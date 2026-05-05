from docplex.mp.model import Model
import math
import time
from itertools import cycle

import requests
import folium


# =============================================================================
# 0. Fallback distance
# =============================================================================

def haversine_km(a, b):
    """
    Fallback straight-line distance.
    a, b: (lat, lon)
    return: km
    """
    earth_radius_km = 6371.0088

    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    h = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    )

    return 2.0 * earth_radius_km * math.asin(math.sqrt(h))


# =============================================================================
# 1. OSRM road router
# =============================================================================

class OSRMRoadRouter:
    """
    OSRM 기반 도로 거리/시간 행렬 및 실제 도로 geometry 호출 클래스.

    입력 좌표:
        (lat, lon)

    OSRM 요청 좌표:
        lon,lat
    """

    def __init__(
        self,
        base_url="https://router.project-osrm.org",
        profile="driving",
        timeout=30,
        request_delay=0.05,
    ):
        self.base_url = base_url.rstrip("/")
        self.profile = profile
        self.timeout = timeout
        self.request_delay = request_delay

        self.session = requests.Session()
        self._table_cache = {}
        self._route_cache = {}

    @staticmethod
    def _point_key(point):
        return (round(float(point[0]), 7), round(float(point[1]), 7))

    @staticmethod
    def _coords_string(points):
        return ";".join(
            f"{float(lon):.8f},{float(lat):.8f}"
            for lat, lon in points
        )

    def _sleep_if_needed(self):
        if self.request_delay and self.request_delay > 0:
            time.sleep(self.request_delay)

    def table_matrix(self, points):
        """
        points:
            [(lat, lon), ...]

        return:
            distance_km[i][j]
            duration_min[i][j]
        """
        if len(points) < 2:
            raise ValueError("OSRM table_matrix requires at least two points.")

        cache_key = tuple(self._point_key(point) for point in points)
        if cache_key in self._table_cache:
            return self._table_cache[cache_key]

        coords_str = self._coords_string(points)
        url = f"{self.base_url}/table/v1/{self.profile}/{coords_str}"

        params = {
            "annotations": "distance,duration",
        }

        self._sleep_if_needed()
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()

        data = response.json()

        if data.get("code") != "Ok":
            raise RuntimeError(
                f"OSRM table API failed: code={data.get('code')}, "
                f"message={data.get('message')}"
            )

        distances_m = data.get("distances")
        durations_s = data.get("durations")

        if distances_m is None or durations_s is None:
            raise RuntimeError("OSRM table response has no distances or durations.")

        size = len(points)
        distance_km = [[0.0 for _ in range(size)] for _ in range(size)]
        duration_min = [[0.0 for _ in range(size)] for _ in range(size)]

        for i in range(size):
            for j in range(size):
                if distances_m[i][j] is None or durations_s[i][j] is None:
                    raise RuntimeError(
                        f"No OSRM road route between point {i} and point {j}."
                    )

                distance_km[i][j] = distances_m[i][j] / 1000.0
                duration_min[i][j] = durations_s[i][j] / 60.0

        self._table_cache[cache_key] = (distance_km, duration_min)
        return distance_km, duration_min

    def route_geometry(self, a, b):
        """
        a, b:
            (lat, lon)

        return:
            geometry_points: [(lat, lon), ...]
            distance_km
            duration_min
        """
        cache_key = (self._point_key(a), self._point_key(b))
        if cache_key in self._route_cache:
            return self._route_cache[cache_key]

        coords_str = self._coords_string([a, b])
        url = f"{self.base_url}/route/v1/{self.profile}/{coords_str}"

        params = {
            "overview": "full",
            "geometries": "geojson",
            "steps": "false",
        }

        self._sleep_if_needed()
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()

        data = response.json()

        if data.get("code") != "Ok":
            raise RuntimeError(
                f"OSRM route API failed: code={data.get('code')}, "
                f"message={data.get('message')}"
            )

        route = data["routes"][0]
        coords_lonlat = route["geometry"]["coordinates"]
        geometry_points = [(lat, lon) for lon, lat in coords_lonlat]

        distance_km = route["distance"] / 1000.0
        duration_min = route["duration"] / 60.0

        self._route_cache[cache_key] = (
            geometry_points,
            distance_km,
            duration_min,
        )
        return geometry_points, distance_km, duration_min


# =============================================================================
# 2. Optimization model
# =============================================================================

def solve_cvrptw_with_satellite_tours(
    coords,
    demand_charge,
    time_windows,
    service_time,
    sat_coords,
    sat_weight,
    sat_service_time,
    num_vehicles,
    cap_weight,
    cap_charge,
    R_km=3.0,
    alpha=1000.0,
    beta=1.0,
    time_limit=None,
    router=None,
    fix_depot_start=True,
    log_output=True,
):
    """
    coords:
        depot + charging customers.
        coords[0] = depot.
        format = [(lat, lon), ...]

    demand_charge:
        customer charging demand.
        demand_charge[0] must be 0.

    time_windows:
        depot/customer time windows in minutes.
        length = len(coords).

    service_time:
        depot/customer service time in minutes.
        customer service_time means EV charging/service duration.
        length = len(coords).

    sat_coords:
        delivery satellite coordinates.
        format = [(lat, lon), ...]

    sat_weight:
        satellite delivery weight demand.
        length = len(sat_coords).

    sat_service_time:
        satellite on-site delivery task time in minutes.
        length = len(sat_coords).

    Capacity logic:
        customer -> charge only.
        satellite -> weight only.

    Hub logic:
        satellite can be assigned to any customer hub within R_km.
        The model chooses the hub by total route + delivery-tour cost, not nearest hub.
    """

    if router is None:
        router = OSRMRoadRouter()

    n = len(coords) - 1
    m = len(sat_coords)

    if n < 1:
        raise ValueError("At least one customer is required.")

    if len(demand_charge) != n + 1:
        raise ValueError("len(demand_charge) must equal len(coords).")

    if len(time_windows) != n + 1:
        raise ValueError("len(time_windows) must equal len(coords).")

    if len(service_time) != n + 1:
        raise ValueError("len(service_time) must equal len(coords).")

    if len(sat_weight) != m:
        raise ValueError("len(sat_weight) must equal len(sat_coords).")

    if len(sat_service_time) != m:
        raise ValueError("len(sat_service_time) must equal len(sat_coords).")

    N = list(range(n + 1))       # depot + customers
    C = list(range(1, n + 1))    # customers
    S = list(range(m))           # satellites
    K = list(range(num_vehicles))

    # -------------------------------------------------------------------------
    # Road distance/time matrices
    # -------------------------------------------------------------------------
    all_points = list(coords) + list(sat_coords)
    road_dist_km, road_time_min = router.table_matrix(all_points)

    def cidx(i):
        return i

    def sidx(s):
        return len(coords) + s

    d = {
        (i, j): road_dist_km[cidx(i)][cidx(j)]
        for i in N for j in N
    }

    t = {
        (i, j): road_time_min[cidx(i)][cidx(j)]
        for i in N for j in N
    }

    hub_to_sat_d = {
        (i, s): road_dist_km[cidx(i)][sidx(s)]
        for i in C for s in S
    }

    sat_to_hub_d = {
        (s, i): road_dist_km[sidx(s)][cidx(i)]
        for s in S for i in C
    }

    hub_to_sat_t = {
        (i, s): road_time_min[cidx(i)][sidx(s)]
        for i in C for s in S
    }

    sat_to_hub_t = {
        (s, i): road_time_min[sidx(s)][cidx(i)]
        for s in S for i in C
    }

    sat_to_sat_d = {
        (s1, s2): road_dist_km[sidx(s1)][sidx(s2)]
        for s1 in S for s2 in S
    }

    sat_to_sat_t = {
        (s1, s2): road_time_min[sidx(s1)][sidx(s2)]
        for s1 in S for s2 in S
    }

    # -------------------------------------------------------------------------
    # Candidate hubs.
    # One-way road effects can make i -> s and s -> i different, so use max.
    # -------------------------------------------------------------------------
    feas = {
        s: [
            i for i in C
            if max(hub_to_sat_d[(i, s)], sat_to_hub_d[(s, i)]) <= R_km
        ]
        for s in S
    }

    for s in S:
        if not feas[s]:
            raise ValueError(
                f"Satellite S{s} has no feasible customer hub within R_km={R_km}. "
                "Increase R_km or check coordinates."
            )

    hub_sats = {
        i: [s for s in S if i in feas[s]]
        for i in C
    }

    def tour_dist(a, b, hub_i):
        """
        Satellite tour arc distance.
        hub is represented by -1.
        """
        if a == -1 and b == -1:
            return 0.0
        if a == -1:
            return hub_to_sat_d[(hub_i, b)]
        if b == -1:
            return sat_to_hub_d[(a, hub_i)]
        return sat_to_sat_d[(a, b)]

    def tour_time(a, b, hub_i):
        """
        Satellite tour arc duration in minutes.
        hub is represented by -1.
        """
        if a == -1 and b == -1:
            return 0.0
        if a == -1:
            return hub_to_sat_t[(hub_i, b)]
        if b == -1:
            return sat_to_hub_t[(a, hub_i)]
        return sat_to_sat_t[(a, b)]

    def safe_max_sat_extra_time(hub_i):
        """
        Conservative upper bound for satellite tour time at hub_i.
        Used only for Big-M.
        """
        Si = hub_sats[hub_i]
        if not Si:
            return 0.0

        nodes = [-1] + Si
        max_arc_time = 0.0

        for a in nodes:
            for b in nodes:
                if a == b:
                    continue
                max_arc_time = max(max_arc_time, tour_time(a, b, hub_i))

        return (len(Si) + 1) * max_arc_time + sum(
            sat_service_time[s] for s in Si
        )

    max_sat_extra_time = {0: 0.0}
    for i in C:
        max_sat_extra_time[i] = safe_max_sat_extra_time(i)

    # -------------------------------------------------------------------------
    # Model variables
    # -------------------------------------------------------------------------
    mdl = Model(name="CVRPTW_ChargingCustomers_DeliverySatellites_OSRM_Folium")

    x = mdl.binary_var_dict(
        ((i, j, k) for i in N for j in N for k in K if i != j),
        name="x",
    )

    y = mdl.binary_var_dict(K, name="y")

    T = mdl.continuous_var_dict(
        ((i, k) for i in N for k in K),
        lb=0,
        name="T",
    )

    T_return = mdl.continuous_var_dict(
        K,
        lb=0,
        name="T_return",
    )

    z = mdl.binary_var_dict(
        ((s, i, k) for s in S for i in feas[s] for k in K),
        name="z",
    )

    p_keys = []
    p_keys_by_hub_vehicle = {}

    for i in C:
        nodes = [-1] + hub_sats[i]

        for k in K:
            p_keys_by_hub_vehicle[(i, k)] = []

        for a in nodes:
            for b in nodes:
                if a == b:
                    continue
                for k in K:
                    key = (a, b, i, k)
                    p_keys.append(key)
                    p_keys_by_hub_vehicle[(i, k)].append(key)

    p = mdl.binary_var_dict(p_keys, name="p")

    u = {}
    for i in C:
        nSi = max(len(hub_sats[i]), 1)
        for s in hub_sats[i]:
            for k in K:
                u[s, i, k] = mdl.continuous_var(
                    lb=0,
                    ub=nSi,
                    name=f"u_{s}_{i}_{k}",
                )

    sat_tour_time_expr = {}
    for i in C:
        for k in K:
            travel_expr = mdl.sum(
                tour_time(a, b, i) * p[a, b, i, k]
                for a, b, _, _ in p_keys_by_hub_vehicle[(i, k)]
            )

            service_expr = mdl.sum(
                sat_service_time[s] * z[s, i, k]
                for s in hub_sats[i]
            )

            sat_tour_time_expr[(i, k)] = travel_expr + service_expr

    # -------------------------------------------------------------------------
    # Basic VRP constraints
    # -------------------------------------------------------------------------
    for j in C:
        mdl.add_constraint(
            mdl.sum(x[i, j, k] for i in N for k in K if i != j) == 1,
            ctname=f"visit_C{j}",
        )

    for k in K:
        for h in C:
            mdl.add_constraint(
                mdl.sum(x[i, h, k] for i in N if i != h)
                == mdl.sum(x[h, j, k] for j in N if j != h),
                ctname=f"flow_C{h}_V{k}",
            )

        mdl.add_constraint(
            mdl.sum(x[0, j, k] for j in C) == y[k],
            ctname=f"depot_out_V{k}",
        )

        mdl.add_constraint(
            mdl.sum(x[i, 0, k] for i in C) == y[k],
            ctname=f"depot_in_V{k}",
        )

        for i in N:
            for j in N:
                if i == j:
                    continue
                mdl.add_constraint(
                    x[i, j, k] <= y[k],
                    ctname=f"x_use_link_{i}_{j}_{k}",
                )

    # -------------------------------------------------------------------------
    # Satellite assignment
    # -------------------------------------------------------------------------
    for s in S:
        mdl.add_constraint(
            mdl.sum(z[s, i, k] for i in feas[s] for k in K) == 1,
            ctname=f"sat_assign_S{s}",
        )

    for s in S:
        for i in feas[s]:
            for k in K:
                mdl.add_constraint(
                    z[s, i, k]
                    <= mdl.sum(x[j, i, k] for j in N if j != i),
                    ctname=f"sat_link_S{s}_C{i}_V{k}",
                )

    # -------------------------------------------------------------------------
    # Capacity constraints
    # -------------------------------------------------------------------------
    # customer charging demand uses only charge capacity.
    # satellite delivery demand uses only weight capacity.
    for k in K:
        mdl.add_constraint(
            mdl.sum(
                sat_weight[s] * z[s, i, k]
                for s in S for i in feas[s]
            )
            <= cap_weight * y[k],
            ctname=f"cap_weight_V{k}",
        )

        mdl.add_constraint(
            mdl.sum(
                demand_charge[j] * x[i, j, k]
                for i in N for j in C if i != j
            )
            <= cap_charge * y[k],
            ctname=f"cap_charge_V{k}",
        )

    # -------------------------------------------------------------------------
    # Time windows
    # -------------------------------------------------------------------------
    depot_start, depot_end = time_windows[0]

    for k in K:
        if fix_depot_start:
            mdl.add_constraint(
                T[0, k] == depot_start,
                ctname=f"fix_depot_start_V{k}",
            )

        mdl.add_constraint(
            T_return[k] <= depot_end * y[k],
            ctname=f"return_upper_V{k}",
        )
        mdl.add_constraint(
            T_return[k] >= depot_start * y[k],
            ctname=f"return_lower_V{k}",
        )

    M_ij = {}
    for i in N:
        for j in N:
            if i == j or j == 0:
                continue

            e_j = time_windows[j][0]
            l_i = time_windows[i][1]

            M_ij[i, j] = max(
                0.0,
                l_i
                + service_time[i]
                + max_sat_extra_time.get(i, 0.0)
                + t[i, j]
                - e_j,
            )

    for k in K:
        for i in N:
            for j in N:
                if i == j or j == 0:
                    continue

                extra_time_at_i = (
                    0.0 if i == 0 else sat_tour_time_expr[(i, k)]
                )

                mdl.add_constraint(
                    T[j, k]
                    >= T[i, k]
                    + service_time[i]
                    + extra_time_at_i
                    + t[i, j]
                    - M_ij[i, j] * (1 - x[i, j, k]),
                    ctname=f"time_{i}_{j}_{k}",
                )

    M_return = {}
    for i in C:
        M_return[i] = max(
            0.0,
            time_windows[i][1]
            + service_time[i]
            + max_sat_extra_time[i]
            + t[i, 0],
        )

    for k in K:
        for i in C:
            mdl.add_constraint(
                T_return[k]
                >= T[i, k]
                + service_time[i]
                + sat_tour_time_expr[(i, k)]
                + t[i, 0]
                - M_return[i] * (1 - x[i, 0, k]),
                ctname=f"return_time_C{i}_V{k}",
            )

    for k in K:
        for i in N:
            e_i, l_i = time_windows[i]
            mdl.add_constraint(T[i, k] >= e_i, ctname=f"tw_start_{i}_{k}")
            mdl.add_constraint(T[i, k] <= l_i, ctname=f"tw_end_{i}_{k}")

    for k in K[:-1]:
        mdl.add_constraint(
            y[k] >= y[k + 1],
            ctname=f"symmetry_V{k}",
        )

    # -------------------------------------------------------------------------
    # Satellite TSP tour constraints
    # -------------------------------------------------------------------------
    for i in C:
        Si = hub_sats[i]
        nodes = [-1] + Si

        for k in K:
            for s in Si:
                mdl.add_constraint(
                    mdl.sum(p[a, s, i, k] for a in nodes if a != s)
                    == z[s, i, k],
                    ctname=f"sat_in_S{s}_C{i}_V{k}",
                )

                mdl.add_constraint(
                    mdl.sum(p[s, b, i, k] for b in nodes if b != s)
                    == z[s, i, k],
                    ctname=f"sat_out_S{s}_C{i}_V{k}",
                )

    for i in C:
        Si = hub_sats[i]
        if not Si:
            continue

        for k in K:
            mdl.add_constraint(
                mdl.sum(p[-1, s, i, k] for s in Si)
                == mdl.sum(p[s, -1, i, k] for s in Si),
                ctname=f"hub_balance_C{i}_V{k}",
            )

            mdl.add_constraint(
                mdl.sum(p[-1, s, i, k] for s in Si) <= 1,
                ctname=f"hub_tour_cap_C{i}_V{k}",
            )

            for s in Si:
                mdl.add_constraint(
                    mdl.sum(p[-1, s2, i, k] for s2 in Si) >= z[s, i, k],
                    ctname=f"hub_required_S{s}_C{i}_V{k}",
                )

    for i in C:
        Si = hub_sats[i]
        nSi = len(Si)

        if nSi <= 1:
            continue

        for k in K:
            for s1 in Si:
                for s2 in Si:
                    if s1 == s2:
                        continue

                    mdl.add_constraint(
                        u[s2, i, k]
                        >= u[s1, i, k]
                        + 1
                        - nSi * (1 - p[s1, s2, i, k]),
                        ctname=f"mtz_S{s1}_S{s2}_C{i}_V{k}",
                    )

    # -------------------------------------------------------------------------
    # Objective
    # -------------------------------------------------------------------------
    veh_distance = mdl.sum(
        d[i, j] * x[i, j, k]
        for i in N for j in N for k in K
        if i != j
    )

    sat_tour_distance = mdl.sum(
        tour_dist(a, b, i) * p[a, b, i, k]
        for a, b, i, k in p_keys
    )

    total_vehicles = mdl.sum(y[k] for k in K)

    mdl.minimize(
        alpha * total_vehicles
        + beta * (veh_distance + sat_tour_distance)
    )

    # -------------------------------------------------------------------------
    # Solve
    # -------------------------------------------------------------------------
    if time_limit is not None:
        mdl.parameters.timelimit = time_limit

    mdl.parameters.emphasis.mip = 1

    sol = mdl.solve(log_output=log_output)

    if sol is None:
        print("No solution found.")

        try:
            from docplex.mp.conflict_refiner import ConflictRefiner

            cr = ConflictRefiner()
            conflicts = cr.refine_conflict(mdl, display=True)
            print(f"\nFound {len(list(conflicts))} conflicting elements.")
        except Exception as exc:
            print(f"Conflict refiner not available: {exc}")

        return None

    # -------------------------------------------------------------------------
    # Extract solution
    # -------------------------------------------------------------------------
    def val(var):
        if var.solution_value is None:
            return 0.0
        return float(var.solution_value)

    routes = {}

    for k in K:
        if val(y[k]) < 0.5:
            continue

        route = [0]
        cur = 0

        for _ in range(len(N) + 5):
            nxt = None
            for j in N:
                if j == cur:
                    continue
                if val(x[cur, j, k]) > 0.5:
                    nxt = j
                    break

            if nxt is None:
                break

            route.append(nxt)

            if nxt == 0:
                break

            cur = nxt

        sat_tours = {}

        for hub in route:
            if hub == 0:
                continue

            if hub not in hub_sats or not hub_sats[hub]:
                continue

            hub_depart_count = sum(
                val(p[-1, s, hub, k])
                for s in hub_sats[hub]
            )

            if hub_depart_count < 0.5:
                continue

            tour = []
            cur_node = -1
            nodes = [-1] + hub_sats[hub]

            for _ in range(len(nodes) + 5):
                nxt_node = None

                for b in nodes:
                    if b == cur_node:
                        continue

                    key = (cur_node, b, hub, k)
                    if key in p and val(p[key]) > 0.5:
                        nxt_node = b
                        break

                if nxt_node is None or nxt_node == -1:
                    break

                tour.append(nxt_node)
                cur_node = nxt_node

            if tour:
                seq = [-1] + tour + [-1]

                travel_len_km = sum(
                    tour_dist(seq[idx], seq[idx + 1], hub)
                    for idx in range(len(seq) - 1)
                )

                travel_time_min = sum(
                    tour_time(seq[idx], seq[idx + 1], hub)
                    for idx in range(len(seq) - 1)
                )

                service_time_min = sum(
                    sat_service_time[s]
                    for s in tour
                )

                sat_tours[hub] = {
                    "sequence": tour,
                    "length_km": travel_len_km,
                    "travel_time_min": travel_time_min,
                    "service_time_min": service_time_min,
                    "duration_min": travel_time_min + service_time_min,
                    "total_weight": sum(sat_weight[s] for s in tour),
                    "satellite_weights": {s: sat_weight[s] for s in tour},
                    "satellite_service_times": {
                        s: sat_service_time[s] for s in tour
                    },
                }

        used_weight = sum(
            sat_weight[s] * val(z[s, i, k])
            for s in S for i in feas[s]
        )

        used_charge = sum(
            demand_charge[j] * val(x[i, j, k])
            for i in N for j in C if i != j
        )

        weight_utilization = used_weight / cap_weight if cap_weight > 0 else 0.0
        charge_utilization = used_charge / cap_charge if cap_charge > 0 else 0.0

        assigned_satellites = []
        for s in S:
            for i in feas[s]:
                if val(z[s, i, k]) > 0.5:
                    assigned_satellites.append(
                        {
                            "satellite": s,
                            "hub": i,
                            "weight": sat_weight[s],
                            "service_time": sat_service_time[s],
                        }
                    )

        schedule = []
        for order, node in enumerate(route):
            if order == 0 and node == 0:
                arrival_min = val(T[0, k])
                node_service_min = service_time[0]
                delivery_tour_min = 0.0
                departure_min = arrival_min + node_service_min

                schedule.append(
                    {
                        "order": order,
                        "node": "Depot",
                        "node_index": 0,
                        "type": "Depot start",
                        "arrival_min": arrival_min,
                        "service_time_min": node_service_min,
                        "delivery_tour_min": delivery_tour_min,
                        "departure_min": departure_min,
                        "note": "",
                    }
                )

            elif node == 0:
                arrival_min = val(T_return[k])

                schedule.append(
                    {
                        "order": order,
                        "node": "Depot",
                        "node_index": 0,
                        "type": "Depot return",
                        "arrival_min": arrival_min,
                        "service_time_min": 0.0,
                        "delivery_tour_min": 0.0,
                        "departure_min": None,
                        "note": "",
                    }
                )

            else:
                arrival_min = val(T[node, k])
                node_service_min = service_time[node]
                delivery_tour_min = (
                    sat_tours[node]["duration_min"]
                    if node in sat_tours
                    else 0.0
                )
                departure_min = (
                    arrival_min
                    + node_service_min
                    + delivery_tour_min
                )

                if node in sat_tours:
                    tour_nodes = [f"S{s}" for s in sat_tours[node]["sequence"]]
                    note = "Delivery: C{} -> {} -> C{}".format(
                        node,
                        " -> ".join(tour_nodes),
                        node,
                    )
                else:
                    note = ""

                schedule.append(
                    {
                        "order": order,
                        "node": f"C{node}",
                        "node_index": node,
                        "type": "Charging customer",
                        "arrival_min": arrival_min,
                        "service_time_min": node_service_min,
                        "delivery_tour_min": delivery_tour_min,
                        "departure_min": departure_min,
                        "note": note,
                    }
                )

        routes[k] = {
            "route": route,
            "schedule": schedule,
            "sat_tours": sat_tours,
            "assigned_satellites": assigned_satellites,
            "used_weight": used_weight,
            "used_charge": used_charge,
            "weight_capacity": cap_weight,
            "charge_capacity": cap_charge,
            "weight_utilization": weight_utilization,
            "charge_utilization": charge_utilization,
            "return_time_min": val(T_return[k]),
        }

    veh_dist_val = sum(
        d[i, j] * val(x[i, j, k])
        for i in N for j in N for k in K
        if i != j
    )

    sat_dist_val = sum(
        tour_dist(a, b, i) * val(p[a, b, i, k])
        for a, b, i, k in p_keys
    )

    sat_duration_val = sum(
        tour_time(a, b, i) * val(p[a, b, i, k])
        for a, b, i, k in p_keys
    ) + sum(
        sat_service_time[s] * val(z[s, i, k])
        for s in S for i in feas[s] for k in K
    )

    return {
        "objective": sol.objective_value,
        "num_vehicles_used": sum(1 for k in K if val(y[k]) > 0.5),
        "vehicle_distance_km": veh_dist_val,
        "satellite_tour_distance_km": sat_dist_val,
        "satellite_tour_duration_min": sat_duration_val,
        "routes": routes,
        "feasible_hubs_by_satellite": feas,
        "hub_candidates": hub_sats,
    }


# =============================================================================
# 3. Folium map
# =============================================================================

def _format_min(value):
    if value is None:
        return "-"
    return f"{value:.1f} min"


def _format_pct(value):
    return f"{100.0 * value:.1f}%"


def draw_solution_map(
    coords,
    sat_coords,
    result,
    router,
    output_html="solution_map.html",
):
    """
    Draw depot, charging customers, delivery satellites, road-based vehicle routes,
    road-based satellite delivery tours, vehicle schedule, and utilization table.
    """

    all_points = list(coords) + list(sat_coords)

    center_lat = sum(point[0] for point in all_points) / len(all_points)
    center_lon = sum(point[1] for point in all_points) / len(all_points)

    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles="OpenStreetMap",
    )

    # -------------------------------------------------------------------------
    # Build satellite assignment dictionary
    # -------------------------------------------------------------------------
    sat_assignment = {}
    for k, info in result["routes"].items():
        for hub, tour_info in info["sat_tours"].items():
            for s in tour_info["sequence"]:
                sat_assignment[s] = {
                    "vehicle": k,
                    "hub": hub,
                    "tour_length_km": tour_info["length_km"],
                    "tour_duration_min": tour_info["duration_min"],
                }

    # -------------------------------------------------------------------------
    # Depot and customer base markers
    # -------------------------------------------------------------------------
    folium.Marker(
        location=coords[0],
        popup="<b>Depot</b>",
        tooltip="Depot",
        icon=folium.Icon(color="red", icon="home"),
        z_index_offset=5000,
    ).add_to(fmap)

    customer_group = folium.FeatureGroup(name="Charging Customers", show=True)

    for i, point in enumerate(coords[1:], start=1):
        hub_satellites = []
        for k, info in result["routes"].items():
            if i in info["sat_tours"]:
                hub_satellites.extend(info["sat_tours"][i]["sequence"])

        hub_text = (
            "None"
            if not hub_satellites
            else ", ".join(f"S{s}" for s in hub_satellites)
        )

        folium.Marker(
            location=point,
            popup=(
                f"<b>Customer C{i}</b><br>"
                f"Type: EV charging<br>"
                f"Delivery satellites handled here: {hub_text}"
            ),
            tooltip=f"C{i} charging",
            icon=folium.Icon(color="blue", icon="bolt", prefix="fa"),
            z_index_offset=3500,
        ).add_to(customer_group)

    customer_group.add_to(fmap)

    # -------------------------------------------------------------------------
    # Vehicle routes and schedule rows
    # -------------------------------------------------------------------------
    colors = cycle([
        "purple",
        "orange",
        "darkred",
        "cadetblue",
        "darkgreen",
        "black",
        "deeppink",
        "gray",
    ])

    utilization_rows = []
    schedule_rows = []
    delivery_rows = []

    for k, info in result["routes"].items():
        color = next(colors)
        route = info["route"]

        route_text = " -> ".join(
            "Depot" if node == 0 else f"C{node}"
            for node in route
        )

        utilization_rows.append(
            f"""
            <tr>
                <td><b>Vehicle {k}</b></td>
                <td>{route_text}</td>
                <td>{info["used_weight"]:.1f} / {info["weight_capacity"]:.1f}
                    ({_format_pct(info["weight_utilization"])})</td>
                <td>{info["used_charge"]:.1f} / {info["charge_capacity"]:.1f}
                    ({_format_pct(info["charge_utilization"])})</td>
            </tr>
            """
        )

        for row in info["schedule"]:
            schedule_rows.append(
                f"""
                <tr>
                    <td>V{k}</td>
                    <td>{row["order"]}</td>
                    <td>{row["node"]}</td>
                    <td>{row["type"]}</td>
                    <td>{_format_min(row["arrival_min"])}</td>
                    <td>{_format_min(row["service_time_min"])}</td>
                    <td>{_format_min(row["delivery_tour_min"])}</td>
                    <td>{_format_min(row["departure_min"])}</td>
                    <td>{row["note"]}</td>
                </tr>
                """
            )

        vehicle_group = folium.FeatureGroup(name=f"Vehicle {k}", show=True)

        # Vehicle main route: actual road geometry.
        for a, b in zip(route[:-1], route[1:]):
            geometry, dist_km, dur_min = router.route_geometry(
                coords[a],
                coords[b],
            )

            a_label = "Depot" if a == 0 else f"C{a}"
            b_label = "Depot" if b == 0 else f"C{b}"

            folium.PolyLine(
                locations=geometry,
                color=color,
                weight=5,
                opacity=0.85,
                tooltip=(
                    f"Vehicle {k}: {a_label} -> {b_label} "
                    f"({dist_km:.2f} km, {dur_min:.1f} min)"
                ),
                smooth_factor=0,
            ).add_to(vehicle_group)

        # Satellite delivery tours: actual road geometry, dashed.
        for hub, tour_info in info["sat_tours"].items():
            sat_seq = tour_info["sequence"]

            tour_points = [coords[hub]]
            tour_points += [sat_coords[s] for s in sat_seq]
            tour_points += [coords[hub]]

            tour_labels = [f"C{hub}"]
            tour_labels += [f"S{s}" for s in sat_seq]
            tour_labels += [f"C{hub}"]

            delivery_rows.append(
                f"""
                <tr>
                    <td>V{k}</td>
                    <td>C{hub}</td>
                    <td>{" -> ".join(tour_labels)}</td>
                    <td>{tour_info["total_weight"]:.1f}</td>
                    <td>{tour_info["length_km"]:.2f} km</td>
                    <td>{tour_info["travel_time_min"]:.1f} min</td>
                    <td>{tour_info["service_time_min"]:.1f} min</td>
                    <td>{tour_info["duration_min"]:.1f} min</td>
                </tr>
                """
            )

            for idx in range(len(tour_points) - 1):
                geometry, dist_km, dur_min = router.route_geometry(
                    tour_points[idx],
                    tour_points[idx + 1],
                )

                folium.PolyLine(
                    locations=geometry,
                    color=color,
                    weight=4,
                    opacity=0.95,
                    dash_array="8, 8",
                    tooltip=(
                        f"V{k} delivery at C{hub}: "
                        f"{tour_labels[idx]} -> {tour_labels[idx + 1]} "
                        f"({dist_km:.2f} km, {dur_min:.1f} min)"
                    ),
                    smooth_factor=0,
                ).add_to(vehicle_group)

        # Visit order markers placed after lines.
        for row in info["schedule"]:
            node_index = row["node_index"]
            if row["type"] == "Depot return":
                # Same depot location. Keep return marker slightly smaller in text only.
                marker_text = "R"
            else:
                marker_text = str(row["order"])

            folium.Marker(
                location=coords[node_index],
                popup=(
                    f"<b>Vehicle {k}</b><br>"
                    f"Order: {row['order']}<br>"
                    f"Node: {row['node']}<br>"
                    f"Type: {row['type']}<br>"
                    f"Arrival: {_format_min(row['arrival_min'])}<br>"
                    f"Service: {_format_min(row['service_time_min'])}<br>"
                    f"Delivery tour: {_format_min(row['delivery_tour_min'])}<br>"
                    f"Departure: {_format_min(row['departure_min'])}<br>"
                    f"{row['note']}"
                ),
                tooltip=f"V{k}-{row['order']}-{row['node']}",
                icon=folium.DivIcon(
                    html=f"""
                    <div style="
                        font-size: 12px;
                        color: white;
                        background: {color};
                        border-radius: 50%;
                        width: 28px;
                        height: 28px;
                        text-align: center;
                        line-height: 28px;
                        border: 2px solid white;">
                        {marker_text}
                    </div>
                    """
                ),
                z_index_offset=4500,
            ).add_to(vehicle_group)

        vehicle_group.add_to(fmap)

    # -------------------------------------------------------------------------
    # Satellite markers are added last so they are visible.
    # -------------------------------------------------------------------------
    satellite_group = folium.FeatureGroup(name="Delivery Satellites", show=True)

    for s, point in enumerate(sat_coords):
        if s in sat_assignment:
            assigned = sat_assignment[s]
            assign_text = (
                f"Vehicle V{assigned['vehicle']}, hub C{assigned['hub']}"
            )
        else:
            assign_text = "Not assigned in extracted tour"

        folium.CircleMarker(
            location=point,
            radius=12,
            popup=(
                f"<b>Satellite S{s}</b><br>"
                f"Type: delivery<br>"
                f"Assignment: {assign_text}"
            ),
            tooltip=f"S{s} delivery",
            color="black",
            weight=3,
            fill=True,
            fill_color="lime",
            fill_opacity=0.95,
        ).add_to(satellite_group)

        folium.Marker(
            location=point,
            icon=folium.DivIcon(
                html=f"""
                <div style="
                    font-size: 12px;
                    font-weight: bold;
                    color: black;
                    background: white;
                    border: 1px solid black;
                    border-radius: 4px;
                    padding: 1px 4px;
                    transform: translate(10px, -12px);">
                    S{s}
                </div>
                """
            ),
            z_index_offset=7000,
        ).add_to(satellite_group)

    satellite_group.add_to(fmap)

    # -------------------------------------------------------------------------
    # HTML schedule panel
    # -------------------------------------------------------------------------
    schedule_panel_html = f"""
    <div style="
        position: fixed;
        bottom: 20px;
        left: 20px;
        width: 980px;
        max-height: 430px;
        overflow-y: auto;
        background-color: white;
        z-index: 9999;
        border: 2px solid #444;
        border-radius: 8px;
        padding: 12px;
        font-size: 12px;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.35);
    ">
        <h3 style="margin-top: 0;">Vehicle Schedule & Utilization</h3>

        <h4>1. Vehicle Utilization</h4>
        <table style="width:100%; border-collapse: collapse;" border="1">
            <thead>
                <tr style="background:#f0f0f0;">
                    <th>Vehicle</th>
                    <th>Main route</th>
                    <th>Weight utilization</th>
                    <th>Charge utilization</th>
                </tr>
            </thead>
            <tbody>
                {''.join(utilization_rows)}
            </tbody>
        </table>

        <h4>2. Vehicle Schedule</h4>
        <table style="width:100%; border-collapse: collapse;" border="1">
            <thead>
                <tr style="background:#f0f0f0;">
                    <th>Vehicle</th>
                    <th>Order</th>
                    <th>Node</th>
                    <th>Type</th>
                    <th>Arrival</th>
                    <th>Service</th>
                    <th>Delivery tour</th>
                    <th>Departure</th>
                    <th>Note</th>
                </tr>
            </thead>
            <tbody>
                {''.join(schedule_rows)}
            </tbody>
        </table>

        <h4>3. Satellite Delivery Tours</h4>
        <table style="width:100%; border-collapse: collapse;" border="1">
            <thead>
                <tr style="background:#f0f0f0;">
                    <th>Vehicle</th>
                    <th>Hub</th>
                    <th>Tour</th>
                    <th>Weight</th>
                    <th>Distance</th>
                    <th>Travel time</th>
                    <th>Service time</th>
                    <th>Total time</th>
                </tr>
            </thead>
            <tbody>
                {''.join(delivery_rows)}
            </tbody>
        </table>

        <p style="margin-bottom:0;">
            <b>Legend:</b>
            red = depot,
            blue = charging customer,
            lime = delivery satellite,
            solid line = main vehicle route,
            dashed line = satellite delivery tour.
        </p>
    </div>
    """

    fmap.get_root().html.add_child(folium.Element(schedule_panel_html))

    folium.LayerControl(collapsed=False).add_to(fmap)

    bounds = [[point[0], point[1]] for point in all_points]
    fmap.fit_bounds(bounds)

    fmap.save(output_html)
    print(f"Map saved to: {output_html}")


# =============================================================================
# 4. Example
# =============================================================================

if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Coordinates are (latitude, longitude).
    # coords[0] = depot.
    # coords[1:] = EV charging customers.
    # -------------------------------------------------------------------------
    coords = [
        (37.5665, 126.9780),  # Depot: Seoul City Hall
        (37.5729, 126.9769),  # C1: Gwanghwamun
        (37.5663, 127.0093),  # C2: Dongdaemun
        (37.5547, 126.9706),  # C3: Seoul Station
        (37.5219, 126.9245),  # C4: Yeouido
        (37.4979, 127.0276),  # C5: Gangnam
        (37.5133, 127.1000),  # C6: Jamsil
        (37.5563, 126.9236),  # C7: Hongdae
        (37.5446, 127.0559),  # C8: Seongsu
    ]

    # Customer charging demand.
    # Weight is intentionally not defined for customers.
    demand_charge = [
        0,   # Depot
        5,   # C1
        7,   # C2
        4,   # C3
        6,   # C4
        8,   # C5
        7,   # C6
        5,   # C7
        6,   # C8
    ]

    # Time windows in minutes.
    time_windows = [
        (0, 600),    # Depot
        (20, 180),   # C1
        (30, 220),   # C2
        (10, 200),   # C3
        (40, 260),   # C4
        (60, 330),   # C5
        (70, 360),   # C6
        (30, 250),   # C7
        (50, 300),   # C8
    ]

    # Customer service time in minutes.
    # This is the charging/customer-handling time at each customer.
    service_time = [
        0,   # Depot
        12,  # C1
        18,  # C2
        10,  # C3
        15,  # C4
        22,  # C5
        17,  # C6
        13,  # C7
        20,  # C8
    ]

    # Satellite delivery coordinates.
    sat_coords = [
        (37.5737, 126.9800),  # S0 near C1
        (37.5708, 126.9735),  # S1 near C1/C3
        (37.5525, 126.9720),  # S2 near C3
        (37.5195, 126.9270),  # S3 near C4
        (37.4995, 127.0310),  # S4 near C5
        (37.5115, 127.0965),  # S5 near C6
        (37.5575, 126.9215),  # S6 near C7
    ]

    # Satellite delivery weight.
    # Charge is intentionally not defined for satellites.
    sat_weight = [
        2,  # S0
        3,  # S1
        3,  # S2
        4,  # S3
        3,  # S4
        2,  # S5
        4,  # S6
    ]

    # Satellite service time in minutes.
    # This is on-site delivery handling time at each satellite.
    sat_service_time = [
        4,  # S0
        6,  # S1
        5,  # S2
        8,  # S3
        7,  # S4
        4,  # S5
        9,  # S6
    ]

    router = OSRMRoadRouter(
        base_url="https://router.project-osrm.org",
        profile="driving",
        timeout=30,
        request_delay=0.05,
    )

    result = solve_cvrptw_with_satellite_tours(
        coords=coords,
        demand_charge=demand_charge,
        time_windows=time_windows,
        service_time=service_time,
        sat_coords=sat_coords,
        sat_weight=sat_weight,
        sat_service_time=sat_service_time,
        num_vehicles=3,
        cap_weight=10,
        cap_charge=18,
        R_km=3.0,
        alpha=1000.0,
        beta=1.0,
        time_limit=120,
        router=router,
        fix_depot_start=True,
        log_output=True,
    )

    if result:
        print("\n=== Result ===")
        print(f"Objective                   : {result['objective']:.2f}")
        print(f"Vehicles used               : {result['num_vehicles_used']}")
        print(f"Vehicle distance            : {result['vehicle_distance_km']:.2f} km")
        print(f"Satellite tour distance     : {result['satellite_tour_distance_km']:.2f} km")
        print(f"Satellite tour duration     : {result['satellite_tour_duration_min']:.2f} min")

        print("\nFeasible hubs by satellite:")
        for s, hubs in result["feasible_hubs_by_satellite"].items():
            print(f"  S{s}: " + ", ".join(f"C{i}" for i in hubs))

        print("\nRoutes and utilization:")
        for k, info in result["routes"].items():
            route_str = " -> ".join(
                "Depot" if node == 0 else f"C{node}"
                for node in info["route"]
            )

            print(f"\n  Vehicle {k}: {route_str}")
            print(
                f"    weight utilization: "
                f"{info['used_weight']:.1f} / {info['weight_capacity']:.1f} "
                f"({_format_pct(info['weight_utilization'])})"
            )
            print(
                f"    charge utilization: "
                f"{info['used_charge']:.1f} / {info['charge_capacity']:.1f} "
                f"({_format_pct(info['charge_utilization'])})"
            )

            print("    schedule:")
            for row in info["schedule"]:
                print(
                    f"      {row['order']:>2} | {row['node']:<6} | "
                    f"arr={_format_min(row['arrival_min']):>9} | "
                    f"service={_format_min(row['service_time_min']):>9} | "
                    f"delivery={_format_min(row['delivery_tour_min']):>9} | "
                    f"dep={_format_min(row['departure_min']):>9} | "
                    f"{row['note']}"
                )

            for hub, tour_info in info["sat_tours"].items():
                tour_str = " -> ".join(f"S{s}" for s in tour_info["sequence"])
                print(
                    f"    delivery tour at C{hub}: "
                    f"C{hub} -> {tour_str} -> C{hub} "
                    f"(weight={tour_info['total_weight']:.1f}, "
                    f"dist={tour_info['length_km']:.2f} km, "
                    f"time={tour_info['duration_min']:.1f} min)"
                )

        draw_solution_map(
            coords=coords,
            sat_coords=sat_coords,
            result=result,
            router=router,
            output_html="solution_map.html",
        )
