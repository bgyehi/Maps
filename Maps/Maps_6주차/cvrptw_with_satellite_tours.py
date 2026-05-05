"""
CVRPTW + Dual Capacity + Satellite TSP Tours
--------------------------------------------
- 일반 고객: 차량이 직접 방문
- 딸린 고객: 직접 방문 불가. hub 고객 i (d(i, s) <= R) 에 할당되면
  차량 k 가 i 에서 시작해 할당된 모든 딸린고객을 TSP 로 돈 뒤 다시 i 로 복귀.
  투어 총거리가 목적함수에 추가된다.
- 목적함수: alpha * 차량수 + beta * (차량 이동거리 + 딸린고객 투어거리)
"""

from docplex.mp.model import Model
import math


def solve_cvrptw_with_satellite_tours(
    coords, demand_weight, demand_charge, time_windows, service_time,
    sat_coords, sat_weight, sat_charge,
    num_vehicles, cap_weight, cap_charge,
    R=3.0, speed=1.0, alpha=1000.0, beta=1.0, time_limit=None,
):
    n = len(coords) - 1
    m = len(sat_coords)
    N = list(range(n + 1))
    C = list(range(1, n + 1))
    S = list(range(m))
    K = list(range(num_vehicles))

    def euc(a, b): return math.hypot(a[0] - b[0], a[1] - b[1])

    d  = {(i, j): euc(coords[i], coords[j]) for i in N for j in N}
    t  = {(i, j): d[i, j] / speed for i in N for j in N}
    ds = {(s, i): euc(sat_coords[s], coords[i]) for s in S for i in C}
    # 딸린고객간 거리
    dss = {(s1, s2): euc(sat_coords[s1], sat_coords[s2])
           for s1 in S for s2 in S}

    # feasibility: 각 딸린고객이 붙을 수 있는 hub 후보
    feas = {s: [i for i in C if ds[s, i] <= R] for s in S}
    for s in S:
        if not feas[s]:
            raise ValueError(f"Satellite {s} has no hub within R={R}.")
    # 역: 각 hub 가 가질 수 있는 딸린고객
    hub_sats = {i: [s for s in S if i in feas[s]] for i in C}

    # 투어 아크 거리: hub = -1
    def tour_dist(a, b, i):
        pa = coords[i]    if a == -1 else sat_coords[a]
        pb = coords[i]    if b == -1 else sat_coords[b]
        return math.hypot(pa[0] - pb[0], pa[1] - pb[1])

    mdl = Model(name="CVRPTW_SatelliteTours")

    # ---------- 변수 ----------
    x = mdl.binary_var_dict(((i, j, k) for i in N for j in N for k in K if i != j),
                            name="x")
    y = mdl.binary_var_dict(K, name="y")
    T = mdl.continuous_var_dict(((i, k) for i in N for k in K), lb=0, name="T")

    z = mdl.binary_var_dict(((s, i, k) for s in S for i in feas[s] for k in K),
                            name="z")

    # 투어 아크: p[a, b, i, k], a,b ∈ {-1} ∪ hub_sats[i]
    p_keys = []
    for i in C:
        nodes = [-1] + hub_sats[i]
        for a in nodes:
            for b in nodes:
                if a != b:
                    for k in K:
                        p_keys.append((a, b, i, k))
    p = mdl.binary_var_dict(p_keys, name="p")

    # MTZ 순서 변수 (위성 노드에만 필요)
    u = {}
    for i in C:
        nSi = max(len(hub_sats[i]), 1)
        for s in hub_sats[i]:
            for k in K:
                u[s, i, k] = mdl.continuous_var(lb=0, ub=nSi,
                                                name=f"u_{s}_{i}_{k}")

    # ---------- 기존 VRP 제약 ----------
    for j in C:
        mdl.add_constraint(
            mdl.sum(x[i, j, k] for i in N for k in K if i != j) == 1,
            ctname=f"visit_{j}")

    for k in K:
        for h in C:
            mdl.add_constraint(
                mdl.sum(x[i, h, k] for i in N if i != h) ==
                mdl.sum(x[h, j, k] for j in N if j != h),
                ctname=f"flow_{h}_{k}")
        mdl.add_constraint(
            mdl.sum(x[0, j, k] for j in C) == y[k], ctname=f"depot_out_{k}")
        mdl.add_constraint(
            mdl.sum(x[i, 0, k] for i in C) == y[k], ctname=f"depot_in_{k}")

    # ---------- 딸린고객 배정 & 방문 연계 ----------
    for s in S:
        mdl.add_constraint(
            mdl.sum(z[s, i, k] for i in feas[s] for k in K) == 1,
            ctname=f"sat_assign_{s}")

    for s in S:
        for i in feas[s]:
            for k in K:
                mdl.add_constraint(
                    z[s, i, k] <= mdl.sum(x[j, i, k] for j in N if j != i),
                    ctname=f"sat_link_{s}_{i}_{k}")

    # ---------- 용량 (투어 딸린고객 수요 포함) ----------
    for k in K:
        mdl.add_constraint(
            mdl.sum(demand_weight[j] * x[i, j, k] for i in N for j in C if i != j)
            + mdl.sum(sat_weight[s] * z[s, i, k]
                      for s in S for i in feas[s])
            <= cap_weight * y[k], ctname=f"cap_w_{k}")
        mdl.add_constraint(
            mdl.sum(demand_charge[j] * x[i, j, k] for i in N for j in C if i != j)
            + mdl.sum(sat_charge[s] * z[s, i, k]
                      for s in S for i in feas[s])
            <= cap_charge * y[k], ctname=f"cap_c_{k}")

    # ---------- 시간 전이 (tight Big-M) + 타임윈도우 ----------
    # ★ depot(j=0) 복귀 arc 는 제외: T[0,k] 가 단일 변수라 닫힌 경로에서
    #   T[0,k] >= T[0,k] + (경로 총 시간) 이라는 불가능 제약이 생김
    M_ij = {}
    for i in N:
        for j in N:
            if i == j or j == 0: continue
            M_ij[i, j] = max(0.0,
                             time_windows[i][1] + service_time[i]
                             + t[i, j] - time_windows[j][0])

    for k in K:
        for i in N:
            for j in N:
                if i == j or j == 0: continue
                mdl.add_constraint(
                    T[j, k] >= T[i, k] + service_time[i] + t[i, j]
                               - M_ij[i, j] * (1 - x[i, j, k]),
                    ctname=f"time_{i}_{j}_{k}")

    for k in K:
        for i in N:
            e_i, l_i = time_windows[i]
            mdl.add_constraint(T[i, k] >= e_i, ctname=f"tw_e_{i}_{k}")
            mdl.add_constraint(T[i, k] <= l_i, ctname=f"tw_l_{i}_{k}")

    for k in K[:-1]:
        mdl.add_constraint(y[k] >= y[k + 1], ctname=f"sym_{k}")

    # ---------- 딸린고객 TSP 투어 제약 ----------
    # (T1, T2) 딸린고객 in/out-degree = z
    for i in C:
        Si = hub_sats[i]
        nodes = [-1] + Si
        for k in K:
            for s in Si:
                mdl.add_constraint(
                    mdl.sum(p[a, s, i, k] for a in nodes if a != s)
                    == z[s, i, k], ctname=f"sat_in_{s}_{i}_{k}")
                mdl.add_constraint(
                    mdl.sum(p[s, b, i, k] for b in nodes if b != s)
                    == z[s, i, k], ctname=f"sat_out_{s}_{i}_{k}")

    # (T3) hub: out = in, at most 1
    for i in C:
        Si = hub_sats[i]
        if not Si: continue
        for k in K:
            mdl.add_constraint(
                mdl.sum(p[-1, s, i, k] for s in Si)
                == mdl.sum(p[s, -1, i, k] for s in Si),
                ctname=f"hub_bal_{i}_{k}")
            mdl.add_constraint(
                mdl.sum(p[-1, s, i, k] for s in Si) <= 1,
                ctname=f"hub_cap_{i}_{k}")
            # hub 은 할당 딸린고객이 있으면 반드시 방문
            for s in Si:
                mdl.add_constraint(
                    mdl.sum(p[-1, s2, i, k] for s2 in Si) >= z[s, i, k],
                    ctname=f"hub_req_{s}_{i}_{k}")

    # (T4) MTZ subtour 제거 (위성간 아크에만 적용)
    for i in C:
        Si = hub_sats[i]
        nSi = len(Si)
        if nSi <= 1: continue
        for k in K:
            for s1 in Si:
                for s2 in Si:
                    if s1 == s2: continue
                    mdl.add_constraint(
                        u[s2, i, k] >= u[s1, i, k] + 1
                                       - nSi * (1 - p[s1, s2, i, k]),
                        ctname=f"mtz_{s1}_{s2}_{i}_{k}")

    # ---------- 목적함수 ----------
    veh_distance = mdl.sum(d[i, j] * x[i, j, k]
                           for i in N for j in N for k in K if i != j)
    sat_tour_distance = mdl.sum(tour_dist(a, b, i) * p[a, b, i, k]
                                for (a, b, i, k) in p_keys)
    total_vehicles = mdl.sum(y[k] for k in K)
    mdl.minimize(alpha * total_vehicles
                 + beta * (veh_distance + sat_tour_distance))

    if time_limit is not None:
        mdl.parameters.timelimit = time_limit
    mdl.parameters.emphasis.mip = 1   # feasibility 우선
    sol = mdl.solve(log_output=True)
    if sol is None:
        print("No solution found.")
        # docplex conflict refiner (정확한 API)
        try:
            from docplex.mp.conflict_refiner import ConflictRefiner
            cr = ConflictRefiner()
            conflicts = cr.refine_conflict(mdl, display=True)
            print(f"\nFound {len(list(conflicts))} conflicting elements.")
        except Exception as e:
            print(f"Conflict refiner not available: {e}")
        return None

    # ---------- 결과 추출 ----------
    routes = {}
    for k in K:
        if y[k].solution_value < 0.5:
            continue
        # 차량 경로
        route, cur = [0], 0
        while True:
            nxt = None
            for j in N:
                if j == cur: continue
                if x[cur, j, k].solution_value > 0.5:
                    nxt = j; break
            if nxt is None or nxt == 0:
                route.append(0); break
            route.append(nxt); cur = nxt
        arrivals = [T[i, k].solution_value for i in route]

        # 각 hub 의 딸린고객 투어 순서 추출
        sat_tours = {}   # hub i -> [s1, s2, ...] (hub 에서 시작 → 복귀)
        for i in route:
            if i == 0 or i not in hub_sats or not hub_sats[i]:
                continue
            if sum(p[-1, s, i, k].solution_value for s in hub_sats[i]) < 0.5:
                continue
            tour, cur_n = [], -1
            while True:
                nxt_n = None
                for b in [-1] + hub_sats[i]:
                    if b == cur_n: continue
                    if p[cur_n, b, i, k].solution_value > 0.5:
                        nxt_n = b; break
                if nxt_n is None or nxt_n == -1:
                    break
                tour.append(nxt_n)
                cur_n = nxt_n
            if tour:
                tour_len = sum(tour_dist(a, b, i)
                               for a, b in zip([-1] + tour, tour + [-1])
                               if any(p[a, b, i, k].solution_value > 0.5
                                      for a_, b_, _, _ in [(a, b, i, k)]))
                # 더 간단히: 직접 합산
                seq = [-1] + tour + [-1]
                tour_len = sum(tour_dist(seq[t1], seq[t1 + 1], i)
                               for t1 in range(len(seq) - 1))
                sat_tours[i] = {"sequence": tour, "length": tour_len}

        routes[k] = {"route": route, "arrivals": arrivals,
                     "sat_tours": sat_tours}

    veh_dist_val = sum(d[i, j] * x[i, j, k].solution_value
                       for i in N for j in N for k in K if i != j)
    sat_dist_val = sum(tour_dist(a, b, i) * p[a, b, i, k].solution_value
                       for (a, b, i, k) in p_keys)

    return {
        "objective": sol.objective_value,
        "num_vehicles_used": sum(1 for k in K if y[k].solution_value > 0.5),
        "vehicle_distance": veh_dist_val,
        "satellite_tour_distance": sat_dist_val,
        "routes": routes,
    }


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    coords = [(0, 0),
              (2, 3), (5, 1), (6, 5), (8, 3),
              (1, 6), (3, 7), (7, 8), (4, 2)]
    demand_weight = [0, 10, 15, 8, 12, 7, 20, 9, 14]
    demand_charge = [0,  5,  7, 3,  6, 4,  8, 5,  6]
    time_windows  = [(0, 200),
                     (10, 60), (20, 80), (30, 90), (40, 100),
                     (15, 70), (25, 85), (50, 120), (35, 95)]
    service_time  = [0, 5, 5, 5, 5, 5, 5, 5, 5]

    # 딸린 고객: 여러 개가 같은 hub 근처에 있도록 배치
    sat_coords = [(2.5, 3.5), (1.8, 2.5),   # C1 근처
                  (6.5, 5.5), (6.2, 4.3),   # C3 근처
                  (1.5, 7.0), (0.8, 5.6),   # C5 근처
                  (4.5, 2.5)]               # C8 근처
    sat_weight = [2, 3, 3, 2, 2, 1, 4]
    sat_charge = [1, 1, 2, 1, 1, 1, 2]

    res = solve_cvrptw_with_satellite_tours(
        coords=coords,
        demand_weight=demand_weight, demand_charge=demand_charge,
        time_windows=time_windows, service_time=service_time,
        sat_coords=sat_coords,
        sat_weight=sat_weight, sat_charge=sat_charge,
        num_vehicles=3, cap_weight=50, cap_charge=20,
        R=3.0, alpha=1000.0, beta=1.0, time_limit=120,
    )

    if res:
        print("\n=== Result ===")
        print(f"Objective                : {res['objective']:.2f}")
        print(f"Vehicles used            : {res['num_vehicles_used']}")
        print(f"Vehicle distance         : {res['vehicle_distance']:.2f}")
        print(f"Satellite tour distance  : {res['satellite_tour_distance']:.2f}")
        for k, info in res["routes"].items():
            print(f"  Vehicle {k}: {info['route']}")
            for hub, t_info in info["sat_tours"].items():
                print(f"    hub C{hub} tour: {hub} -> " +
                      " -> ".join(f"s{s}" for s in t_info["sequence"]) +
                      f" -> {hub}  (len={t_info['length']:.2f})")