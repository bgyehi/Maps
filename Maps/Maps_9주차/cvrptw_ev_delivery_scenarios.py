"""
CVRPTW + Dual Capacity + Satellite TSP Tours (전기차 배달 최적화)
--------------------------------------------
수정 사항:
1. 전기차는 weight 제외, 배달(일반고객)은 charge 제외
2. 딸린고객이 가장 가까운 hub이 아닌 목적함수 최소화 hub 선택
4. 서비스타임이 충전수요(딸린고객)에 따라 차등 적용
5. HTML 지도 시각화
"""

from docplex.mp.model import Model
import math
import os
import json


def solve_cvrptw_with_satellite_tours(
    coords, demand_weight, demand_charge, time_windows, service_time,
    sat_coords, sat_weight, sat_charge, sat_service_time,
    num_vehicles, cap_weight, cap_charge,
    speed=1.0, alpha=1000.0, beta=1.0, time_limit=None,
):
    """
    전기차 배달 최적화

    Parameters:
    - coords: 일반고객(배달) 좌표 [(lng, lat), ...], coords[0] = depot
    - demand_weight: 일반고객 배달 무게 수요 (kg)
    - demand_charge: 일반고객 충전 수요 (사용 안 함, 0으로 설정)
    - time_windows: 시간창 [(early, late), ...]
    - service_time: 일반고객 서비스 타임 (분)
    - sat_coords: 충전소(딸린고객) 좌표 [(lng, lat), ...]
    - sat_weight: 충전소 무게 수요 (사용 안 함, 0으로 설정)
    - sat_charge: 충전소 충전 용량 (kWh)
    - sat_service_time: 충전소별 서비스 타임 (분, 충전량에 비례)
    - num_vehicles: 차량 수
    - cap_weight: 차량 배달 용량 (kg)
    - cap_charge: 차량 충전 용량 (kWh)
    - speed: 차량 속도
    - alpha: 차량수 가중치
    - beta: 거리 가중치
    - time_limit: 최적화 시간 제한 (초)
    """
    n = len(coords) - 1
    m = len(sat_coords)
    N = list(range(n + 1))
    C = list(range(1, n + 1))
    S = list(range(m))
    K = list(range(num_vehicles))

    def euc(a, b): 
        return math.hypot(a[0] - b[0], a[1] - b[1])

    # 거리 계산
    d  = {(i, j): euc(coords[i], coords[j]) for i in N for j in N}
    t  = {(i, j): d[i, j] / speed for i in N for j in N}
    ds = {(s, i): euc(sat_coords[s], coords[i]) for s in S for i in C}
    dss = {(s1, s2): euc(sat_coords[s1], sat_coords[s2])
           for s1 in S for s2 in S}

    # 조건 2: 모든 hub를 후보로 (거리 제약 없음, 최적화로 선택)
    feas = {s: C for s in S}
    hub_sats = {i: S for i in C}

    def tour_dist(a, b, i):
        """투어 거리: hub i에서 충전소 a, b 간 거리"""
        pa = coords[i] if a == -1 else sat_coords[a]
        pb = coords[i] if b == -1 else sat_coords[b]
        return math.hypot(pa[0] - pb[0], pa[1] - pb[1])

    mdl = Model(name="CVRPTW_EV_Delivery")

    # ---------- 변수 ----------
    x = mdl.binary_var_dict(
        ((i, j, k) for i in N for j in N for k in K if i != j), 
        name="x"
    )
    y = mdl.binary_var_dict(K, name="y")
    T = mdl.continuous_var_dict(((i, k) for i in N for k in K), lb=0, name="T")
    z = mdl.binary_var_dict(
        ((s, i, k) for s in S for i in C for k in K), 
        name="z"
    )

    # 투어 아크
    p_keys = []
    for i in C:
        nodes = [-1] + list(S)
        for a in nodes:
            for b in nodes:
                if a != b:
                    for k in K:
                        p_keys.append((a, b, i, k))
    p = mdl.binary_var_dict(p_keys, name="p")

    # MTZ 순서 변수
    u = {}
    for i in C:
        nSi = max(len(S), 1)
        for s in S:
            for k in K:
                u[s, i, k] = mdl.continuous_var(
                    lb=0, ub=nSi, 
                    name=f"u_{s}_{i}_{k}"
                )

    # ---------- 기존 VRP 제약 ----------
    for j in C:
        mdl.add_constraint(
            mdl.sum(x[i, j, k] for i in N for k in K if i != j) == 1,
            ctname=f"visit_{j}"
        )

    for k in K:
        for h in C:
            mdl.add_constraint(
                mdl.sum(x[i, h, k] for i in N if i != h) ==
                mdl.sum(x[h, j, k] for j in N if j != h),
                ctname=f"flow_{h}_{k}"
            )
        mdl.add_constraint(
            mdl.sum(x[0, j, k] for j in C) == y[k], 
            ctname=f"depot_out_{k}"
        )
        mdl.add_constraint(
            mdl.sum(x[i, 0, k] for i in C) == y[k], 
            ctname=f"depot_in_{k}"
        )

    # ---------- 충전소 배정 & 방문 연계 ----------
    for s in S:
        mdl.add_constraint(
            mdl.sum(z[s, i, k] for i in C for k in K) == 1,
            ctname=f"sat_assign_{s}"
        )

    for s in S:
        for i in C:
            for k in K:
                mdl.add_constraint(
                    z[s, i, k] <= mdl.sum(x[j, i, k] for j in N if j != i),
                    ctname=f"sat_link_{s}_{i}_{k}"
                )

    # ---------- 조건 1: 용량 제약 (전기차=charge만, 배달=weight만) ----------
    for k in K:
        # 배달 용량: weight만 (일반고객)
        mdl.add_constraint(
            mdl.sum(demand_weight[j] * x[i, j, k] 
                    for i in N for j in C if i != j)
            <= cap_weight * y[k], 
            ctname=f"cap_weight_{k}"
        )

        # 충전 용량: charge만 (충전소)
        mdl.add_constraint(
            mdl.sum(sat_charge[s] * z[s, i, k] 
                    for s in S for i in C)
            <= cap_charge * y[k], 
            ctname=f"cap_charge_{k}"
        )

    # ---------- 조건 4: 시간 전이 제약 ----------
    M_ij = {}
    for i in N:
        for j in N:
            if i == j or j == 0: 
                continue
            M_ij[i, j] = max(0.0,
                             time_windows[i][1] + service_time[i]
                             + t[i, j] - time_windows[j][0])

    for k in K:
        for i in N:
            for j in N:
                if i == j or j == 0: 
                    continue
                mdl.add_constraint(
                    T[j, k] >= T[i, k] + service_time[i] + t[i, j]
                               - M_ij[i, j] * (1 - x[i, j, k]),
                    ctname=f"time_{i}_{j}_{k}"
                )

    for k in K:
        for i in N:
            e_i, l_i = time_windows[i]
            mdl.add_constraint(T[i, k] >= e_i, ctname=f"tw_e_{i}_{k}")
            mdl.add_constraint(T[i, k] <= l_i, ctname=f"tw_l_{i}_{k}")

    for k in K[:-1]:
        mdl.add_constraint(y[k] >= y[k + 1], ctname=f"sym_{k}")

    # ---------- 충전소 TSP 투어 제약 ----------
    for i in C:
        nodes = [-1] + list(S)
        for k in K:
            for s in S:
                mdl.add_constraint(
                    mdl.sum(p[a, s, i, k] for a in nodes if a != s)
                    == z[s, i, k], 
                    ctname=f"sat_in_{s}_{i}_{k}"
                )
                mdl.add_constraint(
                    mdl.sum(p[s, b, i, k] for b in nodes if b != s)
                    == z[s, i, k], 
                    ctname=f"sat_out_{s}_{i}_{k}"
                )

    for i in C:
        for k in K:
            mdl.add_constraint(
                mdl.sum(p[-1, s, i, k] for s in S)
                == mdl.sum(p[s, -1, i, k] for s in S),
                ctname=f"hub_bal_{i}_{k}"
            )
            mdl.add_constraint(
                mdl.sum(p[-1, s, i, k] for s in S) <= 1,
                ctname=f"hub_cap_{i}_{k}"
            )
            for s in S:
                mdl.add_constraint(
                    mdl.sum(p[-1, s2, i, k] for s2 in S) >= z[s, i, k],
                    ctname=f"hub_req_{s}_{i}_{k}"
                )

    # MTZ subtour 제거
    nSi = len(S)
    if nSi > 1:
        for i in C:
            for k in K:
                for s1 in S:
                    for s2 in S:
                        if s1 == s2: 
                            continue
                        mdl.add_constraint(
                            u[s2, i, k] >= u[s1, i, k] + 1
                                           - nSi * (1 - p[s1, s2, i, k]),
                            ctname=f"mtz_{s1}_{s2}_{i}_{k}"
                        )

    # ---------- 목적함수 ----------
    veh_distance = mdl.sum(
        d[i, j] * x[i, j, k]
        for i in N for j in N for k in K if i != j
    )
    sat_tour_distance = mdl.sum(
        tour_dist(a, b, i) * p[a, b, i, k]
        for (a, b, i, k) in p_keys
    )
    total_vehicles = mdl.sum(y[k] for k in K)

    mdl.minimize(
        alpha * total_vehicles
        + beta * (veh_distance + sat_tour_distance)
    )

    # ---------- 최적화 실행 ----------
    if time_limit is not None:
        mdl.parameters.timelimit = time_limit
    mdl.parameters.emphasis.mip = 1

    sol = mdl.solve(log_output=True)

    if sol is None:
        print("No solution found.")
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

        route, cur = [0], 0
        visited = set([0])
        max_iter = len(N) + 5

        for _ in range(max_iter):
            nxt = None
            for j in N:
                if j in visited and j != 0:
                    continue

                # x 변수는 i != j 인 경우만 생성했으므로,
                # 존재하지 않는 arc는 건너뛰어야 함
                if (cur, j, k) not in x:
                    continue

                if x[cur, j, k].solution_value > 0.5:
                    nxt = j
                    break

            if nxt is None:
                break

            route.append(nxt)
            if nxt == 0:
                break
            visited.add(nxt)
            cur = nxt

        arrivals = [T[i, k].solution_value for i in route]

        sat_tours = {}
        for i in route:
            if i == 0:
                continue
            if sum(p[-1, s, i, k].solution_value for s in S) < 0.5:
                continue

            tour, cur_n = [], -1
            visited_sat = set([-1])
            max_sat_iter = len(S) + 5

            for _ in range(max_sat_iter):
                nxt_n = None
                for b in [-1] + list(S):
                    if b in visited_sat and b != -1:
                        continue

                    # p 변수는 a != b 인 경우만 생성했으므로,
                    # 존재하지 않는 arc는 건너뛰어야 함
                    if (cur_n, b, i, k) not in p:
                        continue

                    if p[cur_n, b, i, k].solution_value > 0.5:
                        nxt_n = b
                        break

                if nxt_n is None or nxt_n == -1:
                    break

                tour.append(nxt_n)
                visited_sat.add(nxt_n)
                cur_n = nxt_n

            if tour:
                seq = [-1] + tour + [-1]
                tour_len = sum(
                    tour_dist(seq[t1], seq[t1 + 1], i)
                    for t1 in range(len(seq) - 1)
                )
                sat_tours[i] = {
                    "sequence": tour, 
                    "length": tour_len
                }

        routes[k] = {
            "route": route, 
            "arrivals": arrivals,
            "sat_tours": sat_tours
        }

    veh_dist_val = sum(
        d[i, j] * x[i, j, k].solution_value
        for i in N for j in N for k in K if i != j
    )
    sat_dist_val = sum(
        tour_dist(a, b, i) * p[a, b, i, k].solution_value
        for (a, b, i, k) in p_keys
    )

    return {
        "objective": sol.objective_value,
        "num_vehicles_used": sum(1 for k in K if y[k].solution_value > 0.5),
        "vehicle_distance": veh_dist_val,
        "satellite_tour_distance": sat_dist_val,
        "routes": routes,
    }


def create_html_map(coords, sat_coords, demand_weight, sat_charge, result, scenario_name):
    """조건 5: HTML 지도 시각화 - Leaflet/OpenStreetMap 버전

    Google Maps API 키/결제/도메인 제한 문제를 피하기 위해
    무료 타일 기반 Leaflet 지도로 HTML을 생성합니다.
    """

    center_lat = sum(c[1] for c in coords) / len(coords)
    center_lng = sum(c[0] for c in coords) / len(coords)

    customers = [
        {"lat": coords[i][1], "lng": coords[i][0], "id": i, "weight": demand_weight[i]}
        for i in range(1, len(coords))
    ]
    satellites = [
        {"lat": sat_coords[s][1], "lng": sat_coords[s][0], "id": s, "charge": sat_charge[s]}
        for s in range(len(sat_coords))
    ]

    route_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]
    routes = []
    if result:
        for k, info in result["routes"].items():
            routes.append({
                "vehicle": k,
                "path": [{"lat": coords[i][1], "lng": coords[i][0]} for i in info["route"]],
                "color": route_colors[k % len(route_colors)],
            })

    customers_json = json.dumps(customers, ensure_ascii=False)
    satellites_json = json.dumps(satellites, ensure_ascii=False)
    routes_json = json.dumps(routes, ensure_ascii=False)

    num_vehicles_used = result['num_vehicles_used'] if result else 0
    vehicle_distance = result['vehicle_distance'] if result else 0
    satellite_tour_distance = result['satellite_tour_distance'] if result else 0
    total_distance = vehicle_distance + satellite_tour_distance
    badge_class = 'easy' if 'Easy' in scenario_name else 'hard'

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>전기차 배달 최적화 - {scenario_name}</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

    <style>
        body {{ margin: 0; font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif; }}
        #map {{ height: 100vh; width: 100%; }}
        .info-box {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            z-index: 1000;
            max-width: 350px;
            font-size: 14px;
        }}
        .info-box h3 {{
            margin: 0 0 15px 0;
            font-size: 18px;
            color: #333;
            border-bottom: 2px solid #4285F4;
            padding-bottom: 10px;
        }}
        .stat-item {{
            margin: 8px 0;
            display: flex;
            justify-content: space-between;
        }}
        .stat-label {{ font-weight: 600; color: #555; }}
        .stat-value {{ color: #4285F4; font-weight: bold; }}
        .legend-item {{
            margin: 8px 0;
            display: flex;
            align-items: center;
            font-size: 13px;
        }}
        .legend-color {{
            width: 24px;
            height: 24px;
            margin-right: 10px;
            border-radius: 4px;
            border: 2px solid white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        }}
        hr {{ border: none; border-top: 1px solid #eee; margin: 15px 0; }}
        .scenario-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        .easy {{ background: #4CAF50; color: white; }}
        .hard {{ background: #F44336; color: white; }}
        .custom-marker {{
            width: 24px;
            height: 24px;
            border-radius: 50%;
            color: white;
            border: 2px solid white;
            box-shadow: 0 1px 4px rgba(0,0,0,0.45);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 11px;
            font-weight: bold;
        }}
        .depot-marker {{ background: #000000; width: 30px; height: 30px; font-size: 14px; }}
        .customer-marker {{ background: #4285F4; }}
        .satellite-marker {{ background: #EA4335; border-radius: 4px; }}
    </style>
</head>
<body>
    <div id="map"></div>
    <div class="info-box">
        <div class="scenario-badge {badge_class}">{scenario_name}</div>
        <h3>📊 최적화 결과</h3>
        <div class="stat-item">
            <span class="stat-label">사용 차량:</span>
            <span class="stat-value">{num_vehicles_used}대</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">차량 거리:</span>
            <span class="stat-value">{vehicle_distance:.2f}km</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">충전소 투어:</span>
            <span class="stat-value">{satellite_tour_distance:.2f}km</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">총 거리:</span>
            <span class="stat-value">{total_distance:.2f}km</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">일반고객:</span>
            <span class="stat-value">{len(coords)-1}개</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">충전소:</span>
            <span class="stat-value">{len(sat_coords)}개</span>
        </div>

        <hr>

        <h4 style="margin: 10px 0 8px 0; font-size: 15px;">범례</h4>
        <div class="legend-item"><div class="legend-color" style="background:#000000;"></div><span>Depot (시청)</span></div>
        <div class="legend-item"><div class="legend-color" style="background:#4285F4;"></div><span>일반고객 (배달)</span></div>
        <div class="legend-item"><div class="legend-color" style="background:#EA4335;"></div><span>충전소</span></div>
        <div class="legend-item"><div class="legend-color" style="background:#FF6B6B;"></div><span>차량 경로</span></div>
    </div>

    <script>
        const map = L.map('map').setView([{center_lat}, {center_lng}], 14);

        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            maxZoom: 19,
            attribution: '&copy; OpenStreetMap contributors'
        }}).addTo(map);

        function makeIcon(className, label) {{
            return L.divIcon({{
                className: '',
                html: `<div class="custom-marker ${{className}}">${{label}}</div>`,
                iconSize: [30, 30],
                iconAnchor: [15, 15],
                popupAnchor: [0, -12]
            }});
        }}

        L.marker([{coords[0][1]}, {coords[0][0]}], {{ icon: makeIcon('depot-marker', 'D') }})
            .addTo(map)
            .bindPopup('Depot (서울시청)');

        const customers = {customers_json};
        customers.forEach(c => {{
            L.marker([c.lat, c.lng], {{ icon: makeIcon('customer-marker', String(c.id)) }})
                .addTo(map)
                .bindPopup(`C${{c.id}} / 배달 ${{c.weight}}kg`);
        }});

        const satellites = {satellites_json};
        satellites.forEach(s => {{
            L.marker([s.lat, s.lng], {{ icon: makeIcon('satellite-marker', `S${{s.id}}`) }})
                .addTo(map)
                .bindPopup(`S${{s.id}} / 충전 ${{s.charge}}kWh`);
        }});

        const routes = {routes_json};
        routes.forEach(r => {{
            const path = r.path.map(p => [p.lat, p.lng]);
            L.polyline(path, {{ color: r.color, weight: 4, opacity: 0.75 }})
                .addTo(map)
                .bindPopup(`차량 ${{r.vehicle + 1}} 경로`);
        }});

        const bounds = [];
        bounds.push([{coords[0][1]}, {coords[0][0]}]);
        customers.forEach(c => bounds.push([c.lat, c.lng]));
        satellites.forEach(s => bounds.push([s.lat, s.lng]));
        if (bounds.length > 0) {{
            map.fitBounds(bounds, {{ padding: [40, 40] }});
        }}
    </script>
</body>
</html>"""

    return html_content


# =============================================================================
# Scenario 1: Easy Problem (4 customers, 3 satellites, 30 sec)
# =============================================================================

def scenario_easy():
    print("\n" + "="*80)
    print("SCENARIO 1: EASY PROBLEM")
    print("="*80)
    print("설정: 4개 고객, 3개 충전소, Time Limit 30초")
    print("-"*80)

    # 서울 중구 중심부 일부 지역만 사용
    coords = [
        (126.9779, 37.5663),  # 0: Depot (서울시청)
        (126.9850, 37.5700),  # 1: 광화문
        (126.9920, 37.5600),  # 2: 명동
        (126.9750, 37.5720),  # 3: 덕수궁
        (126.9900, 37.5680),  # 4: 시청역
    ]

    demand_weight = [0, 12, 15, 10, 14]
    demand_charge = [0,  0,  0,  0,  0]

    time_windows  = [
        (0, 300),      
        (10, 80), (20, 90), (15, 85), (25, 95)
    ]

    service_time_normal = [0, 5, 5, 5, 5]

    # 충전소: 일반고객 근처에 배치 (같은 건물 아닌 근처)
    sat_coords = [
        (126.9855, 37.5705),  # s0: 광화문 근처 (C1 근처, 약 50m 떨어짐)
        (126.9925, 37.5605),  # s1: 명동 근처 (C2 근처)
        (126.9905, 37.5685),  # s2: 시청역 근처 (C4 근처)
    ]

    sat_weight = [0, 0, 0]
    sat_charge = [10, 8, 12]  # kWh
    sat_service_time = [20, 16, 24]  # 충전량에 비례

    res = solve_cvrptw_with_satellite_tours(
        coords=coords,
        demand_weight=demand_weight, 
        demand_charge=demand_charge,
        time_windows=time_windows, 
        service_time=service_time_normal,
        sat_coords=sat_coords,
        sat_weight=sat_weight, 
        sat_charge=sat_charge,
        sat_service_time=sat_service_time,
        num_vehicles=2, 
        cap_weight=30,
        cap_charge=25,
        alpha=1000.0, 
        beta=1.0, 
        time_limit=30,
    )

    if res:
        print("\n[최적화 결과]")
        print(f"목적함수: {res['objective']:.2f}")
        print(f"사용 차량: {res['num_vehicles_used']}대")
        print(f"차량 거리: {res['vehicle_distance']:.4f}km")
        print(f"충전소 투어: {res['satellite_tour_distance']:.4f}km")

        print("\n[차량별 경로]")
        for k, info in res["routes"].items():
            print(f"\n차량 {k+1}: {' → '.join(map(str, info['route']))}")
            if info["sat_tours"]:
                for hub, t_info in info["sat_tours"].items():
                    sat_seq = ', '.join(f'S{s}' for s in t_info["sequence"])
                    print(f"  ★ Hub C{hub} 충전투어: [{sat_seq}] ({t_info['length']:.4f}km)")

        html = create_html_map(coords, sat_coords, demand_weight, sat_charge, res, "Scenario 1: Easy")
        os.makedirs('output', exist_ok=True)
        with open(os.path.join('output', 'scenario_easy_map.html'), 'w', encoding='utf-8') as f:
            f.write(html)
        print("\n✅ HTML 지도 생성: scenario_easy_map.html")

    return res


# =============================================================================
# Scenario 2: Hard Problem (12 customers, 10 satellites, 300 sec)
# =============================================================================

def scenario_hard():
    print("\n" + "="*80)
    print("SCENARIO 2: HARD PROBLEM")
    print("="*80)
    print("설정: 12개 고객, 10개 충전소, Time Limit 300초")
    print("-"*80)

    # 서울 중구 전역 사용
    coords = [
        (126.9779, 37.5663),  # 0: Depot (서울시청)
        (126.9850, 37.5700),  # 1: 광화문
        (127.0000, 37.5650),  # 2: 을지로
        (126.9920, 37.5600),  # 3: 명동
        (127.0050, 37.5550),  # 4: 남산
        (126.9750, 37.5720),  # 5: 덕수궁
        (126.9900, 37.5680),  # 6: 시청역
        (127.0080, 37.5680),  # 7: 동대문
        (126.9880, 37.5620),  # 8: 충무로
        (127.0020, 37.5600),  # 9: 필동
        (126.9800, 37.5640),  # 10: 소공동
        (127.0100, 37.5630),  # 11: 신당동
        (126.9930, 37.5730),  # 12: 종로
    ]

    demand_weight = [0, 10, 15, 8, 12, 7, 20, 9, 14, 11, 13, 16, 10]
    demand_charge = [0,  0,  0, 0,  0, 0,  0, 0,  0,  0,  0,  0,  0]

    time_windows  = [
        (0, 400),
        (10, 80), (20, 100), (30, 110), (40, 130),
        (15, 90), (25, 105), (50, 140), (35, 115),
        (45, 125), (12, 85), (55, 145), (18, 95)
    ]

    service_time_normal = [0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]

    # 충전소: 다양한 고객 근처에 배치
    sat_coords = [
        (126.9855, 37.5705),  # s0: 광화문 근처 (C1)
        (126.9845, 37.5695),  # s1: 광화문 다른 위치 (C1)
        (127.0005, 37.5655),  # s2: 을지로 근처 (C2)
        (126.9925, 37.5605),  # s3: 명동 근처 (C3)
        (126.9918, 37.5598),  # s4: 명동 다른 위치 (C3)
        (127.0055, 37.5555),  # s5: 남산 근처 (C4)
        (126.9755, 37.5725),  # s6: 덕수궁 근처 (C5)
        (127.0085, 37.5685),  # s7: 동대문 근처 (C7)
        (126.9885, 37.5625),  # s8: 충무로 근처 (C8)
        (127.0105, 37.5635),  # s9: 신당동 근처 (C11)
    ]

    sat_weight = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    sat_charge = [12, 8, 10, 9, 7, 11, 10, 13, 8, 9]  # kWh
    sat_service_time = [24, 16, 20, 18, 14, 22, 20, 26, 16, 18]  # 충전량에 비례

    res = solve_cvrptw_with_satellite_tours(
        coords=coords,
        demand_weight=demand_weight, 
        demand_charge=demand_charge,
        time_windows=time_windows, 
        service_time=service_time_normal,
        sat_coords=sat_coords,
        sat_weight=sat_weight, 
        sat_charge=sat_charge,
        sat_service_time=sat_service_time,
        num_vehicles=4, 
        cap_weight=50,
        cap_charge=30,
        alpha=1000.0, 
        beta=1.0, 
        time_limit=300,
    )

    if res:
        print("\n[최적화 결과]")
        print(f"목적함수: {res['objective']:.2f}")
        print(f"사용 차량: {res['num_vehicles_used']}대")
        print(f"차량 거리: {res['vehicle_distance']:.4f}km")
        print(f"충전소 투어: {res['satellite_tour_distance']:.4f}km")

        print("\n[차량별 경로]")
        for k, info in res["routes"].items():
            print(f"\n차량 {k+1}: {' → '.join(map(str, info['route']))}")
            if info["sat_tours"]:
                for hub, t_info in info["sat_tours"].items():
                    sat_seq = ', '.join(f'S{s}' for s in t_info["sequence"])
                    print(f"  ★ Hub C{hub} 충전투어: [{sat_seq}] ({t_info['length']:.4f}km)")

        html = create_html_map(coords, sat_coords, demand_weight, sat_charge, res, "Scenario 2: Hard")
        os.makedirs('output', exist_ok=True)
        with open(os.path.join('output', 'scenario_hard_map.html'), 'w', encoding='utf-8') as f:
            f.write(html)
        print("\n✅ HTML 지도 생성: scenario_hard_map.html")

    return res


# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("전기차 배달 최적화 시스템 - 두 가지 시나리오")
    print("="*80)

    # Scenario 1: Easy
    res_easy = scenario_easy()

    # Scenario 2: Hard
    res_hard = scenario_hard()

    print("\n" + "="*80)
    print("시나리오 비교")
    print("="*80)

    if res_easy and res_hard:
        print(f"\n{'구분':<15} {'Easy':<20} {'Hard':<20}")
        print("-"*55)
        print(f"{'고객 수':<15} {4:<20} {12:<20}")
        print(f"{'충전소 수':<15} {3:<20} {10:<20}")
        print(f"{'Time Limit':<15} {'30초':<20} {'300초':<20}")
        print(f"{'사용 차량':<15} {res_easy['num_vehicles_used']:<20} {res_hard['num_vehicles_used']:<20}")
        print(f"{'목적함수':<15} {res_easy['objective']:<20.2f} {res_hard['objective']:<20.2f}")
        print(f"{'총 거리(km)':<15} {res_easy['vehicle_distance'] + res_easy['satellite_tour_distance']:<20.4f} {res_hard['vehicle_distance'] + res_hard['satellite_tour_distance']:<20.4f}")
