"""
CVRPTW + Dual Capacity + Satellite TSP Tours (전기차 배달 최적화)
--------------------------------------------
수정 사항:
1. 전기차는 weight 제외, 배달(일반고객)은 charge 제외
2. 딸린고객이 가장 가까운 hub이 아닌 목적함수 최소화 hub 선택
3. depot을 중간/최적 거점으로 설정 (경로 중간에 복귀 가능)
4. 서비스타임이 충전수요(딸린고객)에 따라 차등 적용
5. HTML 지도 시각화
"""

from docplex.mp.model import Model
import math


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
    - sat_charge: 충전소 충전 용량 (kW)
    - sat_service_time: 충전소별 서비스 타임 (분, 충전량에 비례)
    - num_vehicles: 차량 수
    - cap_weight: 차량 배달 용량 (kg)
    - cap_charge: 차량 충전 용량 (kW)
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
    feas = {s: C for s in S}  # 모든 일반고객이 hub 후보
    hub_sats = {i: S for i in C}  # 모든 일반고객이 모든 충전소 수용 가능

    def tour_dist(a, b, i):
        """투어 거리: hub i에서 충전소 a, b 간 거리"""
        pa = coords[i] if a == -1 else sat_coords[a]
        pb = coords[i] if b == -1 else sat_coords[b]
        return math.hypot(pa[0] - pb[0], pa[1] - pb[1])

    mdl = Model(name="CVRPTW_EV_Delivery")

    # ---------- 변수 ----------
    # x[i,j,k]: 차량 k가 i에서 j로 이동
    x = mdl.binary_var_dict(
        ((i, j, k) for i in N for j in N for k in K if i != j), 
        name="x"
    )

    # y[k]: 차량 k 사용 여부
    y = mdl.binary_var_dict(K, name="y")

    # T[i,k]: 차량 k가 노드 i에 도착하는 시간
    T = mdl.continuous_var_dict(((i, k) for i in N for k in K), lb=0, name="T")

    # z[s,i,k]: 충전소 s가 차량 k의 hub i에 배정
    z = mdl.binary_var_dict(
        ((s, i, k) for s in S for i in C for k in K), 
        name="z"
    )

    # p[a,b,i,k]: hub i에서 충전소 투어 아크 (a→b)
    p_keys = []
    for i in C:
        nodes = [-1] + list(S)  # -1: hub, 0~m-1: 충전소
        for a in nodes:
            for b in nodes:
                if a != b:
                    for k in K:
                        p_keys.append((a, b, i, k))
    p = mdl.binary_var_dict(p_keys, name="p")

    # u[s,i,k]: MTZ 순서 변수 (충전소 subtour 제거)
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
    # 각 일반고객은 정확히 한 번 방문
    for j in C:
        mdl.add_constraint(
            mdl.sum(x[i, j, k] for i in N for k in K if i != j) == 1,
            ctname=f"visit_{j}"
        )

    # Flow conservation
    for k in K:
        for h in C:
            mdl.add_constraint(
                mdl.sum(x[i, h, k] for i in N if i != h) ==
                mdl.sum(x[h, j, k] for j in N if j != h),
                ctname=f"flow_{h}_{k}"
            )

        # 조건 3: depot 중간 복귀 허용 (출발 횟수 >= 도착 횟수)
        # 차량이 depot을 여러 번 방문 가능
        mdl.add_constraint(
            mdl.sum(x[0, j, k] for j in C) >= mdl.sum(x[i, 0, k] for i in C),
            ctname=f"depot_balance_{k}"
        )

        # 차량 사용 시 최소 1회 출발
        mdl.add_constraint(
            mdl.sum(x[0, j, k] for j in C) >= y[k],
            ctname=f"depot_out_{k}"
        )

    # ---------- 충전소 배정 & 방문 연계 ----------
    # 각 충전소는 정확히 한 hub-차량에 배정
    for s in S:
        mdl.add_constraint(
            mdl.sum(z[s, i, k] for i in C for k in K) == 1,
            ctname=f"sat_assign_{s}"
        )

    # 충전소가 배정되려면 해당 hub를 방문해야 함
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

    # ---------- 조건 4: 시간 전이 제약 (서비스타임 차등 적용) ----------
    # Big-M 계산
    M_ij = {}
    for i in N:
        for j in N:
            if i == j or j == 0: 
                continue
            M_ij[i, j] = max(0.0,
                             time_windows[i][1] + service_time[i]
                             + t[i, j] - time_windows[j][0])

    # 일반고객 간 시간 전이
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

    # 타임윈도우 제약
    for k in K:
        for i in N:
            e_i, l_i = time_windows[i]
            mdl.add_constraint(T[i, k] >= e_i, ctname=f"tw_e_{i}_{k}")
            mdl.add_constraint(T[i, k] <= l_i, ctname=f"tw_l_{i}_{k}")

    # 차량 대칭성 제거
    for k in K[:-1]:
        mdl.add_constraint(y[k] >= y[k + 1], ctname=f"sym_{k}")

    # ---------- 충전소 TSP 투어 제약 ----------
    # (T1, T2) 충전소 in/out-degree = z
    for i in C:
        nodes = [-1] + list(S)
        for k in K:
            for s in S:
                # 충전소 s로 들어오는 아크 = z
                mdl.add_constraint(
                    mdl.sum(p[a, s, i, k] for a in nodes if a != s)
                    == z[s, i, k], 
                    ctname=f"sat_in_{s}_{i}_{k}"
                )
                # 충전소 s에서 나가는 아크 = z
                mdl.add_constraint(
                    mdl.sum(p[s, b, i, k] for b in nodes if b != s)
                    == z[s, i, k], 
                    ctname=f"sat_out_{s}_{i}_{k}"
                )

    # (T3) hub: out = in, at most 1
    for i in C:
        for k in K:
            # hub 진입 = hub 이탈
            mdl.add_constraint(
                mdl.sum(p[-1, s, i, k] for s in S)
                == mdl.sum(p[s, -1, i, k] for s in S),
                ctname=f"hub_bal_{i}_{k}"
            )
            # hub는 최대 1회 투어 시작
            mdl.add_constraint(
                mdl.sum(p[-1, s, i, k] for s in S) <= 1,
                ctname=f"hub_cap_{i}_{k}"
            )
            # 충전소가 배정되면 hub에서 투어 시작
            for s in S:
                mdl.add_constraint(
                    mdl.sum(p[-1, s2, i, k] for s2 in S) >= z[s, i, k],
                    ctname=f"hub_req_{s}_{i}_{k}"
                )

    # (T4) MTZ subtour 제거 (충전소 간)
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
    # 차량 이동거리
    veh_distance = mdl.sum(
        d[i, j] * x[i, j, k]
        for i in N for j in N for k in K if i != j
    )

    # 충전소 투어 거리 (조건 2: 최적화로 hub 선택)
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
    mdl.parameters.emphasis.mip = 1  # feasibility 우선

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

        # 차량 경로
        route, cur = [0], 0
        while True:
            nxt = None
            for j in N:
                if j == cur: 
                    continue
                if x[cur, j, k].solution_value > 0.5:
                    nxt = j
                    break
            if nxt is None or nxt == 0:
                route.append(0)
                break
            route.append(nxt)
            cur = nxt

        arrivals = [T[i, k].solution_value for i in route]

        # 각 hub의 충전소 투어 추출
        sat_tours = {}
        for i in route:
            if i == 0:
                continue
            if sum(p[-1, s, i, k].solution_value for s in S) < 0.5:
                continue

            tour, cur_n = [], -1
            while True:
                nxt_n = None
                for b in [-1] + list(S):
                    if b == cur_n: 
                        continue
                    if p[cur_n, b, i, k].solution_value > 0.5:
                        nxt_n = b
                        break
                if nxt_n is None or nxt_n == -1:
                    break
                tour.append(nxt_n)
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


def create_html_map(coords, sat_coords, demand_weight, sat_charge, result):
    """조건 5: HTML 지도 시각화

    Google Maps API 키 오류를 피하기 위해 Leaflet + OpenStreetMap 기반으로 생성합니다.
    별도 API 키/결제 설정 없이 HTML 파일을 바로 열어 확인할 수 있습니다.
    """
    import json

    # 중심점 계산
    center_lat = sum(c[1] for c in coords) / len(coords)
    center_lng = sum(c[0] for c in coords) / len(coords)

    customers_data = [
        {"lat": coords[i][1], "lng": coords[i][0], "id": i, "weight": demand_weight[i]}
        for i in range(1, len(coords))
    ]

    satellites_data = [
        {"lat": sat_coords[s][1], "lng": sat_coords[s][0], "id": s, "charge": sat_charge[s]}
        for s in range(len(sat_coords))
    ]

    route_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]
    routes_data = [
        {
            "vehicle": k,
            "path": [[coords[i][1], coords[i][0]] for i in info["route"]],  # Leaflet: [lat, lng]
            "color": route_colors[k % len(route_colors)],
        }
        for k, info in (result["routes"].items() if result else {}.items())
    ]

    # 충전소 투어도 점선으로 표시
    sat_tours_data = []
    if result:
        for k, info in result["routes"].items():
            color = route_colors[k % len(route_colors)]
            for hub, t_info in info.get("sat_tours", {}).items():
                path = [[coords[hub][1], coords[hub][0]]]
                path.extend([[sat_coords[s][1], sat_coords[s][0]] for s in t_info["sequence"]])
                path.append([coords[hub][1], coords[hub][0]])
                sat_tours_data.append({
                    "vehicle": k,
                    "hub": hub,
                    "path": path,
                    "color": color,
                    "length": t_info["length"],
                })

    customers_json = json.dumps(customers_data, ensure_ascii=False)
    satellites_json = json.dumps(satellites_data, ensure_ascii=False)
    routes_json = json.dumps(routes_data, ensure_ascii=False)
    sat_tours_json = json.dumps(sat_tours_data, ensure_ascii=False)

    vehicle_distance = result["vehicle_distance"] if result else 0
    satellite_tour_distance = result["satellite_tour_distance"] if result else 0
    total_distance = vehicle_distance + satellite_tour_distance
    num_vehicles_used = result["num_vehicles_used"] if result else 0

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>전기차 배달 최적화 경로</title>
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
            max-width: 320px;
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
        .marker-label {{
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            border: 2px solid white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.35);
        }}
        .depot-marker {{
            width: 30px;
            height: 30px;
            background: #000000;
            border-radius: 50%;
            font-size: 16px;
        }}
        .customer-marker {{
            width: 24px;
            height: 24px;
            background: #4285F4;
            border-radius: 50%;
            font-size: 11px;
        }}
        .satellite-marker {{
            width: 26px;
            height: 26px;
            background: #EA4335;
            border-radius: 6px;
            font-size: 14px;
        }}
        hr {{ border: none; border-top: 1px solid #eee; margin: 15px 0; }}
    </style>
</head>
<body>
    <div id="map"></div>
    <div class="info-box">
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

        <hr>

        <h4 style="margin: 10px 0 8px 0; font-size: 15px;">범례</h4>
        <div class="legend-item"><div class="legend-color" style="background:#000000;"></div><span>🏢 Depot (시청)</span></div>
        <div class="legend-item"><div class="legend-color" style="background:#4285F4;"></div><span>📦 일반고객 (배달)</span></div>
        <div class="legend-item"><div class="legend-color" style="background:#EA4335;"></div><span>⚡ 충전소</span></div>
        <div class="legend-item"><div class="legend-color" style="background:#FF6B6B;"></div><span>🚗 차량 1 경로</span></div>
        <div class="legend-item"><div class="legend-color" style="background:#4ECDC4;"></div><span>🚗 차량 2 경로</span></div>
        <div class="legend-item"><div class="legend-color" style="background:#45B7D1;"></div><span>🚗 차량 3 경로</span></div>
        <div class="legend-item"><div class="legend-color" style="background:white; border:2px dashed #555;"></div><span>점선: 충전소 투어</span></div>
    </div>

    <script>
        const map = L.map('map').setView([{center_lat}, {center_lng}], 14);

        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            maxZoom: 19,
            attribution: '&copy; OpenStreetMap contributors'
        }}).addTo(map);

        function divIcon(className, html) {{
            return L.divIcon({{
                className: '',
                html: `<div class="marker-label ${{className}}">${{html}}</div>`,
                iconSize: [30, 30],
                iconAnchor: [15, 15],
                popupAnchor: [0, -15]
            }});
        }}

        // Depot 마커
        L.marker([{coords[0][1]}, {coords[0][0]}], {{ icon: divIcon('depot-marker', '🏢') }})
            .addTo(map)
            .bindPopup('Depot (시청)');

        // 일반고객 마커
        const customers = {customers_json};
        customers.forEach(c => {{
            L.marker([c.lat, c.lng], {{ icon: divIcon('customer-marker', String(c.id)) }})
                .addTo(map)
                .bindPopup(`고객 C${{c.id}}<br>배달 ${{c.weight}}kg`);
        }});

        // 충전소 마커
        const satellites = {satellites_json};
        satellites.forEach(s => {{
            L.marker([s.lat, s.lng], {{ icon: divIcon('satellite-marker', '⚡') }})
                .addTo(map)
                .bindPopup(`충전소 S${{s.id}}<br>충전 ${{s.charge}}kW`);
        }});

        // 차량 경로
        const routes = {routes_json};
        routes.forEach(r => {{
            L.polyline(r.path, {{
                color: r.color,
                weight: 5,
                opacity: 0.75
            }})
            .addTo(map)
            .bindPopup(`차량 ${{r.vehicle + 1}} 경로`);
        }});

        // 충전소 투어 경로
        const satTours = {sat_tours_json};
        satTours.forEach(t => {{
            L.polyline(t.path, {{
                color: t.color,
                weight: 3,
                opacity: 0.85,
                dashArray: '6, 8'
            }})
            .addTo(map)
            .bindPopup(`차량 ${{t.vehicle + 1}} - Hub C${{t.hub}} 충전소 투어<br>거리: ${{t.length.toFixed(4)}}km`);
        }});

        // 전체 경로가 보이도록 지도 범위 조정
        const allPoints = [];
        routes.forEach(r => allPoints.push(...r.path));
        satTours.forEach(t => allPoints.push(...t.path));
        customers.forEach(c => allPoints.push([c.lat, c.lng]));
        satellites.forEach(s => allPoints.push([s.lat, s.lng]));
        allPoints.push([{coords[0][1]}, {coords[0][0]}]);

        if (allPoints.length > 0) {{
            map.fitBounds(L.latLngBounds(allPoints), {{ padding: [40, 40] }});
        }}
    </script>
</body>
</html>"""

    return html_content

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 실제 서울 중구 좌표 기반 데이터 (경도, 위도)
    # Depot: 서울시청 (126.9779, 37.5663)
    coords = [
        (126.9779, 37.5663),  # 0: Depot (시청)
        (126.9850, 37.5700),  # 1: 광화문 (일반배달)
        (127.0000, 37.5650),  # 2: 을지로 (일반배달)
        (126.9920, 37.5600),  # 3: 명동 (일반배달)
        (127.0050, 37.5550),  # 4: 남산 (일반배달)
        (126.9750, 37.5720),  # 5: 덕수궁 (일반배달)
        (126.9900, 37.5680),  # 6: 시청역 (일반배달)
        (127.0080, 37.5680),  # 7: 동대문 (일반배달)
        (126.9880, 37.5620),  # 8: 충무로 (일반배달)
    ]

    # 조건 1: 일반고객(배달)은 weight만, charge=0
    demand_weight = [0, 10, 15, 8, 12, 7, 20, 9, 14]
    demand_charge = [0,  0,  0, 0,  0, 0,  0, 0,  0]

    time_windows  = [
        (0, 300),      # Depot (넓은 시간창)
        (10, 60), (20, 80), (30, 90), (40, 100),
        (15, 70), (25, 85), (50, 120), (35, 95)
    ]

    # 일반고객 서비스타임 (배달은 짧게)
    service_time_normal = [0, 5, 5, 5, 5, 5, 5, 5, 5]

    # 충전소 (딸린고객): 실제 건물 근처 좌표 (일반고객 근처에 배치)
    sat_coords = [
        (126.9855, 37.5705),  # s0: 광화문 근처 충전소
        (126.9845, 37.5695),  # s1: 광화문 근처 다른 충전소
        (127.0005, 37.5655),  # s2: 을지로 근처 충전소
        (126.9925, 37.5605),  # s3: 명동 근처 충전소
        (126.9918, 37.5598),  # s4: 명동 근처 다른 충전소
        (126.9755, 37.5725),  # s5: 덕수궁 근처 충전소
        (126.9885, 37.5625),  # s6: 충무로 근처 충전소
    ]

    # 조건 1: 충전소(딸린고객)는 charge만, weight=0
    sat_weight = [0, 0, 0, 0, 0, 0, 0]
    sat_charge = [5, 3, 4, 3, 2, 4, 3]  # 충전 용량 (kW)

    # 조건 4: 충전소별 서비스타임 (충전량에 비례하여 설정)
    sat_service_time = [15, 10, 12, 10, 8, 12, 10]  # 분

    print("="*70)
    print("전기차 배달 최적화 시스템")
    print("="*70)
    print("\n[시스템 설정]")
    print(f"- 일반고객(배달): {len(coords)-1}개")
    print(f"- 충전소: {len(sat_coords)}개")
    print(f"- 차량 수: 3대")
    print(f"- 배달 용량: 50kg / 충전 용량: 15kW")
    print("\n[최적화 실행 중...]\n")

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
        num_vehicles=3, 
        cap_weight=50,   # 배달 weight 용량
        cap_charge=15,   # 충전 charge 용량
        alpha=1000.0, 
        beta=1.0, 
        time_limit=180,
    )

    if res:
        print("\n" + "="*70)
        print("최적화 결과")
        print("="*70)
        print(f"목적함수 값             : {res['objective']:.2f}")
        print(f"사용 차량 수           : {res['num_vehicles_used']}대")
        print(f"차량 이동거리          : {res['vehicle_distance']:.4f}km")
        print(f"충전소 투어거리        : {res['satellite_tour_distance']:.4f}km")
        print(f"총 거리                : {res['vehicle_distance'] + res['satellite_tour_distance']:.4f}km")

        print("\n" + "-"*70)
        print("차량별 상세 경로")
        print("-"*70)
        for k, info in res["routes"].items():
            print(f"\n[차량 {k+1}]")
            print(f"  경로: {' → '.join(map(str, info['route']))}")
            print(f"  도착시간: {[f'{t:.1f}' for t in info['arrivals']]}")

            if info["sat_tours"]:
                print(f"  ★ 충전소 투어:")
                for hub, t_info in info["sat_tours"].items():
                    sat_seq = ' → '.join(f'충전소{s}' for s in t_info["sequence"])
                    print(f"     Hub C{hub}: {hub} → {sat_seq} → {hub}")
                    print(f"     투어거리: {t_info['length']:.4f}km")

        # 조건 5: HTML 지도 생성
        print("\n" + "="*70)
        print("HTML 지도 생성")
        print("="*70)
        html = create_html_map(coords, sat_coords, demand_weight, sat_charge, res)

        with open('route_map.html', 'w', encoding='utf-8') as f:
            f.write(html)

        print("✅ HTML 지도 파일 생성 완료: route_map.html")
        print("\n[참고]")
        print("- 일반고객: 배달 무게(weight) 용량만 적용")
        print("- 충전소: 충전 용량(charge)만 적용")
        print("- 충전소는 최적 hub 자동 선택 (거리 제약 없음)")
        print("- Depot 중간 복귀 허용")
        print("- 서비스타임 충전량별 차등 적용")

    else:
        print("\n❌ 최적화 실패")
