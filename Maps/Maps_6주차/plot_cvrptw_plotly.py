"""
CVRPTW + Satellite TSP 해 인터랙티브 시각화 (Plotly)
- Row 1: 경로 지도 (차량 경로 + hub별 딸린고객 TSP 투어 점선)
- Row 2: 차량별 용량 활용률 (무게 / 충전)
- Row 3: 차량별 일정 Gantt 차트
"""

import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors as pc


def _palette(n):
    base = pc.qualitative.Plotly + pc.qualitative.Dark24
    return [base[i % len(base)] for i in range(n)]


def plot_solution_plotly(
    result,
    coords, demand_weight, demand_charge, time_windows, service_time,
    sat_coords, sat_weight, sat_charge,
    cap_weight, cap_charge,
    R=None, save_html=None,
):
    routes = result["routes"]
    vehicles = sorted(routes.keys())
    colors = _palette(max(len(vehicles), 1))
    v_color = {k: colors[idx] for idx, k in enumerate(vehicles)}

    # --- 적재량 집계 ---
    load_w, load_c = {}, {}
    for k, info in routes.items():
        regs = [i for i in info["route"] if i != 0]
        w = sum(demand_weight[i] for i in regs)
        c = sum(demand_charge[i] for i in regs)
        for hub, tinfo in info["sat_tours"].items():
            for s in tinfo["sequence"]:
                w += sat_weight[s]; c += sat_charge[s]
        load_w[k], load_c[k] = w, c

    # --- 딸린고객이 어느 (hub, 차량) 의 몇 번째 방문인지 매핑 ---
    sat_info = {}   # s -> (vehicle, hub, position, tour_len)
    for k, info in routes.items():
        for hub, tinfo in info["sat_tours"].items():
            for pos, s in enumerate(tinfo["sequence"], 1):
                sat_info[s] = (k, hub, pos, tinfo["length"])

    # --- 고객별 방문 차량/시각 매핑 ---
    cust_to_veh, cust_arrival = {}, {}
    for k, info in routes.items():
        for node, t_arr in zip(info["route"], info["arrivals"]):
            if node != 0:
                cust_to_veh[node] = k
                cust_arrival[node] = t_arr

    # --- 서브플롯 ---
    fig = make_subplots(
        rows=3, cols=2,
        row_heights=[0.55, 0.18, 0.27],
        column_widths=[0.5, 0.5],
        specs=[[{"colspan": 2, "type": "xy"}, None],
               [{"type": "xy"}, {"type": "xy"}],
               [{"colspan": 2, "type": "xy"}, None]],
        subplot_titles=("Routes (hover · pan · zoom)",
                        "Weight Utilization (%)", "Charge Utilization (%)",
                        "Vehicle Schedule (Gantt)"),
        vertical_spacing=0.09, horizontal_spacing=0.10,
    )

    # =============================================================
    # Row 1: 경로 지도
    # =============================================================
    # 반경 R
    if R is not None:
        visited = {i for info in routes.values() for i in info["route"] if i != 0}
        theta = [2 * math.pi * t / 60 for t in range(61)]
        for i in visited:
            xi, yi = coords[i]
            fig.add_trace(go.Scatter(
                x=[xi + R * math.cos(a) for a in theta],
                y=[yi + R * math.sin(a) for a in theta],
                mode="lines",
                line=dict(color="lightgray", width=1, dash="dot"),
                hoverinfo="skip", showlegend=False, opacity=0.45,
            ), row=1, col=1)

    # depot
    dx, dy = coords[0]
    fig.add_trace(go.Scatter(
        x=[dx], y=[dy], mode="markers+text",
        marker=dict(symbol="square", size=18, color="black",
                    line=dict(color="white", width=2)),
        text=["Depot"], textposition="top center",
        hovertemplate=f"<b>Depot</b><br>({dx}, {dy})<extra></extra>",
        name="Depot"), row=1, col=1)

    # 일반 고객
    cx, cy, ctxt, chover = [], [], [], []
    for i in range(1, len(coords)):
        xi, yi = coords[i]
        k = cust_to_veh.get(i, None)
        t_arr = cust_arrival.get(i, None)
        e, l = time_windows[i]
        # 이 hub 에 붙은 딸린고객 목록
        attached = [s for s, (kk, hub, _, _) in sat_info.items() if hub == i]
        att_txt = (", ".join(f"s{s}" for s in attached)
                   if attached else "none")
        hover = (f"<b>Customer {i}</b><br>"
                 f"Position: ({xi}, {yi})<br>"
                 f"Vehicle: {k if k is not None else '-'}<br>"
                 f"Arrival: {t_arr:.2f}<br>" if t_arr is not None else
                 f"<b>Customer {i}</b><br>Position: ({xi}, {yi})<br>")
        hover += (f"Time window: [{e}, {l}]<br>"
                  f"Service: {service_time[i]}<br>"
                  f"Demand (w, c): ({demand_weight[i]}, {demand_charge[i]})<br>"
                  f"Satellites attached: {att_txt}<extra></extra>")
        cx.append(xi); cy.append(yi); ctxt.append(f"C{i}"); chover.append(hover)

    fig.add_trace(go.Scatter(
        x=cx, y=cy, mode="markers+text",
        marker=dict(symbol="circle", size=13, color="#4C78A8",
                    line=dict(color="white", width=1.5)),
        text=ctxt, textposition="top right", textfont=dict(size=10),
        hovertext=chover, hoverinfo="text",
        name="Regular customer"), row=1, col=1)

    # 딸린 고객
    sx, sy, stxt, shover = [], [], [], []
    for s, (xs, ys) in enumerate(sat_coords):
        info_s = sat_info.get(s)
        if info_s is not None:
            k_s, hub_s, pos_s, tlen_s = info_s
            hover = (f"<b>Satellite {s}</b><br>"
                     f"Position: ({xs}, {ys})<br>"
                     f"Hub: C{hub_s}<br>"
                     f"Vehicle: {k_s}<br>"
                     f"Tour position: {pos_s}<br>"
                     f"Hub tour length: {tlen_s:.2f}<br>"
                     f"Demand (w, c): ({sat_weight[s]}, {sat_charge[s]})"
                     f"<extra></extra>")
        else:
            hover = f"<b>Satellite {s}</b><br>({xs}, {ys})<extra></extra>"
        sx.append(xs); sy.append(ys); stxt.append(f"s{s}"); shover.append(hover)

    fig.add_trace(go.Scatter(
        x=sx, y=sy, mode="markers+text",
        marker=dict(symbol="triangle-up", size=12, color="#E45756",
                    line=dict(color="white", width=1.2)),
        text=stxt, textposition="top right",
        textfont=dict(size=9, color="#8B2E2A"),
        hovertext=shover, hoverinfo="text",
        name="Satellite customer"), row=1, col=1)

    # --- 차량 경로 + hub TSP 투어 ---
    for k in vehicles:
        info = routes[k]
        route = info["route"]
        color = v_color[k]

        # 주 경로 선
        rx = [coords[n][0] for n in route]
        ry = [coords[n][1] for n in route]
        htxt = [f"Vehicle {k} · step {idx} · node {n}"
                for idx, n in enumerate(route)]
        fig.add_trace(go.Scatter(
            x=rx, y=ry, mode="lines",
            line=dict(color=color, width=2.5),
            hovertext=htxt, hoverinfo="text",
            name=f"Vehicle {k}", legendgroup=f"v{k}"), row=1, col=1)

        # 주 경로 화살표
        for a, b in zip(route[:-1], route[1:]):
            xa, ya = coords[a]; xb, yb = coords[b]
            fig.add_annotation(
                x=xb, y=yb, ax=xa, ay=ya,
                xref="x1", yref="y1", axref="x1", ayref="y1",
                showarrow=True, arrowhead=3, arrowsize=1.2,
                arrowwidth=1.6, arrowcolor=color, opacity=0.9,
                standoff=6, startstandoff=4)

        # hub 별 딸린고객 투어: hub -> s1 -> s2 -> ... -> hub
        for hub, tinfo in info["sat_tours"].items():
            seq = tinfo["sequence"]
            pts = ([coords[hub]]
                   + [sat_coords[s] for s in seq]
                   + [coords[hub]])
            fig.add_trace(go.Scatter(
                x=[p[0] for p in pts], y=[p[1] for p in pts],
                mode="lines",
                line=dict(color=color, width=1.5, dash="dash"),
                hovertext=(f"Vehicle {k} · hub C{hub} tour<br>"
                           f"length = {tinfo['length']:.2f}"),
                hoverinfo="text",
                legendgroup=f"v{k}", showlegend=False,
                opacity=0.85), row=1, col=1)
            # 투어 각 구간 화살표 (작게)
            for a, b in zip(pts[:-1], pts[1:]):
                fig.add_annotation(
                    x=b[0], y=b[1], ax=a[0], ay=a[1],
                    xref="x1", yref="y1", axref="x1", ayref="y1",
                    showarrow=True, arrowhead=2, arrowsize=1.0,
                    arrowwidth=1.0, arrowcolor=color, opacity=0.7,
                    standoff=4, startstandoff=3)

    fig.update_xaxes(title_text="x", row=1, col=1,
                     scaleanchor="y", scaleratio=1)
    fig.update_yaxes(title_text="y", row=1, col=1)

    # =============================================================
    # Row 2: 용량 활용률
    # =============================================================
    v_labels = [f"V{k}" for k in vehicles]
    w_util = [100 * load_w[k] / cap_weight for k in vehicles]
    c_util = [100 * load_c[k] / cap_charge for k in vehicles]
    w_hover = [f"Vehicle {k}<br>Load: {load_w[k]}/{cap_weight}<br>Util: {u:.1f}%"
               for k, u in zip(vehicles, w_util)]
    c_hover = [f"Vehicle {k}<br>Load: {load_c[k]}/{cap_charge}<br>Util: {u:.1f}%"
               for k, u in zip(vehicles, c_util)]
    bar_colors = [v_color[k] for k in vehicles]

    fig.add_trace(go.Bar(
        x=v_labels, y=w_util, marker_color=bar_colors,
        hovertext=w_hover, hoverinfo="text",
        text=[f"{u:.0f}%" for u in w_util], textposition="outside",
        showlegend=False), row=2, col=1)
    fig.add_hline(y=100, line_dash="dash", line_color="red",
                  opacity=0.6, row=2, col=1)

    fig.add_trace(go.Bar(
        x=v_labels, y=c_util, marker_color=bar_colors,
        hovertext=c_hover, hoverinfo="text",
        text=[f"{u:.0f}%" for u in c_util], textposition="outside",
        showlegend=False), row=2, col=2)
    fig.add_hline(y=100, line_dash="dash", line_color="red",
                  opacity=0.6, row=2, col=2)

    max_u = max(max(w_util + c_util, default=0), 100) * 1.15
    fig.update_yaxes(title_text="Utilization %", range=[0, max_u], row=2, col=1)
    fig.update_yaxes(title_text="Utilization %", range=[0, max_u], row=2, col=2)

    # =============================================================
    # Row 3: Gantt
    # =============================================================
    gantt_y = []
    for k in vehicles:
        info = routes[k]
        route = info["route"]; arrivals = info["arrivals"]
        color = v_color[k]
        y_lbl = f"Vehicle {k}"
        gantt_y.append(y_lbl)

        for idx, (node, t_arr) in enumerate(zip(route, arrivals)):
            svc = service_time[node] if node != 0 else 0
            tour_len = (info["sat_tours"][node]["length"]
                        if node in info["sat_tours"] else 0.0)
            # 서비스 블록 (hover에 hub tour 정보 표기)
            label = "Depot" if node == 0 else f"C{node}"
            hover = (f"<b>{label}</b> (Vehicle {k})<br>"
                     f"Arrival: {t_arr:.2f}<br>"
                     f"Service time: {svc}<br>"
                     f"Service end: {t_arr + svc:.2f}")
            if tour_len > 0:
                seq = info["sat_tours"][node]["sequence"]
                hover += (f"<br>Satellite tour: " +
                          " → ".join(f"s{s}" for s in seq) +
                          f"<br>Tour length: {tour_len:.2f} "
                          f"(cost only, not in time axis)")
            fig.add_trace(go.Bar(
                x=[max(svc, 0.3)], y=[y_lbl], base=[t_arr],
                orientation="h",
                marker=dict(color=color, line=dict(color="white", width=1)),
                hovertext=hover, hoverinfo="text", showlegend=False
            ), row=3, col=1)

            # 이동 블록
            if idx < len(route) - 1:
                nxt = route[idx + 1]; t_next = arrivals[idx + 1]
                depart = t_arr + svc
                travel = t_next - depart
                if travel > 1e-6:
                    h = (f"Travel {('Depot' if node==0 else f'C{node}')}"
                         f" → {('Depot' if nxt==0 else f'C{nxt}')}<br>"
                         f"Depart: {depart:.2f}<br>"
                         f"Arrive: {t_next:.2f}<br>"
                         f"Duration: {travel:.2f}")
                    fig.add_trace(go.Bar(
                        x=[travel], y=[y_lbl], base=[depart],
                        orientation="h",
                        marker=dict(color="lightgray",
                                    line=dict(color="white", width=1)),
                        hovertext=h, hoverinfo="text", showlegend=False
                    ), row=3, col=1)

    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_yaxes(title_text="", categoryorder="array",
                     categoryarray=gantt_y[::-1], row=3, col=1)

    # ---------- Layout ----------
    title = (f"CVRPTW + Satellite Tours  |  "
             f"Obj={result['objective']:.2f}  "
             f"Vehicles={result['num_vehicles_used']}  "
             f"VehDist={result['vehicle_distance']:.2f}  "
             f"SatTourDist={result['satellite_tour_distance']:.2f}")
    fig.update_layout(
        title=title, height=1050, barmode="overlay",
        hovermode="closest", template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        margin=dict(l=60, r=30, t=90, b=50),
    )

    if save_html:
        fig.write_html(save_html, include_plotlyjs="cdn")
        print(f"Saved: {save_html}")
    return fig


from cvrptw_with_satellite_tours import solve_cvrptw_with_satellite_tours

if __name__ == "__main__":
    coords = [(0, 0),
              (2, 3),  # C1  ← 3 sats 몰림
              (5, 1),  # C2
              (6, 5),  # C3  ← 2 sats 몰림
              (8, 3),  # C4
              (1, 6),  # C5
              (4, 2)]  # C6
    demand_weight = [0, 10, 12, 8, 11, 7, 9]
    demand_charge = [0, 4, 5, 3, 5, 3, 4]
    time_windows = [(0, 250),
                    (10, 100),
                    (20, 120),
                    (30, 130),
                    (40, 140),
                    (15, 110),
                    (25, 120)]
    service_time = [0, 5, 5, 5, 5, 5, 5]

    # ---- 딸린고객 재배치 ----
    # C1(2,3) 근처 3개, C3(6,5) 근처 2개
    sat_coords = [(2.0, 3.8),  # s0 : C1 북쪽 가까이
                  (1.5, 2.5),  # s1 : C1 남서쪽
                  (2.8, 3.5),  # s2 : C1 북동쪽
                  (6.0, 5.8),  # s3 : C3 북쪽
                  (6.5, 4.0)]  # s4 : C3 남동쪽
    sat_weight = [1, 1, 1, 1, 1]
    sat_charge = [1, 1, 1, 1, 1]

    # 용량: 한 차량이 C1 + 3 딸린고객까지 감당할 수 있게 여유
    cap_weight, cap_charge = 40, 18

    # ---- 거리 체크 (R=3) ----
    # 각 sat 의 hub 후보 확인 (참고용 출력)
    import math

    R = 3.0
    print("Satellite feasibility (R = 3.0):")
    for s, p in enumerate(sat_coords):
        feas = []
        for i, c in enumerate(coords):
            if i == 0: continue
            dd = math.hypot(p[0] - c[0], p[1] - c[1])
            if dd <= R:
                feas.append((f"C{i}", round(dd, 2)))
        print(f"  s{s} {p}: feas = {feas}")
    print()

    res = solve_cvrptw_with_satellite_tours(
        coords=coords,
        demand_weight=demand_weight, demand_charge=demand_charge,
        time_windows=time_windows, service_time=service_time,
        sat_coords=sat_coords,
        sat_weight=sat_weight, sat_charge=sat_charge,
        num_vehicles=3,
        cap_weight=cap_weight, cap_charge=cap_charge,
        R=R, alpha=1000.0, beta=1.0,
        time_limit=600,
    )

    if res:
        print("\n=== Result ===")
        print(f"Objective                : {res['objective']:.2f}")
        print(f"Vehicles used            : {res['num_vehicles_used']}")
        print(f"Vehicle distance         : {res['vehicle_distance']:.2f}")
        print(f"Satellite tour distance  : {res['satellite_tour_distance']:.2f}")
        for k, info in res["routes"].items():
            print(f"  Vehicle {k}: {info['route']}")
            for hub, tinfo in info["sat_tours"].items():
                seq_str = " → ".join(f"s{s}" for s in tinfo["sequence"])
                print(f"    hub C{hub} tour: C{hub} → {seq_str} → C{hub}"
                      f"  (len={tinfo['length']:.2f})")

        fig = plot_solution_plotly(
            res,
            coords=coords,
            demand_weight=demand_weight, demand_charge=demand_charge,
            time_windows=time_windows, service_time=service_time,
            sat_coords=sat_coords,
            sat_weight=sat_weight, sat_charge=sat_charge,
            cap_weight=cap_weight, cap_charge=cap_charge,
            R=R, save_html="cvrptw_sat_tours_clustered.html",
        )
        fig.show()