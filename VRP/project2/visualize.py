"""
plot_solution / export_solution_to_excel / load_solution_from_excel — 한국어 친절 각주 버전

요약
- 차량경로 해(Solution)를 2D로 시각화하고(지연/과적 오버레이),
- 각 Vehicle 경로를 탭별 엑셀 시트로 내보내며,
- 해당 엑셀로부터 원래 객체(Customer, Vehicle, Instance, Solution)를 복원합니다.

사용 전제(주1)
- from module import Customer, Vehicle, Instance, Solution, get_dist 가 가능해야 합니다.
- Solution.instance 는 customers(List[Customer]), vehicles(List[Vehicle]) 를 가집니다.
- Vehicle: ID, loc(=depot 좌표 [x,y]), speed, capacity, schedules(List[Customer])
- Customer: ID, loc([x,y]), tw([ready, due]), serv_time, weight(=demand)

안전장치(주2)
- 값이 비어있을 수 있는 필드는 최대한 기본값으로 처리합니다(예: speed→1.0, capacity→∞, serv_time→0.0).
- 색상/범례/라벨 등은 시각화 가독성을 고려해 기본값을 둡니다.

각주 표기
- 본문 주석에 (주N) 형태로 각주를 달고, 파일 끝부분에 "각주 모음"에서 상세 설명을 제공합니다.
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from module import get_dist  # 사용자 제공 거리 함수 사용(주3)
import math
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from module import Customer, Vehicle, Instance, Solution


# ---- Core plotting utility ----
def plot_solution(
    solution,
    *,
    annotate: bool = True,
    show_time_windows: bool = False,
    show_unserved: bool = True,
    arrows: bool = False,
    figsize: Tuple[int, int] = (9, 7),
    save_path: Optional[str] = None,
    write_back: bool = False,
    # --- New options for CVRP visuals ---
    show_demands: bool = True,         # 고객 수요(demand) 표기(주4)
    mark_overloads: bool = True,       # 용량 초과 구간을 빨간 선으로 강조(주5)
) -> Dict:
    """
    Visualize a routing Solution in 2D, with capacity and tardiness overlays.

    Args:
        solution: your Solution object with .instance.{customers,vehicles} and filled vehicle .schedules.
        annotate: add text labels (ID, demand, tardiness).
        show_time_windows: include [ready, due] in labels.
        show_unserved: mark any customers not present in any vehicle schedule.
        arrows: draw arrowheads along each leg (heavier for many nodes).
        figsize: matplotlib figure size.
        save_path: if provided, saves the figure to this path.
        write_back: if True, writes computed start/end/tardiness back to customer fields. (현재 코드는 주석 처리)(주6)
        show_demands: show customer demand (weight) in annotations.
        mark_overloads: overlay red segments where cumulative load exceeds vehicle capacity.

    Returns:
        A dict with summary metrics:
            {
              "vehicle_distance_km": {veh_id: dist_km, ...},
              "vehicle_tardiness_h": {veh_id: tard_h, ...},
              "vehicle_load": {veh_id: total_demand, ...},
              "vehicle_capacity": {veh_id: capacity_or_inf, ...},
              "vehicle_utilization": {veh_id: load/cap, ... or None if cap inf},
              "capacity_feasible": bool,
              "total_distance_km": float,
              "total_tardiness_h": float
            }
    """
    inst = solution.instance
    customers: List = inst.customers
    vehicles: List = inst.vehicles

    # Colors: Matplotlib 기본 사이클을 우선 사용(주7)
    palette = plt.rcParams.get("axes.prop_cycle", None)
    colors = (palette.by_key()["color"] if palette else
              ["#4e79a7", "#59a14f", "#e15759", "#f28e2b", "#76b7b2", "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ab"])

    fig, ax = plt.subplots(figsize=figsize)

    # Plot depots: 각 Vehicle의 디포를 네모(s)로 표기하고 V<ID> 라벨 추가(주8)
    for idx, v in enumerate(vehicles):
        vx, vy = (float(v.loc[0]), float(v.loc[1]))
        ax.scatter(vx, vy, marker="s", s=90, color=colors[idx % len(colors)], edgecolors="k", zorder=4, label=f"Vehicle {v.ID} depot")
        ax.annotate(f"V{v.ID}", (vx, vy), textcoords="offset points", xytext=(6, 6), fontsize=9)

    # Served set: 어떤 고객이 어느 차량에 의해 방문되었는지 추적(주9)
    served_ids = set()
    for v in vehicles:
        for c in getattr(v, "schedules", []):
            served_ids.add(c.ID)

    # Per-vehicle accumulators
    veh_dist: Dict[int, float] = {}
    veh_tard: Dict[int, float] = {}
    veh_load: Dict[int, float] = {}
    veh_cap: Dict[int, float] = {}
    veh_util: Dict[int, Optional[float]] = {}

    capacity_feasible = True
    # 고객 due time의 범위를 색상값 매핑에 사용(RdYlGn_r: 낮을수록 초록→높을수록 빨강)(주10)
    all_due = [float(c.tw[1]) for c in customers]
    norm = mcolors.Normalize(vmin=min(all_due), vmax=max(all_due))
    cmap = cm.get_cmap("RdYlGn_r")

    for idx, v in enumerate(vehicles):
        col = colors[idx % len(colors)]
        route: List = getattr(v, "schedules", [])
        cap = float(v.capacity) if getattr(v, "capacity", None) is not None else float("inf")
        veh_cap[v.ID] = cap

        if not route:
            # 빈 경로면 요약값 0으로 채움(주11)
            veh_dist[v.ID] = 0.0
            veh_tard[v.ID] = 0.0
            veh_load[v.ID] = 0.0
            veh_util[v.ID] = 0.0 if cap > 0 and math.isfinite(cap) else None
            continue

        # Distance/tardiness/load accumulation
        cur_time = 0
        cur_loc = (float(v.loc[0]), float(v.loc[1]))
        speed = float(v.speed or 1.0)

        xs, ys = [], []

        # Start at depot
        dx, dy = cur_loc
        xs.append(dx); ys.append(dy)

        dist_km = 0.0
        tard_h  = 0.0
        load_cum = 0.0
        overload_segments = []  # (x0,y0,x1,y1) 구간 저장(주5)

        for c in route:
            c_loc = (float(c.loc[0]), float(c.loc[1]))
            leg_km = get_dist(cur_loc, c_loc)
            leg_h  = leg_km / max(1e-9, speed)  # 속도 0 방지(주12)

            # 일정/지연 계산(주13)
            start = max(float(c.tw[0]), cur_time + leg_h)
            end   = start + float(c.serv_time or 0.0)
            tard  = max(0.0, end - float(c.tw[1]))
            # 수요(=weight)
            demand = float(getattr(c, "weight", 0.0) or 0.0)
            load_cum_next = load_cum + demand

            # (선택) write_back: 고객 객체에 스케줄 값을 기록할 수 있으나 현재는 주석 처리
            # if write_back:
            #     c.start = start; c.end = end; c.assigned_vhc = v.ID; c.complete = True
            # v.process(c)  # 사용자의 Vehicle.process()가 있을 수 있어 주석 처리

            dist_km += leg_km
            tard_h  += tard

            # 노드 표시(마커 색은 due 기반 색상)
            cx, cy = c_loc
            due_val = float(c.tw[1])
            color_val = cmap(norm(due_val))
            xs.append(cx); ys.append(cy)
            ax.scatter(cx, cy, s=50, color=color_val, edgecolors="k", zorder=5)

            if annotate:
                parts = [f"{c.ID}"]
                if show_demands:
                    parts.append(f"d={demand:.1f}")
                parts.append(f"T={tard:.2f}h")
                if show_time_windows:
                    parts.append(f"\n[{float(c.tw[0]):.2f},{float(c.tw[1]):.2f}]")
                ax.annotate(" | ".join(parts), (cx, cy), textcoords="offset points", xytext=(6, 4), fontsize=5)

            # 과적 표시(해당 leg 이후 적재가 cap 초과 시 빨간 굵은 선으로 강조)(주5)
            if mark_overloads and load_cum_next - cap > 1e-9 and math.isfinite(cap):
                x0, y0 = xs[-2], ys[-2]
                x1, y1 = xs[-1], ys[-1]
                overload_segments.append((x0, y0, x1, y1))
                capacity_feasible = False

            # advance
            cur_time = end
            cur_loc = c_loc
            load_cum = load_cum_next

        # 궤적(폴리라인) 그리기 및 범례 라벨 구성(주14)
        legend_cap = (f"{cap:.1f}" if math.isfinite(cap) else "∞")
        legend_label = (f"Vehicle {v.ID} route "
                        f"(D={dist_km:.1f} km, T={tard_h:.2f} h, Load={load_cum:.1f}/{legend_cap}), Speed={v.speed:.1f}")
        ax.plot(xs, ys, "-", color=col, lw=2, alpha=0.9, label=legend_label, zorder=3)

        # 과적 구간 오버레이
        if mark_overloads and overload_segments:
            for (x0, y0, x1, y1) in overload_segments:
                ax.plot([x0, x1], [y0, y1], "-", color="red", lw=3, alpha=0.9, zorder=4)

        # 방향 화살표(옵션)
        if arrows:
            for i in range(len(xs) - 1):
                ax.annotate("", xy=(xs[i + 1], ys[i + 1]), xytext=(xs[i], ys[i]),
                            arrowprops=dict(arrowstyle="->", color=col, lw=1, shrinkA=2, shrinkB=2), zorder=2)

        veh_dist[v.ID] = dist_km
        veh_tard[v.ID] = tard_h
        veh_load[v.ID] = load_cum
        veh_util[v.ID] = (load_cum / cap) if math.isfinite(cap) and cap > 0 else None
        if math.isfinite(cap) and load_cum - cap > 1e-9:
            capacity_feasible = False

        # 디포 용량 텍스트(주15)
        dep_label = f"Cap={legend_cap}"
        ax.annotate(dep_label, (dx, dy), textcoords="offset points", xytext=(6, -10), fontsize=8, color=col)
        # --- 여기 추가 --- (원 코드 유지용 주석)

    # 컬러바: due time 기준 색상 범례(주10)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical", fraction=0.03, pad=0.04)
    cbar.set_label("Due Time (tw[1])")

    # Unserved customers (if any)
    if show_unserved:
        unserved = [c for c in customers if c.ID not in served_ids]
        if unserved:
            ux, uy = [], []
            for c in unserved:
                cx, cy = (float(c.loc[0]), float(c.loc[1]))
                ux.append(cx); uy.append(cy)
            ax.scatter(ux, uy, marker="x", s=70, color="red", zorder=6, label="Unserved")

        # Aesthetics (원 코드 위치 그대로 유지: show_unserved=False면 아래 서식이 적용되지 않음에 유의)(주16)
        title_extra = " (CAP OK)" if capacity_feasible else " (CAP VIOLATION!)"
        ax.set_title(
            f"Routes by {solution.algorithm} | status={getattr(solution, 'status', 'N/A')} | "
            f"Total Tardiness={round(getattr(solution, 'objective', 'N/A'), 2)}"
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True, ls="--", alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

        # 범례: 플롯 위쪽 바깥(주17)
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.05),
            fontsize=9,
            frameon=True,
            ncol=2
        )

        # 레이아웃/저장(주18)
        plt.tight_layout(rect=[0, 0, 1, 0.9])
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

    summary = {
        "vehicle_distance_km": veh_dist,
        "vehicle_tardiness_h": veh_tard,
        "vehicle_load": veh_load,
        "vehicle_capacity": veh_cap,
        "vehicle_utilization": veh_util,
        "capacity_feasible": capacity_feasible,
        "total_distance_km": sum(veh_dist.values()),
        "total_tardiness_h": sum(veh_tard.values()),
    }
    return summary


def export_solution_to_excel(
    solution,
    filepath: str,
    *,
    include_customers: bool = True,
    return_to_depot: bool = False
) -> None:
    """
    각 Vehicle의 경로를 개별 시트(V_<veh_id>)로, 전체 문제 요약을 Problem 시트로 저장.

    Parameters
    ----------
    solution : Solution
        .instance (customers, vehicles), .algorithm, .status, .objective 등을 가진 객체
        각 Vehicle은 .ID, .loc(=depot [x,y]), .speed, .capacity, .schedules(List[Customer]) 보유
    filepath : str
        저장할 엑셀 경로 (*.xlsx)
    include_customers : bool, default True
        Problem 시트에 고객 목록 테이블도 함께 기록
    return_to_depot : bool, default False
        True면 마지막 고객에서 depot으로 복귀 leg를 추가로 기록
    """
    inst = solution.instance
    customers: List = inst.customers
    vehicles: List = inst.vehicles

    # ---------- Problem 시트용 테이블 ----------
    # Vehicles 요약(주19)
    vehicles_rows = []
    for v in vehicles:
        vehicles_rows.append({
            "vehicle_id": v.ID,
            "depot_x": float(v.loc[0]),
            "depot_y": float(v.loc[1]),
            "speed": float(getattr(v, "speed", 1.0) or 1.0),
            "capacity": (float(v.capacity) if getattr(v, "capacity", None) is not None else float("inf"))
        })
    df_vehicles = pd.DataFrame(vehicles_rows)

    # Instance/solution 메타(주20)
    meta_rows = [{
        "algorithm": getattr(solution, "algorithm", None),
        "status": getattr(solution, "status", None),
        "objective_total_tardiness": getattr(solution, "objective", None),
        "total_distance_km": None,   # 아래에서 계산 후 채움
        "total_tardiness_h": None,   # 아래에서 계산 후 채움
        "capacity_feasible": None    # 아래에서 계산 후 채움
    }]
    df_meta = pd.DataFrame(meta_rows)

    # Customers 요약 (옵션)
    df_customers = None
    if include_customers:
        cust_rows = []
        for c in customers:
            cust_rows.append({
                "customer_id": c.ID,
                "x": float(c.loc[0]),
                "y": float(c.loc[1]),
                "demand": float(getattr(c, "weight", 0.0) or 0.0),
                "ready": float(c.tw[0]),
                "due": float(c.tw[1]),
                "service_time": float(getattr(c, "serv_time", 0.0) or 0.0),
            })
        df_customers = pd.DataFrame(cust_rows)

    # ---------- Vehicle별 경로 시트 ----------
    total_distance_km = 0.0
    total_tardiness_h = 0.0
    capacity_feasible = True

    veh_dataframes: Dict[str, pd.DataFrame] = {}

    for v in vehicles:
        route: List = getattr(v, "schedules", []) or []
        depot = (float(v.loc[0]), float(v.loc[1]))
        speed = float(getattr(v, "speed", 1.0) or 1.0)
        cap = float(v.capacity) if getattr(v, "capacity", None) is not None else float("inf")

        rows = []
        cur_time = 0.0
        cur_loc = depot
        cum_load = 0.0
        veh_dist = 0.0
        veh_tard = 0.0

        # depot → 각 고객(주21)
        seq = 0
        for c in route:
            c_loc = (float(c.loc[0]), float(c.loc[1]))
            leg_km = float(get_dist(cur_loc, c_loc))
            travel_h = leg_km / max(1e-9, speed)

            start = max(float(c.tw[0]), cur_time + travel_h)
            end = start + float(getattr(c, "serv_time", 0.0) or 0.0)
            tard = max(0.0, end - float(c.tw[1]))
            demand = float(getattr(c, "weight", 0.0) or 0.0)
            cum_load_next = cum_load + demand
            overload = (cum_load_next - cap > 1e-9) and math.isfinite(cap)

            rows.append({
                "seq": seq,
                "from_x": cur_loc[0], "from_y": cur_loc[1],
                "to_id": c.ID, "to_x": c_loc[0], "to_y": c_loc[1],
                "leg_distance_km": round(leg_km, 6),
                "travel_h": round(travel_h, 6),
                "arrive_time": round(cur_time + travel_h, 6),
                "start_time": round(start, 6),
                "end_time": round(end, 6),
                "tardiness_h": round(tard, 6),
                "demand": demand,
                "cum_load_after": cum_load_next,
                "overload_after": overload
            })

            veh_dist += leg_km
            veh_tard += tard
            if overload:
                capacity_feasible = False

            # advance
            cur_loc = c_loc
            cur_time = end
            cum_load = cum_load_next
            seq += 1

        # 고객이 있고 복귀 옵션이 True면: 마지막 고객 → depot(주22)
        if return_to_depot and len(route) > 0:
            back_km = float(get_dist(cur_loc, depot))
            back_h = back_km / max(1e-9, speed)
            rows.append({
                "seq": seq,
                "from_x": cur_loc[0], "from_y": cur_loc[1],
                "to_id": f"DEPOT(V{v.ID})", "to_x": depot[0], "to_y": depot[1],
                "leg_distance_km": round(back_km, 6),
                "travel_h": round(back_h, 6),
                "arrive_time": round(cur_time + back_h, 6),
                "start_time": None,
                "end_time": None,
                "tardiness_h": 0.0,
                "demand": 0.0,
                "cum_load_after": cum_load,
                "overload_after": False
            })
            veh_dist += back_km

        # 요약 행(시트 하단 확인용)(주23)
        rows.append({
            "seq": "SUMMARY",
            "from_x": None, "from_y": None,
            "to_id": None, "to_x": None, "to_y": None,
            "leg_distance_km": round(veh_dist, 6),
            "travel_h": None,
            "arrive_time": None,
            "start_time": None,
            "end_time": None,
            "tardiness_h": round(veh_tard, 6),
            "demand": None,
            "cum_load_after": cum_load,
            "overload_after": (cum_load - cap > 1e-9) and math.isfinite(cap)
        })
        if (cum_load - cap > 1e-9) and math.isfinite(cap):
            capacity_feasible = False

        df_route = pd.DataFrame(rows)
        sheet_name = f"V_{v.ID}"
        veh_dataframes[sheet_name] = df_route

        total_distance_km += veh_dist
        total_tardiness_h += veh_tard

    # 메타 총계 채우기(주20)
    df_meta.loc[0, "total_distance_km"] = round(total_distance_km, 6)
    df_meta.loc[0, "total_tardiness_h"] = round(total_tardiness_h, 6)
    df_meta.loc[0, "capacity_feasible"] = bool(capacity_feasible)

    # ---------- 엑셀로 기록 ----------
    with pd.ExcelWriter(filepath, engine="xlsxwriter") as writer:
        # Problem 시트: 메타, Vehicles, (옵션) Customers를 세 섹션으로(주24)
        start_row = 0
        df_meta.to_excel(writer, sheet_name="Problem", index=False, startrow=start_row)
        start_row += len(df_meta) + 2

        # Vehicles 테이블
        df_vehicles.to_excel(writer, sheet_name="Problem", index=False, startrow=start_row)
        start_row += len(df_vehicles) + 2

        # Customers 테이블(옵션)
        if df_customers is not None:
            df_customers.to_excel(writer, sheet_name="Problem", index=False, startrow=start_row)

        # Vehicle별 시트
        for name, df in veh_dataframes.items():
            df.to_excel(writer, sheet_name=name, index=False)

        # 약간의 서식(첫 행 bold)(주25)
        workbook  = writer.book
        bold_fmt = workbook.add_format({"bold": True})
        # Problem 시트 헤더 bold
        ws_problem = writer.sheets["Problem"]
        # df_meta 헤더
        for col_idx in range(df_meta.shape[1]):
            ws_problem.write(0, col_idx, df_meta.columns[col_idx], bold_fmt)
        # df_vehicles 헤더
        veh_header_row = len(df_meta) + 2
        for col_idx in range(df_vehicles.shape[1]):
            ws_problem.write(veh_header_row, col_idx, df_vehicles.columns[col_idx], bold_fmt)
        # df_customers 헤더(있다면)
        if df_customers is not None:
            cust_header_row = veh_header_row + len(df_vehicles) + 2
            for col_idx in range(df_customers.shape[1]):
                ws_problem.write(cust_header_row, col_idx, df_customers.columns[col_idx], bold_fmt)

        # 각 Vehicle 시트 헤더 bold
        for name, df in veh_dataframes.items():
            ws = writer.sheets[name]
            for col_idx in range(df.shape[1]):
                ws.write(0, col_idx, df.columns[col_idx], bold_fmt)


# ---------------------------------------------------------------------
# Excel -> 객체 복원 (Solution / Instance / Vehicle / Customer)
# ---------------------------------------------------------------------

def load_solution_from_excel(
    filepath: str
) -> Any:
    """
    Export 함수로 저장한 엑셀 파일을 로드하여
    (Instance: vehicles/customers, Vehicle.schedules) 를 복원하고 Solution을 만들어 반환.

    Parameters
    ----------
    filepath : str
        export_solution_to_excel()로 저장된 .xlsx 경로

    Returns
    -------
    solution : Solution
        instance, algorithm/status/objective(있으면) 가 채워진 솔루션 객체
    """
    # 모든 시트를 먼저 읽어둡니다 (Problem + V_<veh_id> ...)
    xls = pd.read_excel(filepath, sheet_name=None, header=None)

    if "Problem" not in xls:
        raise ValueError("Problem 시트를 찾을 수 없습니다.")

    problem_raw = xls["Problem"]

    def _find_header_row(df: pd.DataFrame, header_token: str) -> Optional[int]:
        """헤더로 쓸 특정 토큰이 들어간 행 번호를 찾습니다(열 어디든). (주26)"""
        matches = df.apply(lambda row: row.astype(str).str.strip().eq(header_token).any(), axis=1)
        idx = matches[matches].index.to_list()
        return idx[0] if idx else None

    def _read_table_from_header(df: pd.DataFrame, header_row: int) -> pd.DataFrame:
        """header_row를 헤더로 삼아 그 아래의 연속된 구간을 표로 읽기. (주27)"""
        headers = df.iloc[header_row].tolist()
        n_cols = max([i for i, v in enumerate(headers) if pd.notna(v)] + [0]) + 1
        headers = headers[:n_cols]
        data = df.iloc[header_row+1:, :n_cols].copy()
        # 완전히 빈 행이 나오면 그 위까지 자름(중간 빈줄 2개 가정)(주27)
        empty_rows = data.apply(lambda r: r.isna().all(), axis=1)
        cut = empty_rows.idxmax() if empty_rows.any() else None
        if cut is not None and empty_rows.loc[cut]:
            data = data.loc[:cut-1]
        data.columns = headers
        data = data.dropna(axis=1, how="all")
        return data.reset_index(drop=True)

    # --- 메타(맨 윗줄이 header) 읽기
    meta_header_row = 0  # export 시 항상 0
    meta_df = _read_table_from_header(problem_raw, meta_header_row)
    meta_keys = ["algorithm", "status", "objective_total_tardiness",
                 "total_distance_km", "total_tardiness_h", "capacity_feasible"]
    meta = {k: (meta_df.at[0, k] if k in meta_df.columns and len(meta_df) else None) for k in meta_keys}

    # --- Vehicles 표 찾기
    veh_header_row = _find_header_row(problem_raw, "vehicle_id")
    if veh_header_row is None:
        raise ValueError("Problem 시트에서 vehicle_id 헤더를 찾지 못했습니다.")
    vehicles_df = _read_table_from_header(problem_raw, veh_header_row)

    # --- Customers 표(있을 수도/없을 수도 있음)
    cust_header_row = _find_header_row(problem_raw, "customer_id")
    customers_df = _read_table_from_header(problem_raw, cust_header_row) if cust_header_row is not None else None

    # ---------------- 객체 생성: Vehicles ----------------
    vehicles: List = []
    veh_by_id: Dict[int, Any] = {}

    for _, r in vehicles_df.iterrows():
        vid = int(r["vehicle_id"])
        v = Vehicle(vid)
        # 필드 주입(가능한 경우)
        if "depot_x" in r and "depot_y" in r:
            v.loc = [float(r["depot_x"]), float(r["depot_y"])]
        else:
            v.loc = [0.0, 0.0]
        v.speed = float(r["speed"]) if "speed" in r and pd.notna(r["speed"]) else 1.0
        cap_val = r.get("capacity", None)
        v.capacity = float(cap_val) if (cap_val is not None and pd.notna(cap_val) and str(cap_val) != "inf") else float("inf")
        v.schedules = []  # 빈 스케줄로 초기화
        vehicles.append(v)
        veh_by_id[vid] = v

    # ---------------- 객체 생성: Customers ----------------
    customers: List = []
    cust_by_id: Dict[int, Any] = {}

    if customers_df is not None and len(customers_df):
        # 정식 고객표가 있는 경우
        for _, r in customers_df.iterrows():
            cid = int(r["customer_id"])
            c = Customer(cid)
            c.loc = [float(r["x"]), float(r["y"])]
            ready = float(r["ready"]) if pd.notna(r["ready"]) else 0.0
            due = float(r["due"]) if pd.notna(r["due"]) else float("inf")
            c.tw = [ready, due]
            c.serv_time = float(r["service_time"]) if "service_time" in r and pd.notna(r["service_time"]) else 0.0
            c.weight = float(r["demand"]) if "demand" in r and pd.notna(r["demand"]) else 0.0
            # 러닝타임 필드 기본값(주28)
            c.complete = False; c.start = -1; c.end = -1; c.tardy = -1; c.assigned_vhc = -1
            c.priority = getattr(c, "priority", 0)
            customers.append(c)
            cust_by_id[cid] = c
    else:
        # 고객표가 없으면: 각 V_* 시트에서 고객 id와 좌표를 수집해 재구성(주29)
        seen: Dict[int, Dict] = {}
        for name, df in xls.items():
            if not name.startswith("V_"):
                continue
            # 헤더 행 찾기('seq')
            header_row = None
            for i, row in df.iterrows():
                if row.astype(str).str.strip().eq("seq").any():
                    header_row = i; break
            if header_row is None:
                continue
            # 표로 변환
            temp = df.iloc[header_row:]
            temp.columns = temp.iloc[0]
            temp = temp.iloc[1:].reset_index(drop=True)
            # SUMMARY 및 depot 복귀행 제외
            if "seq" not in temp.columns:
                continue
            temp = temp[temp["seq"].astype(str) != "SUMMARY"]
            if "to_id" in temp.columns:
                mask = ~temp["to_id"].astype(str).str.startswith("DEPOT(")
                temp = temp[mask]
            # 고객 재구성
            for _, r in temp.iterrows():
                if "to_id" not in r or pd.isna(r["to_id"]):
                    continue
                try:
                    cid = int(r["to_id"])
                except:
                    continue
                if cid not in seen:
                    seen[cid] = {
                        "x": float(r.get("to_x", 0.0) or 0.0),
                        "y": float(r.get("to_y", 0.0) or 0.0),
                        "demand": float(r.get("demand", 0.0) or 0.0),
                    }
        # seen으로 Customer 생성
        for cid, info in seen.items():
            c = Customer(cid)
            c.loc = [info["x"], info["y"]]
            c.tw = [0.0, float("inf")]
            c.serv_time = 0.0
            c.weight = info["demand"]
            c.complete = False; c.start = -1; c.end = -1; c.tardy = -1; c.assigned_vhc = -1
            c.priority = getattr(c, "priority", 0)
            customers.append(c)
            cust_by_id[cid] = c

    # ---------------- Vehicle 스케줄 복원 ----------------
    for name, df in xls.items():
        if not name.startswith("V_"):
            continue
        # 헤더 행 찾기 (seq)
        header_row = None
        for i, row in df.iterrows():
            if row.astype(str).str.strip().eq("seq").any():
                header_row = i; break
        if header_row is None:
            continue
        temp = df.iloc[header_row:]
        temp.columns = temp.iloc[0]
        temp = temp.iloc[1:].reset_index(drop=True)
        if "seq" not in temp.columns:
            continue

        # 해당 vehicle id 추출 (시트 이름 V_<id>)
        try:
            vid = int(name.split("_", 1)[1])
        except:
            continue
        vehicle = veh_by_id.get(vid, None)
        if vehicle is None:
            # 시트는 있는데 Problem의 Vehicles에 없으면 스킵
            continue

        # SUMMARY, DEPOT 복귀행 제거 후 seq 기준 정렬
        df_route = temp[temp["seq"].astype(str) != "SUMMARY"].copy()
        df_route = df_route[pd.to_numeric(df_route["seq"], errors="coerce").notna()].copy()
        df_route["seq"] = df_route["seq"].astype(float)
        df_route = df_route.sort_values("seq")

        schedules = []
        for _, r in df_route.iterrows():
            to_id = r.get("to_id", None)
            if pd.isna(to_id):
                continue
            if isinstance(to_id, str) and to_id.startswith("DEPOT("):
                continue
            try:
                cid = int(to_id)
            except:
                continue
            c = cust_by_id.get(cid)
            if c is None:
                # 경로에는 있으나 고객표에 없을 수 있음 → 즉석 생성(주30)
                c = Customer(cid)
                c.loc = [float(r.get("to_x", 0.0) or 0.0), float(r.get("to_y", 0.0) or 0.0)]
                c.tw = [0.0, float("inf")]
                c.serv_time = 0.0
                c.weight = float(r.get("demand", 0.0) or 0.0)
                c.complete = False; c.start = -1; c.end = -1; c.tardy = -1; c.assigned_vhc = -1
                c.priority = getattr(c, "priority", 0)
                customers.append(c)
                cust_by_id[cid] = c
            schedules.append(c)

        vehicle.schedules = schedules  # 복원 완료

    # ---------------- Instance / Solution 조립 ----------------
    try:
        instance = Instance(customers, vehicles)
        instance.customers = customers
        instance.vehicles = vehicles
    except TypeError:
        # 생성자 인자 시나리오 대응(주31)
        instance = Instance()
        instance.customers = customers
        instance.vehicles = vehicles

    try:
        solution = Solution()
    except TypeError:
        solution = Solution  # 팩토리/클래스 특이 케이스 대비

    setattr(solution, "instance", instance)
    # Problem 메타 복원 (있을 때만)
    if meta.get("algorithm") is not None: setattr(solution, "algorithm", meta["algorithm"])
    if meta.get("status") is not None: setattr(solution, "status", meta["status"])
    if meta.get("objective_total_tardiness") is not None: setattr(solution, "objective", meta["objective_total_tardiness"])

    return solution


# ============================== 각주 모음 ==============================
# (주1) 데이터 구조 전제: module 내 클래스/함수 서명이 유사해야 합니다. 커스텀 속성명이 다르면 본 코드를 참조하여 맞춰주세요.
# (주2) 결측/이상치 안전장치: speed=0 방지, capacity=None→∞ 처리 등.
# (주3) get_dist: 사용자 정의 거리 함수. 좌표 단위(예: km) 일관성 중요.
# (주4) show_demands: annotate 활성 시 고객 라벨에 d=<weight> 표기.
# (주5) mark_overloads: 누적 적재가 cap 초과하는 leg를 굵은 빨간 선으로 강조.
# (주6) write_back: 필요 시 Customer에 start/end/tardy/assigned_vhc 등을 기록 가능. 현재는 순수 시각화를 위해 비활성.
# (주7) 색상 주기: 차량 수가 많으면 색상이 재사용될 수 있음.
# (주8) 디포 표기: 각 차량의 출발 위치를 네모 마커로 표시.
# (주9) served_ids: 미방문 고객 표시를 위해 방문 집합을 기록.
# (주10) due time 색상 매핑: 고객 마커 색을 마감시간에 따라 연속 스케일로 시각화.
# (주11) 빈 경로 차량: 요약값을 0으로 설정하고 플롯에는 디포만 표시.
# (주12) 속도 0 보호: 분모 0 회피를 위해 max(1e-9, speed).
# (주13) 일정/지연 계산: start=max(ready, 도착), end=start+service, tard=max(0,end-due).
# (주14) 범례 라벨: 거리/지연/누적적재/용량/속도 요약을 포함.
# (주15) dep_label: 디포 근처에 Cap=... 텍스트 추가.
# (주16) 미방문 옵션: show_unserved=False면 아래 타이틀/축/범례 설정이 적용되지 않습니다(원 코드 보존).
# (주17) 범례 위치: 그래프 상단 바깥에 2열로 배치.
# (주18) 저장: save_path 지정 시 DPI=150, bbox_inches="tight"로 저장.
# (주19) Vehicles 표: vehicle_id, depot 좌표, speed, capacity 요약.
# (주20) Meta 표: 알고리즘/상태/목표치와 총거리/총지연/용량여부를 기록.
# (주21) 경로 행: 각 leg에 대한 거리/이동시간/도착/시작/종료/지연/수요/누적적재/과적 플래그 기록.
# (주22) 복귀 행: 옵션으로 마지막 고객→디포 leg 기록(지연은 0 처리).
# (주23) SUMMARY 행: 차량별 총거리/총지연/최종적재/과적여부를 마지막 행으로 기록.
# (주24) Problem 시트 구성: Meta→(빈줄2)→Vehicles→(빈줄2)→(옵션)Customers 순서.
# (주25) 서식: 각 섹션 첫 행 헤더를 볼드 처리.
# (주26) _find_header_row: 임의 위치 헤더 토큰 탐색(가로줄 어디든 매칭).
# (주27) _read_table_from_header: 헤더 행 이후 연속된 데이터 블록을 표로 변환.
# (주28) Customer 런타임 기본값: complete/start/end/tardy/assigned_vhc 등 초기화.
# (주29) 고객표 부재 시 복원: 각 V_* 시트의 to_id/to_x/to_y/demand로 고객 생성.
# (주30) 경로-전용 고객 생성: 고객표엔 없고 경로에서만 관찰되는 경우 즉석 생성.
# (주31) Instance/Solution 생성자 변동성 대응: 인자 없는/있는 생성자 모두 대응 시도.
