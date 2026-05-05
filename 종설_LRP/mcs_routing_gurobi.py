# -*- coding: utf-8 -*-
"""
Gurobi-based MCS Routing Solver for Jung-gu fast charger saturation events
========================================================================

Problem type
------------
- OPTW when K=1, TOPTW when K>=2.
- Fixed depot: 청계3 노상공영주차장(코리아몰 앞)
- Objective: maximize covered saturation-event profit, where profit = 발생비율_%.
- Time windows: each event can be serviced within [시작시각, 종료시각].
- Service time: configurable from median duration, fixed minutes, or capped minutes.
- Travel time: weekday/saturday/holiday OD matrix.

Outputs
-------
- CSV summary by day type and K
- CSV route details
- Folium HTML route maps

Install
-------
    pip install gurobipy pandas openpyxl folium numpy

Gurobi license
--------------
Gurobi must be installed and licensed. Academic users can normally use an
academic license. If Gurobi is unavailable, this script fails with an explicit
message rather than silently switching to a heuristic.

Example
-------
    python mcs_routing_gurobi.py --base-dir /mnt/data --out-dir /mnt/data/gurobi_outputs \
      --day-types 평일 토요일 일요일 공휴일 --k-min 1 --k-max 5 \
      --top-n 45 --service-mode capped --max-service-minutes 45 \
      --time-limit 1800 --mip-gap 0.0
"""

from __future__ import annotations

import argparse
import html
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import folium
except Exception as exc:  # pragma: no cover
    raise RuntimeError("folium is required. Install with: pip install folium") from exc

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception as exc:  # pragma: no cover
    gp = None
    GRB = None
    GUROBI_IMPORT_ERROR = exc
else:
    GUROBI_IMPORT_ERROR = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEPOT_FULL_NAME = "청계3 노상공영주차장(코리아몰 앞)"
DEPOT_KEY = "청계3_코리아몰앞"
OPERATING_START = 6 * 60   # 06:00
OPERATING_END = 22 * 60    # 22:00

DAY_TO_TRAVEL_FILE = {
    "평일": "weekday_filled.csv",
    "토요일": "saturday_filled.csv",
    # A separate Sunday matrix was not supplied. Use holiday matrix by default.
    "일요일": "holiday_filled.csv",
    "공휴일": "holiday_filled.csv",
}

STATION_NAME_TO_KEY = {
    "기업은행 무교지점 앞 가로등형 충전소": "기업은행_무교지점",
    "서소문청사(지상)": "서소문청사_지상",
    "서소문청사(지하주차장)": "서소문청사_지하",
    "서울시 본관청사": "서울시_본관청사",
    "을지로 노상공영주차장(신한은행 앞)": "을지로_신한은행앞",
    "을지로 노상공영주차장(하나은행 앞)": "을지로_하나은행앞",
    "청계3 노상공영주차장(고려상사 앞)": "청계3_고려상사앞",
    "청계3 노상공영주차장(코리아몰 앞)": "청계3_코리아몰앞",
    "청계5 노상공영주차장": "청계5",
    "청계8가 노상공영주차장": "청계8가",
    "훈련원공원 노상공영주차장": "훈련원공원",
}


@dataclass(frozen=True)
class Event:
    event_id: int
    station_name: str
    station_key: str
    day_type: str
    time_band: str
    a: int
    b: int
    latest_start: int
    service: int
    weight: float
    count: int
    ratio: float
    lat: float
    lon: float


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def parse_hhmm(value) -> int:
    """Parse HH:MM-like values into minutes after midnight."""
    if pd.isna(value):
        raise ValueError("empty time")
    s = str(value).strip()
    m = re.search(r"(\d{1,2}):(\d{2})", s)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    m = re.search(r"(\d{1,2})\s*시", s)
    if m:
        return int(m.group(1)) * 60
    raise ValueError(f"Cannot parse time: {value!r}")


def fmt_time(minutes: float) -> str:
    minutes = int(round(minutes))
    minutes = max(0, min(24 * 60 - 1, minutes))
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def canonical_station_key(full_name: str) -> str:
    if full_name in STATION_NAME_TO_KEY:
        return STATION_NAME_TO_KEY[full_name]
    compact = str(full_name).replace(" ", "")
    for known, key in STATION_NAME_TO_KEY.items():
        if known.replace(" ", "") == compact:
            return key
    raise KeyError(f"No station-key mapping for {full_name!r}. Add it to STATION_NAME_TO_KEY.")


def travel_time(travel_mat: pd.DataFrame, origin_key: str, dest_key: str) -> float:
    if origin_key == dest_key:
        return 0.0
    try:
        return float(travel_mat.loc[origin_key, dest_key])
    except KeyError as exc:
        raise KeyError(f"Travel matrix is missing OD pair {origin_key!r} -> {dest_key!r}") from exc


def ensure_gurobi_available() -> None:
    if gp is None:
        raise RuntimeError(
            "gurobipy could not be imported. Install it with `pip install gurobipy` "
            "and activate a valid Gurobi license. Original error: "
            f"{GUROBI_IMPORT_ERROR}"
        )


# ---------------------------------------------------------------------------
# Data loading and event construction
# ---------------------------------------------------------------------------

def load_inputs(base_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    analysis_path = base_dir / "중구_충전소_분석결과.xlsx"
    station_path = base_dir / "중구_급속충전소_정보.xlsx"
    if not analysis_path.exists():
        raise FileNotFoundError(f"Missing analysis file: {analysis_path}")
    if not station_path.exists():
        raise FileNotFoundError(f"Missing station file: {station_path}")

    analysis = pd.read_excel(analysis_path)
    stations = pd.read_excel(station_path)

    required_analysis = {"충전소명", "요일구분", "시간대", "시작시각", "종료시각", "발생횟수", "발생비율_%"}
    required_station = {"충전소명", "위도", "경도"}
    missing_analysis = required_analysis - set(analysis.columns)
    missing_station = required_station - set(stations.columns)
    if missing_analysis:
        raise ValueError(f"Analysis file missing columns: {sorted(missing_analysis)}")
    if missing_station:
        raise ValueError(f"Station file missing columns: {sorted(missing_station)}")

    analysis["station_key"] = analysis["충전소명"].map(canonical_station_key)
    stations["station_key"] = stations["충전소명"].map(canonical_station_key)

    travel: Dict[str, pd.DataFrame] = {}
    for day_type, filename in DAY_TO_TRAVEL_FILE.items():
        path = base_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing travel matrix for {day_type}: {path}")
        mat = pd.read_csv(path)
        if "origin_name" not in mat.columns:
            raise ValueError(f"Travel matrix {path} must have origin_name column")
        mat = mat.set_index("origin_name")
        travel[day_type] = mat
    return analysis, stations, travel


def build_events(
    analysis: pd.DataFrame,
    stations: pd.DataFrame,
    day_type: str,
    service_mode: str,
    fixed_service_minutes: int,
    max_service_minutes: int,
    min_ratio: float,
    top_n: Optional[int],
) -> List[Event]:
    loc = stations.set_index("station_key")[["위도", "경도"]].to_dict("index")
    df = analysis.loc[analysis["요일구분"].eq(day_type)].copy()
    df = df.loc[df["발생비율_%"].fillna(0).astype(float) >= min_ratio].copy()
    df["weight"] = df["발생비율_%"].astype(float)
    df = df.sort_values(["weight", "발생횟수"], ascending=False)
    if top_n is not None:
        df = df.head(top_n)

    events: List[Event] = []
    for idx, row in df.reset_index(drop=True).iterrows():
        key = row["station_key"]
        if key not in loc:
            continue
        a = parse_hhmm(row["시작시각"])
        b = parse_hhmm(row["종료시각"])
        if b <= a:
            duration_fallback = int(float(row.get("지속시간_분", 0) or 0))
            b = a + max(1, duration_fallback)

        # Intersect with MCS operating hours.
        a = max(a, OPERATING_START)
        b = min(b, OPERATING_END)
        if b <= a:
            continue

        raw_duration = int(round(float(row.get("지속시간_분", b - a) or (b - a))))
        raw_duration = max(1, raw_duration)
        if service_mode == "fixed":
            service = fixed_service_minutes
        elif service_mode == "capped":
            service = min(raw_duration, max_service_minutes)
        elif service_mode == "window_duration":
            service = raw_duration
        else:
            raise ValueError(f"Unknown service_mode: {service_mode}")
        service = int(max(1, min(service, b - a)))
        latest_start = b - service
        if latest_start < a:
            continue

        events.append(Event(
            event_id=idx,
            station_name=str(row["충전소명"]),
            station_key=key,
            day_type=day_type,
            time_band=str(row["시간대"]),
            a=int(a),
            b=int(b),
            latest_start=int(latest_start),
            service=int(service),
            weight=float(row["weight"]),
            count=int(row["발생횟수"]),
            ratio=float(row["발생비율_%"]),
            lat=float(loc[key]["위도"]),
            lon=float(loc[key]["경도"]),
        ))
    return events


# ---------------------------------------------------------------------------
# Gurobi TOPTW model
# ---------------------------------------------------------------------------

def solve_toptw_gurobi(
    events: List[Event],
    travel_mat: pd.DataFrame,
    K: int,
    time_limit: Optional[float],
    mip_gap: float,
    threads: int,
    verbose: bool,
    force_event_station_time_unique: bool = True,
) -> Dict:
    """Solve the TOPTW/OPTW exactly or with a certified MIP gap using Gurobi."""
    ensure_gurobi_available()
    if not events:
        return {
            "status": "EMPTY",
            "objective": 0.0,
            "bound": 0.0,
            "gap": 0.0,
            "routes": [[] for _ in range(K)],
            "arrival_times": {},
            "runtime": 0.0,
            "node_count": 0,
        }

    n = len(events)
    event_ids = list(range(n))  # internal contiguous ids
    start = n
    end = n + 1
    nodes = event_ids + [start, end]
    vehicles = list(range(K))

    station_key = {i: events[i].station_key for i in event_ids}
    station_key[start] = DEPOT_KEY
    station_key[end] = DEPOT_KEY

    # Feasible arcs. Exclude self-arcs, arcs into start, arcs out of end, and direct start->end
    # is allowed so that an unused MCS can leave and return with no event.
    arcs: List[Tuple[int, int]] = []
    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            if j == start or i == end:
                continue
            if i == start and j == end:
                arcs.append((i, j))
                continue
            if i == start and j in event_ids:
                # Can depart depot and start service within j's time window.
                if OPERATING_START + travel_time(travel_mat, DEPOT_KEY, station_key[j]) <= events[j].latest_start:
                    arcs.append((i, j))
                continue
            if i in event_ids and j == end:
                # Can return to depot by operating end after serving i.
                if events[i].a + events[i].service + travel_time(travel_mat, station_key[i], DEPOT_KEY) <= OPERATING_END:
                    arcs.append((i, j))
                continue
            if i in event_ids and j in event_ids:
                # Earliest feasible service sequence i -> j.
                if events[i].a + events[i].service + travel_time(travel_mat, station_key[i], station_key[j]) <= events[j].latest_start:
                    arcs.append((i, j))

    if not arcs:
        return {
            "status": "INFEASIBLE_NO_ARCS",
            "objective": 0.0,
            "bound": 0.0,
            "gap": math.nan,
            "routes": [[] for _ in range(K)],
            "arrival_times": {},
            "runtime": 0.0,
            "node_count": 0,
        }

    model = gp.Model("JungGu_MCS_TOPTW")
    model.Params.OutputFlag = 1 if verbose else 0
    if time_limit is not None and time_limit > 0:
        model.Params.TimeLimit = float(time_limit)
    if mip_gap is not None and mip_gap >= 0:
        model.Params.MIPGap = float(mip_gap)
    if threads is not None and threads > 0:
        model.Params.Threads = int(threads)

    x = model.addVars(arcs, vehicles, vtype=GRB.BINARY, name="x")
    y = model.addVars(event_ids, vehicles, vtype=GRB.BINARY, name="y")
    tau = model.addVars(nodes, vehicles, lb=OPERATING_START, ub=OPERATING_END, vtype=GRB.CONTINUOUS, name="tau")

    # Objective: maximize covered event profit.
    model.setObjective(
        gp.quicksum(events[i].weight * y[i, k] for i in event_ids for k in vehicles),
        GRB.MAXIMIZE,
    )

    outgoing = {(i, k): gp.quicksum(x[i, j, k] for (ii, j) in arcs if ii == i) for i in nodes for k in vehicles}
    incoming = {(i, k): gp.quicksum(x[j, i, k] for (j, ii) in arcs if ii == i) for i in nodes for k in vehicles}

    # Vehicle starts and ends exactly once. Direct start->end means unused vehicle.
    for k in vehicles:
        model.addConstr(outgoing[start, k] == 1, name=f"start_once[{k}]")
        model.addConstr(incoming[end, k] == 1, name=f"end_once[{k}]")
        model.addConstr(incoming[start, k] == 0, name=f"no_in_start[{k}]")
        model.addConstr(outgoing[end, k] == 0, name=f"no_out_end[{k}]")
        model.addConstr(tau[start, k] == OPERATING_START, name=f"start_time[{k}]")
        model.addConstr(tau[end, k] <= OPERATING_END, name=f"end_time[{k}]")

    # Flow conservation and visit definition.
    for i in event_ids:
        for k in vehicles:
            model.addConstr(incoming[i, k] == y[i, k], name=f"in_visit[{i},{k}]")
            model.addConstr(outgoing[i, k] == y[i, k], name=f"out_visit[{i},{k}]")
            model.addConstr(tau[i, k] >= events[i].a * y[i, k], name=f"tw_lb[{i},{k}]")
            # If not visited, this upper bound is relaxed to operating end.
            model.addConstr(
                tau[i, k] <= events[i].latest_start * y[i, k] + OPERATING_END * (1 - y[i, k]),
                name=f"tw_ub[{i},{k}]",
            )

        # Each event can be covered by at most one MCS.
        model.addConstr(gp.quicksum(y[i, k] for k in vehicles) <= 1, name=f"event_once[{i}]")

    # Optional: if duplicate rows represent the same station/time-band event, do not cover duplicates.
    if force_event_station_time_unique:
        groups: Dict[Tuple[str, str], List[int]] = {}
        for i, event in enumerate(events):
            groups.setdefault((event.station_key, event.time_band), []).append(i)
        for (key, band), group in groups.items():
            if len(group) > 1:
                model.addConstr(
                    gp.quicksum(y[i, k] for i in group for k in vehicles) <= 1,
                    name=f"station_time_once[{key},{band}]",
                )

    # Time propagation. This removes subtours because service start times must strictly advance.
    for i, j in arcs:
        if i == end or j == start:
            continue
        if i == start and j == end:
            service_i = 0.0
            tij = 0.0
        elif i == start:
            service_i = 0.0
            tij = travel_time(travel_mat, DEPOT_KEY, station_key[j])
        elif j == end:
            service_i = events[i].service
            tij = travel_time(travel_mat, station_key[i], DEPOT_KEY)
        else:
            service_i = events[i].service
            tij = travel_time(travel_mat, station_key[i], station_key[j])
        # Arc-specific tight M. If arc is not used, constraint is relaxed.
        m_ij = OPERATING_END + service_i + tij - OPERATING_START
        for k in vehicles:
            model.addConstr(
                tau[j, k] >= tau[i, k] + service_i + tij - m_ij * (1 - x[i, j, k]),
                name=f"time[{i},{j},{k}]",
            )

    model.optimize()

    status_code = model.Status
    status_name = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NUMERIC: "NUMERIC",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
    }.get(status_code, str(status_code))

    if model.SolCount == 0:
        return {
            "status": status_name,
            "objective": math.nan,
            "bound": getattr(model, "ObjBound", math.nan),
            "gap": math.nan,
            "routes": [[] for _ in vehicles],
            "arrival_times": {},
            "runtime": model.Runtime,
            "node_count": model.NodeCount,
        }

    routes: List[List[int]] = []
    arrival_times: Dict[Tuple[int, int], float] = {}
    for k in vehicles:
        current = start
        route: List[int] = []
        seen = set()
        while current != end:
            next_nodes = [j for (i, j) in arcs if i == current and x[i, j, k].X > 0.5]
            if not next_nodes:
                break
            nxt = next_nodes[0]
            if nxt == end:
                break
            if nxt in seen:
                raise RuntimeError(f"Cycle detected while extracting route for vehicle {k}")
            seen.add(nxt)
            route.append(nxt)
            arrival_times[(nxt, k)] = float(tau[nxt, k].X)
            current = nxt
        routes.append(route)

    gap = float(model.MIPGap) if model.SolCount > 0 and hasattr(model, "MIPGap") else math.nan
    return {
        "status": status_name,
        "objective": float(model.ObjVal),
        "bound": float(model.ObjBound),
        "gap": gap,
        "routes": routes,
        "arrival_times": arrival_times,
        "runtime": float(model.Runtime),
        "node_count": float(model.NodeCount),
    }


# ---------------------------------------------------------------------------
# Output formatting and visualization
# ---------------------------------------------------------------------------

def flatten_solution(day_type: str, K: int, events: List[Event], sol: Dict) -> pd.DataFrame:
    rows = []
    for k, route in enumerate(sol["routes"], start=1):
        prev_key = DEPOT_KEY
        for seq, event_idx in enumerate(route, start=1):
            event = events[event_idx]
            arr = sol["arrival_times"].get((event_idx, k - 1), event.a)
            start_service = max(arr, event.a)
            end_service = start_service + event.service
            rows.append({
                "요일구분": day_type,
                "K": K,
                "MCS": k,
                "방문순서": seq,
                "충전소명": event.station_name,
                "station_key": event.station_key,
                "시간대": event.time_band,
                "time_window_start": fmt_time(event.a),
                "time_window_end": fmt_time(event.b),
                "arrival_time": fmt_time(arr),
                "service_start": fmt_time(start_service),
                "service_end": fmt_time(end_service),
                "service_minutes": event.service,
                "발생횟수": event.count,
                "발생비율_%": event.ratio,
                "profit": event.weight,
                "위도": event.lat,
                "경도": event.lon,
                "from_station_key": prev_key,
            })
            prev_key = event.station_key
    return pd.DataFrame(rows)


def make_route_map(
    day_type: str,
    K: int,
    events: List[Event],
    sol: Dict,
    stations: pd.DataFrame,
    out_path: Path,
) -> None:
    station_loc = stations.set_index("station_key")[["충전소명", "위도", "경도"]].to_dict("index")
    depot_lat = float(station_loc[DEPOT_KEY]["위도"])
    depot_lon = float(station_loc[DEPOT_KEY]["경도"])
    m = folium.Map(location=[depot_lat, depot_lon], zoom_start=14, tiles="OpenStreetMap")

    folium.Marker(
        [depot_lat, depot_lon],
        popup=f"Depot: {DEPOT_FULL_NAME}",
        tooltip="Depot",
        icon=folium.Icon(color="black", icon="home"),
    ).add_to(m)

    palette = ["red", "blue", "green", "purple", "orange", "darkred", "cadetblue", "darkgreen"]
    for k, route in enumerate(sol["routes"], start=1):
        color = palette[(k - 1) % len(palette)]
        points = [[depot_lat, depot_lon]]
        for seq, event_idx in enumerate(route, start=1):
            event = events[event_idx]
            arr = sol["arrival_times"].get((event_idx, k - 1), event.a)
            popup = (
                f"<b>MCS {k} - #{seq}</b><br>"
                f"충전소: {html.escape(event.station_name)}<br>"
                f"시간창: {fmt_time(event.a)}~{fmt_time(event.b)}<br>"
                f"도착: {fmt_time(arr)}<br>"
                f"서비스: {event.service}분<br>"
                f"발생횟수: {event.count}<br>"
                f"발생비율: {event.ratio:.2f}%"
            )
            folium.Marker(
                [event.lat, event.lon],
                popup=popup,
                tooltip=f"MCS{k}-#{seq}: {event.station_name}",
                icon=folium.Icon(color=color, icon="flash", prefix="fa"),
            ).add_to(m)
            points.append([event.lat, event.lon])
        points.append([depot_lat, depot_lon])
        if len(points) >= 2:
            folium.PolyLine(points, color=color, weight=4, opacity=0.75, tooltip=f"MCS {k}").add_to(m)

    title = (
        f"<div style='position: fixed; top: 10px; left: 50px; z-index: 9999; "
        f"background: white; padding: 10px 14px; border: 1px solid #999; border-radius: 6px;'>"
        f"<b>{day_type} / K={K}</b><br>"
        f"Status: {sol['status']}<br>"
        f"Objective: {sol['objective']:.3f}<br>"
        f"Gap: {sol['gap']:.4f}<br>"
        f"Runtime: {sol['runtime']:.1f}s"
        f"</div>"
    )
    m.get_root().html.add_child(folium.Element(title))
    m.save(str(out_path))


def write_index(out_dir: Path, summary: pd.DataFrame) -> None:
    rows = []
    for _, row in summary.iterrows():
        fn = f"route_{row['요일구분']}_K{int(row['K'])}.html"
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['요일구분']))}</td>"
            f"<td>{int(row['K'])}</td>"
            f"<td>{html.escape(str(row['status']))}</td>"
            f"<td>{row['covered_events']}</td>"
            f"<td>{row['covered_profit']:.3f}</td>"
            f"<td>{row['covered_profit_ratio']:.3f}</td>"
            f"<td>{row['mip_gap']:.6f}</td>"
            f"<td>{row['runtime_sec']:.1f}</td>"
            f"<td><a href='{html.escape(fn)}'>map</a></td>"
            "</tr>"
        )
    page = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>MCS Gurobi Routing Results</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 24px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background: #f5f5f5; }}
</style>
</head>
<body>
<h1>MCS Gurobi Routing Results</h1>
<p>Objective = covered 발생비율_% sum. Gap 0 means Gurobi proved optimality.</p>
<table>
<thead><tr><th>요일</th><th>K</th><th>Status</th><th>Covered events</th><th>Covered profit</th><th>Profit ratio</th><th>MIP gap</th><th>Runtime sec</th><th>Map</th></tr></thead>
<tbody>
{''.join(rows)}
</tbody>
</table>
</body>
</html>"""
    (out_dir / "index.html").write_text(page, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiments(args: argparse.Namespace) -> None:
    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    analysis, stations, travel = load_inputs(base_dir)
    summary_rows = []
    route_frames = []

    for day_type in args.day_types:
        if day_type not in DAY_TO_TRAVEL_FILE:
            raise ValueError(f"Unknown day type: {day_type}. Choose from {list(DAY_TO_TRAVEL_FILE)}")
        events = build_events(
            analysis=analysis,
            stations=stations,
            day_type=day_type,
            service_mode=args.service_mode,
            fixed_service_minutes=args.fixed_service_minutes,
            max_service_minutes=args.max_service_minutes,
            min_ratio=args.min_ratio,
            top_n=args.top_n,
        )
        available_profit = sum(e.weight for e in events)
        print(f"[{day_type}] candidate events={len(events)}, available_profit={available_profit:.3f}")

        for K in range(args.k_min, args.k_max + 1):
            print(f"  Solving K={K} with Gurobi...")
            sol = solve_toptw_gurobi(
                events=events,
                travel_mat=travel[day_type],
                K=K,
                time_limit=args.time_limit,
                mip_gap=args.mip_gap,
                threads=args.threads,
                verbose=args.verbose,
                force_event_station_time_unique=not args.allow_duplicate_station_time,
            )
            route_df = flatten_solution(day_type, K, events, sol)
            route_frames.append(route_df)
            covered_events = int(len(route_df))
            covered_profit = float(route_df["profit"].sum()) if not route_df.empty else 0.0
            profit_ratio = covered_profit / available_profit if available_profit > 0 else 0.0
            summary_rows.append({
                "요일구분": day_type,
                "K": K,
                "candidate_events": len(events),
                "available_profit": available_profit,
                "covered_events": covered_events,
                "covered_profit": covered_profit,
                "covered_profit_ratio": profit_ratio,
                "status": sol["status"],
                "objective": sol["objective"],
                "objective_bound": sol["bound"],
                "mip_gap": sol["gap"],
                "runtime_sec": sol["runtime"],
                "node_count": sol["node_count"],
                "service_mode": args.service_mode,
                "top_n": args.top_n,
                "min_ratio": args.min_ratio,
            })
            map_path = out_dir / f"route_{day_type}_K{K}.html"
            make_route_map(day_type, K, events, sol, stations, map_path)
            print(
                f"    status={sol['status']}, obj={covered_profit:.3f}, "
                f"gap={sol['gap']:.6f}, covered={covered_events}, runtime={sol['runtime']:.1f}s"
            )

    summary = pd.DataFrame(summary_rows)
    routes = pd.concat(route_frames, ignore_index=True) if route_frames else pd.DataFrame()
    summary.to_csv(out_dir / "mcs_gurobi_summary.csv", index=False, encoding="utf-8-sig")
    routes.to_csv(out_dir / "mcs_gurobi_routes.csv", index=False, encoding="utf-8-sig")
    write_index(out_dir, summary)
    print(f"\nDone. Results written to: {out_dir}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gurobi TOPTW solver for Jung-gu MCS routing")
    parser.add_argument(
        "--base-dir",
        default=r"C:\Users\82109\PyCharmMiscProject\종설_LRP\data",
        help="Directory containing input xlsx/csv files"
    )
    parser.add_argument(
        "--out-dir",
        default=r"C:\Users\82109\PyCharmMiscProject\종설_LRP\results",
        help="Directory for output CSV/HTML files"
    )
    parser.add_argument("--day-types", nargs="+", default=["평일", "토요일", "일요일", "공휴일"], help="Day types to solve")
    parser.add_argument("--k-min", type=int, default=1)
    parser.add_argument("--k-max", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=45, help="Use top N events by 발생비율_%; set 0 for all")
    parser.add_argument("--min-ratio", type=float, default=0.0, help="Minimum 발생비율_% threshold")
    parser.add_argument("--service-mode", choices=["window_duration", "fixed", "capped"], default="capped")
    parser.add_argument("--fixed-service-minutes", type=int, default=20)
    parser.add_argument("--max-service-minutes", type=int, default=45)
    parser.add_argument("--time-limit", type=float, default=1800.0, help="Gurobi time limit in seconds; 0 means no limit")
    parser.add_argument("--mip-gap", type=float, default=0.0, help="Relative MIP gap target. 0.0 asks for proven optimum")
    parser.add_argument("--threads", type=int, default=0, help="Gurobi threads; 0 lets Gurobi decide")
    parser.add_argument("--verbose", action="store_true", help="Show Gurobi log")
    parser.add_argument("--allow-duplicate-station-time", action="store_true", help="Allow duplicate events at same station/time-band")
    args = parser.parse_args(argv)
    if args.top_n == 0:
        args.top_n = None
    if args.time_limit == 0:
        args.time_limit = None
    if args.k_min < 1 or args.k_max < args.k_min:
        raise ValueError("Require 1 <= k_min <= k_max")
    return args


if __name__ == "__main__":
    try:
        run_experiments(parse_args())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
