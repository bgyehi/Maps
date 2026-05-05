# gurobi_solver.py
from __future__ import annotations
from typing import List, Tuple, Dict, Optional
import math
import gurobipy as gp
from gurobipy import GRB

# 사용 중인 프로젝트 모듈(사용자 환경에 이미 존재한다고 가정)
import module
from module import Customer, Vehicle, Instance, Solution

# -------------------------
# 보조 함수
# -------------------------
def euclid(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])

# -------------------------
# Gurobi MIP: VRPTW (min total tardiness)
# -------------------------
def solve_vrp_gurobi(
    instance: Instance,
    *,
    time_limit: float = 60.0,
    mip_gap: float = 1e-3,
    use_warmstart: bool = True,
    verbose: bool = True
):
    """
    Gurobi로 푸는 VRPTW + capacity 모델 (objective = 총 tardiness)
    - instance: 기존 Instance 객체 (module.make_random_instance로 생성된 구조와 호환되도록 시도)
    - time_limit: Gurobi time limit (seconds)
    - mip_gap: MIP relative gap
    - use_warmstart: True이면 dispatch heuristic으로 초기해를 주입
    """
    # ---- instance -> 모델 입력 변환 (유연하게 읽음) ----
    customers: List[Customer] = list(getattr(instance, "customers", []))
    vehicles:  List[Vehicle]  = list(getattr(instance, "vehicles", []))

    # depot 좌표 획득: instance.depot 혹은 차량의 시작위치 중 첫번째 사용
    depot = getattr(instance, "depot", None)
    if (not depot) or (not isinstance(depot, (tuple, list))):
        # 차량들에 start_loc 혹은 now_loc이 있으면 사용
        if vehicles and getattr(vehicles[0], "now_loc", None):
            depot = tuple(vehicles[0].now_loc)
        else:
            depot = (0.0, 0.0)

    # 노드 인덱싱: 0 = depot, 1..N = customers
    N = len(customers)
    nodes = [("depot", depot)] + [(c.ID, tuple(getattr(c, "loc", (0.0,0.0)))) for c in customers]

    # demand, tw, service
    demand = {i+1: float(getattr(customers[i], "weight", 0.0)) for i in range(N)}
    ready =  {i+1: float(getattr(customers[i], "tw", [0.0, 1e6])[0]) for i in range(N)}
    due =    {i+1: float(getattr(customers[i], "tw", [0.0, 1e6])[1]) for i in range(N)}
    service = {i+1: float(getattr(customers[i], "serv_time", 0.0)) for i in range(N)}
    # depot service 0, ready -inf, due +inf
    ready[0] = 0.0
    due[0] = 1e6
    service[0] = 0.0

    # 차량 파라미터
    K = len(vehicles)
    veh_cap = {k: float(getattr(vehicles[k], "capacity", 1e9)) for k in range(K)}
    # 속도에 따른 travel time (차량별로 다를 수 있음)
    speeds = {k: float(getattr(vehicles[k], "speed", 1.0)) for k in range(K)}

    # 거리/이동시간 사전: dist[i][j], travel_t[k][i][j]
    coords = {idx: coord for idx, coord in enumerate([depot] + [tuple(getattr(c, "loc", (0.0,0.0))) for c in customers])}
    dist = {}
    for i in coords:
        dist[i] = {}
        for j in coords:
            dist[i][j] = euclid(coords[i], coords[j])
    travel_t = {k: {i: {j: dist[i][j] / max(1e-6, speeds[k]) for j in coords} for i in coords} for k in range(K)}

    # 큰 M 값 (시간용)
    Tmax = max(due.values()) + 1000.0
    M = Tmax + 1000.0

    # -------------------------
    # 모델 생성
    # -------------------------
    model = gp.Model("VRPTW_min_tardiness")
    model.setParam("TimeLimit", time_limit)
    model.setParam("MIPGap", mip_gap)
    model.setParam("OutputFlag", 1 if verbose else 0)

    # 변수
    # x[k,i,j]  binary: vehicle k goes from i to j
    x = {}
    for k in range(K):
        for i in coords:
            for j in coords:
                if i == j:
                    continue
                x[k,i,j] = model.addVar(vtype=GRB.BINARY, name=f"x_{k}_{i}_{j}")

    # y[k,i] = 1 if vehicle k visits node i (i>0 -> customer)
    y = {}
    for k in range(K):
        for i in coords:
            if i == 0:
                continue
            y[k,i] = model.addVar(vtype=GRB.BINARY, name=f"y_{k}_{i}")

    # t[k,i]: arrival time of vehicle k at node i
    t = {}
    for k in range(K):
        for i in coords:
            t[k,i] = model.addVar(lb=0.0, ub=Tmax, vtype=GRB.CONTINUOUS, name=f"t_{k}_{i}")

    # load[k,i]: cumulative load after visiting i on vehicle k
    load = {}
    for k in range(K):
        for i in coords:
            load[k,i] = model.addVar(lb=0.0, ub=veh_cap[k], vtype=GRB.CONTINUOUS, name=f"load_{k}_{i}")

    # tardiness per customer (single var per customer id>0)
    T = {}
    for i in coords:
        if i == 0:
            continue
        T[i] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"T_{i}")

    model.update()

    # -------------------------
    # 제약식
    # -------------------------
    # 1) 각 고객은 정확히 한 차량이 방문되어야 함
    for i in coords:
        if i == 0:
            continue
        expr = gp.quicksum(x[k,j,i] for k in range(K) for j in coords if j != i)
        model.addConstr(expr == 1, name=f"visit_once_{i}")

    # 2) flow conservation for each k and non-depot node: in == out == y[k,i]
    for k in range(K):
        # start from depot: out_of_depot == in_to_depot (allow 0 if vehicle unused)
        model.addConstr(
            gp.quicksum(x[k,0,j] for j in coords if j != 0) == gp.quicksum(x[k,i,0] for i in coords if i != 0),
            name=f"depot_balance_{k}"
        )
        for i in coords:
            if i == 0:
                continue
            in_sum = gp.quicksum(x[k,j,i] for j in coords if j != i)
            out_sum = gp.quicksum(x[k,i,j] for j in coords if j != i)
            model.addConstr(in_sum == y[k,i], name=f"in_is_y_{k}_{i}")
            model.addConstr(out_sum == y[k,i], name=f"out_is_y_{k}_{i}")

    # 3) time propagation: if x[k,i,j]==1 => t[k,j] >= t[k,i] + service_i + travel_t[k][i][j]
    for k in range(K):
        for i in coords:
            for j in coords:
                if i == j:
                    continue
                model.addConstr(
                    t[k,j] >= t[k,i] + service.get(i, 0.0) + travel_t[k][i][j] - M * (1 - x[k,i,j]),
                    name=f"time_link_{k}_{i}_{j}"
                )

    # 4) time window lower bound: if visited then t >= ready
    for k in range(K):
        for i in coords:
            lb = ready.get(i, 0.0)
            model.addConstr(t[k,i] >= lb - M * (1 - (y.get((k,i), 1) if (k,i) in y else 1)),
                            name=f"ready_lb_{k}_{i}")

    # 5) load propagation: if x[k,i,j]==1 => load[k,j] >= load[k,i] + demand_j
    for k in range(K):
        for i in coords:
            for j in coords:
                if i == j:
                    continue
                dj = demand.get(j, 0.0)
                model.addConstr(
                    load[k,j] >= load[k,i] + dj - M * (1 - x[k,i,j]),
                    name=f"load_link_{k}_{i}_{j}"
                )

    # 6) capacity bounds
    for k in range(K):
        for i in coords:
            model.addConstr(load[k,i] <= veh_cap[k], name=f"cap_ub_{k}_{i}")
            model.addConstr(load[k,i] >= 0.0, name=f"cap_lb_{k}_{i}")

    # 7) t and load at depot initialize to 0 for each vehicle when leaving/returning
    for k in range(K):
        model.addConstr(t[k,0] >= 0.0, name=f"depot_time_lb_{k}")
        model.addConstr(load[k,0] == 0.0, name=f"depot_load_zero_{k}")

    # 8) tardiness linearization:
    # For each customer i and vehicle k: T[i] >= t[k,i] - due[i] - M*(1 - y[k,i])
    for i in coords:
        if i == 0:
            continue
        for k in range(K):
            model.addConstr(
                T[i] >= t[k,i] - due[i] - M * (1 - y[k,i]),
                name=f"tard_lin_{i}_{k}"
            )
    for i in coords:
        if i == 0:
            continue
        model.addConstr(T[i] >= 0.0, name=f"t_nonneg_{i}")

    # Optional symmetry breaking: limit vehicles leaving depot in lexicographic order (heuristic)
    # (not necessary; commented out)
    # for k in range(1, K):
    #     model.addConstr(gp.quicksum(x[k,0,j] for j in coords if j!=0) <= gp.quicksum(x[k-1,0,j] for j in coords if j!=0))

    # -------------------------
    # objective: minimize sum_i T[i]
    # -------------------------
    model.setObjective(gp.quicksum(T[i] for i in coords if i != 0), GRB.MINIMIZE)

    model.update()

    # -------------------------
    # warm start (heuristic) - optional
    # -------------------------
    if use_warmstart:
        try:
            # try to use existing dispatch heuristic to get initial routes
            from dispatching import dispatch_earliest_vehicle_best_customer
            heur_sol = dispatch_earliest_vehicle_best_customer(instance, scoring="delta_tardiness", tie_breaker="edd_then_nearest")
            # build initial x,y assignments from heur_sol and instance.vehicles schedules
            # assume each vehicle has .schedules list of customer objects in visit order
            for k, v in enumerate(instance.vehicles):
                route = getattr(v, "schedules", []) or []
                prev = 0
                # set depot->first, intermediate arcs, last->depot
                if len(route) == 0:
                    continue
                for c in route:
                    cid = None
                    # find index of customer in our coords indexing (coords keys are 1..N for customers in same order)
                    # we find by matching ID
                    for idx in range(1, N+1):
                        if coords[idx] == tuple(getattr(c, "loc", (0.0,0.0))) or getattr(customers[idx-1], "ID", None) == getattr(c, "ID", None):
                            cid = idx
                            break
                    if cid is None:
                        continue
                    # set x[k,prev,cid] = 1
                    if (k, prev, cid) in x:
                        x[k, prev, cid].Start = 1.0
                    prev = cid
                # close to depot
                if (k, prev, 0) in x:
                    x[k, prev, 0].Start = 1.0
            # let Gurobi build warm start
        except Exception as e:
            if verbose:
                print("Warmstart skipped (heuristic failed):", e)

    # -------------------------
    # optimize
    # -------------------------
    model.optimize()

    # -------------------------
    # 결과 파싱
    # -------------------------
    if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT or model.Status == GRB.SUBOPTIMAL:
        # reconstruct routes
        routes = {k: [] for k in range(K)}
        for k in range(K):
            # build path by walking from depot
            cur = 0
            visited = set()
            loop_guard = 0
            while True:
                found = False
                for j in coords:
                    if j == cur:
                        continue
                    if (k, cur, j) in x:
                        xv = x[k, cur, j].X
                        if xv > 0.5:
                            if j == 0:
                                found = True
                                cur = 0
                                break
                            routes[k].append(j)
                            cur = j
                            found = True
                            break
                if not found:
                    break
                loop_guard += 1
                if loop_guard > (N + 5):
                    break

        # create Solution-like object
        try:
            sol = Solution("Gurobi-VRPTW-Tardiness", instance, obj=sum(T[i].X for i in coords if i!=0))
        except TypeError:
            class _Sol:
                def __init__(self, alg, inst, obj):
                    self.algorithm = alg; self.instance = inst; self.objective = obj
                    self.comp_time = model.Runtime; self.status = "DONE"
                def __repr__(self):
                    return f"Schedule by {self.algorithm} - Objective: {self.objective}"
            sol = _Sol("Gurobi-VRPTW-Tardiness", instance, sum(T[i].X for i in coords if i!=0))

        sol.total_distance = sum(dist[i][j] * x[k,i,j].X for k in range(K) for i in coords for j in coords if i!=j and (k,i,j) in x)
        sol.unserved_ids = []  # model forced every customer visited exactly once
        sol.status = "DONE" if model.Status == GRB.OPTIMAL or model.Status == GRB.SUBOPTIMAL else "TIME_LIMIT"

        # attach routes as customer ID lists (map indices back to original customer IDs)
        sol.routes = []
        for k in range(K):
            route_cust_ids = []
            for idx in routes[k]:
                # map coords index -> actual customer ID
                # coords idx 1..N correspond to customers list order
                if 1 <= idx <= N:
                    route_cust_ids.append(customers[idx-1].ID)
            sol.routes.append(route_cust_ids)

        # write back arrival/start/end/tardy per customer if desired (non-destructive)
        for i in range(1, N+1):
            # find vehicle k that visited i
            for k in range(K):
                if (k,i) in y and y[k,i].X > 0.5:
                    cust_obj = customers[i-1]
                    cust_obj.assigned_vhc = vehicles[k].ID
                    cust_obj.start = t[k,i].X
                    cust_obj.end = t[k,i].X + service[i]
                    cust_obj.tardy = max(0.0, t[k,i].X + service[i] - due[i])
                    break

        return sol
    else:
        raise RuntimeError(f"Gurobi failed: status={model.Status}")
