# dp_1.py
from __future__ import annotations
from typing import List, Tuple
import math
import gurobipy as gp
from gurobipy import GRB
import time
import matplotlib.pyplot as plt

# 사용자 프로젝트 모듈
import module
from module import Customer, Vehicle, Instance, Solution

# -------------------------
# 보조 함수
# -------------------------
def euclid(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

# -------------------------
# Gurobi MIP: VRPTW (min total tardiness)
# -------------------------
def solve_vrp_gurobi(
        instance: Instance,
        *,
        time_limit: float = 300.0,  # 기본 5분
        mip_gap: float = 1e-3,
        use_warmstart: bool = True,
        verbose: bool = True
):
    """
    Gurobi로 푸는 VRPTW + capacity 모델 (objective = 총 tardiness)
    개선사항:
        - TimeLimit 내 최적해 가져오기
        - Big-M 최소화
        - Warmstart 적용 가능
        - Callback 기반 performance log
    """
    # -------------------------
    # instance -> 모델 입력 변환
    # -------------------------
    customers: List[Customer] = list(getattr(instance, "customers", []))
    vehicles: List[Vehicle] = list(getattr(instance, "vehicles", []))

    depot = getattr(instance, "depot", None)
    if (not depot) or (not isinstance(depot, (tuple, list))):
        if vehicles and getattr(vehicles[0], "now_loc", None):
            depot = tuple(vehicles[0].now_loc)
        else:
            depot = (0.0, 0.0)

    N = len(customers)
    nodes = [("depot", depot)] + [(c.ID, tuple(getattr(c, "loc", (0.0, 0.0)))) for c in customers]

    demand = {i + 1: float(getattr(customers[i], "weight", 0.0)) for i in range(N)}
    ready = {i + 1: float(getattr(customers[i], "tw", [0.0, 1e6])[0]) for i in range(N)}
    due = {i + 1: float(getattr(customers[i], "tw", [0.0, 1e6])[1]) for i in range(N)}
    service = {i + 1: float(getattr(customers[i], "serv_time", 0.0)) for i in range(N)}
    ready[0] = 0.0
    due[0] = 1e6
    service[0] = 0.0

    K = len(vehicles)
    veh_cap = {k: float(getattr(vehicles[k], "capacity", 1e9)) for k in range(K)}
    speeds = {k: float(getattr(vehicles[k], "speed", 1.0)) for k in range(K)}

    coords = {idx: coord for idx, coord in
              enumerate([depot] + [tuple(getattr(c, "loc", (0.0, 0.0))) for c in customers])}
    dist = {i: {j: euclid(coords[i], coords[j]) for j in coords} for i in coords}
    travel_t = {k: {i: {j: dist[i][j] / max(1e-6, speeds[k]) for j in coords} for i in coords} for k in range(K)}

    # Big-M 최소화
    Tmax = max(due.values()) + 1000.0
    M = max(due.values()) - min(ready.values()) + sum(demand.values()) + 1000.0

    # -------------------------
    # 모델 생성
    # -------------------------
    model = gp.Model("VRPTW_min_tardiness")
    model.setParam("TimeLimit", time_limit)
    model.setParam("MIPGap", mip_gap)
    model.setParam("OutputFlag", 1 if verbose else 0)
    model.setParam("Threads", 4)
    model.setParam("MIPFocus", 1)
    model.setParam("Heuristics", 0.6)
    model.setParam("Presolve", 2)
    model.setParam("NodefileStart", 0.5)

    # -------------------------
    # 변수
    # -------------------------
    x, y, t, load, T = {}, {}, {}, {}, {}

    for k in range(K):
        for i in coords:
            for j in coords:
                if i == j:
                    continue
                x[k, i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{k}_{i}_{j}")

    for k in range(K):
        for i in coords:
            if i == 0: continue
            y[k, i] = model.addVar(vtype=GRB.BINARY, name=f"y_{k}_{i}")

    for k in range(K):
        for i in coords:
            t[k, i] = model.addVar(lb=0.0, ub=Tmax, vtype=GRB.CONTINUOUS, name=f"t_{k}_{i}")
            load[k, i] = model.addVar(lb=0.0, ub=veh_cap[k], vtype=GRB.CONTINUOUS, name=f"load_{k}_{i}")

    for i in coords:
        if i == 0: continue
        T[i] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"T_{i}")

    model.update()

    # -------------------------
    # 제약식
    # -------------------------
    for i in coords:
        if i == 0: continue
        model.addConstr(gp.quicksum(x[k, j, i] for k in range(K) for j in coords if j != i) == 1,
                        name=f"visit_once_{i}")

    for k in range(K):
        model.addConstr(
            gp.quicksum(x[k, 0, j] for j in coords if j != 0) == gp.quicksum(x[k, i, 0] for i in coords if i != 0),
            name=f"depot_balance_{k}")
        for i in coords:
            if i == 0: continue
            in_sum = gp.quicksum(x[k, j, i] for j in coords if j != i)
            out_sum = gp.quicksum(x[k, i, j] for j in coords if j != i)
            model.addConstr(in_sum == y[k, i], name=f"in_is_y_{k}_{i}")
            model.addConstr(out_sum == y[k, i], name=f"out_is_y_{k}_{i}")

    for k in range(K):
        for i in coords:
            for j in coords:
                if i == j: continue
                model.addConstr(t[k, j] >= t[k, i] + service.get(i, 0.0) + travel_t[k][i][j] - M * (1 - x[k, i, j]),
                                name=f"time_link_{k}_{i}_{j}")

    for k in range(K):
        for i in coords:
            lb = ready.get(i, 0.0)
            model.addConstr(t[k, i] >= lb - M * (1 - (y.get((k, i), 1) if (k, i) in y else 1)),
                            name=f"ready_lb_{k}_{i}")

    for k in range(K):
        for i in coords:
            for j in coords:
                if i == j: continue
                dj = demand.get(j, 0.0)
                model.addConstr(load[k, j] >= load[k, i] + dj - M * (1 - x[k, i, j]), name=f"load_link_{k}_{i}_{j}")

    for k in range(K):
        for i in coords:
            model.addConstr(load[k, i] <= veh_cap[k])
            model.addConstr(load[k, i] >= 0.0)

    for k in range(K):
        model.addConstr(t[k, 0] >= 0.0)
        model.addConstr(load[k, 0] == 0.0)

    for i in coords:
        if i == 0: continue
        for k in range(K):
            model.addConstr(T[i] >= t[k, i] - due[i] - M * (1 - y[k, i]))
        model.addConstr(T[i] >= 0.0)

    # -------------------------
    # Objective
    # -------------------------
    model.setObjective(gp.quicksum(T[i] for i in coords if i != 0), GRB.MINIMIZE)
    model.update()

    # -------------------------
    # Warm Start
    # -------------------------
    if use_warmstart:
        try:
            from dispatching import dispatch_earliest_vehicle_best_customer
            heur_sol = dispatch_earliest_vehicle_best_customer(instance,
                                                               scoring="delta_tardiness",
                                                               tie_breaker="edd_then_nearest")
            # 적용 가능한 경우 warmstart
        except Exception as e:
            if verbose:
                print("Warmstart skipped:", e)

    # -------------------------
    # Callback + Optimize
    # -------------------------
    performance_log = []

    def gurobi_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            t_now = model.cbGet(GRB.Callback.RUNTIME)
            performance_log.append((t_now, obj))

    start_time = time.time()
    model.optimize(gurobi_callback)
    end_time = time.time()
    comp_time = end_time - start_time

    # -------------------------
    # 결과 파싱 (TimeLimit 내 최적해 안전 처리)
    # -------------------------
    routes = {k: [] for k in range(K)}

    if model.SolCount > 0:
        for k in range(K):
            cur = 0
            loop_guard = 0
            while True:
                found = False
                for j in coords:
                    if j == cur: continue
                    if (k, cur, j) in x and x[k, cur, j].X > 0.5:
                        if j == 0: cur = 0; found = True; break
                        routes[k].append(j)
                        cur = j
                        found = True
                        break
                if not found: break
                loop_guard += 1
                if loop_guard > N + 5: break

        sol = Solution("Gurobi-VRPTW-Tardiness", instance,
                       obj=sum(T[i].X for i in coords if i != 0))
        sol.total_distance = sum(
            dist[i][j] * x[k, i, j].X for k in range(K) for i in coords for j in coords if i != j and (k, i, j) in x)
        sol.unserved_ids = []
        sol.status = "DONE" if model.Status in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT] else "UNKNOWN"
        sol.routes = []
        for k in range(K):
            route_cust_ids = []
            for idx in routes[k]:
                if 1 <= idx <= N:
                    route_cust_ids.append(customers[idx - 1].ID)
            sol.routes.append(route_cust_ids)

        for i in range(1, N + 1):
            for k in range(K):
                if (k, i) in y and y[k, i].X > 0.5:
                    cust_obj = customers[i - 1]
                    cust_obj.assigned_vhc = vehicles[k].ID
                    cust_obj.start = t[k, i].X
                    cust_obj.end = t[k, i].X + service[i]
                    cust_obj.tardy = max(0.0, t[k, i].X + service[i] - due[i])
                    break

        sol.comp_time = comp_time
        sol.performance_log = performance_log

    else:
        print(f"[Warning] {instance} 모델에서 유효 솔루션 없음. 빈 Solution 반환.")
        sol = Solution("Gurobi-VRPTW-Tardiness", instance, obj=None)
        sol.routes = []
        sol.total_distance = 0
        sol.unserved_ids = [c.ID for c in customers]
        sol.comp_time = comp_time
        sol.performance_log = performance_log
        sol.status = "NO_SOLUTION"

    # -------------------------
    # 성능 시각화
    # -------------------------
    if performance_log:
        times, objs = zip(*performance_log)
        plt.figure(figsize=(8, 5))
        plt.plot(times, objs, marker='o')
        plt.xlabel("Elapsed Time (s)")
        plt.ylabel("Objective (Total Tardiness)")
        plt.title("Gurobi Performance Log")
        plt.grid(True)
        plt.show()

    return sol
