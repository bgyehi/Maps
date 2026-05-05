# gurobi_solver_full.py
from __future__ import annotations
from typing import List, Tuple, Dict
import math
import gurobipy as gp
from gurobipy import GRB
import time
import matplotlib.pyplot as plt

import module
from module import Customer, Vehicle, Instance, Solution

def euclid(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])

def solve_vrp_gurobi(
    instance: Instance,
    *,
    time_limit: float = 60.0,
    mip_gap: float = 1e-3,
    use_warmstart: bool = True,
    verbose: bool = True
):
    customers: List[Customer] = getattr(instance, "customers", [])
    vehicles: List[Vehicle] = getattr(instance, "vehicles", [])
    depot = getattr(instance, "depot", (0.0, 0.0))

    N = len(customers)
    K = len(vehicles)

    coords = {i: (depot if i==0 else tuple(getattr(customers[i-1], "loc",(0,0)))) for i in range(N+1)}
    demand = {i: 0.0 if i==0 else float(getattr(customers[i-1], "weight", 0.0)) for i in range(N+1)}
    ready =  {i: 0.0 if i==0 else float(getattr(customers[i-1], "tw",[0,1e6])[0]) for i in range(N+1)}
    due =    {i: 1e6 if i==0 else float(getattr(customers[i-1], "tw",[0,1e6])[1]) for i in range(N+1)}
    service = {i: 0.0 if i==0 else float(getattr(customers[i-1], "serv_time",0.0)) for i in range(N+1)}
    veh_cap = {k: float(getattr(vehicles[k], "capacity", 1e9)) for k in range(K)}
    speeds = {k: float(getattr(vehicles[k], "speed", 1.0)) for k in range(K)}

    dist = {i:{j:euclid(coords[i], coords[j]) for j in coords} for i in coords}
    travel_t = {k:{i:{j:dist[i][j]/max(1e-6,speeds[k]) for j in coords} for i in coords} for k in range(K)}

    Tmax = max(due.values())+1000
    M = max(due.values())-min(ready.values()) + sum(demand.values()) + 1000

    # -------------------------
    # 모델 생성
    # -------------------------
    model = gp.Model("VRPTW_Tardiness")
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
                if i!=j:
                    x[k,i,j] = model.addVar(vtype=GRB.BINARY, name=f"x_{k}_{i}_{j}")
        for i in coords:
            if i!=0:
                y[k,i] = model.addVar(vtype=GRB.BINARY, name=f"y_{k}_{i}")
            t[k,i] = model.addVar(lb=0.0, ub=Tmax, vtype=GRB.CONTINUOUS, name=f"t_{k}_{i}")
            load[k,i] = model.addVar(lb=0.0, ub=veh_cap[k], vtype=GRB.CONTINUOUS, name=f"load_{k}_{i}")
    for i in coords:
        if i!=0:
            T[i] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"T_{i}")

    model.update()

    # -------------------------
    # 제약식 (time, capacity, flow)
    # -------------------------
    for i in range(1,N+1):
        model.addConstr(gp.quicksum(x[k,j,i] for k in range(K) for j in coords if j!=i) == 1)

    for k in range(K):
        model.addConstr(gp.quicksum(x[k,0,j] for j in coords if j!=0) == gp.quicksum(x[k,i,0] for i in range(1,N+1)))
        for i in range(1,N+1):
            in_sum = gp.quicksum(x[k,j,i] for j in coords if j!=i)
            out_sum = gp.quicksum(x[k,i,j] for j in coords if j!=i)
            model.addConstr(in_sum == y[k,i])
            model.addConstr(out_sum == y[k,i])

    for k in range(K):
        for i in coords:
            for j in coords:
                if i!=j:
                    model.addConstr(t[k,j] >= t[k,i] + service[i] + travel_t[k][i][j] - M*(1-x[k,i,j]))

    for k in range(K):
        for i in coords:
            if i!=0:
                model.addConstr(t[k,i] >= ready[i] - M*(1-y[k,i]))
                model.addConstr(load[k,i] <= veh_cap[k])
                model.addConstr(load[k,i] >= 0)
                model.addConstr(T[i] >= t[k,i]-due[i] - M*(1-y[k,i]))
        model.addConstr(t[k,0]>=0)
        model.addConstr(load[k,0]==0)

    for i in range(1,N+1):
        model.addConstr(T[i]>=0)

    model.setObjective(gp.quicksum(T[i] for i in range(1,N+1)), GRB.MINIMIZE)
    model.update()

    # -------------------------
    # Gurobi callback log
    # -------------------------
    performance_log = []
    def log_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            t_now = model.cbGet(GRB.Callback.RUNTIME)
            performance_log.append((t_now, obj))
    model.optimize(log_callback)

    # -------------------------
    # 결과 파싱
    # -------------------------
    sol = Solution("Gurobi-Tardiness", instance, obj=sum(T[i].X for i in range(1,N+1)))
    sol.comp_time = model.Runtime
    sol.status = "DONE" if model.Status in [GRB.OPTIMAL, GRB.SUBOPTIMAL] else "TIME_LIMIT"
    sol.performance_log = performance_log

    sol.routes=[]
    for k in range(K):
        route_cust_ids=[]
        cur=0
        visited=[]
        while True:
            found=False
            for j in coords:
                if j==cur: continue
                if (k,cur,j) in x and x[k,cur,j].X>0.5:
                    if j==0: cur=0; found=True; break
                    route_cust_ids.append(customers[j-1].ID)
                    cur=j
                    found=True
                    break
            if not found: break
        sol.routes.append(route_cust_ids)

    return sol

# -------------------------
# 간단 시각화 함수
# -------------------------
def plot_performance_log(sol: Solution):
    if hasattr(sol, "performance_log") and sol.performance_log:
        times, objs = zip(*sol.performance_log)
        plt.figure(figsize=(8,5))
        plt.plot(times, objs, marker='o')
        plt.xlabel("Elapsed Time (s)")
        plt.ylabel("Objective (Total Tardiness)")
        plt.title("Gurobi Performance Log")
        plt.grid(True)
        plt.show()

