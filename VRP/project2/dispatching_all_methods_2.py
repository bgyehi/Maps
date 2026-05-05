from __future__ import annotations
from typing import List, Optional
import copy
import random
import time
import math
from module import Customer, Vehicle, Instance, Solution, get_dist

def calculate_total_tardiness(vehicles: List[Vehicle]) -> float:
    all_ids = set()
    for v in vehicles:
        for c in v.schedules:
            all_ids.add(c.ID)
    total = 0.0
    for v in vehicles:
        speed = float(v.speed or 30.0)
        curr_loc, curr_time = tuple(v.loc), 0.0
        for c in v.schedules:
            dist = get_dist(curr_loc, tuple(c.loc))
            curr_time += dist / max(speed, 0.01)
            start = max(curr_time, float(c.tw[0]))
            end = start + float(c.serv_time or 0.0)
            total += max(0.0, end - float(c.tw[1]))
            curr_time, curr_loc = end, tuple(c.loc)
    return total

def deep_copy_solution(instance: Instance) -> Instance:
    return copy.deepcopy(instance)

def repair_vehicle_customers(instance: Instance):
    """모든 고객이 1번만 할당(누락/중복 방지)"""
    customers_by_id = {c.ID: c for c in instance.customers}
    all_ids = set(c.ID for c in instance.customers)
    seen = set()
    for v in instance.vehicles:
        clean = []
        for c in v.schedules:
            if c.ID in all_ids and c.ID not in seen:
                clean.append(customers_by_id[c.ID])
                seen.add(c.ID)
        v.schedules = clean
        v.now_capacity = sum(c.weight for c in v.schedules)
    # 빠진 고객 남으면 할당
    remaining = [c for c in instance.customers if c.ID not in seen]
    for c in remaining:
        found = False
        for v in instance.vehicles:
            if v.now_capacity + c.weight <= v.capacity:
                v.schedules.append(customers_by_id[c.ID])
                v.now_capacity += c.weight
                found = True; break
        if not found:  # 어느 차량에도 못 들어감 → 무시(강제 할당)
            instance.vehicles[0].schedules.append(customers_by_id[c.ID])
            instance.vehicles[0].now_capacity += c.weight

def dispatch_greedy(instance: Instance, rule: str = "EDD") -> Solution:
    instance.reset()
    if rule == "SPT":
        customers = sorted(instance.customers, key=lambda c: c.serv_time)
    else:
        customers = sorted(instance.customers, key=lambda c: c.tw[1])
    vehicles = instance.vehicles
    for v in vehicles: v.schedules, v.now_capacity = [], 0
    v_idx = 0
    for c in customers:
        assigned = False
        for _ in range(len(vehicles)):
            v = vehicles[v_idx % len(vehicles)]
            if v.now_capacity + c.weight <= v.capacity:
                v.schedules.append(c); v.now_capacity += c.weight
                assigned=True; v_idx += 1
                break
            v_idx += 1
        if not assigned:
            vehicles[0].schedules.append(c)
            vehicles[0].now_capacity += c.weight
    repair_vehicle_customers(instance)
    return Solution(rule, instance, obj=calculate_total_tardiness(vehicles))

def swap_customers(vehicles):
    vs = [v for v in vehicles if len(v.schedules) > 1]
    if len(vs) == 0: return
    v = random.choice(vs)
    i, j = random.sample(range(len(v.schedules)),2)
    v.schedules[i], v.schedules[j] = v.schedules[j], v.schedules[i]

def relocate_between_vehicles(vehicles):
    srcs = [v for v in vehicles if len(v.schedules)>1]
    if not srcs: return
    v_src = random.choice(srcs)
    dsts = [v for v in vehicles if v != v_src]
    if not dsts or len(v_src.schedules)==0: return
    v_dst = random.choice(dsts)
    idx = random.randrange(len(v_src.schedules))
    customer = v_src.schedules.pop(idx)
    v_dst.schedules.append(customer)

def two_opt_vehicle_route(vehicles):
    vs = [v for v in vehicles if len(v.schedules)>3]
    if not vs: return
    v = random.choice(vs)
    L = len(v.schedules)
    max_i = L-3
    if max_i < 0: return
    i = random.randint(0,max_i)
    j = random.randint(i+2, L-1)
    v.schedules[i:j+1] = reversed(v.schedules[i:j+1])

def strong_perturb(instance: Instance, p=0.2):
    vehicles = instance.vehicles
    all_cust = []
    for v in vehicles:
        all_cust.extend(v.schedules)
        v.schedules.clear(); v.now_capacity=0
    remove_n = max(1, int(len(all_cust)*p))
    to_remove = set(random.sample([c.ID for c in all_cust], min(len(all_cust), remove_n)))
    remaining = [c for c in all_cust if c.ID not in to_remove]
    insert = [c for c in all_cust if c.ID in to_remove]
    for c in remaining:
        v = min(vehicles, key=lambda v: v.now_capacity if v.now_capacity + c.weight <= v.capacity else 1e10)
        v.schedules.append(c); v.now_capacity += c.weight
    for c in insert:
        v = random.choice(vehicles)
        v.schedules.append(c)
        v.now_capacity += c.weight
    repair_vehicle_customers(instance)

def ils_optimize(instance: Instance, time_limit=10.0):
    vehicles = instance.vehicles
    repair_vehicle_customers(instance)
    best_cost = calculate_total_tardiness(vehicles)
    best_state = [list(v.schedules) for v in vehicles]
    start = time.time()
    while time.time() - start < time_limit:
        for _ in range(5):
            move = random.choice([swap_customers, relocate_between_vehicles, two_opt_vehicle_route])
            move(vehicles)
        if random.random() < 0.3:
            strong_perturb(instance)
        repair_vehicle_customers(instance)
        curr = calculate_total_tardiness(vehicles)
        if curr < best_cost:
            best_cost = curr
            best_state = [list(v.schedules) for v in vehicles]
    for v_idx, v in enumerate(vehicles): v.schedules = best_state[v_idx]
    repair_vehicle_customers(instance)
    return Solution("ILS", instance, best_cost)

def vns_optimize(instance: Instance, time_limit=10.0):
    vehicles = instance.vehicles
    best_cost = calculate_total_tardiness(vehicles)
    best_state = [list(v.schedules) for v in vehicles]
    start = time.time()
    while time.time() - start < time_limit:
        move = random.choice([swap_customers, relocate_between_vehicles, two_opt_vehicle_route])
        move(vehicles)
        if random.random() < 0.1:
            strong_perturb(instance)
        repair_vehicle_customers(instance)
        curr = calculate_total_tardiness(vehicles)
        if curr < best_cost:
            best_cost = curr
            best_state = [list(v.schedules) for v in vehicles]
    for v_idx, v in enumerate(vehicles): v.schedules = best_state[v_idx]
    repair_vehicle_customers(instance)
    return Solution("VNS", instance, best_cost)

def sa_optimize(instance: Instance, time_limit=10.0):
    vehicles = instance.vehicles
    repair_vehicle_customers(instance)
    best_cost = curr_cost = calculate_total_tardiness(vehicles)
    best_state = [list(v.schedules) for v in vehicles]
    start = time.time()
    T = best_cost * 0.5
    alpha = 0.98
    while time.time() - start < time_limit and T > 1e-3:
        old_state = [list(v.schedules) for v in vehicles]
        move = random.choice([swap_customers,relocate_between_vehicles,two_opt_vehicle_route])
        move(vehicles)
        repair_vehicle_customers(instance)
        new_cost = calculate_total_tardiness(vehicles)
        delta = new_cost - curr_cost
        if delta < 0 or random.random() < math.exp(-delta/max(T,1e-9)):
            curr_cost = new_cost
            if curr_cost < best_cost:
                best_cost = curr_cost
                best_state = [list(v.schedules) for v in vehicles]
        else:
            for v_idx, v in enumerate(vehicles): v.schedules = old_state[v_idx]
        if random.random() < 0.15: strong_perturb(instance)
        T *= alpha
    for v_idx, v in enumerate(vehicles): v.schedules = best_state[v_idx]
    repair_vehicle_customers(instance)
    return Solution("SA", instance, best_cost)

def gurobi_optimize(instance: Instance, time_limit: float = 600) -> Optional[Solution]:
    try:
        from gurobipy import Model, GRB, quicksum
    except ImportError:
        print("[Gurobi] Guro비 미설치")
        return None
    customers = instance.customers
    vehicles = instance.vehicles
    n = len(customers)
    m = len(vehicles)
    mip_gap = 0.01 if n<=20 else 0.05
    print(f"[Gurobi] 시작 (고객 {n}개, 차량 {m}대, MIPGap={int(mip_gap*100)}%, 시간={time_limit}초)")
    model = Model("VRPTW_Gurobi")
    model.Params.TimeLimit = time_limit
    model.Params.MIPGap = mip_gap
    model.Params.OutputFlag = 0
    model.Params.Threads = 4
    if n > 30:
        model.Params.MIPFocus = 1
        model.Params.Heuristics = 0.5
    x = {}
    for v in range(m):
        for i in range(n):
            for j in range(n):
                if i != j: x[v, i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{v}_{i}_{j}")
    visit = model.addVars(m, n, vtype=GRB.BINARY, name="visit")
    t = model.addVars(m, n, vtype=GRB.CONTINUOUS, lb=0, name="t")
    tardy = model.addVars(m, n, vtype=GRB.CONTINUOUS, lb=0, name="tardy")
    u = model.addVars(m, n, vtype=GRB.CONTINUOUS, lb=0, ub=n, name="u")
    model.update()
    model.setObjective(quicksum(tardy[v,i] for v in range(m) for i in range(n)), GRB.MINIMIZE)
    for i in range(n): model.addConstr(quicksum(visit[v, i] for v in range(m)) == 1)
    for v in range(m):
        for i in range(n):
            model.addConstr(quicksum(x[v, j, i] for j in range(n) if j != i) == visit[v,i])
            model.addConstr(quicksum(x[v, i, j] for j in range(n) if j != i) == visit[v,i])
    for v in range(m): model.addConstr(quicksum(customers[i].weight * visit[v,i] for i in range(n)) <= vehicles[v].capacity)
    for v in range(m):
        for i in range(n):
            for j in range(n):
                if i != j: model.addConstr(u[v,j] >= u[v,i]+1-n*(1-x[v,i,j]))
    M = 100000
    for v in range(m):
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist = get_dist(tuple(customers[i].loc), tuple(customers[j].loc))
                    travel_time = dist / vehicles[v].speed
                    model.addConstr(t[v,j] >= t[v,i]+customers[i].serv_time+travel_time-M*(1-x[v,i,j]))
    for v in range(m):
        for i in range(n):
            model.addConstr(t[v,i] >= customers[i].tw[0] - M*(1-visit[v,i]))
            completion = t[v,i]+customers[i].serv_time
            due = customers[i].tw[1]
            model.addConstr(tardy[v,i] >= completion-due)
    start_time = time.time()
    model.optimize()
    elapsed = time.time() - start_time
    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL] and getattr(model,"SolCount",0)>0 and model.ObjVal<1e8:
        print(f"[Gurobi] 해 발견! Obj={model.ObjVal:.2f}, Gap={getattr(model,'MIPGap',0)*100:.2f}%, Time={elapsed:.1f}s")
        for v_idx in range(m):
            vehicles[v_idx].schedules = []
            visited = [(i, u[v_idx,i].X) for i in range(n) if visit[v_idx,i].X>0.5]
            visited.sort()
            for _, cust_idx in visited:
                vehicles[v_idx].schedules.append(customers[cust_idx])
        final_tard = calculate_total_tardiness(vehicles)
        return Solution("Gurobi", instance, obj=final_tard)
    else:
        print(f"[Gurobi] feasible 정수해 없음 (SolCount={getattr(model,'SolCount',0)})")
        return None
