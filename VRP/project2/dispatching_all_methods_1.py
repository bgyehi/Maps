from __future__ import annotations
from typing import List, Optional
import copy
import random
import time
import math
from module import Customer, Vehicle, Instance, Solution, get_dist

def calculate_total_tardiness(vehicles: List[Vehicle]) -> float:
    total = 0.0
    for v in vehicles:
        speed = float(v.speed or 30.0)
        curr_loc, curr_time = tuple(v.loc), 0.0
        for c in v.schedules:
            dist = get_dist(curr_loc, tuple(c.loc))
            curr_time += dist / max(speed, 1e-9)
            start = max(curr_time, float(c.tw[0]))
            end = start + float(c.serv_time or 0.0)
            total += max(0.0, end - float(c.tw[1]))
            curr_time, curr_loc = end, tuple(c.loc)
    return total

def deep_copy_solution(instance: Instance) -> Instance:
    return copy.deepcopy(instance)

def dispatch_spt(instance: Instance) -> Solution:
    instance.reset()
    customers = sorted(instance.customers, key=lambda c: c.serv_time)
    vehicles = instance.vehicles
    for v in vehicles: v.schedules, v.now_capacity = [], 0
    v_idx = 0
    for c in customers:
        assigned = False
        for _ in range(len(vehicles)):
            v = vehicles[v_idx % len(vehicles)]
            if v.now_capacity + c.weight <= v.capacity:
                v.schedules.append(c)
                v.now_capacity += c.weight
                assigned = True; v_idx += 1
                break
            v_idx += 1
        if not assigned: vehicles[0].schedules.append(c)
    return Solution("SPT", instance, obj=calculate_total_tardiness(vehicles))

def dispatch_edd(instance: Instance) -> Solution:
    instance.reset()
    customers = sorted(instance.customers, key=lambda c: c.tw[1])
    vehicles = instance.vehicles
    for v in vehicles: v.schedules, v.now_capacity = [], 0
    v_idx = 0
    for c in customers:
        assigned = False
        for _ in range(len(vehicles)):
            v = vehicles[v_idx % len(vehicles)]
            if v.now_capacity + c.weight <= v.capacity:
                v.schedules.append(c)
                v.now_capacity += c.weight
                assigned = True; v_idx += 1
                break
            v_idx += 1
        if not assigned: vehicles[0].schedules.append(c)
    return Solution("EDD", instance, obj=calculate_total_tardiness(vehicles))

def dispatch_regret_k(instance: Instance, k: int = 3) -> Solution:
    instance.reset()
    customers = instance.customers
    vehicles = instance.vehicles
    for v in vehicles: v.schedules, v.now_capacity = [], 0.0
    unserved = set(c.ID for c in customers)
    cust_by_id = {c.ID: c for c in customers}
    while unserved:
        best_regret, best = -1, None
        for cid in list(unserved):
            c = cust_by_id[cid]
            costs = []
            for v_idx, v in enumerate(vehicles):
                if v.now_capacity + c.weight > v.capacity: continue
                for pos in range(len(v.schedules)+1):
                    v.schedules.insert(pos, c)
                    cost = calculate_total_tardiness(vehicles)
                    v.schedules.pop(pos)
                    costs.append((cost, v_idx, pos))
            if len(costs) < k: continue
            costs.sort()
            regret = costs[min(k-1, len(costs)-1)][0] - costs[0][0]
            if regret > best_regret:
                best_regret, best = regret, (cid, costs[0][1], costs[0][2])
        if best is None: break
        cid, v_idx, pos = best
        c = cust_by_id[cid]
        vehicles[v_idx].schedules.insert(pos, c)
        vehicles[v_idx].now_capacity += c.weight
        unserved.remove(cid)
    return Solution("Regret-k", instance, obj=calculate_total_tardiness(vehicles))

# --- Neighborhoods for metaheuristics ---
def swap(vehicles):
    candidates = [v for v in vehicles if len(v.schedules) > 1]
    if len(candidates) < 2: return
    v1, v2 = random.sample(candidates, 2)
    i = random.randint(0, len(v1.schedules)-1)
    j = random.randint(0, len(v2.schedules)-1)
    v1.schedules[i], v2.schedules[j] = v2.schedules[j], v1.schedules[i]

def relocate(vehicles):
    srcs = [v for v in vehicles if len(v.schedules) > 1]
    dsts = [v for v in vehicles]
    if not srcs: return
    v_src = random.choice(srcs)
    v_dst = random.choice(dsts)
    if v_src == v_dst or len(v_src.schedules) < 1: return
    i = random.randint(0, len(v_src.schedules)-1)
    customer = v_src.schedules.pop(i)
    v_dst.schedules.append(customer)

def two_opt_route(vehicles):
    candidates = [v for v in vehicles if len(v.schedules) > 3]
    if not candidates: return
    v = random.choice(candidates)
    route_len = len(v.schedules)
    max_i = route_len - 3
    if max_i < 0: return
    i = random.randint(0, max_i)
    j = random.randint(i+2, route_len-1)
    v.schedules[i:j+1] = reversed(v.schedules[i:j+1])

# -------- Metaheuristics --------
def ils_optimize(instance: Instance, time_limit: float = 10.0) -> Solution:
    vehicles = instance.vehicles
    best_cost = calculate_total_tardiness(vehicles)
    best_state = [list(v.schedules) for v in vehicles]
    start = time.time()
    while time.time() - start < time_limit:
        for _ in range(3):
            swap(vehicles)
            relocate(vehicles)
            two_opt_route(vehicles)
        curr_cost = calculate_total_tardiness(vehicles)
        if curr_cost < best_cost:
            best_cost, best_state = curr_cost, [list(v.schedules) for v in vehicles]
        for v in vehicles:
            if random.random() < 0.2 and len(v.schedules) > 4:
                random.shuffle(v.schedules)
    for v_idx, v in enumerate(vehicles): v.schedules = best_state[v_idx]
    return Solution("ILS", instance, obj=best_cost)

def vns_optimize(instance: Instance, time_limit: float = 10.0) -> Solution:
    vehicles = instance.vehicles
    best_cost = calculate_total_tardiness(vehicles)
    best_state = [list(v.schedules) for v in vehicles]
    start = time.time()
    while time.time() - start < time_limit:
        move = random.choice([swap, relocate, two_opt_route])
        move(vehicles)
        curr_cost = calculate_total_tardiness(vehicles)
        if curr_cost < best_cost:
            best_cost, best_state = curr_cost, [list(v.schedules) for v in vehicles]
        else:
            for v in vehicles:
                if random.random() < 0.1 and len(v.schedules) >= 2:
                    random.shuffle(v.schedules)
    for v_idx, v in enumerate(vehicles): v.schedules = best_state[v_idx]
    return Solution("VNS", instance, obj=best_cost)

def sa_optimize(instance: Instance, time_limit: float = 10.0) -> Solution:
    vehicles = instance.vehicles
    current_cost = calculate_total_tardiness(vehicles)
    best_cost, best_state = current_cost, [list(v.schedules) for v in vehicles]
    T = 200.0
    T_min = 0.05
    alpha = 0.98
    start = time.time()
    while time.time()-start < time_limit and T > T_min:
        old_state = [list(v.schedules) for v in vehicles]
        move = random.choice([swap, relocate, two_opt_route])
        move(vehicles)
        new_cost = calculate_total_tardiness(vehicles)
        delta = new_cost - current_cost
        if delta < 0 or random.random() < math.exp(-delta / max(T,1e-9)):
            current_cost = new_cost
            if current_cost < best_cost: best_cost, best_state = current_cost, [list(v.schedules) for v in vehicles]
        else: # revert
            for v_idx, v in enumerate(vehicles):
                v.schedules = old_state[v_idx]
        T *= alpha
    for v_idx, v in enumerate(vehicles): v.schedules = best_state[v_idx]
    return Solution("SA", instance, obj=best_cost)

def ga_optimize(instance: Instance, time_limit: float = 10.0) -> Solution:
    pop_size = 30
    vehicles = instance.vehicles
    population = []
    for _ in range(pop_size//2):
        inst_copy = deep_copy_solution(instance)
        sol = dispatch_regret_k(inst_copy, k=3)
        population.append((sol.objective, [list(v.schedules) for v in inst_copy.vehicles]))
    for _ in range(pop_size//2):
        inst_copy = deep_copy_solution(instance)
        sol = dispatch_edd(inst_copy)
        population.append((sol.objective, [list(v.schedules) for v in inst_copy.vehicles]))
    start = time.time()
    while time.time()-start < time_limit:
        parents = sorted(population, key=lambda x:x[0])[:pop_size//2]
        offspring = []
        for i in range(0, len(parents)-1, 2):
            p1, p2 = parents[i][1], parents[i+1][1]
            child = copy.deepcopy(p1)
            v_idx = random.randint(0, len(child)-1)
            if len(child[v_idx]) > 2:
                cut = len(child[v_idx]) // 2
                child[v_idx][:cut] = p2[v_idx][:cut]
            # mutation
            if random.random() < 0.5:
                mutate_fn = random.choice([swap, relocate, two_opt_route])
                mutate_fn(vehicles)
            offspring.append(child)
        for child in offspring:
            for v_idx, v in enumerate(vehicles):
                v.schedules = child[v_idx]
            cost = calculate_total_tardiness(vehicles)
            population.append((cost, child))
        population = sorted(population, key=lambda x:x[0])[:pop_size]
    best_cost, best_state = population[0]
    for v_idx, v in enumerate(vehicles): v.schedules = best_state[v_idx]
    return Solution("GA", instance, obj=best_cost)

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
                    service_time = customers[i].serv_time
                    model.addConstr(t[v,j] >= t[v,i]+service_time+travel_time-M*(1-x[v,i,j]))
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
