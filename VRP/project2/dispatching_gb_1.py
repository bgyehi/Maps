from __future__ import annotations
from typing import List, Tuple, Dict, Optional
import math
import random
import copy
import time

# 프로젝트 모듈 (사용자 환경에 이미 존재)
import module
from module import Customer, Vehicle, Instance, Solution

# -------------------------
# 보조: 거리
# -------------------------
def euclid(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])

# -------------------------
# 1) Greedy dispatch:
# Earliest-Available Vehicle + Best-Customer (delta tardiness heuristic)
# 반환: Solution-like object (Solution 또는 간단 오브젝트)
# -------------------------
def dispatch_earliest_vehicle_best_customer(instance: Instance, *, scoring: str = "delta_tardiness", tie_breaker: str = "edd_then_nearest") -> Solution:
    """
    Simple greedy dispatcher:
    - iterate until all customers assigned or no feasible assignment
    - choose next vehicle (earliest available) and pick best customer minimizing delta tardiness
    - returns Solution object compatible with main code (set .routes, .objective, .status, .routes, vehicle.schedules)
    Assumes:
      instance.customers : list of Customer with fields ID, loc, tw (ready,due), serv_time, weight(optional)
      instance.vehicles  : list of Vehicle with fields ID, capacity, speed, now_loc (optional)
    """
    customers = [copy.deepcopy(c) for c in getattr(instance, "customers", [])]
    vehicles = [copy.deepcopy(v) for v in getattr(instance, "vehicles", [])]

    N = len(customers)
    K = len(vehicles)
    # coords map customer id -> loc
    cust_by_id = {c.ID: c for c in customers}

    # initialize vehicle schedules
    for v in vehicles:
        v.schedules = []
        v.now = tuple(getattr(v, "now_loc", getattr(v, "start_loc", getattr(instance, "depot", (0.0,0.0)))))

        # vehicle available time and remaining capacity
        v.available_at = 0.0
        v.load = 0.0

    unassigned = set(c.ID for c in customers)

    # helper: compute tardiness contribution if appended after vehicle current route
    def simulate_append(v: Vehicle, cust: Customer):
        # determine arrival time at cust if appended to v.schedules
        locs = [tuple(getattr(cust_by_id[cid], "loc", (0.0,0.0))) for cid in v.schedules]
        start_loc = v.now if not locs else tuple(getattr(cust_by_id[v.schedules[-1]], "loc", (0.0,0.0)))
        travel = euclid(start_loc, tuple(getattr(cust, "loc", (0.0,0.0))))
        arrival = max(getattr(cust, "tw", [0,1e6])[0], v.available_at + travel)
        finish = arrival + getattr(cust, "serv_time", 0.0)
        due = getattr(cust, "tw", [0,1e6])[1]
        tard = max(0.0, finish - due)
        return tard, arrival, finish

    # main greedy loop
    while unassigned:
        # pick vehicle with earliest available time (simple)
        vehicles_sorted = sorted(vehicles, key=lambda vv: vv.available_at)
        assigned_any = False
        for v in vehicles_sorted:
            # build candidate list of unassigned customers that fit capacity (if capacity exists)
            cand = []
            for cid in list(unassigned):
                c = cust_by_id[cid]
                demand = float(getattr(c, "weight", 0.0))
                if getattr(v, "capacity", 1e9) < v.load + demand:
                    continue
                # compute delta tardiness (approx): tardiness if appended - 0 (we consider incremental tardiness)
                tard, arr, fin = simulate_append(v, c)
                cand.append((cid, tard, arr, fin))
            if not cand:
                # this vehicle cannot take further customers now
                continue

            # select best candidate: minimize tard, tie -> earliest due (EDD) -> nearest
            cand.sort(key=lambda x: (x[1], getattr(cust_by_id[x[0]], "tw", [0,1e6])[1], euclid(v.now if not v.schedules else tuple(getattr(cust_by_id[v.schedules[-1]], "loc",(0,0))), tuple(getattr(cust_by_id[x[0]], "loc",(0,0))))))
            best = cand[0]
            cid, tard, arr, fin = best
            # assign
            v.schedules.append(cid)
            v.available_at = fin
            v.load += float(getattr(cust_by_id[cid], "weight", 0.0))
            # update vehicle now location
            v.now = tuple(getattr(cust_by_id[cid], "loc", (0.0,0.0)))
            unassigned.remove(cid)
            assigned_any = True
            # after assigning one customer to this vehicle, continue outer while (re-sort vehicles)
            break

        if not assigned_any:
            # no vehicle could take any remaining customer -> infeasible scheduling for greedy
            break

    # build Solution-like object
    try:
        sol = Solution("Greedy-Dispatch", instance, obj=0.0)
    except TypeError:
        class _Sol:
            def __init__(self, alg, inst, obj):
                self.algorithm = alg; self.instance = inst; self.objective = obj
                self.comp_time = 0.0; self.status = "DONE"
            def __repr__(self):
                return f"Schedule by {self.algorithm} - Obj: {self.objective}"
        sol = _Sol("Greedy-Dispatch", instance, 0.0)

    # attach schedules back to original instance vehicles (by ID order)
    sol.routes = []
    for idx, v_orig in enumerate(getattr(instance, "vehicles", [])):
        if idx < len(vehicles):
            sol.routes.append(list(vehicles[idx].schedules))
            # also update the original vehicle object schedules if present
            try:
                instance.vehicles[idx].schedules = list(vehicles[idx].schedules)
            except Exception:
                pass
        else:
            sol.routes.append([])

    # compute objective (total tardiness) by simulating each vehicle route
    total_tardy = 0.0
    for kidx, r in enumerate(sol.routes):
        time_now = 0.0
        start_loc = tuple(getattr(instance.vehicles[kidx], "now_loc", getattr(instance, "depot", (0.0,0.0))))
        for cid in r:
            c = cust_by_id[cid]
            travel = euclid(start_loc, tuple(getattr(c, "loc", (0.0,0.0))))
            arrival = max(getattr(c, "tw", [0,1e6])[0], time_now + travel)
            finish = arrival + getattr(c, "serv_time", 0.0)
            due = getattr(c, "tw", [0,1e6])[1]
            total_tardy += max(0.0, finish - due)
            time_now = finish
            start_loc = tuple(getattr(c, "loc", (0.0,0.0)))
    sol.objective = total_tardy
    sol.status = "DONE" if len(unassigned) == 0 else "INFEASIBLE_UNSERVED"

    return sol

# -------------------------
# 2) Simple improvement: run_vns_improvement
# A lightweight VNS: try intra-route swaps/inserts to reduce tardiness
# -------------------------
def run_vns_improvement(solution: Solution, time_budget: float = 0.1) -> float:
    """
    Simple improvement routine that mutates solution in-place (expects vehicle.schedules in instance)
    returns new total tardiness (float).
    """
    instance = solution.instance
    customers = {c.ID: c for c in getattr(instance, "customers", [])}
    vehicles = instance.vehicles
    start_time = time.time()

    # helper to compute total tardiness
    def compute_total_tard(routes):
        total = 0.0
        for vidx, route in enumerate(routes):
            time_now = 0.0
            start_loc = tuple(getattr(instance.vehicles[vidx], "now_loc", getattr(instance, "depot", (0.0,0.0))))
            for cid in route:
                c = customers[cid]
                travel = euclid(start_loc, tuple(getattr(c, "loc", (0.0,0.0))))
                arrival = max(getattr(c, "tw", [0,1e6])[0], time_now + travel)
                finish = arrival + getattr(c, "serv_time", 0.0)
                total += max(0.0, finish - getattr(c, "tw", [0,1e6])[1])
                time_now = finish
                start_loc = tuple(getattr(c, "loc", (0.0,0.0)))
        return total

    # initial routes
    routes = [list(getattr(v, "schedules", [])) for v in vehicles]
    best_routes = copy.deepcopy(routes)
    best_cost = compute_total_tard(best_routes)

    # simple VNS neighborhoods: intra-route swap, insert; inter-route move
    while time.time() - start_time < time_budget:
        improved = False
        # intra-route swap
        for r_idx, r in enumerate(routes):
            n = len(r)
            if n < 2:
                continue
            for i in range(n):
                for j in range(i+1, n):
                    new_r = r[:]
                    new_r[i], new_r[j] = new_r[j], new_r[i]
                    cand = routes[:]
                    cand[r_idx] = new_r
                    cand_cost = compute_total_tard(cand)
                    if cand_cost < best_cost:
                        best_cost = cand_cost
                        best_routes = copy.deepcopy(cand)
                        routes = copy.deepcopy(cand)
                        improved = True
                        break
                if improved or time.time() - start_time >= time_budget:
                    break
            if improved:
                break
        if improved:
            continue

        # intra-route insert
        for r_idx, r in enumerate(routes):
            n = len(r)
            if n < 2:
                continue
            done = False
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    cand_r = r[:]
                    task = cand_r.pop(i)
                    cand_r.insert(j, task)
                    cand = routes[:]
                    cand[r_idx] = cand_r
                    cand_cost = compute_total_tard(cand)
                    if cand_cost < best_cost:
                        best_cost = cand_cost
                        best_routes = copy.deepcopy(cand)
                        routes = copy.deepcopy(cand)
                        improved = True
                        done = True
                        break
                if done or time.time() - start_time >= time_budget:
                    break
            if improved or time.time() - start_time >= time_budget:
                break

        # inter-route simple relocate
        if not improved:
            for r1 in range(len(routes)):
                for r2 in range(len(routes)):
                    if r1 == r2 or not routes[r1]:
                        continue
                    for i in range(len(routes[r1])):
                        cand = copy.deepcopy(routes)
                        task = cand[r1].pop(i)
                        cand[r2].append(task)
                        cand_cost = compute_total_tard(cand)
                        if cand_cost < best_cost:
                            best_cost = cand_cost
                            best_routes = copy.deepcopy(cand)
                            routes = copy.deepcopy(cand)
                            improved = True
                            break
                    if improved or time.time() - start_time >= time_budget:
                        break
                if improved or time.time() - start_time >= time_budget:
                    break

        if not improved:
            break

    # write back best_routes into instance vehicles schedules
    for vidx, v in enumerate(vehicles):
        try:
            v.schedules = list(best_routes[vidx])
        except Exception:
            v.schedules = []

    solution.objective = best_cost
    return best_cost

# -------------------------
# 3) perturb_solution: 안전한 교란 (작은 강도)
# -------------------------
def perturb_solution(solution: Solution, *, strength: int = 2) -> None:
    """
    Make a small random perturbation to solution in-place.
    strength: how many random swaps/moves to apply
    """
    instance = solution.instance
    vehicles = instance.vehicles
    # collect all assigned tasks
    all_assigned = []
    for v in vehicles:
        all_assigned += list(getattr(v, "schedules", []))

    if not all_assigned:
        return

    for _ in range(max(1, strength)):
        # choose random vehicle and operation
        v1 = random.choice(vehicles)
        v2 = random.choice(vehicles)
        if not getattr(v1, "schedules", []):
            continue
        if v1 is v2 or not getattr(v2, "schedules", []):
            # intra-route swap
            r = v1.schedules
            if len(r) >= 2:
                i = random.randrange(len(r))
                j = random.randrange(len(r))
                r[i], r[j] = r[j], r[i]
        else:
            # move random task from v1 to v2 (if capacity allows, naive)
            if v1.schedules:
                i = random.randrange(len(v1.schedules))
                task = v1.schedules.pop(i)
                v2.schedules.insert(random.randrange(len(v2.schedules)+1), task)

    # no return; modify instance in-place
    return
