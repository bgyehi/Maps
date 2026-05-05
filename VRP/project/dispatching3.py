# dispatching.py
from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import random
import time
import copy

import module
from module import Customer, Vehicle, Instance, Solution, get_dist

# ----------------------
# Utility: route cost "심판" (stateless)
# ----------------------
def _calculate_route_cost_and_update_customer_info(vehicle: 'Vehicle', route: List['Customer']) -> float:
    """
    route(고객 리스트)만으로 비용(총 tardiness)을 계산.
    용량 초과 시 float('inf') 반환.
    """
    depot_loc = tuple(getattr(vehicle, "loc", getattr(vehicle, "now_loc", (0.0, 0.0))))
    speed = float(getattr(vehicle, "speed", 30.0))
    capacity = float(getattr(vehicle, "capacity", float("inf")))

    cur_time = 0.0
    cur_loc = depot_loc
    total_tard = 0.0
    load = 0.0

    for c in route:
        load += float(c.weight or 0.0)
        if load > capacity + 1e-9:
            return float("inf")
        dist = get_dist(tuple(cur_loc), tuple(c.loc))
        travel_h = dist / max(1e-9, speed)
        arrive = cur_time + travel_h
        start = max(arrive, float(c.tw[0]))
        end = start + float(c.serv_time or 0.0)
        tard = max(0.0, end - float(c.tw[1]))

        # update customer info (for visualization/debug)
        c.start = start
        c.end = end
        c.tardy = tard
        c.assigned_vhc = getattr(vehicle, "ID", -1)
        c.complete = True

        total_tard += tard
        cur_time = end
        cur_loc = tuple(c.loc)

    return total_tard

def calculate_total_tardiness(vehicles: List['Vehicle']) -> float:
    """모든 차량의 schedules를 심판에 넣어 총 tardiness 반환 (용량 초과 시 inf)."""
    tot = 0.0
    for v in vehicles:
        cost = _calculate_route_cost_and_update_customer_info(v, v.schedules)
        if cost == float("inf"):
            return float("inf")
        tot += cost
    return tot

# ----------------------
# Greedy dispatch (초기해)
# ----------------------
def _best_customer_for_vehicle_greedy(v: Vehicle, unserved_ids: List[int], cust_by_id: dict) -> Tuple[Optional[int], Optional[Tuple]]:
    feasible_ids = [cid for cid in unserved_ids if (float(v.now_capacity or 0.0) + float(cust_by_id[cid].weight or 0.0)) <= (float(v.capacity or 0.0) + 1e-9)]
    if not feasible_ids:
        return None, None

    best_key = None
    best_cid = None
    best_metrics = None
    for cid in feasible_ids:
        c = cust_by_id[cid]
        travel_km = get_dist(tuple(v.now_loc), tuple(c.loc))
        travel_h = travel_km / max(1e-9, float(v.speed or 1.0))
        start = max(float(c.tw[0]), float(v.available or 0.0) + travel_h)
        end = start + float(c.serv_time or 0.0)
        tard = max(0.0, end - float(c.tw[1]))
        primary = tard
        secondary = end
        tie1 = float(c.tw[1])
        tie2 = travel_km
        key = (primary, secondary, tie1, tie2, cid)
        if (best_key is None) or (key < best_key):
            best_key = key
            best_cid = cid
            best_metrics = (travel_km, travel_h, start, end, tard)
    return best_cid, best_metrics

def dispatch_earliest_vehicle_best_customer(instance: Instance) -> Solution:
    """
    Greedy dispatch: earliest available vehicle + best customer by delta_tardiness.
    Returns Solution object (and mutates instance.vehicles[].schedules).
    """
    instance.reset()
    customers: List[Customer] = instance.customers
    vehicles: List[Vehicle] = instance.vehicles

    cust_by_id = {c.ID: c for c in customers}
    unserved_ids = list(cust_by_id.keys())

    total_distance = 0.0
    total_tardiness = 0.0

    while unserved_ids:
        order = sorted(range(len(vehicles)), key=lambda kk: (float(vehicles[kk].available or 0.0), vehicles[kk].ID))
        selected = None
        chosen = None
        metrics = None

        for k in order:
            v = vehicles[k]
            cid, m = _best_customer_for_vehicle_greedy(v, unserved_ids, cust_by_id)
            if cid is not None:
                selected = v
                chosen = cid
                metrics = m
                break

        if chosen is None:
            break

        v = selected
        c = cust_by_id[chosen]
        travel_km, travel_h, start, end, tard = metrics

        c.assigned_vhc = v.ID
        c.start = start
        c.end = end
        c.tardy = tard
        c.complete = True

        v.schedules.append(c)
        v.available = end
        v.now_loc = list(c.loc)
        v.now_capacity = float(v.now_capacity or 0.0) + float(c.weight or 0.0)

        total_distance += travel_km
        total_tardiness += tard
        unserved_ids.remove(chosen)

    final_tard = calculate_total_tardiness(vehicles)
    try:
        sol = Solution("Greedy_Dispatch", instance, obj=final_tard)
    except TypeError:
        class _Sol:
            def __init__(self, alg, inst, obj):
                self.algorithm = alg; self.instance = inst; self.objective = obj
                self.comp_time = "N/A"; self.status = "DONE"
            def __repr__(self):
                return f"Schedule by {self.algorithm} - Objective: {self.objective}"
        sol = _Sol("Greedy_Dispatch", instance, final_tard)

    sol.total_distance = total_distance
    sol.unserved_ids = sorted(list(unserved_ids))
    sol.status = "INFEASIBLE_UNSERVED" if unserved_ids else "DONE"
    return sol

# ----------------------
# VNS / Tabu based improvement (Relocate / Swap / 2-opt)
# ----------------------
def _run_first_relocate_move(vehicles: List[Vehicle], current_total_tardiness: float, tabu_list: Dict[int, int], ils_iter: int, best_so_far_tard: float) -> Tuple[bool, float]:
    vehicle_indices = list(range(len(vehicles)))
    random.shuffle(vehicle_indices)
    TABU_TENURE = 5

    for v_from_idx in vehicle_indices:
        v_from = vehicles[v_from_idx]
        if not v_from.schedules:
            continue
        for cust_idx_seq in range(len(v_from.schedules)):
            if cust_idx_seq >= len(v_from.schedules):
                break
            customer_to_move = v_from.schedules[cust_idx_seq]
            is_tabu = tabu_list.get(customer_to_move.ID, 0) > ils_iter
            v_from.schedules.pop(cust_idx_seq)
            for v_to_idx in vehicle_indices:
                v_to = vehicles[v_to_idx]
                for insert_pos in range(len(v_to.schedules) + 1):
                    v_to.schedules.insert(insert_pos, customer_to_move)
                    new_total_tard = calculate_total_tardiness(vehicles)
                    v_to.schedules.pop(insert_pos)
                    if new_total_tard < current_total_tardiness - 1e-5:
                        is_aspirated = new_total_tard < best_so_far_tard
                        if (not is_tabu) or is_aspirated:
                            v_to.schedules.insert(insert_pos, customer_to_move)
                            tabu_list[customer_to_move.ID] = ils_iter + TABU_TENURE
                            return True, new_total_tard
            v_from.schedules.insert(cust_idx_seq, customer_to_move)
    return False, current_total_tardiness

def _run_first_swap_move(vehicles: List[Vehicle], current_total_tardiness: float, tabu_list: Dict[int, int], ils_iter: int, best_so_far_tard: float) -> Tuple[bool, float]:
    vehicle_indices = list(range(len(vehicles)))
    random.shuffle(vehicle_indices)
    TABU_TENURE = 5
    for v_i_idx in vehicle_indices:
        v_i = vehicles[v_i_idx]
        if not v_i.schedules:
            continue
        for c_i_idx in range(len(v_i.schedules)):
            c_i = v_i.schedules[c_i_idx]
            is_tabu_i = tabu_list.get(c_i.ID, 0) > ils_iter
            for v_j_idx in vehicle_indices:
                v_j = vehicles[v_j_idx]
                if not v_j.schedules:
                    continue
                for c_j_idx in range(len(v_j.schedules)):
                    c_j = v_j.schedules[c_j_idx]
                    if v_i.ID == v_j.ID and (c_i_idx == c_j_idx):
                        continue
                    is_tabu_j = tabu_list.get(c_j.ID, 0) > ils_iter
                    # swap
                    v_i.schedules[c_i_idx] = c_j
                    v_j.schedules[c_j_idx] = c_i
                    new_total_tard = calculate_total_tardiness(vehicles)
                    if new_total_tard < current_total_tardiness - 1e-5:
                        is_aspirated = new_total_tard < best_so_far_tard
                        if (not is_tabu_i and not is_tabu_j) or is_aspirated:
                            tabu_list[c_i.ID] = ils_iter + TABU_TENURE
                            tabu_list[c_j.ID] = ils_iter + TABU_TENURE
                            return True, new_total_tard
                    # rollback
                    v_i.schedules[c_i_idx] = c_i
                    v_j.schedules[c_j_idx] = c_j
    return False, current_total_tardiness

def _run_first_2opt_move(vehicles: List[Vehicle], current_total_tardiness: float, tabu_list: Dict[int, int], ils_iter: int, best_so_far_tard: float) -> Tuple[bool, float]:
    vehicle_indices = list(range(len(vehicles)))
    random.shuffle(vehicle_indices)
    TABU_TENURE = 5
    for v_i_idx in vehicle_indices:
        v_i = vehicles[v_i_idx]
        if len(v_i.schedules) < 2:
            continue
        for v_j_idx in vehicle_indices:
            v_j = vehicles[v_j_idx]
            if len(v_j.schedules) < 2:
                continue
            for i in range(len(v_i.schedules) - 1):
                c_i = v_i.schedules[i]
                c_i_next = v_i.schedules[i+1]
                is_tabu_i = tabu_list.get(c_i.ID, 0) > ils_iter or tabu_list.get(c_i_next.ID, 0) > ils_iter
                for j in range(len(v_j.schedules) - 1):
                    c_j = v_j.schedules[j]
                    c_j_next = v_j.schedules[j+1]
                    is_tabu_j = tabu_list.get(c_j.ID, 0) > ils_iter or tabu_list.get(c_j_next.ID, 0) > ils_iter
                    original_i = list(v_i.schedules)
                    original_j = list(v_j.schedules)
                    if v_i.ID == v_j.ID:
                        if i >= j or (i+1) == j:
                            continue
                        seg = original_i[i+1 : j+1]
                        seg.reverse()
                        v_i.schedules = original_i[:i+1] + seg + original_i[j+1:]
                    else:
                        if (i == (len(original_i) - 2)) and (j == (len(original_j) - 2)):
                            continue
                        v_i.schedules = original_i[:i+1] + original_j[j+1:]
                        v_j.schedules = original_j[:j+1] + original_i[i+1:]
                    new_total_tard = calculate_total_tardiness(vehicles)
                    if new_total_tard < current_total_tardiness - 1e-5:
                        is_aspirated = new_total_tard < best_so_far_tard
                        if (not is_tabu_i and not is_tabu_j) or is_aspirated:
                            tabu_list[c_i.ID] = ils_iter + TABU_TENURE
                            tabu_list[c_i_next.ID] = ils_iter + TABU_TENURE
                            tabu_list[c_j.ID] = ils_iter + TABU_TENURE
                            tabu_list[c_j_next.ID] = ils_iter + TABU_TENURE
                            return True, new_total_tard
                    v_i.schedules = original_i
                    v_j.schedules = original_j
    return False, current_total_tardiness

def run_vns_improvement(solution: 'Solution', time_limit_sec: float) -> float:
    vehicles: List[Vehicle] = solution.instance.vehicles
    start = time.time()
    tabu_list: Dict[int,int] = {}
    current_total = calculate_total_tardiness(vehicles)
    best = current_total
    iteration = 0
    # keep iterating neighborhoods until time limit
    while True:
        if (time.time() - start) > time_limit_sec:
            break
        iteration += 1
        # try relocate
        improved, val = _run_first_relocate_move(vehicles, current_total, tabu_list, iteration, best)
        if improved:
            current_total = val
            if current_total < best:
                best = current_total
            continue
        # try swap
        improved, val = _run_first_swap_move(vehicles, current_total, tabu_list, iteration, best)
        if improved:
            current_total = val
            if current_total < best:
                best = current_total
            continue
        # try 2-opt
        improved, val = _run_first_2opt_move(vehicles, current_total, tabu_list, iteration, best)
        if improved:
            current_total = val
            if current_total < best:
                best = current_total
            continue
        # no improvement in any neighborhood -> local optimum
        break
    return best

# ----------------------
# "안전한" 교란 (perturb)
# ----------------------
def perturb_solution(solution: 'Solution', strength: int = 3):
    instance: 'Instance' = solution.instance
    vehicles: List['Vehicle'] = instance.vehicles
    movable = [v for v in vehicles if v.schedules]
    if not movable:
        return
    for _ in range(strength):
        v_from = random.choice(movable)
        if not v_from.schedules:
            continue
        ci = random.randrange(len(v_from.schedules))
        cust = v_from.schedules.pop(ci)
        v_to = random.choice(vehicles)
        pos = random.randint(0, len(v_to.schedules))
        v_to.schedules.insert(pos, cust)

# ----------------------
# LNS repair helper (used externally if desired)
# ----------------------
def greedy_repair_insert(vehicles: List[Vehicle], customers: List[Customer]):
    """
    주어진 customers (unplaced) 를 greedy로 각 차량 최적 위치에 삽입 (심플)
    """
    remaining = list(customers)
    # simple greedy: one by one insert at best (vehicle,pos)
    while remaining:
        best_gain = float("inf")
        best_choice = None
        for c in remaining:
            for v in vehicles:
                for pos in range(len(v.schedules) + 1):
                    v.schedules.insert(pos, c)
                    cost = calculate_total_tardiness(vehicles)
                    v.schedules.pop(pos)
                    if cost < best_gain:
                        best_gain = cost
                        best_choice = (c, v, pos)
        if best_choice is None:
            break
        c, v, pos = best_choice
        v.schedules.insert(pos, c)
        remaining.remove(c)
