from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import random
import time
import math

import module
from module import Customer, Vehicle, Instance, Solution, get_dist

# =========================
# Improved dispatching + ILS/LNS/VNS modules
# Goal: better initial solution (regret-k), LNS-based perturbation/repair,
# adaptive acceptance (simulated annealing style), and strengthened VNS moves.
# These changes are designed to increase improvement over greedy baseline.
# =========================

# -------------------------
# 1) Stateless route cost (same as original, kept for safety)
# -------------------------
def _calculate_route_cost_and_update_customer_info(vehicle: 'Vehicle', route: List['Customer']) -> float:
    depot_loc = tuple(vehicle.loc)
    speed = float(vehicle.speed or 30.0)
    capacity = float(vehicle.capacity or float('inf'))

    current_time = 0.0
    current_loc = depot_loc
    total_tardiness = 0.0
    current_load = 0.0

    for customer in route:
        current_load += float(customer.weight or 0.0)
        if current_load > capacity + 1e-9:
            return float('inf')

        cust_loc = tuple(customer.loc)
        travel_km = get_dist(current_loc, cust_loc)
        travel_h = travel_km / max(1e-9, speed)
        arrival_time = current_time + travel_h

        ready_time = float(customer.tw[0])
        start_time = max(ready_time, arrival_time)

        service_time = float(customer.serv_time or 0.0)
        end_time = start_time + service_time

        due_time = float(customer.tw[1])
        tardiness = max(0.0, end_time - due_time)

        # update for visualization
        customer.start = start_time
        customer.end = end_time
        customer.tardy = tardiness

        total_tardiness += tardiness
        current_time = end_time
        current_loc = cust_loc

    return total_tardiness


def calculate_total_tardiness(vehicles: List['Vehicle']) -> float:
    total_cost = 0.0
    for v in vehicles:
        cost = _calculate_route_cost_and_update_customer_info(v, v.schedules)
        if cost == float('inf'):
            return float('inf')
        total_cost += cost
    return total_cost


# -------------------------
# 2) Improved initial solution: Regret-k insertion (k=2 or 3)
# -------------------------

def dispatch_regret_k(instance: Instance, k: int = 2) -> Solution:
    """
    Regret-k insertion: iteratively insert the customer whose difference between
    best and k-th best insertion costs is largest. Cost measured by added tardiness
    (using the stateless judge) and travel distance as tie-breaker.
    """
    instance.reset()
    customers: List[Customer] = instance.customers
    vehicles: List[Vehicle] = instance.vehicles
    cust_by_id = {c.ID: c for c in customers}
    unserved_ids = set(cust_by_id.keys())

    # initialize empty schedules
    for v in vehicles:
        v.schedules = []
        v.now_loc = list(v.loc)
        v.now_capacity = 0.0
        v.available = 0.0

    total_distance = 0.0

    # helper: compute insertion cost of placing customer c into vehicle v at pos p
    def insertion_cost(v: Vehicle, customer: Customer, pos: int) -> Tuple[float, float]:
        orig = list(v.schedules)
        orig_len = len(orig)
        # insert and evaluate
        v.schedules = orig[:pos] + [customer] + orig[pos:]
        cost = _calculate_route_cost_and_update_customer_info(v, v.schedules)
        # compute incremental travel distance approx: distance from prev to customer + customer to next - prev to next
        prev_loc = tuple(v.loc) if pos == 0 else tuple(orig[pos-1].loc)
        next_loc = tuple(orig[pos].loc) if pos < orig_len else tuple(v.loc)
        dist_inc = get_dist(prev_loc, tuple(customer.loc)) + get_dist(tuple(customer.loc), next_loc) - get_dist(prev_loc, next_loc)
        v.schedules = orig
        return cost, dist_inc

    # main loop: insert until none left or infeasible
    while unserved_ids:
        # for each customer, compute up to k best insertion options (vehicle,pos)
        regret_candidates = []  # (regret_value, best_cost, best_option, cid)

        for cid in list(unserved_ids):
            c = cust_by_id[cid]
            insertion_options = []  # (cost_after_insert, dist_inc, v_idx, pos)
            for v_idx, v in enumerate(vehicles):
                # prune by capacity
                if float(v.now_capacity or 0.0) + float(c.weight or 0.0) > float(v.capacity or float('inf')) + 1e-9:
                    continue
                for pos in range(len(v.schedules) + 1):
                    cost, dist_inc = insertion_cost(v, c, pos)
                    if cost != float('inf'):
                        insertion_options.append((cost, dist_inc, v_idx, pos))

            if not insertion_options:
                continue

            insertion_options.sort(key=lambda x: (x[0], x[1]))
            best = insertion_options[0]
            kth = insertion_options[min(k-1, len(insertion_options)-1)]
            regret = kth[0] - best[0]
            regret_candidates.append((regret, best[0], best, cid))

        if not regret_candidates:
            break

        # choose customer with largest regret (break ties by worst best cost)
        regret_candidates.sort(key=lambda x: (-x[0], -x[1]))
        chosen = regret_candidates[0]
        _, _, best_option, chosen_cid = chosen
        best_cost, best_dist, v_idx, pos = best_option

        v = vehicles[v_idx]
        c = cust_by_id[chosen_cid]

        # commit insertion
        v.schedules.insert(pos, c)
        v.now_capacity = float(v.now_capacity or 0.0) + float(c.weight or 0.0)
        # update v.available and v.now_loc via judge
        _calculate_route_cost_and_update_customer_info(v, v.schedules)
        if v.schedules:
            last = v.schedules[-1]
            v.now_loc = list(last.loc)
            v.available = float(last.end or 0.0)

        unserved_ids.remove(chosen_cid)

    final_tard = calculate_total_tardiness(vehicles)
    try:
        sol = Solution("RegretK_Dispatch", instance, obj=final_tard)
    except TypeError:
        class _Sol:
            def __init__(self, alg, inst, obj):
                self.algorithm = alg; self.instance = inst; self.objective = obj
                self.comp_time = "N/A"; self.status = "DONE"
            def __repr__(self):
                return f"Schedule by {self.algorithm} - Objective: {self.objective}"
        sol = _Sol("RegretK_Dispatch", instance, final_tard)

    sol.total_distance = total_distance
    sol.unserved_ids = sorted(list(unserved_ids))
    sol.status = "INFEASIBLE_UNSERVED" if unserved_ids else "DONE"
    return sol


# -------------------------
# 3) LNS destroy & repair (stronger, but safe)
# -------------------------

def lns_destroy_repair(solution: Solution, remove_count: int = 4, time_limit: float = 0.05):
    """
    Remove 'remove_count' customers (preferentially high-tardiness or random) and reinsert
    them using best insertion (like a repair). This is deterministic and avoids lost customers.
    """
    instance: Instance = solution.instance
    vehicles: List[Vehicle] = instance.vehicles
    all_customers = [c for v in vehicles for c in v.schedules]
    if not all_customers:
        return

    # choose removals: combine worst tardy and some random
    sorted_by_tardy = sorted(all_customers, key=lambda c: float(getattr(c, 'tardy', 0.0)), reverse=True)
    removals = set()
    i = 0
    while len(removals) < remove_count and i < len(sorted_by_tardy):
        removals.add(sorted_by_tardy[i].ID)
        i += 1
    # add random if still short
    all_ids = [c.ID for c in all_customers]
    while len(removals) < remove_count and all_ids:
        removals.add(random.choice(all_ids))

    removed_customers = []
    for v in vehicles:
        new_sched = []
        for c in v.schedules:
            if c.ID in removals:
                removed_customers.append(c)
            else:
                new_sched.append(c)
        v.schedules = new_sched

    # repair: greedy best insertion (min added tardiness, tie by distance)
    for c in removed_customers:
        best = None
        for v_idx, v in enumerate(vehicles):
            # capacity check
            if float(v.now_capacity or 0.0) + float(c.weight or 0.0) > float(v.capacity or float('inf')) + 1e-9:
                continue
            for pos in range(len(v.schedules) + 1):
                orig = list(v.schedules)
                v.schedules = orig[:pos] + [c] + orig[pos:]
                cost = calculate_total_tardiness(vehicles)
                # revert
                v.schedules = orig
                if cost == float('inf'):
                    continue
                if best is None or cost < best[0] or (abs(cost - best[0]) < 1e-9 and get_dist(tuple(v.loc), tuple(c.loc)) < best[1]):
                    best = (cost, get_dist(tuple(v.loc), tuple(c.loc)), v_idx, pos)
        if best is None:
            # can't insert feasibly -> put back to original vehicle end (safe fallback)
            # find original vehicle by ID
            for v in vehicles:
                v.schedules.append(c)
        else:
            _, _, v_idx, pos = best
            v = vehicles[v_idx]
            v.schedules.insert(pos, c)

    # update capacities/locations by running judge per vehicle
    for v in vehicles:
        _calculate_route_cost_and_update_customer_info(v, v.schedules)
        if v.schedules:
            last = v.schedules[-1]
            v.now_loc = list(last.loc)
            v.available = float(last.end or 0.0)
            v.now_capacity = sum(float(cc.weight or 0.0) for cc in v.schedules)
        else:
            v.now_loc = list(v.loc)
            v.available = 0.0
            v.now_capacity = 0.0


# -------------------------
# 4) Strengthened perturbation (uses LNS)
# -------------------------

def perturb_solution_strong(solution: Solution, strength: int = 3):
    # strength controls how many customers removed in LNS
    remove_k = max(2, strength)
    lns_destroy_repair(solution, remove_count=remove_k)


# -------------------------
# 5) VNS-based improvement (keeps Relocate/Swap/2-opt but tries both first- and best-improve with acceptance)
# -------------------------

def _run_relocate_best(vehicles: List[Vehicle], current_total_tardiness: float,
                       tabu_list: Dict[int, int], ils_iter: int, best_so_far_tard: float) -> Tuple[bool, float]:
    # try to find best relocate move across all pairs (not just first-improve) within small budget
    best_improvement = 0.0
    best_move = None
    TABU_TENURE = 5

    vehicle_indices = list(range(len(vehicles)))
    for v_from_idx in vehicle_indices:
        v_from = vehicles[v_from_idx]
        for cust_idx_seq in range(len(v_from.schedules)):
            customer_to_move = v_from.schedules[cust_idx_seq]
            is_tabu = tabu_list.get(customer_to_move.ID, 0) > ils_iter
            # temporarily remove
            v_from.schedules.pop(cust_idx_seq)
            for v_to_idx in vehicle_indices:
                v_to = vehicles[v_to_idx]
                for insert_pos in range(len(v_to.schedules) + 1):
                    v_to.schedules.insert(insert_pos, customer_to_move)
                    new_total_tard = calculate_total_tardiness(vehicles)
                    v_to.schedules.pop(insert_pos)
                    if new_total_tard < current_total_tardiness - 1e-9:
                        improvement = current_total_tardiness - new_total_tard
                        is_asp = new_total_tard < best_so_far_tard
                        if (not is_tabu) or is_asp:
                            if improvement > best_improvement:
                                best_improvement = improvement
                                best_move = (v_from_idx, cust_idx_seq, v_to_idx, insert_pos, new_total_tard, customer_to_move.ID, is_tabu)
            # restore
            v_from.schedules.insert(cust_idx_seq, customer_to_move)

    if best_move:
        v_from_idx, cust_idx_seq, v_to_idx, insert_pos, new_total_tard, cid, was_tabu = best_move
        v_from = vehicles[v_from_idx]
        v_to = vehicles[v_to_idx]
        customer_to_move = v_from.schedules.pop(cust_idx_seq)
        v_to.schedules.insert(insert_pos, customer_to_move)
        tabu_list[cid] = ils_iter + TABU_TENURE
        return True, new_total_tard
    return False, current_total_tardiness


def _run_swap_best(vehicles: List[Vehicle], current_total_tardiness: float,
                   tabu_list: Dict[int, int], ils_iter: int, best_so_far_tard: float) -> Tuple[bool, float]:
    best_improvement = 0.0
    best_move = None
    TABU_TENURE = 5
    vehicle_indices = list(range(len(vehicles)))

    for v_i_idx in vehicle_indices:
        v_i = vehicles[v_i_idx]
        for c_i_idx in range(len(v_i.schedules)):
            c_i = v_i.schedules[c_i_idx]
            is_tabu_i = tabu_list.get(c_i.ID, 0) > ils_iter
            for v_j_idx in vehicle_indices:
                v_j = vehicles[v_j_idx]
                for c_j_idx in range(len(v_j.schedules)):
                    c_j = v_j.schedules[c_j_idx]
                    if v_i_idx == v_j_idx and c_i_idx == c_j_idx:
                        continue
                    # swap
                    v_i.schedules[c_i_idx], v_j.schedules[c_j_idx] = c_j, c_i
                    new_total_tard = calculate_total_tardiness(vehicles)
                    # revert
                    v_i.schedules[c_i_idx], v_j.schedules[c_j_idx] = c_i, c_j
                    if new_total_tard < current_total_tardiness - 1e-9:
                        is_asp = new_total_tard < best_so_far_tard
                        if (not is_tabu_i) or is_asp:
                            improvement = current_total_tardiness - new_total_tard
                            if improvement > best_improvement:
                                best_improvement = improvement
                                best_move = (v_i_idx, c_i_idx, v_j_idx, c_j_idx, new_total_tard, c_i.ID, c_j.ID)

    if best_move:
        v_i_idx, c_i_idx, v_j_idx, c_j_idx, new_total_tard, cid_i, cid_j = best_move
        v_i = vehicles[v_i_idx]
        v_j = vehicles[v_j_idx]
        v_i.schedules[c_i_idx], v_j.schedules[c_j_idx] = v_j.schedules[c_j_idx], v_i.schedules[c_i_idx]
        tabu_list[cid_i] = ils_iter + TABU_TENURE
        tabu_list[cid_j] = ils_iter + TABU_TENURE
        return True, new_total_tard
    return False, current_total_tardiness


def _run_2opt_best(vehicles: List[Vehicle], current_total_tardiness: float,
                   tabu_list: Dict[int, int], ils_iter: int, best_so_far_tard: float) -> Tuple[bool, float]:
    best_improvement = 0.0
    best_move = None
    TABU_TENURE = 5
    vehicle_indices = list(range(len(vehicles)))

    for v_i_idx in vehicle_indices:
        v_i = vehicles[v_i_idx]
        if len(v_i.schedules) < 2:
            continue
        for v_j_idx in vehicle_indices:
            v_j = vehicles[v_j_idx]
            if len(v_j.schedules) < 2:
                continue
            for i in range(len(v_i.schedules) - 1):
                for j in range(len(v_j.schedules) - 1):
                    # save
                    ori_i = list(v_i.schedules)
                    ori_j = list(v_j.schedules)
                    if v_i_idx == v_j_idx:
                        if i >= j:
                            continue
                        segment = ori_i[i+1:j+1]
                        segment.reverse()
                        v_i.schedules = ori_i[:i+1] + segment + ori_i[j+1:]
                    else:
                        v_i.schedules = ori_i[:i+1] + ori_j[j+1:]
                        v_j.schedules = ori_j[:j+1] + ori_i[i+1:]

                    new_total_tard = calculate_total_tardiness(vehicles)
                    # restore
                    v_i.schedules = ori_i
                    v_j.schedules = ori_j
                    if new_total_tard < current_total_tardiness - 1e-9:
                        is_asp = new_total_tard < best_so_far_tard
                        if is_asp or (tabu_list.get(ori_i[i].ID,0) <= ils_iter and tabu_list.get(ori_j[j].ID,0) <= ils_iter):
                            improvement = current_total_tardiness - new_total_tard
                            if improvement > best_improvement:
                                best_improvement = improvement
                                best_move = (v_i_idx, v_j_idx, i, j, new_total_tard, ori_i[i].ID, ori_j[j].ID)

    if best_move:
        v_i_idx, v_j_idx, i, j, new_total_tard, id_i, id_j = best_move
        v_i = vehicles[v_i_idx]
        v_j = vehicles[v_j_idx]
        ori_i = list(v_i.schedules)
        ori_j = list(v_j.schedules)
        if v_i_idx == v_j_idx:
            segment = ori_i[i+1:j+1]
            segment.reverse()
            v_i.schedules = ori_i[:i+1] + segment + ori_i[j+1:]
        else:
            v_i.schedules = ori_i[:i+1] + ori_j[j+1:]
            v_j.schedules = ori_j[:j+1] + ori_i[i+1:]
        tabu_list[id_i] = ils_iter + TABU_TENURE
        tabu_list[id_j] = ils_iter + TABU_TENURE
        return True, new_total_tard
    return False, current_total_tardiness


def run_vns_improvement(solution: 'Solution', time_limit_sec: float, tabu_list: Optional[Dict[int,int]] = None) -> float:
    vehicles: List['Vehicle'] = solution.instance.vehicles
    start_vns_time = time.time()
    if tabu_list is None:
        tabu_list = {}

    current_total_tardiness = calculate_total_tardiness(vehicles)
    best_so_far_tard = current_total_tardiness
    iteration = 0

    while True:
        elapsed = time.time() - start_vns_time
        if elapsed > time_limit_sec:
            break
        iteration += 1
        # try best-relocate
        improved, new_tard = _run_relocate_best(vehicles, current_total_tardiness, tabu_list, iteration, best_so_far_tard)
        if improved:
            current_total_tardiness = new_tard
            if current_total_tardiness < best_so_far_tard:
                best_so_far_tard = current_total_tardiness
            continue
        # swap
        improved, new_tard = _run_swap_best(vehicles, current_total_tardiness, tabu_list, iteration, best_so_far_tard)
        if improved:
            current_total_tardiness = new_tard
            if current_total_tardiness < best_so_far_tard:
                best_so_far_tard = current_total_tardiness
            continue
        # 2-opt
        improved, new_tard = _run_2opt_best(vehicles, current_total_tardiness, tabu_list, iteration, best_so_far_tard)
        if improved:
            current_total_tardiness = new_tard
            if current_total_tardiness < best_so_far_tard:
                best_so_far_tard = current_total_tardiness
            continue
        break

    return best_so_far_tard


# -------------------------
# 6) ILS main with SA acceptance and LNS perturb + adaptive tabu
# -------------------------

def improve_with_ils(solution: 'Solution', time_limit_sec: float) -> float:
    instance: Instance = solution.instance
    vehicles: List[Vehicle] = instance.vehicles
    start = time.time()

    # initialize tabu list
    tabu_list: Dict[int,int] = {}

    # initial local search: run VNS for short time
    vns_init_time = min(0.12, time_limit_sec * 0.4)
    best_tardiness = run_vns_improvement(solution, vns_init_time, tabu_list)
    best_schedules = {v.ID: list(v.schedules) for v in vehicles}

    # temperature schedule for SA acceptance
    T0 = max(1.0, best_tardiness * 0.1)
    end_time = start + time_limit_sec
    iteration = 0

    while time.time() < end_time:
        iteration += 1
        # perturbation: LNS
        remaining = end_time - time.time()
        strength = 2 + (iteration % 4)  # vary strength
        perturb_solution_strong(solution, strength=strength)

        # local search on perturbed solution
        time_budget = min(0.08, max(0.02, remaining * 0.15))
        current_tard = run_vns_improvement(solution, time_budget, tabu_list)

        # acceptance: simulated annealing style
        elapsed = time.time() - start
        frac = min(1.0, elapsed / max(1e-9, time_limit_sec))
        T = T0 * (1.0 - frac)
        delta = current_tard - best_tardiness
        accept = False
        if current_tard < best_tardiness:
            accept = True
        else:
            if T > 1e-9:
                prob = math.exp(-delta / T) if delta > 0 else 1.0
                if random.random() < prob:
                    accept = True

        if accept:
            # accept current as new incumbent (may be worse occasionally)
            if current_tard < best_tardiness:
                best_tardiness = current_tard
                best_schedules = {v.ID: list(v.schedules) for v in vehicles}
        else:
            # revert to best
            for v in vehicles:
                v.schedules = list(best_schedules[v.ID])

    # final restore
    for v in vehicles:
        v.schedules = list(best_schedules[v.ID])

    solution.objective = best_tardiness
    return best_tardiness


# -------------------------
# 7) Compatibility wrappers so main.py can import these functions by name
# -------------------------

def dispatch_earliest_vehicle_best_customer(instance: Instance) -> Solution:
    # keep fallback: use regret-k (k=2) as improved dispatch
    return dispatch_regret_k(instance, k=2)


def run_vns_improvement_wrapper(solution: Solution, time_limit_sec: float) -> float:
    return run_vns_improvement(solution, time_limit_sec)


def perturb_solution(solution: Solution, strength: int = 3):
    # expose stronger perturbation by default
    return perturb_solution_strong(solution, strength)
