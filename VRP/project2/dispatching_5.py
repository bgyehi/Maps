# dispatching_5_improved.py
from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any
import random
import time
import math
import copy
import heapq

import module
from module import Customer, Vehicle, Instance, Solution, get_dist

# -------------------------
# Random seed for reproducibility (optional)
# -------------------------
RANDOM_SEED = None  # set to int for deterministic behaviour
if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)

# -------------------------
# 0) Helper utilities
# -------------------------
def deep_copy_solution(sol: Solution) -> Solution:
    # deep copy instance vehicles' schedules, but keep same Instance reference
    inst = sol.instance
    # We assume Solution holds schedule in vehicles inside instance, so clone vehicles' schedules
    for v in inst.vehicles:
        v.schedules = list(getattr(v, "schedules", []))
    new_sol = Solution(sol.algorithm + "_copy", inst, obj=getattr(sol, "objective", None))
    new_sol.routes = [list(v.schedules) for v in inst.vehicles]
    return new_sol

def clone_instance_with_empty_routes(instance: Instance) -> Instance:
    # shallow clone instance (avoid full deep copy of large object); reset vehicle schedules
    # This helper is used to build new solutions without modifying the original instance's route lists
    inst = instance
    for v in inst.vehicles:
        v.schedules = []
        v.now_loc = list(v.loc)
        v.now_capacity = 0.0
        v.available = 0.0
    return inst

def total_customers_in_vehicles(vehicles: List[Vehicle]) -> int:
    return sum(len(v.schedules) for v in vehicles)

# -------------------------
# 1) Route cost evaluator & customer updates (kept from original)
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

        # update for visualization & downstream heuristics
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
# 2) Regret-k insertion (improved initial solution)
# -------------------------
def dispatch_regret_k(instance: Instance, k: int = 2) -> Solution:
    """
    Regret-k insertion producing a good initial feasible solution.
    """
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    instance.reset()
    customers: List[Customer] = instance.customers
    vehicles: List[Vehicle] = instance.vehicles
    cust_by_id = {c.ID: c for c in customers}
    unserved_ids = set(cust_by_id.keys())

    # initialize vehicles
    for v in vehicles:
        v.schedules = []
        v.now_loc = list(v.loc)
        v.now_capacity = 0.0
        v.available = 0.0

    total_distance = 0.0

    def insertion_cost(v: Vehicle, customer: Customer, pos: int) -> Tuple[float, float]:
        orig = list(v.schedules)
        orig_len = len(orig)
        v.schedules = orig[:pos] + [customer] + orig[pos:]
        cost = _calculate_route_cost_and_update_customer_info(v, v.schedules)
        prev_loc = tuple(v.loc) if pos == 0 else tuple(orig[pos-1].loc)
        next_loc = tuple(orig[pos].loc) if pos < orig_len else tuple(v.loc)
        dist_inc = get_dist(prev_loc, tuple(customer.loc)) + get_dist(tuple(customer.loc), next_loc) - get_dist(prev_loc, next_loc)
        v.schedules = orig
        return cost, dist_inc

    while unserved_ids:
        regret_candidates = []
        for cid in list(unserved_ids):
            c = cust_by_id[cid]
            insertion_options = []
            for v_idx, v in enumerate(vehicles):
                # capacity prune
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

        regret_candidates.sort(key=lambda x: (-x[0], -x[1]))
        _, _, best_option, chosen_cid = regret_candidates[0]
        best_cost, best_dist, v_idx, pos = best_option

        v = vehicles[v_idx]
        c = cust_by_id[chosen_cid]
        v.schedules.insert(pos, c)
        v.now_capacity = float(v.now_capacity or 0.0) + float(c.weight or 0.0)
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

    sol.total_distance = 0.0
    sol.unserved_ids = sorted(list(unserved_ids))
    sol.status = "INFEASIBLE_UNSERVED" if unserved_ids else "DONE"
    return sol

# -------------------------
# 3) LNS destroy & repair
# -------------------------
def lns_destroy_repair(solution: Solution, remove_count: int = 4, time_limit: float = 0.05):
    """
    Remove 'remove_count' customers (preferentially high-tardiness) and reinsert using best insertion.
    """
    instance: Instance = solution.instance
    vehicles: List[Vehicle] = instance.vehicles
    all_customers = [c for v in vehicles for c in v.schedules]
    if not all_customers:
        return

    sorted_by_tardy = sorted(all_customers, key=lambda c: float(getattr(c, 'tardy', 0.0)), reverse=True)
    removals = set()
    i = 0
    while len(removals) < remove_count and i < len(sorted_by_tardy):
        removals.add(sorted_by_tardy[i].ID)
        i += 1
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

    # repair: best insertion (min total tardiness)
    for c in removed_customers:
        best = None
        for v_idx, v in enumerate(vehicles):
            if float(v.now_capacity or 0.0) + float(c.weight or 0.0) > float(v.capacity or float('inf')) + 1e-9:
                continue
            for pos in range(len(v.schedules) + 1):
                orig = list(v.schedules)
                v.schedules = orig[:pos] + [c] + orig[pos:]
                cost = calculate_total_tardiness(vehicles)
                v.schedules = orig
                if cost == float('inf'):
                    continue
                dist_metric = get_dist(tuple(v.loc), tuple(c.loc))
                if best is None or cost < best[0] or (abs(cost - best[0]) < 1e-9 and dist_metric < best[1]):
                    best = (cost, dist_metric, v_idx, pos)
        if best is None:
            # fallback: append to vehicle with most remaining capacity (safe)
            best_v = max(vehicles, key=lambda vv: float(vv.capacity or 0.0) - float(sum(getattr(cc, 'weight', 0.0) for cc in vv.schedules)))
            best_v.schedules.append(c)
        else:
            _, _, v_idx, pos = best
            vehicles[v_idx].schedules.insert(pos, c)

    # update vehicles info
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
# 4) perturbation wrapper
# -------------------------
def perturb_solution_strong(solution: Solution, strength: int = 3):
    remove_k = max(2, strength)
    lns_destroy_repair(solution, remove_count=remove_k)

# -------------------------
# 5) VNS moves (best-improve variants)
# -------------------------
def _run_relocate_best(vehicles: List[Vehicle], current_total_tardiness: float,
                       tabu_list: Dict[int, int], ils_iter: int, best_so_far_tard: float) -> Tuple[bool, float]:
    best_improvement = 0.0
    best_move = None
    TABU_TENURE = 5
    vehicle_indices = list(range(len(vehicles)))

    for v_from_idx in vehicle_indices:
        v_from = vehicles[v_from_idx]
        for cust_idx_seq in range(len(v_from.schedules)):
            customer_to_move = v_from.schedules[cust_idx_seq]
            is_tabu = tabu_list.get(customer_to_move.ID, 0) > ils_iter
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
                    v_i.schedules[c_i_idx], v_j.schedules[c_j_idx] = c_j, c_i
                    new_total_tard = calculate_total_tardiness(vehicles)
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

# -------------------------
# 6) VNS main
# -------------------------
def run_vns_improvement(solution: 'Solution', time_limit_sec: float, tabu_list: Optional[Dict[int,int]] = None) -> float:
    vehicles: List[Vehicle] = solution.instance.vehicles
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
        improved, new_tard = _run_relocate_best(vehicles, current_total_tardiness, tabu_list, iteration, best_so_far_tard)
        if improved:
            current_total_tardiness = new_tard
            best_so_far_tard = min(best_so_far_tard, current_total_tardiness)
            continue
        improved, new_tard = _run_swap_best(vehicles, current_total_tardiness, tabu_list, iteration, best_so_far_tard)
        if improved:
            current_total_tardiness = new_tard
            best_so_far_tard = min(best_so_far_tard, current_total_tardiness)
            continue
        improved, new_tard = _run_2opt_best(vehicles, current_total_tardiness, tabu_list, iteration, best_so_far_tard)
        if improved:
            current_total_tardiness = new_tard
            best_so_far_tard = min(best_so_far_tard, current_total_tardiness)
            continue
        break

    return best_so_far_tard

# -------------------------
# 7) ILS with SA acceptance and LNS perturbation
# -------------------------
def improve_with_ils(solution: 'Solution', time_limit_sec: float) -> float:
    instance: Instance = solution.instance
    vehicles: List[Vehicle] = instance.vehicles
    start = time.time()

    tabu_list: Dict[int,int] = {}
    vns_init_time = min(0.12, time_limit_sec * 0.3)
    best_tardiness = run_vns_improvement(solution, vns_init_time, tabu_list)
    best_schedules = {v.ID: list(v.schedules) for v in vehicles}

    T0 = max(1.0, best_tardiness * 0.1)
    end_time = start + time_limit_sec
    iteration = 0

    while time.time() < end_time:
        iteration += 1
        remaining = end_time - time.time()
        strength = 2 + (iteration % 4)
        perturb_solution_strong(solution, strength=strength)

        time_budget = min(0.08, max(0.02, remaining * 0.15))
        current_tard = run_vns_improvement(solution, time_budget, tabu_list)

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
            if current_tard < best_tardiness:
                best_tardiness = current_tard
                best_schedules = {v.ID: list(v.schedules) for v in vehicles}
        else:
            for v in vehicles:
                v.schedules = list(best_schedules[v.ID])

    for v in vehicles:
        v.schedules = list(best_schedules[v.ID])

    solution.objective = best_tardiness
    return best_tardiness

# -------------------------
# 8) Simple Genetic Algorithm (route-based crossover + LNS mutation)
# -------------------------
def solution_fitness(solution: Solution) -> float:
    # smaller is better (tardiness)
    val = getattr(solution, "objective", None)
    if val is None:
        # evaluate if not set
        val = calculate_total_tardiness(solution.instance.vehicles)
        solution.objective = val
    return val

def encode_solution_routes(solution: Solution) -> List[List[int]]:
    # encode as list of vehicle route customer IDs
    return [[c.ID for c in v.schedules] for v in solution.instance.vehicles]

def decode_and_apply_routes(instance: Instance, routes: List[List[int]]):
    id_to_customer = {c.ID: c for c in instance.customers}
    for v_idx, v in enumerate(instance.vehicles):
        v.schedules = [ id_to_customer[cid] for cid in routes[v_idx] if cid in id_to_customer ]

def route_based_crossover(parentA: Solution, parentB: Solution) -> Solution:
    # create child by choosing for each vehicle either parent's route or mixing
    instance = parentA.instance
    child_inst = instance
    # start from empty
    for v in child_inst.vehicles:
        v.schedules = []
    routesA = encode_solution_routes(parentA)
    routesB = encode_solution_routes(parentB)
    used = set()
    for v_idx in range(len(child_inst.vehicles)):
        pick_from_A = random.random() < 0.5
        chosen_route = routesA[v_idx] if pick_from_A else routesB[v_idx]
        # append only customers not yet used
        chosen_filtered = [cid for cid in chosen_route if cid not in used]
        # simple greedy fill with remaining if route empty
        child_inst.vehicles[v_idx].schedules = [next((c for c in child_inst.customers if c.ID==cid), None) for cid in chosen_filtered]
        used.update(chosen_filtered)
    # append remaining customers greedily (by nearest vehicle)
    remaining = [c for c in child_inst.customers if c.ID not in used]
    for c in remaining:
        best_v_idx, best_pos, best_cost = None, None, float('inf')
        for v_idx, v in enumerate(child_inst.vehicles):
            for pos in range(len(v.schedules)+1):
                orig = list(v.schedules)
                v.schedules = orig[:pos] + [c] + orig[pos:]
                cost = calculate_total_tardiness(child_inst.vehicles)
                v.schedules = orig
                if cost < best_cost:
                    best_cost = cost; best_v_idx = v_idx; best_pos = pos
        if best_v_idx is None:
            # fallback append first vehicle
            child_inst.vehicles[0].schedules.append(c)
        else:
            child_inst.vehicles[best_v_idx].schedules.insert(best_pos, c)
    child_sol = Solution("GA_child", child_inst, obj=calculate_total_tardiness(child_inst.vehicles))
    return child_sol

def mutate_solution_via_lns(sol: Solution, strength: int = 3):
    perturb_solution_strong(sol, strength=strength)

def run_ga_population(instance: Instance, pop_size: int = 10, generations: int = 10, per_ind_time: float = 0.05) -> Solution:
    """
    Run simple GA: build initial population from regret-k variants + random shuffles,
    then iterate: selection (tournament), crossover (route-based), mutate (LNS), evaluate.
    Returns best solution found.
    """
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    # initialize population
    population: List[Solution] = []
    # seed with a few regret-k solutions with different k
    for k in [2,3]:
        inst_copy = instance
        sol = dispatch_regret_k(inst_copy, k=k)
        population.append(sol)
    # fill with randomized inserts
    while len(population) < pop_size:
        inst_copy = instance
        sol = dispatch_regret_k(inst_copy, k=2)
        # random shuffle small parts
        for v in sol.instance.vehicles:
            random.shuffle(v.schedules)
        sol.objective = calculate_total_tardiness(sol.instance.vehicles)
        population.append(sol)

    # evolve
    for gen in range(generations):
        # tournament selection pairs
        new_pop: List[Solution] = []
        while len(new_pop) < pop_size:
            a,b = random.sample(population, 2)
            child = route_based_crossover(a,b)
            # mutate
            mutate_solution_via_lns(child, strength=2 + (gen%3))
            child.objective = calculate_total_tardiness(child.instance.vehicles)
            new_pop.append(child)
        # merge and keep best pop_size
        population += new_pop
        population.sort(key=solution_fitness)
        population = population[:pop_size]
    best = min(population, key=solution_fitness)
    return best

# -------------------------
# 9) High-level I/O wrappers for compatibility
# -------------------------
def dispatch_earliest_vehicle_best_customer(instance: Instance) -> Solution:
    # fallback: improved regret-k (k=2)
    return dispatch_regret_k(instance, k=2)

def run_vns_improvement_wrapper(solution: Solution, time_limit_sec: float) -> float:
    return run_vns_improvement(solution, time_limit_sec)

def perturb_solution(solution: Solution, strength: int = 3):
    return perturb_solution_strong(solution, strength)

# Expose run_ga_population for optional global search
__all__ = [
    "dispatch_earliest_vehicle_best_customer",
    "run_vns_improvement",
    "perturb_solution",
    "improve_with_ils",
    "run_ga_population"
]
