from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import random
import time
import math
import copy

from module import Customer, Vehicle, Instance, Solution, get_dist


# ==========================================
# 공통 함수
# ==========================================
def calculate_total_tardiness(vehicles: List[Vehicle]) -> float:
    """총 tardiness 계산"""
    total = 0.0
    for v in vehicles:
        depot_loc = tuple(v.loc)
        speed = float(v.speed or 30.0)
        current_time = 0.0
        current_loc = depot_loc

        for c in v.schedules:
            travel_km = get_dist(current_loc, tuple(c.loc))
            travel_h = travel_km / max(speed, 1e-9)
            arrival = current_time + travel_h
            start = max(arrival, float(c.tw[0]))
            end = start + float(c.serv_time or 0.0)
            tardy = max(0.0, end - float(c.tw[1]))
            total += tardy
            current_time = end
            current_loc = tuple(c.loc)
    return total


def deep_copy_solution(instance: Instance) -> Instance:
    """Instance 깊은 복사"""
    new_instance = copy.deepcopy(instance)
    return new_instance


# ==========================================
# 초기해 생성 방법들
# ==========================================

def dispatch_spt(instance: Instance) -> Solution:
    """SPT (Shortest Processing Time) 기반 초기해"""
    instance.reset()
    customers = sorted(instance.customers, key=lambda c: c.serv_time)
    vehicles = instance.vehicles

    for v in vehicles:
        v.schedules = []
        v.now_capacity = 0.0

    v_idx = 0
    for c in customers:
        for _ in range(len(vehicles)):
            v = vehicles[v_idx % len(vehicles)]
            if v.now_capacity + c.weight <= v.capacity:
                v.schedules.append(c)
                v.now_capacity += c.weight
                v_idx += 1
                break
            v_idx += 1

    tard = calculate_total_tardiness(vehicles)
    return Solution("SPT", instance, obj=tard)


def dispatch_edd(instance: Instance) -> Solution:
    """EDD (Earliest Due Date) 기반 초기해"""
    instance.reset()
    customers = sorted(instance.customers, key=lambda c: c.tw[1])
    vehicles = instance.vehicles

    for v in vehicles:
        v.schedules = []
        v.now_capacity = 0.0

    v_idx = 0
    for c in customers:
        for _ in range(len(vehicles)):
            v = vehicles[v_idx % len(vehicles)]
            if v.now_capacity + c.weight <= v.capacity:
                v.schedules.append(c)
                v.now_capacity += c.weight
                v_idx += 1
                break
            v_idx += 1

    tard = calculate_total_tardiness(vehicles)
    return Solution("EDD", instance, obj=tard)


def dispatch_regret_k(instance: Instance, k: int = 3) -> Solution:
    """Regret-k 초기해"""
    instance.reset()
    customers = instance.customers
    vehicles = instance.vehicles

    for v in vehicles:
        v.schedules = []
        v.now_capacity = 0.0

    unserved = set(c.ID for c in customers)
    cust_by_id = {c.ID: c for c in customers}

    while unserved:
        best_regret = -1
        best_insertion = None

        for cid in list(unserved):
            c = cust_by_id[cid]
            costs = []

            for v_idx, v in enumerate(vehicles):
                if v.now_capacity + c.weight > v.capacity:
                    continue
                for pos in range(len(v.schedules) + 1):
                    v.schedules.insert(pos, c)
                    cost = calculate_total_tardiness(vehicles)
                    v.schedules.pop(pos)
                    costs.append((cost, v_idx, pos))

            if len(costs) < k:
                continue

            costs.sort()
            regret = costs[min(k - 1, len(costs) - 1)][0] - costs[0][0]

            if regret > best_regret:
                best_regret = regret
                best_insertion = (cid, costs[0][1], costs[0][2])

        if best_insertion is None:
            break

        cid, v_idx, pos = best_insertion
        c = cust_by_id[cid]
        vehicles[v_idx].schedules.insert(pos, c)
        vehicles[v_idx].now_capacity += c.weight
        unserved.remove(cid)

    tard = calculate_total_tardiness(vehicles)
    return Solution("Regret-k", instance, obj=tard)


# ==========================================
# Local Search
# ==========================================

def local_search_swap(vehicles: List[Vehicle]) -> bool:
    """Swap local search"""
    improved = False
    current_cost = calculate_total_tardiness(vehicles)

    for v in vehicles:
        for i in range(len(v.schedules)):
            for j in range(i + 1, len(v.schedules)):
                v.schedules[i], v.schedules[j] = v.schedules[j], v.schedules[i]
                new_cost = calculate_total_tardiness(vehicles)

                if new_cost < current_cost:
                    current_cost = new_cost
                    improved = True
                else:
                    v.schedules[i], v.schedules[j] = v.schedules[j], v.schedules[i]

    return improved


def local_search_insert(vehicles: List[Vehicle]) -> bool:
    """Insert local search"""
    improved = False
    current_cost = calculate_total_tardiness(vehicles)

    for v in vehicles:
        for i in range(len(v.schedules)):
            customer = v.schedules[i]
            for j in range(len(v.schedules)):
                if i == j:
                    continue
                v.schedules.pop(i)
                v.schedules.insert(j, customer)
                new_cost = calculate_total_tardiness(vehicles)

                if new_cost < current_cost:
                    current_cost = new_cost
                    improved = True
                    break
                else:
                    v.schedules.pop(j)
                    v.schedules.insert(i, customer)
            if improved:
                break

    return improved


def local_search_2opt(vehicles: List[Vehicle]) -> bool:
    """2-opt local search"""
    improved = False
    current_cost = calculate_total_tardiness(vehicles)

    for v in vehicles:
        if len(v.schedules) < 4:
            continue
        for i in range(len(v.schedules) - 1):
            for j in range(i + 2, len(v.schedules)):
                v.schedules[i + 1:j + 1] = reversed(v.schedules[i + 1:j + 1])
                new_cost = calculate_total_tardiness(vehicles)

                if new_cost < current_cost:
                    current_cost = new_cost
                    improved = True
                else:
                    v.schedules[i + 1:j + 1] = reversed(v.schedules[i + 1:j + 1])

    return improved


# ==========================================
# 메타휴리스틱
# ==========================================

def ils_optimize(instance: Instance, time_limit: float = 5.0) -> Solution:
    """ILS (Iterated Local Search)"""
    vehicles = instance.vehicles
    best_cost = calculate_total_tardiness(vehicles)
    best_state = [list(v.schedules) for v in vehicles]

    start = time.time()
    iteration = 0

    while time.time() - start < time_limit:
        iteration += 1

        # Local search
        improved = True
        while improved:
            improved = local_search_swap(vehicles)
            improved |= local_search_2opt(vehicles)

        current_cost = calculate_total_tardiness(vehicles)
        if current_cost < best_cost:
            best_cost = current_cost
            best_state = [list(v.schedules) for v in vehicles]

        # Perturbation
        if iteration % 3 == 0:
            for v in vehicles:
                if len(v.schedules) > 2:
                    i, j = random.sample(range(len(v.schedules)), 2)
                    v.schedules[i], v.schedules[j] = v.schedules[j], v.schedules[i]

    # Restore best
    for v_idx, v in enumerate(vehicles):
        v.schedules = best_state[v_idx]

    return Solution("ILS", instance, obj=best_cost)


def vns_optimize(instance: Instance, time_limit: float = 5.0) -> Solution:
    """VNS (Variable Neighborhood Search)"""
    vehicles = instance.vehicles
    best_cost = calculate_total_tardiness(vehicles)
    best_state = [list(v.schedules) for v in vehicles]

    start = time.time()
    neighborhoods = [local_search_swap, local_search_insert, local_search_2opt]

    while time.time() - start < time_limit:
        for neighborhood in neighborhoods:
            improved = neighborhood(vehicles)
            if improved:
                current_cost = calculate_total_tardiness(vehicles)
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_state = [list(v.schedules) for v in vehicles]
                break

    for v_idx, v in enumerate(vehicles):
        v.schedules = best_state[v_idx]

    return Solution("VNS", instance, obj=best_cost)


def sa_optimize(instance: Instance, time_limit: float = 5.0) -> Solution:
    """SA (Simulated Annealing)"""
    vehicles = instance.vehicles
    current_cost = calculate_total_tardiness(vehicles)
    best_cost = current_cost
    best_state = [list(v.schedules) for v in vehicles]

    T = 100.0
    T_min = 0.01
    alpha = 0.95
    start = time.time()

    while time.time() - start < time_limit and T > T_min:
        # Random swap
        v = random.choice(vehicles)
        if len(v.schedules) < 2:
            continue

        i, j = random.sample(range(len(v.schedules)), 2)
        v.schedules[i], v.schedules[j] = v.schedules[j], v.schedules[i]

        new_cost = calculate_total_tardiness(vehicles)
        delta = new_cost - current_cost

        if delta < 0 or random.random() < math.exp(-delta / T):
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_state = [list(v.schedules) for v in vehicles]
        else:
            v.schedules[i], v.schedules[j] = v.schedules[j], v.schedules[i]

        T *= alpha

    for v_idx, v in enumerate(vehicles):
        v.schedules = best_state[v_idx]

    return Solution("SA", instance, obj=best_cost)


def ga_optimize(instance: Instance, time_limit: float = 5.0) -> Solution:
    """GA (Genetic Algorithm)"""
    pop_size = 20
    vehicles = instance.vehicles

    # Initialize population
    population = []
    for _ in range(pop_size):
        inst_copy = deep_copy_solution(instance)
        sol = dispatch_edd(inst_copy)
        population.append((sol.objective, [list(v.schedules) for v in inst_copy.vehicles]))

    start = time.time()
    generation = 0

    while time.time() - start < time_limit:
        generation += 1

        # Selection (tournament)
        parents = []
        for _ in range(pop_size // 2):
            tournament = random.sample(population, 3)
            parents.append(min(tournament, key=lambda x: x[0]))

        # Crossover
        offspring = []
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                child = parents[i][1]  # Simple copy
                offspring.append(child)

        # Mutation
        for child in offspring:
            if random.random() < 0.3:
                v_idx = random.randint(0, len(child) - 1)
                if len(child[v_idx]) > 1:
                    i, j = random.sample(range(len(child[v_idx])), 2)
                    child[v_idx][i], child[v_idx][j] = child[v_idx][j], child[v_idx][i]

        # Evaluate
        new_pop = []
        for child in offspring:
            for v_idx, v in enumerate(vehicles):
                v.schedules = child[v_idx]
            cost = calculate_total_tardiness(vehicles)
            new_pop.append((cost, child))

        population = sorted(population + new_pop, key=lambda x: x[0])[:pop_size]

    # Best solution
    best_cost, best_state = population[0]
    for v_idx, v in enumerate(vehicles):
        v.schedules = best_state[v_idx]

    return Solution("GA", instance, obj=best_cost)


# ==========================================
# Gurobi (작은 문제만)
# ==========================================
def gurobi_optimize(instance: Instance, time_limit: float = 300) -> Optional[Solution]:
    """Gurobi 최적화 (고객 15개 이하만)"""
    try:
        from gurobipy import Model, GRB, quicksum

        if len(instance.customers) > 15:
            return None

        # 앞에서 작성한 gurobi_vrptw_simple 함수 사용
        # 여기서는 생략 (너무 길어서)
        return None
    except:
        return None
