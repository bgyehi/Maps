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
    total = 0.0
    for v in vehicles:
        depot_loc = tuple(v.loc)
        speed = float(v.speed or 30.0)
        cur_t = 0.0
        cur_loc = depot_loc

        for c in v.schedules:
            travel = get_dist(cur_loc, tuple(c.loc)) / max(speed, 1e-9)
            arr = cur_t + travel
            start = max(arr, float(c.tw[0]))
            end = start + float(c.serv_time or 0.0)
            tard = max(0.0, end - float(c.tw[1]))

            total += tard
            cur_t, cur_loc = end, tuple(c.loc)

    return total


def deep_copy_instance(instance: Instance) -> Instance:
    return copy.deepcopy(instance)


# ==========================================
# 초기해 생성
# ==========================================
def dispatch_spt(instance: Instance) -> Solution:
    instance.reset()
    customers = sorted(instance.customers, key=lambda c: c.serv_time)
    vehicles = instance.vehicles

    for v in vehicles:
        v.schedules, v.now_capacity = [], 0.0

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

    cost = calculate_total_tardiness(vehicles)
    return Solution("SPT", instance, obj=cost)


def dispatch_edd(instance: Instance) -> Solution:
    instance.reset()
    customers = sorted(instance.customers, key=lambda c: c.tw[1])
    vehicles = instance.vehicles

    for v in vehicles:
        v.schedules, v.now_capacity = [], 0.0

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

    cost = calculate_total_tardiness(vehicles)
    return Solution("EDD", instance, obj=cost)


# ==========================================
# Local Search (FIRST IMPROVEMENT)
# ==========================================
def ls_swap(vehicles: List[Vehicle], base_cost: float) -> Tuple[bool, float]:
    for v in vehicles:
        sched = v.schedules
        for i in range(len(sched)):
            for j in range(i + 1, len(sched)):
                sched[i], sched[j] = sched[j], sched[i]
                new_cost = calculate_total_tardiness(vehicles)

                if new_cost < base_cost:
                    return True, new_cost
                else:
                    sched[i], sched[j] = sched[j], sched[i]

    return False, base_cost


def ls_insert(vehicles: List[Vehicle], base_cost: float) -> Tuple[bool, float]:
    for v in vehicles:
        sched = v.schedules
        for i in range(len(sched)):
            item = sched[i]
            for j in range(len(sched)):
                if i == j:
                    continue
                sched.pop(i)
                sched.insert(j, item)

                new_cost = calculate_total_tardiness(vehicles)
                if new_cost < base_cost:
                    return True, new_cost

                sched.pop(j)
                sched.insert(i, item)

    return False, base_cost


def ls_2opt(vehicles: List[Vehicle], base_cost: float) -> Tuple[bool, float]:
    for v in vehicles:
        sched = v.schedules
        if len(sched) < 4:
            continue

        for i in range(len(sched) - 1):
            for j in range(i + 2, len(sched)):
                sched[i+1:j+1] = reversed(sched[i+1:j+1])
                new_cost = calculate_total_tardiness(vehicles)

                if new_cost < base_cost:
                    return True, new_cost
                sched[i+1:j+1] = reversed(sched[i+1:j+1])

    return False, base_cost


# ==========================================
# ILS 개선 버전
# ==========================================
def ils_optimize(instance: Instance, time_limit: float = 5.0) -> Solution:
    vehicles = instance.vehicles
    best_cost = calculate_total_tardiness(vehicles)
    best_state = [list(v.schedules) for v in vehicles]

    start = time.time()

    while time.time() - start < time_limit:

        improved = True
        cur_cost = best_cost

        while improved:
            improved, cur_cost = ls_swap(vehicles, cur_cost)
            if improved: continue
            improved, cur_cost = ls_insert(vehicles, cur_cost)
            if improved: continue
            improved, cur_cost = ls_2opt(vehicles, cur_cost)

        if cur_cost < best_cost:
            best_cost = cur_cost
            best_state = [list(v.schedules) for v in vehicles]

        # 강화된 perturbation (조각 랜덤 재배열)
        for v in vehicles:
            if len(v.schedules) > 4:
                a = random.randint(0, len(v.schedules)-2)
                b = random.randint(a+1, len(v.schedules)-1)
                random.shuffle(v.schedules[a:b+1])

    # 복원
    for idx, v in enumerate(vehicles):
        v.schedules = best_state[idx]

    return Solution("ILS", instance, obj=best_cost)


# ==========================================
# VNS 개선 버전
# ==========================================
def vns_optimize(instance: Instance, time_limit: float = 5.0) -> Solution:
    vehicles = instance.vehicles
    best_cost = calculate_total_tardiness(vehicles)
    best_state = [list(v.schedules) for v in vehicles]

    start = time.time()

    N = [ls_insert, ls_swap, ls_2opt]

    while time.time() - start < time_limit:
        for neigh in N:
            improved, new_cost = neigh(vehicles, best_cost)
            if improved:
                best_cost = new_cost
                best_state = [list(v.schedules) for v in vehicles]
                break
        else:
            break

    for idx, v in enumerate(vehicles):
        v.schedules = best_state[idx]

    return Solution("VNS", instance, obj=best_cost)


# ==========================================
# GA 개선 버전
# ==========================================
def ga_optimize(instance: Instance, time_limit: float = 5.0) -> Solution:
    pop_size = 18
    start = time.time()

    # 초기 population
    population = []
    for _ in range(pop_size):
        inst_copy = deep_copy_instance(instance)
        sol = dispatch_edd(inst_copy)
        state = [list(v.schedules) for v in inst_copy.vehicles]
        population.append((sol.objective, state))

    def evaluate(state):
        inst_copy = deep_copy_instance(instance)
        for i, v in enumerate(inst_copy.vehicles):
            v.schedules = [c for c in state[i]]
        return calculate_total_tardiness(inst_copy.vehicles)

    while time.time() - start < time_limit:

        # tournament selection
        parents = []
        for _ in range(pop_size//2):
            cand = random.sample(population, 3)
            parents.append(min(cand, key=lambda x: x[0]))

        offspring = []
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                p1 = parents[i][1]
                p2 = parents[i+1][1]

                # 간단한 RCX-lite crossover
                cut = random.randint(0, len(p1)-1)
                child = []
                for v_idx in range(len(p1)):
                    if v_idx < cut:
                        child.append(list(p1[v_idx]))
                    else:
                        child.append(list(p2[v_idx]))

                offspring.append(child)

        # mutation
        for child in offspring:
            if random.random() < 0.25:
                vid = random.randint(0, len(child)-1)
                if len(child[vid]) > 1:
                    i, j = random.sample(range(len(child[vid])), 2)
                    child[vid][i], child[vid][j] = child[vid][j], child[vid][i]

        # 평가
        new_pop = []
        for child in offspring:
            cost = evaluate(child)
            new_pop.append((cost, child))

        population = sorted(population + new_pop, key=lambda x: x[0])[:pop_size]

    best_cost, best_state = population[0]

    for idx, v in enumerate(instance.vehicles):
        v.schedules = best_state[idx]

    return Solution("GA", instance, obj=best_cost)
