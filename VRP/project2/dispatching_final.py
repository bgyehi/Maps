from __future__ import annotations
from typing import List
import random
import time
import math
import copy

from module import Customer, Vehicle, Instance, Solution, get_dist

# =================================================
# 공통 함수
# =================================================
def calculate_total_tardiness(vehicles: List[Vehicle]) -> float:
    total = 0.0
    for v in vehicles:
        current_time = 0.0
        current_loc = tuple(v.loc)
        speed = float(v.speed or 30.0)

        for c in v.schedules:
            dist = get_dist(current_loc, tuple(c.loc))
            travel = dist / max(speed, 1e-9)
            arrival = current_time + travel
            start = max(arrival, c.tw[0])
            end = start + c.serv_time
            total += max(0.0, end - c.tw[1])
            current_time = end
            current_loc = tuple(c.loc)
    return total


def deep_copy_solution(instance: Instance) -> Instance:
    return copy.deepcopy(instance)


# =================================================
# 초기해
# =================================================
def dispatch_spt(instance: Instance) -> Solution:
    instance.reset()
    customers = sorted(instance.customers, key=lambda c: c.serv_time)
    vehicles = instance.vehicles

    for v in vehicles:
        v.schedules = []
        v.now_capacity = 0.0

    idx = 0
    for c in customers:
        for _ in vehicles:
            v = vehicles[idx % len(vehicles)]
            if v.now_capacity + c.weight <= v.capacity:
                v.schedules.append(c)
                v.now_capacity += c.weight
                idx += 1
                break
            idx += 1

    return Solution("SPT", instance,
                    calculate_total_tardiness(vehicles))


def dispatch_edd(instance: Instance) -> Solution:
    instance.reset()
    customers = sorted(instance.customers, key=lambda c: c.tw[1])
    vehicles = instance.vehicles

    for v in vehicles:
        v.schedules = []
        v.now_capacity = 0.0

    idx = 0
    for c in customers:
        for _ in vehicles:
            v = vehicles[idx % len(vehicles)]
            if v.now_capacity + c.weight <= v.capacity:
                v.schedules.append(c)
                v.now_capacity += c.weight
                idx += 1
                break
            idx += 1

    return Solution("EDD", instance,
                    calculate_total_tardiness(vehicles))


# =================================================
# Local Search
# =================================================
def local_search_swap(vehicles):
    cur = calculate_total_tardiness(vehicles)
    for v in vehicles:
        for i in range(len(v.schedules)):
            for j in range(i + 1, len(v.schedules)):
                v.schedules[i], v.schedules[j] = v.schedules[j], v.schedules[i]
                if calculate_total_tardiness(vehicles) < cur:
                    return True
                v.schedules[i], v.schedules[j] = v.schedules[j], v.schedules[i]
    return False


def local_search_insert(vehicles):
    cur = calculate_total_tardiness(vehicles)
    for v in vehicles:
        for i in range(len(v.schedules)):
            c = v.schedules.pop(i)
            for j in range(len(v.schedules)):
                v.schedules.insert(j, c)
                if calculate_total_tardiness(vehicles) < cur:
                    return True
                v.schedules.pop(j)
            v.schedules.insert(i, c)
    return False


def local_search_2opt(vehicles):
    cur = calculate_total_tardiness(vehicles)
    for v in vehicles:
        if len(v.schedules) < 4:
            continue
        for i in range(len(v.schedules)):
            for j in range(i + 2, len(v.schedules)):
                v.schedules[i:j] = reversed(v.schedules[i:j])
                if calculate_total_tardiness(vehicles) < cur:
                    return True
                v.schedules[i:j] = reversed(v.schedules[i:j])
    return False


# =================================================
# ILS (🔥 수정됨)
# =================================================
def ils_optimize(instance: Instance, time_limit=1.5) -> Solution:
    # 🔥 초기해 없으면 자동 생성
    if all(len(v.schedules) == 0 for v in instance.vehicles):
        dispatch_edd(instance)

    vehicles = instance.vehicles
    assert sum(len(v.schedules) for v in vehicles) > 0, "ILS started with empty solution!"

    best = calculate_total_tardiness(vehicles)
    best_state = [list(v.schedules) for v in vehicles]

    start = time.time()
    while time.time() - start < time_limit:
        improved = True
        while improved:
            improved = local_search_swap(vehicles)
            improved |= local_search_2opt(vehicles)

        cur = calculate_total_tardiness(vehicles)
        if cur < best:
            best = cur
            best_state = [list(v.schedules) for v in vehicles]
        else:
            v = random.choice(vehicles)
            if len(v.schedules) > 1:
                i, j = random.sample(range(len(v.schedules)), 2)
                v.schedules[i], v.schedules[j] = v.schedules[j], v.schedules[i]

    for i, v in enumerate(vehicles):
        v.schedules = best_state[i]

    return Solution("EDD+ILS", instance, best)


# =================================================
# VNS (🔥 수정됨)
# =================================================
def vns_optimize(instance: Instance, time_limit=1.5) -> Solution:
    # 🔥 초기해 없으면 자동 생성
    if all(len(v.schedules) == 0 for v in instance.vehicles):
        dispatch_edd(instance)

    vehicles = instance.vehicles
    assert sum(len(v.schedules) for v in vehicles) > 0, "VNS started with empty solution!"

    best = calculate_total_tardiness(vehicles)
    best_state = [list(v.schedules) for v in vehicles]

    start = time.time()
    neighborhoods = [local_search_swap,
                     local_search_insert,
                     local_search_2opt]

    while time.time() - start < time_limit:
        for nbh in neighborhoods:
            if nbh(vehicles):
                cur = calculate_total_tardiness(vehicles)
                if cur < best:
                    best = cur
                    best_state = [list(v.schedules) for v in vehicles]
                break

    for i, v in enumerate(vehicles):
        v.schedules = best_state[i]

    return Solution("EDD+VNS", instance, best)


# =================================================
# Extreme Optimization (그대로)
# =================================================
def strong_perturbation(vehicles):
    for v in vehicles:
        if len(v.schedules) > 3:
            random.shuffle(v.schedules)


def extreme_optimization(instance: Instance, max_iter=400) -> Solution:
    s1 = dispatch_spt(deep_copy_solution(instance))
    s2 = dispatch_edd(deep_copy_solution(instance))
    current = min([s1, s2], key=lambda s: s.objective)

    vehicles = current.instance.vehicles
    S_cost = calculate_total_tardiness(vehicles)
    best_cost = S_cost
    best_state = [list(v.schedules) for v in vehicles]

    T_init = S_cost * 1.0
    T = T_init

    operators = {
        "swap": lambda: local_search_swap(vehicles),
        "insert": lambda: local_search_insert(vehicles),
        "2opt": lambda: local_search_2opt(vehicles)
    }
    weights = {k: 1.0 for k in operators}
    no_improve = 0

    for it in range(max_iter):
        intensify = it > 0.6 * max_iter

        total = sum(weights.values())
        r = random.random() * total
        acc = 0
        for k, w in weights.items():
            acc += w
            if r <= acc:
                op = k
                break

        backup = [list(v.schedules) for v in vehicles]
        operators[op]()
        new_cost = calculate_total_tardiness(vehicles)
        delta = new_cost - S_cost

        if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-9)):
            S_cost = new_cost
            weights[op] += 0.2
            if S_cost < best_cost:
                best_cost = S_cost
                best_state = [list(v.schedules) for v in vehicles]
                no_improve = 0
            else:
                no_improve += 1
        else:
            for i, v in enumerate(vehicles):
                v.schedules = backup[i]
            weights[op] *= 0.95
            no_improve += 1

        if no_improve >= 50:
            strong_perturbation(vehicles)
            T = T_init * 0.6
            no_improve = 0

        T *= 0.97 if intensify else 0.99

    for i, v in enumerate(vehicles):
        v.schedules = best_state[i]

    return Solution("ExtremeOpt", instance, best_cost)
