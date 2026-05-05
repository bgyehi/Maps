from __future__ import annotations
from typing import List
import random
import time
import copy

from module import Customer, Vehicle, Instance, Solution, get_dist


# =====================================================
# 공통 함수
# =====================================================
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
            start = max(arrival, float(c.tw[0]))
            finish = start + float(c.serv_time or 0.0)
            tard = max(0.0, finish - float(c.tw[1]))

            total += tard
            current_time = finish
            current_loc = tuple(c.loc)

    return total


def deep_copy_solution(instance: Instance) -> Instance:
    return copy.deepcopy(instance)


# =====================================================
# 초기해: SPT / EDD / Regret-3
# =====================================================
def dispatch_spt(instance: Instance) -> Solution:
    instance.reset()
    customers = sorted(instance.customers, key=lambda c: c.serv_time)

    for v in instance.vehicles:
        v.schedules = []
        v.now_capacity = 0.0

    v_idx = 0
    for c in customers:
        for _ in range(len(instance.vehicles)):
            v = instance.vehicles[v_idx % len(instance.vehicles)]
            if v.now_capacity + c.weight <= v.capacity:
                v.schedules.append(c)
                v.now_capacity += c.weight
                break
            v_idx += 1

    obj = calculate_total_tardiness(instance.vehicles)
    return Solution("SPT", instance, obj=obj)


def dispatch_edd(instance: Instance) -> Solution:
    instance.reset()
    customers = sorted(instance.customers, key=lambda c: c.tw[1])

    for v in instance.vehicles:
        v.schedules = []
        v.now_capacity = 0.0

    v_idx = 0
    for c in customers:
        for _ in range(len(instance.vehicles)):
            v = instance.vehicles[v_idx % len(instance.vehicles)]
            if v.now_capacity + c.weight <= v.capacity:
                v.schedules.append(c)
                v.now_capacity += c.weight
                break
            v_idx += 1

    obj = calculate_total_tardiness(instance.vehicles)
    return Solution("EDD", instance, obj=obj)


def dispatch_regret_k(instance: Instance, k: int = 3) -> Solution:
    instance.reset()

    for v in instance.vehicles:
        v.schedules = []
        v.now_capacity = 0.0

    unassigned = instance.customers[:]

    while unassigned:
        best = None
        best_regret = -1

        for c in unassigned:
            costs = []
            for v in instance.vehicles:
                if v.now_capacity + c.weight > v.capacity:
                    continue
                for pos in range(len(v.schedules) + 1):
                    v.schedules.insert(pos, c)
                    cost = calculate_total_tardiness(instance.vehicles)
                    v.schedules.pop(pos)
                    costs.append(cost)

            if len(costs) < k:
                continue

            costs.sort()
            regret = costs[min(k - 1, len(costs) - 1)] - costs[0]

            if regret > best_regret:
                best_regret = regret
                best = c

        if best is None:
            break

        # 최소 cost 위치에 삽입
        best_cost = float("inf")
        best_v, best_pos = None, None
        for v in instance.vehicles:
            if v.now_capacity + best.weight > v.capacity:
                continue
            for pos in range(len(v.schedules) + 1):
                v.schedules.insert(pos, best)
                cost = calculate_total_tardiness(instance.vehicles)
                v.schedules.pop(pos)
                if cost < best_cost:
                    best_cost = cost
                    best_v, best_pos = v, pos

        best_v.schedules.insert(best_pos, best)
        best_v.now_capacity += best.weight
        unassigned.remove(best)

    obj = calculate_total_tardiness(instance.vehicles)
    return Solution("Regret-3", instance, obj=obj)


# =====================================================
# Local Search
# =====================================================
def local_search_swap(vehicles):
    improved = False
    base = calculate_total_tardiness(vehicles)

    for v in vehicles:
        for i in range(len(v.schedules)):
            for j in range(i + 1, len(v.schedules)):
                v.schedules[i], v.schedules[j] = v.schedules[j], v.schedules[i]
                new = calculate_total_tardiness(vehicles)
                if new < base:
                    return True
                v.schedules[i], v.schedules[j] = v.schedules[j], v.schedules[i]
    return improved


def local_search_2opt(vehicles):
    for v in vehicles:
        for i in range(len(v.schedules) - 2):
            for j in range(i + 2, len(v.schedules)):
                v.schedules[i:j] = reversed(v.schedules[i:j])
                return True
    return False


# =====================================================
# ILS
# =====================================================
def ils_optimize(instance: Instance, time_limit=3.0) -> Solution:
    vehicles = instance.vehicles
    best_cost = calculate_total_tardiness(vehicles)
    best_state = [list(v.schedules) for v in vehicles]

    start = time.time()
    while time.time() - start < time_limit:
        improved = True
        while improved:
            improved = local_search_swap(vehicles)
            improved |= local_search_2opt(vehicles)

        cur = calculate_total_tardiness(vehicles)
        if cur < best_cost:
            best_cost = cur
            best_state = [list(v.schedules) for v in vehicles]

        # perturb
        for v in vehicles:
            if len(v.schedules) >= 2:
                i, j = random.sample(range(len(v.schedules)), 2)
                v.schedules[i], v.schedules[j] = v.schedules[j], v.schedules[i]

    for i, v in enumerate(vehicles):
        v.schedules = best_state[i]

    return Solution("ILS", instance, obj=best_cost)


# =====================================================
# VNS
# =====================================================
def vns_optimize(instance: Instance, time_limit=3.0) -> Solution:
    vehicles = instance.vehicles
    best_cost = calculate_total_tardiness(vehicles)
    best_state = [list(v.schedules) for v in vehicles]

    start = time.time()
    while time.time() - start < time_limit:
        if local_search_swap(vehicles) or local_search_2opt(vehicles):
            cur = calculate_total_tardiness(vehicles)
            if cur < best_cost:
                best_cost = cur
                best_state = [list(v.schedules) for v in vehicles]

    for i, v in enumerate(vehicles):
        v.schedules = best_state[i]

    return Solution("VNS", instance, obj=best_cost)
