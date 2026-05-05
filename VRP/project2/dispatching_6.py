from __future__ import annotations
from typing import List
import random
import time
import copy

from module import Customer, Vehicle, Instance, Solution, get_dist


# =========================================================
# 공통 함수
# =========================================================
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
    return copy.deepcopy(instance)


# =========================================================
# 초기해 (SPT / EDD)
# =========================================================
def dispatch_spt(instance: Instance) -> Solution:
    """SPT 기반 초기해"""
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
    """EDD 기반 초기해"""
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


# =========================================================
# Local Search
# =========================================================
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
            c = v.schedules[i]
            for j in range(len(v.schedules)):
                if i == j:
                    continue

                v.schedules.pop(i)
                v.schedules.insert(j, c)
                new_cost = calculate_total_tardiness(vehicles)

                if new_cost < current_cost:
                    improved = True
                    current_cost = new_cost
                    break
                else:
                    v.schedules.pop(j)
                    v.schedules.insert(i, c)

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
                v.schedules[i+1 : j+1] = reversed(v.schedules[i+1 : j+1])
                new_cost = calculate_total_tardiness(vehicles)

                if new_cost < current_cost:
                    current_cost = new_cost
                    improved = True
                else:
                    v.schedules[i+1 : j+1] = reversed(v.schedules[i+1 : j+1])

    return improved


# =========================================================
# ILS
# =========================================================
def ils_optimize(instance: Instance, time_limit: float = 5.0) -> Solution:
    vehicles = instance.vehicles
    best_cost = calculate_total_tardiness(vehicles)
    best_state = [list(v.schedules) for v in vehicles]

    start = time.time()
    iter_count = 0

    while time.time() - start < time_limit:
        iter_count += 1

        improved = True
        while improved:
            improved = local_search_swap(vehicles)
            improved |= local_search_2opt(vehicles)

        cur = calculate_total_tardiness(vehicles)
        if cur < best_cost:
            best_cost = cur
            best_state = [list(v.schedules) for v in vehicles]

        # perturbation
        if iter_count % 3 == 0:
            for v in vehicles:
                if len(v.schedules) > 2:
                    i, j = random.sample(range(len(v.schedules)), 2)
                    v.schedules[i], v.schedules[j] = v.schedules[j], v.schedules[i]

    # restore
    for i, v in enumerate(vehicles):
        v.schedules = best_state[i]

    return Solution("ILS", instance, obj=best_cost)


# =========================================================
# VNS
# =========================================================
def vns_optimize(instance: Instance, time_limit: float = 5.0) -> Solution:
    vehicles = instance.vehicles
    best_cost = calculate_total_tardiness(vehicles)
    best_state = [list(v.schedules) for v in vehicles]

    start = time.time()
    neighborhoods = [local_search_swap, local_search_insert, local_search_2opt]

    while time.time() - start < time_limit:
        for nb in neighborhoods:
            if nb(vehicles):
                cur = calculate_total_tardiness(vehicles)
                if cur < best_cost:
                    best_cost = cur
                    best_state = [list(v.schedules) for v in vehicles]
                break

    for i, v in enumerate(vehicles):
        v.schedules = best_state[i]

    return Solution("VNS", instance, obj=best_cost)
