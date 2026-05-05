from __future__ import annotations
from typing import List, Optional
import random
import time
import math
import copy

from module import Customer, Vehicle, Instance, Solution, get_dist

# ==================================================
# Tardiness 계산 (vehicle 단위, 고속)
# ==================================================

def vehicle_tardiness(v: Vehicle) -> float:
    total = 0.0
    depot = tuple(v.loc)
    speed = float(v.speed or 30.0)
    t = 0.0
    cur = depot

    for c in v.schedules:
        dist = get_dist(cur, tuple(c.loc))
        t += dist / max(speed, 1e-9)
        t = max(t, float(c.tw[0]))
        t += float(c.serv_time or 0.0)
        total += max(0.0, t - float(c.tw[1]))
        cur = tuple(c.loc)

    return total


def calculate_total_tardiness(vehicles: List[Vehicle]) -> float:
    return sum(vehicle_tardiness(v) for v in vehicles)


# ==================================================
# 초기해 생성 (SPT 유지, EDD 제거)
# ==================================================

def dispatch_spt(instance: Instance) -> Solution:
    instance.reset()
    customers = sorted(instance.customers, key=lambda c: c.serv_time)
    vehicles = instance.vehicles

    for v in vehicles:
        v.schedules = []
        v.now_capacity = 0.0

    vidx = 0
    for c in customers:
        for _ in range(len(vehicles)):
            v = vehicles[vidx % len(vehicles)]
            if v.now_capacity + c.weight <= v.capacity:
                v.schedules.append(c)
                v.now_capacity += c.weight
                vidx += 1
                break
            vidx += 1

    return Solution("SPT", instance, obj=calculate_total_tardiness(vehicles))


def dispatch_regret_k(instance: Instance, k: int = 3) -> Solution:
    instance.reset()
    vehicles = instance.vehicles
    customers = instance.customers

    for v in vehicles:
        v.schedules = []
        v.now_capacity = 0.0

    unserved = set(c.ID for c in customers)
    cmap = {c.ID: c for c in customers}

    while unserved:
        best = None
        best_regret = -1

        for cid in list(unserved):
            c = cmap[cid]
            trials = []

            for vi, v in enumerate(vehicles):
                if v.now_capacity + c.weight > v.capacity:
                    continue
                base = vehicle_tardiness(v)
                for pos in range(len(v.schedules) + 1):
                    v.schedules.insert(pos, c)
                    cost = vehicle_tardiness(v)
                    v.schedules.pop(pos)
                    trials.append((base - cost, vi, pos))

            if len(trials) < k:
                continue

            trials.sort(reverse=True)
            regret = trials[0][0] - trials[min(k - 1, len(trials) - 1)][0]
            if regret > best_regret:
                best_regret = regret
                best = (cid, trials[0][1], trials[0][2])

        if best is None:
            break

        cid, vi, pos = best
        c = cmap[cid]
        vehicles[vi].schedules.insert(pos, c)
        vehicles[vi].now_capacity += c.weight
        unserved.remove(cid)

    return Solution("Regret-k", instance, obj=calculate_total_tardiness(vehicles))


# ==================================================
# Local Search (Swap, 2-opt만 사용)
# ==================================================

def ls_swap(v: Vehicle) -> bool:
    base = vehicle_tardiness(v)
    for i in range(len(v.schedules)):
        for j in range(i + 1, len(v.schedules)):
            v.schedules[i], v.schedules[j] = v.schedules[j], v.schedules[i]
            if vehicle_tardiness(v) < base:
                return True
            v.schedules[i], v.schedules[j] = v.schedules[j], v.schedules[i]
    return False


def ls_2opt(v: Vehicle) -> bool:
    if len(v.schedules) < 4:
        return False
    base = vehicle_tardiness(v)
    for i in range(len(v.schedules) - 2):
        for j in range(i + 2, len(v.schedules)):
            v.schedules[i:j] = reversed(v.schedules[i:j])
            if vehicle_tardiness(v) < base:
                return True
            v.schedules[i:j] = reversed(v.schedules[i:j])
    return False


# ==================================================
# ILS + VNS 하이브리드 (SA, GA 제거)
# ==================================================

def ils_vns_optimize(instance: Instance, time_limit: float = 5.0) -> Solution:
    vehicles = instance.vehicles
    best_cost = calculate_total_tardiness(vehicles)
    best_state = [list(v.schedules) for v in vehicles]

    start = time.time()
    while time.time() - start < time_limit:
        improved = True
        while improved:
            improved = False
            for v in vehicles:
                improved |= ls_swap(v)
                improved |= ls_2opt(v)

        cur = calculate_total_tardiness(vehicles)
        if cur < best_cost:
            best_cost = cur
            best_state = [list(v.schedules) for v in vehicles]

        # 경량 perturbation
        v = random.choice(vehicles)
        if len(v.schedules) > 2:
            i, j = random.sample(range(len(v.schedules)), 2)
            v.schedules[i], v.schedules[j] = v.schedules[j], v.schedules[i]

    for i, v in enumerate(vehicles):
        v.schedules = best_state[i]

    return Solution("ILS-VNS", instance, obj=best_cost)


# ==================================================
# Solver wrapper (Instance 1~30용)
# ==================================================

def solve_instance(instance: Instance, time_limit: float = 5.0) -> Solution:
    # SPT 단독 vs SPT+Regret-k 중 더 좋은 초기해 선택
    spt = dispatch_spt(copy.deepcopy(instance))
    rk = dispatch_regret_k(copy.deepcopy(instance))

    base = spt if spt.objective <= rk.objective else rk
    instance = base.instance

    return ils_vns_optimize(instance, time_limit=time_limit)
