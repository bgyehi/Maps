# ✅ 맨 첫 줄에 반드시 위치
from __future__ import annotations

from typing import List, Tuple, Optional
import math
import module
from module import Customer, Vehicle, Instance, Solution


# ---------- 기본 디스패칭 (capacity-aware) ----------
def dispatch_earliest_vehicle_best_customer(
    instance: Instance,
    *,
    scoring: str = "delta_tardiness",
    tie_breaker: str = "edd_then_nearest",  # 동점 시 1순위=EDD, 2순위=최근접
):

    # 항상 초기화하여 일관 상태 유지
    instance.reset()
    customers: List[Customer] = instance.customers
    vehicles:  List[Vehicle]  = instance.vehicles

    # 전체 용량 검사 (참고용)
    total_demand = sum(float(c.weight or 0.0) for c in customers)
    total_capacity = sum(float(v.capacity or 0.0) for v in vehicles)
    if total_demand > total_capacity + 1e-9:
        pass  # 강제종료는 하지 않음 (미배정 고객 발생 가능)

    # 고객 ID 관리
    cust_by_id = {c.ID: c for c in customers}
    unserved = set(cust_by_id.keys())

    total_tardiness = 0.0
    total_distance  = 0.0

    def feasible_for_vehicle(v: Vehicle, c: Customer) -> bool:
        """차량의 용량 확인"""
        cap = float(v.capacity or 0.0)
        used = float(v.now_capacity or 0.0)
        need = float(c.weight or 0.0)
        return (used + need) <= cap + 1e-9

    def best_customer_for_vehicle(v: Vehicle) -> Tuple[Optional[int], Optional[Tuple]]:
        """차량 v에 대해 가장 좋은 고객 선택"""
        best_cid = None
        best_key = None
        best_metrics = None

        feasible_ids = [cid for cid in unserved if feasible_for_vehicle(v, cust_by_id[cid])]
        if not feasible_ids:
            return None, None

        for cid in feasible_ids:
            c = cust_by_id[cid]
            travel_km = module.get_dist(tuple(v.now_loc), tuple(c.loc))
            travel_h  = travel_km / max(1e-9, float(v.speed or 1.0))
            start     = max(float(c.tw[0]), float(v.available or 0.0) + travel_h)
            end       = start + float(c.serv_time or 0.0)
            tard      = max(0.0, end - float(c.tw[1]))
            priority  = float(c.priority or 0.0)

            # 주점수 방식 선택
            if scoring == "delta_tardiness":
                primary = tard
                secondary = end
            elif scoring == "earliest_finish":
                primary = end
                secondary = tard
            elif scoring == "edd":
                primary = float(c.tw[1])
                secondary = tard
            elif scoring in ("shortest_distance", "nearest"):
                primary = travel_km
                secondary = tard
            elif scoring == "given":
                primary = -priority
                secondary = travel_km
            else:
                primary = tard
                secondary = end

            # tie breaker
            if tie_breaker == "edd_then_nearest":
                tie1 = float(c.tw[1])
                tie2 = travel_km
            else:
                tie1 = travel_km
                tie2 = float(c.tw[1])

            key = (primary, secondary, tie1, tie2, cid)
            if (best_key is None) or (key < best_key):
                best_key = key
                best_cid = cid
                best_metrics = (travel_km, travel_h, start, end, tard)

        return best_cid, best_metrics

    # 메인 루프
    while unserved:
        order = sorted(range(len(vehicles)), key=lambda kk: (float(vehicles[kk].available or 0.0), vehicles[kk].ID))
        selected = None
        chosen   = None
        metrics  = None

        for k in order:
            v = vehicles[k]
            cid, m = best_customer_for_vehicle(v)
            if cid is not None:
                selected = v
                chosen = cid
                metrics = m
                break

        if chosen is None:
            break  # 배정 불가 고객만 남음

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

        total_tardiness += tard
        total_distance  += travel_km
        unserved.remove(chosen)

    # 결과 반환
    try:
        sol = Solution("EarliestVhc-BestCust-Capacity", instance, obj=total_tardiness)
    except TypeError:
        class _Sol:
            def __init__(self, alg, inst, obj):
                self.algorithm = alg; self.instance = inst; self.objective = obj
                self.comp_time = "N/A"; self.status = "DONE"
            def __repr__(self):
                return f"Schedule by {self.algorithm} - Objective: {self.objective}"
        sol = _Sol("EarliestVhc-BestCust-Capacity", instance, total_tardiness)

    sol.total_distance = total_distance
    sol.unserved_ids = sorted(list(unserved))
    sol.status = "DONE" if not unserved else "INFEASIBLE_CAPACITY"
    return sol



# ---------- 개선형: Weighted Scoring Dispatch ----------
def dispatch_weighted_score(
    instance: Instance,
    *,
    weights=(1.0, 0.1, 0.5),  # (tardiness, distance, priority)
    tie_breaker="edd_then_nearest",
):
    """
    Weighted scoring dispatching:
    Score = w_tard * tardiness + w_dist * distance - w_prio * priority
    Objective = total tardiness (for fair comparison)
    """
    instance.reset()
    customers = instance.customers
    vehicles = instance.vehicles

    w_tard, w_dist, w_prio = weights

    cust_by_id = {c.ID: c for c in customers}
    unserved = set(cust_by_id.keys())

    total_distance = 0.0
    total_tardiness = 0.0

    def feasible(v: Vehicle, c: Customer) -> bool:
        cap = float(v.capacity or 0.0)
        used = float(v.now_capacity or 0.0)
        need = float(c.weight or 0.0)
        return (used + need) <= cap + 1e-9

    def best_customer_for_vehicle(v: Vehicle):
        best_cid = None
        best_key = None
        best_metrics = None

        feasible_ids = [cid for cid in unserved if feasible(v, cust_by_id[cid])]
        if not feasible_ids:
            return None, None

        for cid in feasible_ids:
            c = cust_by_id[cid]
            travel_km = module.get_dist(tuple(v.now_loc), tuple(c.loc))
            travel_h = travel_km / max(1e-9, float(v.speed or 1.0))
            start = max(float(c.tw[0]), float(v.available or 0.0) + travel_h)
            end = start + float(c.serv_time or 0.0)
            tard = max(0.0, end - float(c.tw[1]))
            priority = float(c.priority or 0.0)

            score = w_tard * tard + w_dist * travel_km - w_prio * priority

            if tie_breaker == "edd_then_nearest":
                tie1 = float(c.tw[1])
                tie2 = travel_km
            else:
                tie1 = travel_km
                tie2 = float(c.tw[1])

            key = (score, tie1, tie2, cid)
            if (best_key is None) or (key < best_key):
                best_key = key
                best_cid = cid
                best_metrics = (travel_km, travel_h, start, end, tard, score)

        return best_cid, best_metrics

    while unserved:
        order = sorted(range(len(vehicles)), key=lambda k: (float(vehicles[k].available or 0.0), vehicles[k].ID))
        selected = None
        chosen = None
        metrics = None

        for k in order:
            v = vehicles[k]
            cid, m = best_customer_for_vehicle(v)
            if cid is not None:
                selected = v
                chosen = cid
                metrics = m
                break

        if chosen is None:
            break

        v = selected
        c = cust_by_id[chosen]
        travel_km, travel_h, start, end, tard, score = metrics

        c.assigned_vhc = v.ID
        c.start = start
        c.end = end
        c.tardy = tard
        c.complete = True

        v.schedules.append(c)
        v.available = end
        v.now_loc = list(c.loc)
        v.now_capacity = float(v.now_capacity or 0.0) + float(c.weight or 0.0)

        total_tardiness += tard
        total_distance += travel_km
        unserved.remove(chosen)

    try:
        sol = Solution("WeightedScoreDispatch", instance, obj=total_tardiness)
    except TypeError:
        class _Sol:
            def __init__(self, alg, inst, obj):
                self.algorithm = alg; self.instance = inst; self.objective = obj
                self.comp_time = "N/A"; self.status = "DONE"
            def __repr__(self):
                return f"Schedule by {self.algorithm} - Objective: {self.objective}"
        sol = _Sol("WeightedScoreDispatch", instance, total_tardiness)

    sol.total_distance = total_distance
    sol.unserved_ids = sorted(list(unserved))
    sol.status = "DONE" if not unserved else "INFEASIBLE_CAPACITY"
    return sol
