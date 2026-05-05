"""
용량을 고려한 탐욕 디스패칭 휴리스틱 — Earliest-Available Vehicle + Best Customer (한국어 친절 각주)

개요
- 핵심 아이디어: "가장 빨리 비는 차량"을 고르고, 그 차량에 실을 수 있는 고객들 중 점수(score)가 가장 좋은 고객을 즉시 배정(주1).
- 점수 산식은 선택형(scoring)이며, 지연증분/종료시점/EDD/최근접/우선순위 기반 중 택1(주2).
- 동률 해소는 EDD→최근접 또는 최근접→EDD 순으로 결정(주3).
- 용량 제약(capacity)을 즉시 반영하여, 남은 적재 여유를 초과하는 고객은 후보에서 제외(주4).

입출력
- 입력: Instance(customers, vehicles)
  * Vehicle: available, now_loc, speed, capacity, now_capacity, schedules 등(주5)
  * Customer: loc, tw=[ready,due], serv_time, weight, priority 등(주6)
- 출력: Solution (objective=총지연), 부가적으로 total_distance, unserved_ids, status(DONE/INFEASIBLE_CAPACITY) 부여(주7)

주의/가정
- 본 함수는 시작 시 instance.reset()을 호출하여 상태를 초기화합니다(주8).
- 시간 단위는 "시간", 속도는 km/h, 거리는 km로 일관(주9).
- 부분 서비스 정책(용량 부족 시 일부 고객 미배정)을 허용합니다(주10).
"""

from __future__ import annotations
from typing import List, Tuple, Optional

import module
from module import Customer, Vehicle, Instance, Solution


# ---------- dispatching heuristic (capacity-aware) ----------
def dispatch_earliest_vehicle_best_customer(
    instance: Instance,
    *,
    scoring: str = "delta_tardiness",
    tie_breaker: str = "edd_then_nearest",  # 동점 시 1순위=EDD, 2순위=최근접(주3)
):
    """
탐욕적(greedy)이며 용량(capacity)을 고려하는 디스패처:
  1) 'available'(다음 가용 시각)가 가장 작은 차량을 선택합니다(동률이면 Vehicle.ID가 작은 차량).
  2) 아직 배정되지 않았고, 해당 차량의 남은 용량에 들어가는 고객들 중에서
     그 차량에 바로 배정했을 때 '가장 좋은' 고객을 고릅니다.
     기본 규칙: 다음으로 배정했을 때의 증분 지연(tardiness)이 가장 작은 고객.

매개변수(Args)
    instance: .customers와 .vehicles를 갖는 Instance. 각 Vehicle은 .capacity와 .now_capacity를 가져야 합니다.
    scoring:
      - "delta_tardiness"(기본): 바로 다음에 서비스했을 때의 지연값을 최소화
      - "earliest_finish": 종료 시각(end)을 최소화(동률 시 tardiness 고려)
      - "edd": 마감시간(due)만 고려하는 EDD 규칙
      - "nearest" / "shortest_distance": 이동거리 최소화(동률 시 tardiness 고려)
      - "given": 고객의 .priority 사용(클수록 우선)
    tie_breaker:
      - "edd_then_nearest"(기본): 동률이면 먼저 EDD(더 이른 due), 그다음 최근접
      - "nearest_then_edd": 반대로 최근접 우선 후 EDD

반환(Returns)
    Solution 객체(목적함수=총 지연)를 반환하며, 다음 부가 속성을 함께 설정합니다.
      - .total_distance: 총 이동거리(km)
      - .unserved_ids: 용량 제약으로 미배정된 고객 ID 목록(list[int])
      - .status: {"DONE", "INFEASIBLE_CAPACITY"} 중 하나
"""
    # 항상 초기화하여 일관 상태 유지(주8)
    instance.reset()
    customers: List[Customer] = instance.customers
    vehicles:  List[Vehicle]  = instance.vehicles

    # 전역 용량 타당성(참고용, 미배정 허용하므로 강제 종료하진 않음)(주10)
    total_demand = sum(float(c.weight or 0.0) for c in customers)
    total_capacity = sum(float(v.capacity or 0.0) for v in vehicles)
    if total_demand > total_capacity + 1e-9:
        # 여전히 최대한 배정은 시도하되, 결과 status는 INFEASIBLE_CAPACITY가 될 수 있음
        pass

    # 고객 인덱스 및 미배정 집합
    cust_by_id = {c.ID: c for c in customers}
    unserved = set(cust_by_id.keys())

    total_tardiness = 0.0
    total_distance  = 0.0

    def feasible_for_vehicle(v: Vehicle, c: Customer) -> bool:
        """차량 v의 남은 용량으로 고객 c 수요를 수용 가능한가? (주4)"""
        cap = float(v.capacity or 0.0)
        used = float(v.now_capacity or 0.0)
        need = float(c.weight or 0.0)
        return (used + need) <= cap + 1e-9

    def best_customer_for_vehicle(v: Vehicle) -> Tuple[Optional[int], Optional[Tuple]]:
        """
        차량 v에 대해 현재 미배정 고객 중 용량상 feasible한 고객들만 고려하여
        (cid, metrics)를 반환. metrics = (travel_km, travel_h, start, end, tard)
        """
        best_cid = None
        best_key = None
        best_metrics = None

        # 용량 feasible 집합
        feasible_ids = [cid for cid in unserved if feasible_for_vehicle(v, cust_by_id[cid])]
        if not feasible_ids:
            return None, None

        for cid in feasible_ids:
            c = cust_by_id[cid]
            travel_km = module.get_dist(tuple(v.now_loc), tuple(c.loc))  # km(주9)
            travel_h  = travel_km / max(1e-9, float(v.speed or 1.0))     # h
            start     = max(float(c.tw[0]), float(v.available or 0.0) + travel_h)
            end       = start + float(c.serv_time or 0.0)
            tard      = max(0.0, end - float(c.tw[1]))
            priority  = float(c.priority or 0.0)

            # 주점수 산식 선택(주2)
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
                primary = -priority  # priority가 클수록 우선
                secondary = travel_km
            else:  # 기본은 delta_tardiness
                primary = tard
                secondary = end

            # 동점 해소 규칙(주3)
            if tie_breaker == "edd_then_nearest":
                tie1 = float(c.tw[1])
                tie2 = travel_km
            else:  # "nearest_then_edd"
                tie1 = travel_km
                tie2 = float(c.tw[1])

            key = (primary, secondary, tie1, tie2, cid)
            if (best_key is None) or (key < best_key):
                best_key = key
                best_cid = cid
                best_metrics = (travel_km, travel_h, start, end, tard)

        return best_cid, best_metrics

    # 메인 탐욕 루프(용량 인지)
    while unserved:
        # 차량을 (available, ID) 오름차순으로 정렬(가장 빨리 비는 차량 우선)(주1)
        order = sorted(range(len(vehicles)), key=lambda kk: (float(vehicles[kk].available or 0.0), vehicles[kk].ID))

        selected = None
        chosen   = None
        metrics  = None

        # 아직 수용 가능한 고객이 남아있는 가장 이른 차량 선택
        for k in order:
            v = vehicles[k]
            cid, m = best_customer_for_vehicle(v)
            if cid is not None:
                selected = v
                chosen = cid
                metrics = m
                break

        if chosen is None:
            # 남은 고객이 모두 용량 제약 때문에 배정 불가 → 중단(주10)
            break

        # 배정 커밋
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

    # 결과 패키징
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

    if unserved:
        sol.status = "INFEASIBLE_CAPACITY"  # 용량 부족으로 미배정 고객 존재
    else:
        sol.status = "DONE"

    return sol


# ============================== 각주 모음 ==============================
# (주1) Earliest-Available Vehicle: v.available(다음 가용 시각)가 가장 작은 차량을 먼저 선택하여 공회전 최소화 유도.
# (주2) scoring:
#   - delta_tardiness: 다음으로 배정했을 때의 지연(tard) 최소화(시간창 위반 우선 억제 효과).
#   - earliest_finish: 종료시점 최소화(throughput 지향). 동점 시 tard 고려.
#   - edd: due만 보고 EDD 규칙 적용. 이때 TW 내 도착 가능성은 secondary에서 보조.
#   - nearest/shortest_distance: 이동거리 최소화(연료/이동비 절감). 동점 시 tard 보조.
#   - given: 고객 priority 기반. priority가 높을수록 먼저.
# (주3) tie_breaker:
#   - edd_then_nearest: due가 더 이른 고객 우선, 같으면 더 가까운 고객.
#   - nearest_then_edd: 반대 순서. 운영 목적에 맞춰 선택.
# (주4) 용량 인지: v.now_capacity + c.weight ≤ v.capacity 여야 후보로 인정. 1e-9 허용오차.
# (주5) Vehicle 필수 동적 필드: available, now_loc, now_capacity는 instance.reset()에서 적절히 초기화되어야 함.
# (주6) Customer 필수 필드: loc, tw, serv_time, weight가 세팅되어야 현실적인 시간/지연 계산 가능.
# (주7) Solution 부가 속성: total_distance(km), unserved_ids(List[int]), status("DONE"/"INFEASIBLE_CAPACITY").
# (주8) reset(): 이전 스케줄 잔존 상태가 해에 영향을 주지 않도록 선초기화.
# (주9) 단위 일관성: 거리(km), 속도(km/h), 시간(h). get_dist는 km 가정.
# (주10) 부분 서비스: 총수요>총용량이면 일부 미배정 허용. 최적성 보장은 없으나 빠른 탐욕 해 제공.
