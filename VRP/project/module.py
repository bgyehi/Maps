"""
VRP with Time Windows (VRPTW) — 최소한의 코어 자료구조 + 랜덤 인스턴스 생성기 (한국어 친절 각주 포함)

구성요약
- 거리 함수(get_dist): (x,y) 좌표 기반 유클리드 거리(km) (주1)
- Customer / Vehicle / Instance / Solution: 간단한 데이터 모델 (주2)
- Vehicle.process(): 한 고객을 차량 경로에 추가하면서 시간/지연/적재 갱신 (주3)
- make_random_instance(): 무작위 VRPTW 인스턴스 생성(수요·용량·시간창 포함) (주4)

안전장치
- None 좌표/용량/속도 등에 대한 방어 코드와 명시적 예외를 포함합니다(주5).
- 정수/실수 범위 인자에 일관되게 대응하고, 정수 수요 옵션(integer_demands)과 연동합니다(주6).

각주 표기 방법
- 본문 주석에 (주N) 꼴 번호를 달았고, 파일 맨 아래 "각주 모음"에서 자세히 설명합니다.
"""

from __future__ import annotations

import math
import random
from typing import Optional, Tuple, List

OBJECTIVE_FUNCTIONS = ['T', 'D']  # 'T': 총 지연(Tardiness), 'D': 총 이동거리(Distance)
OBJECTIVE_FUNC = 'T'
VEHICLE_SPEEDS = [5, 30, 60]  # km/h 후보(주7)

# ------------------------- Distance helpers -------------------------

def _euclid_km(p: Tuple[float, float], q: Tuple[float, float]) -> float:
    return math.hypot(q[0] - p[0], q[1] - p[1])


def get_dist(p: Tuple[float, float], q: Tuple[float, float]) -> float:
    """(x,y) 간 유클리드 거리(km). 좌표 None이 있으면 예외(주1)."""
    if None in p or None in q:
        raise ValueError("All locations must be set (x,y)")
    return _euclid_km(p, q)


# ------------------------------ Customer ------------------------------
class Customer:
    def __init__(self, _id: int):
        # Parameters given by User
        self.ID = _id  # 고유 식별자
        self.tw = [0.0, float("inf")]  # [ready, due]
        self.loc = [None, None]  # [x, y]
        self.serv_time = 0.0  # 서비스 시간(기본 0)(주2)
        self.weight = 0.0     # 수요(demand)
        # Variables changed over time during routing
        self.complete = False
        self.start = -1.0  # 작업 시작 시간
        self.end = -1.0    # 작업 종료 시간
        self.tardy = -1.0
        self.assigned_vhc = -1  # Vehicle ID 또는 참조. 기본은 -1(미배정)
        self.priority = 0

    def __repr__(self):
        return f"Customer {self.ID} at ({self.loc[0]}, {self.loc[1]}) with TW [{self.tw[0]}-{self.tw[1]}]"

    def __eq__(self, other):
        return isinstance(other, Customer) and (other.ID == self.ID)


# ------------------------------ Vehicle ------------------------------
class Vehicle:
    def __init__(self, _id: int):
        self.ID = _id
        self.available = 0.0  # 다음 작업 가능 시간
        self.loc = [None, None]   # 디포 좌표 [x, y]
        self.speed = 30.0         # km/h (주7)
        self.capacity = float("inf")  # 적재 용량(기본 무한)
        self.schedules: List[Customer] = []
        self.priority = 0
        self.now_loc = [None, None]
        self.now_capacity = 0.0

    def __repr__(self):
        return f"Vehicle {self.ID} with {len(self.schedules)} customers"

    def process(self, target: Customer):
        """차량 경로에 고객 한 명 추가. 시간/지연/적재 갱신(주3)."""
        # 사전 방어
        if None in self.now_loc or None in target.loc:
            raise ValueError("Vehicle/Customer location must be set before process().")
        if not math.isfinite(self.speed) or self.speed <= 0:
            raise ValueError("Vehicle speed must be positive and finite.")
        if target.serv_time is None:
            target.serv_time = 0.0
        if self.capacity is None:
            self.capacity = float("inf")

        # 적재 한계 확인
        if self.now_capacity + float(target.weight or 0.0) > float(self.capacity):
            raise ValueError('Capacity exceeds vehicle capacity.')

        # 스케줄링 계산
        travel_h = get_dist(tuple(self.now_loc), tuple(target.loc)) / self.speed
        start = max(float(target.tw[0]), self.available + travel_h)
        end = start + float(target.serv_time)
        tard = max(0.0, end - float(target.tw[1]))

        # 상태 갱신
        target.assigned_vhc = self.ID  # Vehicle 객체 대신 ID로 기록(주8)
        target.start = start
        target.end = end
        target.tardy = tard
        target.complete = True

        self.available = end
        self.now_loc = list(target.loc)
        self.now_capacity += float(target.weight or 0.0)
        self.schedules.append(target)


# ------------------------------ Instance ------------------------------
class Instance:
    type = f"VRP with TW to minimize {OBJECTIVE_FUNC}"

    def __init__(self, customers: List[Customer], vehicles: List[Vehicle]):
        self.customers = customers
        self.vehicles = vehicles
        self.objective = OBJECTIVE_FUNC

    def reset(self):
        """모든 고객/차량을 초기 상태로 복귀(주9)."""
        for customer in self.customers:
            customer.complete = False
            customer.start = -1.0
            customer.end = -1.0
            customer.tardy = -1.0
            customer.assigned_vhc = -1
            customer.priority = 0
        for vehicle in self.vehicles:
            vehicle.schedules = []
            vehicle.priority = 0
            vehicle.available = 0.0
            vehicle.now_loc = list(vehicle.loc)
            vehicle.now_capacity = 0.0


# ------------------------------ Solution ------------------------------
class Solution:
    def __init__(self, _alg: str, instance: Instance, obj: float):
        self.algorithm = _alg
        self.instance = instance
        self.objective = obj
        self.comp_time = 'None'
        self.status = 'None'

    def __repr__(self):
        return f"Schedule obtained by {self.algorithm} - Objective: {self.objective} (Total CPU Time: {self.comp_time})"


# ------------------------------ Random Instance Generator ------------------------------

def make_random_instance(
    num_customers: int,
    num_vehicles: int,
    seed: Optional[int] = None,
    hetero_vhc: bool = True,
    loc_range: Tuple[float, float] = (0.0, 100.0),      # 좌표 범위 (x,y)
    time_window: Tuple[float, float] = (0.0, 20.0),     # due 상한 범위
    service_time_range: Tuple[float, float] = (1.0, 10.0),  # 서비스 시간
    demand_range: Tuple[float, float] = (1.0, 10.0),        # 고객 수요 범위(주10)
    capacity_range: Tuple[float, float] = (30.0, 60.0),     # 차량 용량 범위(주10)
    load_factor: float = 0.7,             # 총용량 대비 목표 사용률 (feasibility 조정)
    integer_demands: bool = True          # 수요를 정수로 만들지 여부
) -> Instance:
    """무작위 VRPTW 인스턴스를 생성합니다(주4)."""
    if seed is not None:
        random.seed(seed)

    # --- Customers with demand (weight) ---
    customers: List[Customer] = []
    for i in range(num_customers):
        c = Customer(i)
        # 위치
        c.loc = [round(random.uniform(*loc_range), 2),
                 round(random.uniform(*loc_range), 2)]
        # 서비스 시간
        c.serv_time = float(random.uniform(*service_time_range))
        # 시간창: ready는 하한~중간, due는 (ready+service)~상한
        ready = random.uniform(time_window[0], (time_window[0] + time_window[1]) / 2.0)
        due = random.uniform(ready + c.serv_time, time_window[1])
        c.tw = [float(ready), float(due)]
        # 수요(demand)
        if integer_demands:
            c.weight = float(max(1, int(round(random.uniform(*demand_range)))))
        else:
            c.weight = float(random.uniform(*demand_range))
        customers.append(c)

    # --- Vehicles with capacities ---
    vehicles: List[Vehicle] = []
    # 동질 차량일 때 기준 속도/용량을 먼저 뽑아둠
    base_speed = float(random.choice(VEHICLE_SPEEDS))
    base_capacity = float(random.uniform(*capacity_range))
    for k in range(num_vehicles):
        v = Vehicle(k)
        v.loc = [round(random.uniform(*loc_range), 2),
                 round(random.uniform(*loc_range), 2)]
        v.now_loc = list(v.loc)
        v.available = 0.0
        v.now_capacity = 0.0
        if hetero_vhc:
            v.speed = float(random.choice(VEHICLE_SPEEDS))
            v.capacity = float(random.uniform(*capacity_range))
        else:
            v.speed = base_speed
            v.capacity = base_capacity
        vehicles.append(v)

    # --- Feasibility adjustment: scale demands to not exceed capacity * load_factor ---
    total_demand = sum(float(c.weight) for c in customers)
    total_capacity = sum(float(v.capacity) for v in vehicles)

    cap_budget = max(1e-9, total_capacity * min(1.0, max(0.0, load_factor)))  # 목표 총수요 한도(주11)
    if total_demand > cap_budget:
        scale = cap_budget / total_demand
        for c in customers:
            c.weight = float(c.weight) * scale

    if integer_demands:
        # 정수 반올림 → 총수요 초과 시 한 번 더 미세 스케일링(보수적)(주12)
        for c in customers:
            c.weight = float(max(1, int(round(c.weight))))
        total_demand = sum(float(c.weight) for c in customers)
        if total_demand > cap_budget:
            scale = cap_budget / total_demand
            # 바닥을 치되 최소 1 유지
            for c in customers:
                c.weight = float(max(1, int(math.floor(c.weight * scale))))

    return Instance(customers, vehicles)


# ============================== 각주 모음 ==============================
# (주1) 거리 단위 일관성: 좌표 단위가 km라고 가정. 속도는 km/h, 시간은 시간 단위로 일관되게 계산됩니다.
# (주2) 기본값: serv_time=0.0, weight=0.0로 초기화하여 연산 시 None 방지. tw는 [0, ∞] 기본.
# (주3) process(): get_dist/속도 기반 이동시간→start/end/tardy 갱신, 적재(now_capacity) 누적.
# (주4) make_random_instance(): 좌표/시간창/수요/용량을 무작위로 생성해 간단한 벤치마크를 만듭니다.
# (주5) 방어 코드: 좌표 None, 속도 0/비유한, 용량 None 등 상황에서 명시적 예외 또는 보정 처리.
# (주6) 정수 수요 옵션: integer_demands=True일 때 수요를 정수로 생성·보정합니다.
# (주7) 속도 후보: VEHICLE_SPEEDS는 예시. 필요 시 문제 특성에 맞춰 조정.
# (주8) assigned_vhc: Vehicle 객체 참조 대신 ID로 기록하여 직렬화/엑셀 저장 시 편의.
# (주9) reset(): 스케줄·시간·지연·적재 상태를 모두 초기화.
# (주10) 범위 형: demand_range/capacity_range는 (min, max) 실수 튜플. 정수 수요 생성 시 uniform→round 사용.
# (주11) load_factor: 총용량 대비 목표 사용률. 0.0~1.0 범위를 벗어나도 내부에서 클램핑.
# (주12) 정수 반올림 후 총수요가 cap_budget 초과하면 floor 재스케일링으로 보수적 조정.
