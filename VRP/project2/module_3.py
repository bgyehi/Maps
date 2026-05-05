import math
from typing import List, Tuple, Optional, Any
from copy import deepcopy


# VRP 핵심 데이터 구조 정의

class Customer:
    """VRP 고객 노드 정보."""

    def __init__(self, ID: int, loc: Tuple[float, float], weight: float, tw: Tuple[float, float], serv_time: float,
                 priority: float = 1.0):
        self.ID = ID  # 고객 ID
        self.loc = loc  # 위치 (x, y)
        self.weight = weight  # 수요량 (배송 무게/부피)
        self.tw = tw  # 시간창 (ready_time, due_time)
        self.serv_time = serv_time  # 서비스 처리 시간
        self.priority = priority  # 우선순위 (높을수록 중요)

        # 동적 상태 필드 (초기값 설정)
        self.assigned_vhc: Optional[int] = None
        self.start: float = 0.0  # 서비스 시작 시각
        self.end: float = 0.0  # 서비스 종료 시각
        self.tardy: float = 0.0  # 지연 시간
        self.complete: bool = False  # 서비스 완료 여부

    # Customer.reset() 메서드를 삭제하고, Instance.reset()에서 직접 속성을 초기화하도록 변경했습니다.
    # 이렇게 하면 pickle로 로드된 객체에 해당 메서드가 없더라도 충돌이 발생하지 않습니다.

    def __repr__(self):
        return f"C{self.ID}({self.loc[0]:.0f},{self.loc[1]:.0f}, W={self.weight}, TW={self.tw})"


class Vehicle:
    """VRP 차량 정보."""

    def __init__(self, ID: int, speed: float, capacity: float):
        self.ID = ID  # 차량 ID
        self.speed = speed  # 속도 (km/h)
        self.capacity = capacity  # 최대 적재 용량

        # 동적 상태 필드 (reset 시 초기화)
        self.now_loc: List[float] = [0.0, 0.0]  # 현재 위치 (창고 시작)
        self.available: float = 0.0  # 다음 서비스 시작 가능한 시각 (창고 출발 시간)
        self.now_capacity: float = 0.0  # 현재 적재량
        self.schedules: List[Customer] = []  # 배정된 고객 목록 (경로 순서대로)

    def reset(self):
        """차량 상태 초기화."""
        self.now_loc = [0.0, 0.0]  # 창고 (0,0) 가정
        self.available = 0.0  # 시작 시간 0 가정
        self.now_capacity = 0.0
        self.schedules = []

    def __repr__(self):
        return f"V{self.ID}(Spd={self.speed}, Cap={self.capacity}, Avail={self.available:.2f})"


class Instance:
    """VRP 문제 인스턴스 전체."""

    def __init__(self, name: str, customers: List[Customer], vehicles: List[Vehicle]):
        self.name = name
        self.customers = customers
        self.vehicles = vehicles

    def reset(self):
        """인스턴스에 속한 모든 차량과 고객 상태 초기화."""
        # 차량 상태 초기화
        for v in self.vehicles:
            v.reset()

        # 고객 상태 수동 초기화 (AttributeError 방지를 위해 직접 속성 설정)
        for c in self.customers:
            c.assigned_vhc = None
            c.start = 0.0
            c.end = 0.0
            c.tardy = 0.0
            c.complete = False

    def __repr__(self):
        return f"Instance('{self.name}', {len(self.customers)}C, {len(self.vehicles)}V)"


class Solution:
    """VRP 해(Solution) 객체."""

    def __init__(self, algorithm: str, instance: Instance, obj: float):
        self.algorithm = algorithm
        self.instance = instance
        self.objective = obj  # 목적 함수 값 (총 지연)
        self.comp_time: Any = "N/A"  # 계산 시간
        self.total_distance: float = 0.0
        self.unserved_ids: List[int] = []
        self.status: str = "PENDING"
        # 복사할 때 사용할 수 있도록 현재 경로 상태 저장 (deepcopy로 상태 복사)
        self.current_vehicle_states = deepcopy(instance.vehicles)
        self.current_customer_states = deepcopy(instance.customers)

    def __repr__(self):
        return f"Solution by {self.algorithm} - Obj: {self.objective:.3f} (Time: {self.comp_time:.3f}s)"

    def calculate_objective(self) -> float:
        """현재 상태의 총 지연을 계산합니다."""
        total_tardiness = sum(c.tardy for c in self.instance.customers)
        return total_tardiness

    def save_state(self):
        """현재 인스턴스 상태를 Solution 객체에 저장합니다."""
        self.current_vehicle_states = deepcopy(self.instance.vehicles)
        self.current_customer_states = deepcopy(self.instance.customers)

    def restore_state(self):
        """저장된 상태를 인스턴스에 복원합니다."""
        self.instance.vehicles = deepcopy(self.current_vehicle_states)
        self.instance.customers = deepcopy(self.current_customer_states)


def get_dist(loc1: Tuple[float, float], loc2: Tuple[float, float]) -> float:
    """유클리드 거리 계산 (km 가정)."""
    return math.hypot(loc1[0] - loc2[0], loc1[1] - loc2[1])


def recalculate_route_metrics(vehicle: Vehicle, instance: Instance) -> float:
    """
    주어진 차량의 경로(v.schedules)를 기반으로
    모든 시간(start, end, tardiness)을 재계산하고 총 이동 거리를 반환합니다.
    """
    route = vehicle.schedules
    if not route:
        vehicle.reset()  # 경로가 비었으면 초기화
        return 0.0

    # 1. 초기 상태 설정 (창고 출발)
    current_time = 0.0
    # 안전한 창고 위치 접근
    depot_loc = getattr(instance, 'depot_loc', (0.0, 0.0))
    current_loc = depot_loc
    total_distance = 0.0
    total_weight = 0.0

    # 2. 경로 순회하며 시간 및 용량 계산
    for customer in route:
        # a. 이동 시간 및 거리
        travel_km = get_dist(current_loc, tuple(customer.loc))
        travel_h = travel_km / max(1e-9, float(vehicle.speed or 1.0))

        # b. 도착 시각 (arrival_time)
        arrival_time = current_time + travel_h

        # c. 서비스 시작 시각 (wait time 고려)
        customer.start = max(float(customer.tw[0]), arrival_time)

        # d. 서비스 종료 시각
        customer.end = customer.start + float(customer.serv_time or 0.0)

        # e. 지연 시간 (tardiness)
        customer.tardy = max(0.0, customer.end - float(customer.tw[1]))
        customer.complete = True

        # f. 차량 상태 업데이트
        current_time = customer.end
        current_loc = tuple(customer.loc)
        total_distance += travel_km
        total_weight += float(customer.weight or 0.0)

    # 3. 창고 복귀
    return_dist = get_dist(current_loc, depot_loc)
    total_distance += return_dist

    # 4. 차량 동적 필드 최종 업데이트
    vehicle.available = current_time  # 마지막 고객 서비스 종료 시각
    vehicle.now_capacity = total_weight

    return total_distance