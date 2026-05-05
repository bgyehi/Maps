import random
import time
from copy import deepcopy
from typing import List, Tuple, Dict, Any, Optional

from module_3 import Instance, Solution, Vehicle, Customer, get_dist, recalculate_route_metrics


# --- 헬퍼 함수 ---

def evaluate_solution(solution: Solution, instance: Instance):
    """
    차량 경로를 기반으로 모든 고객의 메트릭을 재계산하고 총 지연 및 거리를 업데이트합니다.
    """
    total_tardiness = 0.0
    total_distance = 0.0

    # 1. 모든 차량에 대해 경로 재계산 (시간, 용량, 지연 업데이트)
    for v in instance.vehicles:
        total_distance += recalculate_route_metrics(v, instance)

    # 2. 총 지연 계산 (모든 고객의 tardiness 합)
    for c in instance.customers:
        # 안전을 위해 c.tardy 속성이 없으면 0.0을 사용
        total_tardiness += getattr(c, 'tardy', 0.0)

    solution.objective = total_tardiness
    solution.total_distance = total_distance
    solution.status = "FEASIBLE" if total_tardiness < float('inf') else "INFEASIBLE"
    return solution.objective


# --- 초기 해 생성 (Greedy Dispatching) ---

def dispatch_earliest_vehicle_best_customer(
        instance: Instance,
) -> Solution:
    """
    차량이 가장 빨리 사용 가능해지는 시점에 가장 좋은 고객을 할당하는 탐욕적 휴리스틱.
    """
    # instance.reset()는 main.py에서 이미 호출되었거나, 이 함수 시작 전에 호출되어야 합니다.
    # 안전을 위해 여기서 차량만 초기화 (고객은 Instance.reset()에 의존)
    for v in instance.vehicles:
        v.reset()

    vehicles = instance.vehicles
    unassigned_customers = list(instance.customers)

    # 차량의 초기 위치 설정 (0,0 가정)
    depot_loc = getattr(instance, 'depot_loc', (0.0, 0.0))
    for v in vehicles:
        v.now_loc = list(depot_loc)
        v.available = 0.0

    while unassigned_customers:
        best_assignment = None
        best_cost_overall = float('inf')

        # 1. 고객 순회: 이번 라운드에 할당할 최적 고객 C* 선정
        for customer in unassigned_customers:

            # 2. 차량 순회: C*를 배정할 최적 차량 V* 선정 (경로 끝 삽입만 고려)
            for vehicle in vehicles:

                # 용량 제약 확인
                if vehicle.now_capacity + customer.weight > vehicle.capacity:
                    continue

                # 임시 삽입 (경로 끝)
                vehicle.schedules.append(customer)

                # 삽입 후 메트릭 재계산 및 비용 확인 (총 지연)
                recalculate_route_metrics(vehicle, instance)
                cost_after_insertion = sum(getattr(c, 'tardy', 0.0) for c in vehicle.schedules)

                # 임시 제거
                vehicle.schedules.pop()

                # (중요) 임시 제거 후 메트릭 재계산 필요 (경로가 변경되었으므로)
                # 이 로직은 복잡해지므로, 현재 라운드에서 가장 좋은 단일 삽입을 찾고
                # 그 삽입만 최종적으로 수행하는 방식으로 단순화합니다.

                # 현재 최적 할당 갱신
                if cost_after_insertion < best_cost_overall:
                    best_cost_overall = cost_after_insertion
                    best_assignment = (customer, vehicle)

        # 3. 최적 할당 수행
        if best_assignment:
            customer_to_assign, vehicle_to_assign = best_assignment

            # 최종 할당 (경로 끝)
            vehicle_to_assign.schedules.append(customer_to_assign)
            vehicle_to_assign.now_capacity += customer_to_assign.weight
            unassigned_customers.remove(customer_to_assign)

            # 경로 재계산 및 차량 상태 업데이트 (이 시점에서 고객 상태도 업데이트됨)
            recalculate_route_metrics(vehicle_to_assign, instance)

        else:
            # 더 이상 할당할 수 없는 고객이 있다면
            # print(f"경고: {len(unassigned_customers)}명의 고객을 할당할 수 없습니다 (용량/시간 제약).")
            break

    # 4. 최종 솔루션 평가
    solution = Solution("GreedyDispatch", instance, obj=0.0)
    # 모든 차량에 대해 최종 평가를 수행하여 모든 고객의 상태가 확정되도록 합니다.
    evaluate_solution(solution, instance)
    return solution


# --- Local Search Operators ---

def apply_swap(instance: Instance, vehicle: Vehicle) -> Optional[float]:
    """단일 차량 내 2-Swap 오퍼레이터."""
    best_cost = sum(getattr(c, 'tardy', 0.0) for c in vehicle.schedules)
    best_swap: Optional[Tuple[int, int]] = None

    route = vehicle.schedules
    if len(route) < 2: return None

    # 현재 상태를 저장하고 개선된 경우에만 최종적으로 업데이트하도록 합니다.
    original_route = list(route)

    for i in range(len(route)):
        for j in range(i + 1, len(route)):
            # Swap
            route[i], route[j] = route[j], route[i]

            # Evaluate
            recalculate_route_metrics(vehicle, instance)
            new_cost = sum(getattr(c, 'tardy', 0.0) for c in vehicle.schedules)

            if new_cost < best_cost:
                best_cost = new_cost
                best_swap = (i, j)

            # Revert (개선되지 않았거나, 더 나은 스왑을 찾기 위해 원본 복구)
            route[i], route[j] = route[j], route[i]

    if best_swap:
        i, j = best_swap
        # 최적의 스왑만 최종 적용
        route[i], route[j] = route[j], route[i]
        recalculate_route_metrics(vehicle, instance)  # 최종 메트릭 재계산
        return best_cost

    # 개선되지 않았으므로 차량 경로는 원본(original_route)과 동일해야 합니다.
    return None


def apply_2opt(instance: Instance, vehicle: Vehicle) -> Optional[float]:
    """단일 차량 내 2-opt 오퍼레이터."""
    best_cost = sum(getattr(c, 'tardy', 0.0) for c in vehicle.schedules)
    best_segment: Optional[Tuple[int, int]] = None

    route = vehicle.schedules
    n = len(route)
    if n < 4: return None

    original_route = list(route)

    for i in range(n - 1):
        for j in range(i + 1, n):
            # 2-opt: Reverse segment [i+1, j]
            # new_route = route[:i+1] + route[i+1:j+1][::-1] + route[j+1:]

            # Segment Reversal
            temp_segment = route[i + 1:j + 1]
            route[i + 1:j + 1] = temp_segment[::-1]  # Reverse in place

            # Evaluate
            recalculate_route_metrics(vehicle, instance)
            new_cost = sum(getattr(c, 'tardy', 0.0) for c in vehicle.schedules)

            if new_cost < best_cost:
                best_cost = new_cost
                best_segment = (i, j)
            else:
                # Revert (원래 상태로 되돌림)
                route[i + 1:j + 1] = temp_segment[::-1]  # 다시 뒤집어서 원본으로 복구

    if best_segment:
        # 최적의 변경은 이미 적용된 상태이거나, 루프를 벗어나면 최종 적용됨
        recalculate_route_metrics(vehicle, instance)  # 최종 메트릭 재계산
        return best_cost

    # 개선되지 않았으면, 차량 경로는 원본과 동일해야 합니다.
    return None


# --- Local Search Engine ---

def run_local_search(instance: Instance, max_iter: int = 50) -> Solution:
    """
    모든 차량에 대해 반복적으로 Local Search 오퍼레이터를 적용합니다.
    """
    vehicles = instance.vehicles

    # 초기 비용 설정
    current_sol = Solution("LS_Temp", instance, obj=0.0)
    current_cost = evaluate_solution(current_sol, instance)

    for _ in range(max_iter):
        improved = False

        # 차량 순회하며 Swap 및 2-opt 적용
        for v in vehicles:
            # 1. Swap 시도
            swap_cost = apply_swap(instance, v)
            if swap_cost is not None and swap_cost < current_cost:
                current_cost = swap_cost
                improved = True

            # 2. 2-opt 시도
            opt2_cost = apply_2opt(instance, v)
            if opt2_cost is not None and opt2_cost < current_cost:
                current_cost = opt2_cost
                improved = True

            # 개선되었다면 다시 처음부터 다른 차량도 시도 (Best-improvement search)
            if improved:
                break

        # 개선이 없으면 종료 (First-improvement 또는 Best-improvement 후)
        if not improved:
            break

    # 최종 솔루션 상태 업데이트 및 반환
    final_sol = Solution("LS_Final", instance, obj=0.0)
    evaluate_solution(final_sol, instance)
    return final_sol


# --- 메타휴리스틱 (ILS) ---

def perturb_solution(solution: Solution, strength: int = 2):
    """
    현재 해를 무작위로 교란(Perturb)하는 오퍼레이터 (무작위 Swap).
    """
    instance = solution.instance
    vehicles = instance.vehicles

    for _ in range(strength):
        # 1. 무작위 차량 선택
        v = random.choice(vehicles)

        # 2. 무작위 고객 두 명 선택 후 Swap
        if len(v.schedules) >= 2:
            i, j = random.sample(range(len(v.schedules)), 2)
            v.schedules[i], v.schedules[j] = v.schedules[j], v.schedules[i]

            # 메트릭 재계산
            recalculate_route_metrics(v, instance)


def iterated_local_search(
        initial_sol: Solution,
        max_ils_iterations: int = 10,
        max_ls_iter: int = 50,
        perturbation_strength: int = 2
) -> Tuple[Solution, float]:
    """
    반복 지역 탐색 (ILS) 메타휴리스틱.
    """
    start_time = time.time()

    # 1. 초기 해 (best_sol) 설정 및 상태 저장
    best_sol = deepcopy(initial_sol)
    best_sol.save_state()
    current_sol = deepcopy(initial_sol)

    # 2. 초기 지역 탐색 적용
    ls_result = run_local_search(current_sol.instance, max_iter=max_ls_iter)
    current_sol = ls_result

    if current_sol.objective < best_sol.objective:
        best_sol = deepcopy(current_sol)
        best_sol.save_state()

    # 3. ILS 반복
    for _ in range(max_ils_iterations):
        # a. Perturbation (교란)
        # deepcopy를 통해 상태 복원 후 교란
        perturb_sol = deepcopy(best_sol)
        perturb_sol.restore_state()
        perturb_solution(perturb_sol, strength=perturbation_strength)

        # b. Local Search (지역 탐색)
        ls_result = run_local_search(perturb_sol.instance, max_iter=max_ls_iter)
        current_sol = ls_result

        # c. Acceptance Criteria (수락 기준)
        if current_sol.objective < best_sol.objective:
            # 개선되면 수락 및 상태 저장
            best_sol = deepcopy(current_sol)
            best_sol.save_state()

    end_time = time.time()
    total_comp_time = end_time - start_time

    # 최종적으로 최적 상태를 인스턴스에 복원
    best_sol.restore_state()
    evaluate_solution(best_sol, best_sol.instance)
    best_sol.comp_time = total_comp_time

    return best_sol, best_sol.comp_time