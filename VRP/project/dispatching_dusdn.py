"""
[수정됨] VRP 해법 핵심 모듈 (초기해 생성 + 해 개선)

(수정: main.py의 ILS가 호출할 '안전한 교란' 함수 추가)
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import random
import time

import module
from module import Customer, Vehicle, Instance, Solution, get_dist


# === 1. (신규) VNS/ILS용 '상태 없는(Stateless)' 비용 계산 함수 ===
def _calculate_route_cost_and_update_customer_info(
        vehicle: 'Vehicle',
        route: List['Customer']
) -> float:
    """
    (중요) 이 함수는 v.now_capacity나 v.available을 '읽지' 않습니다.
    오직 입력받은 'route' (고객 리스트)만을 기반으로 비용을 계산합니다.
    계산 중 용량 초과 시 'inf'를 반환합니다.
    (모든 개선 로직의 '심판' 역할을 합니다.)
    """
    depot_loc = tuple(vehicle.loc)
    speed = float(vehicle.speed or 30.0)
    capacity = float(vehicle.capacity or float('inf'))

    current_time = 0.0
    current_loc = depot_loc
    total_tardiness = 0.0
    current_load = 0.0

    for customer in route:
        # 1. 용량 체크
        current_load += float(customer.weight or 0.0)
        if current_load > capacity + 1e-9:
            return float('inf')  # 용량 초과

        # 2. 시간 계산
        cust_loc = tuple(customer.loc)
        travel_km = get_dist(current_loc, cust_loc)
        travel_h = travel_km / max(1e-9, speed)
        arrival_time = current_time + travel_h

        ready_time = float(customer.tw[0])
        start_time = max(ready_time, arrival_time)

        service_time = float(customer.serv_time or 0.0)
        end_time = start_time + service_time

        due_time = float(customer.tw[1])
        tardiness = max(0.0, end_time - due_time)

        # 3. 고객 객체 상태 업데이트 (시각화를 위해)
        customer.start = start_time
        customer.end = end_time
        customer.tardy = tardiness

        total_tardiness += tardiness
        current_time = end_time
        current_loc = cust_loc

    return total_tardiness


def calculate_total_tardiness(vehicles: List['Vehicle']) -> float:
    """모든 차량의 '현재 schedules'를 기반으로 총 Tardiness를 계산합니다."""
    total_cost = 0.0
    for v in vehicles:
        # (수정) v.schedules (현재 경로)를 심판에게 넘겨서 비용 계산
        cost = _calculate_route_cost_and_update_customer_info(v, v.schedules)
        if cost == float('inf'):
            return float('inf')  # 한 대라도 용량 초과 시 'inf'
        total_cost += cost
    return total_cost


# === 2. (수정) '초기 해 생성' (Greedy Dispatching) ===

def _best_customer_for_vehicle_greedy(
        v: Vehicle,
        unserved_ids: List[int],
        cust_by_id: dict
) -> Tuple[Optional[int], Optional[Tuple]]:
    """
    (수정) GRASP(k=3) 제거.
    오직 '가장 좋은 고객 1명' (best_cid)만 찾아 반환하는 순수 Greedy 함수.
    """

    # (중요) v.now_capacity는 instance.reset()으로 0에서 시작하여 누적됨
    feasible_ids = [cid for cid in unserved_ids if
                    (float(v.now_capacity or 0.0) + float(cust_by_id[cid].weight or 0.0)) <= (
                                float(v.capacity or 0.0) + 1e-9)]

    if not feasible_ids:
        return None, None

    best_key = None
    best_cid = None
    best_metrics = None

    for cid in feasible_ids:
        c = cust_by_id[cid]
        travel_km = get_dist(tuple(v.now_loc), tuple(c.loc))
        travel_h = travel_km / max(1e-9, float(v.speed or 1.0))
        start = max(float(c.tw[0]), float(v.available or 0.0) + travel_h)
        end = start + float(c.serv_time or 0.0)
        tard = max(0.0, end - float(c.tw[1]))

        # (수정) 'delta_tardiness'와 'edd_then_nearest'로 고정 (Tardiness 최적화)
        primary = tard
        secondary = end
        tie1 = float(c.tw[1])
        tie2 = travel_km
        key = (primary, secondary, tie1, tie2, cid)

        if (best_key is None) or (key < best_key):
            best_key = key
            best_cid = cid
            best_metrics = (travel_km, travel_h, start, end, tard)

    return best_cid, best_metrics


def dispatch_earliest_vehicle_best_customer(instance: Instance) -> Solution:
    """
    (수정) 'randomness_k' 제거.
    'delta_tardiness'를 사용하는 순수 Greedy 디스패처로 복귀.
    """
    instance.reset()  # (중요) v.now_capacity, v.available 0으로 초기화
    customers: List[Customer] = instance.customers
    vehicles: List[Vehicle] = instance.vehicles

    cust_by_id = {c.ID: c for c in customers}
    unserved_ids = list(cust_by_id.keys())

    total_distance = 0.0
    total_tardiness = 0.0

    while unserved_ids:
        order = sorted(range(len(vehicles)), key=lambda kk: (float(vehicles[kk].available or 0.0), vehicles[kk].ID))

        selected = None
        chosen = None
        metrics = None

        for k_idx in order:
            v = vehicles[k_idx]

            # (수정) k=1 (Greedy)
            chosen_cid, chosen_metrics = _best_customer_for_vehicle_greedy(
                v, unserved_ids, cust_by_id
            )

            if chosen_cid is not None:
                selected = v
                chosen = chosen_cid
                metrics = chosen_metrics
                break

        if chosen is None:
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

        total_distance += travel_km
        total_tardiness += tard
        unserved_ids.remove(chosen)

    # (수정) 이 함수는 초기 해의 Tardiness를 반환
    # (버그 방지를 위해 '심판' 함수로 최종 검증)
    final_dispatch_tardiness = calculate_total_tardiness(vehicles)

    try:
        sol = Solution("Greedy_Dispatch", instance, obj=final_dispatch_tardiness)
    except TypeError:
        class _Sol:
            def __init__(self, alg, inst, obj):
                self.algorithm = alg;
                self.instance = inst;
                self.objective = obj
                self.comp_time = "N/A";
                self.status = "DONE"

            def __repr__(self):
                return f"Schedule by {self.algorithm} - Objective: {self.objective}"

        sol = _Sol("Greedy_Dispatch", instance, final_dispatch_tardiness)

    sol.total_distance = total_distance
    sol.unserved_ids = sorted(list(unserved_ids))

    if unserved_ids:
        sol.status = "INFEASIBLE_UNSERVED"
    else:
        sol.status = "DONE"

    return sol


# === 3. (신규) '해 개선' (VNS + Tabu Search) ===
# (main.py에서 이 함수들을 호출)

def _run_first_relocate_move(
        vehicles: List[Vehicle],
        current_total_tardiness: float,
        tabu_list: Dict[int, int],
        ils_iter: int,
        best_so_far_tard: float
) -> Tuple[bool, float]:
    """(Relocate) Tabu+Aspiration을 적용한 최초 개선 탐색"""

    vehicle_indices = list(range(len(vehicles)))
    random.shuffle(vehicle_indices)
    TABU_TENURE = 5

    for v_from_idx in vehicle_indices:
        v_from = vehicles[v_from_idx]
        if not v_from.schedules:
            continue

        for cust_idx_seq in range(len(v_from.schedules)):
            if cust_idx_seq >= len(v_from.schedules):
                break

            customer_to_move = v_from.schedules[cust_idx_seq]
            is_tabu = tabu_list.get(customer_to_move.ID, 0) > ils_iter

            v_from.schedules.pop(cust_idx_seq)

            for v_to_idx in vehicle_indices:
                v_to = vehicles[v_to_idx]

                for insert_pos in range(len(v_to.schedules) + 1):
                    v_to.schedules.insert(insert_pos, customer_to_move)

                    new_total_tard = calculate_total_tardiness(vehicles)

                    v_to.schedules.pop(insert_pos)

                    if new_total_tard < current_total_tardiness - 1e-5:
                        is_aspirated = new_total_tard < best_so_far_tard

                        if not is_tabu or is_aspirated:
                            v_to.schedules.insert(insert_pos, customer_to_move)
                            tabu_list[customer_to_move.ID] = ils_iter + TABU_TENURE
                            return True, new_total_tard

            v_from.schedules.insert(cust_idx_seq, customer_to_move)

    return False, current_total_tardiness


def _run_first_swap_move(
        vehicles: List[Vehicle],
        current_total_tardiness: float,
        tabu_list: Dict[int, int],
        ils_iter: int,
        best_so_far_tard: float
) -> Tuple[bool, float]:
    """(Swap) Tabu+Aspiration을 적용한 최초 개선 탐색"""

    vehicle_indices = list(range(len(vehicles)))
    random.shuffle(vehicle_indices)
    TABU_TENURE = 5

    for v_i_idx in vehicle_indices:
        v_i = vehicles[v_i_idx]
        if not v_i.schedules:
            continue

        for c_i_idx in range(len(v_i.schedules)):
            if c_i_idx >= len(v_i.schedules): break
            c_i = v_i.schedules[c_i_idx]
            is_tabu_i = tabu_list.get(c_i.ID, 0) > ils_iter

            for v_j_idx in vehicle_indices:
                v_j = vehicles[v_j_idx]
                if not v_j.schedules:
                    continue

                for c_j_idx in range(len(v_j.schedules)):
                    if c_j_idx >= len(v_j.schedules): break
                    c_j = v_j.schedules[c_j_idx]

                    if v_i.ID == v_j.ID and (c_i_idx == c_j_idx):
                        continue

                    is_tabu_j = tabu_list.get(c_j.ID, 0) > ils_iter

                    v_i.schedules[c_i_idx] = c_j
                    v_j.schedules[c_j_idx] = c_i

                    new_total_tard = calculate_total_tardiness(vehicles)

                    if new_total_tard < current_total_tardiness - 1e-5:
                        is_aspirated = new_total_tard < best_so_far_tard

                        if (not is_tabu_i and not is_tabu_j) or is_aspirated:
                            tabu_list[c_i.ID] = ils_iter + TABU_TENURE
                            tabu_list[c_j.ID] = ils_iter + TABU_TENURE
                            return True, new_total_tard

                    v_i.schedules[c_i_idx] = c_i
                    v_j.schedules[c_j_idx] = c_j

    return False, current_total_tardiness


def _run_first_2opt_move(
        vehicles: List[Vehicle],
        current_total_tardiness: float,
        tabu_list: Dict[int, int],
        ils_iter: int,
        best_so_far_tard: float
) -> Tuple[bool, float]:
    """(2-opt) Tabu+Aspiration을 적용한 최초 개선 탐색 (Inter/Intra 통합)"""

    vehicle_indices = list(range(len(vehicles)))
    random.shuffle(vehicle_indices)
    TABU_TENURE = 5

    for v_i_idx in vehicle_indices:
        v_i = vehicles[v_i_idx]
        # (수정) 2-opt는 intra-route는 3명, inter-route는 2명부터 가능
        if len(v_i.schedules) < 2:
            continue

        for v_j_idx in vehicle_indices:
            v_j = vehicles[v_j_idx]
            if len(v_j.schedules) < 2:
                continue

            for i in range(len(v_i.schedules) - 1):
                c_i = v_i.schedules[i]
                c_i_next = v_i.schedules[i + 1]
                is_tabu_i = tabu_list.get(c_i.ID, 0) > ils_iter or tabu_list.get(c_i_next.ID, 0) > ils_iter

                for j in range(len(v_j.schedules) - 1):
                    c_j = v_j.schedules[j]
                    c_j_next = v_j.schedules[j + 1]
                    is_tabu_j = tabu_list.get(c_j.ID, 0) > ils_iter or tabu_list.get(c_j_next.ID, 0) > ils_iter

                    # (수정) 시뮬레이션 전에 '항상' 원본 경로를 저장
                    original_route_i = list(v_i.schedules)
                    original_route_j = list(v_j.schedules)  # v_i와 v_j가 같아도 일단 저장

                    if v_i.ID == v_j.ID:
                        if i >= j or (i + 1) == j:  # 겹치거나 순서가 맞지 않으면 스킵
                            continue

                        # (수정) Intra-route 2-opt: [i+1]...[j] 구간을 뒤집음
                        segment_to_reverse = original_route_i[i + 1: j + 1]
                        segment_to_reverse.reverse()
                        v_i.schedules = original_route_i[:i + 1] + segment_to_reverse + original_route_i[j + 1:]

                    else:
                        # Inter-route (다른 차량) 2-opt (꼬리 교환)
                        if (i == (len(original_route_i) - 2)) and (j == (len(original_route_j) - 2)):
                            # (버그 방지) 두 경로의 마지막 엣지를 교환하면 (꼬리 전체)
                            # 2-opt가 아닌 단순 Relocate와 같아져, Tabu 로직에 혼선
                            continue

                        v_i.schedules = original_route_i[:i + 1] + original_route_j[j + 1:]
                        v_j.schedules = original_route_j[:j + 1] + original_route_i[i + 1:]

                    new_total_tard = calculate_total_tardiness(vehicles)

                    if new_total_tard < current_total_tardiness - 1e-5:
                        is_aspirated = new_total_tard < best_so_far_tard

                        if (not is_tabu_i and not is_tabu_j) or is_aspirated:
                            tabu_list[c_i.ID] = ils_iter + TABU_TENURE
                            tabu_list[c_i_next.ID] = ils_iter + TABU_TENURE
                            tabu_list[c_j.ID] = ils_iter + TABU_TENURE
                            tabu_list[c_j_next.ID] = ils_iter + TABU_TENURE
                            return True, new_total_tard

                    # (원상 복구)
                    v_i.schedules = original_route_i
                    v_j.schedules = original_route_j  # (수정) v_j도 항상 복구

    return False, current_total_tardiness


def run_vns_improvement(solution: 'Solution', time_limit_sec: float) -> float:
    """
    (VNS+TS) Tabu List를 참조하여 Relocate -> Swap -> 2-opt 순서로 탐색
    """
    vehicles: List['Vehicle'] = solution.instance.vehicles
    start_vns_time = time.time()

    tabu_list: Dict[int, int] = {}

    print(f"  [VNS] VNS (Relocate+Swap+2opt+TS) 시작...")

    current_total_tardiness = calculate_total_tardiness(vehicles)
    best_so_far_tard = current_total_tardiness

    iteration = 0
    while True:
        elapsed = time.time() - start_vns_time
        if elapsed > time_limit_sec:
            break

        iteration += 1

        print(
            f"    [VNS Iter {iteration}] Time: {elapsed:.2f}s / {time_limit_sec:.2f}s. Current Best: {best_so_far_tard:.3f}",
            end='\r')

        # 1. [Neighborhood 1] Relocate 시도
        relocate_improved, tard_after_relocate = _run_first_relocate_move(
            vehicles, current_total_tardiness, tabu_list, iteration, best_so_far_tard
        )
        if relocate_improved:
            current_total_tardiness = tard_after_relocate
            if current_total_tardiness < best_so_far_tard:
                best_so_far_tard = current_total_tardiness
                print(f"\n    [VNS Iter {iteration}] New Best Found! (Relocate) Tardiness: {best_so_far_tard:.3f}")
            continue

            # 2. [Neighborhood 2] Swap 시도
        swap_improved, tard_after_swap = _run_first_swap_move(
            vehicles, current_total_tardiness, tabu_list, iteration, best_so_far_tard
        )
        if swap_improved:
            current_total_tardiness = tard_after_swap
            if current_total_tardiness < best_so_far_tard:
                best_so_far_tard = current_total_tardiness
                print(f"\n    [VNS Iter {iteration}] New Best Found! (Swap) Tardiness: {best_so_far_tard:.3f}")
            continue

            # 3. [Neighborhood 3] 2-opt 시도
        two_opt_improved, tard_after_2opt = _run_first_2opt_move(
            vehicles, current_total_tardiness, tabu_list, iteration, best_so_far_tard
        )
        if two_opt_improved:
            current_total_tardiness = tard_after_2opt
            if current_total_tardiness < best_so_far_tard:
                best_so_far_tard = current_total_tardiness
                print(f"\n    [VNS Iter {iteration}] New Best Found! (2-opt) Tardiness: {best_so_far_tard:.3f}")
            continue

        # 4. 모든 탐색 실패 시 Local Optimum -> 종료
        break

    print(f"\n  [VNS] Finished. Total Iterations: {iteration}. Final Best Tardiness: {best_so_far_tard:.3f}")

    return best_so_far_tard


# === 4. (신규) '안전한 교란' (Random Move) ===
def perturb_solution(solution: 'Solution', strength: int = 3):
    """
    (신규) '랜덤 Relocate'를 'strength' 횟수만큼 강제 수행합니다.
    이 방식은 고객 유실(Unserved) 버그가 100% 없습니다.
    """
    instance: 'Instance' = solution.instance
    vehicles: List['Vehicle'] = instance.vehicles

    # 교란 대상이 될 고객 리스트
    movable_customers = []
    for v in vehicles:
        if len(v.schedules) > 0:
            movable_customers.append(v)

    if not movable_customers:
        return  # 교란할 고객이 없음

    for _ in range(strength):
        try:
            # 1. 랜덤한 (비어있지 않은) 차량에서 고객 뽑기
            v_from = random.choice(movable_customers)
            if not v_from.schedules:  # 방어 코드
                continue
            cust_idx = random.randrange(len(v_from.schedules))
            customer_to_move = v_from.schedules.pop(cust_idx)

            # 2. 랜덤한 차량(본인 포함)의 랜덤한 위치에 삽입
            v_to = random.choice(vehicles)
            insert_pos = random.randint(0, len(v_to.schedules))
            v_to.schedules.insert(insert_pos, customer_to_move)

        except (ValueError, IndexError):
            continue