import pickle
import time
from module import get_dist


# ============================================================
# 1) Instance 로드 유틸 (다른 곳에서도 쓸 수 있게)
# ============================================================
def load_instance(filepath):
    with open(filepath, "rb") as f:
        instance = pickle.load(f)
    return instance


# ============================================================
# 2) objective 값 추출 유틸
# ============================================================
def get_objective_value(solution):
    """
    solution.obj 또는 solution.objective 중 존재하는 값을 가져옴.
    """
    val = getattr(solution, "obj", None)
    if val is None:
        val = getattr(solution, "objective", 0.0)
    return float(val)


# ============================================================
# 3) 차량 경로(tardy, dist) 평가 함수
# ============================================================
def eval_vehicle_route_tardy_dist(vehicle, route_customers):
    """
    vehicle: Vehicle 객체
    route_customers: 방문 순서 리스트 (Customer 객체 리스트)
    return: (tardiness 합, distance 합)

    시작 위치는 vehicle.loc 기준.
    (필요하면 vehicle.start_loc / instance.depot 등으로 확장 가능)
    """
    available = 0.0
    now_loc = vehicle.loc[:]  # 일반적으로 depot에서 시작한다고 가정
    now_capacity = 0.0
    total_tardy = 0.0
    total_dist = 0.0
    cap = float(vehicle.capacity or 0.0)

    for c in route_customers:
        travel_km = get_dist(tuple(now_loc), tuple(c.loc))
        travel_h = travel_km / max(1e-9, float(vehicle.speed or 1.0))

        # 도착 후 대기/시작 시각
        start = max(float(c.tw[0]), available + travel_h)
        end = start + float(c.serv_time or 0.0)
        tard = max(0.0, end - float(c.tw[1]))

        need = float(c.weight or 0.0)
        if (now_capacity + need) > cap + 1e-9:
            # 용량 위반 시, 불가능한 해로 간주
            return float("inf"), float("inf")

        total_tardy += tard
        total_dist += travel_km

        now_loc = list(c.loc)
        available = end
        now_capacity += need

    return total_tardy, total_dist


# ============================================================
# 4) 차량별 tardiness-aware 2-opt
# ============================================================
def two_opt_for_vehicle_tardy(vehicle):
    """
    현재 vehicle.schedules 순서를 기반으로,
    tardiness 최소화(동률 시 거리 최소) 2-opt를 수행.
    """
    route = vehicle.schedules[:]
    n = len(route)
    if n < 4:
        # 방문 고객이 너무 적으면 2-opt 할 게 없음
        return route

    best_tardy, best_dist = eval_vehicle_route_tardy_dist(vehicle, route)
    best_route = route[:]

    improved = True
    while improved:
        improved = False
        # 2-opt 표준 루프 (1부터 n-2까지, i<k)
        for i in range(1, n - 2):
            for k in range(i + 1, n):
                candidate = (
                    best_route[:i]
                    + best_route[i:k+1][::-1]
                    + best_route[k+1:]
                )
                cand_tardy, cand_dist = eval_vehicle_route_tardy_dist(
                    vehicle, candidate
                )

                better = False
                # tardiness 우선
                if cand_tardy + 1e-9 < best_tardy:
                    better = True
                # tardiness 같으면 거리로 tie-break
                elif abs(cand_tardy - best_tardy) <= 1e-9 and cand_dist + 1e-9 < best_dist:
                    better = True

                if better:
                    best_route = candidate
                    best_tardy = cand_tardy
                    best_dist = cand_dist
                    improved = True
        # 더 이상 개선 없으면 while 루프 종료

    return best_route


# ============================================================
# 5) 1-0 relocate: 차량 간 고객 한 명 이동해서 tardiness 감소 시도
# ============================================================
def improve_routes_by_relocate_tardy(instance, routes_dict, max_outer_iter=100):
    """
    routes_dict: {vehicle.ID: [Customer, ...]} 형태의 현재 라우트 집합

    1-0 relocate (한 고객을 다른 차량의 라우트로 옮기는 move)를
    first-improvement 방식으로 반복 적용하여
    총 tardiness를 줄인다.
    """
    # vehicle ID → Vehicle 객체 맵
    veh_map = {v.ID: v for v in instance.vehicles}

    outer_iter = 0
    improved = True

    while improved and outer_iter < max_outer_iter:
        outer_iter += 1
        improved = False

        # 현재 각 차량의 tardiness 및 전체 tardiness 계산
        per_v_tardy = {}
        current_total_tardy = 0.0
        infeasible = False

        for v in instance.vehicles:
            route = routes_dict[v.ID]
            tard, _ = eval_vehicle_route_tardy_dist(v, route)
            if tard == float("inf"):
                infeasible = True
                break
            per_v_tardy[v.ID] = tard
            current_total_tardy += tard

        if infeasible:
            # 현재 routes_dict 자체가 말이 안 되면 그냥 종료
            break

        # --- 모든 (from vehicle, to vehicle, 고객, 삽입 위치)에 대해 개선 move 탐색 ---
        for v_from in instance.vehicles:
            route_from = routes_dict[v_from.ID]
            if len(route_from) == 0:
                continue

            orig_from_tard = per_v_tardy[v_from.ID]

            # route_from의 각 고객을 하나씩 꺼내서 다른 차량에 보내보기
            for idx_c, cust in enumerate(route_from):
                # v_from에서 해당 고객 제거한 경로
                route_from_wo = route_from[:idx_c] + route_from[idx_c+1:]

                # 제거 후 tardiness
                tard_from_wo, _ = eval_vehicle_route_tardy_dist(v_from, route_from_wo)
                if tard_from_wo == float("inf"):
                    continue

                for v_to in instance.vehicles:
                    # 같은 차량 안으로의 이동은 건너뜀 (within-route 최적화는 2-opt가 담당)
                    if v_to.ID == v_from.ID:
                        continue

                    route_to = routes_dict[v_to.ID]
                    orig_to_tard = per_v_tardy[v_to.ID]

                    # v_to 라우트의 모든 위치에 삽입해 보기
                    for insert_pos in range(len(route_to) + 1):
                        new_route_to = (
                            route_to[:insert_pos] + [cust] + route_to[insert_pos:]
                        )

                        tard_to_new, _ = eval_vehicle_route_tardy_dist(
                            v_to, new_route_to
                        )
                        if tard_to_new == float("inf"):
                            # 용량/시간창 위반이면 스킵
                            continue

                        # 총 tardiness 변화 계산
                        new_total_tardy = (
                            current_total_tardy
                            - orig_from_tard
                            - orig_to_tard
                            + tard_from_wo
                            + tard_to_new
                        )

                        # 개선되면 move 채택 (first-improvement)
                        if new_total_tardy + 1e-9 < current_total_tardy:
                            routes_dict[v_from.ID] = route_from_wo
                            routes_dict[v_to.ID] = new_route_to
                            improved = True
                            break

                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

    return routes_dict


# ============================================================
# 6) 인스턴스 전체 개선 (2-opt + 1-0 relocate)
# ============================================================
def improve_instance_by_2opt_tardy(instance):
    """
    1단계: 각 vehicle에 대해 tardiness 기준 2-opt 수행
    2단계: 1-0 relocate를 통해 차량 간 고객 이동으로 추가 개선
    이후 instance를 reset하고, 개선된 라우트를 다시 process() 적용.

    return: 전체 tardiness (모든 고객 tardy 합)
    """
    # 1) 차량별 2-opt 결과 라우트 저장
    improved_routes = {}
    for v in instance.vehicles:
        improved_routes[v.ID] = two_opt_for_vehicle_tardy(v)

    # 2) 차량 간 1-0 relocate로 추가 개선
    improved_routes = improve_routes_by_relocate_tardy(instance, improved_routes)

    # 3) instance 초기화 후, 최종 라우트대로 다시 process
    instance.reset()

    for v in instance.vehicles:
        for c in improved_routes[v.ID]:
            v.process(c)

    # 4) 전체 tardiness 계산
    total_tardy = 0.0
    for c in instance.customers:
        if getattr(c, "tardy", None) is not None and c.tardy > 0:
            total_tardy += c.tardy

    return total_tardy


# ============================================================
# 7) 전체 거리 계산
# ============================================================
def compute_total_distance(instance):
    """
    인스턴스 내 모든 vehicle의 총 주행 거리 계산.

    - instance.depot 속성이 있고, 그 안에 loc가 있으면 depot 기준 시작
    - 없으면 vehicle.loc를 시작점으로 사용 (기존 코드와 호환성 유지)
    """
    total = 0.0
    depot = getattr(instance, "depot", None)

    for v in instance.vehicles:
        if depot is not None:
            prev = depot.loc
        else:
            prev = v.loc  # 구조를 모를 경우 기존 방식 유지

        # v.schedules 순회하며 거리 합산
        for c in v.schedules:
            total += get_dist(tuple(prev), tuple(c.loc))
            prev = c.loc

    return total


# ============================================================
# 8) 통합 실행 함수: run_local_search()
# ============================================================
def run_local_search(instance, init_solution):
    """
    주어진 초기해(init_solution)를 기준으로 instance 내 route를 개선하고,
    개선 후 tardiness, distance, runtime을 반환한다.

    init_solution은 구조상 필요하지 않지만, 인터페이스 맞추기용 인자.
    """
    start = time.time()

    tard_after = improve_instance_by_2opt_tardy(instance)
    total_dist_after = compute_total_distance(instance)

    runtime = round(time.time() - start, 4)

    return {
        "tardiness_after_ls": round(tard_after, 3),
        "total_distance_after_ls": round(total_dist_after, 3),
        # LS(2-opt + relocate)만 수행하는 데 걸린 시간
        "runtime_sec": runtime,
    }
