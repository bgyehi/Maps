from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import random
import time
import math

from module import Customer, Vehicle, Instance, Solution, get_dist
from gurobipy import Model, GRB, quicksum


# ========================
# Tardiness 계산 함수
# ========================
def calculate_total_tardiness(vehicles: List[Vehicle]) -> float:
    """차량별 경로의 총 지연 계산"""
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


# ========================
# 휴리스틱 초기해 (Greedy EDD)
# ========================
def dispatch_earliest_vehicle_best_customer(instance: Instance) -> Solution:
    """EDD 기반 간단한 초기해 생성"""
    instance.reset()
    customers: List[Customer] = instance.customers
    vehicles: List[Vehicle] = instance.vehicles

    # Due date 순으로 정렬
    sorted_customers = sorted(customers, key=lambda c: c.tw[1])

    # 차량 초기화
    for v in vehicles:
        v.schedules = []
        v.now_capacity = 0.0

    # Round-robin 방식으로 할당
    v_idx = 0
    for c in sorted_customers:
        # 용량 체크
        assigned = False
        for _ in range(len(vehicles)):
            v = vehicles[v_idx % len(vehicles)]
            if v.now_capacity + c.weight <= v.capacity:
                v.schedules.append(c)
                v.now_capacity += c.weight
                assigned = True
                v_idx += 1
                break
            v_idx += 1

        if not assigned:
            # 용량 초과시 그냥 첫 차량에 할당
            vehicles[0].schedules.append(c)

    total_tard = calculate_total_tardiness(vehicles)
    sol = Solution("Dispatch_EDD", instance, obj=total_tard)
    sol.status = "DONE"
    return sol


# ========================
# 간단한 ILS 개선
# ========================
def improve_with_ils(solution: Solution, time_limit_sec: float = 2.0) -> float:
    """간단한 swap 기반 local search"""
    vehicles = solution.instance.vehicles
    best_tard = calculate_total_tardiness(vehicles)
    start_time = time.time()

    iteration = 0
    while time.time() - start_time < time_limit_sec:
        iteration += 1
        improved = False

        # Swap within route
        for v in vehicles:
            if len(v.schedules) < 2:
                continue
            for i in range(len(v.schedules)):
                for j in range(i + 1, len(v.schedules)):
                    # Swap
                    v.schedules[i], v.schedules[j] = v.schedules[j], v.schedules[i]
                    new_tard = calculate_total_tardiness(vehicles)

                    if new_tard < best_tard:
                        best_tard = new_tard
                        improved = True
                        break
                    else:
                        # Revert
                        v.schedules[i], v.schedules[j] = v.schedules[j], v.schedules[i]
                if improved:
                    break

        if not improved:
            break

    solution.objective = best_tard
    return best_tard


# ========================
# Gurobi 단순 모델 (MTZ)
# ========================
def gurobi_vrptw_simple(instance: Instance, time_limit_sec: int = 600) -> Solution:
    """
    단순화된 VRPTW Gurobi 모델
    - 고객 수를 20개로 제한
    - MTZ 제약 사용
    - 시간창 soft constraint
    """
    original_customers = instance.customers
    original_vehicles = instance.vehicles

    # 문제 크기 제한
    max_customers = 20
    max_vehicles = 3

    customers = original_customers[:min(len(original_customers), max_customers)]
    vehicles = original_vehicles[:min(len(original_vehicles), max_vehicles)]

    n = len(customers)
    m = len(vehicles)

    print(f"[Gurobi] 고객 {n}개, 차량 {m}대로 최적화 시작...")

    # Gurobi 모델
    model = Model("VRPTW_Simple")
    model.Params.TimeLimit = time_limit_sec
    model.Params.MIPGap = 0.05  # 5% gap 허용
    model.Params.OutputFlag = 1

    # 변수
    x = {}  # x[v,i,j]: 차량 v가 고객 i→j 이동
    for v in range(m):
        for i in range(n):
            for j in range(n):
                if i != j:
                    x[v, i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{v}_{i}_{j}")

    # visit[v,i]: 차량 v가 고객 i 방문 여부
    visit = model.addVars(m, n, vtype=GRB.BINARY, name="visit")

    # t[v,i]: 차량 v의 고객 i 도착 시간
    t = model.addVars(m, n, vtype=GRB.CONTINUOUS, lb=0, name="t")

    # tardiness
    tardy = model.addVars(m, n, vtype=GRB.CONTINUOUS, lb=0, name="tardy")

    # MTZ 변수 (subtour elimination)
    u = model.addVars(m, n, vtype=GRB.CONTINUOUS, lb=0, ub=n, name="u")

    model.update()

    # 목적함수: 총 tardiness 최소화
    model.setObjective(
        quicksum(tardy[v, i] for v in range(m) for i in range(n)),
        GRB.MINIMIZE
    )

    # 제약 1: 각 고객은 정확히 한 번 방문
    for i in range(n):
        model.addConstr(
            quicksum(visit[v, i] for v in range(m)) == 1,
            name=f"visit_once_{i}"
        )

    # 제약 2: 방문하면 들어오고 나가는 경로 존재
    for v in range(m):
        for i in range(n):
            model.addConstr(
                quicksum(x[v, j, i] for j in range(n) if j != i) == visit[v, i],
                name=f"flow_in_{v}_{i}"
            )
            model.addConstr(
                quicksum(x[v, i, j] for j in range(n) if j != i) == visit[v, i],
                name=f"flow_out_{v}_{i}"
            )

    # 제약 3: 용량 제약
    for v in range(m):
        model.addConstr(
            quicksum(customers[i].weight * visit[v, i] for i in range(n)) <= vehicles[v].capacity,
            name=f"capacity_{v}"
        )

    # 제약 4: MTZ (subtour elimination)
    for v in range(m):
        for i in range(n):
            for j in range(n):
                if i != j:
                    model.addConstr(
                        u[v, j] >= u[v, i] + 1 - n * (1 - x[v, i, j]),
                        name=f"mtz_{v}_{i}_{j}"
                    )

    # 제약 5: 시간 연속성
    M = 10000  # Big-M
    for v in range(m):
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist = get_dist(tuple(customers[i].loc), tuple(customers[j].loc))
                    travel_time = dist / vehicles[v].speed
                    service_time = customers[i].serv_time

                    model.addConstr(
                        t[v, j] >= t[v, i] + service_time + travel_time - M * (1 - x[v, i, j]),
                        name=f"time_{v}_{i}_{j}"
                    )

    # 제약 6: 시간창 (soft - 위반 가능)
    for v in range(m):
        for i in range(n):
            # 최소 ready time
            model.addConstr(
                t[v, i] >= customers[i].tw[0] - M * (1 - visit[v, i]),
                name=f"tw_ready_{v}_{i}"
            )
            # tardiness 정의
            completion = t[v, i] + customers[i].serv_time
            due = customers[i].tw[1]
            model.addConstr(
                tardy[v, i] >= completion - due,
                name=f"tardy_def_{v}_{i}"
            )

    # 최적화
    start = time.time()
    model.optimize()
    elapsed = time.time() - start

    # 해 추출
    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
        print(f"[Gurobi] 해 발견! Obj={model.ObjVal:.2f}, Gap={model.MIPGap:.4f}, Time={elapsed:.2f}s")

        # 경로 복원
        for v_idx in range(m):
            vehicles[v_idx].schedules = []
            visited_customers = [(i, u[v_idx, i].X) for i in range(n) if visit[v_idx, i].X > 0.5]
            # MTZ 순서대로 정렬
            visited_customers.sort(key=lambda x: x[1])
            for cust_idx, _ in visited_customers:
                vehicles[v_idx].schedules.append(customers[cust_idx])

        # 나머지 고객들 처리 (축소했을 경우)
        if len(original_customers) > len(customers):
            remaining = original_customers[len(customers):]
            for c in remaining:
                vehicles[0].schedules.append(c)

        final_tard = calculate_total_tardiness(original_vehicles)
        sol = Solution("Gurobi_MTZ", instance, obj=final_tard)
        sol.status = "OPTIMAL" if model.status == GRB.OPTIMAL else "TIME_LIMIT"
        return sol
    else:
        print(f"[Gurobi] 해를 찾지 못함 (Status={model.status})")
        return dispatch_earliest_vehicle_best_customer(instance)
