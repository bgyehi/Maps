from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import random
import time
import math

from module import Customer, Vehicle, Instance, Solution, get_dist
from gurobipy import Model, GRB, quicksum


# ========================
# 초기해 함수 (휴리스틱)
# ========================
def dispatch_earliest_vehicle_best_customer(instance: Instance) -> Solution:
    """
    기존 휴리스틱 초기해 생성 (Regret-k 등)
    반드시 Solution 객체를 반환하고, objective 값을 채워야 함
    """
    # 여기서는 기존 dispatching_1.py의 함수 사용
    # 아래는 예시일 뿐이며, 실제로는 기존 구현을 import하거나 복사
    instance.reset()
    customers: List[Customer] = instance.customers
    vehicles: List[Vehicle] = instance.vehicles

    # 간단한 greedy dispatch (예시)
    unserved = set(c.ID for c in customers)
    for v in vehicles:
        v.schedules = []

    # 매우 간단한 예시: due date 순으로 차량에 할당
    sorted_customers = sorted(customers, key=lambda c: c.tw[1])
    v_idx = 0
    for c in sorted_customers:
        if c.ID in unserved:
            vehicles[v_idx % len(vehicles)].schedules.append(c)
            unserved.remove(c.ID)

    # Tardiness 계산
    total_tard = calculate_total_tardiness(vehicles)

    sol = Solution("Dispatch_Heuristic", instance, obj=total_tard)
    sol.status = "DONE" if not unserved else "INFEASIBLE_UNSERVED"
    return sol


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
            travel_h = travel_km / speed
            arrival = current_time + travel_h
            start = max(arrival, float(c.tw[0]))
            end = start + float(c.serv_time or 0.0)
            tardy = max(0.0, end - float(c.tw[1]))
            total += tardy
            current_time = end
            current_loc = tuple(c.loc)
    return total


# ========================
# Gurobi 최적화 함수
# ========================
def gurobi_vrptw_optimize(instance: Instance, time_limit_sec: int = 1800) -> Solution:
    """
    Gurobi를 이용한 VRPTW 최적화
    - 이진변수 y[v,i,j]: 차량 v가 고객 i 방문 후 j 방문 여부
    - 연속변수 C[v,i]: 차량 v의 고객 i 서비스 완료 시각
    - 연속변수 T[v,i]: 차량 v의 고객 i에서의 지연
    - 목적함수: 총 지연 최소화
    """
    customers = instance.customers
    vehicles = instance.vehicles
    n = len(customers)
    m = len(vehicles)

    # Big-M 값 설정
    M = 10000.0  # 충분히 큰 값

    # Gurobi 모델 생성
    model = Model("VRPTW_Gurobi")
    model.Params.TimeLimit = time_limit_sec
    model.Params.MIPGap = 0.01  # 1% gap 허용

    # ===== 변수 정의 =====
    # x[v,i,j]: 차량 v가 고객 i에서 고객 j로 이동하는지 여부
    x = {}
    for v_idx in range(m):
        for i in range(n):
            for j in range(n):
                if i != j:
                    x[v_idx, i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{v_idx}_{i}_{j}")

    # t[v,i]: 차량 v가 고객 i에서 서비스 시작하는 시각
    t = model.addVars(m, n, vtype=GRB.CONTINUOUS, name="t")

    # tardy[v,i]: 차량 v의 고객 i에서의 지연
    tardy = model.addVars(m, n, vtype=GRB.CONTINUOUS, lb=0, name="tardy")

    # visit[v,i]: 차량 v가 고객 i를 방문하는지 여부
    visit = model.addVars(m, n, vtype=GRB.BINARY, name="visit")

    model.update()

    # ===== 목적함수 =====
    model.setObjective(quicksum(tardy[v, i] for v in range(m) for i in range(n)), GRB.MINIMIZE)

    # ===== 제약조건 =====
    # 1) 각 고객은 정확히 한 차량에 의해 방문
    for i in range(n):
        model.addConstr(quicksum(visit[v, i] for v in range(m)) == 1, name=f"customer_once_{i}")

    # 2) 차량별 용량 제약
    for v in range(m):
        model.addConstr(
            quicksum(customers[i].weight * visit[v, i] for i in range(n)) <= vehicles[v].capacity,
            name=f"capacity_{v}"
        )

    # 3) 방문 연결: visit[v,i]=1이면 해당 차량이 i로 들어오고 나가는 경로가 있어야
    for v in range(m):
        for i in range(n):
            # 들어오는 경로
            model.addConstr(
                quicksum(x[v, j, i] for j in range(n) if j != i) == visit[v, i],
                name=f"flow_in_{v}_{i}"
            )
            # 나가는 경로
            model.addConstr(
                quicksum(x[v, i, j] for j in range(n) if j != i) == visit[v, i],
                name=f"flow_out_{v}_{i}"
            )

    # 4) 시간창 제약
    for v in range(m):
        for i in range(n):
            model.addConstr(t[v, i] >= customers[i].tw[0] - M * (1 - visit[v, i]), name=f"tw_early_{v}_{i}")
            model.addConstr(t[v, i] <= customers[i].tw[1] + M * (1 - visit[v, i]), name=f"tw_late_{v}_{i}")

    # 5) 시간 연속성 제약 (i에서 j로 이동 시)
    for v in range(m):
        for i in range(n):
            for j in range(n):
                if i != j:
                    travel_time = get_dist(tuple(customers[i].loc), tuple(customers[j].loc)) / vehicles[v].speed
                    service_time = customers[i].serv_time
                    model.addConstr(
                        t[v, j] >= t[v, i] + service_time + travel_time - M * (1 - x[v, i, j]),
                        name=f"time_cont_{v}_{i}_{j}"
                    )

    # 6) 지연 정의
    for v in range(m):
        for i in range(n):
            completion = t[v, i] + customers[i].serv_time
            due = customers[i].tw[1]
            model.addConstr(tardy[v, i] >= completion - due - M * (1 - visit[v, i]), name=f"tardy_def_{v}_{i}")

    # ===== 최적화 실행 =====
    print(f"\n[Gurobi] 최적화 시작 (시간 제한: {time_limit_sec}초)...")
    start_time = time.time()
    model.optimize()
    elapsed = time.time() - start_time

    # ===== 해 추출 =====
    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
        print(f"[Gurobi] 해 발견! (Status: {model.status}, Time: {elapsed:.2f}s)")

        # 차량별 경로 복원
        for v_idx, v in enumerate(vehicles):
            v.schedules = []
            for i in range(n):
                if visit[v_idx, i].X > 0.5:
                    v.schedules.append(customers[i])

            # 방문 순서 정렬 (시작 시각 기준)
            v.schedules.sort(key=lambda c: t[v_idx, customers.index(c)].X)

        final_tardiness = model.ObjVal
        sol = Solution("Gurobi_VRPTW", instance, obj=final_tardiness)
        sol.status = "OPTIMAL" if model.status == GRB.OPTIMAL else "TIME_LIMIT"
        sol.gap = model.MIPGap if hasattr(model, 'MIPGap') else 0.0

        return sol
    else:
        print(f"[Gurobi] 해를 찾지 못함 (Status: {model.status})")
        # 휴리스틱 fallback
        return dispatch_earliest_vehicle_best_customer(instance)
