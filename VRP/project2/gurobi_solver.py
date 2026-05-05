# ================================================================
# gurobi_solver.py
# 강화된 VRPTW-Tardiness 최소화 MIP (SEC + Valid Inequalities)
# ================================================================

from gurobipy import Model, GRB, quicksum
import module


def solve_vrptw_gurobi(instance, time_limit=300, mipgap=0.01):
    customers = instance.customers
    vehicles  = instance.vehicles
    N = len(customers)
    V = len(vehicles)

    # 노드 인덱스 재구성
    # depot = 0, customers = 1..N
    # 각 vehicle 출발/종료 depot 공통
    nodes = [0] + [c.ID for c in customers]

    # 거리/시간 미리 계산
    dist = {}
    travel = {}
    for i in nodes:
        for j in nodes:
            if i == j:
                dist[i, j] = 0
                travel[i, j] = 0
            else:
                pi = instance.depot if i == 0 else instance.get_customer(i).loc
                pj = instance.depot if j == 0 else instance.get_customer(j).loc
                d = module.get_dist(tuple(pi), tuple(pj))
                dist[i, j] = d
                travel[i, j] = d / instance.vehicle_speed

    # -------------------------
    # 모델 생성
    # -------------------------
    m = Model("VRPTW_Tardiness")
    m.Params.TimeLimit = time_limit
    m.Params.MIPGap = mipgap
    m.Params.OutputFlag = 0
    m.Params.Presolve = 2
    m.Params.Heuristics = 0.2
    m.Params.MIPFocus = 1    # 1=Feasible, 2=Optimal, 3=Bound
    m.Params.Cuts = 2        # aggressive cuts

    # -------------------------
    # 변수
    # -------------------------
    x = m.addVars(nodes, nodes, V, vtype=GRB.BINARY, name="x")      # vehicle k 이동 여부
    t = m.addVars(nodes, V, lb=0, vtype=GRB.CONTINUOUS, name="t")   # service start
    u = m.addVars(nodes, V, lb=0, ub=N, vtype=GRB.CONTINUOUS, name="u")  # MTZ ordering
    tard = m.addVars(nodes, V, lb=0, vtype=GRB.CONTINUOUS, name="tard")

    # -------------------------
    # 목적함수: 총 tardiness 최소화
    # -------------------------
    m.setObjective(quicksum(tard[i, k] for i in nodes for k in range(V)), GRB.MINIMIZE)

    # -------------------------
    # 제약식
    # -------------------------

    # 1) depot 출발 = 1
    for k in range(V):
        m.addConstr(quicksum(x[0, j, k] for j in nodes if j != 0) <= 1)

    # 2) depot 도착 = 1
    for k in range(V):
        m.addConstr(quicksum(x[i, 0, k] for i in nodes if i != 0) <= 1)

    # 3) 각 고객은 정확히 한 번 방문
    for i in nodes[1:]:
        m.addConstr(quicksum(x[j, i, k] for j in nodes for k in range(V) if j != i) == 1)
        m.addConstr(quicksum(x[i, j, k] for j in nodes for k in range(V) if j != i) == 1)

    # 4) 시간창 + Big-M
    BIGM = max(c.tw[1] for c in customers) + 10
    for k in range(V):
        for i in nodes:
            for j in nodes:
                if i != j:
                    serv_i = 0 if i == 0 else instance.get_customer(i).serv_time
                    m.addConstr(
                        t[j, k] >= t[i, k] + serv_i + travel[i, j] - BIGM * (1 - x[i, j, k])
                    )

    # 5) tardiness 정의
    for k in range(V):
        for i in nodes[1:]:
            due = instance.get_customer(i).tw[1]
            m.addConstr(tard[i, k] >= t[i, k] + instance.get_customer(i).serv_time - due)

    # -------------------------
    # 6) MTZ Subtour Elimination
    # -------------------------
    for k in range(V):
        for i in nodes:
            for j in nodes:
                if i != j and i != 0 and j != 0:
                    m.addConstr(u[j, k] >= u[i, k] + 1 - BIGM * (1 - x[i, j, k]))

    # -------------------------
    # 7) Capacity Cuts (강한 valid inequality)
    # -------------------------
    for k in range(V):
        capacity = vehicles[k].capacity
        m.addConstr(
            quicksum(instance.get_customer(i).weight *
                     quicksum(x[i, j, k] for j in nodes if j != i)
                     for i in nodes[1:]) <= capacity
        )

    # -------------------------
    # 최적화 실행
    # -------------------------
    m.optimize()

    # -------------------------
    # 결과 수집
    # -------------------------
    sol = {
        "status": m.Status,
        "objective": m.ObjVal if m.Status == GRB.OPTIMAL or m.Status == GRB.TIME_LIMIT else None,
        "x": m.getAttr("x", x),
        "t": m.getAttr("x", t),
        "tard": m.getAttr("x", tard)
    }
    return sol
