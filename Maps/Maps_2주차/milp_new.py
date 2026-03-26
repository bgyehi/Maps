from module import *  # Instance, Schedule, 보조 함수 등 공통 클래스/함수
from docplex.mp.model import Model  # CPLEX용 MILP 모델 생성
from ortools.linear_solver import pywraplp
import numpy as np  # setup time의 최댓값 등 수치 계산용
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, getSolver


# import gurobipy as grb

# 함수 선언: 입력 인스턴스에 대해 CPLEX 기반 MILP 스케줄링 수행
def milp_scheduling(_prob: Instance, time_limit=300, init_sol: Schedule = None):
    prob = copy.deepcopy(_prob)  # To prevent changing the original copy # 원본 인스턴스를 직접 변경하지 않기 위해 사용
    # Job / Machine 인덱스 집합
    SJ = range(0, prob.numJob)
    SM = range(0, prob.numMch)
    # setup time, processing time 데이터
    s = prob.setup
    p = prob.ptime
    # Big-M 부분
    M = 0
    max_s = np.array(s).max()
    for i in SJ:
        M = M + max([row[i] for row in p])
        M = M + max_s
    M2 = M + max_s
    M = 1000000  # 고정 값 (제약식에 들어가는 충분히 큰 수?)
    M2 = 1000000  # 고정 값 (")
    model = Model(name='PMSP')  # CPLEX MILP 모델 생성

    # 결정변수
    C_max = model.continuous_var(lb=0, name='C_max')  # 전체 최대 완료시간
    C_i = {i: model.continuous_var(lb=0, name='C_' + str(i)) for i in SJ}  # job i 완료시간
    T_i = {i: model.continuous_var(lb=0, name='T_' + str(i)) for i in SJ}  # job i tardiness
    C_ik = {(i, k): model.continuous_var(lb=0, name='C_' + str(i) + '_' + str(k)) for i in SJ for k in
            SM}  # machine k에서의 완료시간
    S_ik = {(i, k): model.continuous_var(lb=0, name='S_' + str(i) + '_' + str(k)) for i in SJ for k in
            SM}  # machine k에서의 시작시간
    y_ik = {(i, k): model.binary_var(name='y_' + str(i) + '_' + str(k)) for i in SJ for k in
            SM}  # job i가 machine k에 배정되면 1 (배정 변수)
    z_ijk = {(i, j, k): model.binary_var(name='z_' + str(i) + '_' + str(j) + '_' + str(k)) for i in SJ for j in SJ for k
             in SM if i < j}  # 같은 machine에서 i,j의 선후관계 (순서 변수)

    # 배정되지 않은 machine에서는 시작/완료시간이 활성화되지 않도록 하는 제약
    constraint_1 = {(i, k): model.add_constraint(
        ct=C_ik[i, k] + S_ik[i, k] <= M * y_ik[i, k],
        ctname="constraint_1_{0}_{1}".format(i, k)) for i in SJ for k in SM}

    # 배정된 machine에서는 완료시간 >= 시작시간 + 처리시간
    constraint_2 = {(i, k): model.add_constraint(
        ct=C_ik[i, k] >= S_ik[i, k] + p[k][i] - M * (1 - y_ik[i, k]),
        ctname="constraint_2_{0}_{1}".format(i, k)) for i in SJ for k in SM}

    if prob.with_setup:
        constraint_3 = {(i, j, k): model.add_constraint(
            ct=S_ik[i, k] >= C_ik[j, k] + s[k][j][i] * y_ik[j, k] - M2 * z_ijk[i, j, k],
            ctname="constraint_3_{0}_{1}_{2}".format(i, j, k)) for k in SM for i in SJ for j in SJ if i < j}

        constraint_4 = {(i, j, k): model.add_constraint(
            ct=S_ik[j, k] >= C_ik[i, k] + s[k][i][j] * y_ik[i, k] - M2 * (1 - z_ijk[i, j, k]),
            ctname="constraint_4_{0}_{1}_{2}".format(i, j, k)) for k in SM for i in SJ for j in SJ if i < j}
    else:
        constraint_3 = {(i, j, k): model.add_constraint(
            ct=S_ik[i, k] >= C_ik[j, k] - M2 * z_ijk[i, j, k],
            ctname="constraint_3_{0}_{1}_{2}".format(i, j, k)) for k in SM for i in SJ for j in SJ if i < j}

        constraint_4 = {(i, j, k): model.add_constraint(
            ct=S_ik[j, k] >= C_ik[i, k] - M2 * (1 - z_ijk[i, j, k]),
            ctname="constraint_4_{0}_{1}_{2}".format(i, j, k)) for k in SM for i in SJ for j in SJ if i < j}

    # 각 job은 정확히 한 개의 machine에만 배정
    constraint_5 = {(i): model.add_constraint(
        ct=model.sum(y_ik[i, k] for k in SM) == 1,
        ctname="constraint_5_{0}".format(i)) for i in SJ}

    # 각 machine별 완료시간 변수와 job의 전체 완료시간 C_i를 연결
    constraint_6 = {(i): model.add_constraint(
        ct=model.sum(C_ik[i, k] for k in SM) <= C_i[i],
        ctname="constraint_6_{0}".format(i)) for i in SJ}

    # 목적함수
    if prob.objective == 'T':
        constraint_T = {(i): model.add_constraint(
            ct=T_i[i] >= C_i[i] - prob.job_list[i].due,
            ctname="constraint_T".format(i)) for i in SJ}
        model.minimize(model.sum(T_i[i] for i in SJ))
    elif prob.objective == 'wT':
        constraint_T = {(i): model.add_constraint(
            ct=T_i[i] >= C_i[i] - prob.job_list[i].due,
            ctname="constraint_T".format(i)) for i in SJ}
        model.minimize(model.sum(prob.job_list[i].weight * T_i[i] for i in SJ))
    elif prob.objective == 'C':
        model.minimize(model.sum(C_i[i] for i in SJ))
    else:
        constraint_C_max = {(i): model.add_constraint(
            ct=C_max >= C_i[i],
            ctname="constraint_C_max".format(i)) for i in SJ}
        model.minimize(C_max)

    if init_sol is not None:
        sol = model.new_solution()
        for job in init_sol.instance.job_list:
            i = job.ID
            k = job.assignedMch
            if (i, k) in y_ik:
                sol.add_var_value(y_ik[i, k], 1.0)

        for mch in init_sol.instance.machine_list:
            matched_mch = [b for b in init_sol.bars if b.machine == mch.ID]
            matched_mch.sort(key=lambda x: x.start)
            for bar_id in range(len(matched_mch)):
                if bar_id != (len(matched_mch) - 1):
                    job_prev = matched_mch[bar_id].job.ID
                    job_next = matched_mch[bar_id + 1].job.ID
                    i, j, k = job_prev, job_next, mch.ID
                    if (i, j, k) in z_ijk:
                        sol.add_var_value(z_ijk[i, j, k], 1.0)
                    elif (j, i, k) in z_ijk:
                        sol.add_var_value(z_ijk[j, i, k], 0.0)
        model.add_mip_start(sol)

    # MILP gap 허용오차를 0으로 설정하고 시간 제한 내에서 CPLEX 실행
    model.parameters.mip.tolerances.mipgap.set(float(0.0))
    model.parameters.mip.tolerances.absmipgap.set(float(0.0))
    model.set_time_limit(time_limit)
    result = model.solve(log_output=True)
    print('Cplex Objective - ' + str(result.objective_value))

    # z_ijk 해를 바탕으로 작업 간 상대적 순서 정보를 priority에 반영
    for i in SJ:
        for j in SJ:
            if i < j:
                for k in SM:
                    if round(result.get_value(z_ijk[(i, j, k)])) > 0:
                        job_i = prob.findJob(i)
                        job_i.priority -= 1

    # 배정변수 y_ik 해를 읽어 각 job의 기계 배정 및 완료시간을 복원
    MA = {i: [] for i in SM}
    for i in SJ:
        for k in SM:
            if round(result.get_value(y_ik[i, k])) > 0:
                print('Job {0} on Machine {1} completed at {2}'.format(i, k, int(result.get_value(C_ik[i, k]))))
                job_i = prob.findJob(i)
                job_i.end = result.get_value(C_ik[i, k])
                MA[k].append(job_i)
    # 각 machine의 작업들을 완료시간 기준으로 정렬한 뒤 스케줄 객체에 반영
    for k in SM:
        MA[k] = sorted((job for job in MA[k]), key=lambda m: m.end)
        machine = prob.findMch(k)
        for job in MA[k]:
            machine.process(job)

    # 복원된 스케줄의 실제 목적함수값을 다시 계산하여 MILP 결과와 일치하는지 검증
    obj = get_obj(prob)
    result_check = Schedule('MILP_CPLEX', prob, obj=obj)
    result_check.print_schedule()
    result_check.comp_time = result.solve_details.time
    result_check.status = result.solve_status.name
    if obj != result.objective_value:
        result_check.status = 'Different Objective Values: MILP {0} - Actual {1}'.format(result.objective_value, obj)
    return result_check

# OR-Tools, gurobi 관련 코드라 일단 해석 X
def milp_scheduling_ortools(prob:Instance, time_limit=300):
    solver = pywraplp.Solver.CreateSolver('SCIP')
    SJ = range(0, prob.numJob)
    SM = range(0, prob.numMch)
    s = prob.setup
    p = prob.ptime
    M = 0
    max_s = np.array(s).max()
    for i in SJ:
        M = M + max([row[i] for row in p])
        M = M + max_s
    M2 = M + max_s
    infinity = solver.infinity()

    C_i= {i: solver.NumVar(0, infinity, 'C_' + str(i)) for i in SJ}
    C_ik ={(i,k): solver.NumVar(0, infinity, 'C_' + str(i) + '_' + str(k)) for i in SJ for k in SM}
    S_ik ={(i,k): solver.NumVar(0, infinity, 'S_' + str(i) + '_' + str(k)) for i in SJ for k in SM}
    y_ik ={(i,k): solver.IntVar(0, 1, 'y_' + str(i) + '_' + str(k)) for i in SJ for k in SM}
    z_ijk ={(i,j,k) : solver.IntVar(0, 1, 'z_' + str(i) + '_' + str(j) + '_' + str(k)) for i in SJ for j in SJ for k in SM}

    # Add Constraints
    constraint_1 = {(i,k) : solver.Add(C_ik[i,k] + S_ik[i,k] <= M * y_ik[i,k]) for i in SJ for k in SM }
    constraint_2 = {(i,k) : solver.Add(C_ik[i,k] >= S_ik[i,k] + p[k][i] - M * (1 - y_ik[i,k])) for i in SJ for k in SM }
    constraint_3 = {(i,j,k) : solver.Add(S_ik[i,k] >= C_ik[j,k] + s[k][j][i] * y_ik[j,k] - M2 * z_ijk[i,j,k]) for k in SM for i in SJ for j in SJ if i < j}
    constraint_4 = {(i,j,k) : solver.Add(S_ik[j,k] >= C_ik[i,k] + s[k][i][j] * y_ik[i,k] - M2 * (1 - z_ijk[i,j,k])) for k in SM for i in SJ for j in SJ if i < j}
    constraint_5 = {(i) : solver.Add(solver.Sum(y_ik[i,k] for k in SM) == 1) for i in SJ}
    constraint_6 = {(i) : solver.Add(solver.Sum(C_ik[i,k] for k in SM) <= C_i[i]) for i in SJ}

    solver.Minimize(sum([C_i[i] for i in SJ]))
    solver.set_time_limit(time_limit*1000)
    solver.EnableOutput()
    status = solver.Solve()



    if status == pywraplp.Solver.OPTIMAL:
        print('Solution:')
        print('Objective value =', solver.Objective().Value())
        return solver
    elif status == pywraplp.Solver.FEASIBLE:
        print('Solution:')
        print('Objective value =', solver.Objective().Value())
        return solver
    else:
        print("답이 없습니다.")
        return solver

def milp_scheduling_pulp(prob:Instance, time_limit=300):
    SJ = range(0, prob.numJob)
    SM = range(0, prob.numMch)
    s = prob.setup
    p = prob.ptime
    M = 0
    max_s = np.array(s).max()
    for i in SJ:
        M = M + max([row[i] for row in p])
        M = M + max_s
    M2 = M + max_s
    model = LpProblem("pulp_model", LpMinimize)

    # 결정변수
    C_i = {i: LpVariable(name=f'C_{i}', lowBound=0, cat=pl.LpContinuous) for i in SJ}
    C_ik = {(i, k): LpVariable(name=('C_' + str(i) + '_' + str(k)), lowBound=0, cat=pl.LpContinuous) for i in SJ for k in SM}
    S_ik = {(i, k): LpVariable(name=('S_' + str(i) + '_' + str(k)), lowBound=0, cat=pl.LpContinuous) for i in SJ for k in SM}
    y_ik = {(i, k): LpVariable(name=('Y_' + str(i) + '_' + str(k)), lowBound=0, cat=pl.LpBinary) for i in SJ for k in SM}
    z_ijk = {(i, j, k): LpVariable(name=('Z_' + str(i) + '_' + str(j) + '_' + str(k)), lowBound=0, cat=pl.LpBinary) for i in SJ for j in SJ for k
             in SM if i < j}

    for i in SJ:
        for k in SM:
            model += C_ik[i, k] + S_ik[i, k] <= M * y_ik[i, k], f"constraint_1_{i}_{k}"
            model += C_ik[i, k] >= S_ik[i, k] + p[k][i] - M * (1 - y_ik[i, k]), f"constraint_2_{i}_{k}"

    for i in SJ:
        for j in SJ:
            for k in SM:
                if i < j:
                    model += S_ik[i, k] >= C_ik[j, k] + s[k][j][i] * y_ik[j, k] - M2 * z_ijk[
                        i, j, k], f"constraint_3_{i}_{j}_{k}"
                    model += S_ik[j, k] >= C_ik[i, k] + s[k][i][j] * y_ik[i, k] - M2 * (
                                1 - z_ijk[i, j, k]), f"constraint_4_{i}_{j}_{k}"

    for i in SJ:
        model += lpSum(y_ik[i, k] for k in SM) == 1, f"constraint_5_{i}"

    for i in SJ:
        model += lpSum(C_ik[i, k] for k in SM) <= C_i[i], f"constraint_6_{i}"

    model += lpSum(C_i[i] for i in SJ)
    solver = getSolver('PULP_CBC_CMD', timeLimit=time_limit)
    result = model.solve(solver)
    return model