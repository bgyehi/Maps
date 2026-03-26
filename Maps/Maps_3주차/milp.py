from module import *
from docplex.mp.model import Model
from ortools.linear_solver import pywraplp
import numpy as np
# pulp import LpProblem, LpMinimize, LpVariable, lpSum, getSolver
# import gurobipy as grb

def milp_scheduling(_prob:Instance, time_limit=300, init_sol: Schedule = None):
    prob = copy.deepcopy(_prob)  # To prevent changing the original copy
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
    M = 1000000
    M2 = 1000000
    model = Model(name='PMSP')

    # 결정변수
    C_max = model.continuous_var(lb=0, name='C_max')
    C_i = {i: model.continuous_var(lb=0, name='C_' + str(i)) for i in SJ}
    T_i = {i: model.continuous_var(lb=0, name='T_' + str(i)) for i in SJ}
    C_ik = {(i, k): model.continuous_var(lb=0, name='C_' + str(i) + '_' + str(k)) for i in SJ for k in SM}
    S_ik = {(i, k): model.continuous_var(lb=0, name='S_' + str(i) + '_' + str(k)) for i in SJ for k in SM}
    y_ik = {(i, k): model.binary_var(name='y_' + str(i) + '_' + str(k)) for i in SJ for k in SM}
    z_ijk = {(i, j, k): model.binary_var(name='z_' + str(i) + '_' + str(j) + '_' + str(k)) for i in SJ for j in SJ for k
             in SM if i < j}

    constraint_1 = {(i, k): model.add_constraint(
        ct=C_ik[i, k] + S_ik[i, k] <= M * y_ik[i, k],
        ctname="constraint_1_{0}_{1}".format(i, k)) for i in SJ for k in SM}

    constraint_2 = {(i, k): model.add_constraint(
        ct=C_ik[i, k] >= S_ik[i, k] + p[k][i] - M * (1 - y_ik[i, k]),
        ctname="constraint_2_{0}_{1}".format(i, k)) for i in SJ for k in SM}

    if prob.with_setup:
        constraint_3 = {(i, j, k): model.add_constraint(
            ct=S_ik[i, k] >= C_ik[j, k] + s[k][j][i]*y_ik[j, k] - M2*z_ijk[i, j, k],
            ctname="constraint_3_{0}_{1}_{2}".format(i, j, k)) for k in SM for i in SJ for j in SJ if i < j}

        constraint_4 = {(i, j, k): model.add_constraint(
            ct=S_ik[j, k] >= C_ik[i, k] + s[k][i][j]*y_ik[i, k] - M2*(1 - z_ijk[i, j, k]),
            ctname="constraint_4_{0}_{1}_{2}".format(i, j, k)) for k in SM for i in SJ for j in SJ if i < j}
    else:
        constraint_3 = {(i, j, k): model.add_constraint(
            ct=S_ik[i, k] >= C_ik[j, k] - M2 * z_ijk[i, j, k],
            ctname="constraint_3_{0}_{1}_{2}".format(i, j, k)) for k in SM for i in SJ for j in SJ if i < j}

        constraint_4 = {(i, j, k): model.add_constraint(
            ct=S_ik[j, k] >= C_ik[i, k] - M2 * (1 - z_ijk[i, j, k]),
            ctname="constraint_4_{0}_{1}_{2}".format(i, j, k)) for k in SM for i in SJ for j in SJ if i < j}

    constraint_5 = {(i): model.add_constraint(
        ct=model.sum(y_ik[i, k] for k in SM) == 1,
        ctname="constraint_5_{0}".format(i)) for i in SJ}

    constraint_6 = {(i): model.add_constraint(
        ct=model.sum(C_ik[i, k] for k in SM) <= C_i[i],
        ctname="constraint_6_{0}".format(i)) for i in SJ}

    # constraint_check = model.add_constraint(
    #     ct=model.sum(prob.job_list[i].weight*T_i[i] for i in SJ) <= 7081,
    #     ctname="constraint_check")

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
        model.minimize(model.sum(prob.job_list[i].weight*T_i[i] for i in SJ))
    elif prob.objective == 'C':
        model.minimize(model.sum(C_i[i] for i in SJ))
    else:
        constraint_C_max = {(i): model.add_constraint(
            ct= C_max >= C_i[i],
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
            matched_mch.sort(key=lambda x : x.start)
            for bar_id in range(len(matched_mch)):
                if bar_id != (len(matched_mch)-1):
                    job_prev = matched_mch[bar_id].job.ID
                    job_next = matched_mch[bar_id+1].job.ID
                    i, j, k = job_prev, job_next, mch.ID
                    if (i, j, k) in z_ijk:
                        sol.add_var_value(z_ijk[i, j, k], 1.0)
                    elif (j, i, k) in z_ijk:
                        sol.add_var_value(z_ijk[j, i, k], 0.0)
        model.add_mip_start(sol)

    model.parameters.mip.tolerances.mipgap.set(float(0.0))
    model.parameters.mip.tolerances.absmipgap.set(float(0.0))
    model.set_time_limit(time_limit)
    result = model.solve(log_output=True)
    print('Cplex Objective - '+ str(result.objective_value))

    for i in SJ:
        for j in SJ:
            if i < j:
                for k in SM:
                    if round(result.get_value(z_ijk[(i, j, k)])) > 0:
                        job_i = prob.findJob(i)
                        job_i.priority -= 1

    MA = {i: [] for i in SM}
    for i in SJ:
       for k in SM:
           if round(result.get_value(y_ik[i, k])) > 0:
               print('Job {0} on Machine {1} completed at {2}'.format(i, k, int(result.get_value(C_ik[i, k]))))
               job_i = prob.findJob(i)
               job_i.end = result.get_value(C_ik[i, k])
               MA[k].append(job_i)
    for k in SM:
        MA[k] = sorted((job for job in MA[k]), key=lambda m: m.end)
        machine = prob.findMch(k)
        for job in MA[k]:
            machine.process(job)

    obj = get_obj(prob)
    result_check = Schedule('MILP_CPLEX', prob, obj=obj)
    result_check.print_schedule()
    result_check.comp_time = result.solve_details.time
    result_check.status = result.solve_status.name
    if obj != result.objective_value:
        result_check.status = 'Different Objective Values: MILP {0} - Actual {1}'.format(result.objective_value, obj)
    return result_check


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

# def milp_scheduling_gurobi(prob:Instance, time_limit=300):
#
#     SJ = range(0, prob.numJob)
#     SM = range(0, prob.numMch)
#     s = prob.setup
#     p = prob.ptime
#     M = 0
#     max_s = np.array(s).max()
#     for i in SJ:
#         M = M + max([row[i] for row in p])
#         M = M + max_s
#     M2 = M + max_s
#
#     opt_model = grb.Model(name="MIP Model")
#
#     #결정변수
#     C_i = {i: opt_model.addVar(vtype=grb.GRB.CONTINUOUS,lb=0,name="C_{0}".format(i)) for i in SJ}
#     C_ik = {(i, k): opt_model.addVar(vtype=grb.GRB.CONTINUOUS,lb=0,name="C_{0}_{1}".format(i, k)) for i in SJ for k in SM}
#     S_ik = {(i, k): opt_model.addVar(vtype=grb.GRB.CONTINUOUS, lb=0, name="S_{0}_{1}".format(i, k)) for i in SJ for k in SM}
#     y_ik = {(i, k): opt_model.addVar(vtype=grb.GRB.BINARY, name="y_{0}_{1}".format(i, k))for i in SJ for k in SM}
#     z_ijk = {(i,j,k): opt_model.addVar(vtype=grb.GRB.BINARY, name="z_{0}_{1}_{2}".format(i,j,k)) for i in SJ for j in SJ for k in SM if i<j}
#
#     #제약조건
#     constraint_1 = {(i, k): opt_model.addConstr(lhs=(C_ik[i, k] + S_ik[i,k]),sense=grb.GRB.LESS_EQUAL,rhs=M * y_ik[i, k],
#             name="constraint1_{0}_{1}".format(i, k)) for i in SJ for k in SM}
#     constraint_2 = {(i, k): opt_model.addConstr(lhs=(C_ik[i, k] ),sense=grb.GRB.GREATER_EQUAL, rhs=S_ik[i, k] + p[k][i] - M * (1 - y_ik[i, k]),
#             name="constraint2_{0}_{1}".format(i, k)) for i in SJ for k in SM}
#     constraint_3 = {(i, j, k): opt_model.addConstr(lhs=(S_ik[i, k] ), sense=grb.GRB.GREATER_EQUAL, rhs=C_ik[j, k] + s[k][j][i]*y_ik[j, k] - M2*z_ijk[i, j, k],
#                                                 name="constraint3_{0}_{1}_{2}".format(i, j, k)) for k in SM for i in SJ for j in SJ if i < j}
#     constraint_4 = {(i, j, k): opt_model.addConstr(lhs=(S_ik[j, k]), sense=grb.GRB.GREATER_EQUAL,
#                                     rhs=C_ik[i, k] + s[k][i][j]*y_ik[i,k] - M2*(1 - z_ijk[i,j,k]), name="constraint4_{0}_{1}_{2}".format(i, j, k)) for k in SM for i in SJ for j in SJ if i < j}
#     constraint_5 = {i :opt_model.addConstr(lhs=grb.quicksum(y_ik[i, k] for k in SM),sense=grb.GRB.EQUAL,rhs=1,name="constraint5_{0}".format(i)) for i in SJ}
#     constraint_6 = {i: opt_model.addConstr(lhs=grb.quicksum(C_ik[i, k] for k in SM),sense=grb.GRB.LESS_EQUAL, rhs=C_i[i], name="constraint6_{0}".format(i)) for i in SJ}
#
#     #목적함수
#     objective = grb.quicksum(C_i[i] for i in SJ)
#
#     opt_model.ModelSense = grb.GRB.MINIMIZE
#     opt_model.setObjective(objective)
#     opt_model.setParam('TimeLimit', 3600)
#     opt_model.optimize()
#
#     return opt_model