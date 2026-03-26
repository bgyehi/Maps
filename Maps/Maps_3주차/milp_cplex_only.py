import copy
import numpy as np
from docplex.mp.model import Model
from module import Instance, Schedule, get_obj


def milp_scheduling(prob: Instance, time_limit=300):
    prob = copy.deepcopy(prob)

    SJ = range(prob.numJob)
    SM = range(prob.numMch)
    s = prob.setup
    p = prob.ptime

    max_p = max(max(row) for row in p) if p else 0
    max_s = np.array(s).max() if s else 0
    M = prob.numJob * (max_p + max_s) + max_s + 1
    M2 = M

    model = Model(name="PMSP_SDST_CPLEX")

    C_max = model.continuous_var(lb=0, name="C_max")
    C_i = {i: model.continuous_var(lb=0, name=f"C_{i}") for i in SJ}
    T_i = {i: model.continuous_var(lb=0, name=f"T_{i}") for i in SJ}
    C_ik = {(i, k): model.continuous_var(lb=0, name=f"C_{i}_{k}") for i in SJ for k in SM}
    S_ik = {(i, k): model.continuous_var(lb=0, name=f"S_{i}_{k}") for i in SJ for k in SM}
    y_ik = {(i, k): model.binary_var(name=f"y_{i}_{k}") for i in SJ for k in SM}
    z_ijk = {(i, j, k): model.binary_var(name=f"z_{i}_{j}_{k}")
             for i in SJ for j in SJ for k in SM if i < j}

    for i in SJ:
        for k in SM:
            model.add_constraint(C_ik[i, k] <= M * y_ik[i, k], ctname=f"c1_{i}_{k}")
            model.add_constraint(C_ik[i, k] >= S_ik[i, k] + p[k][i] - M * (1 - y_ik[i, k]), ctname=f"c2_{i}_{k}")

    for k in SM:
        for i in SJ:
            for j in SJ:
                if i < j:
                    model.add_constraint(
                        S_ik[i, k] >= C_ik[j, k] + s[k][j][i] * y_ik[j, k] - M2 * z_ijk[i, j, k],
                        ctname=f"c3_{i}_{j}_{k}"
                    )
                    model.add_constraint(
                        S_ik[j, k] >= C_ik[i, k] + s[k][i][j] * y_ik[i, k] - M2 * (1 - z_ijk[i, j, k]),
                        ctname=f"c4_{i}_{j}_{k}"
                    )

    for i in SJ:
        model.add_constraint(model.sum(y_ik[i, k] for k in SM) == 1, ctname=f"assign_{i}")

    for i in SJ:
        for k in SM:
            model.add_constraint(C_i[i] >= C_ik[i, k], ctname=f"linkC_{i}_{k}")

    if prob.objective == "T":
        for i in SJ:
            model.add_constraint(T_i[i] >= C_i[i] - prob.job_list[i].due, ctname=f"T_{i}")
            model.add_constraint(T_i[i] >= 0, ctname=f"Tlb_{i}")
        model.minimize(model.sum(T_i[i] for i in SJ))

    elif prob.objective == "wT":
        for i in SJ:
            model.add_constraint(T_i[i] >= C_i[i] - prob.job_list[i].due, ctname=f"wT_{i}")
            model.add_constraint(T_i[i] >= 0, ctname=f"wTlb_{i}")
        model.minimize(model.sum(prob.job_list[i].weight * T_i[i] for i in SJ))

    elif prob.objective == "C":
        model.minimize(model.sum(C_i[i] for i in SJ))

    else:
        for i in SJ:
            model.add_constraint(C_max >= C_i[i], ctname=f"cmax_{i}")
        model.minimize(C_max)

    model.parameters.mip.tolerances.mipgap.set(0.0)
    model.parameters.mip.tolerances.absmipgap.set(0.0)
    model.set_time_limit(time_limit)

    result = model.solve(log_output=True)
    if result is None:
        raise RuntimeError("CPLEX가 해를 찾지 못했습니다.")

    for i in SJ:
        for j in SJ:
            if i < j:
                for k in SM:
                    if round(result.get_value(z_ijk[i, j, k])) > 0:
                        prob.findJob(i).priority -= 1

    MA = {k: [] for k in SM}
    for i in SJ:
        for k in SM:
            if round(result.get_value(y_ik[i, k])) > 0:
                job_i = prob.findJob(i)
                job_i.end = result.get_value(C_ik[i, k])
                MA[k].append(job_i)

    for k in SM:
        MA[k] = sorted(MA[k], key=lambda job: job.end)
        machine = prob.findMch(k)
        for job in MA[k]:
            machine.process(job)

    obj = get_obj(prob)
    sched = Schedule("MILP_CPLEX", prob, obj=obj)
    sched.comp_time = result.solve_details.time
    sched.status = result.solve_status.name
    return sched