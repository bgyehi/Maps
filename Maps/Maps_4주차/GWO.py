import copy
import time
import random
import numpy as np

from module_modified import Instance, Schedule, get_obj


def _reset_instance_state(prob: Instance):
    """Deep-copied instance의 machine/job 상태 초기화"""
    for job in prob.job_list:
        job.complete = False
        job.start = -1
        job.end = -1
        job.assignedMch = -1
        job.priority = 0

    for mch in prob.machine_list:
        mch.available = 0
        mch.assigned = []
        mch.schedules = []
        mch.priority = 0


def _decode_random_keys_to_sequence(position):
    """Continuous vector -> permutation (random keys)"""
    return list(np.argsort(position))


def _incremental_cost(machine, job, objective):
    """
    현재 machine의 마지막 뒤에 job을 붙였을 때의 증가 비용
    """
    if len(machine.assigned) == 0:
        setup_time = 0
    else:
        prev_job = machine.assigned[-1]
        setup_time = machine.setup[prev_job.ID][job.ID]

    start = machine.available + setup_time
    end = start + machine.ptime[job.ID]

    if objective == "C":
        return end
    elif objective == "Cmax":
        return end
    elif objective == "T":
        return max(end - job.due, 0)
    else:  # wT
        return job.weight * max(end - job.due, 0)


def _decode_position_to_schedule(_prob: Instance, position):
    """
    늑대 위치 벡터를 실제 스케줄로 변환
    - job permutation은 random keys
    - machine assignment는 greedy incremental objective
    """
    prob = copy.deepcopy(_prob)
    _reset_instance_state(prob)

    seq = _decode_random_keys_to_sequence(position)

    for job_id in seq:
        job = prob.findJob(int(job_id))

        best_mch = None
        best_cost = float("inf")

        for mch in prob.machine_list:
            cost = _incremental_cost(mch, job, prob.objective)
            if cost < best_cost:
                best_cost = cost
                best_mch = mch

        best_mch.process(job)

    return prob, get_obj(prob, prob.objective)


def grey_wolf_scheduling(_prob: Instance, time_limit=300, n_wolves=30, seed=None):
    """
    Original GWO adapted to scheduling.
    Core GWO update equations follow Mirjalili et al. (2014).
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    start_time = time.time()

    dim = _prob.numJob
    lb, ub = -1.0, 1.0

    # Step 1: Initialize grey wolf population
    X = np.random.uniform(lb, ub, size=(n_wolves, dim))

    alpha_pos = np.zeros(dim)
    beta_pos = np.zeros(dim)
    delta_pos = np.zeros(dim)

    alpha_score = float("inf")
    beta_score = float("inf")
    delta_score = float("inf")

    best_prob = None
    iteration = 0
    max_iter = 10**9  # time-based stop dominates

    while iteration < max_iter and (time.time() - start_time) < time_limit:
        # Step 2: Evaluate all wolves and update alpha, beta, delta
        for i in range(n_wolves):
            decoded_prob, fitness = _decode_position_to_schedule(_prob, X[i])

            if fitness < alpha_score:
                delta_score, delta_pos = beta_score, beta_pos.copy()
                beta_score, beta_pos = alpha_score, alpha_pos.copy()
                alpha_score, alpha_pos = fitness, X[i].copy()
                best_prob = decoded_prob
            elif fitness < beta_score:
                delta_score, delta_pos = beta_score, beta_pos.copy()
                beta_score, beta_pos = fitness, X[i].copy()
            elif fitness < delta_score:
                delta_score, delta_pos = fitness, X[i].copy()

        # Step 3: linearly decrease a from 2 to 0
        # Original paper style
        a = 2 - 2 * (iteration / max(1, max_iter))

        # Since we are using time-limit stopping,
        # a is alternatively decayed by elapsed ratio
        elapsed_ratio = min((time.time() - start_time) / time_limit, 1.0)
        a = 2 - 2 * elapsed_ratio

        # Step 4: Update each wolf position using alpha, beta, delta
        for i in range(n_wolves):
            for j in range(dim):
                r1, r2 = np.random.rand(), np.random.rand()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha_pos[j] - X[i, j])
                X1 = alpha_pos[j] - A1 * D_alpha

                r1, r2 = np.random.rand(), np.random.rand()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta_pos[j] - X[i, j])
                X2 = beta_pos[j] - A2 * D_beta

                r1, r2 = np.random.rand(), np.random.rand()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta_pos[j] - X[i, j])
                X3 = delta_pos[j] - A3 * D_delta

                X[i, j] = (X1 + X2 + X3) / 3.0

            X[i] = np.clip(X[i], lb, ub)

        iteration += 1

    sched = Schedule("GWO_2014", best_prob, obj=alpha_score)
    sched.comp_time = time.time() - start_time
    sched.status = "TimeLimit" if sched.comp_time >= time_limit else "Finished"
    return sched