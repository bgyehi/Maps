import copy
import time
import random
import numpy as np

from GWO import _reset_instance_state, _incremental_cost
from module_ import Schedule, get_obj


def _evaluate_perm(_prob, perm):
    prob = copy.deepcopy(_prob)
    _reset_instance_state(prob)

    for job_id in perm:
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


def _order_crossover(p1, p2):
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))
    child = [-1] * n
    child[a:b] = p1[a:b]

    fill = [x for x in p2 if x not in child]
    ptr = 0
    for i in range(n):
        if child[i] == -1:
            child[i] = fill[ptr]
            ptr += 1
    return child


def _swap_mutation(perm, pm=0.2):
    perm = perm[:]
    if random.random() < pm:
        i, j = random.sample(range(len(perm)), 2)
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def ga_scheduling(_prob, time_limit=300, pop_size=50, elite_size=5, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    start_time = time.time()
    n = _prob.numJob
    jobs = list(range(n))

    population = []
    for _ in range(pop_size):
        perm = jobs[:]
        random.shuffle(perm)
        population.append(perm)

    best_prob = None
    best_score = float("inf")

    while (time.time() - start_time) < time_limit:
        scored = []
        for perm in population:
            prob, fit = _evaluate_perm(_prob, perm)
            scored.append((fit, perm, prob))
            if fit < best_score:
                best_score = fit
                best_prob = prob

        scored.sort(key=lambda x: x[0])
        elites = [x[1] for x in scored[:elite_size]]

        new_population = elites[:]
        while len(new_population) < pop_size:
            p1 = random.choice(elites)
            p2 = random.choice(elites)
            child = _order_crossover(p1, p2)
            child = _swap_mutation(child)
            new_population.append(child)

        population = new_population

    sched = Schedule("GA", best_prob, obj=best_score)
    sched.comp_time = time.time() - start_time
    sched.status = "TimeLimit" if sched.comp_time >= time_limit else "Finished"
    return sched