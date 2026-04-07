import copy
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from milp import milp_scheduling
from module import generate_prob, get_obj, Schedule


# =========================
# 공통 유틸
# =========================
def clone_instance(prob):
    return copy.deepcopy(prob)


def safe_time_value(x):
    try:
        return float(x)
    except:
        return None


# =========================
# 디코더: priority vector -> schedule
# =========================
def decode_priority_to_schedule(_prob, priority_vector, alg_name="META"):
    """
    priority_vector 길이 = numJob
    값이 작은 순서대로 job를 꺼내고,
    각 job를 넣었을 때 weighted tardiness 증가가 가장 작은 machine에 배정
    """
    prob = clone_instance(_prob)

    pairs = list(zip(priority_vector, prob.job_list))
    ordered_jobs = [job for _, job in sorted(pairs, key=lambda x: x[0])]

    for job in ordered_jobs:
        best_machine = None
        best_score = float("inf")
        best_completion = float("inf")

        for mch in prob.machine_list:
            setup = mch.get_setup(job)
            ptime = mch.get_ptime(job)
            completion = mch.available + setup + ptime
            tardiness = max(completion - job.due, 0)
            score = job.weight * tardiness

            if score < best_score:
                best_score = score
                best_completion = completion
                best_machine = mch
            elif score == best_score and completion < best_completion:
                best_completion = completion
                best_machine = mch

        best_machine.process(job)

    obj = get_obj(prob)
    result = Schedule(alg_name, prob, obj)
    result.status = "FEASIBLE"
    return result


# =========================
# GA
# =========================
def tournament_selection(pop, fitness, k=3):
    idxs = np.random.choice(len(pop), size=k, replace=False)
    best_idx = idxs[0]
    for idx in idxs[1:]:
        if fitness[idx] < fitness[best_idx]:
            best_idx = idx
    return pop[best_idx].copy()


def order_crossover(parent1, parent2):
    n = len(parent1)
    a, b = sorted(np.random.choice(range(n), size=2, replace=False))
    child = [-1] * n
    child[a:b+1] = parent1[a:b+1]

    fill_values = [x for x in parent2 if x not in child]
    ptr = 0
    for i in range(n):
        if child[i] == -1:
            child[i] = fill_values[ptr]
            ptr += 1
    return child


def swap_mutation(perm, mutation_rate=0.2):
    child = perm.copy()
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(child)), 2)
        child[i], child[j] = child[j], child[i]
    return child


def permutation_to_priority_vector(perm):
    """
    permutation [2,0,1,3] -> priority vector
    job 2가 0등, job 0이 1등 ...
    """
    n = len(perm)
    vec = np.zeros(n)
    for rank, job_id in enumerate(perm):
        vec[job_id] = rank
    return vec


def ga_scheduling(prob, time_limit=300, pop_size=40, mutation_rate=0.2, elite_size=4):
    start = time.time()
    n = prob.numJob

    population = [random.sample(range(n), n) for _ in range(pop_size)]

    best_sched = None
    best_obj = float("inf")
    history = []

    while time.time() - start < time_limit:
        fitness = []
        schedules = []

        for perm in population:
            priority_vec = permutation_to_priority_vector(perm)
            sched = decode_priority_to_schedule(prob, priority_vec, alg_name="GA")
            schedules.append(sched)
            fitness.append(sched.objective)

            if sched.objective < best_obj:
                best_obj = sched.objective
                best_sched = sched

        history.append(best_obj)

        # 정렬
        ranked = sorted(zip(population, fitness), key=lambda x: x[1])
        new_population = [ind.copy() for ind, _ in ranked[:elite_size]]

        while len(new_population) < pop_size:
            p1 = tournament_selection(population, fitness, k=3)
            p2 = tournament_selection(population, fitness, k=3)
            child = order_crossover(p1, p2)
            child = swap_mutation(child, mutation_rate=mutation_rate)
            new_population.append(child)

        population = new_population

    best_sched.comp_time = time.time() - start
    best_sched.status = "FEASIBLE"
    return best_sched, history


# =========================
# FOA (초파리 최적화 알고리즘)
# =========================
def foa_scheduling(prob, time_limit=300, pop_size=30, step_size=0.15):
    start = time.time()
    n = prob.numJob

    population = np.random.rand(pop_size, n)

    best_sched = None
    best_obj = float("inf")
    best_pos = None
    history = []

    # 초기 평가
    for i in range(pop_size):
        sched = decode_priority_to_schedule(prob, population[i], alg_name="FOA")
        if sched.objective < best_obj:
            best_obj = sched.objective
            best_sched = sched
            best_pos = population[i].copy()

    history.append(best_obj)

    while time.time() - start < time_limit:
        new_population = []

        for _ in range(pop_size):
            candidate = best_pos + np.random.normal(0, step_size, size=n)
            candidate = np.clip(candidate, 0.0, 1.0)

            sched = decode_priority_to_schedule(prob, candidate, alg_name="FOA")

            if sched.objective < best_obj:
                best_obj = sched.objective
                best_sched = sched
                best_pos = candidate.copy()

            new_population.append(candidate)

        # 일부 랜덤 재시작
        new_population = np.array(new_population)
        restart_num = max(1, pop_size // 5)
        restart_idx = np.random.choice(pop_size, restart_num, replace=False)
        new_population[restart_idx] = np.random.rand(restart_num, n)

        population = new_population
        history.append(best_obj)

    best_sched.comp_time = time.time() - start
    best_sched.status = "FEASIBLE"
    return best_sched, history


# =========================
# 그래프 업데이트
# =========================
def update_plot(job_sizes, milp_objs, ga_objs, foa_objs):
    plt.clf()
    plt.plot(job_sizes, milp_objs, marker='o', label='MILP')
    plt.plot(job_sizes, ga_objs, marker='s', label='GA')
    plt.plot(job_sizes, foa_objs, marker='^', label='FOA')

    plt.xlabel('Number of Jobs')
    plt.ylabel('Objective (wT)')
    plt.title('MILP vs GA vs FOA')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.pause(0.1)


# =========================
# 메인 실험
# =========================
if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)

    TIME_LIMIT = 300
    NUM_MCH = 2

    results = []

    job_sizes = []
    milp_objs = []
    ga_objs = []
    foa_objs = []

    plt.ion()
    plt.figure(figsize=(10, 6))

    for i in range(15, 16):
        num_job = 5 + i

        instance = generate_prob(
            numJob=num_job,
            numMch=NUM_MCH,
            setup=True,
            family=False,
            method='Schutten',
            identical_mch=False
        )

        print("=" * 80)
        print(f"job: {num_job}, mch: {NUM_MCH}")

        # -----------------
        # 1) MILP
        # -----------------
        milp_result = milp_scheduling(copy.deepcopy(instance), TIME_LIMIT)

        print("[MILP]")
        print("Objective (wT):", milp_result.objective)
        print("Status:", milp_result.status)
        print("Solve Time:", milp_result.comp_time)

        # -----------------
        # 2) GA
        # -----------------
        ga_result, ga_history = ga_scheduling(
            copy.deepcopy(instance),
            time_limit=TIME_LIMIT,
            pop_size=40,
            mutation_rate=0.2,
            elite_size=4
        )

        print("[GA]")
        print("Objective (wT):", ga_result.objective)
        print("Status:", ga_result.status)
        print("Solve Time:", ga_result.comp_time)

        # -----------------
        # 3) FOA
        # -----------------
        foa_result, foa_history = foa_scheduling(
            copy.deepcopy(instance),
            time_limit=TIME_LIMIT,
            pop_size=30,
            step_size=0.15
        )

        print("[FOA]")
        print("Objective (wT):", foa_result.objective)
        print("Status:", foa_result.status)
        print("Solve Time:", foa_result.comp_time)

        # -----------------
        # 결과 저장
        # -----------------
        milp_obj = milp_result.objective
        ga_obj = ga_result.objective
        foa_obj = foa_result.objective

        milp_time = safe_time_value(milp_result.comp_time)
        ga_time = safe_time_value(ga_result.comp_time)
        foa_time = safe_time_value(foa_result.comp_time)

        gap_ga = None if milp_obj == 0 else 100.0 * (ga_obj - milp_obj) / milp_obj
        gap_foa = None if milp_obj == 0 else 100.0 * (foa_obj - milp_obj) / milp_obj

        results.append({
            "num_job": num_job,
            "num_mch": NUM_MCH,

            "milp_obj": milp_obj,
            "milp_status": milp_result.status,
            "milp_time": milp_time,

            "ga_obj": ga_obj,
            "ga_status": ga_result.status,
            "ga_time": ga_time,
            "ga_gap_percent_vs_milp": gap_ga,

            "foa_obj": foa_obj,
            "foa_status": foa_result.status,
            "foa_time": foa_time,
            "foa_gap_percent_vs_milp": gap_foa
        })

        # 그래프용 리스트
        job_sizes.append(num_job)
        milp_objs.append(milp_obj)
        ga_objs.append(ga_obj)
        foa_objs.append(foa_obj)

        # 매 for 종료 시 그래프 갱신
        update_plot(job_sizes, milp_objs, ga_objs, foa_objs)

    plt.ioff()
    plt.show()

    df = pd.DataFrame(results)
    print("\n=== Summary ===")
    print(df)

    df.to_csv("compare_milp_ga_foa.csv", index=False, encoding="utf-8-sig")
    print("\nSaved: compare_milp_ga_foa.csv")
    print("Done")