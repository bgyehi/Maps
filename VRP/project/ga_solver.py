import random
import copy
import time
from typing import List

def run_genetic_algorithm(instance, dispatch_func, time_limit_sec=0.5, population_size=20, mutation_rate=0.1, verbose=False):
    """
    간단한 GA 버전 (교수님 프레임워크와 호환)
    """
    start_time = time.time()

    # 초기 population 생성
    population = []
    for _ in range(population_size):
        sol = dispatch_func(instance)
        population.append(sol)

    def evaluate(sol):
        return sol.objective

    def crossover(parent1, parent2):
        child = copy.deepcopy(parent1)
        for v in child.instance.vehicles:
            if random.random() < 0.5:
                v.schedules = list(parent2.instance.vehicles[v.ID].schedules)
        return child

    def mutate(sol):
        v = random.choice(sol.instance.vehicles)
        if len(v.schedules) > 1:
            i, j = random.sample(range(len(v.schedules)), 2)
            v.schedules[i], v.schedules[j] = v.schedules[j], v.schedules[i]

    # 초기 평가
    best_sol = min(population, key=evaluate)
    best_val = evaluate(best_sol)

    while time.time() - start_time < time_limit_sec:
        parents = random.sample(population, 2)
        child = crossover(parents[0], parents[1])
        if random.random() < mutation_rate:
            mutate(child)

        val = evaluate(child)
        if val < best_val:
            best_sol, best_val = child, val
            if verbose:
                print(f"  [GA] New best found: {best_val:.3f}")

    return best_sol, best_val
