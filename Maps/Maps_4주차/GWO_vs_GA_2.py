#오로지 GWO 논문 그대로 바이브코딩과 GA의 비교
import csv
import time
import random
import statistics
import numpy as np
import matplotlib.pyplot as plt

from GWO_2 import GreyWolfOptimizer, BENCHMARKS


# =========================
# Simple real-coded GA
# =========================
class GeneticAlgorithm:
    def __init__(
        self,
        obj_func,
        dim,
        lb,
        ub,
        pop_size=50,
        max_iter=1000,
        time_limit=300,
        crossover_rate=0.9,
        mutation_rate=0.1,
        seed=None,
    ):
        self.obj_func = obj_func
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.time_limit = time_limit
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def _initialize_population(self):
        return np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))

    def _clip(self, x):
        return np.clip(x, self.lb, self.ub)

    def _tournament_selection(self, pop, fitness, k=3):
        idxs = np.random.choice(len(pop), k, replace=False)
        best_idx = idxs[np.argmin(fitness[idxs])]
        return pop[best_idx].copy()

    def _crossover(self, p1, p2):
        if np.random.rand() > self.crossover_rate:
            return p1.copy(), p2.copy()

        alpha = np.random.rand(self.dim)
        c1 = alpha * p1 + (1 - alpha) * p2
        c2 = alpha * p2 + (1 - alpha) * p1
        return self._clip(c1), self._clip(c2)

    def _mutate(self, x):
        for j in range(self.dim):
            if np.random.rand() < self.mutation_rate:
                x[j] += np.random.normal(0, 0.1 * (self.ub - self.lb))
        return self._clip(x)

    def optimize(self):
        start_time = time.time()

        pop = self._initialize_population()
        fitness = np.array([self.obj_func(ind) for ind in pop])

        best_idx = np.argmin(fitness)
        best_score = fitness[best_idx]
        best_position = pop[best_idx].copy()

        convergence_curve = []

        for _ in range(self.max_iter):
            if time.time() - start_time >= self.time_limit:
                break

            new_pop = []

            while len(new_pop) < self.pop_size:
                p1 = self._tournament_selection(pop, fitness)
                p2 = self._tournament_selection(pop, fitness)

                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)

                new_pop.append(c1)
                if len(new_pop) < self.pop_size:
                    new_pop.append(c2)

            pop = np.array(new_pop)
            fitness = np.array([self.obj_func(ind) for ind in pop])

            current_best_idx = np.argmin(fitness)
            current_best_score = fitness[current_best_idx]

            if current_best_score < best_score:
                best_score = current_best_score
                best_position = pop[current_best_idx].copy()

            convergence_curve.append(best_score)

        elapsed = time.time() - start_time

        return {
            "best_score": best_score,
            "best_position": best_position,
            "iterations": len(convergence_curve),
            "runtime": elapsed,
            "convergence": convergence_curve,
        }


# =========================
# Experiment utilities
# =========================
def summarize_results(scores, runtimes):
    return {
        "best_fitness": min(scores),
        "mean_fitness": statistics.mean(scores),
        "std_fitness": statistics.stdev(scores) if len(scores) > 1 else 0.0,
        "mean_runtime": statistics.mean(runtimes),
    }


def run_trials(algorithm_name, algo_class, benchmark_name, benchmark_info, n_trials=20, dim=30):
    scores = []
    runtimes = []
    convergence_curves = []

    for trial in range(n_trials):
        seed = 1000 + trial

        optimizer = algo_class(
            obj_func=benchmark_info["func"],
            dim=dim,
            lb=benchmark_info["lb"],
            ub=benchmark_info["ub"],
            time_limit=300,
            seed=seed,
        )

        result = optimizer.optimize()

        scores.append(result["best_score"])
        runtimes.append(result["runtime"])
        convergence_curves.append(result["convergence"])

        print(
            f"[{algorithm_name}] {benchmark_name} | Trial {trial + 1}/{n_trials} "
            f"| Best={result['best_score']:.6e} | Time={result['runtime']:.2f}s"
        )

    summary = summarize_results(scores, runtimes)

    return {
        "algorithm": algorithm_name,
        "benchmark": benchmark_name,
        "scores": scores,
        "runtimes": runtimes,
        "curves": convergence_curves,
        "summary": summary,
    }


def save_summary_csv(results, filename="results_summary.csv"):
    with open(filename, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Benchmark", "Algorithm", "BestFitness", "MeanFitness",
            "StdFitness", "MeanRuntime"
        ])

        for r in results:
            s = r["summary"]
            writer.writerow([
                r["benchmark"],
                r["algorithm"],
                s["best_fitness"],
                s["mean_fitness"],
                s["std_fitness"],
                s["mean_runtime"],
            ])


def save_trial_csv(results, filename="results_trials.csv"):
    with open(filename, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Benchmark", "Algorithm", "Trial", "BestFitness", "Runtime"
        ])

        for r in results:
            for i, (score, runtime) in enumerate(zip(r["scores"], r["runtimes"]), start=1):
                writer.writerow([
                    r["benchmark"],
                    r["algorithm"],
                    i,
                    score,
                    runtime,
                ])


def plot_mean_fitness(results, filename="mean_fitness_comparison.png"):
    benchmarks = sorted(list(set(r["benchmark"] for r in results)))
    algorithms = sorted(list(set(r["algorithm"] for r in results)))

    x = np.arange(len(benchmarks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, algo in enumerate(algorithms):
        means = []
        for bench in benchmarks:
            matched = [r for r in results if r["benchmark"] == bench and r["algorithm"] == algo][0]
            means.append(matched["summary"]["mean_fitness"])
        ax.bar(x + i * width - width / 2, means, width, label=algo)

    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks)
    ax.set_ylabel("Mean Fitness")
    ax.set_title("Mean Fitness Comparison: GWO vs GA")
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_mean_runtime(results, filename="mean_runtime_comparison.png"):
    benchmarks = sorted(list(set(r["benchmark"] for r in results)))
    algorithms = sorted(list(set(r["algorithm"] for r in results)))

    x = np.arange(len(benchmarks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, algo in enumerate(algorithms):
        means = []
        for bench in benchmarks:
            matched = [r for r in results if r["benchmark"] == bench and r["algorithm"] == algo][0]
            means.append(matched["summary"]["mean_runtime"])
        ax.bar(x + i * width - width / 2, means, width, label=algo)

    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks)
    ax.set_ylabel("Mean Runtime (s)")
    ax.set_title("Mean Runtime Comparison: GWO vs GA")
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_convergence(results, benchmark_name, filename_prefix="convergence"):
    plt.figure(figsize=(10, 6))

    for algo in sorted(list(set(r["algorithm"] for r in results))):
        matched = [r for r in results if r["benchmark"] == benchmark_name and r["algorithm"] == algo][0]
        min_len = min(len(curve) for curve in matched["curves"])
        trimmed = np.array([curve[:min_len] for curve in matched["curves"]])
        mean_curve = np.mean(trimmed, axis=0)
        plt.plot(mean_curve, label=algo)

    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")
    plt.title(f"Convergence Curve: {benchmark_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_{benchmark_name}.png")
    plt.close()


if __name__ == "__main__":
    dim = 30
    n_trials = 20
    all_results = []

    for benchmark_name, benchmark_info in BENCHMARKS.items():
        # GWO
        gwo_result = run_trials(
            algorithm_name="GWO",
            algo_class=GreyWolfOptimizer,
            benchmark_name=benchmark_name,
            benchmark_info=benchmark_info,
            n_trials=n_trials,
            dim=dim,
        )
        all_results.append(gwo_result)

        # GA
        ga_result = run_trials(
            algorithm_name="GA",
            algo_class=GeneticAlgorithm,
            benchmark_name=benchmark_name,
            benchmark_info=benchmark_info,
            n_trials=n_trials,
            dim=dim,
        )
        all_results.append(ga_result)

    save_summary_csv(all_results, "results_summary.csv")
    save_trial_csv(all_results, "results_trials.csv")

    plot_mean_fitness(all_results, "mean_fitness_comparison.png")
    plot_mean_runtime(all_results, "mean_runtime_comparison.png")

    for benchmark_name in BENCHMARKS.keys():
        plot_convergence(all_results, benchmark_name)

    print("\nDone.")
    print("Saved files:")
    print("- results_summary.csv")
    print("- results_trials.csv")
    print("- mean_fitness_comparison.png")
    print("- mean_runtime_comparison.png")
    for benchmark_name in BENCHMARKS.keys():
        print(f"- convergence_{benchmark_name}.png")