#오로지 논문대로 바이브코딩
import time
import math
import random
import numpy as np


# =========================
# Benchmark functions
# =========================
def sphere(x):
    return np.sum(x ** 2)


def rastrigin(x):
    n = len(x)
    return 10 * n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


def rosenbrock(x):
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2)


BENCHMARKS = {
    "Sphere": {
        "func": sphere,
        "lb": -100.0,
        "ub": 100.0,
    },
    "Rastrigin": {
        "func": rastrigin,
        "lb": -5.12,
        "ub": 5.12,
    },
    "Rosenbrock": {
        "func": rosenbrock,
        "lb": -30.0,
        "ub": 30.0,
    },
}


# =========================
# Original GWO (Mirjalili et al., 2014 style)
# =========================
class GreyWolfOptimizer:
    def __init__(
        self,
        obj_func,
        dim,
        lb,
        ub,
        n_wolves=30,
        max_iter=1000,
        time_limit=300,
        seed=None,
    ):
        self.obj_func = obj_func
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.n_wolves = n_wolves
        self.max_iter = max_iter
        self.time_limit = time_limit

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def _initialize_population(self):
        # Step 1: Initialize grey wolf population uniformly in bounds
        return np.random.uniform(self.lb, self.ub, (self.n_wolves, self.dim))

    def _clip(self, x):
        return np.clip(x, self.lb, self.ub)

    def optimize(self):
        start_time = time.time()

        # Initialize wolves
        X = self._initialize_population()

        # Alpha, beta, delta
        alpha_pos = np.zeros(self.dim)
        beta_pos = np.zeros(self.dim)
        delta_pos = np.zeros(self.dim)

        alpha_score = float("inf")
        beta_score = float("inf")
        delta_score = float("inf")

        convergence_curve = []

        # Main loop
        for t in range(self.max_iter):
            # Stop if 300 seconds exceeded
            if time.time() - start_time >= self.time_limit:
                break

            # Step 2: Evaluate each wolf and update alpha, beta, delta
            for i in range(self.n_wolves):
                X[i] = self._clip(X[i])
                fitness = self.obj_func(X[i])

                if fitness < alpha_score:
                    delta_score = beta_score
                    delta_pos = beta_pos.copy()

                    beta_score = alpha_score
                    beta_pos = alpha_pos.copy()

                    alpha_score = fitness
                    alpha_pos = X[i].copy()

                elif fitness < beta_score:
                    delta_score = beta_score
                    delta_pos = beta_pos.copy()

                    beta_score = fitness
                    beta_pos = X[i].copy()

                elif fitness < delta_score:
                    delta_score = fitness
                    delta_pos = X[i].copy()

            # Save current best
            convergence_curve.append(alpha_score)

            # Step 3: Decrease parameter a linearly from 2 to 0
            a = 2 - 2 * (t / self.max_iter)

            # Step 4: Update positions according to alpha, beta, delta
            for i in range(self.n_wolves):
                for j in range(self.dim):
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

                    # Original GWO update
                    X[i, j] = (X1 + X2 + X3) / 3.0

        elapsed = time.time() - start_time

        return {
            "best_score": alpha_score,
            "best_position": alpha_pos,
            "iterations": len(convergence_curve),
            "runtime": elapsed,
            "convergence": convergence_curve,
        }