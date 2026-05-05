from docplex.mp.model import Model
import random
import time


def generate_instance(n_jobs, seed=0, p_low=1, p_high=20, w_low=1, w_high=10, due_factor=0.6):
    rnd = random.Random(seed)

    p = {j: rnd.randint(p_low, p_high) for j in range(n_jobs)}
    w = {j: rnd.randint(w_low, w_high) for j in range(n_jobs)}

    total_p = sum(p.values())
    d_low = max(1, int(total_p * due_factor * 0.4))
    d_high = max(d_low + 1, int(total_p * due_factor * 0.9))
    d = {j: rnd.randint(d_low, d_high) for j in range(n_jobs)}

    return p, d, w


def solve_two_machine_twt_cplex(n_jobs, p, d, w, time_limit=300):
    machines = [0, 1]
    jobs = list(range(n_jobs))

    mdl = Model(name=f"2M_TWT_{n_jobs}")
    mdl.parameters.timelimit = time_limit

    # 충분히 큰 M
    H = sum(p[j] for j in jobs)
    M = H

    # x[j,k] = job j가 machine k에 배정되면 1
    x = mdl.binary_var_dict(((j, k) for j in jobs for k in machines), name="x")

    # C[j] = completion time
    C = mdl.continuous_var_dict(jobs, lb=0, name="C")

    # T[j] = tardiness
    T = mdl.continuous_var_dict(jobs, lb=0, name="T")

    # y[i,j,k] = machine k에서 i가 j보다 먼저면 1
    y = mdl.binary_var_dict(
        ((i, j, k) for i in jobs for j in jobs for k in machines if i != j),
        name="y"
    )

    # 각 job은 정확히 한 머신에 배정
    for j in jobs:
        mdl.add_constraint(mdl.sum(x[j, k] for k in machines) == 1, ctname=f"assign_{j}")

    # completion time
    for j in jobs:
        for k in machines:
            mdl.add_constraint(C[j] >= p[j] - M * (1 - x[j, k]), ctname=f"comp_lb_{j}_{k}")

    # 같은 머신에 배정된 두 job의 순서 제약
    for i in jobs:
        for j in jobs:
            if i >= j:
                continue
            for k in machines:
                # 둘 다 k에 배정되면 둘 중 하나가 먼저
                mdl.add_constraint(
                    y[i, j, k] + y[j, i, k] >= x[i, k] + x[j, k] - 1,
                    ctname=f"order1_{i}_{j}_{k}"
                )
                mdl.add_constraint(
                    y[i, j, k] + y[j, i, k] <= 1,
                    ctname=f"order2_{i}_{j}_{k}"
                )

                # i가 j보다 먼저면 C[j] >= C[i] + p[j]
                mdl.add_constraint(
                    C[j] >= C[i] + p[j] - M * (1 - y[i, j, k]),
                    ctname=f"seq_{i}_{j}_{k}"
                )
                # j가 i보다 먼저면 C[i] >= C[j] + p[i]
                mdl.add_constraint(
                    C[i] >= C[j] + p[i] - M * (1 - y[j, i, k]),
                    ctname=f"seq_{j}_{i}_{k}"
                )

    # tardiness
    for j in jobs:
        mdl.add_constraint(T[j] >= C[j] - d[j], ctname=f"tard_{j}")
        mdl.add_constraint(T[j] >= 0, ctname=f"tard_nonneg_{j}")

    # objective
    mdl.minimize(mdl.sum(w[j] * T[j] for j in jobs))

    start_time = time.time()
    sol = mdl.solve(log_output=True)
    elapsed = time.time() - start_time

    result = {
        "status": mdl.solve_details.status if mdl.solve_details else None,
        "time": elapsed,
        "objective": None,
        "gap": None,
        "schedule": {0: [], 1: []}
    }

    if sol is not None:
        result["objective"] = sol.objective_value
        try:
            result["gap"] = mdl.solve_details.mip_relative_gap
        except Exception:
            result["gap"] = None

        for j in jobs:
            assigned_k = None
            for k in machines:
                if x[j, k].solution_value > 0.5:
                    assigned_k = k
                    break

            c_j = C[j].solution_value
            t_j = T[j].solution_value
            s_j = c_j - p[j]

            result["schedule"][assigned_k].append({
                "job": j,
                "start": s_j,
                "end": c_j,
                "p": p[j],
                "d": d[j],
                "w": w[j],
                "tardiness": t_j
            })

        for k in machines:
            result["schedule"][k].sort(key=lambda z: z["start"])

    return result


def run_scaling_experiment(start_n=5, step=5, max_n=50, seed=0, time_limit=300):
    summary = []

    for n_jobs in range(start_n, max_n + 1, step):
        print(f"\n===== n_jobs = {n_jobs} =====")
        p, d, w = generate_instance(n_jobs=n_jobs, seed=seed + n_jobs)

        result = solve_two_machine_twt_cplex(
            n_jobs=n_jobs,
            p=p,
            d=d,
            w=w,
            time_limit=time_limit
        )

        row = {
            "n_jobs": n_jobs,
            "status": result["status"],
            "objective": result["objective"],
            "gap": result["gap"],
            "time": result["time"]
        }
        summary.append(row)
        print(row)

    return summary


def print_schedule(result):
    print("\nStatus:", result["status"])
    print("Objective:", result["objective"])
    print("Gap:", result["gap"])
    print("Time:", result["time"])

    for k in [0, 1]:
        print(f"\nMachine {k}")
        for item in result["schedule"][k]:
            print(
                f"Job {item['job']:>3} | "
                f"[{item['start']:>6.1f}, {item['end']:>6.1f}] | "
                f"p={item['p']:>3}, d={item['d']:>4}, w={item['w']:>3}, T={item['tardiness']:>6.1f}"
            )


if __name__ == "__main__":
    # 단일 테스트
    n_jobs = 20
    p, d, w = generate_instance(n_jobs=n_jobs, seed=42)
    result = solve_two_machine_twt_cplex(n_jobs, p, d, w, time_limit=300)
    print_schedule(result)

    # 문제 크기 증가 실험
    summary = run_scaling_experiment(
        start_n=5,
        step=5,
        max_n=50,
        seed=100,
        time_limit=300
    )