import copy
import os
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
    except Exception:
        return None


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


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

        for perm in population:
            priority_vec = permutation_to_priority_vector(perm)
            sched = decode_priority_to_schedule(prob, priority_vec, alg_name="GA")
            fitness.append(sched.objective)

            if sched.objective < best_obj:
                best_obj = sched.objective
                best_sched = sched

        history.append(best_obj)

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
# FOA
# =========================
def foa_scheduling(prob, time_limit=300, pop_size=30, step_size=0.15, restart_ratio=0.2):
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
        restart_num = max(1, int(pop_size * restart_ratio))
        restart_idx = np.random.choice(pop_size, restart_num, replace=False)
        new_population[restart_idx] = np.random.rand(restart_num, n)

        population = new_population
        history.append(best_obj)

    best_sched.comp_time = time.time() - start
    best_sched.status = "FEASIBLE"
    return best_sched, history


# =========================
# FOA Grid Search 평가
# =========================
def evaluate_foa_grid(instance, time_limit, pop_size, step_size, restart_ratio, n_runs=20, base_seed=42):
    objs = []
    times = []
    histories = []

    best_run_obj = float("inf")
    best_run_sched = None
    best_run_history = None
    best_run_seed = None

    for run_idx in range(n_runs):
        seed = base_seed + run_idx
        random.seed(seed)
        np.random.seed(seed)

        result, history = foa_scheduling(
            copy.deepcopy(instance),
            time_limit=time_limit,
            pop_size=pop_size,
            step_size=step_size,
            restart_ratio=restart_ratio
        )

        objs.append(result.objective)
        times.append(safe_time_value(result.comp_time))
        histories.append(history)

        if result.objective < best_run_obj:
            best_run_obj = result.objective
            best_run_sched = result
            best_run_history = history
            best_run_seed = seed

    return {
        "mean_obj": float(np.mean(objs)),
        "std_obj": float(np.std(objs)),
        "min_obj": float(np.min(objs)),
        "max_obj": float(np.max(objs)),
        "mean_time": float(np.mean(times)),
        "best_run_obj": best_run_obj,
        "best_run_sched": best_run_sched,
        "best_run_history": best_run_history,
        "best_run_seed": best_run_seed,
        "all_objs": objs,
    }


# =========================
# 시각화: Heatmap
# =========================
def plot_foa_grid_heatmaps(df_job, job_size, pop_candidates, step_candidates, restart_candidates, save_dir):
    fig, axes = plt.subplots(1, len(restart_candidates), figsize=(24, 4.8), constrained_layout=True)

    if len(restart_candidates) == 1:
        axes = [axes]

    global_vmin = df_job["mean_obj"].min()
    global_vmax = df_job["mean_obj"].max()

    for ax, rr in zip(axes, restart_candidates):
        sub = df_job[df_job["restart_ratio"] == rr].copy()
        pivot = sub.pivot(index="pop_size", columns="step_size", values="mean_obj")
        pivot = pivot.reindex(index=pop_candidates, columns=step_candidates)

        im = ax.imshow(
            pivot.values,
            aspect="auto",
            origin="lower",
            vmin=global_vmin,
            vmax=global_vmax
        )

        ax.set_title(f"restart_ratio={rr}")
        ax.set_xlabel("step_size")
        ax.set_ylabel("pop_size")
        ax.set_xticks(range(len(step_candidates)))
        ax.set_xticklabels(step_candidates)
        ax.set_yticks(range(len(pop_candidates)))
        ax.set_yticklabels(pop_candidates)

        for i in range(len(pop_candidates)):
            for j in range(len(step_candidates)):
                val = pivot.values[i, j]
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=8)

    fig.suptitle(f"FOA Grid Search Heatmaps (job={job_size})", fontsize=14)
    cbar = fig.colorbar(im, ax=axes, shrink=0.9)
    cbar.set_label("Mean Objective (wT)")

    out_path = os.path.join(save_dir, f"foa_grid_heatmap_job_{job_size}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# =========================
# 시각화: GA vs Best FOA
# =========================
def plot_best_foa_vs_fixed_ga(summary_df, save_dir):
    fig = plt.figure(figsize=(10, 6))
    plt.plot(summary_df["num_job"], summary_df["ga_obj"], marker="s", label="GA (fixed)")
    plt.plot(summary_df["num_job"], summary_df["best_foa_mean_obj"], marker="o", label="Best FOA (grid mean)")
    plt.xlabel("Number of Jobs")
    plt.ylabel("Objective (wT)")
    plt.title("Fixed GA vs Best FOA by Job Size")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(save_dir, "best_foa_vs_fixed_ga.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# =========================
# 시각화: FOA best history
# =========================
def plot_best_foa_history(history, job_size, params, save_dir):
    if history is None or len(history) == 0:
        return

    fig = plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(history) + 1), history, marker="o", markersize=3)
    plt.xlabel("Iteration")
    plt.ylabel("Best Objective so far")
    plt.title(
        f"Best FOA History (job={job_size})\n"
        f"pop={params[0]}, step={params[1]}, restart={params[2]}"
    )
    plt.grid(True)
    plt.tight_layout()

    out_path = os.path.join(save_dir, f"best_foa_history_job_{job_size}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# =========================
# 시각화: Gantt schedule
# =========================
def plot_schedule_gantt(schedule, title, save_path):
    """
    schedule.instance.machine_list / assigned job 정보를 이용해 gantt chart 저장
    """
    instance = schedule.instance
    machines = instance.machine_list

    fig, ax = plt.subplots(figsize=(14, max(4, len(machines) * 1.2)))

    cmap = plt.get_cmap("tab20")
    color_map = {}

    for job in instance.job_list:
        color_map[job.ID] = cmap(job.ID % 20)

    y_ticks = []
    y_labels = []

    for row_idx, mch in enumerate(machines):
        y_base = row_idx * 10
        y_ticks.append(y_base + 4)
        y_labels.append(f"Machine {mch.ID}")

        assigned_jobs = sorted(mch.assigned, key=lambda j: j.start)

        prev_end = 0
        for job in assigned_jobs:
            setup_start = prev_end
            setup_end = job.start
            proc_start = job.start
            proc_end = job.end

            # setup 구간
            if setup_end > setup_start:
                ax.broken_barh(
                    [(setup_start, setup_end - setup_start)],
                    (y_base, 8),
                    facecolors="lightgray",
                    edgecolors="black",
                    alpha=0.7
                )

            # processing 구간
            ax.broken_barh(
                [(proc_start, proc_end - proc_start)],
                (y_base, 8),
                facecolors=color_map[job.ID],
                edgecolors="black"
            )

            # job label
            ax.text(
                x=(proc_start + proc_end) / 2,
                y=y_base + 4,
                s=f"J{job.ID}",
                ha="center",
                va="center",
                fontsize=8,
                color="black"
            )

            # due date line
            ax.axvline(job.due, color="red", linestyle="--", alpha=0.15)

            prev_end = job.end

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Time")
    ax.set_ylabel("Machines")
    ax.set_title(title)
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# =========================
# 메인
# =========================
if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)

    # -----------------
    # 저장 폴더
    # -----------------
    OUTPUT_DIR = "foa_grid_outputs"
    ensure_dir(OUTPUT_DIR)

    # -----------------
    # 고정 설정
    # -----------------
    NUM_MCH = 2
    JOB_SIZES = [10, 50, 100]

    # 요청사항 반영
    GA_TIME_LIMIT = 60
    FOA_TIME_LIMIT = 60
    FOA_RUNS_PER_GRID = 10

    # -----------------
    # GA는 고정
    # -----------------
    GA_POP_SIZE = 40
    GA_MUTATION_RATE = 0.2
    GA_ELITE_SIZE = 4

    # -----------------
    # FOA Grid Search (5x5x5)
    # -----------------
    FOA_POP_CANDIDATES = [10, 30, 50]
    FOA_STEP_CANDIDATES = [0.05,  0.15, 0.30]
    FOA_RESTART_CANDIDATES = [0.10, 0.20, 0.30]

    all_grid_rows = []
    summary_rows = []

    for num_job in JOB_SIZES:
        print("=" * 120)
        print(f"[START] job={num_job}, mch={NUM_MCH}")

        # -----------------
        # instance 생성
        # -----------------
        instance = generate_prob(
            numJob=num_job,
            numMch=NUM_MCH,
            setup=True,
            family=False,
            method='Schutten',
            identical_mch=False
        )

        # -----------------
        # GA 실행 (고정)
        # -----------------
        random.seed(42)
        np.random.seed(42)
        ga_result, ga_history = ga_scheduling(
            copy.deepcopy(instance),
            time_limit=GA_TIME_LIMIT,
            pop_size=GA_POP_SIZE,
            mutation_rate=GA_MUTATION_RATE,
            elite_size=GA_ELITE_SIZE
        )

        print("[GA FIXED]")
        print("Objective (wT):", ga_result.objective)
        print("Status:", ga_result.status)
        print("Solve Time:", ga_result.comp_time)

        # GA gantt 저장
        ga_gantt_path = os.path.join(OUTPUT_DIR, f"ga_schedule_job_{num_job}.png")
        plot_schedule_gantt(
            ga_result,
            title=f"GA Fixed Schedule (job={num_job})",
            save_path=ga_gantt_path
        )

        # -----------------
        # FOA Grid Search
        # -----------------
        best_foa_mean_score = float("inf")
        best_foa_params = None
        best_foa_schedule = None
        best_foa_history = None
        best_foa_seed = None
        best_foa_best_run_obj = None

        total_cases = (
            len(FOA_POP_CANDIDATES)
            * len(FOA_STEP_CANDIDATES)
            * len(FOA_RESTART_CANDIDATES)
        )
        case_no = 0

        for pop_size in FOA_POP_CANDIDATES:
            for step_size in FOA_STEP_CANDIDATES:
                for restart_ratio in FOA_RESTART_CANDIDATES:
                    case_no += 1
                    print(
                        f"[FOA GRID] job={num_job} "
                        f"({case_no}/{total_cases}) | "
                        f"pop={pop_size}, step={step_size}, restart={restart_ratio}"
                    )

                    summary = evaluate_foa_grid(
                        instance=instance,
                        time_limit=FOA_TIME_LIMIT,
                        pop_size=pop_size,
                        step_size=step_size,
                        restart_ratio=restart_ratio,
                        n_runs=FOA_RUNS_PER_GRID,
                        base_seed=1000
                    )

                    row = {
                        "num_job": num_job,
                        "num_mch": NUM_MCH,
                        "pop_size": pop_size,
                        "step_size": step_size,
                        "restart_ratio": restart_ratio,
                        "mean_obj": summary["mean_obj"],
                        "std_obj": summary["std_obj"],
                        "min_obj": summary["min_obj"],
                        "max_obj": summary["max_obj"],
                        "mean_time": summary["mean_time"],
                        "best_run_obj": summary["best_run_obj"],
                        "best_run_seed": summary["best_run_seed"],
                        "ga_obj_fixed": ga_result.objective,
                        "ga_time_fixed": safe_time_value(ga_result.comp_time),
                    }
                    all_grid_rows.append(row)

                    if summary["mean_obj"] < best_foa_mean_score:
                        best_foa_mean_score = summary["mean_obj"]
                        best_foa_params = (pop_size, step_size, restart_ratio)
                        best_foa_schedule = summary["best_run_sched"]
                        best_foa_history = summary["best_run_history"]
                        best_foa_seed = summary["best_run_seed"]
                        best_foa_best_run_obj = summary["best_run_obj"]

        print(f"[BEST FOA] job={num_job}")
        print("best mean objective:", best_foa_mean_score)
        print("best params:", best_foa_params)
        print("best single run objective:", best_foa_best_run_obj)
        print("best seed:", best_foa_seed)

        # -----------------
        # job size별 heatmap 저장
        # -----------------
        df_job = pd.DataFrame([r for r in all_grid_rows if r["num_job"] == num_job])
        plot_foa_grid_heatmaps(
            df_job=df_job,
            job_size=num_job,
            pop_candidates=FOA_POP_CANDIDATES,
            step_candidates=FOA_STEP_CANDIDATES,
            restart_candidates=FOA_RESTART_CANDIDATES,
            save_dir=OUTPUT_DIR
        )

        # -----------------
        # best FOA history 저장
        # -----------------
        plot_best_foa_history(
            history=best_foa_history,
            job_size=num_job,
            params=best_foa_params,
            save_dir=OUTPUT_DIR
        )

        # -----------------
        # best FOA gantt 저장
        # -----------------
        foa_gantt_path = os.path.join(OUTPUT_DIR, f"best_foa_schedule_job_{num_job}.png")
        plot_schedule_gantt(
            best_foa_schedule,
            title=(
                f"Best FOA Schedule (job={num_job})\n"
                f"pop={best_foa_params[0]}, step={best_foa_params[1]}, "
                f"restart={best_foa_params[2]}, best_run_obj={best_foa_best_run_obj:.2f}"
            ),
            save_path=foa_gantt_path
        )

        summary_rows.append({
            "num_job": num_job,
            "num_mch": NUM_MCH,
            "ga_obj": ga_result.objective,
            "ga_time": safe_time_value(ga_result.comp_time),
            "best_foa_mean_obj": best_foa_mean_score,
            "best_foa_best_run_obj": best_foa_best_run_obj,
            "best_foa_pop_size": best_foa_params[0],
            "best_foa_step_size": best_foa_params[1],
            "best_foa_restart_ratio": best_foa_params[2],
            "best_foa_seed": best_foa_seed,
            "ga_schedule_path": ga_gantt_path,
            "best_foa_schedule_path": foa_gantt_path,
        })

    # -----------------
    # 결과 저장
    # -----------------
    df_grid = pd.DataFrame(all_grid_rows)
    df_summary = pd.DataFrame(summary_rows)

    grid_csv = os.path.join(OUTPUT_DIR, "foa_grid_search_all_results.csv")
    summary_csv = os.path.join(OUTPUT_DIR, "foa_grid_search_summary.csv")

    df_grid.to_csv(grid_csv, index=False, encoding="utf-8-sig")
    df_summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print("\nSaved:", grid_csv)
    print("Saved:", summary_csv)

    print("\n=== SUMMARY ===")
    print(df_summary)

    plot_best_foa_vs_fixed_ga(df_summary, save_dir=OUTPUT_DIR)
    print("Done")