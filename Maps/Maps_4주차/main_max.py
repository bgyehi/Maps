from module_ import load_instance_from_csv
from module_max import create_random_instance_from_template
from milp_max import milp_scheduling


TEMPLATE_JOBS_CSV = "jobs.csv"
TEMPLATE_SETUP_CSV = "setup_times.csv"

TIME_LIMIT = 300

# 교수님 의도대로 machine=2 고정, job 수만 증가
JOB_SIZES = [5, 10, 15, 20, 25, 30, 35, 40]
SEEDS = [0, 1, 2]


def print_result(result):
    print("\n===== SOLUTION SUMMARY =====")
    print("Objective :", result.objective)
    print("Status    :", result.status)
    print("SolveTime :", result.comp_time)
    print("Gap       :", getattr(result, "gap", None))

    print("\n===== JOB SEQUENCE =====")
    for m in result.instance.machine_list:
        assigned = sorted(m.assigned, key=lambda j: j.start)
        print(f"Machine {m.ID}")
        for idx, job in enumerate(assigned):
            if idx == 0:
                setup_time = 0
            else:
                prev_job = assigned[idx - 1]
                setup_time = result.instance.setup[m.ID][prev_job.ID][job.ID]

            tardiness = max(job.end - job.due, 0)
            wt = job.weight * tardiness
            print(
                f"  Job {job.ID + 1} | start={job.start:.0f}, end={job.end:.0f}, "
                f"due={job.due}, weight={job.weight}, setup={setup_time}, "
                f"tardiness={tardiness:.0f}, weighted tardiness={wt:.0f}"
            )


def run_benchmark():
    template_inst = load_instance_from_csv(TEMPLATE_JOBS_CSV, TEMPLATE_SETUP_CSV)

    if template_inst.numMch < 2:
        raise ValueError("템플릿 CSV에는 최소 2대의 machine 정보가 있어야 합니다.")

    all_rows = []

    for n_jobs in JOB_SIZES:
        print("\n" + "=" * 80)
        print(f"JOB SIZE = {n_jobs}")
        print("=" * 80)

        for seed in SEEDS:
            print(f"\n--- instance: n_jobs={n_jobs}, seed={seed} ---")

            inst = create_random_instance_from_template(
                template_inst=template_inst,
                n_jobs=n_jobs,
                seed=seed,
                p_low=1,
                p_high=20,
                s_low=0,
                s_high=10,
                w_low=1,
                w_high=10,
                due_tightness=0.6
            )

            result = milp_scheduling(inst, TIME_LIMIT)

            row = {
                "n_jobs": n_jobs,
                "seed": seed,
                "objective": result.objective,
                "status": result.status,
                "solve_time": result.comp_time,
                "gap": getattr(result, "gap", None),
            }
            all_rows.append(row)

            print(
                f"n_jobs={row['n_jobs']:>3}, seed={row['seed']:>2}, "
                f"obj={row['objective']}, status={row['status']}, "
                f"time={row['solve_time']:.2f}, gap={row['gap']}"
            )

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    for row in all_rows:
        print(
            f"n_jobs={row['n_jobs']:>3}, seed={row['seed']:>2}, "
            f"obj={row['objective']}, status={row['status']:<20}, "
            f"time={row['solve_time']:>8.2f}, gap={row['gap']}"
        )

    return all_rows


if __name__ == "__main__":
    run_benchmark()