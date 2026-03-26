from module_modified import load_instance_from_csv
from milp_modified import milp_scheduling


JOBS_CSV = "jobs.csv"
SETUP_CSV = "setup_times.csv"
TIME_LIMIT = 3600


def print_result(result):
    print("\n===== SOLUTION SUMMARY =====")
    print("Objective :", result.objective)
    print("Status    :", result.status)
    print("SolveTime :", result.comp_time)

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


if __name__ == "__main__":
    inst = load_instance_from_csv(JOBS_CSV, SETUP_CSV)
    result = milp_scheduling(inst, TIME_LIMIT)
    print_result(result)