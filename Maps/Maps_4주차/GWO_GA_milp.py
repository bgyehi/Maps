from module_modified import load_instance_from_csv
from milp_modified import milp_scheduling
from GWO import grey_wolf_scheduling
from GWO_vs_GA import ga_scheduling
import pandas as pd

JOBS_CSV = "jobs.csv"
SETUP_CSV = "setup_times.csv"
TIME_LIMIT = 300


def print_result(name, result):
    print(f"\n===== {name} =====")
    print("Objective :", result.objective)
    print("Status    :", result.status)
    print("SolveTime :", result.comp_time)

    for m in result.instance.machine_list:
        assigned = sorted(m.assigned, key=lambda j: j.start)
        print(f"Machine {m.ID}")
        for job in assigned:
            tardiness = max(job.end - job.due, 0)
            wt = job.weight * tardiness
            print(
                f"  Job {job.ID + 1} | start={job.start:.0f}, end={job.end:.0f}, "
                f"due={job.due}, weight={job.weight}, "
                f"tardiness={tardiness:.0f}, weighted tardiness={wt:.0f}"
            )


if __name__ == "__main__":
    milp_inst = load_instance_from_csv(JOBS_CSV, SETUP_CSV)
    gwo_inst = load_instance_from_csv(JOBS_CSV, SETUP_CSV)
    ga_inst = load_instance_from_csv(JOBS_CSV, SETUP_CSV)

    milp_res = milp_scheduling(milp_inst, time_limit=TIME_LIMIT)
    gwo_res = grey_wolf_scheduling(gwo_inst, time_limit=TIME_LIMIT, n_wolves=30, seed=42)
    ga_res = ga_scheduling(ga_inst, time_limit=TIME_LIMIT, pop_size=50, elite_size=5, seed=42)

    print_result("MILP", milp_res)
    print_result("GWO", gwo_res)
    print_result("GA", ga_res)

    df = pd.DataFrame([
        {"Algorithm": "MILP", "Objective": milp_res.objective, "Runtime": milp_res.comp_time, "Status": milp_res.status},
        {"Algorithm": "GWO", "Objective": gwo_res.objective, "Runtime": gwo_res.comp_time, "Status": gwo_res.status},
        {"Algorithm": "GA", "Objective": ga_res.objective, "Runtime": ga_res.comp_time, "Status": ga_res.status},
    ])
    df.to_csv("compare_gwo_ga_milp.csv", index=False)
    print("\nSaved: compare_gwo_ga_milp.csv")