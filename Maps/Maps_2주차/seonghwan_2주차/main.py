import os
from module import load_or_create_instance
from milp import milp_scheduling


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))

    json_path = os.path.join(base_dir, "instance_1.json")
    jobs_csv = os.path.join(base_dir, "jobs.csv")
    setup_csv = os.path.join(base_dir, "setup_times.csv")

    inst = load_or_create_instance(
        json_path=json_path,
        jobs_csv=jobs_csv,
        setup_csv=setup_csv
    )

    result = milp_scheduling(inst, 3600)

    print("Objective (wT):", result.objective)
    print("Status:", result.status)
    print("Solve Time:", result.comp_time)