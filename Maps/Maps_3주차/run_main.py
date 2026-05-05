from module import Job, Machine, Instance
from milp_cplex_only import milp_scheduling

# -------------------------
# 1) Job 생성
# -------------------------
due_dates = [26, 20, 41, 22, 26]
weights = [2, 1, 8, 8, 7]

jobs = []
for i in range(5):
    job = Job(i)
    job.due = due_dates[i]
    job.weight = weights[i]
    jobs.append(job)

# -------------------------
# 2) Machine 생성
# -------------------------
machines = [Machine(0), Machine(1)]

# -------------------------
# 3) Processing Time
# -------------------------
ptime = [
    [57, 50, 32, 42, 55],  # Machine 0
    [40, 53, 49, 40, 32],  # Machine 1
]

# -------------------------
# 4) Setup Time
# setup[k][i][j] = machine k에서 job i 다음 job j로 갈 때 setup
# job 번호 1~5를 내부에서는 0~4로 씀
# -------------------------
setup = [
    [  # Machine 0
        [0, 23, 65, 75, 26],
        [62, 0, 30, 38, 19],
        [50, 84, 0, 52, 18],
        [41, 89, 73, 0, 52],
        [85, 14, 50, 71, 0],
    ],
    [  # Machine 1
        [0, 45, 50, 26, 81],
        [68, 0, 75, 55, 88],
        [22, 38, 0, 45, 78],
        [81, 34, 83, 0, 38],
        [20, 44, 25, 19, 0],
    ],
]

for k in range(2):
    machines[k].ptime = ptime[k]
    machines[k].setup = setup[k]

# -------------------------
# 5) Instance 생성
# -------------------------
inst = Instance(jobs, machines, ptime, setup)
inst.with_setup = True
inst.objective = "wT"

# -------------------------
# 6) CPLEX 실행
# -------------------------
result = milp_scheduling(inst, time_limit=3600)

# -------------------------
# 7) 결과 출력
# -------------------------
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
            setup_time = inst.setup[m.ID][prev_job.ID][job.ID]

        tardiness = max(job.end - job.due, 0)
        wt = job.weight * tardiness

        print(
            f"  Job {job.ID + 1} | start={job.start:.0f}, end={job.end:.0f}, "
            f"due={job.due}, weight={job.weight}, setup={setup_time}, "
            f"tardiness={tardiness:.0f}, weighted tardiness={wt:.0f}"
        )