import pickle
import time
import pandas as pd
import glob
import os
import copy
import matplotlib.pyplot as plt

# 최종 사용 알고리즘만 import
from dispatching_final_1 import (
    dispatch_spt,
    dispatch_regret_k,
    ils_vns_optimize,
    calculate_total_tardiness,
)

# -------------------------------------------
# deepcopy용 함수
# -------------------------------------------

def deep_copy_instance(inst):
    return copy.deepcopy(inst)


# -------------------------------------------
# 최종 솔루션 시각화 함수 (schedules 기반)
# -------------------------------------------

def plot_solution(instance, title="Solution", filename=None):
    plt.figure(figsize=(7, 7))

    # depot 좌표
    depot_x, depot_y = instance.vehicles[0].loc
    plt.scatter(depot_x, depot_y, c='black', s=120, marker='s', label='Depot')

    # 고객 좌표
    xs = [c.loc[0] for c in instance.customers]
    ys = [c.loc[1] for c in instance.customers]
    plt.scatter(xs, ys, c='gray', s=25, label='Customers')

    # 차량별 경로
    for v in instance.vehicles:
        if len(v.schedules) == 0:
            continue
        rx = [depot_x] + [c.loc[0] for c in v.schedules]
        ry = [depot_y] + [c.loc[1] for c in v.schedules]
        plt.plot(rx, ry, marker='o', alpha=0.7)

    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.legend()

    if filename:
        plt.savefig(filename, dpi=150)
        plt.close()
    else:
        plt.show()


# ===============================================================
# Instance 실행
# ===============================================================

instance_files = sorted(glob.glob("hw2/instance_*.prob"))
os.makedirs("results_fig", exist_ok=True)

results = []

total_start = time.time()

for prob_file in instance_files:
    print(f"\n{'=' * 70}")
    print(f"Processing: {prob_file}")
    print('=' * 70)

    with open(prob_file, "rb") as f:
        instance_orig = pickle.load(f)

    # -----------------------------
    # SPT 초기해
    # -----------------------------
    t0 = time.time()
    inst_spt = deep_copy_instance(instance_orig)
    sol_spt = dispatch_spt(inst_spt)
    time_spt = time.time() - t0

    # -----------------------------
    # Regret-k 초기해
    # -----------------------------
    t0 = time.time()
    inst_regret = deep_copy_instance(instance_orig)
    sol_regret = dispatch_regret_k(inst_regret, k=3)
    time_regret = time.time() - t0

    # -----------------------------
    # 초기해 선택
    # -----------------------------
    if sol_spt.objective <= sol_regret.objective:
        base_inst = inst_spt
        base_name = "SPT"
        base_obj = sol_spt.objective
    else:
        base_inst = inst_regret
        base_name = "Regret-3"
        base_obj = sol_regret.objective

    print(f"Initial Best: {base_name} | Tardiness = {base_obj:.2f}")

    # -----------------------------
    # ILS + VNS 최적화
    # -----------------------------
    t0 = time.time()
    sol_final = ils_vns_optimize(base_inst, time_limit=5.0)
    elapsed = time.time() - t0

    best_obj = sol_final.objective

    print(f"Final: ILS-VNS | Tardiness = {best_obj:.2f} | Time = {elapsed:.2f}s")

    # -----------------------------
    # 그림 저장
    # -----------------------------
    fig_name = f"results_fig/{os.path.basename(prob_file).replace('.prob','')}_final.png"
    plot_solution(
        base_inst,
        title=f"{os.path.basename(prob_file)} | ILS-VNS | tardiness={best_obj:.2f}",
        filename=fig_name,
    )

    # -----------------------------
    # 결과 저장
    # -----------------------------
    results.append({
        "Problem": os.path.basename(prob_file),
        "Initial_SPT": sol_spt.objective,
        "Initial_Regret3": sol_regret.objective,
        "Best_Tardiness": best_obj,
        "Solve_Time": elapsed,
    })

# ===============================================================
# 전체 결과 요약
# ===============================================================

df = pd.DataFrame(results)
df.set_index("Problem", inplace=True)

print("\n" + "=" * 100)
print("최종 요약 결과 (Summary Results)")
print("=" * 100)
print(df.to_markdown(floatfmt=".2f"))

avg_tard = df["Best_Tardiness"].mean()
total_tard = df["Best_Tardiness"].sum()

avg_time = df["Solve_Time"].mean()
total_time = df["Solve_Time"].sum()

print("\n" + "-" * 60)
print("전체 평균/총합 성능")
print(f"- 평균 Tardiness : {avg_tard:.2f}")
print(f"- 총 Tardiness   : {total_tard:.2f}")
print(f"- 평균 실행시간  : {avg_time:.2f} s")
print(f"- 총 실행시간    : {total_time:.2f} s")
print("=" * 100)

# CSV 저장
df.to_csv("results_final_summary.csv")
print("Saved results_final_summary.csv ✅")
