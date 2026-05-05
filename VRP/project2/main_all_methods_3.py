import pickle
import time
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt

from dispatching_all_methods import *


# ==============================
# 플롯 저장 폴더 생성
# ==============================
if not os.path.exists("plots"):
    os.makedirs("plots")


# ==============================
# 성능 개선 그래프 저장 함수
# ==============================
def plot_trajectory(cost_list, title, save_path):
    if len(cost_list) == 0:
        return

    plt.figure(figsize=(6, 4))
    plt.plot(cost_list, linewidth=2)
    plt.xlabel("Iteration / Improvement #")
    plt.ylabel("Objective (Tardiness)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ==============================
# 각 instance 실행
# ==============================
instance_files = sorted(glob.glob("hw2/instance_*.prob"))
results = []

for prob_file in instance_files:
    print(f"\n{'=' * 70}")
    print(f"Processing: {prob_file}")
    print('=' * 70)

    with open(prob_file, "rb") as f:
        instance_orig = pickle.load(f)

    prob_name = os.path.basename(prob_file)

    # =====================================
    # 1) 초기해 생성
    # =====================================
    print("[1] 초기해 생성 중...")
    candidates = []

    inst_spt = deep_copy_solution(instance_orig)
    sol_spt = dispatch_spt(inst_spt)
    candidates.append(("SPT", sol_spt.objective, inst_spt))
    print(f"  SPT: {sol_spt.objective:.2f}")

    inst_edd = deep_copy_solution(instance_orig)
    sol_edd = dispatch_edd(inst_edd)
    candidates.append(("EDD", sol_edd.objective, inst_edd))
    print(f"  EDD: {sol_edd.objective:.2f}")

    inst_regret = deep_copy_solution(instance_orig)
    sol_regret = dispatch_regret_k(inst_regret, k=3)
    candidates.append(("Regret-3", sol_regret.objective, inst_regret))
    print(f"  Regret-3: {sol_regret.objective:.2f}")

    # =====================================
    # 2) 메타휴리스틱 최적화 (SA/Gurobi 제외됨)
    # =====================================
    print("[2] 메타휴리스틱 최적화 중...")

    # ---- ILS ----
    inst_ils = deep_copy_solution(inst_regret)
    sol_ils, ils_trace = ils_optimize(inst_ils, time_limit=3.0)
    candidates.append(("ILS", sol_ils.objective, inst_ils))
    print(f"  ILS: {sol_ils.objective:.2f}")

    # 성능 그래프 저장
    plot_trajectory(
        ils_trace,
        f"{prob_name} ILS Performance",
        f"plots/{prob_name}_ILS_perf.png"
    )

    # ---- VNS ----
    inst_vns = deep_copy_solution(inst_regret)
    sol_vns, vns_trace = vns_optimize(inst_vns, time_limit=3.0)
    candidates.append(("VNS", sol_vns.objective, inst_vns))
    print(f"  VNS: {sol_vns.objective:.2f}")

    plot_trajectory(
        vns_trace,
        f"{prob_name} VNS Performance",
        f"plots/{prob_name}_VNS_perf.png"
    )

    # ---- GA ----
    inst_ga = deep_copy_solution(instance_orig)
    sol_ga, ga_trace = ga_optimize(inst_ga, time_limit=3.0)
    candidates.append(("GA", sol_ga.objective, inst_ga))
    print(f"  GA: {sol_ga.objective:.2f}")

    plot_trajectory(
        ga_trace,
        f"{prob_name} GA Performance",
        f"plots/{prob_name}_GA_perf.png"
    )

    # =====================================
    # 3) 최선해 선택
    # =====================================
    best_method, best_obj, best_inst = min(candidates, key=lambda x: x[1])
    print(f"\n[최선] {best_method}: {best_obj:.2f}")

    # =====================================
    # 4) 라우트 플롯 저장
    # =====================================
    try:
        fig = plot_routes_solution(best_inst, title=f"{prob_name} | Best={best_method} | Obj={best_obj:.2f}")
        fig.savefig(f"plots/{prob_name}_{best_method}.png")
        plt.close(fig)
        print(f"  → Route plot saved: plots/{prob_name}_{best_method}.png")
    except Exception as e:
        print("Plot failed:", e)

    # =====================================
    # 결과 기록
    # =====================================
    results.append({
        "Problem": prob_name,
        "Best_Method": best_method,
        "Best_Tardiness": best_obj,
        "SPT": sol_spt.objective,
        "EDD": sol_edd.objective,
        "Regret-3": sol_regret.objective,
        "ILS": sol_ils.objective,
        "VNS": sol_vns.objective,
        "GA": sol_ga.objective
    })


# ==============================
# 5) 결과 요약 테이블
# ==============================
df = pd.DataFrame(results)
df.set_index("Problem", inplace=True)

print("\n" + "=" * 100)
print("최종 요약 결과 (Summary Results)")
print("=" * 100)
print(df.to_markdown(floatfmt=".2f"))

avg_best = df["Best_Tardiness"].mean()
total_best = df["Best_Tardiness"].sum()

print("\n" + "-" * 60)
print("전체 평균/총합 (Average/Total Performance)")
print(f"- 평균 최적 Tardiness : {avg_best:.2f}")
print(f"- 총 최적 Tardiness : {total_best:.2f}")
print("=" * 100)

df.to_csv("results_all_methods.csv")
print("\nSaved results_all_methods.csv ✅")
print("All plots saved in /plots folder!")
