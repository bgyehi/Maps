import pickle
import time
import pandas as pd
import glob
import os
import copy
import matplotlib.pyplot as plt

from dispatching_all_methods import *  # SPT, EDD, Regret, ILS, VNS, SA, GA, Gurobi 포함


# -------------------------------------------
# deepcopy용 함수 (Solution/Instance 복제)
# -------------------------------------------
def deep_copy_solution(inst):
    return copy.deepcopy(inst)


# -------------------------------------------
# 최종 솔루션 시각화 함수
# -------------------------------------------
def plot_solution(instance, title="Solution", filename=None):
    """Vehicle routes를 2D 평면에 표시"""
    plt.figure(figsize=(7, 7))

    # depot이 없으면 customers[0]를 depot으로 사용
    if hasattr(instance, 'depot'):
        depot_x = instance.depot.x
        depot_y = instance.depot.y
    else:
        depot_x = instance.customers[0].x
        depot_y = instance.customers[0].y

    # depot 표시
    plt.scatter(depot_x, depot_y, c='red', s=120, marker='s', label="Depot")

    # 고객 표시
    xs = [c.x for c in instance.customers]
    ys = [c.y for c in instance.customers]
    plt.scatter(xs, ys, c='blue', s=40, label="Customers")

    # 라우팅 선 연결
    for v in instance.vehicles:
        if len(v.route) >= 2:
            rx = [instance.customers[idx].x for idx in v.route]
            ry = [instance.customers[idx].y for idx in v.route]
            plt.plot(rx, ry, alpha=0.6)

    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)

    if filename:
        plt.savefig(filename, dpi=150)
        plt.close()
    else:
        plt.show()


# -------------------------------------------
# 인스턴스 실행 시작
# -------------------------------------------

instance_files = sorted(glob.glob("hw2/instance_*.prob"))
results = []

os.makedirs("results_fig", exist_ok=True)

for prob_file in instance_files:
    print(f"\n{'=' * 70}")
    print(f"Processing: {prob_file}")
    print('=' * 70)

    with open(prob_file, "rb") as f:
        instance_orig = pickle.load(f)

    n = len(instance_orig.customers)
    candidates = []

    # -----------------------------------------------------------
    # 1. 초기해들
    # -----------------------------------------------------------
    print("[1] 초기해 생성 중...")

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

    # -----------------------------------------------------------
    # 2. 메타휴리스틱 (각 초기해에서 출발)
    # -----------------------------------------------------------
    print("[2] 메타휴리스틱 최적화 중...")

    inst_ils = deep_copy_solution(inst_regret)
    sol_ils = ils_optimize(inst_ils, time_limit=3.0)
    candidates.append(("ILS", sol_ils.objective, inst_ils))
    print(f"  ILS: {sol_ils.objective:.2f}")

    inst_vns = deep_copy_solution(inst_regret)
    sol_vns = vns_optimize(inst_vns, time_limit=3.0)
    candidates.append(("VNS", sol_vns.objective, inst_vns))
    print(f"  VNS: {sol_vns.objective:.2f}")

    inst_sa = deep_copy_solution(inst_edd)
    sol_sa = sa_optimize(inst_sa, time_limit=3.0)
    candidates.append(("SA", sol_sa.objective, inst_sa))
    print(f"  SA: {sol_sa.objective:.2f}")

    inst_ga = deep_copy_solution(instance_orig)
    sol_ga = ga_optimize(inst_ga, time_limit=3.0)
    candidates.append(("GA", sol_ga.objective, inst_ga))
    print(f"  GA: {sol_ga.objective:.2f}")

    # -----------------------------------------------------------
    # 3. Gurobi (작은 문제만)
    # -----------------------------------------------------------
    if n <= 15:
        print("[3] Gurobi 최적화 중...")
        inst_gurobi = deep_copy_solution(instance_orig)
        sol_gurobi = gurobi_optimize(inst_gurobi, time_limit=300)
        if sol_gurobi:
            candidates.append(("Gurobi", sol_gurobi.objective, inst_gurobi))
            print(f"  Gurobi: {sol_gurobi.objective:.2f}")

    # -----------------------------------------------------------
    # 4. 최선 선택
    # -----------------------------------------------------------
    best_method, best_obj, best_inst = min(candidates, key=lambda x: x[1])
    print(f"\n[최선] {best_method}: {best_obj:.2f}")

    # -----------------------------------------------------------
    # 5. 최종 솔루션 그림 저장
    # -----------------------------------------------------------
    fig_name = f"results_fig/{os.path.basename(prob_file).replace('.prob','')}_{best_method}.png"
    plot_solution(
        best_inst,
        title=f"{os.path.basename(prob_file)} | {best_method} | tardiness={best_obj:.2f}",
        filename=fig_name
    )
    print(f"  ➜ 그림 저장됨: {fig_name}")

    # -----------------------------------------------------------
    # 6. 기록 저장
    # -----------------------------------------------------------
    results.append({
        "Problem": os.path.basename(prob_file),
        "Best_Method": best_method,
        "Best_Tardiness": best_obj,
        "SPT": sol_spt.objective,
        "EDD": sol_edd.objective,
        "Regret-3": sol_regret.objective,
        "ILS": sol_ils.objective,
        "VNS": sol_vns.objective,
        "SA": sol_sa.objective,
        "GA": sol_ga.objective
    })
    if n <= 15 and sol_gurobi:
        results[-1]["Gurobi"] = sol_gurobi.objective

# ===============================================================
# 전체 결과 요약
# ===============================================================
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
