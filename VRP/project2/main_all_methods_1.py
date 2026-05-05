import pickle
import glob
import pandas as pd
import os
import time
from dispatching_all_methods_1 import *

try:
    from visualize import plot_solution
except ImportError:
    plot_solution = None

instance_files = sorted(glob.glob("hw2/instance_*.prob"))
results = []
GUROBI_TIME_LIMIT = 600
META_TIME_LIMIT = 10.0

total_start_time = time.time()

for i, prob_file in enumerate(instance_files, 1):
    prob_start = time.time()
    with open(prob_file, "rb") as f:
        instance_orig = pickle.load(f)
    n = len(instance_orig.customers)
    m = len(instance_orig.vehicles)
    print(f"\n{'=' * 70}\n[{i}/{len(instance_files)}] {os.path.basename(prob_file)} | Customers:{n} | Vehicles:{m}")
    candidates = []

    # SPT
    inst_spt = deep_copy_solution(instance_orig)
    timer1 = time.time()
    sol_spt = dispatch_spt(inst_spt)
    t_spt = time.time() - timer1
    candidates.append(("SPT", sol_spt.objective, inst_spt))
    print(f"  SPT      : Tard={sol_spt.objective:.1f} (시간={t_spt:.2f}s)")

    # EDD
    inst_edd = deep_copy_solution(instance_orig)
    timer2 = time.time()
    sol_edd = dispatch_edd(inst_edd)
    t_edd = time.time() - timer2
    candidates.append(("EDD", sol_edd.objective, inst_edd))
    print(f"  EDD      : Tard={sol_edd.objective:.1f} (시간={t_edd:.2f}s)")

    # Regret-3
    inst_regret = deep_copy_solution(instance_orig)
    timer3 = time.time()
    sol_regret = dispatch_regret_k(inst_regret, k=3)
    t_regret = time.time() - timer3
    candidates.append(("Regret-3", sol_regret.objective, inst_regret))
    print(f"  Regret-3 : Tard={sol_regret.objective:.1f} (시간={t_regret:.2f}s)")

    # ILS
    inst_ils = deep_copy_solution(inst_regret)
    timer4 = time.time()
    sol_ils = ils_optimize(inst_ils, time_limit=META_TIME_LIMIT)
    t_ils = time.time() - timer4
    candidates.append(("ILS", sol_ils.objective, inst_ils))
    print(f"  ILS      : Tard={sol_ils.objective:.1f} (시간={t_ils:.2f}s)")

    # VNS
    inst_vns = deep_copy_solution(inst_regret)
    timer5 = time.time()
    sol_vns = vns_optimize(inst_vns, time_limit=META_TIME_LIMIT)
    t_vns = time.time() - timer5
    candidates.append(("VNS", sol_vns.objective, inst_vns))
    print(f"  VNS      : Tard={sol_vns.objective:.1f} (시간={t_vns:.2f}s)")

    # SA
    inst_sa = deep_copy_solution(inst_edd)
    timer6 = time.time()
    sol_sa = sa_optimize(inst_sa, time_limit=META_TIME_LIMIT)
    t_sa = time.time() - timer6
    candidates.append(("SA", sol_sa.objective, inst_sa))
    print(f"  SA       : Tard={sol_sa.objective:.1f} (시간={t_sa:.2f}s)")

    # GA
    inst_ga = deep_copy_solution(instance_orig)
    timer7 = time.time()
    sol_ga = ga_optimize(inst_ga, time_limit=META_TIME_LIMIT)
    t_ga = time.time() - timer7
    candidates.append(("GA", sol_ga.objective, inst_ga))
    print(f"  GA       : Tard={sol_ga.objective:.1f} (시간={t_ga:.2f}s)")

    # Gurobi
    inst_gurobi = deep_copy_solution(instance_orig)
    timer8 = time.time()
    sol_gurobi = gurobi_optimize(inst_gurobi, time_limit=GUROBI_TIME_LIMIT)
    t_gurobi = time.time() - timer8
    if sol_gurobi:
        candidates.append(("Gurobi", sol_gurobi.objective, inst_gurobi))
        gurobi_obj = sol_gurobi.objective
        print(f"  Gurobi   : Tard={gurobi_obj:.1f} (시간={t_gurobi / 60:.1f}분)")
    else:
        gurobi_obj = float('inf')
        print(f"  Gurobi   : 미해결 (시간={t_gurobi / 60:.1f}분)")

    # 최적
    best_method, best_obj, best_inst = min(candidates, key=lambda x: x[1])
    print(f" → [Best] {best_method} | Obj={best_obj:.1f}")

    if plot_solution:
        try:
            save_path_img = prob_file.replace('.prob', f'_{best_method}_bestroute.png')
            plot_solution(best_inst, annotate=True, show_time_windows=True, arrows=True, save_path=save_path_img)
            print(f"[그림 저장] {save_path_img}")
        except Exception as e:
            print(f"[그림 오류]: {e}")

    prob_time = time.time() - prob_start

    result_row = {
        "Problem": os.path.basename(prob_file),
        "Customers": n,
        "Vehicles": m,
        "Best_Method": best_method,
        "Best_Tardiness": best_obj,
        "SPT": sol_spt.objective,
        "EDD": sol_edd.objective,
        "Regret-3": sol_regret.objective,
        "ILS": sol_ils.objective,
        "VNS": sol_vns.objective,
        "SA": sol_sa.objective,
        "GA": sol_ga.objective,
        "Gurobi": gurobi_obj,
        "TotalTime(s)": prob_time
    }
    results.append(result_row)
    print(f" => 문제별 소요시간: {prob_time:.2f}초")

# 전체 시간/평균
total_elapsed = time.time() - total_start_time

df = pd.DataFrame(results)
df.set_index("Problem", inplace=True)
print("\n[===== 전체 결과 표 =====]\n")
print(df.to_markdown(floatfmt=".2f"))
avg_best = df["Best_Tardiness"].mean()
total_best = df["Best_Tardiness"].sum()
avg_time = df["TotalTime(s)"].mean()
print(f"\n[전체 최적 Tardiness] 평균: {avg_best:.2f}, 총합: {total_best:.2f}")
print(f"[평균 실행시간/문제] {avg_time:.2f}초, [전체 총 실행시간] {total_elapsed / 60:.1f}분")

df.to_csv("results_all_methods.csv")
print(f"\n결과 파일 저장 완료: results_all_methods.csv ✅")
