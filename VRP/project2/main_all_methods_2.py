import pickle
import glob
import pandas as pd
import os
import time
from dispatching_all_methods_2 import *
try:
    from visualize import plot_solution
except ImportError:
    plot_solution = None

instance_files = sorted(glob.glob("hw2/instance_*.prob"))
results = []
GUROBI_TIME_LIMIT = 600
META_TIME_LIMIT = 10.0

total_start_time = time.time()

for idx, prob_file in enumerate(instance_files, 1):
    print(f"\n{'='*60}\n[{idx}/{len(instance_files)}] {os.path.basename(prob_file)}")
    start = time.time()
    with open(prob_file, "rb") as f:
        instance_orig = pickle.load(f)
    n = len(instance_orig.customers)
    m = len(instance_orig.vehicles)
    print(f"Customers={n} | Vehicles={m}")

    methods = []

    inst_edd = deep_copy_solution(instance_orig)
    t0 = time.time()
    sol_edd = dispatch_greedy(inst_edd, rule="EDD")
    t_edd = time.time() - t0
    print(f"  EDD   : {sol_edd.objective:.1f} (시간 {t_edd:.2f}s)")
    methods.append(("EDD", sol_edd.objective, inst_edd))

    inst_spt = deep_copy_solution(instance_orig)
    t0 = time.time()
    sol_spt = dispatch_greedy(inst_spt, rule="SPT")
    t_spt = time.time() - t0
    print(f"  SPT   : {sol_spt.objective:.1f} (시간 {t_spt:.2f}s)")
    methods.append(("SPT", sol_spt.objective, inst_spt))

    inst_ils = deep_copy_solution(instance_orig)
    t0 = time.time()
    sol_ils = ils_optimize(inst_ils, time_limit=META_TIME_LIMIT)
    t_ils = time.time() - t0
    print(f"  ILS   : {sol_ils.objective:.1f} (시간 {t_ils:.2f}s)")
    methods.append(("ILS", sol_ils.objective, inst_ils))

    inst_vns = deep_copy_solution(instance_orig)
    t0 = time.time()
    sol_vns = vns_optimize(inst_vns, time_limit=META_TIME_LIMIT)
    t_vns = time.time() - t0
    print(f"  VNS   : {sol_vns.objective:.1f} (시간 {t_vns:.2f}s)")
    methods.append(("VNS", sol_vns.objective, inst_vns))

    inst_sa = deep_copy_solution(instance_orig)
    t0 = time.time()
    sol_sa = sa_optimize(inst_sa, time_limit=META_TIME_LIMIT)
    t_sa = time.time() - t0
    print(f"  SA    : {sol_sa.objective:.1f} (시간 {t_sa:.2f}s)")
    methods.append(("SA", sol_sa.objective, inst_sa))

    inst_gurobi = deep_copy_solution(instance_orig)
    t0 = time.time()
    sol_gurobi = gurobi_optimize(inst_gurobi, time_limit=GUROBI_TIME_LIMIT)
    t_gurobi = time.time() - t0
    if sol_gurobi:
        print(f"  Gurobi: {sol_gurobi.objective:.1f} (시간 {t_gurobi/60:.2f}분)")
        methods.append(("Gurobi", sol_gurobi.objective, inst_gurobi))
    else:
        print(f"  Gurobi: 해 없음 (시간 {t_gurobi/60:.2f}분)")

    best_method, best_obj, best_inst = min(methods, key=lambda x: x[1])

    if plot_solution:
        try:
            img_path = prob_file.replace('.prob', f'_{best_method}_best.png')
            plot_solution(best_inst, annotate=True, show_time_windows=True, arrows=True, save_path=img_path)
            print(f"[그림 저장] {img_path}")
        except Exception as e:
            print(f"[그림 오류]: {e}")

    elapsed = time.time() - start

    results.append({
        "Problem": os.path.basename(prob_file),
        "Customers": n,
        "Vehicles": m,
        "Best_Method": best_method,
        "Best_Tardiness": best_obj,
        "EDD": sol_edd.objective,
        "SPT": sol_spt.objective,
        "ILS": sol_ils.objective,
        "VNS": sol_vns.objective,
        "SA": sol_sa.objective,
        "Gurobi": sol_gurobi.objective if sol_gurobi else None,
        "TotalTime(s)": elapsed
    })
    print(f"=> 문제 소요시간: {elapsed:.2f}초 | Best={best_method}")

total_elapsed = time.time() - total_start_time
df = pd.DataFrame(results)
df.set_index("Problem", inplace=True)
print("\n--- 전체 결과 표 ---\n")
print(df.to_markdown(floatfmt=".2f"))
print(f"\n전체 평균(최적): {df['Best_Tardiness'].mean():.2f}")
print(f"전체 총합(최적): {df['Best_Tardiness'].sum():.2f}")
print(f"평균 소요시간 : {df['TotalTime(s)'].mean():.2f}s | 전체: {total_elapsed/60:.1f}분")
df.to_csv("results_all_methods_vrp.csv")
print(f"\n결과 파일 저장 완료: results_all_methods_vrp.csv ✅")
