import pickle
import glob
import pandas as pd
import os
from dispatching_all_methods import *
try:
    from visualize import plot_solution
except ImportError:
    plot_solution = None

instance_files = sorted(glob.glob("hw2/instance_*.prob"))
results = []
GUROBI_TIME_LIMIT = 600
META_TIME_LIMIT = 5.0

for prob_file in instance_files:
    with open(prob_file, "rb") as f:
        instance_orig = pickle.load(f)
    n = len(instance_orig.customers)
    m = len(instance_orig.vehicles)
    print(f"\n=== {os.path.basename(prob_file)} / Customers:{n} / Vehicles:{m} ===")
    candidates = []
    inst_spt = deep_copy_solution(instance_orig)
    sol_spt = dispatch_spt(inst_spt)
    candidates.append(("SPT", sol_spt.objective, inst_spt))
    inst_edd = deep_copy_solution(instance_orig)
    sol_edd = dispatch_edd(inst_edd)
    candidates.append(("EDD", sol_edd.objective, inst_edd))
    inst_regret = deep_copy_solution(instance_orig)
    sol_regret = dispatch_regret_k(inst_regret, k=3)
    candidates.append(("Regret-3", sol_regret.objective, inst_regret))
    inst_ils = deep_copy_solution(inst_regret)
    sol_ils = ils_optimize(inst_ils, time_limit=META_TIME_LIMIT)
    candidates.append(("ILS", sol_ils.objective, inst_ils))
    inst_vns = deep_copy_solution(inst_regret)
    sol_vns = vns_optimize(inst_vns, time_limit=META_TIME_LIMIT)
    candidates.append(("VNS", sol_vns.objective, inst_vns))
    inst_sa = deep_copy_solution(inst_edd)
    sol_sa = sa_optimize(inst_sa, time_limit=META_TIME_LIMIT)
    candidates.append(("SA", sol_sa.objective, inst_sa))
    inst_ga = deep_copy_solution(instance_orig)
    sol_ga = ga_optimize(inst_ga, time_limit=META_TIME_LIMIT)
    candidates.append(("GA", sol_ga.objective, inst_ga))
    inst_gurobi = deep_copy_solution(instance_orig)
    sol_gurobi = gurobi_optimize(inst_gurobi, time_limit=GUROBI_TIME_LIMIT)
    if sol_gurobi:
        candidates.append(("Gurobi", sol_gurobi.objective, inst_gurobi))
        gurobi_obj = sol_gurobi.objective
    else:
        gurobi_obj = float('inf')
    best_method, best_obj, best_inst = min(candidates, key=lambda x: x[1])
    if plot_solution:
        try:
            save_path_img = prob_file.replace('.prob', f'_{best_method}_bestroute.png')
            plot_solution(best_inst, annotate=True, show_time_windows=True, arrows=True, save_path=save_path_img)
            print(f"[그림 저장] {save_path_img}")
        except Exception as e:
            print(f"[그림 오류]: {e}")
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
        "Gurobi": gurobi_obj
    }
    results.append(result_row)

df = pd.DataFrame(results)
df.set_index("Problem", inplace=True)
print(df.to_markdown(floatfmt=".2f"))
avg_best = df["Best_Tardiness"].mean()
total_best = df["Best_Tardiness"].sum()
print(f"\n전체 평균: {avg_best:.2f}, 전체 총합: {total_best:.2f}")
df.to_csv("results_all_methods.csv")
