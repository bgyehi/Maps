import pickle
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt

from dispatching_all_methods_ import (
    deep_copy_solution,
    dispatch_spt,
    dispatch_edd,
    dispatch_regret_k,
    ils_optimize,
    vns_optimize
)


def plot_solution(instance, title, filename):
    plt.figure(figsize=(7, 7))
    for v in instance.vehicles:
        if len(v.schedules) > 1:
            xs = [c.loc[0] for c in v.schedules]
            ys = [c.loc[1] for c in v.schedules]
            plt.plot(xs, ys, marker="o")
    plt.title(title)
    plt.savefig(filename)
    plt.close()


# =====================================================
# 실행
# =====================================================
instance_files = sorted(glob.glob("hw2/instance_*.prob"))
results = []

os.makedirs("results_fig", exist_ok=True)

for prob in instance_files:
    print(f"\nProcessing {prob}")

    with open(prob, "rb") as f:
        inst = pickle.load(f)

    candidates = []

    spt = deep_copy_solution(inst)
    sol_spt = dispatch_spt(spt)
    candidates.append(("SPT", sol_spt.objective, spt))

    edd = deep_copy_solution(inst)
    sol_edd = dispatch_edd(edd)
    candidates.append(("EDD", sol_edd.objective, edd))

    reg = deep_copy_solution(inst)
    sol_reg = dispatch_regret_k(reg, 3)
    candidates.append(("Regret-3", sol_reg.objective, reg))

    ils = deep_copy_solution(reg)
    sol_ils = ils_optimize(ils)
    candidates.append(("ILS", sol_ils.objective, ils))

    vns = deep_copy_solution(reg)
    sol_vns = vns_optimize(vns)
    candidates.append(("VNS", sol_vns.objective, vns))

    best_name, best_obj, best_inst = min(candidates, key=lambda x: x[1])

    plot_solution(
        best_inst,
        f"{os.path.basename(prob)} | {best_name} | {best_obj:.2f}",
        f"results_fig/{os.path.basename(prob)}_{best_name}.png"
    )

    results.append({
        "Problem": os.path.basename(prob),
        "Best": best_name,
        "Best_Tardiness": best_obj,
        "SPT": sol_spt.objective,
        "EDD": sol_edd.objective,
        "Regret-3": sol_reg.objective,
        "ILS": sol_ils.objective,
        "VNS": sol_vns.objective
    })

df = pd.DataFrame(results)
df.to_csv("results_final.csv", index=False)
print("\n모든 인스턴스 완료 ✅")
