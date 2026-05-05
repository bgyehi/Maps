import pickle
import time
import pandas as pd
import glob
import os
import copy
import matplotlib.pyplot as plt

# 👉 최종 디스패칭 / 메타휴리스틱 코드
from dispatching_final import (
    dispatch_spt,
    dispatch_edd,
    ils_optimize,
    vns_optimize,
    extreme_optimization,
    calculate_total_tardiness
)

# =====================================================
# deepcopy
# =====================================================
def deep_copy_solution(inst):
    return copy.deepcopy(inst)


# =====================================================
# 시각화 함수 (Vehicle schedules 기반)
# =====================================================
def plot_solution(instance, title="", filename=None):
    plt.figure(figsize=(7, 7))

    # depot
    depot = instance.vehicles[0].loc
    plt.scatter(depot[0], depot[1], c='red', s=120, marker='s', label="Depot")

    # customers
    xs = [c.loc[0] for c in instance.customers]
    ys = [c.loc[1] for c in instance.customers]
    plt.scatter(xs, ys, c='black', s=30, label="Customers")

    colors = plt.cm.tab10.colors

    for i, v in enumerate(instance.vehicles):
        if len(v.schedules) == 0:
            continue
        route = [depot] + [c.loc for c in v.schedules]
        rx = [p[0] for p in route]
        ry = [p[1] for p in route]
        plt.plot(rx, ry, color=colors[i % len(colors)],
                 linewidth=1.5, label=f"Vehicle {i}")

    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(fontsize=8)
    plt.grid(True)

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# =====================================================
# 실험 설정
# =====================================================
METHODS = {
    "SPT": dispatch_spt,
    "EDD": dispatch_edd,
    "EDD+ILS": lambda inst: ils_optimize(inst, time_limit=1.5),
    "EDD+VNS": lambda inst: vns_optimize(inst, time_limit=1.5),
    "ExtremeOpt": lambda inst: extreme_optimization(inst, max_iter=400)
}

instance_files = sorted(glob.glob("hw2/instance_*.prob"))
os.makedirs("results_fig", exist_ok=True)

results = []
total_time = 0.0

# =====================================================
# 메인 루프
# =====================================================
for prob_file in instance_files:
    print("\n" + "=" * 70)
    print(f"Processing: {prob_file}")
    print("=" * 70)

    with open(prob_file, "rb") as f:
        instance_orig = pickle.load(f)

    candidates = []

    # -----------------------------
    # 알고리즘 실행
    # -----------------------------
    for name, method in METHODS.items():
        inst_copy = deep_copy_solution(instance_orig)

        start = time.time()
        sol = method(inst_copy)
        elapsed = time.time() - start

        total_time += elapsed

        candidates.append({
            "method": name,
            "tardiness": sol.objective,
            "time": elapsed,
            "instance": inst_copy
        })

        print(f"  {name:<12} | tardiness = {sol.objective:8.2f} | time = {elapsed:5.2f}s")

    # -----------------------------
    # 최선 선택
    # -----------------------------
    best = min(candidates, key=lambda x: x["tardiness"])

    print("-" * 60)
    print(f"[BEST] {best['method']} | tardiness = {best['tardiness']:.2f}")

    # -----------------------------
    # 시각화 저장
    # -----------------------------
    fig_name = f"results_fig/{os.path.basename(prob_file).replace('.prob','')}_{best['method']}.png"
    plot_solution(
        best["instance"],
        title=f"{os.path.basename(prob_file)} | {best['method']} | tardiness={best['tardiness']:.2f}",
        filename=fig_name
    )
    print(f"  ➜ Saved figure: {fig_name}")

    # -----------------------------
    # 결과 기록
    # -----------------------------
    row = {
        "Problem": os.path.basename(prob_file),
        "Best_Method": best["method"],
        "Best_Tardiness": best["tardiness"]
    }

    for c in candidates:
        row[c["method"]] = c["tardiness"]

    results.append(row)


# =====================================================
# 전체 결과 요약
# =====================================================
df = pd.DataFrame(results)
df.set_index("Problem", inplace=True)

print("\n" + "=" * 100)
print("📊 Summary Results")
print("=" * 100)
print(df.to_markdown(floatfmt=".2f"))

avg_tard = df["Best_Tardiness"].mean()
total_tard = df["Best_Tardiness"].sum()
avg_time = total_time / (len(instance_files) * len(METHODS))

print("\n" + "-" * 60)
print("📈 Overall Performance")
print(f"- Total Best Tardiness   : {total_tard:.2f}")
print(f"- Average Best Tardiness : {avg_tard:.2f}")
print(f"- Total Time (sec)       : {total_time:.2f}")
print(f"- Average Time (sec)     : {avg_time:.2f}")
print("=" * 100)

df.to_csv("results_final_methods.csv")
print("\nSaved results_final_methods.csv ✅")
