import pickle
import time
import pandas as pd
import glob
import os

from dispatching_2 import dispatch_earliest_vehicle_best_customer, gurobi_vrptw_optimize

USE_GUROBI = True
TIME_LIMIT_PER_INST = 1800  # 30분

instance_files = sorted(glob.glob("hw2/instance_*.prob"))
results = []

for prob_file in instance_files:
    print(f"\n{'=' * 60}")
    print(f"Processing: {prob_file}")
    print('=' * 60)

    with open(prob_file, "rb") as f:
        instance = pickle.load(f)

    # 1. 초기해 생성
    start_dispatch = time.time()
    sol_dispatch = dispatch_earliest_vehicle_best_customer(instance)
    time_dispatch = time.time() - start_dispatch
    obj_dispatch = sol_dispatch.objective

    print(f"[Dispatch] Tardiness: {obj_dispatch:.3f}, Time: {time_dispatch:.4f}s")

    # 2. Gurobi 최적화
    if USE_GUROBI:
        start_opt = time.time()
        sol_opt = gurobi_vrptw_optimize(instance, time_limit_sec=TIME_LIMIT_PER_INST)
        time_opt = time.time() - start_opt
        obj_opt = sol_opt.objective

        print(f"[Gurobi] Tardiness: {obj_opt:.3f}, Time: {time_opt:.2f}s, Status: {sol_opt.status}")

        results.append({
            "Problem": os.path.basename(prob_file),
            "Initial_Tardiness": obj_dispatch,
            "Final_Tardiness": obj_opt,
            "Dispatch_Time(s)": time_dispatch,
            "Total_Time(s)": time_dispatch + time_opt,
            "Status": sol_opt.status
        })
    else:
        results.append({
            "Problem": os.path.basename(prob_file),
            "Initial_Tardiness": obj_dispatch,
            "Final_Tardiness": obj_dispatch,
            "Dispatch_Time(s)": time_dispatch,
            "Total_Time(s)": time_dispatch,
            "Status": sol_dispatch.status
        })

# 결과 요약
df = pd.DataFrame(results)
df.set_index("Problem", inplace=True)

print("\n" + "=" * 100)
print("최종 요약 결과 (Summary Results)")
print("=" * 100)
print(df.to_markdown(floatfmt=".3f"))

avg_initial = df["Initial_Tardiness"].mean()
avg_final = df["Final_Tardiness"].mean()
avg_time = df["Total_Time(s)"].mean()
total_initial = df["Initial_Tardiness"].sum()
total_final = df["Final_Tardiness"].sum()
improv = ((avg_initial - avg_final) / avg_initial * 100) if avg_initial > 1e-9 else 0

print("\n" + "-" * 60)
print("전체 평균 (Average Performance)")
print(f"- 평균 초기 Tardiness : {avg_initial:.3f}")
print(f"- 평균 최종 Tardiness : {avg_final:.3f}")
print(f"- 개선율 (Improvement) : {improv:.2f} %")
print(f"- 문제당 평균 총 시간 : {avg_time:.3f} 초")
print("-" * 60)
print("전체 총합 (Total Performance)")
print(f"- 총 초기 Tardiness : {total_initial:.3f}")
print(f"- 총 최종 Tardiness : {total_final:.3f}")
print("=" * 100)

df.to_csv("results_summary.csv")
print("\nSaved results_summary.csv ✅")

print("\n[구로비] --- [평균 성능] ---")
print(f"평균 초기 Tardiness : {avg_initial:.3f}")
print(f"평균 최종 Tardiness : {avg_final:.3f}")
print(f"개선율 : {improv:.2f} %")
print(f"평균 총 실행시간 : {avg_time:.3f} 초")
print("=" * 100)
