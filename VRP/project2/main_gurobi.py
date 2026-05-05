import pickle
import time
import pandas as pd
import glob
import os
from dispatching_gurobi import (
    dispatch_earliest_vehicle_best_customer,
    gurobi_vrptw_simple,
    improve_with_ils
)

# ========================
# 설정
# ========================
USE_GUROBI = True
GUROBI_TIME_LIMIT = 600  # 10분
HEURISTIC_TIME_LIMIT = 2.0

instance_files = sorted(glob.glob("hw2/instance_*.prob"))

if not instance_files:
    print("=" * 50)
    print("  [오류] 'hw2/instance_*.prob' 파일을 찾을 수 없습니다.")
    print("=" * 50)
    exit(1)

results = []

# ========================
# 문제별 처리
# ========================
for prob_file in instance_files:
    print(f"\n{'=' * 70}")
    print(f"Processing: {prob_file}")
    print('=' * 70)

    with open(prob_file, "rb") as f:
        instance = pickle.load(f)

    n_customers = len(instance.customers)
    n_vehicles = len(instance.vehicles)

    print(f"고객: {n_customers}개, 차량: {n_vehicles}대")

    # 1. 초기해 (항상 생성)
    start_dispatch = time.time()
    sol_init = dispatch_earliest_vehicle_best_customer(instance)
    time_dispatch = time.time() - start_dispatch
    obj_init = sol_init.objective

    print(f"[초기해] Tardiness: {obj_init:.3f}, Time: {time_dispatch:.4f}s")

    # 2. 최적화 전략 선택
    start_opt = time.time()

    if USE_GUROBI and n_customers <= 20:
        # 작은 문제: Gurobi 사용
        print(f"[전략] Gurobi 최적화 (고객 {n_customers}개)")
        try:
            sol_final = gurobi_vrptw_simple(instance, time_limit_sec=GUROBI_TIME_LIMIT)
            obj_final = sol_final.objective
        except Exception as e:
            print(f"[Gurobi 오류] {e}")
            print("[Fallback] 휴리스틱 개선으로 전환")
            improve_with_ils(sol_init, HEURISTIC_TIME_LIMIT)
            sol_final = sol_init
            obj_final = sol_init.objective
    else:
        # 큰 문제: 휴리스틱만
        print(f"[전략] 휴리스틱 개선 (고객 {n_customers}개)")
        improve_with_ils(sol_init, HEURISTIC_TIME_LIMIT)
        sol_final = sol_init
        obj_final = sol_init.objective

    time_opt = time.time() - start_opt
    total_time = time.time() - start_dispatch

    print(f"[최종해] Tardiness: {obj_final:.3f}, Total Time: {total_time:.2f}s")

    # 결과 기록
    results.append({
        "Problem": os.path.basename(prob_file),
        "Initial_Tardiness": obj_init,
        "Final_Tardiness": obj_final,
        "Dispatch_Time(s)": time_dispatch,
        "Total_Time(s)": total_time,
        "Status": sol_final.status
    })

# ========================
# 결과 요약
# ========================
df = pd.DataFrame(results)
df.set_index("Problem", inplace=True)

print("\n" + "=" * 100)
print("최종 요약 결과 (Summary Results)")
print("=" * 100)
try:
    print(df.to_markdown(floatfmt=".3f"))
except:
    print(df)

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
if avg_initial > 1e-9:
    print(f"- 개선율 (Improvement) : {improv:.2f} %")
print(f"- 문제당 평균 총 시간 : {avg_time:.3f} 초")
print("-" * 60)
print("전체 총합 (Total Performance)")
print(f"- 총 초기 Tardiness : {total_initial:.3f}")
print(f"- 총 최종 Tardiness : {total_final:.3f}")
print("=" * 100)

df.to_csv("results_summary.csv")
print("\nSaved results_summary.csv ✅")

print("\n[구로비 + 휴리스틱 하이브리드] --- [평균 성능] ---")
print(f"평균 초기 Tardiness : {avg_initial:.3f}")
print(f"평균 최종 Tardiness : {avg_final:.3f}")
print(f"개선율 : {improv:.2f} %")
print(f"평균 총 실행시간 : {avg_time:.3f} 초")
print("--- [총합 성능] ---")
print(f"총 초기 Tardiness : {total_initial:.3f}")
print(f"총 최종 Tardiness : {total_final:.3f}")
print("=" * 100)
