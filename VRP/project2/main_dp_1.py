# main_dp1.py
import pickle
import time
import glob
import sys
import pandas as pd
from dp_1 import solve_vrp_gurobi
from module import Instance

# ===========================================
# 인스턴스 파일 탐색
# ===========================================
instance_files = sorted(glob.glob("hw2/*.prob"))
if not instance_files:
    print("[오류] 'instances/*.prob' 파일이 없습니다.")
    sys.exit(1)

results = []

# ===========================================
# 인스턴스별 실행
# ===========================================
for idx, file_path in enumerate(instance_files, start=1):
    print(f"\n=== [Instance {idx:02d}] {file_path} ===")

    try:
        with open(file_path, "rb") as f:
            instance: Instance = pickle.load(f)
    except Exception as e:
        print(f"[Error] {file_path} 로딩 실패: {e}")
        continue

    start_time = time.time()
    try:
        solution = solve_vrp_gurobi(instance, time_limit=60.0, use_warmstart=False, verbose=True)
    except Exception as e:
        print(f"[Error] {file_path} 실행 실패: {e}")
        continue
    elapsed_time = time.time() - start_time

    obj_val = getattr(solution, "objective", getattr(solution, "obj", None))
    print(f"  Objective (총 Tardiness): {obj_val}")
    print(f"  실행 시간: {elapsed_time:.3f}s")

    results.append({
        "Instance": idx,
        "Objective": obj_val,
        "Elapsed_Time(s)": elapsed_time
    })

# ===========================================
# 결과 요약
# ===========================================
if results:
    df = pd.DataFrame(results).set_index("Instance")
    print("\n=== 요약 결과 ===")
    print(df.to_markdown(floatfmt=".3f"))
    print(f"\n평균 Objective: {df['Objective'].mean():.3f}")
    print(f"평균 실행 시간: {df['Elapsed_Time(s)'].mean():.3f}초")
