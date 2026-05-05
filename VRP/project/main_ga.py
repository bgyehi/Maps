import pickle
import time
import pandas as pd
import glob
import sys
from typing import List

# 모듈 불러오기
try:
    from module import Instance, Customer, Vehicle, Solution
    from dispatching import dispatch_earliest_vehicle_best_customer  # 초기 해 생성용
    from ga_solver import run_genetic_algorithm  # GA 실행 함수
    try:
        from visualize import plot_solution, export_solution_to_excel
    except ImportError:
        plot_solution = None
        export_solution_to_excel = None
except ImportError as e:
    print("="*50)
    print(f"  [필수 오류] 파일 import에 실패했습니다: {e}")
    print("  VS Code 작업 폴더에 module.py, dispatching.py, ga_solver.py 파일이")
    print("  모두 올바르게 업로드되었는지 확인해주세요.")
    print("="*50)
    raise


# =========================================================
#                   메인 실행부
# =========================================================
if __name__ == "__main__":
    # ✅ 수정: instances 폴더 내부의 .prob 파일을 탐색
    instance_files = sorted(glob.glob("instances/*.prob"))
    if not instance_files:
        print("="*50)
        print("  [오류] 'instances/*.prob' 파일을 찾을 수 없습니다.")
        print("  project/instances 폴더에 .prob 파일이 30개 들어있는지 확인하세요.")
        print("="*50)
        sys.exit()

    results = []
    GA_TIME_PER_INSTANCE = 0.5  # GA 수행 시간 (초)

    for prob_file in instance_files:
        print(f"\n--- Processing: {prob_file} ---")

        # 1️⃣ 인스턴스 로드
        try:
            with open(prob_file, 'rb') as f:
                instance = pickle.load(f)
        except Exception as e:
            print(f"  [오류] {prob_file} 로딩 중 문제 발생: {e}")
            continue

        # 2️⃣ 초기 해 생성 (Dispatching)
        start_disp = time.time()
        init_solution = dispatch_earliest_vehicle_best_customer(instance)
        end_disp = time.time()
        dispatch_time = end_disp - start_disp
        init_obj = init_solution.objective

        print(f"  [Dispatch] 초기 해 생성 완료 (Tardiness: {init_obj:.3f}, Time: {dispatch_time:.3f}s)")

        # 3️⃣ GA 실행
        start_ga = time.time()
        best_sol, best_val = run_genetic_algorithm(
            instance,
            dispatch_func=dispatch_earliest_vehicle_best_customer,
            time_limit_sec=GA_TIME_PER_INSTANCE,
            population_size=20,
            mutation_rate=0.1,
            verbose=True
        )
        end_ga = time.time()

        total_time = end_ga - start_disp
        best_sol.objective = best_val
        best_sol.algorithm = "Greedy + GA"

        print(f"  [GA] 최종 Best: {best_val:.3f} (총 시간: {total_time:.3f}s)")

        results.append({
            "Problem": prob_file,
            "Initial_Tardiness": init_obj,
            "Final_Tardiness(GA)": best_val,
            "Dispatch_Time(s)": dispatch_time,
            "Total_Time(s)": total_time
        })

        # 4️⃣ 시각화
        if plot_solution:
            try:
                save_path = prob_file.replace('.prob', '_ga_solution.png')
                plot_solution(best_sol, annotate=True, show_time_windows=False, arrows=False, save_path=save_path)
                print(f"  [Plot] {save_path} 저장 완료")
            except Exception as e:
                print(f"  [Plot] 오류: {e}")

    # 5️⃣ 결과 요약
    print("\n\n" + "="*50)
    print("               GA Summary Results")
    print("="*50)

    if results:
        df = pd.DataFrame(results)
        df.set_index("Problem", inplace=True)
        print(df.to_markdown(floatfmt=".3f"))

        avg_init = df["Initial_Tardiness"].mean()
        avg_final = df["Final_Tardiness(GA)"].mean()
        print("\n평균 초기 Tardiness:", avg_init)
        print("평균 최종 Tardiness:", avg_final)
        if avg_init > 0:
            print(f"평균 개선율: {((avg_init - avg_final) / avg_init * 100):.2f}%")
    else:
        print("처리된 결과가 없습니다.")
