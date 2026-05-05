import pickle
import time
import pandas as pd
import glob
import random
import math
import sys
from typing import TYPE_CHECKING, Tuple, Optional, List, Dict

# 1. 업로드한 파일에서 클래스와 함수들을 불러옵니다.
try:
    from module import Instance, Customer, Vehicle, Solution, make_random_instance, get_dist

    # 개선된 dispatching 모듈에서 초기해, VNS, ILS, 교란 함수들을 가져옵니다.
    # (이전과 같은 함수명으로 호환되도록 dispatching_improved.py는 wrapper들을 제공합니다.)
    from dispatching_1 import (
        dispatch_earliest_vehicle_best_customer,
        run_vns_improvement,
        perturb_solution,
        improve_with_ils,
    )

    try:
        from visualize import plot_solution, export_solution_to_excel, load_solution_from_excel
    except ImportError:
        print("참고: visualize.py 파일을 찾을 수 없습니다. 시각화 및 엑셀 내보내기를 건너뜁니다.")
        plot_solution = None
        export_solution_to_excel = None
        load_solution_from_excel = None

except ImportError as e:
    print("="*50)
    print(f"  [필수 오류] 파일 import에 실패했습니다: {e}")
    print("  VS Code 작업 폴더에 module.py, dispatching.py 파일이")
    print("  모두 올바르게 업로드되었는지 확인해주세요.")
    print("="*50)
    raise


# ---------------------------------------------------------------------
#               메인 실행 로직 (수정됨)
# ---------------------------------------------------------------------

if __name__ == "__main__":

    instance_files = sorted(glob.glob("hw2/instance_*.prob"))

    if not instance_files:
        print("="*50)
        print("  [오류] 'instance_*.prob' 파일을 찾을 수 없습니다.")
        print("  VS Code 작업 폴더에 30개의 .prob 파일을 업로드했는지 확인하세요.")
        print("="*50)

    results = []

    # ILS 총 시간 (초). 필요하면 커맨드라인 또는 직접 수정하세요.
    ILS_TIME_PER_INSTANCE = 1.0  # 기본 1.0초로 늘려서 더 깊은 탐색 허용

    SAVE_EXCEL = False
    SAVE_PLOT = True

    for prob_file in instance_files:
        print(f"\n--- Processing: {prob_file} ---")

        try:
            with open(prob_file, 'rb') as f:
                instance = pickle.load(f)
        except Exception as e:
            print(f"  오류: {prob_file} 로딩 중 문제 발생: {e}")
            continue

        # --- 1. 초기 해 생성 (Regret-k 또는 모듈에서 설정한 방식) ---
        start_time_dispatch = time.time()

        solution_disp = dispatch_earliest_vehicle_best_customer(instance)

        end_time_dispatch = time.time()
        time_dispatch = end_time_dispatch - start_time_dispatch
        obj_dispatch = solution_disp.objective

        print(f"  [Dispatch] 초기 해 생성 완료 (Tardiness: {obj_dispatch:.3f}, Time: {time_dispatch:.4f}s)")

        # --- 2. 해 개선 (ILS 호출) ---
        start_time_ils = time.time()

        if solution_disp.status == "INFEASIBLE_UNSERVED":
            print(f"  [ILS] Skipped. Dispatching failed to serve all customers.")
            obj_ils_final = obj_dispatch
            time_total = time_dispatch
        else:
            obj_ils_final = improve_with_ils(solution_disp, time_limit_sec=ILS_TIME_PER_INSTANCE)
            end_time_ils = time.time()
            time_total = end_time_ils - start_time_dispatch

        solution_disp.objective = obj_ils_final
        solution_disp.algorithm = f"RegretK + ILS(VNS+TS+2opt)"

        # --- 상태 업데이트 및 확인 ---
        total_customers_in_solution = sum(len(v.schedules) for v in instance.vehicles)
        if total_customers_in_solution < len(instance.customers):
            solution_disp.status = "BUG_UNSERVED"
        elif obj_ils_final == float('inf'):
            solution_disp.status = "BUG_INFEASIBLE_CAP"
        elif solution_disp.status == "INFEASIBLE_UNSERVED":
            pass
        else:
            solution_disp.status = "DONE"

        # 결과 저장
        results.append({
            "Problem": prob_file,
            "Initial_Tardiness": obj_dispatch,
            "Final_Tardiness(ILS)": obj_ils_final,
            "Dispatch_Time(s)": time_dispatch,
            "Total_Time(s)": time_total,
            "Status": solution_disp.status
        })

        # 엑셀/플롯 저장 (옵션)
        if SAVE_EXCEL and export_solution_to_excel:
            try:
                save_path_excel = prob_file.replace('.prob', '_solution.xlsx')
                export_solution_to_excel(solution=solution_disp, filepath=save_path_excel)
                print(f"  [Excel] {save_path_excel} (으)로 상세 경로 저장됨.")
            except Exception as e:
                print(f"  [Excel] 엑셀 저장 중 오류 발생: {e}")

        if SAVE_PLOT and plot_solution:
            try:
                save_path_img = prob_file.replace('.prob', '_solution.png')
                metrics = plot_solution(
                    solution_disp,
                    annotate=True,
                    show_time_windows=True,
                    arrows=False,
                    save_path=save_path_img,
                    write_back=True
                )
                print(f"  [Plot] {save_path_img} (으)로 최종 경로 저장됨.")
            except Exception as e:
                print(f"  [Plot] 시각화 중 오류 발생: {e}")

    # --- 3. 최종 요약 결과 출력 ---
    print("\n\n" + "="*50)
    print("           최종 요약 결과 (Summary Results)")
    print("="*50)

    if results:
        df_results = pd.DataFrame(results)
        df_results.set_index("Problem", inplace=True)

        avg_initial_tard = df_results["Initial_Tardiness"].mean()
        avg_final_tard = df_results["Final_Tardiness(ILS)"].mean()
        avg_total_time = df_results["Total_Time(s)"].mean()
        total_initial_tard = df_results["Initial_Tardiness"].sum()
        total_final_tard = df_results["Final_Tardiness(ILS)"].sum()

        try:
            print(df_results.to_markdown(floatfmt=".3f"))
        except ImportError:
            print(df_results)

        print("\n" + "-"*50)
        print("          전체 평균 (Average Performance)")
        print(f"  - 평균 초기 Tardiness: {avg_initial_tard:.3f}")
        print(f"  - 평균 최종 Tardiness: {avg_final_tard:.3f}")

        if avg_initial_tard > 1e-9:
            print(f"  - 개선율 (Improvement): {((avg_initial_tard - avg_final_tard) / avg_initial_tard * 100):.2f} %")

        print(f"  - 문제당 평균 총 시간: {avg_total_time:.4f} 초 (Dispatch + ILS {ILS_TIME_PER_INSTANCE}s)")
        print("\n" + "-"*50)
        print("          전체 총합 (Total Performance)")
        print(f"  - 총 초기 Tardiness: {total_initial_tard:.3f}")
        print(f"  - 총 최종 Tardiness: {total_final_tard:.3f}")
        print("="*50)
    else:
        print("처리된 결과가 없습니다. 'instance_*.prob' 파일들을 VS Code 작업 폴더에 업로드했는지 확인하세요.")