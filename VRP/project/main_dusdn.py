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

    # (수정) dispatching에서 '초기 해 생성', 'VNS', '교란' 함수를 모두 import
    from dispatching_dusdn import dispatch_earliest_vehicle_best_customer, run_vns_improvement, perturb_solution

    try:
        from visualize import plot_solution, export_solution_to_excel, load_solution_from_excel
    except ImportError:
        print("참고: visualize.py 파일을 찾을 수 없습니다. 시각화 및 엑셀 내보내기를 건너뜁니다.")
        plot_solution = None
        export_solution_to_excel = None
        load_solution_from_excel = None

except ImportError as e:
    print("=" * 50)
    print(f"  [필수 오류] 파일 import에 실패했습니다: {e}")
    print("  VS Code 작업 폴더에 module.py, dispatching.py 파일이")
    print("  모두 올바르게 업로드되었는지 확인해주세요.")
    print("=" * 50)
    raise


# === (신규) ILS (하이브리드 메타휴리스틱) 메인 함수 ===
def improve_with_ils(solution: 'Solution', time_limit_sec: float) -> float:
    """
    (수정됨) ILS가 '안전한 교란(Random Move)'을 사용하고,
    VNS(Relocate+Swap+2opt) 탐색 시 Tabu List를 사용합니다.
    """
    instance: 'Instance' = solution.instance
    vehicles: List['Vehicle'] = instance.vehicles
    start_ils_time = time.time()

    tabu_list: Dict[int, int] = {}

    print(f"  [ILS] Initial Local Search (VNS+TS+2opt) 시작...")

    # (수정) VNS 함수는 이제 dispatching.py에 있으므로,
    #       VNS가 ILS의 Tabu List, Iteration, Best Tardiness를 참조할 수 없음.
    #       (단순화) VNS를 0.1초 실행하여 '최초의 지역 최적해'를 찾음
    VNS_INIT_TIME = 0.1  # 0.3초 중 0.1초는 초기 해 최적화에 사용

    best_tardiness = run_vns_improvement(solution, VNS_INIT_TIME)

    print(f"  [ILS] Initial Local Optimum found. Tardiness: {best_tardiness:.3f}")

    best_schedules: Dict[int, List['Customer']] = {v.ID: list(v.schedules) for v in vehicles}

    iteration = 0
    # (수정) 남은 시간 동안 ILS 반복
    remaining_time = time_limit_sec - VNS_INIT_TIME

    while True:
        elapsed = time.time() - start_ils_time
        if elapsed > time_limit_sec:
            break

        iteration += 1

        print(
            f"    [ILS Iter {iteration}] Time: {elapsed:.2f}s / {time_limit_sec:.2f}s. Current Best: {best_tardiness:.3f}",
            end='\r')
        sys.stdout.flush()

        # 4. Perturbation (안전한 랜덤 이동)
        perturb_solution(solution, strength=3)

        # 5. Local Search (VNS)
        # (수정) ILS 루프 내에서는 VNS를 짧게(0.05초) 실행
        current_tardiness = run_vns_improvement(solution, 0.05)

        # 6. (Acceptance)
        if current_tardiness < best_tardiness:
            best_tardiness = current_tardiness
            best_schedules = {v.ID: list(v.schedules) for v in vehicles}
            print(f"\n    [ILS Iter {iteration}] New Best Found! (Random Move) Tardiness: {best_tardiness:.3f}")

        # 7. '최고 해'의 상태로 되돌림
        for v in vehicles:
            v.schedules = list(best_schedules[v.ID])

    print()
    print(f"  [ILS] Finished. Total Iterations: {iteration}. Final Best Tardiness: {best_tardiness:.3f}")

    solution.objective = best_tardiness
    return best_tardiness


# ---------------------------------------------------------------------
#               (수정된) 메인 실행 로직
# ---------------------------------------------------------------------

if __name__ == "__main__":

    instance_files = sorted(glob.glob("instances/*.prob"))

    if not instance_files:
        print("=" * 50)
        print("  [오류] 'instances/*.prob' 파일을 찾을 수 없습니다.")
        print("  VS Code 작업 폴더에 30개의 .prob 파일을 업로드했는지 확인하세요.")
        print("=" * 50)

    results = []

    # (수정) ILS 총 시간 0.3초
    ILS_TIME_PER_INSTANCE = 0.3

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

        # --- 1. 초기 해 생성 (Greedy Dispatching) ---
        start_time_dispatch = time.time()

        solution_disp = dispatch_earliest_vehicle_best_customer(
            instance
        )

        end_time_dispatch = time.time()
        time_dispatch = end_time_dispatch - start_time_dispatch
        obj_dispatch = solution_disp.objective  # 초기 Tardiness

        print(f"  [Dispatch] Greedy 초기 해 생성 완료 (Tardiness: {obj_dispatch:.3f}, Time: {time_dispatch:.4f}s)")

        # --- 2. 해 개선 (Iterated Local Search - ILS) ---
        start_time_ils = time.time()

        if solution_disp.status == "INFEASIBLE_UNSERVED":
            print(f"  [ILS] Skipped. Dispatching failed to serve all customers.")
            obj_ils_final = obj_dispatch
            time_total = time_dispatch
        else:
            # (수정) VNS가 아닌 ILS 함수 호출
            obj_ils_final = improve_with_ils(solution_disp, time_limit_sec=ILS_TIME_PER_INSTANCE)
            end_time_ils = time.time()
            time_total = end_time_ils - start_time_dispatch  # dispatch + ILS 총 시간

        solution_disp.objective = obj_ils_final
        solution_disp.algorithm = f"Greedy + ILS(VNS+TS+2opt)"

        # (수정) 최종 Status 업데이트 (Unserved 고객 확인)
        total_customers_in_solution = sum(len(v.schedules) for v in instance.vehicles)
        if total_customers_in_solution < len(instance.customers):
            solution_disp.status = "BUG_UNSERVED"  # (버그)
        elif obj_ils_final == float('inf'):
            solution_disp.status = "BUG_INFEASIBLE_CAP"  # (버그)
        elif solution_disp.status == "INFEASIBLE_UNSERVED":
            pass
        else:
            solution_disp.status = "DONE"  # 모든 고객이 배송 완료됨

        # 결과 저장
        results.append({
            "Problem": prob_file,
            "Initial_Tardiness": obj_dispatch,
            "Final_Tardiness(ILS)": obj_ils_final,  # (수정)
            "Dispatch_Time(s)": time_dispatch,
            "Total_Time(s)": time_total,
            "Status": solution_disp.status
        })

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

    # --- 3. 최종 요약 결과 출력 (제출물) ---
    print("\n\n" + "=" * 50)
    print("           최종 요약 결과 (Summary Results)")
    print("=" * 50)

    if results:
        df_results = pd.DataFrame(results)
        df_results.set_index("Problem", inplace=True)

        # 총합/평균 계산
        avg_initial_tard = df_results["Initial_Tardiness"].mean()
        avg_final_tard = df_results["Final_Tardiness(ILS)"].mean()
        avg_total_time = df_results["Total_Time(s)"].mean()
        total_initial_tard = df_results["Initial_Tardiness"].sum()
        total_final_tard = df_results["Final_Tardiness(ILS)"].sum()

        # Markdown 형식으로 깔끔하게 출력
        try:
            print(df_results.to_markdown(floatfmt=".3f"))
        except ImportError:
            print(df_results)

        print("\n" + "-" * 50)
        print("          전체 평균 (Average Performance)")
        print(f"  - 평균 초기 Tardiness: {avg_initial_tard:.3f} (Greedy 결과)")
        print(f"  - 평균 최종 Tardiness: {avg_final_tard:.3f} (ILS 적용 후)")

        if avg_initial_tard > 1e-9:
            print(f"  - 개선율 (Improvement): {((avg_initial_tard - avg_final_tard) / avg_initial_tard * 100):.2f} %")

        print(f"  - 문제당 평균 총 시간: {avg_total_time:.4f} 초 (Dispatch + ILS {ILS_TIME_PER_INSTANCE}s)")
        print("\n" + "-" * 50)
        print("          전체 총합 (Total Performance)")
        print(f"  - 총 초기 Tardiness: {total_initial_tard:.3f}")
        print(f"  - 총 최종 Tardiness: {total_final_tard:.3f}")
        print("=" * 50)
    else:
        print("처리된 결과가 없습니다. 'instance_*.prob' 파일들을 VS Code 작업 폴더에 업로드했는지 확인하세요.")