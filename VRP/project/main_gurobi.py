import pickle
import time
import pandas as pd
import glob
import sys
from typing import List, Dict

# ===========================================
#   (1) 외부 모듈 import
# ===========================================
try:
    from module import Instance, Customer, Vehicle, Solution
    from dispatching_gurobi import (
        dispatch_earliest_vehicle_best_customer,
        run_vns_improvement,
        perturb_solution
    )
    try:
        from visualize import plot_solution, export_solution_to_excel
    except ImportError:
        print("[참고] visualize.py 없음 → 시각화 및 엑셀 저장 생략.")
        plot_solution = None
        export_solution_to_excel = None
except ImportError as e:
    print("=" * 60)
    print(f"[오류] 필수 파일 import 실패: {e}")
    print("module.py, dispatching_gurobi.py가 프로젝트 폴더에 있는지 확인하세요.")
    print("=" * 60)
    sys.exit(1)


# ===========================================
#   (2) 개선 알고리즘 (ILS)
# ===========================================
def improve_with_ils(solution: 'Solution', time_limit_sec: float) -> float:
    """
    Iterated Local Search (VNS + 안전한 교란)
    """
    instance: 'Instance' = solution.instance
    vehicles: List['Vehicle'] = instance.vehicles
    start_time = time.time()

    best_tardiness = run_vns_improvement(solution, 0.1)
    best_routes = {v.ID: list(v.schedules) for v in vehicles}

    iter_count = 0
    while time.time() - start_time < time_limit_sec:
        iter_count += 1

        perturb_solution(solution, strength=3)
        current_tardiness = run_vns_improvement(solution, 0.05)

        if current_tardiness < best_tardiness:
            best_tardiness = current_tardiness
            best_routes = {v.ID: list(v.schedules) for v in vehicles}
            print(f"    [ILS] New Best Found: {best_tardiness:.3f}")

        # 복원
        for v in vehicles:
            v.schedules = list(best_routes[v.ID])

    solution.objective = best_tardiness
    return best_tardiness


# ===========================================
#   (3) 메인 실행 루프
# ===========================================
if __name__ == "__main__":
    instance_files = sorted(glob.glob("hw2/*.prob"))
    if not instance_files:
        print("[오류] 'hw2/*.prob' 파일을 찾을 수 없습니다.")
        sys.exit(1)

    results = []
    ILS_TIME = 0.3  # 인스턴스당 ILS 실행 시간
    SAVE_EXCEL = False
    SAVE_PLOT = True

    for idx, file_path in enumerate(instance_files, start=1):
        print(f"\n=== [Instance {idx:02d}] {file_path} ===")

        # -------------------------------
        # 1. 인스턴스 로드
        # -------------------------------
        try:
            with open(file_path, "rb") as f:
                instance = pickle.load(f)
        except Exception as e:
            print(f"[Error] {file_path} 로딩 실패: {e}")
            continue

        # -------------------------------
        # 2. 초기 해 (Greedy Dispatch)
        # -------------------------------
        start_dispatch = time.time()
        solution = dispatch_earliest_vehicle_best_customer(instance)
        dispatch_time = time.time() - start_dispatch
        init_tard = getattr(solution, "objective", float('inf'))

        print(f"  [Dispatch] 완료 - 초기 Tardiness: {init_tard:.3f}, Time: {dispatch_time:.3f}s")

        # -------------------------------
        # 3. ILS 개선
        # -------------------------------
        if solution.status == "INFEASIBLE_UNSERVED":
            print("  [ILS] Skipped (Unserved 고객 존재)")
            final_tard = init_tard
            total_time = dispatch_time
        else:
            start_ils = time.time()
            final_tard = improve_with_ils(solution, time_limit_sec=ILS_TIME)
            total_time = time.time() - start_dispatch
            print(f"  [ILS] 최종 Tardiness: {final_tard:.3f}, 총 시간: {total_time:.3f}s")

        # -------------------------------
        # 4. 결과 상태 저장
        # -------------------------------
        total_customers = sum(len(v.schedules) for v in instance.vehicles)
        if total_customers < len(instance.customers):
            solution.status = "UNSERVED"
        else:
            solution.status = "DONE"

        results.append({
            "Instance": idx,
            "Initial_Tardiness": init_tard,
            "Final_Tardiness": final_tard,
            "Dispatch_Time(s)": dispatch_time,
            "Total_Time(s)": total_time,
            "Status": solution.status
        })

        # -------------------------------
        # 5. 저장 (엑셀, 그래프)
        # -------------------------------
        if SAVE_EXCEL and export_solution_to_excel:
            export_solution_to_excel(solution, file_path.replace(".prob", "_solution.xlsx"))

        if SAVE_PLOT and plot_solution:
            try:
                img_path = file_path.replace(".prob", "_solution.png")
                plot_solution(
                    solution,
                    annotate=True,
                    show_time_windows=True,
                    arrows=False,
                    save_path=img_path
                )
                print(f"  [Plot] {img_path} 저장 완료")
            except Exception as e:
                print(f"  [Plot Error] {e}")

    # ===========================================
    #   (4) 최종 요약 결과 출력 (제출용)
    # ===========================================
    print("\n\n" + "=" * 60)
    print("📊 최종 요약 결과 (Summary of 30 Instances)")
    print("=" * 60)

    if results:
        df = pd.DataFrame(results)
        df.set_index("Instance", inplace=True)

        # 평균 및 총합 계산
        avg_tard_init = df["Initial_Tardiness"].mean()
        avg_tard_final = df["Final_Tardiness"].mean()
        avg_time_total = df["Total_Time(s)"].mean()

        total_tard_init = df["Initial_Tardiness"].sum()
        total_tard_final = df["Final_Tardiness"].sum()

        # 표 출력
        try:
            print(df.to_markdown(floatfmt=".3f"))
        except ImportError:
            print(df)

        print("\n--- [평균 성능] ---")
        print(f"  평균 초기 Tardiness : {avg_tard_init:.3f}")
        print(f"  평균 최종 Tardiness : {avg_tard_final:.3f}")
        if avg_tard_init > 1e-9:
            print(f"  개선율              : {(avg_tard_init - avg_tard_final) / avg_tard_init * 100:.2f} %")
        print(f"  평균 총 실행시간     : {avg_time_total:.3f} 초")

        print("\n--- [총합 성능] ---")
        print(f"  총 초기 Tardiness   : {total_tard_init:.3f}")
        print(f"  총 최종 Tardiness   : {total_tard_final:.3f}")

        print("=" * 60)
        print("✅ 평균 Tardiness와 평균 시간은 낮을수록 우수한 성능입니다.")
    else:
        print("결과가 없습니다. 인스턴스 로드를 확인하세요.")
