# main_gurobi.py
import pickle
import time
import pandas as pd
import glob
import sys
from typing import List, Dict

# ===========================================
#   (1) 외부 모듈 import (유연하게 처리)
# ===========================================
try:
    from module import Instance, Customer, Vehicle, Solution
except ImportError as e:
    print("=" * 60)
    print(f"[오류] 필수 파일 import 실패: {e}")
    print("module.py가 프로젝트 폴더에 있는지 확인하세요.")
    print("=" * 60)
    sys.exit(1)

# try to import Gurobi solver (dispatching_gurobi.py)
have_gurobi_solver = False
have_local_improve = False
try:
    from dispatching_gurobi import solve_vrp_gurobi
    have_gurobi_solver = True
    # optional: try to import improvement primitives if present in dispatching_gurobi
    try:
        from dispatching_gurobi import run_vns_improvement, perturb_solution
        have_local_improve = True
    except Exception:
        have_local_improve = False
except Exception:
    # fallback: try dispatching_gb_1 heuristic-only
    try:
        from dispatching_gb_1 import dispatch_earliest_vehicle_best_customer, run_vns_improvement, perturb_solution
        have_gurobi_solver = False
        have_local_improve = True
    except Exception as e:
        print("=" * 60)
        print(f"[오류] dispatching_gurobi.py 또는 dispatching_gb_1.py에서 필요한 함수들을 찾을 수 없습니다: {e}")
        print("dispatching_gurobi.py 또는 dispatching_gb_1.py가 프로젝트 폴더에 있는지 확인하세요.")
        print("=" * 60)
        sys.exit(1)

# optional visualization helpers
try:
    from visualize import plot_solution, export_solution_to_excel
except Exception:
    print("[참고] visualize.py 없음 → 시각화 및 엑셀 저장 생략.")
    plot_solution = None
    export_solution_to_excel = None

# ===========================================
#   (2) 개선 알고리즘 (ILS style) — optional
# ===========================================
def improve_with_ils(solution: 'Solution', time_limit_sec: float) -> float:
    """
    Iterated Local Search wrapper. 동작 가능한 개선 함수가 있으면 호출.
    - 만약 dispatching_dusdn.run_vns_improvement/perturb_solution이 있으면 사용.
    - 없으면 바로 현재 objective 반환.
    """
    try:
        from dispatching_gb_1 import run_vns_improvement as local_run, perturb_solution as local_perturb
        have = True
    except Exception:
        have = False

    if not have:
        return getattr(solution, "objective", float('inf'))

    instance: 'Instance' = solution.instance
    vehicles: List['Vehicle'] = instance.vehicles
    start_time = time.time()

    best_tardiness = local_run(solution, 0.1)
    best_routes = {v.ID: list(getattr(v, "schedules", [])) for v in vehicles}

    iter_count = 0
    while time.time() - start_time < time_limit_sec:
        iter_count += 1
        local_perturb(solution, strength=3)
        current_tardiness = local_run(solution, 0.05)

        if current_tardiness < best_tardiness:
            best_tardiness = current_tardiness
            best_routes = {v.ID: list(getattr(v, "schedules", [])) for v in vehicles}
            print(f"    [ILS] New Best Found: {best_tardiness:.3f}")

        # restore best
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
    # 권장: Gurobi run parameters (적절히 조정)
    GUROBI_TIME_LIMIT = 120.0   # 권장: 120초 이상
    GUROBI_GAP = 1e-2           # 0.01 (초기에는 넉넉히)
    GUROBI_NUMERIC_FOCUS = 2    # 숫자 안정성 강화
    ILS_TIME = 0.3              # 인스턴스당 ILS 개선 시간
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
        # 2. 초기 해 생성: greedy (optional, for warmstart)
        # -------------------------------
        start_dispatch = time.time()
        warmstart_sol = None
        try:
            # use heuristic to create warmstart if available
            from dispatching_gb_1 import dispatch_earliest_vehicle_best_customer
            warmstart_sol = dispatch_earliest_vehicle_best_customer(instance)
            dispatch_time = time.time() - start_dispatch
            print(f"  [Dispatch] Greedy completed - init tardiness: {getattr(warmstart_sol,'objective',float('inf')):.3f}, time {dispatch_time:.3f}s")
        except Exception:
            warmstart_sol = None
            dispatch_time = time.time() - start_dispatch
            print("  [Dispatch] Greedy not available or failed, continuing without warmstart.")

        # -------------------------------
        # 3. Gurobi solve (with safety fallbacks)
        # -------------------------------
        solution = None
        if have_gurobi_solver:
            try:
                # call with recommended params (these default to safer values)
                solution = solve_vrp_gurobi(
                    instance,
                    time_limit=GUROBI_TIME_LIMIT,
                    mip_gap=GUROBI_GAP,
                    use_warmstart=bool(warmstart_sol),
                    verbose=False,
                    numeric_focus=GUROBI_NUMERIC_FOCUS
                )
                print("  [Gurobi Solver] Completed normally.")
            except Exception as e:
                print(f"  [Gurobi Solver] 실행 실패: {e}")
                # Try again with more relaxed settings and no warmstart
                try:
                    print("    [Gurobi] Retrying: loosening gap, disabling warmstart, increasing time.")
                    solution = solve_vrp_gurobi(
                        instance,
                        time_limit=max(GUROBI_TIME_LIMIT, 300.0),
                        mip_gap=1e-1,
                        use_warmstart=False,
                        verbose=False,
                        numeric_focus=2
                    )
                except Exception as e2:
                    print(f"    [Gurobi] fallback failed: {e2}")
                    solution = None
        else:
            print("  [Info] Gurobi solver not available; using pure heuristic solution.")
            try:
                from dispatching_gb_1 import dispatch_earliest_vehicle_best_customer
                solution = dispatch_earliest_vehicle_best_customer(instance)
            except Exception as e:
                print(f"  [Heuristic] failed: {e}")
                solution = None

        if solution is None:
            print("  [Error] 초기 해 생성 실패 — 다음 인스턴스로 넘어갑니다.")
            continue

        # -------------------------------
        # 4. ILS 개선 (가능하면)
        # -------------------------------
        init_tard = getattr(solution, "objective", float('inf'))
        print(f"  [Result] Initial Tardiness: {init_tard:.3f}")
        if getattr(solution, "status", "") != "INFEASIBLE_UNSERVED":
            final_tard = improve_with_ils(solution, time_limit_sec=ILS_TIME)
            total_time = time.time() - start_dispatch
            print(f"  [ILS] Final Tardiness: {final_tard:.3f}, total time: {total_time:.3f}s")
        else:
            final_tard = init_tard
            total_time = dispatch_time

        # -------------------------------
        # 5. record results
        # -------------------------------
        total_customers = sum(len(getattr(v, "schedules", [])) for v in getattr(instance, "vehicles", []))
        if total_customers < len(getattr(instance, "customers", [])):
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
        # 6. optional save/plot
        # -------------------------------
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
                print(f"  [Plot] {img_path} saved")
            except Exception as e:
                print(f"  [Plot Error] {e}")

        if SAVE_EXCEL and export_solution_to_excel:
            try:
                export_solution_to_excel(solution, file_path.replace(".prob", "_solution.xlsx"))
            except Exception as e:
                print(f"  [Export Excel Error] {e}")

    # ===========================================
    #   (4) summary
    # ===========================================
    print("\n\n" + "=" * 60)
    print("📊 Final Summary")
    print("=" * 60)

    if results:
        df = pd.DataFrame(results)
        df.set_index("Instance", inplace=True)

        avg_tard_init = df["Initial_Tardiness"].mean()
        avg_tard_final = df["Final_Tardiness"].mean()
        avg_time_total = df["Total_Time(s)"].mean()

        total_tard_init = df["Initial_Tardiness"].sum()
        total_tard_final = df["Final_Tardiness"].sum()

        try:
            print(df.to_markdown(floatfmt=".3f"))
        except Exception:
            print(df)

        print("\n--- [Average] ---")
        print(f"  Avg Initial Tardiness : {avg_tard_init:.3f}")
        print(f"  Avg Final Tardiness   : {avg_tard_final:.3f}")
        if avg_tard_init > 1e-9:
            print(f"  Improvement (%)       : {(avg_tard_init - avg_tard_final) / avg_tard_init * 100:.2f} %")
        print(f"  Avg Total Time (s)    : {avg_time_total:.3f}")

        print("\n--- [Sum] ---")
        print(f"  Total Initial Tardiness : {total_tard_init:.3f}")
        print(f"  Total Final Tardiness   : {total_tard_final:.3f}")
    else:
        print("No results collected.")
