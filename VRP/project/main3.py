import os
import glob
import pickle
import time
import pandas as pd
import random
import sys

# imports from our dispatching module
from dispatching3 import dispatch_earliest_vehicle_best_customer, run_vns_improvement, perturb_solution, calculate_total_tardiness

# optional visualize (if present)
try:
    from visualize import plot_solution, export_solution_to_excel
except Exception:
    plot_solution = None
    export_solution_to_excel = None


def improve_with_ils(solution: 'Solution', time_limit_sec: float) -> float:
    """
    ILS outer loop: perturb -> short VNS -> accept if improved.
    Returns best tardiness found (and mutates solution.instance.vehicles schedules).
    """
    instance = solution.instance
    vehicles = instance.vehicles
    start_time = time.time()

    # initial local search to improve initial solution a bit
    best_tard = run_vns_improvement(solution, 0.1)
    best_schedules = {v.ID: list(v.schedules) for v in vehicles}

    iteration = 0
    # continue until time limit
    while (time.time() - start_time) < time_limit_sec:
        iteration += 1
        # perturb (safe)
        perturb_solution(solution, strength=3)
        # short local search
        cur_tard = run_vns_improvement(solution, 0.05)
        if cur_tard < best_tard:
            best_tard = cur_tard
            best_schedules = {v.ID: list(v.schedules) for v in vehicles}
            # debug print
            print(f"\n    [ILS] Iter {iteration} improved -> {best_tard:.3f}")
        else:
            # revert to best schedules
            for v in vehicles:
                v.schedules = list(best_schedules[v.ID])

    # write back best objective
    solution.objective = best_tard
    return best_tard


def main():
    random.seed(1)
    instance_files = sorted(glob.glob("instances/instance_*.prob"))
    if not instance_files:
        print("No instance_*.prob files found in current folder.")
        return

    results = []
    ILS_TIME_PER_INSTANCE = 0.3  # per-instance time budget

    for prob_file in instance_files:
        print(f"\n--- Processing: {prob_file} ---")
        try:
            with open(prob_file, "rb") as f:
                instance = pickle.load(f)
        except Exception as e:
            print(f"Failed to load {prob_file}: {e}")
            continue

        # 1) Greedy dispatch (baseline)
        t0 = time.time()
        sol = dispatch_earliest_vehicle_best_customer(instance)
        t_dispatch = time.time() - t0
        initial_tard = sol.objective if hasattr(sol, "objective") else calculate_total_tardiness(instance.vehicles)
        print(f"  [Dispatch] initial tardiness={initial_tard:.3f}  (time {t_dispatch:.3f}s)")

        # 2) ILS improvement
        if sol.status == "INFEASIBLE_UNSERVED":
            print("  [ILS] skipped due to unserved customers in dispatch.")
            final_tard = initial_tard
            total_time = t_dispatch
        else:
            t_start_ils = time.time()
            final_tard = improve_with_ils(sol, ILS_TIME_PER_INSTANCE)
            total_time = time.time() - t_start_ils + t_dispatch
            print(f"  [ILS] final tardiness={final_tard:.3f} (ILS time approx {ILS_TIME_PER_INSTANCE}s)")

        # sanity check
        total_assigned = sum(len(v.schedules) for v in instance.vehicles)
        status = sol.status
        if total_assigned < len(instance.customers):
            status = "BUG_UNSERVED"
        elif final_tard == float("inf"):
            status = "BUG_INFEASIBLE_CAP"
        else:
            status = "DONE"

        results.append({
            "Problem": os.path.basename(prob_file),
            "Initial_Tardiness": initial_tard,
            "Final_Tardiness": final_tard,
            "Dispatch_Time(s)": round(t_dispatch, 4),
            "Total_Time(s)": round(total_time, 4),
            "Status": status
        })

        # optional: per-instance exports
        if export_solution_to_excel:
            try:
                out_xlsx = prob_file.replace(".prob", "_solution.xlsx")
                export_solution_to_excel(solution=sol, filepath=out_xlsx)
            except Exception:
                pass
        if plot_solution:
            try:
                out_img = prob_file.replace(".prob", "_solution.png")
                plot_solution(sol, annotate=True, show_time_windows=True, arrows=False, save_path=out_img, write_back=True)
            except Exception:
                pass

    # --- 추가된 요약 출력 ---
    if results:
        df = pd.DataFrame(results).set_index("Problem")
        print("\n\n" + "=" * 60)
        print("           최종 요약 결과 (Summary Results)")
        print("=" * 60)

        # 표 출력
        try:
            print(df.to_markdown(floatfmt=".3f"))
        except Exception:
            print(df)

        # 총합 / 평균 계산
        avg_initial = df["Initial_Tardiness"].mean()
        avg_final = df["Final_Tardiness"].mean()
        avg_time = df["Total_Time(s)"].mean()
        total_initial = df["Initial_Tardiness"].sum()
        total_final = df["Final_Tardiness"].sum()

        print("\n" + "-" * 60)
        print("          전체 평균 (Average Performance)")
        print(f"  - 평균 초기 Tardiness : {avg_initial:.3f}")
        print(f"  - 평균 최종 Tardiness : {avg_final:.3f}")
        if avg_initial > 1e-9:
            print(f"  - 개선율 (Improvement) : {((avg_initial - avg_final) / avg_initial * 100):.2f} %")
        print(f"  - 문제당 평균 총 시간 : {avg_time:.3f} 초")

        print("\n" + "-" * 60)
        print("          전체 총합 (Total Performance)")
        print(f"  - 총 초기 Tardiness : {total_initial:.3f}")
        print(f"  - 총 최종 Tardiness : {total_final:.3f}")
        print("=" * 60)

        # CSV 저장
        df.to_csv("results_summary.csv", index=True, encoding="utf-8-sig")
        print("\nSaved results_summary.csv ✅")
    else:
        print("No valid results generated. Check instance files or dispatching logic.")


if __name__ == "__main__":
    main()
