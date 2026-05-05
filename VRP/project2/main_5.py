# main_5.py
import pickle
import time
import pandas as pd
import glob
import math
import sys
import os
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt

# -------------------------
# 사용자 모듈 import (변경 금지)
# -------------------------
try:
    from module import Instance, Customer, Vehicle, Solution, get_dist
    from dispatching_5 import (
        dispatch_earliest_vehicle_best_customer,
        run_vns_improvement,
        perturb_solution,
        improve_with_ils,   # (선택적) 내부 ILS가 있으면 사용 가능
    )
    try:
        from visualize import plot_solution, export_solution_to_excel
    except ImportError:
        plot_solution = None
        export_solution_to_excel = None
except ImportError as e:
    print("=" * 60)
    print(f"[필수 오류] 파일 import 실패: {e}")
    print("module.py, dispatching_4.py 가 프로젝트 폴더에 있는지 확인하세요.")
    print("=" * 60)
    raise

# -------------------------
# 설정
# -------------------------
INSTANCE_GLOB = "hw2/instance_*.prob"   # 네가 말한 폴더
ILS_TIME_PER_INSTANCE = 1.0             # 초 단위 (인스턴스당 ILS 전체 시간)
VNS_STEP_TIME = 0.08                    # 각 VNS 호출 시간(초) — ILS 내에서 사용
PERTURB_STRENGTH_BASE = 2               # 교란 강도 기본값
SAVE_PLOTS = True
SAVE_EXCEL = False
OUTPUT_DIR = "results_dispatching4"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# 보조: 안전한 평가/읽기
# -------------------------
def safe_get_obj(sol: Solution) -> float:
    # Solution 객체에 .objective 또는 .obj 가 있는 경우 반환, 없으면 계산 시도
    val = getattr(sol, "objective", None)
    if val is None:
        val = getattr(sol, "obj", None)
    if val is None:
        # 계산 시도: 차량들에 schedule이 들어있으면 계산
        try:
            vehs = sol.instance.vehicles
            # use module-provided judge if available; otherwise use naive evaluator
            total = 0.0
            for v in vehs:
                # assume each customer has start/end/tardy set by heuristics; fallback compute
                # We will try to compute tardiness by walking route
                cur_loc = tuple(v.loc)
                cur_time = 0.0
                for c in v.schedules:
                    d = get_dist(cur_loc, tuple(c.loc))
                    travel = d / max(1e-9, float(getattr(v, "speed", 30.0)))
                    arr = cur_time + travel
                    st = max(arr, float(c.tw[0]))
                    end = st + float(getattr(c, "serv_time", 0.0))
                    tard = max(0.0, end - float(c.tw[1]))
                    total += tard
                    cur_time = end
                    cur_loc = tuple(c.loc)
            val = total
        except Exception:
            val = float("inf")
    return float(val) if val is not None else float("inf")

# -------------------------
# ILS implemented in main (so we can record history)
# -------------------------
def ils_with_logging(solution: Solution, time_limit_sec: float,
                     perturb_fn, vns_fn,
                     step_vns_time: float = VNS_STEP_TIME,
                     base_strength: int = PERTURB_STRENGTH_BASE) -> Tuple[float, List[Tuple[float,float]]]:
    """
    Main-level ILS loop (so we capture improvement history).
    - solution: initial Solution (must have instance and vehicles populated)
    - time_limit_sec: total ILS time
    - perturb_fn: function(solution, strength)
    - vns_fn: function(solution, time_limit_sec) -> best_tardiness
    Returns (best_objective, history_list) where history_list = [(elapsed_time, best_obj), ...]
    """
    start = time.time()
    end_time = start + time_limit_sec
    vehicles = solution.instance.vehicles

    # initial evaluation
    best_obj = safe_get_obj(solution)
    history: List[Tuple[float,float]] = [(0.0, best_obj)]

    iteration = 0
    # if provided module has improve_with_ils we could call but we implement our own loop to log
    while time.time() < end_time:
        iteration += 1
        remaining = end_time - time.time()
        if remaining <= 0:
            break

        # 1) perturb
        strength = base_strength + (iteration % 3)
        try:
            perturb_fn(solution, strength=strength)
        except TypeError:
            # maybe signature perturb_solution(solution, strength)
            perturb_fn(solution, strength)

        # 2) local search (VNS)
        # allocate small chunk of time, keep it reasonable
        vns_budget = min(step_vns_time, max(0.02, remaining * 0.25))
        try:
            new_obj = vns_fn(solution, vns_budget)
        except TypeError:
            # fallback if wrapper name differs
            new_obj = vns_fn(solution, vns_budget)

        # vns_fn should update solution in-place (as our dispatching_4 does)
        # recompute objective for safety
        cur_obj = safe_get_obj(solution)

        # if improved, record and keep; otherwise revert to best-known schedules
        if cur_obj < best_obj - 1e-9:
            best_obj = cur_obj
            # snapshot schedules (store copies)
            best_snapshot = {v.ID: list(v.schedules) for v in solution.instance.vehicles}
            history.append((time.time() - start, best_obj))
        else:
            # if not improved, try occasional acceptance (simulated annealing idea)
            # small probability to accept worse solution to escape local minima
            # compute temperature schedule
            frac = min(1.0, (time.time() - start) / max(1e-9, time_limit_sec))
            T0 = max(1.0, max(1.0, best_obj * 0.05))
            T = T0 * (1.0 - frac)
            delta = cur_obj - best_obj
            if delta > 0 and T > 1e-9:
                p = math.exp(-delta / T)
                if random.random() < p:
                    # accept worse solution (but do not change best_obj)
                    history.append((time.time() - start, cur_obj))

        # small safety: if we have saved a best_snapshot and current solution worse, revert
        try:
            if 'best_snapshot' in locals():
                if safe_get_obj(solution) > best_obj + 1e-9:
                    # revert schedules
                    for v in solution.instance.vehicles:
                        v.schedules = list(best_snapshot.get(v.ID, []))
        except Exception:
            pass

    # final restore to best_snapshot if exist
    if 'best_snapshot' in locals():
        for v in solution.instance.vehicles:
            v.schedules = list(best_snapshot.get(v.ID, []))

    solution.objective = best_obj
    return best_obj, history

# -------------------------
# 메인 루프
# -------------------------
def main():
    instance_files = sorted(glob.glob(INSTANCE_GLOB))
    if not instance_files:
        print(f"[오류] '{INSTANCE_GLOB}' 위치에 파일이 없습니다.")
        sys.exit(1)

    results: List[Dict] = []
    # For summary plot: record initial vs final
    summary_initial = []
    summary_final = []

    for prob_file in instance_files:
        print(f"\n=== Processing: {prob_file} ===")
        try:
            with open(prob_file, "rb") as f:
                instance: Instance = pickle.load(f)
        except Exception as e:
            print(f"  [Error] {prob_file} 로딩 실패: {e}")
            continue

        # --- 1) 초기 해 (dispatch) ---
        t0 = time.time()
        try:
            sol0: Solution = dispatch_earliest_vehicle_best_customer(instance)
        except Exception as e:
            print(f"  [Error] dispatch 실패: {e}")
            continue
        t_dispatch = time.time() - t0
        init_obj = safe_get_obj(sol0)
        print(f"  [Dispatch] initial tardiness = {init_obj:.3f}  (time {t_dispatch:.3f}s)")

        # --- 2) ILS (main-level) with logging ---
        ils_start = time.time()
        best_obj, history = ils_with_logging(sol0,
                                            time_limit_sec=ILS_TIME_PER_INSTANCE,
                                            perturb_fn=perturb_solution,
                                            vns_fn=run_vns_improvement,
                                            step_vns_time=VNS_STEP_TIME,
                                            base_strength=PERTURB_STRENGTH_BASE)
        ils_time = time.time() - ils_start
        total_time = time.time() - t0
        print(f"  [ILS] final tardiness = {best_obj:.3f}  (ILS time {ils_time:.3f}s, total {total_time:.3f}s)")

        # --- status ---
        total_customers_in_solution = sum(len(v.schedules) for v in instance.vehicles)
        if total_customers_in_solution < len(instance.customers):
            status = "UNSERVED"
        elif best_obj == float('inf'):
            status = "INFEASIBLE_CAP"
        else:
            status = "DONE"

        # collect results
        results.append({
            "Problem": os.path.basename(prob_file),
            "Initial_Tardiness": init_obj,
            "Final_Tardiness": best_obj,
            "Dispatch_Time(s)": t_dispatch,
            "ILS_Time(s)": ils_time,
            "Total_Time(s)": total_time,
            "Status": status,
            "history": history
        })
        summary_initial.append(init_obj)
        summary_final.append(best_obj)

        # --- save per-instance improvement plot ---
        if SAVE_PLOTS:
            try:
                times = [h[0] for h in history]
                objs = [h[1] for h in history]
                if len(times) >= 1:
                    plt.figure(figsize=(6,4))
                    plt.step(times, objs, where='post', marker='o')
                    plt.xlabel("Elapsed time (s) since ILS start")
                    plt.ylabel("Best objective (total tardiness)")
                    plt.title(f"Improvement trace - {os.path.basename(prob_file)}")
                    plt.grid(True)
                    fname = os.path.join(OUTPUT_DIR, os.path.basename(prob_file).replace(".prob","_trace.png"))
                    plt.savefig(fname, bbox_inches="tight", dpi=150)
                    plt.close()
                    print(f"  [Plot] improvement trace saved -> {fname}")
            except Exception as e:
                print(f"  [Plot] 실패: {e}")

        # optional: export final route image or excel using visualize if available
        if SAVE_PLOTS and plot_solution is not None:
            try:
                imgpath = os.path.join(OUTPUT_DIR, os.path.basename(prob_file).replace(".prob","_final.png"))
                plot_solution(sol0, annotate=True, show_time_windows=True, save_path=imgpath, write_back=True)
                print(f"  [Plot] final route image saved -> {imgpath}")
            except Exception as e:
                print(f"  [Plot] final route save 실패: {e}")

        if SAVE_EXCEL and export_solution_to_excel is not None:
            try:
                excel_path = os.path.join(OUTPUT_DIR, os.path.basename(prob_file).replace(".prob","_solution.xlsx"))
                export_solution_to_excel(solution=sol0, filepath=excel_path)
                print(f"  [Excel] 솔루션 저장 -> {excel_path}")
            except Exception as e:
                print(f"  [Excel] 저장 실패: {e}")

    # -------------------------
    # 결과 요약 및 전체 그래프
    # -------------------------
    if results:
        df = pd.DataFrame(results)
        df_summary = df.drop(columns=["history"])
        df_summary.set_index("Problem", inplace=True)
        print("\n" + "="*60)
        print("Summary results:")
        try:
            print(df_summary.to_markdown(floatfmt=".3f"))
        except Exception:
            print(df_summary)

        # summary bar chart: initial vs final
        if SAVE_PLOTS:
            try:
                problems = df_summary.index.tolist()
                x = range(len(problems))
                fig, ax = plt.subplots(figsize=(max(8, len(problems)*0.25), 4))
                ax.bar(x, df_summary["Initial_Tardiness"].values, width=0.4, label="Initial")
                ax.bar([xi+0.4 for xi in x], df_summary["Final_Tardiness"].values, width=0.4, label="Final")
                ax.set_xticks([xi+0.2 for xi in x])
                ax.set_xticklabels(problems, rotation=90, fontsize=8)
                ax.set_ylabel("Total tardiness")
                ax.set_title("Initial vs Final Tardiness per instance")
                ax.legend()
                plt.tight_layout()
                fname = os.path.join(OUTPUT_DIR, "summary_initial_vs_final.png")
                plt.savefig(fname, dpi=150)
                plt.close()
                print(f"\n[Plot] summary saved -> {fname}")
            except Exception as e:
                print(f"[Plot] summary plot 실패: {e}")

        # print averages
        avg_init = df_summary["Initial_Tardiness"].mean()
        avg_final = df_summary["Final_Tardiness"].mean()
        avg_time = df_summary["Total_Time(s)"].mean()
        print(f"\nAverage initial tardiness : {avg_init:.3f}")
        print(f"Average final tardiness   : {avg_final:.3f}")
        if avg_init > 1e-9:
            print(f"Overall improvement       : {(avg_init - avg_final)/avg_init*100:.2f} %")
        print(f"Average total time (per inst): {avg_time:.3f} s")

        # save results table CSV
        out_csv = os.path.join(OUTPUT_DIR, "results_summary.csv")
        df_summary.to_csv(out_csv)
        print(f"\n[Save] results csv -> {out_csv}")

    else:
        print("No results. Check instances path and imports.")

if __name__ == "__main__":
    main()
