# ================================================================
# main.py  (Dispatch + Gurobi + Summary Table)
# ================================================================

import os
import pandas as pd
import time
from tabulate import tabulate
from module import load_instance
from dispatching import dispatch_earliest_vehicle_best_customer
from gurobi_solver import solve_vrptw_gurobi


INSTANCE_DIR = "./instances"


def run_all():
    rows = []
    rows_gurobi = []

    inst_files = sorted([f for f in os.listdir(INSTANCE_DIR) if f.endswith(".prob")])

    for fname in inst_files:
        path = os.path.join(INSTANCE_DIR, fname)
        instance = load_instance(path)

        # ---------- Dispatch ----------
        t0 = time.time()
        disp_sol = dispatch_earliest_vehicle_best_customer(instance)
        t1 = time.time()
        disp_time = t1 - t0

        initial_tardiness = sum(c.tardy for c in instance.customers)

        # ---------- Gurobi ----------
        t2 = time.time()
        gurobi = solve_vrptw_gurobi(instance, time_limit=120, mipgap=0.01)
        t3 = time.time()
        gurobi_time = t3 - t2

        final_tardiness = gurobi["objective"]

        rows.append([
            fname,
            initial_tardiness,
            final_tardiness,
            disp_time,
            disp_time + gurobi_time,
            "DONE"
        ])

        rows_gurobi.append([initial_tardiness, final_tardiness, disp_time + gurobi_time])

    # ===========================
    # 표 출력
    # ===========================
    df = pd.DataFrame(rows,
        columns=[
            "Problem", "Initial_Tardiness", "Final_Tardiness",
            "Dispatch_Time(s)", "Total_Time(s)", "Status"
        ])

    print("\n============================================================")
    print("           최종 요약 결과 (Summary Results)")
    print("============================================================")
    print(tabulate(df, headers="keys", tablefmt="github", floatfmt=".3f"))

    # ===== 평균 =====
    avg_init = df["Initial_Tardiness"].mean()
    avg_final = df["Final_Tardiness"].mean()
    avg_time = df["Total_Time(s)"].mean()
    imp = (avg_init - avg_final) / avg_init * 100

    print("\n------------------------------------------------------------")
    print("          전체 평균 (Average Performance)")
    print(f"  - 평균 초기 Tardiness : {avg_init:.3f}")
    print(f"  - 평균 최종 Tardiness : {avg_final:.3f}")
    print(f"  - 개선율 (Improvement) : {imp:.2f} %")
    print(f"  - 문제당 평균 총 시간 : {avg_time:.3f} 초")

    # ===== 총합 =====
    sum_init = df["Initial_Tardiness"].sum()
    sum_final = df["Final_Tardiness"].sum()

    print("\n------------------------------------------------------------")
    print("          전체 총합 (Total Performance)")
    print(f"  - 총 초기 Tardiness : {sum_init:.3f}")
    print(f"  - 총 최종 Tardiness : {sum_final:.3f}")
    print("============================================================\n")

    df.to_csv("results_summary.csv", index=False)
    print("Saved results_summary.csv ✅\n")

    # ===========================
    # Gurobi 성능 별도 요약
    # ===========================
    gdf = pd.DataFrame(rows_gurobi, columns=["Initial", "Final", "Time"])
    avg_i = gdf["Initial"].mean()
    avg_f = gdf["Final"].mean()
    sum_i = gdf["Initial"].sum()
    sum_f = gdf["Final"].sum()

    imp2 = (avg_i - avg_f) / avg_i * 100

    print("[구로비]\n")
    print("--- [평균 성능] ---")
    print(f"  평균 초기 Tardiness : {avg_i:.3f}")
    print(f"  평균 최종 Tardiness : {avg_f:.3f}")
    print(f"  개선율              : {imp2:.2f} %")
    print(f"  평균 총 실행시간     : {gdf['Time'].mean():.3f} 초\n")

    print("--- [총합 성능] ---")
    print(f"  총 초기 Tardiness   : {sum_i:.3f}")
    print(f"  총 최종 Tardiness   : {sum_f:.3f}")
    print("============================================================")


if __name__ == "__main__":
    run_all()
