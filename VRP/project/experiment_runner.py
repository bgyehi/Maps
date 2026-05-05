# experiment_runner.py

import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import pickle
import glob
import csv
import statistics
from copy import deepcopy

# ✅ 현재 스크립트 기준으로 project 폴더를 sys.path 최상단에 강제 추가
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# ✅ 현재 경로 확인용 (임시로 확인 후 나중에 주석 처리해도 됨)
print("현재 경로:", CURRENT_DIR)
print("sys.path 첫 항목:", sys.path[0])

from dispatching import dispatch_earliest_vehicle_best_customer, dispatch_weighted_score
import local_search as ls


# settings
INSTANCE_DIR = "instances"
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

# algorithm configs
weights = (1.0, 0.05, 0.2)  # (w_tard, w_dist, w_prio) - 튜닝 가능
sa_params = dict(initial_temp=1.0, final_temp=1e-3, alpha=0.9, iter_per_temp=100)

# helper loader: try pickle
def load_instance_from_prob(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj

# get instance files (sorted)
files = sorted(glob.glob(os.path.join(INSTANCE_DIR, "*.prob")))
files = files[:30]  # ensure only 30

rows = []
summary = []

for idx, fpath in enumerate(files, start=1):
    print(f"[{idx}/{len(files)}] Loading {fpath} ...")
    try:
        instance = load_instance_from_prob(fpath)
    except Exception as e:
        print("Failed to load instance:", e)
        continue

    # ensure fresh copy per algorithm
    inst0 = deepcopy(instance)
    inst1 = deepcopy(instance)
    inst2 = deepcopy(instance)
    inst3 = deepcopy(instance)

    # --- baseline greedy dispatch (original) ---
    sol_base = dispatch_earliest_vehicle_best_customer(inst0, scoring="delta_tardiness")
    base_tard = sol_base.objective
    base_dist = getattr(sol_base, "total_distance", 0.0)
    base_unserved = len(getattr(sol_base, "unserved_ids", []))

    # --- weighted dispatch (no local search) ---
    sol_w = dispatch_weighted_score(inst1, weights=weights)
    w_tard = sol_w.objective
    w_dist = getattr(sol_w, "total_distance", 0.0)
    w_unserved = len(getattr(sol_w, "unserved_ids", []))

    # --- weighted + greedy local search ---
    # note: dispatch_weighted_score already filled inst2 vehicles' schedules
    # we will apply local search on inst2 (deepcopy) and then re-evaluate tardiness
    # Make sure instance passed to local search has vehicles.schedules set (we used inst1 for that)
    inst2 = deepcopy(inst1)
    inst2 = ls.local_search_improve(inst2, max_iters=200)
    ls_tard, ls_dist, ls_unserved = ls.evaluate_solution(inst2)

    # --- weighted + SA ---
    inst3 = deepcopy(inst1)
    inst3_sa = ls.simulated_annealing_on_solution(inst3, **sa_params)
    sa_tard, sa_dist, sa_unserved = ls.evaluate_solution(inst3_sa)

    rows.append({
        "instance_file": os.path.basename(fpath),
        "base_tard": base_tard,
        "base_dist": base_dist,
        "base_unserved": base_unserved,
        "weighted_tard": w_tard,
        "weighted_dist": w_dist,
        "weighted_unserved": w_unserved,
        "local_tard": ls_tard,
        "local_dist": ls_dist,
        "local_unserved": ls_unserved,
        "sa_tard": sa_tard,
        "sa_dist": sa_dist,
        "sa_unserved": sa_unserved
    })

# write CSV
csv_path = os.path.join(OUT_DIR, "experiment_results.csv")
with open(csv_path, "w", newline='') as csvfile:
    fieldnames = list(rows[0].keys()) if rows else []
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

# summary stats
def stats(vals):
    return {"mean": statistics.mean(vals), "stdev": statistics.pstdev(vals) if len(vals)>1 else 0.0}

base_tards = [r["base_tard"] for r in rows]
w_tards = [r["weighted_tard"] for r in rows]
local_tards = [r["local_tard"] for r in rows]
sa_tards = [r["sa_tard"] for r in rows]

summary = {
    "base": stats(base_tards),
    "weighted": stats(w_tards),
    "local": stats(local_tards),
    "sa": stats(sa_tards)
}

import json
with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("Experiment finished. Results saved to:", OUT_DIR)
