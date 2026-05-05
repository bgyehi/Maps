# main_local_search.py
import random
import pandas as pd
import matplotlib.pyplot as plt
from module import make_random_instance
from visualize import export_solution_to_excel
from dispatching_추가ver import dispatch_earliest_vehicle_best_customer
from local_search import local_search_improve, evaluate_solution

if __name__ == "__main__":
    results = []

    for i in range(30):
        num_customers = random.randint(5, 15)
        num_vehicles = random.randint(2, 5)

        # 초기해 생성 (기본 디스패칭 기반)
        instance = make_random_instance(num_customers=num_customers, num_vehicles=num_vehicles, seed=i+1)
        base_solution = dispatch_earliest_vehicle_best_customer(instance)
        base_tard, _, _ = evaluate_solution(instance)

        # 로컬 서치 개선
        improved_instance = local_search_improve(instance, max_iters=300)
        tard, dist, unserved = evaluate_solution(improved_instance)

        print(f"Instance {i+1}: before={base_tard:.3f} → after={tard:.3f}")

        results.append({
            "instance": i+1,
            "num_customers": num_customers,
            "num_vehicles": num_vehicles,
            "before_tardiness": base_tard,
            "after_tardiness": tard,
            "improvement": base_tard - tard
        })

    df = pd.DataFrame(results)
    print("\n=== Average Improvement (Local Search) ===")
    print(df.mean(numeric_only=True))

    df.to_excel("results_local_search.xlsx", index=False)
    print("✅ Saved to results_local_search.xlsx")

    plt.figure(figsize=(7,5))
    plt.plot(df["instance"], df["before_tardiness"], marker="x", label="Before (Dispatch)")
    plt.plot(df["instance"], df["after_tardiness"], marker="o", label="After (Local Search)")
    plt.xlabel("Instance")
    plt.ylabel("Total Tardiness")
    plt.title("Local Search Improvement per Instance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("tardiness_local_search_comparison.png")
    plt.show()
