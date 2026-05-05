# main_dispatch_basic.py
import random
import pandas as pd
import matplotlib.pyplot as plt
from module import make_random_instance
from visualize import export_solution_to_excel
from dispatching_추가ver import dispatch_earliest_vehicle_best_customer
from local_search import evaluate_solution

if __name__ == "__main__":
    results = []

    for i in range(30):
        num_customers = random.randint(5, 15)
        num_vehicles = random.randint(2, 5)

        instance = make_random_instance(num_customers=num_customers, num_vehicles=num_vehicles, seed=i+1)
        solution = dispatch_earliest_vehicle_best_customer(instance)

        tard, dist, unserved = evaluate_solution(instance)

        print(f"Instance {i+1}: tardiness={tard:.3f}, distance={dist:.3f}, unserved={unserved}")

        results.append({
            "instance": i+1,
            "num_customers": num_customers,
            "num_vehicles": num_vehicles,
            "tardiness": tard,
            "distance": dist,
            "unserved": unserved
        })

    df = pd.DataFrame(results)
    print("\n=== Average Results (Basic Dispatching) ===")
    print(df.mean(numeric_only=True))

    df.to_excel("results_basic_dispatch.xlsx", index=False)
    print("✅ Saved to results_basic_dispatch.xlsx")

    plt.figure(figsize=(7,5))
    plt.plot(df["instance"], df["tardiness"], marker="o")
    plt.xlabel("Instance")
    plt.ylabel("Total Tardiness")
    plt.title("Basic Dispatching: Tardiness per Instance")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("tardiness_basic_dispatch.png")
    plt.show()
