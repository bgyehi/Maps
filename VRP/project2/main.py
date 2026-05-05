import random
from module import make_random_instance
from visualize import plot_solution, export_solution_to_excel, load_solution_from_excel
from dispatching import dispatch_earliest_vehicle_best_customer
import pandas as pd

if __name__ == "__main__":

    for i in range(0, 30):
        num_customers = random.randint(5, 15)
        num_vehicles = random.randint(2, 5)
        instance = make_random_instance(num_customers=num_customers, num_vehicles=num_vehicles, seed=i+1)
        solution_disp = dispatch_earliest_vehicle_best_customer(
            instance,
            scoring="shortest_distance",  # or "earliest_finish", "edd"
            tie_breaker="edd_then_nearest",
        )
        for k, v in enumerate(instance.vehicles):
            print(f"Vehicle {v.ID} route:", [c.ID for c in v.schedules])
            for c in v.schedules:
                print(f"  Customer {c.ID}: start={c.start:.3f}, end={c.end:.3f}, due={c.tw[1]:.3f}")
        sol = load_solution_from_excel(filepath='test.xlsx')
        export_solution_to_excel(solution=solution_disp, filepath='test.xlsx')

        metrics = plot_solution(
            solution_disp,
            annotate=True,
            show_time_windows=True,
            arrows=False,
            save_path="routes_disp.png",  # e.g., "routes.png"
            write_back=False  # set True if you want start/end/tardiness written into customer objs
        )

