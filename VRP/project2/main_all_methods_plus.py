# ========================================
# VRP 통합 실행 코드 (VNS, SPT, EDD, ILS 포함)
# Customer 좌표 문제 수정 완료
# ========================================

import pickle
import time
import pandas as pd
import glob
import os
import copy
import matplotlib.pyplot as plt


# --------------------------
# Customer / Instance / Solution 클래스
# --------------------------
class Customer:
    def __init__(self, id, demand, location):
        self.id = id
        self.demand = demand
        self.location = location  # (x, y) 튜플
        self.x = location[0]  # plot_solution 호환
        self.y = location[1]


class Instance:
    def __init__(self, customers):
        self.customers = customers  # list of Customer objects


class Solution:
    def __init__(self, routes):
        self.routes = routes  # list of lists: [[customer indices], ...]


# --------------------------
# deepcopy 용 함수
# --------------------------
def deep_copy_solution(inst):
    return copy.deepcopy(inst)


# --------------------------
# plot_solution
# --------------------------
def plot_solution(instance, solution, title="", filename=None):
    depot_x = instance.customers[0].x
    depot_y = instance.customers[0].y

    plt.figure(figsize=(8, 6))

    # 고객 좌표
    x_coords = [c.x for c in instance.customers[1:]]  # depot 제외
    y_coords = [c.y for c in instance.customers[1:]]
    plt.scatter(x_coords, y_coords, c='blue', label='Customers')

    # depot 표시
    plt.scatter(depot_x, depot_y, c='red', marker='s', label='Depot')

    # 경로 plot
    for route in solution.routes:
        route_x = [instance.customers[i].x for i in route]
        route_y = [instance.customers[i].y for i in route]
        plt.plot(route_x, route_y, alpha=0.6)

    plt.title(title)
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")

    if filename:
        plt.savefig(filename)
    plt.show()


# --------------------------
# 예시 dispatching 방법 (SPT, EDD, ILS, VNS) 모듈
# 실제 구현은 dispatching_all_methods.py에서 import 가능
# --------------------------
def SPT(instance):
    # 간단 예시: 고객 demand 기준 정렬
    sorted_customers = sorted(range(1, len(instance.customers)), key=lambda i: instance.customers[i].demand)
    return Solution([sorted_customers])


def EDD(instance):
    # 간단 예시: customer id 기준
    sorted_customers = sorted(range(1, len(instance.customers)), key=lambda i: instance.customers[i].id)
    return Solution([sorted_customers])


def ILS(instance):
    # dummy 예시
    sorted_customers = list(range(1, len(instance.customers)))
    return Solution([sorted_customers])


def VNS(instance):
    # dummy 예시
    sorted_customers = list(range(1, len(instance.customers)))
    return Solution([sorted_customers])


# --------------------------
# 인스턴스 파일 로딩
# --------------------------
instance_files = sorted(glob.glob("hw2/instance_*.prob"))  # 실제 경로 확인 필요
results = []

for prob_file in instance_files:
    print(f"Processing {prob_file}...")

    # pickle 로드 예시 (Customer list 포함)
    with open(prob_file, "rb") as f:
        customers = pickle.load(f)  # instance.customers가 pickle에 저장되어 있다고 가정

    inst = Instance(customers)

    # 모든 알고리즘 실행
    methods = {"SPT": SPT, "EDD": EDD, "ILS": ILS, "VNS": VNS}
    best_obj = float('inf')
    best_solution = None
    best_method = ""

    for name, func in methods.items():
        start_time = time.time()
        sol = func(inst)
        end_time = time.time()

        # 예시 objective: 경로 길이 합 (간단히 고객 수)
        obj = sum(len(route) for route in sol.routes)
        print(f"[{name}] objective: {obj}")

        if obj < best_obj:
            best_obj = obj
            best_solution = sol
            best_method = name

    # plot 저장
    fig_name = os.path.basename(prob_file).replace(".prob", f"_{best_method}.png")
    plot_solution(inst, best_solution, title=f"{os.path.basename(prob_file)} | {best_method} | obj={best_obj}",
                  filename=fig_name)

    results.append({
        "file": prob_file,
        "best_method": best_method,
        "best_obj": best_obj
    })

# 결과 summary
df = pd.DataFrame(results)
print(df)
df.to_csv("summary_results.csv", index=False)
