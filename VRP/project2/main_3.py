import os
import pickle
import time
from typing import List, Dict, Any, Optional
import pandas as pd
from tabulate import tabulate

# VRP 모듈 및 Solver 로직 임포트
from module import Instance, Solution
from dispatching_3 import dispatch_earliest_vehicle_best_customer, iterated_local_search

# 필요한 상수 및 설정
NUM_PROBLEMS = 30
FILE_PREFIX = "instance_"
FILE_SUFFIX = ".prob"
FOLDER_NAME = "hw2"


def load_instance(problem_id: int) -> Optional[Instance]:
    """
    문제 인스턴스 파일을 pickle을 사용하여 역직렬화(Deserialization)하여 복원합니다.
    """
    filename = f"{FILE_PREFIX}{problem_id}{FILE_SUFFIX}"
    filepath = os.path.join(FOLDER_NAME, filename)

    try:
        with open(filepath, 'rb') as f:
            instance = pickle.load(f)
            # 복원된 객체가 Instance 클래스인지 확인
            if isinstance(instance, Instance):
                return instance
            else:
                print(f"경고: {filepath} 파일이 VRP Instance 객체가 아닙니다.")
                return None
    except FileNotFoundError:
        print(f"오류: 파일 경로를 찾을 수 없습니다: {filepath}")
        print("실행을 위해서는 '{FOLDER_NAME}' 폴더 안에 '{FILE_PREFIX}X{FILE_SUFFIX}' 파일이 있어야 합니다.")
        return None
    except Exception as e:
        print(f"오류: {filepath} 로드 중 예외 발생: {e}")
        return None


def solve_and_record(problem_id: int) -> Dict[str, Any]:
    """
    단일 인스턴스를 로드, 디스패칭, ILS로 최적화하고 결과를 기록합니다.
    """
    instance = load_instance(problem_id)

    if instance is None:
        return {
            "Problem": os.path.join(FOLDER_NAME, f"{FILE_PREFIX}{problem_id}{FILE_SUFFIX}"),
            "Initial_Tardiness": 0.0,
            "Final_Tardiness(ILS)": 0.0,
            "Dispatch_Time(s)": 0.0,
            "Total_Time(s)": 0.0,
            "Status": "MISSING_FILE",
        }

    problem_name = os.path.join(FOLDER_NAME, f"{FILE_PREFIX}{problem_id}{FILE_SUFFIX}")

    # **핵심 수정**: 매 실행 전에 모든 인스턴스 상태 초기화
    # Instance.reset()은 이제 Customer의 속성도 수동으로 초기화합니다.
    instance.reset()

    # 1. 초기 해 (Greedy Dispatch) 계산 및 시간 측정
    dispatch_start_time = time.time()
    initial_sol = dispatch_earliest_vehicle_best_customer(
        instance,
    )
    dispatch_time = time.time() - dispatch_start_time

    initial_tardiness = initial_sol.objective

    # 2. 최적화 (Iterated Local Search) 실행 및 시간 측정
    MAX_ILS_ITER = 5  # ILS 반복 횟수
    MAX_LS_ITER = 50  # Local Search 반복 횟수

    best_sol, ils_time = iterated_local_search(
        initial_sol,
        max_ils_iterations=MAX_ILS_ITER,
        max_ls_iter=MAX_LS_ITER,
        perturbation_strength=2
    )

    total_time = dispatch_time + ils_time
    final_tardiness = best_sol.objective

    return {
        "Problem": problem_name,
        "Initial_Tardiness": final_tardiness,  # ILS 전의 initial_sol.objective를 사용합니다.
        "Final_Tardiness(ILS)": final_tardiness,
        "Dispatch_Time(s)": dispatch_time,
        "Total_Time(s)": total_time,
        "Status": best_sol.status,
    }


def main():
    """메인 실행 함수: 30개 인스턴스 해결 및 결과 출력."""

    all_results: List[Dict[str, Any]] = []

    # 1부터 30까지 인스턴스 처리
    for i in range(1, NUM_PROBLEMS + 1):
        result = solve_and_record(i)
        all_results.append(result)
        # print(f"Processing {result['Problem']}... Done. Final Tardiness: {result['Final_Tardiness(ILS)']:.3f}")

    # 데이터프레임으로 변환하여 표 형식 준비
    df = pd.DataFrame(all_results)

    # 계산된 값의 정확도를 맞춤 (소수점 3자리)
    df['Initial_Tardiness'] = df['Initial_Tardiness'].round(3)
    df['Final_Tardiness(ILS)'] = df['Final_Tardiness(ILS)'].round(3)
    df['Dispatch_Time(s)'] = df['Dispatch_Time(s)'].round(3)
    df['Total_Time(s)'] = df['Total_Time(s)'].round(3)

    # 요청하신 형식에 맞춰 표 출력 (tabulate 라이브러리 사용)
    table = tabulate(
        df,
        headers='keys',
        tablefmt='pipe',
        showindex=False,
        numalign='right',
        stralign='left'
    )

    print("\n" + "=" * 50)
    print("           최종 요약 결과 (Summary Results)")
    print("=" * 50)
    print(table)

    # 전체 요약 통계 계산 및 출력
    avg_initial_tardiness = df['Initial_Tardiness'].mean()
    avg_final_tardiness = df['Final_Tardiness(ILS)'].mean()

    # 개선율 계산
    improvement_rate = 0.0
    if avg_initial_tardiness > 1e-9:
        improvement_rate = (avg_initial_tardiness - avg_final_tardiness) / avg_initial_tardiness * 100

    avg_total_time = df['Total_Time(s)'].mean()

    total_initial_tardiness = df['Initial_Tardiness'].sum()
    total_final_tardiness = df['Final_Tardiness(ILS)'].sum()

    print("-" * 50)
    print("          전체 평균 (Average Performance)")
    print(f"  - 평균 초기 Tardiness: {avg_initial_tardiness:.3f}")
    print(f"  - 평균 최종 Tardiness: {avg_final_tardiness:.3f}")
    print(f"  - 개선율 (Improvement): {improvement_rate:.2f} %")
    print(f"  - 문제당 평균 총 시간: {avg_total_time:.4f} 초 (Dispatch + ILS {MAX_ILS_ITER}회)")

    print("-" * 50)
    print("          전체 총합 (Total Performance)")
    print(f"  - 총 초기 Tardiness: {total_initial_tardiness:.3f}")
    print(f"  - 총 최종 Tardiness: {total_final_tardiness:.3f}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()