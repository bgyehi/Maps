import pandas as pd  # 결과를 DataFrame 형태로 저장하기 위한 라이브러리
import config  # 실험 환경설정(모델명, 파라미터 등)을 관리하는 설정 모듈
from exact import milp_scheduling  # MILP 기반 exact 스케줄링 함수
from module import *  # 인스턴스 생성/로드 등 공통 함수
from exact import cp, milp  # exact 해법들(CP, MILP) 비교 실험용 모듈
from heuristic import dispatching  # dispatching rule 기반 휴리스틱 모듈
from llm import PLAID, ATONCE, test, train  # LLM 기반 스케줄링 및 학습/테스트 실험용 모듈
import exact.milpcg2  # 자동 exact solver 모듈

Rules = ['SPT', 'EDD', 'MST', 'COVERT_nosetup', 'CR', 'SLACK']  # 비교 가능한 dispatching rule 목록

if __name__ == '__main__':  # 이 파일을 직접 실행할 때만 아래 실험 코드 수행

    summary = pd.DataFrame(columns=['Num', 'wT', 'Time', 'Status'])
    # 여러 문제 인스턴스에 대해 반복 실험
    for i in range(1, 31):
        inst = load_instance_from_json("EO_data\Training_Small\instance_{}.json".format(i))
        solver = exact.milpcg2.solve_instance_auto(inst, 3600, 3600)  # exact solver 실행_시간 제한 3600초
        new_row = {}
        new_row['Num'] = i
        new_row['wT'] = solver['objective']
        new_row['Time'] = solver['total_time_spent_sec']
        new_row['Status'] = solver['status']
        summary = summary._append(new_row, ignore_index=True)
        summary.to_excel("Lag_Test.xlsx", sheet_name='test')

    result = milp_scheduling(inst, 3600)  # 마지막으로 불러온 인스턴스에 대해 기본 MILP 스케줄링 실행

    summary = pd.DataFrame(columns=['Num', 'wT', 'Time', 'h'])

    instance = generate_prob(numJob=5, numMch=2, setup=True, family=False, method='Schutten',
                             identical_mch=False)  # 스케줄링 문제 인스턴스 객체 생성
    instance.loadFile('dataset/example/instance_{0}.prob'.format(16))

    for i in range(0, 36):
        # num_job = random.randint(100, 300)
        # num_mch = random.randint(4, 7)
        num_job = 5
        num_mch = 2
        instance = generate_prob(numJob=num_job, numMch=num_mch, setup=True, family=False, method='Schutten',
                                 identical_mch=False)
        instance.loadFile('dataset/noweight/instance_{0}.prob'.format(i + 1))

        new_row = {}
        new_row['Num'] = "Train_Small_{}".format(i + 1)

        atc_solution = dispatching.scheduling(instance,
                                              'ATCS')  # Fluctuate Performance # dispatching rule(ATCS)을 사용해 휴리스틱 스케줄 생성
        atc_solution2 = PLAID.scheduling_openai_keep(instance, with_ft=False)  # Fluctuate Performance

        new_row['wT'] = atc_solution.objective
        new_row['Time'] = atc_solution.comp_time
        new_row['h'] = atc_solution.objective

        summary = summary._append(new_row, ignore_index=True)
        summary.to_excel("test_1.xlsx", sheet_name='test')
        print(i + 1)

    edd_solution = dispatching.scheduling(instance, rule='MST')

    openai_solution_1 = PLAID.scheduling(instance, model='openai')
    print('Done')
