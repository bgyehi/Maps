
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
from ortools.sat.python import cp_model
from datetime import datetime, timedelta
import warnings

# AUTO-INJECTED: Korean font setup for matplotlib
import os as _os
import matplotlib.font_manager as _fm
import matplotlib.pyplot as _plt
if not any('NanumGothic' in f.name for f in _fm.fontManager.ttflist):
    for _font in ['/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
                  '/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf']:
        if _os.path.exists(_font):
            _fm.fontManager.addfont(_font)
_plt.rcParams.update({'font.family': 'NanumGothic', 'axes.unicode_minus': False})
del _os, _fm, _plt
# END AUTO-INJECTED Korean font setup

warnings.filterwarnings('ignore')

print("=" * 80)
print("MCS ROUTING 최적화 시스템 - 최적해 보장 (OR-Tools CP-SAT)")
print("Location-Routing Problem for Mobile Charging Station")
print("=" * 80)

# ============================================================================
# 1. 데이터 로드 및 전처리
# ============================================================================

# 데이터 로드 (경로는 실제 환경에 맞게 수정)
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

saturation_data = pd.read_excel(DATA_DIR / "중구_충전소_분석결과.xlsx")
station_info = pd.read_excel(DATA_DIR / "중구_급속충전소_정보.xlsx")

weekday_travel = pd.read_csv(DATA_DIR / "weekday_filled.csv")
saturday_travel = pd.read_csv(DATA_DIR / "saturday_filled.csv")
holiday_travel = pd.read_csv(DATA_DIR / "holiday_filled.csv")

print(f"\n[데이터 로드 완료]")
print(f"포화 데이터: {saturation_data.shape}")

# 충전소명 매핑
station_mapping = {
    '기업은행 무교지점 앞 가로등형 충전소': '기업은행_무교지점',
    '서소문청사(지상)': '서소문청사_지상',
    '서소문청사(지하주차장)': '서소문청사_지하',
    '서울시 본관청사': '서울시_본관청사',
    '을지로 노상공영주차장(신한은행 앞)': '을지로_신한은행앞',
    '을지로 노상공영주차장(하나은행 앞)': '을지로_하나은행앞',
    '청계3 노상공영주차장(고려상사 앞)': '청계3_고려상사앞',
    '청계3 노상공영주차장(코리아몰 앞)': '청계3_코리아몰앞',
    '청계5 노상공영주차장': '청계5',
    '청계8가 노상공영주차장': '청계8가',
    '훈련원공원 공영주차장': '훈련원공원'
}

saturation_data['station_short'] = saturation_data['충전소명'].map(station_mapping)

# 일요일과 공휴일 통합
saturation_data['요일구분'] = saturation_data['요일구분'].replace({
    '일요일': '일요일/공휴일', 
    '공휴일': '일요일/공휴일'
})

# NaN 제거
saturation_data_clean = saturation_data[saturation_data['station_short'].notna()].copy()

# 시간을 분으로 변환
def time_to_minutes(time_str):
    h, m = map(int, time_str.split(':'))
    return h * 60 + m

saturation_data_clean['start_min'] = saturation_data_clean['시작시각'].apply(time_to_minutes)
saturation_data_clean['end_min'] = saturation_data_clean['종료시각'].apply(time_to_minutes)

print(f"\n정제 후 데이터: {saturation_data_clean.shape}")
print(f"요일구분별 데이터 수:")
print(saturation_data_clean['요일구분'].value_counts())

# 이동시간 행렬을 딕셔너리로 변환
def create_travel_time_dict(df):
    travel_dict = {}
    for _, row in df.iterrows():
        origin = row['origin_name']
        for col in df.columns[1:]:
            travel_dict[(origin, col)] = row[col]
    return travel_dict

travel_time_weekday = create_travel_time_dict(weekday_travel)
travel_time_saturday = create_travel_time_dict(saturday_travel)
travel_time_holiday = create_travel_time_dict(holiday_travel)

print(f"\n이동시간 딕셔너리 생성 완료")

# ============================================================================
# 2. CP-SAT Optimal Solver 클래스 정의
# ============================================================================

class MCS_Optimal_Solver:
    '''
    MCS Routing을 위한 최적해 보장 Solver (OR-Tools CP-SAT)
    TOPTW (Team Orienteering Problem with Time Windows)

    Parameters:
    - day_type: '평일', '토요일', '일요일/공휴일'
    - depot: 거점 충전소
    - operation_hours: 운영시간 (분 단위)
    - time_limit: 최적화 시간 제한 (초)
    - top_n: 상위 N개 노드만 선택 (문제 규모 축소)
    '''

    def __init__(self, day_type, depot='청계3_코리아몰앞', operation_hours=(360, 1320), 
                 time_limit=300, top_n=50):
        self.day_type = day_type
        self.depot = depot
        self.t_start, self.t_end = operation_hours
        self.time_limit = time_limit

        # 요일별 데이터 및 이동시간 선택
        day_data = saturation_data_clean[saturation_data_clean['요일구분'] == day_type].copy()

        # 발생비율 기준 상위 N개 선택
        day_data = day_data.nlargest(top_n, '발생비율_%')

        if day_type == '평일':
            self.travel_time = travel_time_weekday
        elif day_type == '토요일':
            self.travel_time = travel_time_saturday
        else:  # 일요일/공휴일
            self.travel_time = travel_time_holiday

        # 수요 노드 생성
        self.demands = []
        for idx, row in day_data.iterrows():
            self.demands.append({
                'id': len(self.demands),
                'original_id': idx,
                'station': row['station_short'],
                'time_slot': row['시간대'],
                'start_time': int(row['start_min']),
                'end_time': int(row['end_min']),
                'duration': int(row['지속시간_분']),
                'profit': int(row['발생비율_%'] * 10),  # 정수로 변환 (x10)
                'occurrences': row['발생횟수']
            })

        self.N = len(self.demands)

        print(f"\n{'='*80}")
        print(f"CP-SAT Optimal Solver 초기화 - {day_type}")
        print(f"{'='*80}")
        print(f"거점(Depot): {depot}")
        print(f"운영시간: {self.t_start//60:02d}:00 ~ {self.t_end//60:02d}:00")
        print(f"수요 노드 수: {self.N} (상위 {top_n}개)")
        print(f"총 발생비율 합: {sum(d['profit']/10 for d in self.demands):.1f}%")
        print(f"최적화 시간 제한: {time_limit}초")

    def get_travel_time(self, from_station, to_station):
        '''두 충전소 간 이동시간 반환'''
        if from_station == to_station:
            return 0
        return int(self.travel_time.get((from_station, to_station), 0))

    def solve(self, num_mcs):
        '''CP-SAT 모델로 최적 경로 찾기'''

        print(f"\n{'─'*80}")
        print(f"MCS {num_mcs}대 최적화 시작 (OR-Tools CP-SAT)")
        print(f"{'─'*80}")

        # 노드 인덱스
        nodes = list(range(self.N + 2))
        demand_nodes = list(range(1, self.N + 1))
        depot_start = 0
        depot_end = self.N + 1

        # MCS 인덱스
        vehicles = list(range(num_mcs))

        # CP-SAT 모델 생성
        model = cp_model.CpModel()

        # 결정변수
        # x[i,j,k]: MCS k가 노드 i에서 j로 이동
        x = {}
        for i in nodes:
            for j in nodes:
                if i != j:
                    for k in vehicles:
                        x[i, j, k] = model.NewBoolVar(f"x_{i}_{j}_{k}")

        # tau[i,k]: MCS k가 노드 i에 도착하는 시각 (분)
        tau = {}
        for i in nodes:
            for k in vehicles:
                tau[i, k] = model.NewIntVar(0, self.t_end, f"tau_{i}_{k}")

        # y[i]: 노드 i 방문 여부
        y = {}
        for i in demand_nodes:
            y[i] = model.NewBoolVar(f"y_{i}")

        # 목적함수: 발생비율 최대화
        total_profit = []
        for i in demand_nodes:
            total_profit.append(self.demands[i-1]['profit'] * y[i])

        model.Maximize(sum(total_profit))

        # 제약조건

        # (1) 노드 방문 여부와 경로 연결
        for i in demand_nodes:
            arcs_out = [x[i, j, k] for j in nodes if j != i for k in vehicles]
            model.Add(sum(arcs_out) == y[i])

        # (2) Flow conservation
        for i in demand_nodes:
            for k in vehicles:
                arcs_in = [x[j, i, k] for j in nodes if j != i]
                arcs_out = [x[i, j, k] for j in nodes if j != i]
                model.Add(sum(arcs_in) == sum(arcs_out))

        # (3) 각 MCS는 depot_start에서 정확히 1회 출발
        for k in vehicles:
            model.Add(sum(x[depot_start, j, k] for j in demand_nodes) == 1)

        # (4) 각 MCS는 depot_end로 정확히 1회 복귀
        for k in vehicles:
            model.Add(sum(x[i, depot_end, k] for i in demand_nodes) == 1)

        # (5) 수요 노드 도착 시간은 포화 시작 전
        for i in demand_nodes:
            for k in vehicles:
                for j in nodes:
                    if i != j:
                        model.Add(tau[i, k] <= self.demands[i-1]['start_time']).OnlyEnforceIf(x[i, j, k])

        # (6) 시간 전파 제약
        for i in nodes:
            for j in demand_nodes:
                if i != j:
                    for k in vehicles:
                        # 이동시간
                        if i == depot_start or i == depot_end:
                            origin_station = self.depot
                        else:
                            origin_station = self.demands[i-1]['station']

                        dest_station = self.demands[j-1]['station']
                        t_ij = self.get_travel_time(origin_station, dest_station)

                        # 서비스 시간
                        if i in demand_nodes:
                            service_time = self.demands[i-1]['duration']
                        else:
                            service_time = 0

                        # x[i,j,k] = 1이면 tau[j,k] >= tau[i,k] + service_time + t_ij
                        model.Add(tau[j, k] >= tau[i, k] + service_time + t_ij).OnlyEnforceIf(x[i, j, k])

        # (7) 운영시간 시작
        for k in vehicles:
            model.Add(tau[depot_start, k] >= self.t_start)

        # (8) 운영시간 종료
        for k in vehicles:
            model.Add(tau[depot_end, k] <= self.t_end)

        # 최적화 실행
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit
        solver.parameters.log_search_progress = False
        solver.parameters.num_search_workers = 8  # 병렬 처리

        print("OR-Tools CP-SAT 최적화 실행 중...")
        status = solver.Solve(model)

        # 결과 추출
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            if status == cp_model.OPTIMAL:
                print(f"✓ 최적해 발견!")
            else:
                print(f"✓ 실행 가능해 발견 (시간 제한)")

            obj_value = solver.ObjectiveValue() / 10  # 원래 스케일로 복원
            print(f"목적함수 값: {obj_value:.1f}%")
            print(f"해 시간: {solver.WallTime():.2f}초")

            # 경로 추출
            routes = []

            for k in vehicles:
                route = []
                current = depot_start
                visited = set([depot_start])

                while current != depot_end and len(visited) < self.N + 2:
                    found = False
                    for j in nodes:
                        if j != current and (current, j, k) in x:
                            if solver.Value(x[current, j, k]) == 1:
                                if j != depot_end:
                                    route.append({
                                        'node_id': j - 1,
                                        'station': self.demands[j-1]['station'],
                                        'time_slot': self.demands[j-1]['time_slot'],
                                        'arrival': solver.Value(tau[j, k]),
                                        'start_window': self.demands[j-1]['start_time'],
                                        'end_window': self.demands[j-1]['end_time'],
                                        'duration': self.demands[j-1]['duration'],
                                        'profit': self.demands[j-1]['profit'] / 10,
                                        'occurrences': self.demands[j-1]['occurrences']
                                    })
                                current = j
                                visited.add(j)
                                found = True
                                break

                    if not found:
                        break

                if route:
                    total_profit = sum(stop['profit'] for stop in route)
                    routes.append({
                        'mcs_id': k + 1,
                        'route': route,
                        'profit': total_profit,
                        'num_visits': len(route),
                        'final_time': solver.Value(tau[depot_end, k])
                    })
                    print(f"MCS {k+1}: {len(route)}개 노드, profit {total_profit:.1f}%")

            return {
                'status': 'optimal' if status == cp_model.OPTIMAL else 'feasible',
                'objective': obj_value,
                'routes': routes,
                'total_visits': sum(r['num_visits'] for r in routes),
                'day_type': self.day_type,
                'num_mcs': num_mcs,
                'solve_time': solver.WallTime()
            }
        else:
            print(f"✗ 최적화 실패 (Status: {status})")
            return None

# ============================================================================
# 3. 결과 출력 함수
# ============================================================================

def print_route_details(result):
    '''경로 상세 정보 출력'''
    print(f"\n{'='*80}")
    print(f"경로 상세 정보 - {result['day_type']}, MCS {result['num_mcs']}대")
    print(f"{'='*80}")
    print(f"상태: {result['status']}")
    print(f"총 커버 발생비율: {result['objective']:.2f}%")
    print(f"총 방문 노드 수: {result['total_visits']}")
    print(f"해 시간: {result['solve_time']:.2f}초")

    for mcs_result in result['routes']:
        mcs_id = mcs_result['mcs_id']
        route = mcs_result['route']
        final_time = int(mcs_result['final_time'])

        print(f"\n[MCS {mcs_id}]")
        print(f"방문 노드 수: {mcs_result['num_visits']}")
        print(f"총 Profit: {mcs_result['profit']:.1f}%")
        print(f"복귀 시각: {final_time//60:02d}:{final_time%60:02d}")
        print(f"\n순서  충전소명          시간대  도착시각  시간창(시작-종료)  지속시간  발생비율")
        print("-" * 80)

        for i, stop in enumerate(route, 1):
            arrival = int(stop['arrival'])
            start_window = int(stop['start_window'])
            end_window = int(stop['end_window'])

            arrival_str = f"{arrival//60:02d}:{arrival%60:02d}"
            window_str = f"{start_window//60:02d}:{start_window%60:02d}-{end_window//60:02d}:{end_window%60:02d}"

            print(f"{i:3d}   {stop['station']:15s} {stop['time_slot']:4s}  {arrival_str:5s}  "
                  f"{window_str:13s}  {stop['duration']:4.0f}분  {stop['profit']:6.1f}%")

# ============================================================================
# 4. 메인 실행
# ============================================================================

if __name__ == "__main__":
    # 요일별 최적화
    day_types = ['평일', '토요일', '일요일/공휴일']
    all_results = {}

    for day_type in day_types:
        print(f"\n{'='*80}")
        print(f"{day_type} MCS Routing 최적화")
        print(f"{'='*80}")

        # top_n을 적절히 조정 (평일은 더 많은 노드, 주말은 적게)
        if day_type == '평일':
            solver = MCS_Optimal_Solver(day_type, time_limit=300, top_n=50)
        else:
            solver = MCS_Optimal_Solver(day_type, time_limit=180, top_n=40)

        day_results = {}

        for num_mcs in [1, 2, 3]:
            result = solver.solve(num_mcs=num_mcs)

            if result:
                day_results[num_mcs] = result
                print_route_details(result)
            else:
                print(f"\n✗ MCS {num_mcs}대 실패")
                break

        all_results[day_type] = day_results

    # 결과 출력
    print(f"\n{'='*80}")
    print("최적화 완료 - 전체 결과 요약")
    print(f"{'='*80}")

    for day_type, day_results in all_results.items():
        print(f"\n{day_type}:")
        for num_mcs, result in day_results.items():
            print(f"  MCS {num_mcs}대: 방문 {result['total_visits']}개, 커버 {result['objective']:.1f}%, "
                  f"상태: {result['status']}, 시간: {result['solve_time']:.1f}초")
