
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
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
print("MCS ROUTING 최적화 시스템 - Greedy Heuristic")
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
print(f"충전소 정보: {station_info.shape}")

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
# 2. Greedy Heuristic Solver 클래스 정의
# ============================================================================

class MCS_Greedy_Solver:
    '''
    MCS Routing을 위한 Greedy Heuristic Solver
    TOPTW (Team Orienteering Problem with Time Windows)

    Parameters:
    - day_type: '평일', '토요일', '일요일/공휴일'
    - depot: 거점 충전소 (default: '청계3_코리아몰앞')
    - operation_hours: 운영시간 (분 단위, default: 06:00-22:00)
    '''

    def __init__(self, day_type, depot='청계3_코리아몰앞', operation_hours=(360, 1320)):
        self.day_type = day_type
        self.depot = depot
        self.t_start, self.t_end = operation_hours

        # 요일별 데이터 및 이동시간 선택
        self.data = saturation_data_clean[saturation_data_clean['요일구분'] == day_type].copy()

        if day_type == '평일':
            self.travel_time = travel_time_weekday
        elif day_type == '토요일':
            self.travel_time = travel_time_saturday
        else:  # 일요일/공휴일
            self.travel_time = travel_time_holiday

        # 수요 노드 생성
        self.demands = []
        for idx, row in self.data.iterrows():
            self.demands.append({
                'id': idx,
                'station': row['station_short'],
                'time_slot': row['시간대'],
                'start_time': row['start_min'],
                'end_time': row['end_min'],
                'duration': row['지속시간_분'],
                'profit': row['발생비율_%'],
                'occurrences': row['발생횟수']
            })

        print(f"\n{'='*80}")
        print(f"Greedy Solver 초기화 - {day_type}")
        print(f"{'='*80}")
        print(f"거점(Depot): {depot}")
        print(f"운영시간: {self.t_start//60:02d}:00 ~ {self.t_end//60:02d}:00")
        print(f"수요 노드 수: {len(self.demands)}")
        print(f"총 발생비율 합: {sum(d['profit'] for d in self.demands):.1f}%")

    def get_travel_time(self, from_station, to_station):
        '''두 충전소 간 이동시간 반환'''
        if from_station == to_station:
            return 0
        return self.travel_time.get((from_station, to_station), 0)

    def can_visit(self, current_time, current_station, demand, route_visited):
        '''해당 수요 노드를 방문 가능한지 확인'''
        if demand['id'] in route_visited:
            return False

        travel_time = self.get_travel_time(current_station, demand['station'])
        arrival_time = current_time + travel_time

        # 포화 시작 전에 도착해야 함
        if arrival_time > demand['start_time']:
            return False

        # 서비스 완료 후 depot 복귀 가능 여부
        service_end = arrival_time + demand['duration']
        return_time = self.get_travel_time(demand['station'], self.depot)

        if service_end + return_time > self.t_end:
            return False

        return True

    def calculate_priority(self, current_time, current_station, demand):
        '''노드 방문 우선순위 계산'''
        travel_time = self.get_travel_time(current_station, demand['station'])
        arrival_time = current_time + travel_time

        # Profit-to-time ratio
        total_time = travel_time + demand['duration']
        if total_time == 0:
            profit_ratio = demand['profit']
        else:
            profit_ratio = demand['profit'] / total_time

        # Time window urgency
        time_slack = demand['start_time'] - arrival_time
        urgency = 1 / (time_slack + 1)

        # 가중치 조합
        priority = profit_ratio * 0.7 + urgency * demand['profit'] * 0.3

        return priority

    def solve_single_vehicle(self, initial_demands=None):
        '''단일 MCS 경로 최적화'''
        if initial_demands is None:
            available_demands = self.demands.copy()
        else:
            available_demands = initial_demands

        route = []
        current_time = self.t_start
        current_station = self.depot
        visited_ids = set()
        total_profit = 0

        while True:
            # 방문 가능한 노드 찾기
            feasible = []
            for demand in available_demands:
                if self.can_visit(current_time, current_station, demand, visited_ids):
                    priority = self.calculate_priority(current_time, current_station, demand)
                    feasible.append((priority, demand))

            if not feasible:
                break

            # 우선순위가 가장 높은 노드 선택
            feasible.sort(key=lambda x: x[0], reverse=True)
            _, next_demand = feasible[0]

            # 이동 및 서비스
            travel_time = self.get_travel_time(current_station, next_demand['station'])
            arrival_time = current_time + travel_time

            route.append({
                'station': next_demand['station'],
                'time_slot': next_demand['time_slot'],
                'arrival': arrival_time,
                'start_window': next_demand['start_time'],
                'end_window': next_demand['end_time'],
                'duration': next_demand['duration'],
                'profit': next_demand['profit'],
                'occurrences': next_demand['occurrences'],
                'travel_time': travel_time
            })

            # 상태 업데이트
            visited_ids.add(next_demand['id'])
            current_time = arrival_time + next_demand['duration']
            current_station = next_demand['station']
            total_profit += next_demand['profit']

        # Depot 복귀
        return_time = self.get_travel_time(current_station, self.depot)
        final_time = current_time + return_time

        return {
            'route': route,
            'total_profit': total_profit,
            'num_visits': len(route),
            'final_time': final_time,
            'visited_ids': visited_ids
        }

    def solve_multi_vehicle(self, num_mcs):
        '''다중 MCS 경로 최적화'''
        all_routes = []
        total_profit = 0
        all_visited = set()

        for k in range(num_mcs):
            # 아직 방문하지 않은 노드만 대상
            available = [d for d in self.demands if d['id'] not in all_visited]

            if not available:
                print(f"MCS {k+1}: 더 이상 방문할 노드가 없습니다.")
                break

            result = self.solve_single_vehicle(available)

            if result['num_visits'] > 0:
                all_routes.append({
                    'mcs_id': k + 1,
                    'route': result['route'],
                    'profit': result['total_profit'],
                    'num_visits': result['num_visits'],
                    'final_time': result['final_time']
                })

                total_profit += result['total_profit']
                all_visited.update(result['visited_ids'])

                print(f"MCS {k+1}: {result['num_visits']}개 노드 방문, profit {result['total_profit']:.1f}%")
            else:
                print(f"MCS {k+1}: 방문 가능한 노드 없음")

        return {
            'routes': all_routes,
            'total_profit': total_profit,
            'total_visits': len(all_visited),
            'day_type': self.day_type,
            'num_mcs': num_mcs
        }

# ============================================================================
# 3. 결과 출력 함수
# ============================================================================

def print_route_details(result):
    '''경로 상세 정보 출력'''
    print(f"\n{'='*80}")
    print(f"경로 상세 정보 - {result['day_type']}, MCS {result['num_mcs']}대")
    print(f"{'='*80}")
    print(f"총 커버 발생비율: {result['total_profit']:.2f}%")
    print(f"총 방문 노드 수: {result['total_visits']}")

    for mcs_result in result['routes']:
        mcs_id = mcs_result['mcs_id']
        route = mcs_result['route']
        final_time = int(mcs_result['final_time'])

        print(f"\n[MCS {mcs_id}]")
        print(f"방문 노드 수: {mcs_result['num_visits']}")
        print(f"총 Profit: {mcs_result['profit']:.1f}%")
        print(f"복귀 시각: {final_time//60:02d}:{final_time%60:02d}")
        print(f"\n순서  충전소명          시간대  도착시각  시간창(시작-종료)  지속시간  발생비율  이동시간")
        print("-" * 90)

        for i, stop in enumerate(route, 1):
            arrival = int(stop['arrival'])
            start_window = int(stop['start_window'])
            end_window = int(stop['end_window'])

            arrival_str = f"{arrival//60:02d}:{arrival%60:02d}"
            window_str = f"{start_window//60:02d}:{start_window%60:02d}-{end_window//60:02d}:{end_window%60:02d}"

            print(f"{i:3d}   {stop['station']:15s} {stop['time_slot']:4s}  {arrival_str:5s}  "
                  f"{window_str:13s}  {stop['duration']:4.0f}분  {stop['profit']:6.1f}%  {stop['travel_time']:5.1f}분")

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

        solver = MCS_Greedy_Solver(day_type)
        day_results = {}

        for num_mcs in [1, 2, 3, 4, 5]:
            print(f"\n{'─'*80}")
            print(f"MCS {num_mcs}대 최적화")
            print(f"{'─'*80}")

            result = solver.solve_multi_vehicle(num_mcs=num_mcs)
            day_results[num_mcs] = result

            print(f"\n총 커버 발생비율: {result['total_profit']:.2f}%")
            print(f"총 방문 노드 수: {result['total_visits']}")

        all_results[day_type] = day_results

    # 결과 출력
    print(f"\n{'='*80}")
    print("최적화 완료 - 전체 결과 요약")
    print(f"{'='*80}")

    for day_type, day_results in all_results.items():
        print(f"\n{day_type}:")
        for num_mcs, result in day_results.items():
            print(f"  MCS {num_mcs}대: 방문 {result['total_visits']}개, 커버 {result['total_profit']:.1f}%")
