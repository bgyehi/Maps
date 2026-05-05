import pandas as pd
import os
import time
from datetime import datetime

# ---------------------------------------------
# 설정 영역
# ---------------------------------------------
ESP32_CSV = "esp32_data.csv"        # 조원한테 받은 센서 원본 파일
DASHBOARD_CSV = "dashboard_data.csv"  # 대시보드용 결과 파일
UPDATE_INTERVAL = 2  # 초 단위 (갱신 주기)

# 가중치 설정
WEIGHTS = {
    'temperature': 0.45,
    'wbgt': 0.30,
    'heartrate': 0.20,
    'spo2': 0.05
}

# 임계값 설정 (보고서 기준)
THRESHOLDS = {
    'temperature': {'warning': 38.0, 'danger': 40.0},
    'wbgt': {'warning': 28.0, 'danger': 31.0},
    'heartrate': {'warning': 100, 'danger': 120},
    'spo2': {'warning': 94, 'danger': 90}
}


# ---------------------------------------------
# 위험도 계산 함수
# ---------------------------------------------
def calculate_risk(temp, wbgt, hr, spo2):
    score = 0.0

    # 체온
    if temp >= THRESHOLDS['temperature']['danger']:
        score += WEIGHTS['temperature'] * 1.0
    elif temp >= THRESHOLDS['temperature']['warning']:
        score += WEIGHTS['temperature'] * 0.6

    # WBGT
    if wbgt >= THRESHOLDS['wbgt']['danger']:
        score += WEIGHTS['wbgt'] * 1.0
    elif wbgt >= THRESHOLDS['wbgt']['warning']:
        score += WEIGHTS['wbgt'] * 0.6

    # 심박수
    if hr >= THRESHOLDS['heartrate']['danger']:
        score += WEIGHTS['heartrate'] * 1.0
    elif hr >= THRESHOLDS['heartrate']['warning']:
        score += WEIGHTS['heartrate'] * 0.6

    # SpO2 (낮을수록 위험)
    if spo2 <= THRESHOLDS['spo2']['danger']:
        score += WEIGHTS['spo2'] * 1.0
    elif spo2 <= THRESHOLDS['spo2']['warning']:
        score += WEIGHTS['spo2'] * 0.6

    # 위험도 등급
    if score < 0.25:
        status = "✅ 안전"
    elif score < 0.5:
        status = "⚠️ 주의"
    elif score < 0.75:
        status = "🚨 위험"
    else:
        status = "⛔ 휴식 필요"

    return round(score, 2), status


# ---------------------------------------------
# CSV 파일 준비
# ---------------------------------------------
def ensure_files_exist():
    # 원본 파일이 없을 경우 빈 CSV 생성
    if not os.path.exists(ESP32_CSV):
        pd.DataFrame(columns=['timestamp', 'temperature', 'wbgt', 'heartrate', 'spo2']).to_csv(ESP32_CSV, index=False, encoding='utf-8-sig')
        print(f"[생성됨] {ESP32_CSV} (빈 원본 CSV 생성)\n")

    # 대시보드용 파일이 없을 경우 생성
    if not os.path.exists(DASHBOARD_CSV):
        pd.DataFrame(columns=['timestamp', 'temperature', 'wbgt', 'heartrate', 'spo2', 'risk_score', 'status']).to_csv(DASHBOARD_CSV, index=False, encoding='utf-8-sig')
        print(f"[생성됨] {DASHBOARD_CSV} (빈 대시보드 CSV 생성)\n")


# ---------------------------------------------
# 메인 루프
# ---------------------------------------------
def main():
    print("\n==============================")
    print("HEAT GUARD 위험도 실시간 분석 시스템")
    print("==============================\n")

    ensure_files_exist()
    last_size = 0

    while True:
        try:
            # 파일 사이즈가 변했는지 감지
            new_size = os.path.getsize(ESP32_CSV)
            if new_size == last_size:
                time.sleep(UPDATE_INTERVAL)
                continue
            last_size = new_size

            # CSV 읽기
            df = pd.read_csv(ESP32_CSV)

            if len(df) == 0:
                print("⚠️ esp32_data.csv 파일이 비어 있습니다. 데이터가 들어오면 분석 시작.")
                time.sleep(UPDATE_INTERVAL)
                continue

            latest = df.iloc[-1]
            temp = float(latest['temperature'])
            wbgt = float(latest['wbgt'])
            hr = float(latest['heartrate'])
            spo2 = float(latest['spo2'])

            score, status = calculate_risk(temp, wbgt, hr, spo2)

            now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{now_str}] 체온:{temp:.1f}°C | WBGT:{wbgt:.1f} | HR:{hr:.0f}bpm | SpO₂:{spo2:.1f}% "
                  f"=> 위험점수:{score:.2f} → {status}")

            # 대시보드용 CSV에 결과 저장 (누적)
            new_row = pd.DataFrame([{
                'timestamp': now_str,
                'temperature': temp,
                'wbgt': wbgt,
                'heartrate': hr,
                'spo2': spo2,
                'risk_score': score,
                'status': status
            }])
            new_row.to_csv(DASHBOARD_CSV, mode='a', header=False, index=False, encoding='utf-8-sig')

            time.sleep(UPDATE_INTERVAL)

        except KeyboardInterrupt:
            print("\n프로그램 종료됨.")
            break
        except Exception as e:
            print(f"⚠️ 오류 발생: {e}")
            time.sleep(UPDATE_INTERVAL)


if __name__ == "__main__":
    main()
