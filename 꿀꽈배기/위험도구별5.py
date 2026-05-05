import pandas as pd
import os
import time
from datetime import datetime

# ---------------------------------------------
# ⚙️ 설정 영역
# ---------------------------------------------
INPUT_CSV = "esp32_data.csv"       # ESP32 센서로 측정한 결과
OUTPUT_CSV = "heat_risk_output.csv"  # 대시보드로 넘길 결과 파일
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
# ⚡ 위험도 계산 함수
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

    # SpO₂ (낮을수록 위험)
    if spo2 <= THRESHOLDS['spo2']['danger']:
        score += WEIGHTS['spo2'] * 1.0
    elif spo2 <= THRESHOLDS['spo2']['warning']:
        score += WEIGHTS['spo2'] * 0.6

    # 위험도 등급 (1~5 단계)
    if score < 0.15:
        status = 1  # 매우 안전
    elif score < 0.35:
        status = 2  # 안전
    elif score < 0.55:
        status = 3  # 주의
    elif score < 0.75:
        status = 4  # 위험
    else:
        status = 5  # 휴식 필요

    return round(score, 2), status


# ---------------------------------------------
# 🧾 CSV 파일 확인
# ---------------------------------------------
def ensure_input_csv_exists():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"{INPUT_CSV} 파일이 없습니다. ESP32 측에서 전달받은 파일을 프로젝트 폴더에 넣어주세요.")


# ---------------------------------------------
# 🧠 메인 루프: 실시간 분석 + 결과 저장
# ---------------------------------------------
def main():
    print("\n==============================")
    print("HEAT GUARD 위험도 분석 + 대시보드용 CSV 생성 시스템 (수치 등급 버전)")
    print("==============================\n")

    ensure_input_csv_exists()

    while True:
        try:
            df = pd.read_csv(INPUT_CSV)

            if len(df) == 0:
                print("⚠️ CSV 파일이 비어 있습니다. 센서 데이터가 기록되면 자동 분석됩니다.")
                time.sleep(UPDATE_INTERVAL)
                continue

            required_cols = {'timestamp', 'temperature', 'wbgt', 'heartrate', 'spo2'}
            if not required_cols.issubset(df.columns):
                raise ValueError(f"입력 CSV에 필요한 컬럼이 없습니다: {required_cols}")

            # 최신 데이터
            latest = df.iloc[-1]
            temp = float(latest['temperature'])
            wbgt = float(latest['wbgt'])
            hr = float(latest['heartrate'])
            spo2 = float(latest['spo2'])

            score, status = calculate_risk(temp, wbgt, hr, spo2)

            # 콘솔 출력
            print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                  f"체온:{temp:.1f}°C | WBGT:{wbgt:.1f} | HR:{hr:.0f}bpm | SpO₂:{spo2:.1f}% "
                  f"=> 위험점수:{score:.2f} → 위험등급:{status}")

            # CSV 파일 업데이트
            df['risk_score'] = df.apply(lambda row: calculate_risk(
                row['temperature'], row['wbgt'], row['heartrate'], row['spo2'])[0], axis=1)
            df['risk_status'] = df.apply(lambda row: calculate_risk(
                row['temperature'], row['wbgt'], row['heartrate'], row['spo2'])[1], axis=1)

            df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

            print(f"✅ 분석 완료 → '{OUTPUT_CSV}' 저장됨 (대시보드 전달용)\n")

            time.sleep(UPDATE_INTERVAL)

        except KeyboardInterrupt:
            print("\n프로그램 종료됨.")
            break
        except Exception as e:
            print(f"⚠️ 오류 발생: {e}")
            time.sleep(UPDATE_INTERVAL)


if __name__ == "__main__":
    main()
