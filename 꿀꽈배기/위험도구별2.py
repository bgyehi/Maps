import csv
import time
import os
import shutil
from datetime import datetime

# ---------------------------------------------
# 설정
# ---------------------------------------------
ESP32_CSV = "esp32_data.csv"        # 센서가 계속 쓰는 파일
TEMP_COPY = "temp_copy.csv"         # 복사본 (읽기 전용)
DASHBOARD_CSV = "dashboard_data.csv"  # 결과 저장
CHECK_INTERVAL = 1.0  # 초 단위 갱신 간격

WEIGHTS = {'temperature': 0.45, 'wbgt': 0.30, 'heartrate': 0.20, 'spo2': 0.05}
THRESHOLDS = {
    'temperature': {'warning': 38.0, 'danger': 40.0},
    'wbgt': {'warning': 28.0, 'danger': 31.0},
    'heartrate': {'warning': 100, 'danger': 120},
    'spo2': {'warning': 94, 'danger': 90}
}


def calculate_risk(temp, wbgt, hr, spo2):
    score = 0.0
    def add_score(value, key):
        nonlocal score
        w = WEIGHTS[key]
        th = THRESHOLDS[key]
        if key == 'spo2':
            if value <= th['danger']:
                score += w * 1.0
            elif value <= th['warning']:
                score += w * 0.6
        else:
            if value >= th['danger']:
                score += w * 1.0
            elif value >= th['warning']:
                score += w * 0.6
    add_score(temp, 'temperature')
    add_score(wbgt, 'wbgt')
    add_score(hr, 'heartrate')
    add_score(spo2, 'spo2')

    if score < 0.25:
        status = "✅ 안전"
    elif score < 0.5:
        status = "⚠️ 주의"
    elif score < 0.75:
        status = "🚨 위험"
    else:
        status = "⛔ 휴식 필요"

    return round(score, 2), status


def ensure_files():
    if not os.path.exists(ESP32_CSV):
        with open(ESP32_CSV, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'temperature', 'wbgt', 'heartrate', 'spo2'])
    if not os.path.exists(DASHBOARD_CSV):
        with open(DASHBOARD_CSV, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'temperature', 'wbgt', 'heartrate', 'spo2', 'risk_score', 'status'])


def read_last_line(filepath):
    """파일 끝 한 줄만 읽기"""
    with open(filepath, 'rb') as f:
        try:
            f.seek(-300, os.SEEK_END)
        except OSError:
            f.seek(0)
        lines = f.readlines()
    return lines[-1].decode('utf-8').strip()


def main():
    print("\n⚡ HEAT GUARD 초고속 위험도 분석 시작 ⚡")
    ensure_files()
    last_line_cache = ""

    while True:
        try:
            # 잠금 회피용 복사 (읽기 전용 복제)
            shutil.copyfile(ESP32_CSV, TEMP_COPY)

            last_line = read_last_line(TEMP_COPY)
            if not last_line or "temperature" in last_line:
                time.sleep(CHECK_INTERVAL)
                continue

            if last_line == last_line_cache:
                time.sleep(CHECK_INTERVAL)
                continue
            last_line_cache = last_line

            parts = last_line.split(',')
            if len(parts) < 5:
                time.sleep(CHECK_INTERVAL)
                continue

            ts, temp, wbgt, hr, spo2 = parts
            temp, wbgt, hr, spo2 = float(temp), float(wbgt), float(hr), float(spo2)

            score, status = calculate_risk(temp, wbgt, hr, spo2)

            print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                  f"체온:{temp:.1f}°C | WBGT:{wbgt:.1f} | HR:{hr:.0f}bpm | SpO₂:{spo2:.1f}% "
                  f"=> 위험점수:{score:.2f} → {status}")

            # 대시보드용 파일에 결과 추가
            with open(DASHBOARD_CSV, 'a', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                 temp, wbgt, hr, spo2, score, status])

            time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            print("\n프로그램 종료됨.")
            break
        except Exception as e:
            print(f"⚠️ 오류 발생: {e}")
            time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
