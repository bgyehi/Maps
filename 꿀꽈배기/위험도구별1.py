import csv
import time
import os
from datetime import datetime

ESP32_CSV = "esp32_data.csv"
DASHBOARD_CSV = "dashboard_data.csv"
UPDATE_INTERVAL = 0.5  # 0.5초마다 체크 (더 빠르게 가능)

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
        if key == 'spo2':  # 낮을수록 위험
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


def ensure_files_exist():
    if not os.path.exists(ESP32_CSV):
        with open(ESP32_CSV, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'temperature', 'wbgt', 'heartrate', 'spo2'])
    if not os.path.exists(DASHBOARD_CSV):
        with open(DASHBOARD_CSV, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'temperature', 'wbgt', 'heartrate', 'spo2', 'risk_score', 'status'])


def get_last_line(file_path):
    """파일 끝 한 줄만 읽기 (매우 빠름)"""
    with open(file_path, 'rb') as f:
        f.seek(-200, os.SEEK_END)  # 끝에서 200바이트만 읽기
        last = f.readlines()[-1].decode('utf-8').strip()
    return last


def main():
    print("\n🚀 HEAT GUARD 고속 실시간 분석 시작")
    ensure_files_exist()
    last_size = 0

    while True:
        try:
            new_size = os.path.getsize(ESP32_CSV)
            if new_size == last_size:
                time.sleep(UPDATE_INTERVAL)
                continue
            last_size = new_size

            last_line = get_last_line(ESP32_CSV)
            if not last_line or "temperature" in last_line:
                continue

            parts = last_line.split(',')
            if len(parts) < 5:
                continue

            temp = float(parts[1])
            wbgt = float(parts[2])
            hr = float(parts[3])
            spo2 = float(parts[4])

            score, status = calculate_risk(temp, wbgt, hr, spo2)
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{now}] 체온:{temp:.1f}°C | WBGT:{wbgt:.1f} | HR:{hr:.0f}bpm | SpO₂:{spo2:.1f}% "
                  f"=> 위험점수:{score:.2f} → {status}")

            with open(DASHBOARD_CSV, 'a', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow([now, temp, wbgt, hr, spo2, score, status])

            time.sleep(UPDATE_INTERVAL)

        except KeyboardInterrupt:
            print("\n프로그램 종료됨.")
            break
        except Exception as e:
            print(f"⚠️ 오류 발생: {e}")
            time.sleep(1)


if __name__ == "__main__":
    main()
