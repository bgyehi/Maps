
import pandas as pd
import numpy as np
from pathlib import Path

# =========================
# 설정값
# =========================
INPUT_FILE = "mcs_schedule_raw.xlsx"
OUTPUT_DIR = Path("mcs_method1_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# 커버율 목표 후보
TARGET_COVERAGES = [0.80, 0.90, 0.95]

# 엘보우 판단 기준:
# 차량 1대 추가 시 커버율 증가분이 이 값보다 작아지면 엘보우 후보로 간주
MARGINAL_GAIN_THRESHOLD = 0.05  # 5%p

# K 탐색 상한: 각 구의 최대 필요_MCS수까지 자동 설정
USE_AUTO_KMAX = True
MANUAL_KMAX = 10  # USE_AUTO_KMAX=False 일 때 사용


# =========================
# 데이터 로드/전처리
# =========================
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)

    required_cols = {"구", "date", "hour", "필요_MCS수"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {missing}")

    df = df.copy()
    df["필요_MCS수"] = pd.to_numeric(df["필요_MCS수"], errors="coerce").fillna(0).astype(int)
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce").fillna(0).astype(int)
    df["date"] = df["date"].astype(str)

    # 시간 슬롯 식별자 생성
    df["time_slot"] = df["date"] + "_" + df["hour"].astype(str)

    # 혹시 중복 행이 있다면 합산
    grouped = (
        df.groupby(["구", "time_slot"], as_index=False)["필요_MCS수"]
        .sum()
        .sort_values(["구", "time_slot"])
        .reset_index(drop=True)
    )
    return grouped


# =========================
# 방법1: 구별 독립 스케줄링
# served(K) = min(필요_MCS수, K)
# =========================
def evaluate_district(district_df: pd.DataFrame, k_max: int) -> pd.DataFrame:
    demand = district_df["필요_MCS수"].to_numpy()
    total_demand = demand.sum()

    rows = []
    for k in range(k_max + 1):
        served = np.minimum(demand, k).sum()
        unmet = np.maximum(demand - k, 0).sum()

        coverage = served / total_demand if total_demand > 0 else 0.0
        unmet_ratio = unmet / total_demand if total_demand > 0 else 0.0
        saturation_reduction = coverage  # baseline(K=0) 대비 포화 대응률로 해석

        rows.append({
            "K": k,
            "total_demand": int(total_demand),
            "served_total": int(served),
            "unmet_total": int(unmet),
            "coverage_rate": coverage,
            "unmet_ratio": unmet_ratio,
            "saturation_reduction_rate": saturation_reduction,
        })

    result = pd.DataFrame(rows)
    result["marginal_gain"] = result["coverage_rate"].diff().fillna(result["coverage_rate"])
    return result


def find_target_k(result_df: pd.DataFrame, target: float):
    hit = result_df[result_df["coverage_rate"] >= target]
    if hit.empty:
        return None
    return int(hit.iloc[0]["K"])


def find_elbow_k(result_df: pd.DataFrame, threshold: float = 0.05):
    """
    엘보우 판단:
    - coverage가 90% 이상인 상태에서
    - marginal_gain < threshold 인 가장 작은 K
    우선 적용
    없으면 marginal_gain이 threshold 아래로 처음 내려가는 K 반환
    그것도 없으면 마지막 K 반환
    """
    cond1 = result_df[
        (result_df["coverage_rate"] >= 0.90) &
        (result_df["marginal_gain"] < threshold)
    ]
    if not cond1.empty:
        return int(cond1.iloc[0]["K"])

    cond2 = result_df[result_df["marginal_gain"] < threshold]
    if not cond2.empty:
        return int(cond2.iloc[0]["K"])

    return int(result_df["K"].max())


def summarize_all(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    districts = sorted(df["구"].unique())
    summary_rows = []
    detailed_results = {}

    for district in districts:
        district_df = df[df["구"] == district].copy()

        k_max = int(district_df["필요_MCS수"].max()) if USE_AUTO_KMAX else MANUAL_KMAX
        result_df = evaluate_district(district_df, k_max=k_max)

        elbow_k = find_elbow_k(result_df, threshold=MARGINAL_GAIN_THRESHOLD)
        k80 = find_target_k(result_df, 0.80)
        k90 = find_target_k(result_df, 0.90)
        k95 = find_target_k(result_df, 0.95)

        detailed_results[district] = result_df

        summary_rows.append({
            "구": district,
            "총_필요량합": int(district_df["필요_MCS수"].sum()),
            "최대_시간대_필요량": int(district_df["필요_MCS수"].max()),
            "80%_커버_최소대수": k80,
            "90%_커버_최소대수": k90,
            "95%_커버_최소대수": k95,
            "엘보우_대수": elbow_k,
            "엘보우_커버율": float(result_df.loc[result_df["K"] == elbow_k, "coverage_rate"].iloc[0]),
            "엘보우_미대응비율": float(result_df.loc[result_df["K"] == elbow_k, "unmet_ratio"].iloc[0]),
            "엘보우_포화감소율": float(result_df.loc[result_df["K"] == elbow_k, "saturation_reduction_rate"].iloc[0]),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("구").reset_index(drop=True)
    return summary_df, detailed_results


def build_citywide_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for col in ["80%_커버_최소대수", "90%_커버_최소대수", "95%_커버_최소대수", "엘보우_대수"]:
        total = pd.to_numeric(summary_df[col], errors="coerce").fillna(0).sum()
        rows.append({
            "기준": col,
            "서울시_총_MCS대수": int(total)
        })

    return pd.DataFrame(rows)


def save_results(summary_df: pd.DataFrame, detailed_results: dict, citywide_df: pd.DataFrame):
    summary_path = OUTPUT_DIR / "district_summary.csv"
    citywide_path = OUTPUT_DIR / "citywide_summary.csv"
    excel_path = OUTPUT_DIR / "method1_results.xlsx"

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    citywide_df.to_csv(citywide_path, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="district_summary", index=False)
        citywide_df.to_excel(writer, sheet_name="citywide_summary", index=False)

        for district, result_df in detailed_results.items():
            safe_name = str(district)[:31]
            result_df.to_excel(writer, sheet_name=safe_name, index=False)

    print("결과 저장 완료")
    print(f"- 구별 요약: {summary_path}")
    print(f"- 서울시 요약: {citywide_path}")
    print(f"- 엑셀 결과: {excel_path}")


def main():
    df = load_data(INPUT_FILE)
    summary_df, detailed_results = summarize_all(df)
    citywide_df = build_citywide_summary(summary_df)
    save_results(summary_df, detailed_results, citywide_df)

    print("\n[구별 요약 상위 10개]")
    print(summary_df.head(10).to_string(index=False))

    print("\n[서울시 총량 요약]")
    print(citywide_df.to_string(index=False))


if __name__ == "__main__":
    main()
