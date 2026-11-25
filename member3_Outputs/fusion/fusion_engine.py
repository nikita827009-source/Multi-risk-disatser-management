import os
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_integration import integrate_disaster_data

def calculate_fused_risk(df, w_sat=0.65, w_soc=0.25, w_urg=0.10):
    """
    Computes multi-modal fused risk score combining satellite perception,
    social text sentiment, confidence, and urgency.
    """
    df = df.copy()
    df["fused_risk"] = (
        w_sat * df["S_sat"]
        + w_soc * df["confidence"] * df["S_soc"]
        + w_urg * df["confidence"] * df["urgency"]
    ).round(4)
    return df

def classify_risk(score):
    """
    Categorizes numerical risk scores into empirical disaster response tiers:
    - Critical (>= 0.40): Severe structural devastation and casualty exposure (Immediate Priority)
    - High (>= 0.30 to < 0.40): Substantial hazard impact requiring urgent logistical dispatch
    - Moderate (>= 0.18 to < 0.30): Moderate disruption, localized infrastructure damage
    - Low (< 0.18): Minor / baseline event severity
    """
    if score >= 0.40:
        return "Critical"
    elif score >= 0.30:
        return "High"
    elif score >= 0.18:
        return "Moderate"
    else:
        return "Low"

def run_fusion_pipeline(base_dir=None):
    """
    Executes end-to-end data fusion, normalization, and risk categorization.
    """
    if base_dir is None:
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.abspath(os.path.join(curr_dir, "..", ".."))

    unified_path = os.path.join(base_dir, "member3_Outputs", "data", "unified_disaster_zones.csv")
    fused_path = os.path.join(base_dir, "member3_Outputs", "data", "fused_zone_risk.csv")
    final_path = os.path.join(base_dir, "member3_Outputs", "data", "final_zone_risk.csv")

    if not os.path.exists(unified_path):
        integrate_disaster_data(base_dir)

    df = pd.read_csv(unified_path)

    required = ["S_sat", "S_soc", "confidence", "urgency"]
    assert df[required].notna().all().all(), "Missing values found in required fusion columns"
    assert df[required].apply(lambda col: col.between(0, 1).all()).all(), "Required values must be bounded in [0, 1]"

    result = calculate_fused_risk(df)
    result["risk_category"] = result["fused_risk"].apply(classify_risk)

    # Normalized risk index (0 - 100%)
    min_r = result["fused_risk"].min()
    max_r = result["fused_risk"].max()
    result["risk_index_pct"] = (((result["fused_risk"] - min_r) / (max_r - min_r)) * 100).round(1)

    # Sort descending by fused risk so priority zones are first
    result = result.sort_values(by="fused_risk", ascending=False).reset_index(drop=True)

    # Save to both fused_zone_risk and final_zone_risk
    result.to_csv(fused_path, index=False)
    result.to_csv(final_path, index=False)

    print(f"[Fusion Engine] Fusion completed successfully. Processed {len(result)} zones.")
    print(f"[Fusion Engine] Output saved to:\n  - {fused_path}\n  - {final_path}")
    print("\nRisk Category Breakdown:")
    print(result["risk_category"].value_counts().to_string())
    return result

if __name__ == "__main__":
    run_fusion_pipeline()
