import os
import sys
from pathlib import Path
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fusion_engine import calculate_fused_risk, run_fusion_pipeline

def validate_fusion_weights(base_dir=None):
    if base_dir is None:
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.abspath(os.path.join(curr_dir, "..", ".."))

    final_path = os.path.join(base_dir, "member3_Outputs", "data", "final_zone_risk.csv")
    if not os.path.exists(final_path):
        run_fusion_pipeline(base_dir)

    df = pd.read_csv(final_path)

    # Reference risk: baseline satellite mean severity or Member 2 fused risk score
    ref_col = "sat_mean_severity"
    if "fused_risk_score" in df.columns:
        ref_col = "fused_risk_score"

    experiments = [
        (0.65, 0.25, 0.10),
        (0.55, 0.30, 0.15),
        (0.70, 0.20, 0.10),
        (0.60, 0.30, 0.10),
        (0.50, 0.35, 0.15)
    ]

    records = []
    print("\n--- Weight Sensitivity Analysis ---")
    for w_sat, w_soc, w_urg in experiments:
        test_df = calculate_fused_risk(df, w_sat, w_soc, w_urg)
        corr, p_val = spearmanr(test_df["fused_risk"], test_df[ref_col])
        mean_risk = test_df["fused_risk"].mean()
        max_risk = test_df["fused_risk"].max()
        print(f"Weights (Sat={w_sat:.2f}, Soc={w_soc:.2f}, Urg={w_urg:.2f}) -> Spearman r: {corr:.4f} (p={p_val:.4e})")

        records.append({
            "w_satellite": w_sat,
            "w_social": w_soc,
            "w_urgency": w_urg,
            "spearman_correlation": round(corr, 4),
            "p_value": round(p_val, 6),
            "mean_fused_risk": round(mean_risk, 4),
            "max_fused_risk": round(max_risk, 4)
        })

    results_df = pd.DataFrame(records)
    out_path = os.path.join(base_dir, "outputs", "weight_sensitivity.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    results_df.to_csv(out_path, index=False)
    print(f"\n[Validation] Saved weight sensitivity analysis to {out_path}")
    return results_df

if __name__ == "__main__":
    validate_fusion_weights()
