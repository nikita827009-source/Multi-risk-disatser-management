import os
import pandas as pd
import numpy as np

def integrate_disaster_data(base_dir=None):
    """
    Integrates spatial zone summaries, social text scores, and resource databases
    into a unified dataset for multimodal risk fusion.
    """
    if base_dir is None:
        # Determine base directory (repo root)
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.abspath(os.path.join(curr_dir, "..", ".."))

    spatial_path = os.path.join(base_dir, "CV_Spatial_Outputs", "02_spatial_zones_summary.csv")
    social_path = os.path.join(base_dir, "CV_Spatial_Outputs", "text_zone_scores.csv")
    resource_path = os.path.join(base_dir, "CV_Spatial_Outputs", "03_disaster_resource_db.csv")
    output_dir = os.path.join(base_dir, "member3_Outputs", "data")
    output_path = os.path.join(output_dir, "unified_disaster_zones.csv")

    os.makedirs(output_dir, exist_ok=True)

    print(f"[Data Integration] Loading datasets from {base_dir}...")
    zones = pd.read_csv(spatial_path)
    social = pd.read_csv(social_path)
    resources = pd.read_csv(resource_path)

    # 1. Merge zones with social text scores
    unified = pd.merge(zones, social, on="zone_id", how="left")
    
    # Fill any missing text score (e.g. Z12) with the mean text severity
    mean_text = unified["text_severity"].dropna().mean()
    unified["text_severity"] = unified["text_severity"].fillna(mean_text)

    # 2. Merge with resource database
    resource_cols = [
        "zone_id", "total_population_Pk", "injured_total_Ik", "injured_critical",
        "injured_serious", "injured_moderate", "displaced_total_Dk",
        "hospital_beds_Bk", "shelter_capacity_Sk", "ambulances_Ak",
        "buses_Uk", "boats_Bok"
    ]
    res_sub = resources[[c for c in resource_cols if c in resources.columns]]
    unified = pd.merge(unified, res_sub, on="zone_id", how="left")

    # 3. Standardize and normalize multimodal columns required for fusion
    unified["S_sat"] = unified["sat_mean_severity"].clip(0.0, 1.0)
    unified["S_soc"] = unified["text_severity"].clip(0.0, 1.0)

    # Confidence derived from observation density delta_k (normalized to [0.50, 0.95])
    min_obs = unified["observation_density_delta_k"].min()
    max_obs = unified["observation_density_delta_k"].max()
    if max_obs > min_obs:
        unified["confidence"] = (0.50 + 0.45 * (unified["observation_density_delta_k"] - min_obs) / (max_obs - min_obs)).round(4)
    else:
        unified["confidence"] = 0.75

    # Urgency derived from social severity and casualty indicators
    max_inj = unified["injured_total_Ik"].max() if "injured_total_Ik" in unified.columns and unified["injured_total_Ik"].max() > 0 else 1.0
    inj_ratio = (unified["injured_total_Ik"] / max_inj) if "injured_total_Ik" in unified.columns else 0.0
    unified["urgency"] = (0.65 * unified["S_soc"] + 0.35 * inj_ratio).clip(0.05, 1.0).round(4)

    # Provide both centroid and standard lat/lon column names for mapping
    unified["latitude"] = unified["centroid_lat"]
    unified["longitude"] = unified["centroid_lon"]

    # Save output
    unified.to_csv(output_path, index=False)
    print(f"[Data Integration] Saved {len(unified)} integrated zone records to {output_path}")
    return unified

if __name__ == "__main__":
    integrate_disaster_data()
