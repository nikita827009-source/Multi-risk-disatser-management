import pandas as pd

def aggregate_zone_risk(fused_points_csv, output_zones_csv="fused_zones_for_gcn.csv"):
    """
    Takes Member 3's point-level fused risk CSV (must contain 'zone_id' and 'R_fused')
    and computes Equation (12) Mean Fused Risk (mu_k) and Equation (13) Peak Fused Risk (rho_k).
    """
    df = pd.read_csv(fused_points_csv)
    zones = df.groupby("zone_id").agg(
        centroid_lat=("latitude", "mean"),
        centroid_lon=("longitude", "mean"),
        observation_density_delta_k=("R_fused", "count"),
        mean_fused_risk_mu_k=("R_fused", "mean"),
        peak_fused_risk_rho_k=("R_fused", "max"),
        sat_mean_severity=("S_sat", "mean"),
    ).reset_index()
    zones.to_csv(output_zones_csv, index=False)
    print(f"Saved aggregated zone features to {output_zones_csv}")
    return zones
