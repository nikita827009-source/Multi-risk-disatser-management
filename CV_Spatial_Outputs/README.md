# Satellite Perception, Spatial Zoning & Resource Grounding Outputs

All files in this folder were generated from the real **xView2 / xBD** post-disaster building polygon annotations and calibrated using the paper's WorldPop & FEMA/WHO logistics formulas[cite: 5, 11, 12].

## Files Included
1. **01_satellite_severity.csv** (For Multimodal Fusion)
   - Columns: image_id, disaster_event, disaster_type, latitude, longitude, total_buildings, damage counts, and normalized satellite damage severity S_sat in [0, 1][cite: 6, 7, 18].
2. **02_image_zone_mapping.csv** (For Multimodal Fusion)
   - Maps every satellite image/coordinate to its K-Means spatial cluster (zone_id: Z0 to Z28)[cite: 10, 19].
3. **02_spatial_zones_summary.csv** (For GCN Spatial Graph)
   - Contains zone centroids (centroid_lat, centroid_lon), observation density (delta_k), and baseline satellite severity for building the Haversine adjacency graph[cite: 10, 11].
4. **aggregate_fused_zones.py** (Helper Script for Fusion -> GCN)
   - Call aggregate_zone_risk("your_fused_points.csv") to compute Mean Fused Risk (mu_k) and Peak Fused Risk (rho_k) per zone[cite: 10].
5. **03_disaster_resource_db.csv** & **03_disaster_resource_db.json** (For RL Environment)
   - Ready-to-use zone-level population exposure (P_k), injured/displaced counts (I_k, D_k), hospital beds (B_k), shelter capacities (S_k), and vehicle counts (ambulances, buses, boats) for the Gymnasium RL environment[cite: 11, 12, 42].
