import pandas as pd
import os

folder = "CV_Spatial_Outputs"

files = [
    "01_satellite_severity.csv",
    "02_image_zone_mapping.csv",
    "02_spatial_zones_summary.csv",
    "text_zone_scores.csv",
    "top_20_critical_zones.csv",
    "03_disaster_resource_db.csv"
]

for file in files:
    path = os.path.join(folder, file)

    print("\n", "=" * 60)
    print("FILE:", file)

    if os.path.exists(path):
        df = pd.read_csv(path)

        print("Shape:", df.shape)
        print("Columns:", df.columns.tolist())
        print(df.head(3).to_string(index=False))
    else:
        print("FILE NOT FOUND")