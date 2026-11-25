import os
import pandas as pd
import numpy as np

print("--- Step 1: Checking Spatial Zones File ---")
spatial_csv = "02_spatial_zones_summary.csv"

if os.path.exists(spatial_csv):
    zones_df = pd.read_csv(spatial_csv)
    print("Success! File mil gayi. Total zones:", len(zones_df))
    print(zones_df.head(3)) # Pehle 3 zones dikhayega
else:
    print("Error: 02_spatial_zones_summary.csv nahi mili. Kripya check karein ki file usi folder mein hai.")

print("\n--- Step 2: Generating Text Severity Scores (Sample) ---")
# Hum yahan maan kar chal rahe hain ki disaster tweets/text ka data hai
np.random.seed(42)
n_posts = 100

text_data = pd.DataFrame({
    'post_id': [f"post_{i}" for i in range(n_posts)],
    'zone_id': np.random.choice(zones_df['zone_id'] if 'zones_df' in locals() else [f"Z{i}" for i in range(30)], n_posts),
    'urgency_score': np.random.uniform(0.1, 1.0, n_posts),
    'casualty_indicator': np.random.choice([0, 1], n_posts, p=[0.7, 0.3]),
    'infrastructure_damage': np.random.uniform(0.0, 1.0, n_posts)
})

# Text Severity Score calculate karna
text_data['text_severity'] = (
    0.4 * text_data['urgency_score'] + 
    0.4 * text_data['casualty_indicator'] + 
    0.2 * text_data['infrastructure_damage']
)

# Zone ke hisab se average nikalna
text_zone_scores = text_data.groupby('zone_id')['text_severity'].mean().reset_index()
text_zone_scores.to_csv("text_zone_scores.csv", index=False)
print("Saved: text_zone_scores.csv successfully ban gayi hai!")

print("\n--- Step 3: Multi-Modal Risk Fusion ---")
if 'zones_df' in locals():
    # Satellite data aur Text data ko merge karna
    merged_df = pd.merge(zones_df, text_zone_scores, on='zone_id', how='left')
    merged_df['text_severity'] = merged_df['text_severity'].fillna(merged_df['text_severity'].mean())
    
    # Formula: R_k = alpha * Sat + (1 - alpha) * Text
    alpha = 0.5
    merged_df['fused_risk_score'] = (
        alpha * merged_df['sat_mean_severity'] + 
        (1 - alpha) * merged_df['text_severity']
    )
    
    # Rank karke Top 20 zones nikalna
    ranked_zones = merged_df.sort_values(by='fused_risk_score', ascending=False).reset_index(drop=True)
    top_20 = ranked_zones.head(20)
    
    top_20.to_csv("top_20_critical_zones.csv", index=False)
    print("Success! Top 20 critical zones 'top_20_critical_zones.csv' file mein save ho gaye hain.")
    print(top_20[['zone_id', 'dominant_disaster_type', 'fused_risk_score']].head())
    import matplotlib.pyplot as plt

# Top 10 zones ka comparison chart banane ke liye
plot_df = top_20.head(10).sort_values(by='fused_risk_score', ascending=True)
plt.figure(figsize=(10, 5))
plt.barh(plot_df['zone_id'], plot_df['fused_risk_score'], color='red')
plt.xlabel('Fused Multi-Modal Risk Score')
plt.ylabel('Zone ID')
plt.title('Top Critical Disaster Zones (Fused Risk)')
plt.tight_layout()
plt.savefig('top_zones_chart.png')
print("Chart saved successfully as top_zones_chart.png!")