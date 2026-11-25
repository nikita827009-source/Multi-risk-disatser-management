import folium
from folium.plugins import HeatMap

RISK_COLORS = {
    "Critical": "#d32f2f",
    "High": "#f57c00",
    "Moderate": "#fbc02d",
    "Low": "#388e3c"
}

EVENT_REGIONS = {
    "🌍 All Zones Overview": {"center": [25.0, -85.0], "zoom": 3},
    "🌀 Hurricane Matthew (Haiti) [Critical]": {"center": [18.40, -73.90], "zoom": 9},
    "🌊 Hurricane Harvey (Texas) [Critical / High]": {"center": [29.75, -95.55], "zoom": 9},
    "🔥 Santa Rosa Wildfire (California) [Critical]": {"center": [38.50, -122.71], "zoom": 10},
    "🌪️ Hurricane Florence (North Carolina) [High]": {"center": [34.40, -78.20], "zoom": 8},
    "🔥 SoCal Fire (Malibu / California) [High]": {"center": [34.10, -118.80], "zoom": 10},
    "🌀 Hurricane Michael (Florida) [High]": {"center": [30.20, -85.65], "zoom": 9},
    "🌊 Palu Tsunami (Indonesia)": {"center": [-0.85, 119.85], "zoom": 10},
    "🌧️ Midwest Flooding (Oklahoma / Arkansas)": {"center": [36.20, -96.30], "zoom": 8},
    "🌋 Guatemala Volcano": {"center": [14.40, -90.84], "zoom": 10}
}

def get_risk_color(category):
    return RISK_COLORS.get(category, "#1976d2")

def create_risk_map(df, center=None, zoom_start=None):
    """
    Creates an interactive Folium map with:
    - Multimodal Risk Heatmap layer
    - Color-coded Zone Markers with comprehensive disaster & resource popups
    - Proper regional framing and visual risk legend
    """
    if df is None or len(df) == 0:
        return folium.Map(location=[25.0, -85.0], zoom_start=3, tiles="OpenStreetMap")

    lat_col = "latitude" if "latitude" in df.columns else "centroid_lat"
    lon_col = "longitude" if "longitude" in df.columns else "centroid_lon"

    valid_coords = df.dropna(subset=[lat_col, lon_col])
    if len(valid_coords) == 0:
        return folium.Map(location=[25.0, -85.0], zoom_start=3, tiles="OpenStreetMap")

    # Determine optimal initial center and zoom
    if center is None:
        if len(valid_coords) <= 5:
            center = [valid_coords[lat_col].mean(), valid_coords[lon_col].mean()]
            zoom_start = 8 if zoom_start is None else zoom_start
        else:
            # Default to framing the primary disaster cluster (Americas) nicely
            americas = valid_coords[valid_coords[lon_col] < 0]
            if len(americas) > 0:
                center = [americas[lat_col].mean(), americas[lon_col].mean()]
            else:
                center = [valid_coords[lat_col].mean(), valid_coords[lon_col].mean()]
            zoom_start = 3 if zoom_start is None else zoom_start

    m = folium.Map(
        location=center,
        zoom_start=zoom_start,
        tiles="OpenStreetMap"
    )

    # 1. HeatMap Layer
    risk_col = "fused_risk" if "fused_risk" in df.columns else "sat_mean_severity"
    heat_data = [
        [row[lat_col], row[lon_col], float(row[risk_col])]
        for _, row in valid_coords.iterrows()
    ]
    heat_layer = folium.FeatureGroup(name="🔥 Risk Heatmap (Spatial Density)", show=True)
    HeatMap(heat_data, radius=28, blur=18, min_opacity=0.45).add_to(heat_layer)
    heat_layer.add_to(m)

    # 2. Zone Markers Layer
    markers_layer = folium.FeatureGroup(name="📍 Disaster Zones (Interactive Markers)", show=True)

    for _, row in valid_coords.iterrows():
        zone_id = str(row.get("zone_id", "N/A"))
        event = str(row.get("dominant_event", row.get("disaster_event", "Disaster Event")))
        dtype = str(row.get("dominant_disaster_type", row.get("disaster_type", "General")))
        score = float(row.get(risk_col, 0.0))
        category = str(row.get("risk_category", "Unknown"))
        color = get_risk_color(category)

        s_sat = float(row.get("S_sat", row.get("sat_mean_severity", 0.0)))
        s_soc = float(row.get("S_soc", row.get("text_severity", 0.0)))
        buildings = int(row.get("total_buildings", 0))
        pop = int(row.get("total_population_Pk", 0))
        injured = int(row.get("injured_total_Ik", 0))
        displaced = int(row.get("displaced_total_Dk", 0))
        beds = int(row.get("hospital_beds_Bk", 0))
        shelter = int(row.get("shelter_capacity_Sk", 0))
        ambulances = int(row.get("ambulances_Ak", 0))
        buses = int(row.get("buses_Uk", 0))
        boats = int(row.get("boats_Bok", 0))

        popup_html = f"""
        <div style="font-family: Arial, sans-serif; min-width: 270px; max-width: 320px; font-size: 13px;">
            <div style="background-color: {color}; color: white; padding: 7px 10px; border-radius: 4px 4px 0 0; font-weight: bold; font-size: 14px;">
                Zone {zone_id} &bull; {category.upper()} RISK
            </div>
            <div style="padding: 10px; border: 1px solid #ddd; border-top: none; background: #ffffff;">
                <p style="margin: 3px 0;"><strong>Event:</strong> {event} (<em>{dtype.title()}</em>)</p>
                <p style="margin: 3px 0;"><strong>Fused Risk Score:</strong> <span style="color: {color}; font-weight: bold; font-size: 14px;">{score:.4f}</span></p>
                <hr style="margin: 6px 0; border: 0; border-top: 1px solid #eee;">
                <p style="margin: 2px 0;"><strong>Satellite Damage (S_sat):</strong> {s_sat:.3f}</p>
                <p style="margin: 2px 0;"><strong>Social Text Score (S_soc):</strong> {s_soc:.3f}</p>
                <p style="margin: 2px 0;"><strong>Total Buildings:</strong> {buildings:,}</p>
                <hr style="margin: 6px 0; border: 0; border-top: 1px solid #eee;">
                <p style="margin: 2px 0;"><strong>Population Exposed:</strong> {pop:,}</p>
                <p style="margin: 2px 0;"><strong>Casualties (Injured):</strong> <span style="color: #d32f2f; font-weight: bold;">{injured:,}</span></p>
                <p style="margin: 2px 0;"><strong>Displaced Residents:</strong> {displaced:,}</p>
                <hr style="margin: 6px 0; border: 0; border-top: 1px solid #eee;">
                <p style="margin: 2px 0;"><strong>Resources Available:</strong></p>
                <p style="margin: 2px 0; font-size: 12px; color: #555;">
                    🛏️ {beds} beds &bull; ⛺ {shelter} shelters &bull; 🚑 {ambulances} amb &bull; 🚤 {boats} boats
                </p>
            </div>
        </div>
        """

        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=8 + int(score * 14),
            color=color,
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=350),
            tooltip=f"Zone {zone_id} | {event} | {category} ({score:.3f})"
        ).add_to(markers_layer)

    markers_layer.add_to(m)
    folium.LayerControl(position="topright", collapsed=False).add_to(m)

    return m
