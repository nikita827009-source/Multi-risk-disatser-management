import os
import sys
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

# Ensure local imports work reliably
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from maps import create_risk_map, EVENT_REGIONS, RISK_COLORS
from charts import (
    plot_top_critical_zones,
    plot_risk_category_distribution,
    plot_multimodal_correlation,
    plot_disaster_type_breakdown,
    plot_resource_allocation_chart
)

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Multi-Disaster Risk Management System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 4px solid #1e88e5;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .badge-critical {
        background-color: #ffebee;
        color: #c62828;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-high {
        background-color: #fff3e0;
        color: #e65100;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        font-weight: 600;
        font-size: 15px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner="Loading disaster zone data...")
def load_data():
    """
    Loads zone-level risk dataset, executing the fusion pipeline if not yet generated.
    """
    candidate_paths = [
        CURRENT_DIR.parent / "data" / "final_zone_risk.csv",
        REPO_ROOT / "member3_Outputs" / "data" / "final_zone_risk.csv",
        CURRENT_DIR.parent / "data" / "fused_zone_risk.csv",
        REPO_ROOT / "CV_Spatial_Outputs" / "top_20_critical_zones.csv"
    ]

    for path in candidate_paths:
        if path.exists() and path.stat().st_size > 0:
            df = pd.read_csv(path)
            return df

    # If file does not exist, run the fusion pipeline automatically
    try:
        from member3_Outputs.fusion.fusion_engine import run_fusion_pipeline
        df = run_fusion_pipeline(str(REPO_ROOT))
        return df
    except Exception as e:
        st.error(f"Error initializing fusion pipeline: {e}")
        return pd.DataFrame()

raw_df = load_data()

if raw_df.empty:
    st.error("No disaster data found. Please run the fusion pipeline to generate the dataset.")
    st.stop()

# --- Sidebar Controls ---
st.sidebar.header("🔍 Filters & Controls")

# Threshold Configuration in Sidebar
with st.sidebar.expander("⚙️ Calibrate Risk Thresholds", expanded=False):
    st.caption("Adjust risk classification boundaries to re-tier disaster zones:")
    crit_threshold = st.slider("Critical Threshold", 0.35, 0.70, 0.40, 0.05)
    high_threshold = st.slider("High Risk Threshold", 0.25, 0.50, 0.30, 0.05)
    mod_threshold = st.slider("Moderate Threshold", 0.10, 0.30, 0.18, 0.02)

# Dynamically re-classify based on active thresholds
df = raw_df.copy()
def dynamic_classify(score):
    if score >= crit_threshold:
        return "Critical"
    elif score >= high_threshold:
        return "High"
    elif score >= mod_threshold:
        return "Moderate"
    else:
        return "Low"

if "fused_risk" in df.columns:
    df["risk_category"] = df["fused_risk"].apply(dynamic_classify)

# Event Filter
all_events = sorted(df["dominant_event"].dropna().unique().tolist()) if "dominant_event" in df.columns else []
selected_events = st.sidebar.multiselect("Disaster Event", options=all_events, default=[])

# Disaster Type Filter
all_types = sorted(df["dominant_disaster_type"].dropna().unique().tolist()) if "dominant_disaster_type" in df.columns else []
selected_types = st.sidebar.multiselect("Disaster Hazard Type", options=all_types, default=[])

# Risk Category Filter
all_categories = ["Critical", "High", "Moderate", "Low"]
existing_categories = [c for c in all_categories if c in df["risk_category"].unique()]
selected_categories = st.sidebar.multiselect("Risk Category", options=existing_categories, default=[])

# Minimum Risk Slider
min_risk = st.sidebar.slider("Minimum Fused Risk Score", 0.0, 1.0, 0.0, 0.05)

# Pipeline Re-run button
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Re-run Fusion Pipeline"):
    st.cache_data.clear()
    try:
        from member3_Outputs.fusion.fusion_engine import run_fusion_pipeline
        run_fusion_pipeline(str(REPO_ROOT))
        st.sidebar.success("Pipeline re-run complete!")
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Failed to re-run pipeline: {e}")

# Apply Filters
filtered_df = df.copy()

if selected_events:
    filtered_df = filtered_df[filtered_df["dominant_event"].isin(selected_events)]

if selected_types:
    filtered_df = filtered_df[filtered_df["dominant_disaster_type"].isin(selected_types)]

if selected_categories:
    filtered_df = filtered_df[filtered_df["risk_category"].isin(selected_categories)]

if "fused_risk" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["fused_risk"] >= min_risk]

# Ensure sorted by priority
filtered_df = filtered_df.sort_values(by="fused_risk", ascending=False).reset_index(drop=True)

st.sidebar.info(f"Showing **{len(filtered_df)}** of **{len(df)}** monitored disaster zones.")

# --- Header Section ---
st.title("🚨 Multi-Disaster Risk Management System")
st.caption("AI-Powered Multimodal Fusion, Geospatial Hazard Mapping & Emergency Logistics Coordination")

# --- Key Metrics Overview ---
st.subheader("Disaster Risk Overview")
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

total_zones = len(filtered_df)
critical_risk = (filtered_df["risk_category"] == "Critical").sum() if "risk_category" in filtered_df.columns else 0
high_risk = (filtered_df["risk_category"] == "High").sum() if "risk_category" in filtered_df.columns else 0
total_pop = int(filtered_df["total_population_Pk"].sum()) if "total_population_Pk" in filtered_df.columns else 0
total_injured = int(filtered_df["injured_total_Ik"].sum()) if "injured_total_Ik" in filtered_df.columns else 0

kpi1.metric("Monitored Zones", total_zones)
kpi2.metric("Critical Priority Zones", critical_risk, delta=f"{critical_risk} Urgent" if critical_risk > 0 else "None", delta_color="inverse")
kpi3.metric("High Risk Zones", high_risk, delta=f"{high_risk} High Alert" if high_risk > 0 else "None", delta_color="inverse")
kpi4.metric("Exposed Population", f"{total_pop:,}")
kpi5.metric("Total Casualties (Injured)", f"{total_injured:,}")

st.markdown("---")

# --- Interactive Main Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Geospatial Risk Map",
    "📊 Risk Analytics & Rankings",
    "🚑 Resource Logistics & Allocation",
    "📋 Zone Data Explorer"
])

# TAB 1: GEOSPATIAL MAP
with tab1:
    st.markdown("#### Interactive Geospatial Risk Map & Density Heatmap")
    st.caption("Circle markers denote spatial zone centroids colored by risk severity: **Red: Critical**, **Orange: High**, **Yellow: Moderate**, **Green: Low**. Click any marker to view damage, population, and resource logistics.")

    map_ctrl1, map_ctrl2 = st.columns([2, 1])
    with map_ctrl1:
        region_choice = st.selectbox(
            "📍 Quick Regional Focus (Zoom to Disaster Event):",
            options=list(EVENT_REGIONS.keys()),
            index=0
        )
    with map_ctrl2:
        st.markdown("""
        <div style="padding-top: 28px; font-size: 13px;">
            <span style="color:#d32f2f; font-weight:bold;">● Critical</span> &nbsp;
            <span style="color:#f57c00; font-weight:bold;">● High</span> &nbsp;
            <span style="color:#fbc02d; font-weight:bold;">● Moderate</span> &nbsp;
            <span style="color:#388e3c; font-weight:bold;">● Low</span>
        </div>
        """, unsafe_allow_html=True)

    if len(filtered_df) > 0:
        reg_cfg = EVENT_REGIONS.get(region_choice, {"center": None, "zoom": None})
        risk_map = create_risk_map(filtered_df, center=reg_cfg["center"], zoom_start=reg_cfg["zoom"])
        components.html(risk_map._repr_html_(), height=600)
    else:
        st.warning("No zones match the current filter selection.")

# TAB 2: RISK ANALYTICS
with tab2:
    st.markdown("#### Multimodal Risk Analytics & Severity Rankings")
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        fig_top = plot_top_critical_zones(filtered_df, top_n=10)
        st.plotly_chart(fig_top, width="stretch")

    with col_chart2:
        fig_dist = plot_risk_category_distribution(filtered_df)
        st.plotly_chart(fig_dist, width="stretch")

    col_chart3, col_chart4 = st.columns(2)

    with col_chart3:
        fig_scatter = plot_multimodal_correlation(filtered_df)
        st.plotly_chart(fig_scatter, width="stretch")

    with col_chart4:
        fig_disaster = plot_disaster_type_breakdown(filtered_df)
        st.plotly_chart(fig_disaster, width="stretch")

# TAB 3: RESOURCE ALLOCATION
with tab3:
    st.markdown("#### Emergency Resource Grounding & Logistics Allocation")
    st.caption("Logistics metrics calibrated with WorldPop and FEMA/WHO guidelines for disaster relief operations.")

    r_col1, r_col2, r_col3, r_col4 = st.columns(4)
    beds = int(filtered_df["hospital_beds_Bk"].sum()) if "hospital_beds_Bk" in filtered_df.columns else 0
    shelters = int(filtered_df["shelter_capacity_Sk"].sum()) if "shelter_capacity_Sk" in filtered_df.columns else 0
    ambulances = int(filtered_df["ambulances_Ak"].sum()) if "ambulances_Ak" in filtered_df.columns else 0
    boats = int(filtered_df["boats_Bok"].sum()) if "boats_Bok" in filtered_df.columns else 0

    r_col1.metric("Hospital Beds Available", f"{beds:,}")
    r_col2.metric("Shelter Capacity", f"{shelters:,}")
    r_col3.metric("Ambulances Deployed", f"{ambulances:,}")
    r_col4.metric("Rescue Boats Deployed", f"{boats:,}")

    fig_res = plot_resource_allocation_chart(filtered_df, top_n=8)
    st.plotly_chart(fig_res, width="stretch")

# TAB 4: DATA EXPLORER
with tab4:
    st.markdown("#### Detailed Zone Risk & Logistics Table (Ranked by Risk)")
    st.caption("Inspect, sort, and export the processed multi-modal disaster risk records.")

    display_mapping = {
        "zone_id": "Zone ID",
        "dominant_event": "Disaster Event",
        "dominant_disaster_type": "Hazard Type",
        "risk_category": "Risk Level",
        "fused_risk": "Fused Risk Score",
        "S_sat": "Satellite Damage (S_sat)",
        "S_soc": "Social Score (S_soc)",
        "confidence": "Confidence",
        "urgency": "Urgency",
        "total_buildings": "Buildings",
        "total_population_Pk": "Exposed Population",
        "injured_total_Ik": "Injured Casualties",
        "displaced_total_Dk": "Displaced Residents",
        "hospital_beds_Bk": "Hospital Beds",
        "shelter_capacity_Sk": "Shelter Capacity",
        "ambulances_Ak": "Ambulances",
        "boats_Bok": "Rescue Boats"
    }

    cols_to_use = [c for c in display_mapping.keys() if c in filtered_df.columns]
    table_df = filtered_df[cols_to_use].rename(columns=display_mapping)

    st.dataframe(
        table_df,
        width="stretch",
        height=450,
        column_config={
            "Fused Risk Score": st.column_config.NumberColumn(format="%.4f"),
            "Satellite Damage (S_sat)": st.column_config.NumberColumn(format="%.3f"),
            "Social Score (S_soc)": st.column_config.NumberColumn(format="%.3f"),
            "Confidence": st.column_config.NumberColumn(format="%.2f"),
            "Urgency": st.column_config.NumberColumn(format="%.2f"),
            "Exposed Population": st.column_config.NumberColumn(format="%d"),
            "Injured Casualties": st.column_config.NumberColumn(format="%d"),
            "Displaced Residents": st.column_config.NumberColumn(format="%d"),
            "Buildings": st.column_config.NumberColumn(format="%d")
        }
    )

    csv_data = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Full Disaster Risk Report (CSV)",
        data=csv_data,
        file_name="disaster_risk_management_report.csv",
        mime="text/csv"
    )
