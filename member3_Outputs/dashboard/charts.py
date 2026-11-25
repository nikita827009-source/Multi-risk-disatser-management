import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

RISK_COLORS = {
    "Critical": "#d32f2f",
    "High": "#f57c00",
    "Moderate": "#fbc02d",
    "Low": "#388e3c"
}

def plot_top_critical_zones(df, top_n=10):
    """
    Horizontal bar chart showing the top critical disaster zones by fused risk score.
    """
    if df is None or len(df) == 0:
        return go.Figure()

    risk_col = "fused_risk" if "fused_risk" in df.columns else "sat_mean_severity"
    top_df = df.sort_values(by=risk_col, ascending=False).head(top_n).copy()
    
    # Create descriptive zone label
    event_col = "dominant_event" if "dominant_event" in top_df.columns else "disaster_event"
    top_df["display_label"] = top_df.apply(
        lambda r: f"{r['zone_id']} ({r[event_col]})" if event_col in r and pd.notna(r[event_col]) else str(r['zone_id']),
        axis=1
    )
    # Reverse order so highest risk appears at top of horizontal chart
    top_df = top_df.iloc[::-1]

    fig = px.bar(
        top_df,
        x=risk_col,
        y="display_label",
        orientation="h",
        color="risk_category" if "risk_category" in top_df.columns else None,
        color_discrete_map=RISK_COLORS,
        text=risk_col,
        hover_data={
            "dominant_event": True,
            "dominant_disaster_type": True,
            "S_sat": ":.3f" if "S_sat" in top_df.columns else False,
            "S_soc": ":.3f" if "S_soc" in top_df.columns else False,
            "injured_total_Ik": True if "injured_total_Ik" in top_df.columns else False,
            "display_label": False
        },
        title=f"🚨 Top {min(top_n, len(df))} Priority Disaster Zones (Fused Multi-Modal Risk)"
    )

    fig.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside",
        marker_line_color="rgba(0,0,0,0.25)",
        marker_line_width=1
    )
    fig.update_layout(
        xaxis_title="Fused Risk Score (0.00 - 1.00)",
        yaxis_title="Disaster Zone (ID & Event)",
        xaxis_range=[0, max(top_df[risk_col].max() * 1.2, 0.75)],
        margin=dict(l=40, r=40, t=50, b=40),
        height=420,
        legend_title="Risk Level"
    )
    return fig

def plot_risk_category_distribution(df):
    """
    Donut chart displaying the breakdown of zones by risk severity category.
    """
    if df is None or len(df) == 0:
        return go.Figure()

    cat_order = ["Critical", "High", "Moderate", "Low"]
    counts = df["risk_category"].value_counts()
    
    records = []
    for cat in cat_order:
        records.append({
            "risk_category": cat,
            "count": int(counts.get(cat, 0))
        })
    category_counts = pd.DataFrame(records)
    # Only show non-zero categories in pie slices
    non_zero = category_counts[category_counts["count"] > 0]

    fig = px.pie(
        non_zero,
        names="risk_category",
        values="count",
        color="risk_category",
        color_discrete_map=RISK_COLORS,
        hole=0.45,
        title="📊 Disaster Risk Category Distribution"
    )
    fig.update_traces(
        textinfo="label+value+percent",
        textposition="inside",
        pull=[0.08 if cat in ["Critical", "High"] else 0 for cat in non_zero["risk_category"]]
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=20),
        height=420,
        legend_title="Severity Tier"
    )
    return fig

def plot_multimodal_correlation(df):
    """
    Scatter plot comparing Satellite Severity vs Social Text Severity.
    """
    if df is None or len(df) == 0:
        return go.Figure()

    sat_col = "S_sat" if "S_sat" in df.columns else "sat_mean_severity"
    soc_col = "S_soc" if "S_soc" in df.columns else "text_severity"

    fig = px.scatter(
        df,
        x=sat_col,
        y=soc_col,
        color="risk_category" if "risk_category" in df.columns else None,
        color_discrete_map=RISK_COLORS,
        size="observation_density_delta_k" if "observation_density_delta_k" in df.columns else None,
        hover_name="zone_id",
        hover_data=["dominant_event", "dominant_disaster_type", "fused_risk"],
        title="🛰️ Multimodal Correlation: Satellite Vision (S_sat) vs. Social Text (S_soc)"
    )

    # Diagonal reference line
    fig.add_shape(
        type="line",
        x0=0, y0=0, x1=1, y1=1,
        line=dict(color="#888", dash="dash", width=1.5)
    )

    fig.update_layout(
        xaxis_title="Satellite Damage Severity (S_sat)",
        yaxis_title="Social Text Severity (S_soc)",
        xaxis_range=[-0.05, 1.05],
        yaxis_range=[-0.05, 1.05],
        margin=dict(l=40, r=40, t=50, b=40),
        height=400,
        legend_title="Risk Level"
    )
    return fig

def plot_disaster_type_breakdown(df):
    """
    Bar chart showing total zones and average risk score by disaster type.
    """
    if df is None or len(df) == 0:
        return go.Figure()

    dtype_col = "dominant_disaster_type" if "dominant_disaster_type" in df.columns else "disaster_type"
    risk_col = "fused_risk" if "fused_risk" in df.columns else "sat_mean_severity"

    grouped = df.groupby(dtype_col).agg(
        zone_count=("zone_id", "count"),
        avg_risk=(risk_col, "mean")
    ).reset_index().sort_values(by="avg_risk", ascending=False)

    fig = px.bar(
        grouped,
        x=dtype_col,
        y="avg_risk",
        text="zone_count",
        color="avg_risk",
        color_continuous_scale="Reds",
        title="🌪️ Average Risk Score & Zone Count by Hazard Type"
    )

    fig.update_traces(
        texttemplate="%{text} zones",
        textposition="outside"
    )
    fig.update_layout(
        xaxis_title="Disaster Hazard Type",
        yaxis_title="Average Fused Risk Score",
        yaxis_range=[0, max(grouped["avg_risk"].max() * 1.3, 0.7)],
        margin=dict(l=40, r=40, t=50, b=40),
        height=400,
        coloraxis_colorbar=dict(title="Avg Risk")
    )
    return fig

def plot_resource_allocation_chart(df, top_n=6):
    """
    Subplots comparing:
    1. Casualties (Injured & Displaced) vs Shelter & Hospital Capacity
    2. Evacuation Fleet (Ambulances, Buses, Rescue Boats)
    """
    if df is None or len(df) == 0:
        return go.Figure()

    risk_col = "fused_risk" if "fused_risk" in df.columns else "sat_mean_severity"
    top_df = df.sort_values(by=risk_col, ascending=False).head(top_n).copy()

    # Create 2 subplots with shared x-axis
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.15,
        subplot_titles=(
            "🏥 Casualties vs. Medical & Shelter Capacity",
            "🚑 Emergency Evacuation & Rescue Fleet"
        )
    )

    # Subplot 1: Casualties vs Medical / Shelters
    if "injured_total_Ik" in top_df.columns:
        fig.add_trace(go.Bar(
            name="Injured Casualties",
            x=top_df["zone_id"],
            y=top_df["injured_total_Ik"],
            marker_color="#d32f2f"
        ), row=1, col=1)

    if "hospital_beds_Bk" in top_df.columns:
        fig.add_trace(go.Bar(
            name="Hospital Beds",
            x=top_df["zone_id"],
            y=top_df["hospital_beds_Bk"],
            marker_color="#1976d2"
        ), row=1, col=1)

    if "shelter_capacity_Sk" in top_df.columns:
        fig.add_trace(go.Bar(
            name="Shelter Capacity",
            x=top_df["zone_id"],
            y=top_df["shelter_capacity_Sk"],
            marker_color="#388e3c"
        ), row=1, col=1)

    # Subplot 2: Fleet (Ambulances, Buses, Boats)
    if "ambulances_Ak" in top_df.columns:
        fig.add_trace(go.Bar(
            name="Ambulances",
            x=top_df["zone_id"],
            y=top_df["ambulances_Ak"],
            marker_color="#fb8c00"
        ), row=2, col=1)

    if "buses_Uk" in top_df.columns:
        fig.add_trace(go.Bar(
            name="Transit Buses",
            x=top_df["zone_id"],
            y=top_df["buses_Uk"],
            marker_color="#8e24aa"
        ), row=2, col=1)

    if "boats_Bok" in top_df.columns:
        fig.add_trace(go.Bar(
            name="Rescue Boats",
            x=top_df["zone_id"],
            y=top_df["boats_Bok"],
            marker_color="#00acc1"
        ), row=2, col=1)

    fig.update_layout(
        barmode="group",
        height=540,
        margin=dict(l=40, r=40, t=85, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.10, xanchor="center", x=0.5)
    )
    return fig
