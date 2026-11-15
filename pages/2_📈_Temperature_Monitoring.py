"""
ColdGuard AI - PAGE 3: TEMPERATURE & HUMIDITY MONITORING
Displays interactive multi-sensor time-series charts, configurable safe thermal band,
sensor-wise comparisons, and highlighted temperature excursions.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import config
from src.ui_components import inject_custom_css, initialize_session_state, render_sidebar, apply_plotly_theme

st.set_page_config(
    page_title=f"{config.APP_NAME} - Thermal Monitoring",
    page_icon="📈",
    layout="wide"
)

inject_custom_css()
initialize_session_state()
render_sidebar()

st.title("📈 Temperature & Humidity Monitoring")
st.markdown(
    "Analyze temporal environmental patterns across monitored cold-storage zones. "
    "Excursions beyond the configured operating band are dynamically highlighted."
)

df = st.session_state.df

if df.empty:
    st.warning("⚠️ No dataset loaded. Please go to **Data Upload** to load a dataset.")
    st.stop()

# Sensor Filter Selector
all_sensors = sorted(df["sensor_id"].unique().tolist())
fcol1, fcol2 = st.columns([8, 4])
with fcol1:
    selected_sensors = st.multiselect(
        "Filter Sensors to Compare",
        options=all_sensors,
        default=all_sensors,
        help="Select one or multiple sensors to display and compare simultaneously."
    )
with fcol2:
    highlight_excursions = st.checkbox("Highlight Excursion Points", value=True)

if not selected_sensors:
    st.info("Please select at least one sensor to display charts.")
    st.stop()

filtered_df = df[df["sensor_id"].isin(selected_sensors)].copy()

# ==========================================
# 1. TEMPERATURE TIME-SERIES WITH BAND
# ==========================================
st.subheader("🌡️ Cold-Storage Temperature Trajectory")

fig_temp = go.Figure()

# Add acceptable temperature band (shaded rectangle)
min_t = st.session_state.min_temp
max_t = st.session_state.max_temp
min_ts = filtered_df["timestamp"].min()
max_ts = filtered_df["timestamp"].max()

fig_temp.add_hrect(
    y0=min_t,
    y1=max_t,
    fillcolor="rgba(16, 185, 129, 0.12)",
    line_width=1,
    line_color="rgba(16, 185, 129, 0.4)",
    annotation_text=f"Safe Operating Band ({min_t}°C to {max_t}°C)",
    annotation_position="top left"
)

# Plot sensor temperature lines
palette = ["#38BDF8", "#F59E0B", "#A855F7", "#EC4899", "#10B981", "#6366F1", "#14B8A6", "#F97316", "#E11D48"]
for i, sensor_id in enumerate(selected_sensors):
    s_data = filtered_df[filtered_df["sensor_id"] == sensor_id]
    color = palette[i % len(palette)]
    
    fig_temp.add_trace(go.Scatter(
        x=s_data["timestamp"],
        y=s_data["temperature"],
        mode="lines",
        name=f"{sensor_id} Temp",
        line=dict(color=color, width=2),
        hovertemplate="<b>%{x}</b><br>Sensor: " + sensor_id + "<br>Temp: %{y:.2f}°C<extra></extra>"
    ))

    # Highlight Excursion points if enabled
    if highlight_excursions:
        excursion_pts = s_data[(s_data["temperature"] > max_t) | (s_data["temperature"] < min_t)]
        if not excursion_pts.empty:
            fig_temp.add_trace(go.Scatter(
                x=excursion_pts["timestamp"],
                y=excursion_pts["temperature"],
                mode="markers",
                name=f"{sensor_id} Excursions",
                marker=dict(color="#EF4444", size=6, symbol="x"),
                showlegend=False,
                hovertemplate="<b>EXCURSION: %{x}</b><br>Sensor: " + sensor_id + "<br>Temp: %{y:.2f}°C<extra></extra>"
            ))

fig_temp.update_layout(
    xaxis_title="Timestamp",
    yaxis_title="Temperature (°C)",
    hovermode="x unified"
)
fig_temp = apply_plotly_theme(fig_temp, "Temperature Time Series with Safe Operating Band & Excursions")
st.plotly_chart(fig_temp, use_container_width=True)

# ==========================================
# 2. HUMIDITY TIME-SERIES
# ==========================================
st.subheader("💧 Relative Humidity Trajectory")

if "humidity" in filtered_df.columns:
    fig_hum = go.Figure()
    for i, sensor_id in enumerate(selected_sensors):
        s_data = filtered_df[filtered_df["sensor_id"] == sensor_id]
        color = palette[i % len(palette)]
        
        fig_hum.add_trace(go.Scatter(
            x=s_data["timestamp"],
            y=s_data["humidity"],
            mode="lines",
            name=f"{sensor_id} Humidity",
            line=dict(color=color, width=1.5),
            hovertemplate="<b>%{x}</b><br>Sensor: " + sensor_id + "<br>RH: %{y:.1f}%<extra></extra>"
        ))

    fig_hum.update_layout(
        xaxis_title="Timestamp",
        yaxis_title="Relative Humidity (%)",
        hovermode="x unified"
    )
    fig_hum = apply_plotly_theme(fig_hum, "Relative Humidity (% RH) Over Time")
    st.plotly_chart(fig_hum, use_container_width=True)
else:
    st.info("Humidity column not available in active dataset.")

# ==========================================
# 3. SENSOR COMPARISON METRICS TABLE
# ==========================================
st.subheader("📊 Sensor-Wise Comparative Analytics")

summary_rows = []
for sensor_id in selected_sensors:
    s_data = filtered_df[filtered_df["sensor_id"] == sensor_id]
    temp = s_data["temperature"]
    excursions = ((temp > max_t) | (temp < min_t)).sum()
    pct_excursion = (excursions / max(len(temp), 1)) * 100
    
    summary_rows.append({
        "Sensor ID": sensor_id,
        "Total Readings": len(temp),
        "Min Temp (°C)": round(temp.min(), 1),
        "Max Temp (°C)": round(temp.max(), 1),
        "Mean Temp (°C)": round(temp.mean(), 1),
        "Std Dev (°C)": round(temp.std(), 2) if len(temp) > 1 else 0.0,
        "Excursion Count": excursions,
        "Excursion Rate (%)": round(pct_excursion, 1)
    })

st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
