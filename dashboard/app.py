# to run: streamlit run dashboard/app.py

import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Indian Railways Punctuality, Bottleneck & Delay Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Resolve project root directory whether launched from root or dashboard/
CWD = Path.cwd()
BASE_DIR = CWD.parent if CWD.name == "dashboard" else CWD
RESULT_DIR = BASE_DIR / "results"
FIG_DIR = RESULT_DIR / "figures"

@st.cache_data
def load_dashboard_data():
    data = {}
    csv_files = {
        "partition_summary": "olap_step03_02_partition_summary.csv",
        "denom_benchmarks": "olap_step03_02_denominator_benchmarks.csv",
        "train_cat": "olap_step03_03_train_category.csv",
        "premium_drill": "olap_step03_03_train_category_premium_drilldown.csv",
        "zone": "olap_step03_03_railway_zone.csv",
        "top20_bottlenecks": "olap_step03_03_station_top20_bottlenecks_ge1000.csv",
        "top20_hubs": "olap_step03_03_station_top20_major_hubs_ge25000.csv",
        "quarterly": "olap_step03_03_temporal_quarterly.csv",
        "monthly": "olap_step03_03_temporal_monthly.csv",
        "weekday": "olap_step03_03_temporal_weekday.csv",
        "zone_x_type": "olap_step03_03_cuboid_zone_x_traintype.csv",
        "route_stage": "olap_step03_04_route_stage_progression.csv",
        "distance_band": "olap_step03_04_distance_band_progression.csv",
        "diurnal": "olap_step03_04_diurnal_time_of_day.csv",
        "top20_amp_stations": "olap_step03_04_station_top20_delay_amplifiers_ge1000.csv",
        "top20_rec_stations": "olap_step03_04_station_top20_delay_recoverers_ge1000.csv",
        "composite_hubs": "step04_01_top25_composite_bottleneck_hubs.csv",
        "trunk_amplifiers": "step04_01_top25_trunk_corridor_amplifiers_ge3000.csv",
        "model_overall": "step04_03_model_comparison_overall.csv",
        "model_severity": "step04_03_test_performance_by_severity.csv",
        "model_stop_type": "step04_03_test_performance_by_stop_type.csv",
        "model_train_cat": "step04_03_test_performance_by_train_category.csv",
        "feature_imp": "step04_03_feature_importance_comparison.csv",
    }
    for key, fname in csv_files.items():
        fp = RESULT_DIR / fname
        if fp.exists():
            data[key] = pd.read_csv(fp)
    return data

data = load_dashboard_data()

st.title("Indian Railways Network Punctuality, Bottleneck & Delay Prediction Dashboard")
st.caption(
    "Full-Year Operational Analysis (2025-02-08 to 2026-02-07) | 36,614,824 Cleaned Records | "
    "8,720 Trains | 9,027 Stations | 33,158,834 Scheduled Prediction Transitions"
)

tabs = st.tabs([
    "1. Network Overview",
    "2. Train Category & Zone Comparison",
    "3. Station & Route Bottlenecks",
    "4. Temporal & Route-Stage Trends",
    "5. Regression Models & Delay Predictor",
])

# ==========================================
# TAB 1: NETWORK-WIDE PUNCTUALITY OVERVIEW
# ==========================================
with tabs[0]:
    st.subheader("Network-Wide Punctuality & Structural Partition Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Cleaned Warehouse Rows", "36,614,824", "97.18% Observed")
    c2.metric("Observed Mean Delay", "35.68 min", "Median: 16.0 min")
    c3.metric("Origin On-Time (<=15m)", "86.56%", "Median: 1.0 min")
    c4.metric("En-Route Arrival On-Time", "47.36%", "Median: 17.0 min")
    c5.metric("Bottleneck Rate (>2 SD)", "3.47%", "> 167.65 min")

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("#### Origin Departure (`station_no = 1`) vs. En-Route Arrival (`station_no > 1`)")
        if "denom_benchmarks" in data:
            st.dataframe(data["denom_benchmarks"], use_container_width=True)
    with col_right:
        st.markdown("#### 2x2 Warehouse Partition Matrix (`is_imputed` × `has_schedule`)")
        if "partition_summary" in data:
            st.dataframe(data["partition_summary"], use_container_width=True)

# ==========================================
# TAB 2: TRAIN-CATEGORY & ZONE COMPARISON
# ==========================================
with tabs[1]:
    st.subheader("Service Category & Railway Zone Performance Comparison")
    col1, col2 = st.columns(2)
    with col1:
        if "train_cat" in data:
            tc = data["train_cat"].sort_values("mean_delay_min", ascending=False)
            fig_tc = px.bar(
                tc, x="primary_type_code", y="mean_delay_min",
                color="bottleneck_gt_2sd_pct",
                title="Train Categories: Mean Delay (min) & Bottleneck Rate (>2 SD %)",
                labels={"mean_delay_min": "Mean Delay (min)", "primary_type_code": "Train Category"},
                color_continuous_scale="YlOrRd",
            )
            st.plotly_chart(fig_tc, use_container_width=True)
            st.dataframe(tc, use_container_width=True)
    with col2:
        if "zone" in data:
            zn = data["zone"][data["zone"]["observed_stops"] >= 10000].sort_values("bottleneck_gt_2sd_pct", ascending=True)
            fig_zn = px.bar(
                zn, y="station_zone", x="bottleneck_gt_2sd_pct", orientation="h",
                color="mean_delay_min",
                title="Railway Zones: Bottleneck Rate (>2 SD %) & Mean Delay",
                labels={"bottleneck_gt_2sd_pct": "Bottleneck Rate (%)", "station_zone": "Railway Zone"},
                color_continuous_scale="Reds",
            )
            st.plotly_chart(fig_zn, use_container_width=True)
            st.dataframe(zn.sort_values("bottleneck_gt_2sd_pct", ascending=False), use_container_width=True)

# ==========================================
# TAB 3: STATION & ROUTE BOTTLENECK EXPLORATION
# ==========================================
with tabs[2]:
    st.subheader("Topological Centrality, Composite Bottleneck Hubs & Transition Amplifiers")
    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("#### Top Composite Structural-Operational Bottleneck Hubs (>= 20,000 Training Stops)")
        if "composite_hubs" in data:
            hubs = data["composite_hubs"]
            fig_hub = px.scatter(
                hubs, x="betweenness_centrality", y="train_arr_mean_delay_min",
                size="total_degree", color="train_bottleneck_gt_2sd_pct",
                hover_name="station_full_name", text="station_name",
                title="Static Schedule Betweenness Centrality vs. Training Mean Arrival Delay",
                color_continuous_scale="Magma",
            )
            fig_hub.update_traces(textposition="top center")
            st.plotly_chart(fig_hub, use_container_width=True)
            st.dataframe(hubs, use_container_width=True)
    with c_right:
        st.markdown("#### High-Volume Trunk Corridor Delay Amplifiers (>= 3,000 Runs)")
        if "trunk_amplifiers" in data:
            tr = data["trunk_amplifiers"].copy()
            tr["corridor"] = tr["from_station"] + " -> " + tr["to_station"] + " (" + tr["to_station_zone"] + ")"
            fig_tr = px.bar(
                tr.head(15).iloc[::-1], x="train_mean_delay_delta_min", y="corridor", orientation="h",
                color="train_amp_gt_5m_rate",
                title="Top 15 Trunk Transitions by Mean Delay Amplification (min)",
                color_continuous_scale="OrRd",
            )
            st.plotly_chart(fig_tr, use_container_width=True)
            st.dataframe(tr, use_container_width=True)

# ==========================================
# TAB 4: TEMPORAL & ROUTE-STAGE TRENDS
# ==========================================
with tabs[3]:
    st.subheader("Seasonal Trajectory, Diurnal Cycle & Route Progression (Terminal Padding)")
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        if "monthly" in data:
            m_df = data["monthly"].copy()
            m_df["period_label"] = m_df["month_name"].str.slice(0, 3) + " " + m_df["year"].astype(str)
            fig_m = go.Figure()
            fig_m.add_trace(go.Scatter(x=m_df["period_label"], y=m_df["mean_delay_min"], mode="lines+markers", name="Mean Delay (min)"))
            fig_m.add_trace(go.Scatter(x=m_df["period_label"], y=m_df["median_delay_min"], mode="lines+markers", name="Median Delay (min)"))
            fig_m.update_layout(title="Monthly Delay Trajectory (Winter Fog Surge in Dec 2025 – Jan 2026)", xaxis_title="Month", yaxis_title="Minutes")
            st.plotly_chart(fig_m, use_container_width=True)
    with r1c2:
        if "route_stage" in data:
            rs_df = data["route_stage"]
            fig_rs = px.bar(
                rs_df, x="route_stage", y="mean_delay_min", color="early_lt_0m_pct",
                title="Route Progression & Terminal Schedule Padding (31.61% Early at Terminal)",
                color_continuous_scale="Tealgrn",
            )
            st.plotly_chart(fig_rs, use_container_width=True)

    if "diurnal" in data:
        st.markdown("#### Diurnal (Time-of-Day) Arrival Performance")
        st.dataframe(data["diurnal"], use_container_width=True)

# ==========================================
# TAB 5: REGRESSION PREDICTION & MODEL COMPARISON
# ==========================================
with tabs[4]:
    st.subheader("Independent Test Set Evaluation (n = 3,351,989 Observed Transitions)")
    if "model_overall" in data:
        m_df = data["model_overall"].copy()
        m_df["train_MSE"] = np.round(m_df["train_RMSE"] ** 2, 4)
        m_df["val_MSE"] = np.round(m_df["val_RMSE"] ** 2, 4)
        m_df["test_MSE"] = np.round(m_df["test_RMSE"] ** 2, 4)
        st.dataframe(m_df, use_container_width=True)

        fig_mod = px.bar(
            m_df, x="model", y=["test_MAE", "test_RMSE"],
            barmode="group",
            title="Independent Test Set Error Comparison: test_MAE & test_RMSE (minutes)",
            labels={"value": "Error (minutes)", "model": "Model", "variable": "Metric"},
        )
        st.plotly_chart(fig_mod, use_container_width=True)

    mc1, mc2 = st.columns(2)
    with mc1:
        st.markdown("#### Performance Across Delay Severity Regimes (`target_severity`)")
        if "model_severity" in data:
            sev = data["model_severity"].copy()
            sev["MSE"] = np.round(sev["RMSE"] ** 2, 4)
            st.dataframe(sev, use_container_width=True)

        st.markdown("#### Performance by Stop Context (`stop_context`)")
        if "model_stop_type" in data:
            stp = data["model_stop_type"].copy()
            stp["MSE"] = np.round(stp["RMSE"] ** 2, 4)
            st.dataframe(stp, use_container_width=True)

    with mc2:
        st.markdown("#### Performance by Train Category (`primary_type_code`)")
        if "model_train_cat" in data:
            tcat = data["model_train_cat"].copy()
            tcat["MSE"] = np.round(tcat["RMSE"] ** 2, 4)
            st.dataframe(tcat, use_container_width=True)

        st.markdown("#### Feature Importance Comparison (`feature_name`)")
        if "feature_imp" in data:
            st.dataframe(data["feature_imp"], use_container_width=True)

    st.markdown("---")
    st.markdown("### Interactive Next-Station Arrival Delay Simulator")
    sc1, sc2, sc3, sc4 = st.columns(4)
    curr_d = sc1.number_input("Current Station Delay (min)", min_value=-60.0, max_value=1440.0, value=25.0, step=1.0)
    prev_d = sc2.number_input("Previous Station Delay (min)", min_value=-60.0, max_value=1440.0, value=20.0, step=1.0)
    trans_delta = sc3.number_input("Historical Edge Amplification Δd (min)", min_value=-80.0, max_value=90.0, value=2.5, step=0.5)
    sched_run = sc4.number_input("Scheduled Segment Running Time (min)", min_value=1.0, max_value=240.0, value=18.0, step=1.0)
    is_term = st.checkbox("Next Station is Terminal Destination (Schedule Slack Recovery)", value=False)

    persistence_pred = curr_d
    slack_adj = -8.5 if is_term else 0.0
    ml_est = max(-120.0, min(1440.0, 0.945 * curr_d + 0.035 * prev_d + 0.85 * trans_delta - 0.12 * max(0.0, sched_run - 12.0) + slack_adj))

    p1, p2, p3 = st.columns(3)
    p1.metric("Persistence Baseline Forecast", f"{persistence_pred:.1f} min")
    p2.metric("ML Model Adjusted Forecast", f"{ml_est:.1f} min", f"{ml_est - persistence_pred:+.1f} min vs Baseline")
    p3.metric("Expected Trajectory Shift (next - curr)", f"{ml_est - curr_d:+.1f} min")