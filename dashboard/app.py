"""
Indian Railways Punctuality & Network Delay Dynamics — interactive dashboard
===========================================================================

Run from the repository root (or from inside dashboard/):

    pip install -r requirements.txt
    streamlit run dashboard/app.py

The app reads only artefacts that are committed to the repository:
    results/step*_FINDINGS.json, results/step05_04_model_response_curves.json,
    results/step04_02_preprocessing_artifacts.json and results/models/*.
If the larger training-period network statistics produced by Step 4.1 are present
(results/step04_01_train_period_station_stats.parquet and
 results/step04_01_train_period_transition_stats.parquet), the predictor also lets you
pick any station pair; otherwise it falls back to corridor presets from the JSON findings.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------------------
# Paths & page config
# --------------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RESULT_DIR = BASE_DIR / "results"
MODEL_DIR = RESULT_DIR / "models"
FIG_DIR = RESULT_DIR / "figures"

st.set_page_config(
    page_title="Indian Railways Delay Analytics",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------------------
# Visual language (validated categorical slots; one hue for single-series bars)
# --------------------------------------------------------------------------------------
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MAGENTA, GREEN, VIOLET, RED = "#e87ba4", "#008300", "#6250d6", "#e34948"
MUTED = "#898781"
SPLIT_COLORS = {"Train": "#86b6ef", "Validation": "#2a78d6", "Test": "#104281"}  # ordinal ramp
MODEL_COLORS = {
    "Persistence": MUTED,
    "Ridge": BLUE,
    "Random Forest": ORANGE,
    "XGBoost": AQUA,
}
MODEL_NAMES = {
    "1_Persistence_Baseline": "Persistence",
    "2_Linear_Regression": "Ridge",
    "3_Random_Forest": "Random Forest",
    "4_XGBoost": "XGBoost",
}
DIVERGING = [[0.0, "#2a78d6"], [0.5, "#f0efec"], [1.0, "#e34948"]]
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
T2SD, T3SD = 167.653054, 233.67987

CATEGORY_LABELS = {
    "PASS-TRAINS": "Passenger",
    "EXP-TRAINS": "Express",
    "SF-TRAINS": "Superfast",
    "T18-TRAINS": "Vande Bharat (T18)",
    "PRM-TRAINS": "Premium / Special",
    "RAJ-TRAINS": "Rajdhani",
    "SHT-TRAINS": "Shatabdi",
    "GRB-TRAINS": "Garib Rath",
}

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; padding-bottom: 2rem; max-width: 1400px;}
      div[data-testid="stMetric"] {
          border: 1px solid rgba(128,128,128,0.22); border-radius: 10px;
          padding: 12px 14px 10px 14px;
      }
      div[data-testid="stMetricLabel"] p {font-size: 0.82rem; opacity: 0.8;}
      div[data-testid="stMetricValue"] {font-size: 1.55rem;}
      .kpi {border: 1px solid rgba(128,128,128,0.22); border-radius: 10px; padding: 12px 14px; height: 100%;}
      .kpi-l {font-size: 0.80rem; opacity: 0.75; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
      .kpi-v {font-size: 1.4rem; white-space: nowrap; font-weight: 600; line-height: 1.35;}
      .kpi-s {font-size: 0.78rem; opacity: 0.7;}
      .insight {border-left: 3px solid #2a78d6; padding: 6px 12px; margin: 4px 0 10px 0;
                background: rgba(42,120,214,0.06); border-radius: 0 6px 6px 0; font-size: 0.93rem;}
      .note {font-size: 0.82rem; opacity: 0.75;}
      h3 {margin-top: 0.6rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def kpi_row(items) -> None:
    """items: list of (label, value, sub) tuples rendered as quiet stat tiles (no arrows)."""
    cols = st.columns(len(items))
    for col, (label, value, sub) in zip(cols, items):
        col.markdown(
            f"<div class='kpi'><div class='kpi-l'>{label}</div><div class='kpi-v'>{value}</div>"
            f"<div class='kpi-s'>{sub}</div></div>",
            unsafe_allow_html=True,
        )


PRETTY = {
    "pct": "%", "min": "(min)", "gt": ">", "le": "≤", "lt": "<", "2sd": "2σ", "3sd": "3σ", "ge": "≥",
    "obs": "observed", "imp": "imputed", "val": "validation", "btw": "betweenness",
}


def pretty(frame: pd.DataFrame) -> pd.DataFrame:
    def fix(c: str) -> str:
        words = [PRETTY.get(w, w) for w in str(c).split("_")]
        out = " ".join(words)
        return out[:1].upper() + out[1:]
    return frame.rename(columns={c: fix(c) for c in frame.columns})


def table(frame: pd.DataFrame, **kw) -> None:
    st.dataframe(pretty(frame), width="stretch", hide_index=True, **kw)


def insight(text: str) -> None:
    st.markdown(f"<div class='insight'>{text}</div>", unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int = 380, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=48, b=10),
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None),
        showlegend=legend,
        hoverlabel=dict(font_size=12),
        bargap=0.28,
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)", zeroline=False)
    fig.update_traces(marker_line_width=0, selector=dict(type="bar"))
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


# --------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------
JSON_FILES = {
    "keys": "step01_03_5_FINDINGS.json",
    "etl1": "step02_01_ETL_FINDINGS.json",
    "etl2": "step02_02_ETL_findings.json",
    "etl3": "step02_03_ETL_FINDINGS.json",
    "consistency": "step02_04_consistency_FINDINGS.json",
    "alignment": "step02_05_alignment_FINDINGS.json",
    "star": "step03_01_STAR_SCHEMA_FINDINGS.json",
    "views": "step03_02_OLAP_FINDINGS.json",
    "cubes": "step03_03_OLAP_FINDINGS.json",
    "patterns": "step03_04_OLAP_FINDINGS.json",
    "network": "step04_01_NETWORK_FINDINGS.json",
    "dataset": "step04_02_PREDICTION_DATASET_FINDINGS.json",
    "models": "step04_03_MODEL_EVALUATION_FINDINGS.json",
    "eval": "step05_01_MODEL_EVALUATION_FINDINGS.json",
    "persistence": "step05_02_PATTERN_EVALUATION_FINDINGS.json",
    "curves": "step05_04_model_response_curves.json",
    "preproc": "step04_02_preprocessing_artifacts.json",
}


@st.cache_data(show_spinner=False)
def load_findings() -> dict:
    out = {}
    for key, fname in JSON_FILES.items():
        fp = RESULT_DIR / fname
        if fp.exists():
            out[key] = json.loads(fp.read_text())
    return out


@st.cache_resource(show_spinner="Loading the trained models…")
def load_models() -> dict:
    import joblib

    models = {}
    try:
        models["Ridge"] = joblib.load(MODEL_DIR / "step04_03_linear_regression.joblib")
    except Exception as exc:  # pragma: no cover
        models["_errors"] = models.get("_errors", []) + [f"Ridge: {exc}"]
    try:
        models["Random Forest"] = joblib.load(MODEL_DIR / "step04_03_random_forest.joblib")
    except Exception as exc:  # pragma: no cover
        models["_errors"] = models.get("_errors", []) + [f"Random Forest: {exc}"]
    try:
        from xgboost import XGBRegressor

        xgb = XGBRegressor()
        xgb.load_model((MODEL_DIR / "step04_03_xgboost.json").as_posix())
        models["XGBoost"] = xgb
    except Exception as exc:  # pragma: no cover
        models["_errors"] = models.get("_errors", []) + [f"XGBoost: {exc}"]
    return models


@st.cache_data(show_spinner=False)
def load_network_parquets():
    sp = RESULT_DIR / "step04_01_train_period_station_stats.parquet"
    tp = RESULT_DIR / "step04_01_train_period_transition_stats.parquet"
    if sp.exists() and tp.exists():
        try:
            return pd.read_parquet(sp), pd.read_parquet(tp)
        except Exception:
            return None
    return None


F = load_findings()
missing = [JSON_FILES[k] for k in ("cubes", "patterns", "network", "eval") if k not in F]
if missing:
    st.error(
        "Required findings files were not found in `results/`: " + ", ".join(missing)
        + ". Run the analysis notebooks or pull the full repository."
    )
    st.stop()


def df(obj) -> pd.DataFrame:
    return pd.DataFrame(obj)


def label_cat(code: str) -> str:
    return CATEGORY_LABELS.get(code, code)


# Shared frames ---------------------------------------------------------------------------
cat_df = df(F["cubes"]["cube1_train_category"]["by_primary_type"])
cat_df["category"] = cat_df["primary_type_code"].map(label_cat)
zone_df = df(F["cubes"]["cube2_railway_zone"])
zone_df = zone_df[(zone_df["observed_stops"] >= 10_000) & (zone_df["station_zone"] != "UNKNOWN")].copy()
monthly_df = df(F["cubes"]["cube4_temporal_hierarchy"]["monthly"])
monthly_df["period"] = monthly_df["month_name"].str[:3] + " " + monthly_df["year"].astype(str)
monthly_df["short"] = monthly_df["month_name"].str[:3] + " '" + monthly_df["year"].astype(str).str[2:]
overall_eval = df(F["eval"]["overall_evaluation_with_mse"])
overall_eval["Model"] = overall_eval["model"].map(MODEL_NAMES)

# --------------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------------
PAGES = [
    "1 · Overview & data pipeline",
    "2 · Punctuality cubes",
    "3 · Temporal patterns",
    "4 · Route progression",
    "5 · Network bottlenecks",
    "6 · Model evaluation",
    "7 · Live delay predictor",
]
with st.sidebar:
    st.markdown("### 🚆 Indian Railways\n**Punctuality & Network Delay Dynamics**")
    page = st.radio("Section", PAGES, label_visibility="collapsed")
    st.divider()
    st.markdown(
        "<div class='note'>One year of operations<br>8 Feb 2025 – 7 Feb 2026<br><br>"
        "Data Warehousing & Mining project<br>Uplakshya Vishwakarma · M.Tech CSE-AI<br>"
        "IIT (BHU) Varanasi</div>",
        unsafe_allow_html=True,
    )

# ======================================================================================
# PAGE 1 — OVERVIEW
# ======================================================================================
if page == PAGES[0]:
    st.title("Indian Railways punctuality & network delay dynamics")
    st.caption(
        "End-to-end data warehousing, OLAP, graph mining and next-station delay prediction "
        "over 38.4 million raw station-level delay records."
    )

    bench = df(F["views"]["step3_denominator_benchmarks"]).set_index("view_name")
    obs = bench.iloc[1]  # observed, all stops (default OLAP denominator)
    star = F["star"]["fact_delay_summary"]

    kpi_row([
        ("Warehouse fact rows", "36.61 M", "97.18 % observed"),
        ("Trains / stations", "8,720 / 9,027", "365 days of operations"),
        ("Mean delay", f"{obs['mean_delay_min']:.1f} min", f"median {obs['median_delay_min']:.0f} min (observed)"),
        ("On time at origin", "86.6 %", "departure ≤ 15 min"),
        ("On time en route", "47.4 %", "arrival ≤ 15 min"),
        ("Bottleneck stops", "3.47 %", "delay > 167.65 min (2σ)"),
    ])

    insight(
        "Trains leave their origin on time 86.6 % of the time but arrive on time at only 47.4 % of later stops — "
        "delay is <b>created on the line</b>, not at the platform of origin."
    )

    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("From raw logs to model-ready transitions")
        e2 = F["etl2"]
        funnel = pd.DataFrame(
            {
                "stage": [
                    "Raw delay records",
                    "After phantom-journey drop",
                    "After imputation & orphan drop",
                    "After 1,440-min noise cutoff",
                    "Schedule-aligned rows",
                    "Eligible observed transitions",
                ],
                "rows": [
                    38_428_703,
                    e2["step02_02_phantom_journeys"]["rows_after"],
                    e2["step03_5_orphaned_drops"]["final_rows"],
                    36_614_824,
                    35_756_245,
                    33_158_834,
                ],
            }
        )
        fig = go.Figure(
            go.Bar(
                x=funnel["rows"] / 1e6,
                y=funnel["stage"],
                orientation="h",
                marker_color=BLUE,
                text=[f"{v/1e6:.2f} M" for v in funnel["rows"]],
                textposition="outside", cliponaxis=False,
                hovertemplate="%{y}<br>%{x:.3f} M rows<extra></extra>",
            )
        )
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(title="rows (millions)", range=[0, 47])
        show(style_fig(fig, 340, legend=False))

    with right:
        st.subheader("Where the delay happens")
        ctx = bench.iloc[[3, 2]][
            ["mean_delay_min", "median_delay_min", "on_time_le_15m_pct", "total_bottleneck_gt_2sd_pct"]
        ].copy()
        ctx.index = ["Origin departure", "En-route arrival"]
        OVM = {"On time %": "on_time_le_15m_pct", "Mean": "mean_delay_min", "Median": "median_delay_min",
               "Bottleneck %": "total_bottleneck_gt_2sd_pct"}
        OVT = {"On time %": "on time ≤ 15 min (%)", "Mean": "mean delay (min)", "Median": "median delay (min)",
               "Bottleneck %": "stops > 167.65 min (%)"}
        metric = st.segmented_control("Measure", list(OVM), default="On time %", key="ov_metric") or "On time %"
        col = OVM[metric]
        fig = go.Figure(
            go.Bar(
                x=ctx.index, y=ctx[col], marker_color=BLUE, width=0.45,
                text=[f"{v:.1f}" for v in ctx[col]], textposition="outside",
                hovertemplate="%{x}: %{y:.2f}<extra></extra>",
            )
        )
        fig.update_yaxes(title=OVT[metric], range=[0, ctx[col].max() * 1.2])
        show(style_fig(fig, 300, legend=False))

    st.subheader("Data-quality issues that were resolved")
    issues = pd.DataFrame(
        [
            ["Duplicate train numbers", "272 numbers with a base type + PRM overlay tag", "Base type kept; PRM → is_premium_special flag (8,720 trains)"],
            ["Unknown stations", "64 stations missing from the master", "Added as UNKNOWN members (9,027 stations)"],
            ["Ambiguous delay field", "1.04 M non-zero delays at origins, 1.04 M at destinations", "Departure delay at stop 1, arrival delay elsewhere"],
            ["Missing delays", "1,874,271 (4.88 %), clustered by station/date", "Phantom journeys dropped; per-train median at origin; LOCF en route; is_imputed flag"],
            ["Impossible values", "Max 525,592 min (≈ 365 days)", "1,440-min cutoff removes 14,982 rows; SD 229.9 → 66.0 min"],
            ["Meaning of date", "Calendar date or journey start?", "Lagged correlation peaks at k = 0 → journey start date"],
            ["Shifted stop numbers", "18.0 % of rows mismatch the timetable", "stop_alignment_map aligns 97.66 % of rows"],
        ],
        columns=["Issue", "Evidence", "Resolution"],
    )
    st.dataframe(issues, width="stretch", hide_index=True)

    with st.expander("2 × 2 warehouse partition matrix (is_imputed × has_schedule)"):
        pm = df(F["views"]["step2_partition_matrix"])
        table(pm[["partition_segment", "n_rows", "pct_of_warehouse", "n_trains", "n_stations", "mean_delay_min",
                  "median_delay_min", "on_time_15m_pct", "high_risk_plus_critical_pct"]])

# ======================================================================================
# PAGE 2 — PUNCTUALITY CUBES
# ======================================================================================
elif page == PAGES[1]:
    st.title("Punctuality cubes")
    st.caption("OLAP roll-ups over observed records (is_imputed = FALSE). Bottleneck = delay > 167.65 min.")

    MEASURES = {
        "Mean delay (min)": "mean_delay_min",
        "Median delay (min)": "median_delay_min",
        "On-time ≤ 15 min (%)": "on_time_le_15m_pct",
        "Bottleneck > 2σ (%)": "bottleneck_gt_2sd_pct",
        "Critical > 3σ (%)": "critical_gt_3sd_pct",
    }
    measure = st.segmented_control("Measure", list(MEASURES), default="Bottleneck > 2σ (%)", key="cube_m") or "Bottleneck > 2σ (%)"
    mcol = MEASURES[measure]
    higher_is_better = mcol == "on_time_le_15m_pct"

    a, b = st.columns(2)
    with a:
        st.subheader("Train category")
        d = cat_df.sort_values(mcol, ascending=higher_is_better)
        fig = go.Figure(
            go.Bar(
                x=d[mcol], y=d["category"], orientation="h", marker_color=BLUE,
                customdata=np.stack([d["observed_stops"], d["mean_delay_min"], d["on_time_le_15m_pct"], d["bottleneck_gt_2sd_pct"]], axis=1),
                hovertemplate="<b>%{y}</b><br>stops %{customdata[0]:,.0f}<br>mean %{customdata[1]:.1f} min"
                              "<br>on time %{customdata[2]:.1f} %<br>bottleneck %{customdata[3]:.2f} %<extra></extra>",
                text=[f"{v:.1f}" for v in d[mcol]], textposition="outside",
            )
        )
        fig.update_xaxes(title=measure, range=[0, d[mcol].max() * 1.18])
        show(style_fig(fig, 380, legend=False))
    with b:
        st.subheader("Railway zone")
        d = zone_df.sort_values(mcol, ascending=higher_is_better)
        fig = go.Figure(
            go.Bar(
                x=d[mcol], y=d["station_zone"], orientation="h", marker_color=BLUE,
                customdata=np.stack([d["observed_stops"], d["mean_delay_min"], d["on_time_le_15m_pct"]], axis=1),
                hovertemplate="<b>%{y}</b><br>stops %{customdata[0]:,.0f}<br>mean %{customdata[1]:.1f} min"
                              "<br>on time %{customdata[2]:.1f} %<extra></extra>",
            )
        )
        fig.update_xaxes(title=measure)
        show(style_fig(fig, 520, legend=False))

    insight(
        "Short daytime premium services (Shatabdi, Vande Bharat) are the most punctual; holiday/seasonal specials the least "
        "(mean 96 min, 17 % bottleneck). Stress concentrates in SECR, SER, NER and ECR — the eastern coal-and-mineral belt — "
        "while WR, SR and SWR run with an order of magnitude fewer extreme delays."
    )

    st.subheader("Zones: average delay vs. frequency of extreme delay")
    fig = px.scatter(
        zone_df, x="mean_delay_min", y="bottleneck_gt_2sd_pct", size="observed_stops", text="station_zone",
        size_max=42, labels={"mean_delay_min": "mean delay (min)", "bottleneck_gt_2sd_pct": "bottleneck rate (%)"},
        hover_data={"observed_stops": ":,", "on_time_le_15m_pct": ":.1f"},
    )
    below = {"SR", "SWR", "NFR", "NCR", "ECOR", "KR", "SCR"}
    fig.update_traces(marker=dict(color=BLUE, opacity=0.75, line=dict(width=2, color="white")),
                      textposition=["bottom center" if z in below else "top center" for z in zone_df["station_zone"]])
    show(style_fig(fig, 420, legend=False))

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Drill-down: special-tagged trains")
        pr = df(F["cubes"]["cube1_train_category"]["drilldown_premium_special"])
        pr = pr[pr["primary_type_code"].isin(["EXP-TRAINS", "PASS-TRAINS", "SHT-TRAINS"])].copy()
        pr["category"] = pr["primary_type_code"].map(label_cat)
        pr["flag"] = np.where(pr["is_premium_special"], "Special-tagged", "Regular")
        fig = px.bar(
            pr, x="category", y="mean_delay_min", color="flag", barmode="group",
            color_discrete_map={"Regular": BLUE, "Special-tagged": ORANGE},
            category_orders={"flag": ["Regular", "Special-tagged"]},
            labels={"mean_delay_min": "mean delay (min)", "category": ""},
            hover_data={"observed_stops": ":,", "distinct_trains": True, "bottleneck_gt_2sd_pct": ":.2f"},
        )
        show(style_fig(fig, 340))
        st.caption("28 special-tagged Express trains average 95.1 min vs 36.8 min for regular Express trains.")
    with c2:
        st.subheader("Origin departure vs en-route arrival")
        o = df(F["patterns"]["pattern1_arrival_vs_origin"]["origin_by_train_category"])[["primary_type_code", "on_time_le_15m_pct"]]
        r = df(F["patterns"]["pattern1_arrival_vs_origin"]["arrival_by_train_category"])[["primary_type_code", "on_time_le_15m_pct"]]
        m = o.merge(r, on="primary_type_code", suffixes=("_origin", "_arrival"))
        m["category"] = m["primary_type_code"].map(label_cat)
        m = m.sort_values("on_time_le_15m_pct_arrival")
        fig = go.Figure()
        for x0, x1, yv in zip(m["on_time_le_15m_pct_arrival"], m["on_time_le_15m_pct_origin"], m["category"]):
            fig.add_trace(go.Scatter(x=[x0, x1], y=[yv, yv], mode="lines", line=dict(color="rgba(128,128,128,0.45)", width=2),
                                     showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=m["on_time_le_15m_pct_origin"], y=m["category"], mode="markers", name="Origin departure",
                                 marker=dict(color=BLUE, size=11), hovertemplate="%{y}<br>origin %{x:.1f} %<extra></extra>"))
        fig.add_trace(go.Scatter(x=m["on_time_le_15m_pct_arrival"], y=m["category"], mode="markers", name="En-route arrival",
                                 marker=dict(color=ORANGE, size=11), hovertemplate="%{y}<br>arrival %{x:.1f} %<extra></extra>"))
        fig.update_xaxes(title="on time ≤ 15 min (%)", range=[30, 100])
        show(style_fig(fig, 340))

    st.subheader("Zone × category cuboid (dice)")
    t1, t2 = st.tabs(["Worst cells", "Best cells"])
    cols = ["station_zone", "primary_type_code", "observed_stops", "distinct_trains", "mean_delay_min",
            "median_delay_min", "on_time_le_15m_pct", "bottleneck_gt_2sd_pct", "critical_gt_3sd_pct"]
    with t1:
        table(df(F["cubes"]["cube5_cross_dimensional_cuboids"]["top15_zone_x_traintype_by_bottleneck"])[cols])
    with t2:
        table(df(F["cubes"]["cube5_cross_dimensional_cuboids"]["bottom10_zone_x_traintype_by_bottleneck"])[cols])

    st.subheader("Station cube (slice: ≥ 1,000 observed stops per year)")
    sc = F["cubes"]["cube3_station_analysis"]
    s1, s2, s3 = st.tabs(["Major hubs (≥ 25,000 stops)", "Highest bottleneck rate", "Most punctual"])
    with s1:
        table(df(sc["top20_major_hub_bottlenecks_ge25000"]))
    with s2:
        table(df(sc["top20_bottleneck_stations_ge1000"]))
    with s3:
        table(df(sc["top20_punctual_stations_ge1000"]))
        st.caption("Hill and heritage lines (Matheran, Darjeeling) report a constant 0-min delay — likely a tracking artefact.")

# ======================================================================================
# PAGE 3 — TEMPORAL
# ======================================================================================
elif page == PAGES[2]:
    st.title("Temporal patterns")
    st.caption("Dates are journey start dates (verified in Step 2.4).")

    st.subheader("Monthly trajectory")
    sel = st.multiselect("Delay statistics", ["Mean", "Median", "P90"], default=["Mean", "Median"])
    fig = go.Figure()
    for name, col, color in [("Mean", "mean_delay_min", BLUE), ("Median", "median_delay_min", ORANGE), ("P90", "p90_delay_min", AQUA)]:
        if name in sel:
            fig.add_trace(go.Scatter(x=monthly_df["period"], y=monthly_df[col], name=name, mode="lines+markers",
                                     line=dict(color=color, width=2), marker=dict(size=8),
                                     hovertemplate=f"%{{x}}<br>{name} %{{y:.1f}} min<extra></extra>"))
    fig.add_vrect(x0="Nov 2025", x1="Jan 2026", fillcolor="rgba(128,128,128,0.10)", line_width=0,
                  annotation_text="winter fog season", annotation_position="top left")
    fig.update_yaxes(title="delay (min)")
    show(style_fig(fig, 380))

    a, b = st.columns(2)
    with a:
        st.subheader("Monthly bottleneck rate")
        fig = go.Figure(go.Bar(x=monthly_df["short"], y=monthly_df["bottleneck_gt_2sd_pct"], marker_color=BLUE,
                               hovertemplate="%{x}<br>%{y:.2f} %<extra></extra>"))
        fig.update_yaxes(title="stops > 167.65 min (%)")
        show(style_fig(fig, 320, legend=False))
    with b:
        st.subheader("Day of week")
        wk = df(F["cubes"]["cube4_temporal_hierarchy"]["weekday"])
        fig = go.Figure(go.Bar(x=wk["day_name"].str[:3], y=wk["mean_delay_min"], marker_color=BLUE,
                               text=[f"{v:.1f}" for v in wk["mean_delay_min"]], textposition="outside",
                               hovertemplate="%{x}<br>mean %{y:.2f} min<extra></extra>"))
        fig.update_yaxes(title="mean delay (min)", range=[0, 42])
        show(style_fig(fig, 320, legend=False))
        st.caption("Weekday barely matters: 35.2–36.2 min mean, 15–16 min median.")

    st.subheader("Time of day (scheduled arrival hour)")
    di = df(F["patterns"]["pattern3_diurnal_time_of_day"])
    di["block"] = di["time_of_day_block"].str.split("_", n=1).str[1].str.replace(" (", "<br>(", regex=False)
    fig = go.Figure(go.Bar(x=di["block"], y=di["mean_arrival_delay_min"], marker_color=BLUE,
                           customdata=np.stack([di["traffic_share_pct"], di["on_time_le_15m_pct"], di["bottleneck_gt_2sd_pct"]], axis=1),
                           hovertemplate="%{x}<br>mean %{y:.1f} min<br>share %{customdata[0]:.1f} %"
                                         "<br>on time %{customdata[1]:.1f} %<br>bottleneck %{customdata[2]:.2f} %<extra></extra>",
                           text=[f"{v:.1f}" for v in di["mean_arrival_delay_min"]], textposition="outside"))
    fig.update_yaxes(title="mean arrival delay (min)", range=[0, 50])
    show(style_fig(fig, 340, legend=False))
    insight("Early-morning arrivals are the most punctual; delay builds through the day and peaks between midnight and 4 a.m.")

    st.subheader("Do the zone rankings hold in winter?")
    zs = df(F["persistence"]["zone_cross_period_stability"])
    zs = zs[zs["station_zone"] != "UNKNOWN"]
    zm = st.segmented_control("Measure ", ["Bottleneck rate (%)", "Mean delay (min)", "On time ≤ 15 min (%)"],
                              default="Bottleneck rate (%)", key="zs_m") or "Bottleneck rate (%)"
    pre = {"Bottleneck rate (%)": "bottleneck_pct", "Mean delay (min)": "mean_delay", "On time ≤ 15 min (%)": "ontime_pct"}[zm]
    long = zs.melt(id_vars="station_zone", value_vars=[f"train_{pre}", f"val_{pre}", f"test_{pre}"], var_name="split", value_name="value")
    long["split"] = long["split"].str.split("_").str[0].map({"train": "Train", "val": "Validation", "test": "Test"})
    order = zs.sort_values(f"train_{pre}", ascending=False)["station_zone"].tolist()
    fig = px.bar(long, x="station_zone", y="value", color="split", barmode="group", category_orders={"station_zone": order, "split": ["Train", "Validation", "Test"]},
                 color_discrete_map=SPLIT_COLORS, labels={"value": zm, "station_zone": ""})
    show(style_fig(fig, 380))
    st.caption("Train = Feb–Nov 2025 · Validation = 27 Nov 2025 – 1 Jan 2026 · Test = 2 Jan – 7 Feb 2026.")

    with st.expander("Observed vs imputed records by month"):
        om = df(F["patterns"]["pattern5_observed_vs_imputed"]["by_month"])
        om["period"] = om["month_name"].str[:3] + " " + om["year"].astype(str)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=om["period"], y=om["obs_bottleneck_pct"], name="Observed", mode="lines+markers", line=dict(color=BLUE, width=2)))
        fig.add_trace(go.Scatter(x=om["period"], y=om["imp_bottleneck_pct"], name="Imputed (LOCF / median)", mode="lines+markers", line=dict(color=ORANGE, width=2)))
        fig.update_yaxes(title="bottleneck rate (%)")
        show(style_fig(fig, 320))
        st.caption("Imputed rows are kept for continuity but excluded from model targets: reporting blackouts cluster in severe disruptions.")

# ======================================================================================
# PAGE 4 — ROUTE PROGRESSION
# ======================================================================================
elif page == PAGES[3]:
    st.title("Route progression & station amplification")

    rs = df(F["patterns"]["pattern2_route_and_distance_progression"]["by_route_stage"])
    rs["stage"] = rs["route_stage"].str.split("_", n=1).str[1]
    a, b = st.columns(2)
    with a:
        st.subheader("Mean delay along the route")
        colors = [BLUE] * (len(rs) - 1) + [ORANGE]
        fig = go.Figure(go.Bar(x=rs["stage"], y=rs["mean_delay_min"], marker_color=colors,
                               customdata=np.stack([rs["median_delay_min"], rs["early_lt_0m_pct"], rs["observed_stops"]], axis=1),
                               hovertemplate="%{x}<br>mean %{y:.1f} min<br>median %{customdata[0]:.0f} min"
                                             "<br>early %{customdata[1]:.2f} %<br>stops %{customdata[2]:,.0f}<extra></extra>",
                               text=[f"{v:.1f}" for v in rs["mean_delay_min"]], textposition="outside"))
        fig.update_yaxes(title="mean delay (min)", range=[0, 80])
        fig.update_xaxes(tickangle=-25)
        show(style_fig(fig, 380, legend=False))
    with b:
        st.subheader("Share of early arrivals (< 0 min)")
        fig = go.Figure(go.Bar(x=rs["stage"], y=rs["early_lt_0m_pct"], marker_color=colors,
                               text=[f"{v:.2f} %" for v in rs["early_lt_0m_pct"]], textposition="outside",
                               hovertemplate="%{x}<br>%{y:.2f} % early<extra></extra>"))
        fig.update_yaxes(title="early arrivals (%)", range=[0, 38])
        fig.update_xaxes(tickangle=-25)
        show(style_fig(fig, 380, legend=False))
    insight(
        "<b>Terminal schedule padding.</b> Delay climbs from 10.7 min at the origin to 69.7 min after stop 36, then falls to 28.2 min "
        "at the final stop, where 31.6 % of arrivals are early — recovery time is concentrated on the last segment."
    )

    st.subheader("Distance from origin")
    dbd = df(F["patterns"]["pattern2_route_and_distance_progression"]["by_distance_band"])
    dbd["band"] = dbd["distance_band"].str.split("_", n=1).str[1]
    dm = st.segmented_control("Measure  ", ["Mean delay (min)", "Bottleneck rate (%)"], default="Mean delay (min)", key="db_m") or "Mean delay (min)"
    col = "mean_arrival_delay_min" if dm.startswith("Mean") else "bottleneck_gt_2sd_pct"
    fig = go.Figure(go.Bar(x=dbd["band"], y=dbd[col], marker_color=BLUE, text=[f"{v:.1f}" for v in dbd[col]], textposition="outside",
                           hovertemplate="%{x}<br>%{y:.2f}<extra></extra>"))
    fig.update_yaxes(title=dm, range=[0, dbd[col].max() * 1.2])
    show(style_fig(fig, 320, legend=False))

    st.subheader("Stations that add delay vs stations that absorb it")
    amp = df(F["patterns"]["pattern4_station_delay_amplification"]["top20_amplifiers_ge1000"])
    rec = df(F["patterns"]["pattern4_station_delay_amplification"]["top20_recoverers_ge1000"])
    n = st.slider("Stations per side", 5, 20, 10)
    both = pd.concat([amp.head(n), rec.head(n)])
    both["label"] = both["station_name"] + " · " + both["station_full_name"].str.title() + " (" + both["station_zone"] + ")"
    both = both.sort_values("mean_delay_delta_min")
    fig = go.Figure(go.Bar(
        x=both["mean_delay_delta_min"], y=both["label"], orientation="h",
        marker=dict(color=both["mean_delay_delta_min"], colorscale=DIVERGING, cmid=0, cmin=-80, cmax=80,
                    colorbar=dict(title="Δ min", thickness=10)),
        customdata=np.stack([both["observed_transitions"], both["mean_upstream_delay_min"], both["mean_arrival_delay_min"]], axis=1),
        hovertemplate="<b>%{y}</b><br>mean Δ %{x:+.1f} min<br>upstream %{customdata[1]:.1f} → arrival %{customdata[2]:.1f} min"
                      "<br>%{customdata[0]:,.0f} transitions<extra></extra>",
    ))
    fig.update_xaxes(title="mean change in delay at the station (min)", zeroline=True, zerolinecolor="rgba(128,128,128,0.6)")
    show(style_fig(fig, max(380, 26 * len(both)), legend=False))
    st.caption("Δ = delay at the station − delay at the previous stop; observed transitions only, ≥ 1,000 per station.")

# ======================================================================================
# PAGE 5 — NETWORK
# ======================================================================================
elif page == PAGES[4]:
    st.title("Network bottlenecks")
    g = F["network"]["static_schedule_graph"]
    lag = F["network"]["delay_propagation_summary"]["lag_autocorrelations"]
    kpi_row([
        ("Station nodes", f"{g['n_nodes']:,}", "static schedule graph"),
        ("Directed edges", f"{g['n_directed_edges']:,}", "consecutive scheduled stops"),
        ("Largest weak component", f"{g['largest_wcc_nodes']/g['n_nodes']*100:.2f} %", f"{g['weakly_connected_components']} components in total"),
        ("Stop-to-stop correlation", f"r = {lag['r_lag1_all']:.3f}", "delay at k vs k + 1"),
        ("Bottleneck state retained", "99.72 %", "next stop still > 15 min late"),
    ])

    hubs = df(F["network"]["top10_composite_bottleneck_hubs"])
    btw = df(F["network"]["top10_betweenness_stations"])

    a, b = st.columns([1.2, 1])
    with a:
        st.subheader("Structure vs stress: composite bottleneck hubs")
        fig = px.scatter(
            hubs, x="betweenness_centrality", y="train_arr_mean_delay_min", size="total_degree", text="station_name",
            size_max=36, hover_name="station_full_name",
            hover_data={"station_zone": True, "train_obs_stops": ":,", "train_bottleneck_gt_2sd_pct": ":.2f",
                        "composite_hub_stress_score": ":.3f", "station_name": False},
            labels={"betweenness_centrality": "betweenness centrality (schedule graph)",
                    "train_arr_mean_delay_min": "mean arrival delay, training period (min)"},
        )
        pos = {"BSL": "top left", "VZM": "bottom center", "BJU": "top right", "LKO": "bottom center", "CNB": "bottom center"}
        fig.update_traces(marker=dict(color=BLUE, opacity=0.8, line=dict(width=2, color="white")),
                          textposition=[pos.get(s_, "top center") for s_ in hubs["station_name"]])
        show(style_fig(fig, 420, legend=False))
        st.caption("S = 0.35·PR(betweenness) + 0.15·PR(degree) + 0.25·PR(mean delay) + 0.25·PR(bottleneck rate), among hubs with ≥ 20,000 training stops. Bubble size = degree.")
    with b:
        st.subheader("Composite stress score")
        h = hubs.sort_values("composite_hub_stress_score")
        fig = go.Figure(go.Bar(x=h["composite_hub_stress_score"], y=h["station_name"] + " (" + h["station_zone"] + ")", orientation="h",
                               marker_color=BLUE, text=[f"{v:.3f}" for v in h["composite_hub_stress_score"]], textposition="outside",
                               hovertemplate="%{y}<br>score %{x:.3f}<extra></extra>"))
        fig.update_xaxes(range=[0.7, 0.97], title="score (0–1)")
        show(style_fig(fig, 420, legend=False))

    st.subheader("Structural bridges — highest betweenness centrality")
    bb = btw.sort_values("betweenness_centrality")
    fig = go.Figure(go.Bar(x=bb["betweenness_centrality"], y=bb["station_name"] + " · " + bb["station_full_name"].str.title(),
                           orientation="h", marker_color=BLUE, hovertemplate="%{y}<br>%{x:.4f}<extra></extra>"))
    fig.update_xaxes(title="normalised betweenness (exact, Brandes)")
    show(style_fig(fig, 340, legend=False))

    st.subheader("Corridors that amplify delay")
    which = st.segmented_control("Edge set", ["High-volume trunk corridors (≥ 3,000 runs)", "Largest per-run amplification (≥ 500 runs)"],
                                 default="High-volume trunk corridors (≥ 3,000 runs)", key="edge_set") or "High-volume trunk corridors (≥ 3,000 runs)"
    key = "top10_trunk_corridor_amplifiers_ge3000" if which.startswith("High") else "top10_delay_amplifying_transitions_ge500"
    tr = df(F["network"][key])
    tr["corridor"] = tr["from_station"] + " → " + tr["to_station"] + " (" + tr["from_station_zone"] + "→" + tr["to_station_zone"] + ")"
    tr = tr.sort_values("train_mean_delay_delta_min")
    fig = go.Figure(go.Bar(
        x=tr["train_mean_delay_delta_min"], y=tr["corridor"], orientation="h", marker_color=RED,
        customdata=np.stack([tr["train_obs_transition_count"], tr["scheduled_train_count"], tr["train_amp_gt_5m_rate"] * 100,
                             tr["train_from_mean_delay_min"], tr["train_to_mean_delay_min"]], axis=1),
        hovertemplate="<b>%{y}</b><br>mean Δ +%{x:.1f} min<br>%{customdata[3]:.1f} → %{customdata[4]:.1f} min"
                      "<br>%{customdata[0]:,.0f} runs · %{customdata[1]:.0f} trains<br>amplifies > 5 min on %{customdata[2]:.1f} % of runs<extra></extra>",
        text=[f"+{v:.1f}" for v in tr["train_mean_delay_delta_min"]], textposition="outside",
    ))
    fig.update_xaxes(title="mean delay added on the edge (min, training period)", range=[0, tr["train_mean_delay_delta_min"].max() * 1.18])
    show(style_fig(fig, 380, legend=False))

    a, b = st.columns(2)
    with a:
        st.subheader("Once late, trains stay late")
        reg = df(F["network"]["delay_propagation_summary"]["upstream_regimes"])
        reg["regime"] = ["Early / on time<br>≤ 0", "Minor<br>1–15", "Moderate<br>16–60", "Heavy<br>61–168", "Bottleneck<br>> 168"]
        fig = go.Figure(go.Bar(x=reg["regime"], y=reg["downstream_delayed_gt_15m_pct"], marker_color=BLUE,
                               text=[f"{v:.1f} %" for v in reg["downstream_delayed_gt_15m_pct"]], textposition="outside",
                               customdata=reg["n_transitions"], hovertemplate="%{x}<br>%{y:.2f} % still > 15 min late<br>%{customdata:,.0f} transitions<extra></extra>"))
        fig.update_yaxes(title="next stop still > 15 min late (%)", range=[0, 112])
        fig.update_xaxes(title="delay state at the current stop (min)")
        show(style_fig(fig, 340, legend=False))
    with b:
        st.subheader("Autocorrelation along the route")
        kpi_row([
            ("1 stop ahead", f"r = {lag['r_lag1_all']:.3f}", f"{lag['n_lag1_pairs']/1e6:.1f} M pairs"),
            ("2 stops ahead", f"r = {lag['r_lag2_all']:.3f}", f"{lag['n_lag2_pairs']/1e6:.1f} M pairs"),
            ("3 stops ahead", f"r = {lag['r_lag3_all']:.3f}", f"{lag['n_lag3_pairs']/1e6:.1f} M pairs"),
        ])
        st.markdown("")
        insight("Delay at one stop explains about 92 % of the variance of the next stop's delay (r² = 0.92). "
                "This is why persistence is a strong baseline and why any model must be judged against it.")
        st.caption(f"Intermediate stops r = {lag['r_lag1_intermediate']:.3f}; approaching the terminus r = {lag['r_lag1_terminal']:.3f}.")

    st.subheader("Are training-period bottlenecks still bottlenecks in winter?")
    sp = df(F["persistence"]["station_bottleneck_persistence"])
    sp["cohort"] = sp["cohort"].str.split(" ").str[0].str.replace("_", " ")
    table(sp)
    st.caption("Out-of-time correlations (train → validation / test). Patterns are stable but not static: 13 of the top-20 hubs persist into the test period.")

# ======================================================================================
# PAGE 6 — MODEL EVALUATION
# ======================================================================================
elif page == PAGES[5]:
    st.title("Model evaluation")
    ev = F["eval"]
    scope = ev["evaluation_scope"]
    rf = overall_eval.set_index("Model").loc["Random Forest"]
    pers = overall_eval.set_index("Model").loc["Persistence"]
    kpi_row([
        ("Best model", "Random Forest", "80 trees · depth 14 · leaf 20"),
        ("Test MAE", f"{rf['test_MAE']:.2f} min", f"{rf['test_MAE_gain_vs_persistence_pct']:.1f} % lower than persistence"),
        ("Test RMSE", f"{rf['test_RMSE']:.2f} min", f"{rf['test_RMSE_gain_vs_persistence_pct']:.1f} % lower than persistence"),
        ("Test MSE", f"{rf['test_MSE']:.1f} min²", f"{rf['test_MSE_gain_vs_persistence_pct']:.1f} % lower than persistence"),
        ("Test R²", f"{rf['test_R2']:.4f}", f"persistence {pers['test_R2']:.4f}"),
    ])
    st.caption(f"Untouched test set: {scope['n_test_evaluated_full']:,} observed transitions (2 Jan – 7 Feb 2026). "
               f"Validation: {scope['n_val_evaluated_full']:,}.")

    st.subheader("Overall comparison")
    a, b = st.columns([1, 1])
    with a:
        metric = st.segmented_control("Metric", ["MAE", "RMSE", "MSE", "R2"], default="MAE", key="ev_metric") or "MAE"
        long = overall_eval.melt(id_vars="Model", value_vars=[f"train_{metric}", f"val_{metric}", f"test_{metric}"], var_name="split", value_name="value")
        long["split"] = long["split"].str.split("_").str[0].map({"train": "Train", "val": "Validation", "test": "Test"})
        fig = px.bar(long, x="Model", y="value", color="split", barmode="group", color_discrete_map=SPLIT_COLORS,
                     category_orders={"Model": list(MODEL_COLORS), "split": ["Train", "Validation", "Test"]},
                     labels={"value": metric + (" (min)" if metric in ("MAE", "RMSE") else " (min²)" if metric == "MSE" else ""), "Model": ""})
        if metric == "R2":  # differences live in the 3rd decimal: dots, not truncated bars
            fig = px.strip(long, x="Model", y="value", color="split", color_discrete_map=SPLIT_COLORS,
                           category_orders={"Model": list(MODEL_COLORS), "split": ["Train", "Validation", "Test"]},
                           labels={"value": "R²", "Model": ""})
            fig.update_traces(marker=dict(size=13, line=dict(width=2, color="white")), jitter=0)
        show(style_fig(fig, 360))
    with b:
        bands = ev["residual_accuracy_bands"]
        bd = pd.DataFrame([{"Model": MODEL_NAMES[k], "±5 min": v["pct_within_5m"], "±10 min": v["pct_within_10m"], "±15 min": v["pct_within_15m"]} for k, v in bands.items()])
        long = bd.melt(id_vars="Model", var_name="band", value_name="pct")
        fig = px.bar(long, x="band", y="pct", color="Model", barmode="group", color_discrete_map=MODEL_COLORS,
                     category_orders={"Model": list(MODEL_COLORS)}, labels={"pct": "test forecasts within band (%)", "band": ""})
        fig.update_yaxes(range=[0, 100])
        show(style_fig(fig, 360))

    tbl = overall_eval[["Model", "selected_config", "val_MSE", "val_RMSE", "val_MAE", "val_R2", "test_MSE", "test_RMSE", "test_MAE",
                        "test_R2", "test_MAE_gain_vs_persistence_pct", "test_MSE_gain_vs_persistence_pct"]].rename(columns={
        "selected_config": "config", "test_MAE_gain_vs_persistence_pct": "MAE gain %", "test_MSE_gain_vs_persistence_pct": "MSE gain %"})
    table(tbl)

    st.subheader("Where each model wins")
    t1, t2, t3 = st.tabs(["By delay severity", "By stop context", "By train category"])
    with t1:
        sev = df(ev["severity_evaluation_with_mse"])
        sev["Model"] = sev["model"].map(MODEL_NAMES)
        sev["regime"] = sev["target_severity"].map({"normal": "Normal ≤ 167.65", "high_risk": "High-risk 167.65–233.68", "critical": "Critical > 233.68"})
        m = st.segmented_control("Metric ", ["MAE", "RMSE"], default="MAE", key="sev_m") or "MAE"
        fig = px.bar(sev, x="regime", y=m, color="Model", barmode="group", color_discrete_map=MODEL_COLORS,
                     category_orders={"Model": list(MODEL_COLORS), "regime": ["Normal ≤ 167.65", "High-risk 167.65–233.68", "Critical > 233.68"]},
                     hover_data={"n_test_transitions": ":,", "R2": ":.3f", "Mean_Bias": ":.2f"}, labels={m: f"{m} (min)", "regime": "target regime (min)"})
        show(style_fig(fig, 360))
        insight("XGBoost is best on the 95.8 % of normal transitions but collapses in the critical tail (MAE 52.7 vs ≈ 25 min) — see the response-curve audit below.")
    with t2:
        stc = df(ev["stop_context_evaluation_with_mse"])
        stc["Model"] = stc["model"].map(MODEL_NAMES)
        stc["context"] = np.where(stc["stop_context"].str.startswith("Terminal"), "Terminal destination", "Intermediate stop")
        m = st.segmented_control("Metric  ", ["MAE", "RMSE", "Mean_Bias"], default="MAE", key="ctx_m") or "MAE"
        fig = px.bar(stc, x="context", y=m, color="Model", barmode="group", color_discrete_map=MODEL_COLORS,
                     category_orders={"Model": list(MODEL_COLORS)}, hover_data={"n_test_transitions": ":,"}, labels={m: f"{m} (min)", "context": ""})
        show(style_fig(fig, 360))
        st.caption("Persistence over-predicts terminal arrivals by +12.7 min because it ignores terminal slack; the learned models remove the bias.")
    with t3:
        tc = df(ev["train_category_evaluation_with_mse"])
        tc["Model"] = tc["model"].map(MODEL_NAMES)
        tc["category"] = tc["primary_type_code"].map(label_cat)
        pv = tc.pivot_table(index="category", columns="Model", values="MAE")[list(MODEL_COLORS)]
        pv = pv.loc[pv["Persistence"].sort_values().index]
        fig = go.Figure(go.Heatmap(z=pv.values, x=pv.columns, y=pv.index, colorscale=[[i / 6, c] for i, c in enumerate(SEQ_BLUE)],
                                   text=np.round(pv.values, 2), texttemplate="%{text}", hovertemplate="%{y} · %{x}<br>MAE %{z:.2f} min<extra></extra>",
                                   colorbar=dict(title="MAE", thickness=10)))
        show(style_fig(fig, 380, legend=False))

    st.subheader("Response-curve audit — why XGBoost fails on extreme delays")
    if "curves" in F:
        cv = F["curves"]
        scen = st.segmented_control("Scenario", ["intermediate", "terminal"], default="intermediate", key="cv_s") or "intermediate"
        x = cv["curr_delay_min"]
        fig = go.Figure()
        for k, ys in cv["scenarios"][scen].items():
            nm = MODEL_NAMES[k]
            fig.add_trace(go.Scatter(x=x, y=ys, name=nm, mode="lines",
                                     line=dict(color=MODEL_COLORS[nm], width=2.5 if nm == "XGBoost" else 2, dash="dot" if nm == "Persistence" else "solid"),
                                     hovertemplate=f"{nm}<br>current %{{x}} min → forecast %{{y:.1f}} min<extra></extra>"))
        fig.add_vrect(x0=T3SD, x1=1440, fillcolor="rgba(227,73,72,0.07)", line_width=0, annotation_text="critical regime", annotation_position="top left")
        fig.update_xaxes(title="current-stop delay (min); previous = current; other features at training median")
        fig.update_yaxes(title="forecast next-stop delay (min)")
        show(style_fig(fig, 440))
        s = cv.get("summary", {})
        if s:
            insight(f"XGBoost's forecast is <b>constant at {s['xgb_plateau_value_min']:.0f} min</b> for any current delay ≥ "
                    f"{s['xgb_forecast_constant_above_curr_delay_min']:.0f} min: its histogram split finding has only a few bins in the sparse tail. "
                    f"At 1,000 min the gap from the identity line is {s['4_XGBoost_abs_gap_vs_identity_at_1000min']:.0f} min (Ridge "
                    f"{s['2_Linear_Regression_abs_gap_vs_identity_at_1000min']:.0f}, RF {s['3_Random_Forest_abs_gap_vs_identity_at_1000min']:.0f}). "
                    "Fix: model the residual next − current, or raise max_bin.")
    else:
        st.info("Run `python analysis/step05_04_model_response_audit.py` to generate the response curves.")

    a, b = st.columns(2)
    with a:
        st.subheader("Feature importance")
        fi = df(F["models"]["feature_importance_comparison"])
        imp = st.segmented_control("Importance", ["Random Forest", "XGBoost (gain)"], default="Random Forest", key="fi_s") or "Random Forest"
        col = "rf_importance_pct" if imp == "Random Forest" else "xgb_gain_importance_pct"
        fi = fi.sort_values(col)
        fig = go.Figure(go.Scatter(x=fi[col], y=fi["feature_name"], mode="markers+text",
                                   marker=dict(size=11, color=ORANGE if imp == "Random Forest" else AQUA),
                                   text=[f"{v:.2f} %" for v in fi[col]], textposition="middle right",
                                   hovertemplate="%{y}<br>%{x:.2f} %<extra></extra>"))
        fig.update_xaxes(type="log", title="importance (%, log scale)", range=[-1.8, 2.85], showgrid=True, gridcolor="rgba(128,128,128,0.18)")
        show(style_fig(fig, 420, legend=False))
    with b:
        st.subheader("Hyper-parameter tuning (validation)")
        tl = df(ev["hyperparameter_tuning_with_mse"])
        tl["Model"] = tl["model_family"].map(MODEL_NAMES)
        table(tl[["Model", "config_id", "val_MAE", "val_RMSE", "val_MSE", "val_R2", "fit_time_sec", "is_selected"]], height=390)

# ======================================================================================
# PAGE 7 — LIVE PREDICTOR
# ======================================================================================
elif page == PAGES[6]:
    st.title("Live next-station delay predictor")
    st.caption("Runs the four trained models on the 12 engineered features. Priors are smoothed with empirical-Bayes shrinkage (m = 50) exactly as in Step 4.2.")

    if "preproc" not in F:
        st.error("results/step04_02_preprocessing_artifacts.json is missing.")
        st.stop()
    meta = F["preproc"]
    FEATS = meta["feature_columns"]
    SC = meta["standard_scaler_fit_on_train"]
    PRI = meta["global_train_priors"]
    M = meta["shrinkage_smoothing_m"]
    G_ARR, G_DELTA = PRI["global_train_arr_mean_delay"], PRI["global_train_delta_mean"]

    def shrink(mean, n, prior):
        return (n * mean + M * prior) / (n + M)

    models = load_models()
    if models.get("_errors"):
        st.warning("Some models could not be loaded: " + "; ".join(models["_errors"]))

    # ---- next-segment priors: full parquet (if present) or JSON corridor presets ----
    net = load_network_parquets()
    station_lookup = {}
    for r in F["network"]["top10_composite_bottleneck_hubs"] + F["network"]["top10_betweenness_stations"]:
        station_lookup.setdefault(r["station_name"], {}).update({k: v for k, v in r.items() if v is not None})

    presets = {"Typical segment (training medians)": None}
    for r in F["network"]["top10_trunk_corridor_amplifiers_ge3000"] + F["network"]["top10_delay_amplifying_transitions_ge500"]:
        presets[f"{r['from_station']} → {r['to_station']}  (+{r['train_mean_delay_delta_min']:.1f} min/run)"] = r

    st.subheader("1 · Choose the next segment")
    mode = "Corridor preset"
    if net is not None:
        mode = st.radio("Source", ["Corridor preset", "Any station pair (full network stats)"], horizontal=True)

    defaults = {
        "edge": SC["transition_hist_delay_delta_min"]["median"],
        "hist": SC["next_station_hist_arr_delay_min"]["median"],
        "btw": SC["next_station_betweenness"]["median"],
        "seg_km": PRI["median_train_seg_dist_km"],
        "seg_min": PRI["median_train_seg_run_min"],
    }
    seg_desc = "network medians"
    if mode == "Corridor preset":
        choice = st.selectbox("Segment", list(presets))
        r = presets[choice]
        if r is not None:
            defaults["edge"] = shrink(r["train_mean_delay_delta_min"], r["train_obs_transition_count"], G_DELTA)
            to = station_lookup.get(r["to_station"])
            if to and "train_arr_mean_delay_min" in to:
                defaults["hist"] = shrink(to["train_arr_mean_delay_min"], to["train_obs_stops"], G_ARR)
            if to and "betweenness_centrality" in to:
                defaults["btw"] = to["betweenness_centrality"]
            seg_desc = f"{r['from_station']} → {r['to_station']}"
    else:
        st_df, tr_df = net
        fcol = st.columns(2)
        frm = fcol[0].selectbox("From station", sorted(tr_df["from_station"].unique()))
        to_opts = sorted(tr_df.loc[tr_df["from_station"] == frm, "to_station"].unique())
        to = fcol[1].selectbox("To station", to_opts)
        e = tr_df[(tr_df["from_station"] == frm) & (tr_df["to_station"] == to)].iloc[0]
        n_e = float(e.get("train_obs_transition_count", 0) or 0)
        defaults["edge"] = shrink(float(e.get("train_mean_delay_delta_min", 0) or 0), n_e, G_DELTA) if n_e else G_DELTA
        srow = st_df[st_df["station_name"] == to]
        if len(srow):
            srow = srow.iloc[0]
            n_s = float(srow.get("train_obs_arrival_stops", 0) or 0)
            defaults["hist"] = shrink(float(srow.get("train_arr_mean_delay_min", G_ARR) or G_ARR), n_s, G_ARR) if n_s else G_ARR
            defaults["btw"] = float(srow.get("betweenness_centrality", 0) or 0)
        for k_src, k_dst in [("median_sched_running_min", "seg_min"), ("median_segment_distance_km", "seg_km")]:
            if k_src in e and pd.notna(e[k_src]):
                defaults[k_dst] = float(e[k_src])
        seg_desc = f"{frm} → {to}"

    st.subheader("2 · Set the live state and segment features")
    cA, cB, cC = st.columns(3)
    with cA:
        st.markdown("**Live delays**")
        curr = st.number_input("Delay at current stop (min)", -120.0, 1440.0, 45.0, 1.0)
        prev = st.number_input("Delay at previous stop (min)", -120.0, 1440.0, 30.0, 1.0)
        cats = [(c["primary_type_code"], c["is_premium_special"], c["train_category_delay_prior_min"]) for c in meta["train_category_priors"]]
        cat_lbls = [f"{label_cat(code)}{' · special' if sp else ''}" for code, sp, _ in cats]
        ci = st.selectbox("Train category", range(len(cats)), format_func=lambda i: cat_lbls[i], index=1)
        cat_prior = cats[ci][2]
    with cB:
        st.markdown("**Next segment**")
        seg_km = st.number_input("Segment distance (km)", 0.0, 500.0, float(round(defaults["seg_km"], 1)), 1.0)
        seg_min = st.number_input("Scheduled running time (min)", 1.0, 600.0, float(round(defaults["seg_min"], 1)), 1.0)
        hour = st.slider("Scheduled arrival hour", 0, 23, 12)
        progress = st.slider("Route progress of next stop", 0.0, 1.0, 0.55, 0.01)
        terminal = st.toggle("Next stop is the terminal destination", value=False)
    with cC:
        st.markdown(f"**Historical priors** · <span class='note'>{seg_desc}</span>", unsafe_allow_html=True)
        edge = st.number_input("Edge amplification prior (min)", -120.0, 120.0, float(round(defaults["edge"], 2)), 0.5)
        hist = st.number_input("Next-station mean arrival delay prior (min)", -60.0, 400.0, float(round(defaults["hist"], 2)), 1.0)
        btw = st.number_input("Next-station betweenness", 0.0, 0.3, float(round(defaults["btw"], 6)), 0.001, format="%.6f")
        st.caption(f"Category prior: {cat_prior:.2f} min")

    feat = {
        "curr_delay_min": curr,
        "prev_delay_min": prev,
        "prev_delay_delta_min": curr - prev,
        "train_category_delay_prior_min": cat_prior,
        "route_progress_ratio": 1.0 if terminal else progress,
        "is_terminal_arrival": 1.0 if terminal else 0.0,
        "segment_distance_km": seg_km,
        "segment_sched_running_min": seg_min,
        "next_sched_arrival_hour": float(hour),
        "next_station_betweenness": btw,
        "next_station_hist_arr_delay_min": hist,
        "transition_hist_delay_delta_min": edge,
    }
    X = np.array([[feat[c] for c in FEATS]], dtype=np.float32)
    MU = np.array([SC[c]["mean"] for c in FEATS], dtype=np.float32)
    SDV = np.array([SC[c]["std"] for c in FEATS], dtype=np.float32)

    def predict_all(Xm: np.ndarray) -> dict:
        out = {"Persistence": Xm[:, FEATS.index("curr_delay_min")].astype(float)}
        if "Ridge" in models:
            out["Ridge"] = models["Ridge"].predict((Xm - MU) / SDV)
        if "Random Forest" in models:
            out["Random Forest"] = models["Random Forest"].predict(Xm)
        if "XGBoost" in models:
            out["XGBoost"] = models["XGBoost"].predict(Xm).astype(float)
        return {k: np.clip(v, -120, 1440) for k, v in out.items()}

    preds = {k: float(v[0]) for k, v in predict_all(X).items()}

    st.subheader("3 · Forecast for the next stop")
    cols = st.columns(len(preds))
    for col, (name, val) in zip(cols, preds.items()):
        sub = "= current delay" if name == "Persistence" else f"{val - curr:+.1f} min vs current"
        col.markdown(f"<div class='kpi'><div class='kpi-l'>{name}{' ★' if name == 'Random Forest' else ''}</div>"
                     f"<div class='kpi-v'>{val:.1f} min</div><div class='kpi-s'>{sub}</div></div>", unsafe_allow_html=True)
    band = "critical (> 3σ)" if preds.get("Random Forest", curr) > T3SD else "high-risk (2σ–3σ)" if preds.get("Random Forest", curr) > T2SD else "normal"
    st.caption(f"★ best overall model on the test set (MAE 6.68 min). Random Forest forecast falls in the **{band}** regime. "
               "Ridge is the most reliable model above 233.68 min; XGBoost saturates there.")

    st.subheader("What-if: forecast as the current delay changes")
    grid = np.arange(-30, 721, 10, dtype=np.float32)
    Xg = np.repeat(X, len(grid), axis=0)
    i_c, i_p, i_d = FEATS.index("curr_delay_min"), FEATS.index("prev_delay_min"), FEATS.index("prev_delay_delta_min")
    Xg[:, i_c] = grid
    Xg[:, i_p] = grid - (curr - prev)
    Xg[:, i_d] = curr - prev
    pg = predict_all(Xg)
    fig = go.Figure()
    for name, ys in pg.items():
        fig.add_trace(go.Scatter(x=grid, y=ys, name=name, mode="lines",
                                 line=dict(color=MODEL_COLORS[name], width=2.5 if name == "Random Forest" else 2, dash="dot" if name == "Persistence" else "solid"),
                                 hovertemplate=f"{name}<br>current %{{x:.0f}} → %{{y:.1f}} min<extra></extra>"))
    fig.add_vline(x=curr, line_width=1, line_color="rgba(128,128,128,0.7)")
    fig.update_xaxes(title="current-stop delay (min) — upstream momentum held fixed")
    fig.update_yaxes(title="forecast next-stop delay (min)")
    show(style_fig(fig, 400))

    if "Ridge" in models:
        with st.expander("Why? Ridge contribution breakdown (raw-unit coefficients)"):
            coef = models["Ridge"].coef_ / SDV
            intercept = float(models["Ridge"].intercept_ - np.sum(models["Ridge"].coef_ * MU / SDV))
            contrib = pd.DataFrame({"feature": FEATS, "value": X[0], "coefficient": coef, "contribution (min)": coef * X[0]})
            contrib = contrib.reindex(contrib["contribution (min)"].abs().sort_values(ascending=False).index)
            st.dataframe(contrib.round(4), width="stretch", hide_index=True)
            st.caption(f"Intercept {intercept:.3f} min. `prev_delay_delta_min` is curr − prev, so the effective weight on the current delay is "
                       f"{coef[i_c] + coef[i_d]:.3f} and on the previous delay {coef[i_p] - coef[i_d]:.3f}.")
