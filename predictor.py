# to run: streamlit run predictor.py


import json
from pathlib import Path
import duckdb
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBRegressor

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Indian Railways Multi-Station Delay Predictor",
    page_icon="🚆",
    layout="centered",
)

# --- PATHS & ARTIFACT LOADING ---
BASE_DIR = Path(__file__).parent
RESULT_DIR = BASE_DIR / "results"
MODEL_DIR = RESULT_DIR / "models"


@st.cache_resource
def load_artifacts():
  with open(RESULT_DIR / "step04_02_preprocessing_artifacts.json", "r") as f:
    preproc_meta = json.load(f)

  rf_model = joblib.load(MODEL_DIR / "step04_03_random_forest.joblib")
  lin_model = joblib.load(MODEL_DIR / "step04_03_linear_regression.joblib")

  xgb_model = XGBRegressor()
  xgb_model.load_model((MODEL_DIR / "step04_03_xgboost.json").as_posix())

  con = duckdb.connect(database=":memory:")
  con.execute(f"""
        CREATE VIEW dim_train AS SELECT * FROM read_csv_auto('{RESULT_DIR / "dim_train.csv"}');
        CREATE VIEW dim_station AS SELECT * FROM read_csv_auto('{RESULT_DIR / "dim_station.csv"}');
        CREATE VIEW schedule AS SELECT * FROM read_csv_auto('{RESULT_DIR / "combined_schedule_imputed.csv"}');
        CREATE VIEW station_stats AS SELECT * FROM read_parquet('{RESULT_DIR / "step04_01_train_period_station_stats.parquet"}');
        CREATE VIEW transition_stats AS SELECT * FROM read_parquet('{RESULT_DIR / "step04_01_train_period_transition_stats.parquet"}');
    """)
  return preproc_meta, rf_model, lin_model, xgb_model, con


preproc_meta, rf_model, lin_model, xgb_model, con = load_artifacts()
FEATURE_COLS = preproc_meta["feature_columns"]
GLOBAL_MEAN = preproc_meta["global_train_priors"]["global_train_arr_mean_delay"]

# --- UI HEADER ---
st.title("🚆 Indian Railways Multi-Station Delay Predictor")
st.markdown(
    "Enter a train number, current position, live delays, and **any arbitrary"
    " target station** down the route to forecast delays via recursive"
    " multi-step rollout."
)

# --- USER INPUT FORM ---
with st.form("prediction_form"):
  model_choice = st.selectbox(
      "Select Prediction Model",
      [
          "Random Forest Regressor (Best Overall - Test R²: 0.932)",
          "XGBoost Regressor",
          "Linear Regression (Ridge)",
          "Persistence Baseline",
      ],
  )

  col1, col2 = st.columns(2)
  with col1:
    train_no_input = st.text_input(
        "Train Number (e.g., 12951, 12656)", value="12951"
    )
    journey_date = st.date_input("Journey Date", value=pd.to_datetime("2026-02-01"))
  with col2:
    curr_station = st.text_input(
        "Current Station Code (e.g., NDLS, KYN)", value="NDLS"
    )
    target_station = st.text_input(
        "Target Station Code (e.g., CNB, BSL, HWH)", value="CNB"
    )

  st.subheader("Real-Time Operational State at Current Station")
  col3, col4 = st.columns(2)
  with col3:
    curr_delay = st.number_input(
        "Delay at Current Station (minutes)", value=15.0, step=1.0
    )
  with col4:
    prev_delay = st.number_input(
        "Delay at Previous Station (minutes)", value=10.0, step=1.0
    )

  submitted = st.form_submit_button("Calculate Multi-Step Prediction")

# --- PREDICTION & ROUTING LOGIC ---
if submitted:
  with st.spinner(
      "Fetching route sequence and executing recursive trajectory rollout..."
  ):
    # 1. Fetch train schedule sequence for this train number
    route_df = con.execute(f"""
            SELECT s.station_no, s.station_name, d.station_full_name, s.distance_from_origin, 
                   s.arrival_day, s.arrival_time, s.departure_time
            FROM schedule s
            LEFT JOIN dim_station d ON s.station_name = d.station_name
            WHERE s.train_no = {train_no_input}
            ORDER BY s.station_no ASC
        """).df()

    if route_df.empty:
      st.error(
          f"Train number {train_no_input} not found in schedule database."
      )
    else:
      curr_matches = route_df[route_df["station_name"] == curr_station]
      target_matches = route_df[route_df["station_name"] == target_station]

      if curr_matches.empty or target_matches.empty:
        st.error(
            "Could not find either the current or target station code in this"
            " train's schedule sequence."
        )
      else:
        curr_idx = curr_matches.iloc[0]["station_no"]
        target_idx = target_matches.iloc[0]["station_no"]

        if target_idx <= curr_idx:
          st.error(
              "Target station must appear *after* the current station along the"
              " train's route sequence!"
          )
        else:
          sub_route = route_df[
              (route_df["station_no"] >= curr_idx)
              & (route_df["station_no"] <= target_idx)
          ].reset_index(drop=True)
          max_station_no = route_df["station_no"].max()

          train_info = con.execute(f"""
                        SELECT primary_type_code, is_premium_special 
                        FROM dim_train WHERE train_no = {train_no_input} LIMIT 1
                    """).fetchone()
          type_code, is_prem = (
              train_info if train_info else ("EXP-TRAINS", False)
          )

          cat_prior = GLOBAL_MEAN
          for cp in preproc_meta["train_category_priors"]:
            if (
                cp["primary_type_code"] == type_code
                and cp["is_premium_special"] == is_prem
            ):
              cat_prior = cp["train_category_delay_prior_min"]
              break

          active_curr_delay = float(curr_delay)
          active_prev_delay = float(prev_delay)

          step_history = []

          for i in range(len(sub_route) - 1):
            s_from = sub_route.iloc[i]
            s_to = sub_route.iloc[i + 1]

            f_name = s_from["station_name"]
            t_name = s_to["station_name"]
            t_full_name = s_to["station_full_name"] or t_name
            t_no = int(s_to["station_no"])

            station_info = con.execute(f"""
                            SELECT betweenness_centrality, train_arr_mean_delay_min, train_obs_arrival_stops
                            FROM station_stats WHERE station_name = '{t_name}' LIMIT 1
                        """).fetchone()

            if station_info:
              nb = float(station_info[0] or 0.0)
              ham = float(station_info[1] or GLOBAL_MEAN)
              obs = float(station_info[2] or 0.0)
              ns_hist = (obs * ham + 50.0 * GLOBAL_MEAN) / (obs + 50.0)
            else:
              nb, ns_hist = 0.0, GLOBAL_MEAN

            edge_info = con.execute(f"""
                            SELECT train_mean_delay_delta_min, train_obs_transition_count
                            FROM transition_stats WHERE from_station = '{f_name}' AND to_station = '{t_name}' LIMIT 1
                        """).fetchone()

            if edge_info:
              ed = float(edge_info[0] or 0.0)
              ec = float(edge_info[1] or 0.0)
              trans_delta = (ec * ed + 50.0 * 0.8528) / (ec + 50.0)
            else:
              trans_delta = 0.8528

            seg_dist = float(
                s_to["distance_from_origin"] - s_from["distance_from_origin"]
            )
            is_term = 1.0 if t_no == max_station_no else 0.0

            feature_dict = {
                "curr_delay_min": active_curr_delay,
                "prev_delay_min": active_prev_delay,
                "prev_delay_delta_min": float(
                    active_curr_delay - active_prev_delay
                ),
                "train_category_delay_prior_min": float(cat_prior),
                "route_progress_ratio": float(t_no / max_station_no),
                "is_terminal_arrival": is_term,
                "segment_distance_km": max(seg_dist, 0.0),
                "segment_sched_running_min": 30.0,
                "next_sched_arrival_hour": float(journey_date.day % 24),
                "next_station_betweenness": nb,
                "next_station_hist_arr_delay_min": ns_hist,
                "transition_hist_delay_delta_min": trans_delta,
            }

            X_input = np.array(
                [[feature_dict[col] for col in FEATURE_COLS]], dtype=np.float32
            )

            if model_choice.startswith("Random Forest"):
              raw_pred = rf_model.predict(X_input)[0]
            elif model_choice.startswith("XGBoost"):
              raw_pred = xgb_model.predict(X_input)[0]
            elif model_choice.startswith("Linear Regression"):
              train_means = np.array(
                  [
                      preproc_meta["standard_scaler_fit_on_train"][c]["mean"]
                      for c in FEATURE_COLS
                  ],
                  dtype=np.float32,
              )
              train_stds = np.array(
                  [
                      preproc_meta["standard_scaler_fit_on_train"][c]["std"]
                      for c, _ in preproc_meta[
                          "standard_scaler_fit_on_train"
                      ].items()
                  ],
                  dtype=np.float32,
              )
              X_std = (X_input - train_means) / train_stds
              raw_pred = lin_model.predict(X_std)[0]
            else:
              raw_pred = active_curr_delay

            predicted_delay = float(np.clip(raw_pred, -120.0, 1440.0))

            step_history.append({
                "Station Code": t_name,
                "Station Name": t_full_name,
                "Stop #": t_no,
                "Predicted Arrival Delay (mins)": round(predicted_delay, 1),
            })

            active_prev_delay = active_curr_delay
            active_curr_delay = predicted_delay

          st.success(
              f"Successfully calculated trajectory from {curr_station} to"
              f" {target_station} across {len(step_history)} segment steps!"
          )

          final_target_delay = step_history[-1]["Predicted Arrival Delay (mins)"]

          m1, m2, m3 = st.columns(3)
          m1.metric(
              f"Delay at Target ({target_station})",
              f"{final_target_delay:.1f} mins",
          )
          m2.metric("Starting Delay", f"{curr_delay:.1f} mins")
          m3.metric(
              "Net Trajectory Change",
              f"{final_target_delay - curr_delay:+.1f} mins",
          )

          st.markdown("### Step-by-Step Trajectory Rollout Breakdown")
          st.dataframe(pd.DataFrame(step_history), use_container_width=True)