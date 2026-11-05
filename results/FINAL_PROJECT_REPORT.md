# Indian Railways Network Punctuality, Structural Bottleneck Mining & Delay Prediction
## Final Project Report — Data Warehousing and Data Mining

---

## 1. Executive Summary & Methodology
Passenger train delays across the Indian Railways network cascade across densely interconnected routes, disrupting commuter schedules and compounding congestion at major transit junctions. This project implements an end-to-end **Data Warehousing, Online Analytical Processing (OLAP), Graph Mining, and Machine Learning Regression** pipeline over a full year of operational records (`2025-02-08` to `2026-02-07`, spanning `38,428,703` raw delay logs across `8,992` train entries and `8,963` station records).

The methodology follows a strict five-phase architecture:
1. **Data Quality & Structural Key Resolution (Steps 1.1–1.3):** Out-of-core DuckDB auditing to resolve entity identification ambiguities, composite overlay tags, referential integrity gaps, and global fact-grain uniqueness.
2. **Domain-Aware ETL & Spatiotemporal Cleaning (Steps 2.1–2.5):** Logical Zero-Halt schedule imputation, hypothesis testing to resolve contextual delay semantics, multi-day origin-date-aware Last-Observation-Carried-Forward (LOCF) imputation with explicit provenance tracking (`is_imputed`), domain cutoff noise removal (`<= 1,440` minutes), and stop sequence alignment (`stop_alignment_map.csv`).
3. **ROLAP Star Schema & Multidimensional OLAP (Steps 3.1–3.4):** Construction of a centralized `36,614,824`-row Star Schema (`fact_delay.parquet` joined to `dim_train`, `dim_station`, and `dim_date`), standardized analytical denominators, 1D/2D OLAP cuboids, route-progression analysis, and station-to-station delay amplification auditing.
4. **Network Graph Mining & Leakage-Free Delay Prediction (Steps 4.1–4.3):** Directed schedule topology construction (`8,629` nodes, `31,514` edges), betweenness centrality and composite bottleneck scoring, chronological 80/10/10 partitioning (`33,158,834` eligible observed transitions), 12-feature engineering, and evaluation of four regression families.
5. **Independent Pattern & Model Evaluation (Steps 5.1–5.4):** Full evaluation across MSE, RMSE, MAE, and $R^2$ on the untouched test partition, cross-period bottleneck persistence verification, interactive dashboarding, and synthesis of operational insights.

---

## 2. Dataset Architecture & Preprocessing Pipeline

### 2.1 Source Tables & Structural Key Resolution (Step 1.3)
- **Train Dimension (`dim_train_resolved.csv`):** Raw `train_details.csv` contained `8,992` rows but only `8,720` unique `train_no` keys due to `272` duplicate train numbers carrying both a base service type and a `PRM-TRAINS` special overlay tag. Extracting the base category into `primary_type_code` and creating a boolean `is_premium_special` flag—alongside manual SQL resolution of 4 Namo Bharat Rapid Rail records to `T18-TRAINS`—produced a conflict-free `8,720`-row dimension.
- **Station Dimension (`dim_station_resolved.csv`):** Referential integrity checks against `station_full_names.csv` (`8,963` rows) identified `1` unmatched schedule station and `63` unmatched delay stations. Appending these `64` operational stations with `"UNKNOWN"` metadata placeholders and imputing `94` missing `station_address` fields preserved `100%` foreign-key integrity across `9,027` unique stations.
- **Global Fact Grain Verification:** An out-of-core DuckDB grouping query across all `38,428,703` raw delay records confirmed `0` duplicate `(date, train_no, station_no)` tuples, establishing `(date_id, train_no, station_no)` as the exact atomic grain.

### 2.2 Domain-Guided ETL, Imputation & Noise Reduction (Steps 2.1–2.3)
- **Schedule Zero-Halt Imputation:** Exactly `8,673` missing `arrival_time` values occurred exclusively at origin stops (`station_no = 1`) and `8,673` missing `departure_time` values occurred exclusively at final destination stops in `combined_schedule.csv` (`172,112` rows). Setting origin arrivals equal to scheduled departure and destination departures equal to scheduled arrival eliminated all schedule nulls.
- **Delay Context Resolution:** Out-of-core boundary hypothesis testing confirmed `1,038,047` non-zero delays at origins (`station_no = 1`) and `1,044,006` at destinations, proving that `delay` records **Departure Delay** at `station_no = 1` and **Arrival Delay** at `station_no > 1`.
- **Multi-Day Spatiotemporal Imputation (`is_imputed` Flag):** To address `1,874,271` structured missing delay records (`4.88%`) without injecting synthetic congestion:
  1. Completely dark ("phantom") journeys with zero valid observations across their entire run (grouped by `origin_date`) were dropped.
  2. Missing origin departures (`station_no = 1`) were imputed via per-train historical median (with global origin median fallback).
  3. Mid-route blackouts (`station_no > 1`) were interpolated via Last Observation Carried Forward (`LAST_VALUE ... IGNORE NULLS`) partitioned by `(origin_date, train_no)`.
  4. Leading orphaned nulls (`10,990` rows) were dropped, yielding `36,629,806` complete rows with `is_imputed` tracking every synthetic value.
- **Noise Cutoff vs. Bottleneck Preservation:** All `609,727` negative delay values (down to `-120.0` minutes, representing valid early arrivals) were retained. Impossible transmission errors (up to `525,592.0` minutes / 365 days) were filtered using Indian Railways' 24-hour (`1,440`-minute) operational rescheduling cutoff, removing `14,982` noise rows and finalizing `combined_delay_cleaned.csv` at **`36,614,824` rows** (`35,582,318` observed, `1,032,506` imputed / `2.8199%`).
- **Statistical Bottleneck Thresholds:** On the cleaned dataset (`Mean = 35.60` min, `SD = 66.03` min), positive-tail dispersion boundaries were established at **`> 2 SD = 167.6531` min (High-Risk: `613,515` rows)** and **`> 3 SD = 233.6799` min (Critical: `659,943` rows)**, totaling `1,273,458` bottleneck records (`3.4780%`).

### 2.3 Cross-Table Validation & Stop Alignment Resolution (Steps 2.4–2.5)
- **Journey Start-Date Semantics:** Demeaned origin-to-destination persistence correlations peaked sharply at `k = 0` across 1-day (`r = 0.4715`), 2-day (`r = 0.6131`), 3-day (`r = 0.5262`), and 4-day (`r = 0.4363`) journeys, proving that `date` in `combined_delay.csv` represents the **journey origin start date** for every stop of a run.
- **Stop Sequence Alignment (`stop_alignment_map.csv`):** Direct joins on `(train_no, station_no)` revealed an `18.0147%` (`6,596,040`-row) station name mismatch caused by unscheduled intermediate stops shifting downstream `station_no` indices. Constructing `stop_alignment_map.csv` across `141,646` distinct train-stop pairs cleanly aligned **`35,756,245` rows (`97.6551%`)** (`30,018,784` exact + `5,737,461` shifted) with zero collisions, isolating `858,579` unscheduled-station rows (`2.3449%`) with `has_schedule = FALSE`.

---

## 3. Data Warehouse Architecture & Multidimensional OLAP Findings

### 3.1 Star Schema & Partition Denominators (Steps 3.1–3.2)
The Star Schema (`olap_star_schema` in `step03_01_star_schema.duckdb` and `fact_delay.parquet`) links the `36,614,824`-row central fact table to `dim_train` (`8,720` trains), `dim_station` (`9,027` stations), and `dim_date` (`365` days).
- **Origin vs. En-Route Structural Contrast (`vw_olap_observed`, `35,582,318` rows):** Trains depart their origin (`station_no = 1`, `1,808,957` rows) within 15 minutes of schedule **`86.56%`** of the time (`Mean = 10.73` min, `Median = 1.0` min, `Bottleneck = 1.0436%`). However, cumulative en-route delays reduce intermediate/destination arrival punctuality (`station_no > 1`, `33,773,361` rows) to **`47.36%`** (`Mean = 37.01` min, `Median = 17.0` min, `Bottleneck = 3.6011%`).

### 3.2 Multidimensional OLAP Cubes (Steps 3.3–3.4)
- **Train-Category Cube:** `PASS-TRAINS` (`49.24%` of stops, `Mean = 32.29` min) and `EXP-TRAINS` (`35.43%`, `Mean = 37.03` min) account for `84.66%` of network volume. Daytime intercity premium trains achieve the best punctuality—`SHT-TRAINS` (`76.20%` `<=15m`, `Mean = 14.83` min, `Bottleneck = 1.11%`) and `T18-TRAINS` Vande Bharat (`73.11%` `<=15m`, `Mean = 14.71` min, `Bottleneck = 0.99%`)—whereas `PRM-TRAINS` holiday/seasonal specials suffer severe congestion (`Mean = 96.33` min, `P90 = 316.0` min, `Bottleneck = 17.40%`), followed by `GRB-TRAINS` (`Mean = 51.44` min, `Bottleneck = 7.53%`) and `SF-TRAINS` (`Mean = 43.68` min, `Bottleneck = 6.00%`). Drilling down on `EXP-TRAINS` by `is_premium_special` exposes a `95.11`-min mean delay (`15.19%` bottleneck) on special-tagged Express runs vs. `36.83` min (`3.68%` bottleneck) on regular Express runs.
- **Railway-Zone Cube:** Bottlenecks concentrate heavily in coal, mineral, and dense eastern/northern trunk zones: `SECR` (`7.74%` bottleneck, `Mean = 57.82` min, `On-Time = 35.97%`), `SER` (`7.71%` bottleneck, `Mean = 53.93` min), `NER` (`6.43%`), `ECR` (`5.94%`, `On-Time = 34.79%`), and `WCR` (`5.22%`). Conversely, `SWR` (`0.58%` bottleneck), `SR` (`0.74%`), `WR` (`0.89%`, `On-Time = 72.77%`), and `NWR` (`1.02%`) operate with minimal extreme congestion.
- **Temporal & Diurnal Cubes:** Network performance peaks in Q3 monsoon/late-summer (`Aug 2025 Mean = 31.02` min, `Bottleneck = 2.56%`) and deteriorates sharply during North Indian winter fog (`Dec 2025 Mean = 44.97` min, `Bottleneck = 5.41%`; `Jan 2026 Mean = 40.19` min). Diurnally, arrivals between `04:00–07:59` are most punctual (`Mean = 30.38` min, `On-Time = 54.39%`), whereas Late Night arrivals (`00:00–03:59`) suffer peak knock-on delays (`Mean = 43.56` min, `Bottleneck = 4.97%`).
- **Route Progression & Terminal Schedule Padding Discovery:** Mean delay climbs monotonically from `10.73` min at Stop 1 to `69.65` min (`Bottleneck = 10.98%`) at Stops `36+`. However, at the final terminal destination (`1,849,874` stops), mean delay drops to `28.15` min (`Median = 6.0` min, `On-Time = 64.38%`), and **`31.61%` of all terminal arrivals are early (`delay < 0`)** compared to `0.00%` at intermediate stops—proving that Indian Railways concentrates schedule slack buffers immediately before the destination terminal.

---

## 4. Railway Graph Mining & Network Bottleneck Findings (Step 4.1)

Constructing the directed schedule graph (`8,629` nodes, `31,514` directed edges; `99.64%` in the largest WCC) and combining static topological centrality with training-period (`2025-02-08` to `2025-11-26`) empirical stress revealed:
1. **Structural Bridge Hubs (Betweenness Centrality):** `DDU` Pt Deen Dayal Upadhyay Jn (`0.2674`), `BZA` Vijayawada Jn (`0.2499`), `NGP` Nagpur (`0.2310`), `CNB` Kanpur Central (`0.2055`), `NDLS` New Delhi (`0.2010`), and `BPQ` Balharshah (`0.1959`) serve as the primary inter-regional bridge junctions of the Indian rail network.
2. **Composite Structural-Operational Bottleneck Hubs (`>= 20,000` Training Stops):** Ranking hubs jointly by centrality, degree, mean arrival delay, and bottleneck rate identifies `KGP` Kharagpur Jn (`SER`, score `0.9382`, mean delay `94.44` min, bottleneck `15.36%`), `NGP` Nagpur (`CR`, `0.9241`), `TATA` Tatanagar Jn (`SER`, `0.9141`, mean delay `110.19` min, bottleneck `17.62%`), `BJU` Barauni Jn (`ECR`, `0.9065`), and `BSL` Bhusaval Jn (`CR`, `0.9062`) as the top systemic congestion nuclei.
3. **Transition Delay Amplification vs. Recovery:** Major trunk corridors actively inject delay—led by `KGP -> SRC` (`+39.83` min across `4,027` runs), `BVRT -> AKVD` (`+25.30` min), `NDLS -> CNB` (`+21.45` min), `BLS -> KGP` (`+21.05` min), and `HWH -> BWN` (`+19.61` min, amplifying `>5` min on `94.69%` of runs)—while padded terminal approaches (`COA` `-79.68` min, `PURI` `-41.77` min, `SHM` `-39.10` min) absorb upstream delay.
4. **Spatial-Temporal Autocorrelation & Regime Retention:** Consecutive stop delays exhibit `r = 0.9600` 1-step autocorrelation (`r = 0.9657` across intermediate stops vs. `r = 0.9068` at terminals; `r = 0.9375` at Lag 2; `r = 0.9162` at Lag 3). Trains entering a bottleneck state (`> 167.65` min) remain delayed `> 15` min at the next station `99.72%` of the time.

---

## 5. Regression Modelling & Independent Evaluation (Steps 4.2, 4.3 & 5.1)

### 5.1 Leakage-Free Feature Engineering & Chronological Split
Across `33,158,834` contiguous observed scheduled transitions (`has_schedule = TRUE`, `curr_is_imputed = FALSE`, `next_is_imputed = FALSE`), an 80/10/10 chronological date split was enforced:
- **Training (`2025-02-08` to `2025-11-26`, 292 days):** `26,528,080` transitions (`Target Mean = 35.37` min, `Bottleneck = 3.28%`).
- **Validation (`2025-11-27` to `2026-01-01`, 36 days):** `3,278,765` transitions (`Target Mean = 46.09` min, `Bottleneck = 5.46%` — winter fog surge).
- **Independent Test (`2026-01-02` to `2026-02-07`, 37 days):** `3,351,989` transitions (`Target Mean = 41.16` min, `Bottleneck = 4.18%`).

Twelve strictly prediction-time features were engineered using empirical Bayes smoothed priors (`m = 50`) fitted solely on the Training partition: `curr_delay_min` (`r = +0.9600`), `prev_delay_min` (`r = +0.9377`), `next_station_hist_arr_delay_min` (`r = +0.3374`), `prev_delay_delta_min` (`r = +0.1717`), `transition_hist_delay_delta_min` (`r = +0.1140`), `route_progress_ratio` (`r = +0.1062`), `train_category_delay_prior_min` (`r = +0.0946`), `segment_distance_km` (`r = +0.0726`), `segment_sched_running_min` (`r = +0.0615`), `is_terminal_arrival` (`r = -0.0357`), `next_sched_arrival_hour` (`r = +0.0332`), and `next_station_betweenness` (`r = +0.0265`).

### 5.2 Independent Test Set Results (`n = 3,351,989` Observed Transitions)

| Model | Validation MSE ($\text{min}^2$) | Validation RMSE (min) | Validation MAE (min) | Validation $R^2$ | Test MSE ($\text{min}^2$) | Test RMSE (min) | Test MAE (min) | Test $R^2$ | Test MAE vs. Baseline | Test MSE vs. Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Persistence Baseline** | 414.0696 | 20.3487 | 8.4207 | 0.935176 | 408.1370 | 20.2024 | 8.2098 | 0.910606 | 0.00% | 0.00% |
| **2. Linear Regression (Ridge $\alpha=10$)** | 316.9539 | 17.8032 | 7.1502 | 0.950380 | 324.3349 | 18.0093 | 7.1739 | 0.928961 | -12.62% | -20.53% |
| **3. Random Forest (`RF_d14_leaf20`)** | **302.6730** | **17.3975** | **6.6395** | **0.952615** | **309.2639** | **17.5859** | **6.6842** | **0.932263** | **-18.58%** | **-24.23%** |
| **4. XGBoost (`XGB_d10_lr0.05`)** | 503.9980 | 22.4499 | 7.3643 | 0.921097 | 429.8739 | 20.7334 | 7.0557 | 0.905846 | -14.06% | +5.33% |

### 5.3 Key Diagnostic Insights
- **Overall Winner:** **Random Forest (`RF_d14_leaf20`)** achieves the best overall generalization on the untouched test set, reducing **MSE by `24.23%`** (`309.26` vs. `408.14` $\text{min}^2$), **RMSE by `12.95%`** (`17.59` vs. `20.20` min), and **MAE by `18.58%`** (`6.68` vs. `8.21` min), while raising $R^2$ to **`0.932263`** (`89.45%` of predictions within $\pm 15$ min vs. `84.41%` for Persistence).
- **Terminal Schedule Padding Correction:** At terminal destinations (`181,678` test rows), Persistence suffers a massive `+12.75`-min over-prediction bias (`MAE = 20.55` min, `MSE = 1,701.18` $\text{min}^2$) because it ignores terminal slack recovery. Incorporating `is_terminal_arrival`, `transition_hist_delay_delta_min`, and `segment_sched_running_min` reduces terminal MAE by **`>40%`** (`12.07` min for XGBoost, `12.25` min for Random Forest with near-zero bias `+0.14` min).
- **Severity Regime Trade-off:** In the Normal regime (`<= 167.65` min, `95.82%` of test data), **XGBoost** achieves the lowest MAE (`5.92` min) and MSE (`136.55` $\text{min}^2$, `RMSE = 11.69` min). However, in the Critical regime (`> 233.68` min, `2.08%` of test data), XGBoost's histogram quantile binning (`tree_method = 'hist'`) under-fits the sparse extreme upper tail (`MAE = 52.68` min, `RMSE = 116.99` min), whereas **Ridge Regression** (`MAE = 24.79` min, `RMSE = 84.87` min) and **Random Forest** (`MAE = 25.14` min, `RMSE = 85.18` min) extrapolate smoothly across severe disruptions.

---

## 6. Pattern Persistence, Limitations, Imputation Effects & Future Scope (Step 5.2)

1. **Cross-Period Bottleneck Persistence:** Station mean arrival delays and bottleneck rates exhibit `r > 0.98` correlation between the Training window and full-year operations, and the top structural-operational bottleneck zones (`SECR`, `SER`, `NER`, `ECR`, `WCR`) and hubs (`TATA`, `KGP`, `G`, `GKP`, `BSP`, `NGP`, `BSL`) consistently occupy the top stress tiers across Training, Validation, and Testing splits despite the seasonal winter fog shift.
2. **Missing Data & Spatiotemporal Imputation Effects:**
   - **Origin Median Smoothing:** Imputed origin rows (`42,392` rows / `2.29%` of origins) exhibit lower variance (`SD = 24.63` min vs. `38.78` min observed) and lower bottleneck rates (`0.125%` vs. `1.044%` observed) because per-train median replacement compresses extreme origin departure spikes.
   - **En-Route LOCF Tail Preservation:** Imputed intermediate/destination rows (`990,114` rows / `2.85%` of arrivals) exhibit higher dispersion (`SD = 72.94` min vs. `66.73` min observed) and elevated critical delay rates (`2.23%` vs. `1.86%` observed; peaking at `8.04%` bottleneck rate in Dec 2025 vs. `5.41%` observed) because reporting blackouts occur disproportionately during severe disruption cascades. Tracking `is_imputed` and excluding imputed records from prediction training/evaluation prevented synthetic zero-change transitions (`curr_delay == next_delay`) from inflating model accuracy.
3. **Limitations & Future Scope:**
   - **Single-Year Weather Horizon:** Because the dataset spans exactly one annual cycle (`Feb 2025 – Feb 2026`), chronological validation and testing fall inside the winter fog regime (`Dec 2025 – Feb 2026`). Incorporating multi-year archives and exogenous meteorological telemetry (visibility, rainfall, temperature) would improve out-of-regime calibration.
   - **Multi-Horizon Rollout & Dispatch Optimization:** Extending the 1-step transition regressor into a multi-step recursive trajectory forecaster with uncertainty intervals (quantile regression) will enable real-time platform allocation and crew connection alerts.