"""
Build the standalone executive dashboard: dashboard/indian_railways_dashboard.html
=================================================================================

The page is a single self-contained HTML file (data embedded as JSON, charts drawn
with Plotly.js) that opens in any browser — no Python or server needed to view it.

    python dashboard/build_dashboard_html.py            # Plotly.js loaded from the CDN (small file)
    python dashboard/build_dashboard_html.py --inline   # embed Plotly.js too (works fully offline, ~4.5 MB)

Inputs: the committed JSON findings in results/ (and results/step05_04_model_response_curves.json,
produced by analysis/step05_04_model_response_audit.py).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULT_DIR = BASE_DIR / "results"
OUT = Path(__file__).resolve().parent / "indian_railways_dashboard.html"
PLOTLY_CDN = "https://cdn.plot.ly/plotly-4.1.1.min.js"


def j(name: str):
    return json.loads((RESULT_DIR / name).read_text())


def keep(rows, cols):
    return [{c: r.get(c) for c in cols} for r in rows]


def build_data() -> dict:
    views = j("step03_02_OLAP_FINDINGS.json")
    cubes = j("step03_03_OLAP_FINDINGS.json")
    pat = j("step03_04_OLAP_FINDINGS.json")
    net = j("step04_01_NETWORK_FINDINGS.json")
    ds = j("step04_02_PREDICTION_DATASET_FINDINGS.json")
    mdl = j("step04_03_MODEL_EVALUATION_FINDINGS.json")
    ev = j("step05_01_MODEL_EVALUATION_FINDINGS.json")
    per = j("step05_02_PATTERN_EVALUATION_FINDINGS.json")
    pre = j("step04_02_preprocessing_artifacts.json")
    e2 = j("step02_02_ETL_findings.json")
    curves_path = RESULT_DIR / "step05_04_model_response_curves.json"
    curves = json.loads(curves_path.read_text()) if curves_path.exists() else None

    d = {
        "bench": views["step3_denominator_benchmarks"],
        "funnel": [
            ["Raw delay records", 38_428_703],
            ["After phantom-journey drop", e2["step02_02_phantom_journeys"]["rows_after"]],
            ["After imputation & orphan drop", e2["step03_5_orphaned_drops"]["final_rows"]],
            ["After 1,440-min noise cutoff", 36_614_824],
            ["Schedule-aligned rows", 35_756_245],
            ["Eligible observed transitions", 33_158_834],
        ],
        "category": cubes["cube1_train_category"]["by_primary_type"],
        "premium": cubes["cube1_train_category"]["drilldown_premium_special"],
        "zone": [z for z in cubes["cube2_railway_zone"] if z["observed_stops"] >= 10000 and z["station_zone"] != "UNKNOWN"],
        "cuboid_top": cubes["cube5_cross_dimensional_cuboids"]["top15_zone_x_traintype_by_bottleneck"],
        "cuboid_bottom": cubes["cube5_cross_dimensional_cuboids"]["bottom10_zone_x_traintype_by_bottleneck"],
        "hubs25k": cubes["cube3_station_analysis"]["top20_major_hub_bottlenecks_ge25000"],
        "stations_top": cubes["cube3_station_analysis"]["top20_bottleneck_stations_ge1000"],
        "monthly": cubes["cube4_temporal_hierarchy"]["monthly"],
        "weekday": cubes["cube4_temporal_hierarchy"]["weekday"],
        "origin_cat": pat["pattern1_arrival_vs_origin"]["origin_by_train_category"],
        "arrival_cat": pat["pattern1_arrival_vs_origin"]["arrival_by_train_category"],
        "route": pat["pattern2_route_and_distance_progression"]["by_route_stage"],
        "distance": pat["pattern2_route_and_distance_progression"]["by_distance_band"],
        "diurnal": pat["pattern3_diurnal_time_of_day"],
        "amp": pat["pattern4_station_delay_amplification"]["top20_amplifiers_ge1000"],
        "rec": pat["pattern4_station_delay_amplification"]["top20_recoverers_ge1000"],
        "obs_imp_month": pat["pattern5_observed_vs_imputed"]["by_month"],
        "graph": net["static_schedule_graph"],
        "betweenness": net["top10_betweenness_stations"],
        "hubs": net["top10_composite_bottleneck_hubs"],
        "trunk": net["top10_trunk_corridor_amplifiers_ge3000"],
        "edges": net["top10_delay_amplifying_transitions_ge500"],
        "regimes": net["delay_propagation_summary"]["upstream_regimes"],
        "lags": net["delay_propagation_summary"]["lag_autocorrelations"],
        "splits": ds["chronological_split_summary"],
        "features": keep(ds["feature_statistics_and_correlations_train"], ["feature_name", "train_mean", "train_std", "train_median", "corr_with_target_train"]),
        "overall": ev["overall_evaluation_with_mse"],
        "bands": ev["residual_accuracy_bands"],
        "severity": ev["severity_evaluation_with_mse"],
        "context": ev["stop_context_evaluation_with_mse"],
        "cat_eval": ev["train_category_evaluation_with_mse"],
        "tuning": ev["hyperparameter_tuning_with_mse"],
        "importance": mdl["feature_importance_comparison"],
        "persistence": per["station_bottleneck_persistence"],
        "zone_stability": [z for z in per["zone_cross_period_stability"] if z["station_zone"] != "UNKNOWN"],
        "cat_priors": pre["train_category_priors"],
        "scaler": pre["standard_scaler_fit_on_train"],
        "priors": pre["global_train_priors"],
        "m": pre["shrinkage_smoothing_m"],
    }
    if curves:
        d["curves"] = {k: curves[k] for k in ("curr_delay_min", "scenarios", "summary", "ridge_raw_unit_equation")}
    return d


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Indian Railways Delay Analytics</title>
__PLOTLY__
<style>
:root{
  color-scheme: light;
  --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,0.10); --wash:rgba(42,120,214,0.07);
  --blue:#2a78d6; --orange:#eb6834; --aqua:#1baf7a; --red:#e34948; --gray:#898781;
  --s-train:#86b6ef; --s-val:#2a78d6; --s-test:#104281; --mid:#f0efec;
  --seq1:#cde2fb; --seq2:#86b6ef; --seq3:#3987e5; --seq4:#1c5cab; --seq5:#0d366b;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    color-scheme: dark;
    --page:#0d0d0d; --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10); --wash:rgba(57,135,229,0.12);
    --blue:#3987e5; --orange:#d95926; --aqua:#199e70; --red:#e66767; --gray:#898781;
    --s-train:#184f95; --s-val:#3987e5; --s-test:#9ec5f4; --mid:#383835;
    --seq1:#104281; --seq2:#1c5cab; --seq3:#2a78d6; --seq4:#6da7ec; --seq5:#cde2fb;
  }
}
:root[data-theme="dark"]{
    color-scheme: dark;
    --page:#0d0d0d; --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10); --wash:rgba(57,135,229,0.12);
    --blue:#3987e5; --orange:#d95926; --aqua:#199e70; --red:#e66767; --gray:#898781;
    --s-train:#184f95; --s-val:#3987e5; --s-test:#9ec5f4; --mid:#383835;
    --seq1:#104281; --seq2:#1c5cab; --seq3:#2a78d6; --seq4:#6da7ec; --seq5:#cde2fb;
}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
header{padding:22px 28px 0;max-width:1320px;margin:0 auto}
.top{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;flex-wrap:wrap}
h1{font-size:24px;margin:0 0 4px;letter-spacing:-0.01em}
.sub{color:var(--ink2);font-size:13px}
.who{color:var(--muted);font-size:12px;text-align:right}
button.theme{background:var(--surface);color:var(--ink);border:1px solid var(--border);border-radius:8px;padding:6px 10px;cursor:pointer;font:inherit;font-size:12px;margin-top:6px}
nav{display:flex;gap:4px;flex-wrap:wrap;margin:18px 0 0;border-bottom:1px solid var(--grid)}
nav button{background:none;border:none;color:var(--ink2);font:inherit;padding:9px 12px;cursor:pointer;border-bottom:2px solid transparent;margin-bottom:-1px}
nav button[aria-selected="true"]{color:var(--ink);border-bottom-color:var(--blue);font-weight:600}
nav button:hover{color:var(--ink)}
main{max-width:1320px;margin:0 auto;padding:18px 28px 40px}
section{display:none} section.active{display:block}
.grid{display:grid;gap:16px}
.g2{grid-template-columns:repeat(2,minmax(0,1fr))}
.g3{grid-template-columns:repeat(3,minmax(0,1fr))}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-bottom:16px}
.kpi{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:12px 14px}
.kpi .l{color:var(--ink2);font-size:12px}
.kpi .v{font-size:24px;font-weight:600;margin:2px 0}
.kpi .s{color:var(--muted);font-size:12px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:14px 16px;min-width:0}
.card h3{margin:0 0 2px;font-size:15px}
.card .cap{color:var(--muted);font-size:12px;margin:0 0 6px}
.chart{width:100%;height:340px}
.tall{height:440px}
.insight{border-left:3px solid var(--blue);background:var(--wash);padding:8px 12px;border-radius:0 8px 8px 0;margin:0 0 16px;color:var(--ink)}
.controls{display:flex;gap:6px;flex-wrap:wrap;margin:4px 0 8px}
.seg{display:inline-flex;border:1px solid var(--border);border-radius:8px;overflow:hidden}
.seg button{background:var(--surface);color:var(--ink2);border:none;border-right:1px solid var(--border);padding:5px 10px;font:inherit;font-size:12px;cursor:pointer}
.seg button:last-child{border-right:none}
.seg button[aria-pressed="true"]{background:var(--wash);color:var(--ink);font-weight:600}
table{width:100%;border-collapse:collapse;font-size:12.5px;font-variant-numeric:tabular-nums}
th,td{padding:6px 8px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}
th:first-child,td:first-child,th.t,td.t{text-align:left}
th{color:var(--ink2);font-weight:600;cursor:pointer;user-select:none;position:sticky;top:0;background:var(--surface)}
th:hover{color:var(--ink)}
.tbl{max-height:380px;overflow:auto}
.form{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}
.form label{display:block;color:var(--ink2);font-size:12px;margin-bottom:3px}
.form input,.form select{width:100%;padding:7px 9px;border:1px solid var(--border);border-radius:8px;background:var(--page);color:var(--ink);font:inherit}
.big{font-size:30px;font-weight:600}
.note{color:var(--muted);font-size:12px}
footer{max-width:1320px;margin:0 auto;padding:0 28px 30px;color:var(--muted);font-size:12px}
@media (max-width:900px){.g2,.g3,.form{grid-template-columns:1fr} header,main,footer{padding-left:16px;padding-right:16px}}
</style>
</head>
<body>
<header>
  <div class="top">
    <div>
      <h1>Indian Railways Punctuality &amp; Network Delay Dynamics</h1>
      <div class="sub">Data warehousing · OLAP · graph mining · next-station delay prediction — 8 Feb 2025 to 7 Feb 2026, 38.4 M raw delay records</div>
    </div>
    <div class="who">Uplakshya Vishwakarma · M.Tech CSE-AI · Roll No. 26072011<br>IIT (BHU) Varanasi<br>
      <button class="theme" id="themeBtn" type="button">Toggle light / dark</button></div>
  </div>
  <nav role="tablist" id="tabs"></nav>
</header>
<main>
<!-- OVERVIEW -->
<section id="t-overview">
  <div class="kpis" id="kpi-overview"></div>
  <div class="insight">Trains leave their origin on time <b>86.6 %</b> of the time but arrive on time at only <b>47.4 %</b> of later stops — delay is created on the line, not at the platform of origin.</div>
  <div class="grid g2">
    <div class="card"><h3>From raw logs to model-ready transitions</h3><p class="cap">Row counts after each ETL step</p><div class="chart" id="c-funnel"></div></div>
    <div class="card"><h3>Origin departure vs en-route arrival</h3><p class="cap">Observed records</p>
      <div class="controls"><div class="seg" id="s-ctx"></div></div><div class="chart" id="c-ctx" style="height:300px"></div></div>
  </div>
</section>
<!-- PUNCTUALITY -->
<section id="t-punct">
  <div class="controls"><span class="note" style="align-self:center">Measure&nbsp;</span><div class="seg" id="s-cube"></div></div>
  <div class="grid g2">
    <div class="card"><h3>Train category</h3><p class="cap">Roll-up to primary_type_code</p><div class="chart" id="c-cat"></div></div>
    <div class="card"><h3>Railway zone</h3><p class="cap">Zones with ≥ 10,000 observed stops</p><div class="chart tall" id="c-zone"></div></div>
  </div>
  <div class="insight" style="margin-top:16px">Shatabdi and Vande Bharat services are the most punctual; special trains the least (mean 96 min, 17 % bottleneck). Stress concentrates in SECR, SER, NER and ECR.</div>
  <div class="grid g2">
    <div class="card"><h3>Zones: average delay vs frequency of extreme delay</h3><p class="cap">Bubble area = observed stops</p><div class="chart" id="c-zscatter"></div></div>
    <div class="card"><h3>Punctuality at origin vs en route, by category</h3><p class="cap">On time ≤ 15 min (%)</p><div class="chart" id="c-dumbbell"></div></div>
  </div>
  <div class="grid g2" style="margin-top:16px">
    <div class="card"><h3>Zone × category cuboid — worst cells</h3><p class="cap">Click a column header to sort</p><div class="tbl" id="tb-cuboid"></div></div>
    <div class="card"><h3>Major hubs (≥ 25,000 observed stops)</h3><p class="cap">Ranked by bottleneck rate</p><div class="tbl" id="tb-hubs25"></div></div>
  </div>
</section>
<!-- TEMPORAL -->
<section id="t-time">
  <div class="card"><h3>Monthly delay trajectory</h3><p class="cap">Journey start month · shaded = winter fog season</p><div class="chart" id="c-month"></div></div>
  <div class="grid g3" style="margin-top:16px">
    <div class="card"><h3>Monthly bottleneck rate</h3><p class="cap">Stops &gt; 167.65 min (%)</p><div class="chart" id="c-mbn"></div></div>
    <div class="card"><h3>Day of week</h3><p class="cap">Mean delay (min) — nearly flat</p><div class="chart" id="c-week"></div></div>
    <div class="card"><h3>Time of day</h3><p class="cap">Mean arrival delay by scheduled arrival hour</p><div class="chart" id="c-diurnal"></div></div>
  </div>
  <div class="card" style="margin-top:16px"><h3>Do zone rankings hold in winter?</h3><p class="cap">Bottleneck rate by chronological split</p>
    <div class="chart" id="c-zstab"></div></div>
</section>
<!-- ROUTE -->
<section id="t-route">
  <div class="insight"><b>Terminal schedule padding.</b> Delay climbs from 10.7 min at the origin to 69.7 min after stop 36, then falls to 28.2 min at the final stop, where 31.6 % of arrivals are early.</div>
  <div class="grid g2">
    <div class="card"><h3>Mean delay along the route</h3><p class="cap">Observed stops by route stage</p><div class="chart" id="c-route"></div></div>
    <div class="card"><h3>Early arrivals (&lt; 0 min)</h3><p class="cap">Share of stops by route stage</p><div class="chart" id="c-early"></div></div>
  </div>
  <div class="grid g2" style="margin-top:16px">
    <div class="card"><h3>Distance from origin</h3><p class="cap">Mean arrival delay (min)</p><div class="chart" id="c-dist"></div></div>
    <div class="card"><h3>Stations that add vs absorb delay</h3><p class="cap">Mean Δ = delay − previous-stop delay (≥ 1,000 transitions)</p><div class="chart tall" id="c-ampr"></div></div>
  </div>
</section>
<!-- NETWORK -->
<section id="t-net">
  <div class="kpis" id="kpi-net"></div>
  <div class="grid g2">
    <div class="card"><h3>Structure vs stress — composite bottleneck hubs</h3><p class="cap">S = 0.35·PR(betweenness) + 0.15·PR(degree) + 0.25·PR(mean delay) + 0.25·PR(bottleneck rate)</p><div class="chart tall" id="c-hubs"></div></div>
    <div class="card"><h3>Structural bridges</h3><p class="cap">Exact normalised betweenness centrality (Brandes)</p><div class="chart tall" id="c-btw"></div></div>
  </div>
  <div class="grid g2" style="margin-top:16px">
    <div class="card"><h3>Corridors that amplify delay</h3><div class="controls"><div class="seg" id="s-edges"></div></div><div class="chart" id="c-edges"></div></div>
    <div class="card"><h3>Once late, trains stay late</h3><p class="cap">Next stop still &gt; 15 min late, by current delay state</p><div class="chart" id="c-regime"></div></div>
  </div>
  <div class="card" style="margin-top:16px"><h3>Out-of-time persistence of station patterns</h3><p class="cap">Correlation of station statistics between training and later periods</p><div class="tbl" id="tb-persist"></div></div>
</section>
<!-- MODELS -->
<section id="t-models">
  <div class="kpis" id="kpi-models"></div>
  <div class="grid g2">
    <div class="card"><h3>Overall comparison</h3><div class="controls"><div class="seg" id="s-metric"></div></div><div class="chart" id="c-overall"></div></div>
    <div class="card"><h3>Test forecasts within an error band</h3><p class="cap">Share of 3.35 M test transitions (%)</p><div class="chart" id="c-bands"></div></div>
  </div>
  <div class="grid g2" style="margin-top:16px">
    <div class="card"><h3>By target severity</h3><p class="cap">Test MAE (min)</p><div class="chart" id="c-sev"></div></div>
    <div class="card"><h3>By stop context</h3><p class="cap">Test MAE (min) — persistence ignores terminal slack</p><div class="chart" id="c-ctxm"></div></div>
  </div>
  <div class="card" style="margin-top:16px"><h3>Response-curve audit — why XGBoost fails on extreme delays</h3>
    <p class="cap">Current delay swept; previous delay = current; every other feature at its training median</p>
    <div class="controls"><div class="seg" id="s-scen"></div></div><div class="chart tall" id="c-curve"></div><div class="insight" id="curve-note" style="margin:8px 0 0"></div></div>
  <div class="grid g2" style="margin-top:16px">
    <div class="card"><h3>Test MAE by train category</h3><p class="cap">min · colour scale at right</p><div class="chart tall" id="c-heat"></div></div>
    <div class="card"><h3>Feature importance</h3><div class="controls"><div class="seg" id="s-imp"></div></div><div class="chart tall" id="c-imp"></div></div>
  </div>
  <div class="card" style="margin-top:16px"><h3>Full results table</h3><div class="tbl" id="tb-overall"></div></div>
</section>
<!-- PREDICTOR -->
<section id="t-pred">
  <div class="insight">This in-browser predictor runs the trained <b>Ridge</b> model (exact coefficients) next to the persistence baseline. The Streamlit app (<code>streamlit run dashboard/app.py</code>) also runs the Random Forest and XGBoost models.</div>
  <div class="card">
    <div class="form" id="pred-form"></div>
  </div>
  <div class="kpis" id="kpi-pred" style="margin-top:16px"></div>
  <div class="card"><h3>What-if: how much the model adjusts persistence</h3><p class="cap">Ridge forecast − current delay, as the current delay changes (upstream momentum held fixed). Above 0 = the model expects the train to lose more time on this segment.</p><div class="chart" id="c-whatif"></div></div>
</section>
</main>
<footer>Source: Indian Railways Delay Dataset (Kaggle, RXYDENXD). Built from the JSON findings in <code>results/</code> by <code>dashboard/build_dashboard_html.py</code>. Bottleneck = delay &gt; 167.65 min (2σ); critical = &gt; 233.68 min (3σ).</footer>

<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const CAT = {"PASS-TRAINS":"Passenger","EXP-TRAINS":"Express","SF-TRAINS":"Superfast","T18-TRAINS":"Vande Bharat","PRM-TRAINS":"Special (PRM)","RAJ-TRAINS":"Rajdhani","SHT-TRAINS":"Shatabdi","GRB-TRAINS":"Garib Rath"};
const MN = {"1_Persistence_Baseline":"Persistence","2_Linear_Regression":"Ridge","3_Random_Forest":"Random Forest","4_XGBoost":"XGBoost"};
const MODELS = ["Persistence","Ridge","Random Forest","XGBoost"];
const T2 = 167.653054, T3 = 233.67987;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const MC = () => ({"Persistence":css('--gray'),"Ridge":css('--blue'),"Random Forest":css('--orange'),"XGBoost":css('--aqua')});
const SC = () => ({"Train":css('--s-train'),"Validation":css('--s-val'),"Test":css('--s-test')});
const fmt = (v, d=1) => (v==null||isNaN(v)) ? '—' : Number(v).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const int = v => Number(v).toLocaleString('en-US');

function layout(extra={}){
  const ink=css('--ink2'), grid=css('--grid');
  return Object.assign({
    paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
    font:{family:'system-ui,-apple-system,Segoe UI,sans-serif',size:12,color:ink},
    margin:{l:10,r:16,t:28,b:10}, bargap:0.3, hovermode:'closest',
    legend:{orientation:'h',x:0,y:1.12,font:{size:12}},
    xaxis:{automargin:true,gridcolor:grid,zeroline:false,linecolor:css('--axis'),showgrid:false},
    yaxis:{automargin:true,gridcolor:grid,zeroline:false,linecolor:css('--axis')},
    hoverlabel:{bgcolor:css('--surface'),bordercolor:css('--axis'),font:{color:css('--ink')}}
  }, extra);
}
function deep(a,b){for(const k in b){if(b[k]&&typeof b[k]==='object'&&!Array.isArray(b[k])){a[k]=deep(a[k]||{},b[k]);}else a[k]=b[k];}return a;}
function plot(id, traces, lay={}){Plotly.react(id, traces, deep(layout(), lay), {displaylogo:false,responsive:true,modeBarButtonsToRemove:['lasso2d','select2d']});}

function kpis(id, items){document.getElementById(id).innerHTML = items.map(([l,v,s])=>`<div class="kpi"><div class="l">${l}</div><div class="v">${v}</div><div class="s">${s}</div></div>`).join('');}
function seg(id, opts, init, onchange){
  const el=document.getElementById(id); let cur=init;
  el.innerHTML = opts.map(o=>`<button type="button" aria-pressed="${o===init}" data-v="${o}">${o}</button>`).join('');
  el.querySelectorAll('button').forEach(b=>b.onclick=()=>{cur=b.dataset.v;el.querySelectorAll('button').forEach(x=>x.setAttribute('aria-pressed',x===b));onchange(cur);});
  return ()=>cur;
}
function tableEl(id, rows, cols){
  const el=document.getElementById(id); let sortK=null, asc=false;
  const draw=()=>{
    let r=[...rows]; if(sortK){r.sort((a,b)=>{const x=a[sortK],y=b[sortK];return (typeof x==='number'?x-y:String(x).localeCompare(String(y)))*(asc?1:-1);});}
    el.innerHTML = `<table><thead><tr>${cols.map(c=>`<th class="${c.t?'t':''}" data-k="${c.k}">${c.h}${sortK===c.k?(asc?' ▲':' ▼'):''}</th>`).join('')}</tr></thead><tbody>`+
      r.map(x=>`<tr>${cols.map(c=>`<td class="${c.t?'t':''}">${c.f?c.f(x[c.k],x):x[c.k]}</td>`).join('')}</tr>`).join('')+`</tbody></table>`;
    el.querySelectorAll('th').forEach(th=>th.onclick=()=>{const k=th.dataset.k; if(sortK===k)asc=!asc; else {sortK=k;asc=false;} draw();});
  }; draw();
}

/* ---------------- tabs ---------------- */
const TABS=[["overview","Overview"],["punct","Punctuality cubes"],["time","Temporal"],["route","Route & stations"],["net","Network"],["models","Models"],["pred","Predictor"]];
const rendered={};
function show(t){
  document.querySelectorAll('nav button').forEach(b=>b.setAttribute('aria-selected',b.dataset.t===t));
  document.querySelectorAll('section').forEach(s=>s.classList.toggle('active',s.id==='t-'+t));
  RENDER[t](); rendered[t]=true;
  try{history.replaceState(null,'','#'+t);}catch(e){}
}
document.getElementById('tabs').innerHTML = TABS.map(([k,l])=>`<button role="tab" type="button" data-t="${k}">${l}</button>`).join('');
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>show(b.dataset.t));

/* ---------------- renderers ---------------- */
const RENDER = {};
RENDER.overview = () => {
  const obs=D.bench[1];
  kpis('kpi-overview',[["Warehouse fact rows","36.61 M","97.18 % observed"],["Trains / stations","8,720 / 9,027","365 days"],
    ["Mean delay",fmt(obs.mean_delay_min)+" min","median "+fmt(obs.median_delay_min,0)+" min"],["On time at origin","86.6 %","departure ≤ 15 min"],
    ["On time en route","47.4 %","arrival ≤ 15 min"],["Bottleneck stops","3.47 %","delay > 167.65 min"]]);
  const f=D.funnel;
  plot('c-funnel',[{type:'bar',orientation:'h',y:f.map(x=>x[0]),x:f.map(x=>x[1]/1e6),marker:{color:css('--blue')},text:f.map(x=>fmt(x[1]/1e6,2)+' M'),textposition:'outside',cliponaxis:false,
    hovertemplate:'%{y}<br>%{x:.3f} M rows<extra></extra>'}],{yaxis:{autorange:'reversed'},xaxis:{title:{text:'rows (millions)'},range:[0,47]},showlegend:false});
  const ctx=[D.bench[3],D.bench[2]], names=["Origin departure","En-route arrival"];
  const M={"On time %":["on_time_le_15m_pct","on time ≤ 15 min (%)"],"Mean":["mean_delay_min","mean delay (min)"],"Median":["median_delay_min","median delay (min)"],"Bottleneck %":["total_bottleneck_gt_2sd_pct","stops > 167.65 min (%)"]};
  const draw=m=>{const [k,t]=M[m];plot('c-ctx',[{type:'bar',x:names,y:ctx.map(c=>c[k]),width:0.45,marker:{color:css('--blue')},text:ctx.map(c=>fmt(c[k])),textposition:'outside',cliponaxis:false,hovertemplate:'%{x}: %{y:.2f}<extra></extra>'}],
    {yaxis:{title:{text:t},rangemode:'tozero'},showlegend:false});};
  const g=seg('s-ctx',Object.keys(M),"On time %",draw); draw(g());
};
RENDER.punct = () => {
  const M={"Mean delay":"mean_delay_min","Median":"median_delay_min","On time %":"on_time_le_15m_pct","Bottleneck %":"bottleneck_gt_2sd_pct","Critical %":"critical_gt_3sd_pct"};
  const draw=m=>{
    const k=M[m], better=k==='on_time_le_15m_pct';
    const c=[...D.category].sort((a,b)=>better?a[k]-b[k]:a[k]-b[k]);
    plot('c-cat',[{type:'bar',orientation:'h',y:c.map(x=>CAT[x.primary_type_code]),x:c.map(x=>x[k]),marker:{color:css('--blue')},text:c.map(x=>fmt(x[k])),textposition:'outside',cliponaxis:false,
      customdata:c.map(x=>[x.observed_stops,x.mean_delay_min,x.on_time_le_15m_pct,x.bottleneck_gt_2sd_pct]),
      hovertemplate:'<b>%{y}</b><br>stops %{customdata[0]:,}<br>mean %{customdata[1]:.1f} min<br>on time %{customdata[2]:.1f} %<br>bottleneck %{customdata[3]:.2f} %<extra></extra>'}],{showlegend:false,xaxis:{title:{text:m}}});
    const z=[...D.zone].sort((a,b)=>a[k]-b[k]);
    plot('c-zone',[{type:'bar',orientation:'h',y:z.map(x=>x.station_zone),x:z.map(x=>x[k]),marker:{color:css('--blue')},
      customdata:z.map(x=>[x.observed_stops,x.mean_delay_min,x.on_time_le_15m_pct]),
      hovertemplate:'<b>%{y}</b><br>stops %{customdata[0]:,}<br>mean %{customdata[1]:.1f} min<br>on time %{customdata[2]:.1f} %<extra></extra>'}],{showlegend:false,xaxis:{title:{text:m}}});
  };
  const g=seg('s-cube',Object.keys(M),"Bottleneck %",draw); draw(g());
  const below=new Set(["SR","SWR","NFR","NCR","ECOR","KR","SCR"]);
  const z=D.zone, mx=Math.max(...z.map(x=>x.observed_stops));
  plot('c-zscatter',[{type:'scatter',mode:'markers+text',x:z.map(x=>x.mean_delay_min),y:z.map(x=>x.bottleneck_gt_2sd_pct),text:z.map(x=>x.station_zone),
    textposition:z.map(x=>below.has(x.station_zone)?'bottom center':'top center'),textfont:{color:css('--ink2'),size:11},
    marker:{size:z.map(x=>8+30*Math.sqrt(x.observed_stops/mx)),color:css('--blue'),opacity:0.75,line:{width:2,color:css('--surface')}},
    customdata:z.map(x=>[x.observed_stops,x.on_time_le_15m_pct]),hovertemplate:'<b>%{text}</b><br>mean %{x:.1f} min<br>bottleneck %{y:.2f} %<br>stops %{customdata[0]:,}<extra></extra>'}],
    {showlegend:false,xaxis:{title:{text:'mean delay (min)'},showgrid:true},yaxis:{title:{text:'bottleneck rate (%)'}}});
  const o=Object.fromEntries(D.origin_cat.map(x=>[x.primary_type_code,x.on_time_le_15m_pct]));
  const a=[...D.arrival_cat].sort((p,q)=>p.on_time_le_15m_pct-q.on_time_le_15m_pct);
  const lines=a.map(x=>({type:'scatter',mode:'lines',x:[x.on_time_le_15m_pct,o[x.primary_type_code]],y:[CAT[x.primary_type_code],CAT[x.primary_type_code]],line:{color:css('--axis'),width:2},hoverinfo:'skip',showlegend:false}));
  plot('c-dumbbell',[...lines,
    {type:'scatter',mode:'markers',name:'Origin departure',x:a.map(x=>o[x.primary_type_code]),y:a.map(x=>CAT[x.primary_type_code]),marker:{size:11,color:css('--blue')},hovertemplate:'%{y}<br>origin %{x:.1f} %<extra></extra>'},
    {type:'scatter',mode:'markers',name:'En-route arrival',x:a.map(x=>x.on_time_le_15m_pct),y:a.map(x=>CAT[x.primary_type_code]),marker:{size:11,color:css('--orange')},hovertemplate:'%{y}<br>arrival %{x:.1f} %<extra></extra>'}],
    {xaxis:{title:{text:'on time ≤ 15 min (%)'},range:[30,100],showgrid:true}});
  tableEl('tb-cuboid',D.cuboid_top,[{k:'station_zone',h:'Zone',t:1},{k:'primary_type_code',h:'Category',t:1,f:v=>CAT[v]||v},{k:'observed_stops',h:'Stops',f:int},
    {k:'mean_delay_min',h:'Mean (min)',f:v=>fmt(v)},{k:'on_time_le_15m_pct',h:'On time %',f:v=>fmt(v)},{k:'bottleneck_gt_2sd_pct',h:'Bottleneck %',f:v=>fmt(v,2)}]);
  tableEl('tb-hubs25',D.hubs25k,[{k:'station_name',h:'Code',t:1},{k:'station_full_name',h:'Station',t:1},{k:'station_zone',h:'Zone',t:1},{k:'observed_stops',h:'Stops',f:int},
    {k:'mean_delay_min',h:'Mean (min)',f:v=>fmt(v)},{k:'bottleneck_gt_2sd_pct',h:'Bottleneck %',f:v=>fmt(v,2)}]);
};
RENDER.time = () => {
  const m=D.monthly, lab=m.map(x=>x.month_name.slice(0,3)+" '"+String(x.year).slice(2));
  plot('c-month',[
    {type:'scatter',mode:'lines+markers',name:'Mean',x:lab,y:m.map(x=>x.mean_delay_min),line:{color:css('--blue'),width:2},marker:{size:8},hovertemplate:'%{x}<br>mean %{y:.1f} min<extra></extra>'},
    {type:'scatter',mode:'lines+markers',name:'Median',x:lab,y:m.map(x=>x.median_delay_min),line:{color:css('--orange'),width:2},marker:{size:8},hovertemplate:'%{x}<br>median %{y:.0f} min<extra></extra>'},
    {type:'scatter',mode:'lines+markers',name:'P90',x:lab,y:m.map(x=>x.p90_delay_min),line:{color:css('--aqua'),width:2},marker:{size:8},visible:'legendonly',hovertemplate:'%{x}<br>P90 %{y:.0f} min<extra></extra>'}],
    {yaxis:{title:{text:'delay (min)'},rangemode:'tozero'},hovermode:'x unified',
     shapes:[{type:'rect',xref:'x',yref:'paper',x0:"Nov '25",x1:"Jan '26",y0:0,y1:1,fillcolor:css('--grid'),opacity:0.5,line:{width:0},layer:'below'}]});
  plot('c-mbn',[{type:'bar',x:lab,y:m.map(x=>x.bottleneck_gt_2sd_pct),marker:{color:css('--blue')},hovertemplate:'%{x}<br>%{y:.2f} %<extra></extra>'}],{showlegend:false,yaxis:{title:{text:'%'}}});
  const w=D.weekday;
  plot('c-week',[{type:'bar',x:w.map(x=>x.day_name.slice(0,3)),y:w.map(x=>x.mean_delay_min),marker:{color:css('--blue')},hovertemplate:'%{x}<br>mean %{y:.2f} min<br><extra></extra>'}],{showlegend:false,yaxis:{title:{text:'min'},range:[0,40]}});
  const di=D.diurnal;
  plot('c-diurnal',[{type:'bar',x:di.map(x=>x.time_of_day_block.split('_')[1].replace(' (','<br>(')),y:di.map(x=>x.mean_arrival_delay_min),marker:{color:css('--blue')},
    customdata:di.map(x=>[x.traffic_share_pct,x.bottleneck_gt_2sd_pct]),hovertemplate:'%{x}<br>mean %{y:.1f} min<br>share %{customdata[0]:.1f} %<br>bottleneck %{customdata[1]:.2f} %<extra></extra>'}],
    {showlegend:false,yaxis:{title:{text:'min'}},xaxis:{tickfont:{size:10}}});
  const zs=[...D.zone_stability].sort((a,b)=>b.train_bottleneck_pct-a.train_bottleneck_pct), c=SC();
  plot('c-zstab',[["Train","train_bottleneck_pct"],["Validation","val_bottleneck_pct"],["Test","test_bottleneck_pct"]].map(([n,k])=>({type:'bar',name:n,x:zs.map(z=>z.station_zone),y:zs.map(z=>z[k]),marker:{color:c[n]},hovertemplate:n+' · %{x}<br>%{y:.2f} %<extra></extra>'})),
    {barmode:'group',yaxis:{title:{text:'bottleneck rate (%)'}}});
};
RENDER.route = () => {
  const r=D.route, lab=r.map(x=>x.route_stage.split('_')[1].replace(' (','<br>(')), col=r.map((x,i)=>i===r.length-1?css('--orange'):css('--blue'));
  plot('c-route',[{type:'bar',x:lab,y:r.map(x=>x.mean_delay_min),marker:{color:col},text:r.map(x=>fmt(x.mean_delay_min)),textposition:'outside',cliponaxis:false,
    customdata:r.map(x=>[x.median_delay_min,x.observed_stops]),hovertemplate:'%{x}<br>mean %{y:.1f} min<br>median %{customdata[0]} min<br>stops %{customdata[1]:,}<extra></extra>'}],{showlegend:false,yaxis:{title:{text:'mean delay (min)'}},xaxis:{tickfont:{size:10}}});
  plot('c-early',[{type:'bar',x:lab,y:r.map(x=>x.early_lt_0m_pct),marker:{color:col},text:r.map(x=>fmt(x.early_lt_0m_pct,2)+' %'),textposition:'outside',cliponaxis:false,hovertemplate:'%{x}<br>%{y:.2f} % early<extra></extra>'}],
    {showlegend:false,yaxis:{title:{text:'early arrivals (%)'}},xaxis:{tickfont:{size:10}}});
  const d=D.distance;
  plot('c-dist',[{type:'bar',x:d.map(x=>x.distance_band.split('_')[1]),y:d.map(x=>x.mean_arrival_delay_min),marker:{color:css('--blue')},text:d.map(x=>fmt(x.mean_arrival_delay_min)),textposition:'outside',cliponaxis:false,
    customdata:d.map(x=>x.bottleneck_gt_2sd_pct),hovertemplate:'%{x}<br>mean %{y:.1f} min<br>bottleneck %{customdata:.2f} %<extra></extra>'}],{showlegend:false,yaxis:{title:{text:'min'}}});
  const both=[...D.amp.slice(0,10),...D.rec.slice(0,10)].sort((a,b)=>a.mean_delay_delta_min-b.mean_delay_delta_min);
  plot('c-ampr',[{type:'bar',orientation:'h',y:both.map(x=>x.station_name+' · '+x.station_full_name+' ('+x.station_zone+')'),x:both.map(x=>x.mean_delay_delta_min),
    marker:{color:both.map(x=>x.mean_delay_delta_min),cmin:-80,cmax:80,colorscale:[[0,css('--blue')],[0.5,css('--mid')],[1,css('--red')]],colorbar:{title:{text:'Δ min'},thickness:10}},
    customdata:both.map(x=>[x.mean_upstream_delay_min,x.mean_arrival_delay_min,x.observed_transitions]),
    hovertemplate:'<b>%{y}</b><br>mean Δ %{x:+.1f} min<br>%{customdata[0]:.1f} → %{customdata[1]:.1f} min<br>%{customdata[2]:,} transitions<extra></extra>'}],
    {showlegend:false,yaxis:{tickfont:{size:10}},xaxis:{title:{text:'mean change in delay (min)'},zeroline:true,zerolinecolor:css('--axis')}});
};
RENDER.net = () => {
  const g=D.graph, l=D.lags;
  kpis('kpi-net',[["Station nodes",int(g.n_nodes),"static schedule graph"],["Directed edges",int(g.n_directed_edges),"consecutive scheduled stops"],
    ["Largest weak component",fmt(g.largest_wcc_nodes/g.n_nodes*100,2)+" %",g.weakly_connected_components+" components"],
    ["Stop-to-stop correlation","r = "+fmt(l.r_lag1_all,3),"lag 2: "+fmt(l.r_lag2_all,3)+" · lag 3: "+fmt(l.r_lag3_all,3)],["Bottleneck state retained","99.72 %","next stop still > 15 min late"]]);
  const h=D.hubs, pos={"BSL":"top left","VZM":"bottom center","BJU":"top right","LKO":"bottom center","CNB":"bottom center"};
  plot('c-hubs',[{type:'scatter',mode:'markers+text',x:h.map(x=>x.betweenness_centrality),y:h.map(x=>x.train_arr_mean_delay_min),text:h.map(x=>x.station_name),
    textposition:h.map(x=>pos[x.station_name]||'top center'),textfont:{color:css('--ink2')},marker:{size:h.map(x=>x.total_degree/2.2),color:css('--blue'),opacity:0.8,line:{width:2,color:css('--surface')}},
    customdata:h.map(x=>[x.station_full_name,x.station_zone,x.train_bottleneck_gt_2sd_pct,x.composite_hub_stress_score,x.total_degree]),
    hovertemplate:'<b>%{customdata[0]}</b> (%{customdata[1]})<br>betweenness %{x:.4f} · degree %{customdata[4]}<br>mean %{y:.1f} min · bottleneck %{customdata[2]:.2f} %<br>score %{customdata[3]:.3f}<extra></extra>'}],
    {showlegend:false,xaxis:{title:{text:'betweenness centrality'},showgrid:true},yaxis:{title:{text:'mean arrival delay, training (min)'}}});
  const b=[...D.betweenness].sort((p,q)=>p.betweenness_centrality-q.betweenness_centrality);
  plot('c-btw',[{type:'bar',orientation:'h',y:b.map(x=>x.station_name+' · '+x.station_full_name),x:b.map(x=>x.betweenness_centrality),marker:{color:css('--blue')},
    text:b.map(x=>fmt(x.betweenness_centrality,3)),textposition:'outside',cliponaxis:false,hovertemplate:'%{y}<br>%{x:.4f}<extra></extra>'}],{showlegend:false,xaxis:{title:{text:'normalised betweenness'}}});
  const draw=set=>{const e=[...(set.startsWith('Trunk')?D.trunk:D.edges)].sort((p,q)=>p.train_mean_delay_delta_min-q.train_mean_delay_delta_min);
    plot('c-edges',[{type:'bar',orientation:'h',y:e.map(x=>x.from_station+' → '+x.to_station+' ('+x.to_station_zone+')'),x:e.map(x=>x.train_mean_delay_delta_min),marker:{color:css('--red')},
      text:e.map(x=>'+'+fmt(x.train_mean_delay_delta_min)),textposition:'outside',cliponaxis:false,
      customdata:e.map(x=>[x.train_obs_transition_count,x.scheduled_train_count,x.train_amp_gt_5m_rate*100,x.train_from_mean_delay_min,x.train_to_mean_delay_min]),
      hovertemplate:'<b>%{y}</b><br>+%{x:.1f} min per run<br>%{customdata[3]:.1f} → %{customdata[4]:.1f} min<br>%{customdata[0]:,} runs · %{customdata[1]} trains<br>amplifies > 5 min on %{customdata[2]:.1f} %<extra></extra>'}],
      {showlegend:false,xaxis:{title:{text:'mean delay added on the edge (min)'}}});};
  const gs=seg('s-edges',["Trunk corridors (≥ 3,000 runs)","Largest per run (≥ 500 runs)"],"Trunk corridors (≥ 3,000 runs)",draw); draw(gs());
  const rg=D.regimes, rl=["Early / on time<br>≤ 0","Minor<br>1–15","Moderate<br>16–60","Heavy<br>61–168","Bottleneck<br>> 168"];
  plot('c-regime',[{type:'bar',x:rl,y:rg.map(x=>x.downstream_delayed_gt_15m_pct),marker:{color:css('--blue')},text:rg.map(x=>fmt(x.downstream_delayed_gt_15m_pct)+' %'),textposition:'outside',cliponaxis:false,
    customdata:rg.map(x=>x.n_transitions),hovertemplate:'%{x}<br>%{y:.2f} % still late<br>%{customdata:,} transitions<extra></extra>'}],{showlegend:false,yaxis:{title:{text:'%'},range:[0,112]},xaxis:{title:{text:'delay at current stop (min)'}}});
  tableEl('tb-persist',D.persistence,[{k:'cohort',h:'Cohort',t:1,f:v=>v.split(' ')[0].replace(/_/g,' ')},{k:'n_stations',h:'Stations',f:int},
    {k:'mean_delay_pearson_train_val',h:'Mean delay r (→ val)',f:v=>fmt(v,3)},{k:'mean_delay_pearson_train_test',h:'Mean delay r (→ test)',f:v=>fmt(v,3)},
    {k:'bottleneck_pearson_train_val',h:'Bottleneck r (→ val)',f:v=>fmt(v,3)},{k:'bottleneck_pearson_train_test',h:'Bottleneck r (→ test)',f:v=>fmt(v,3)},
    {k:'top20_hub_overlap_train_vs_test',h:'Top-20 overlap (test)',f:v=>v==null?'—':v+' / 20'}]);
};
RENDER.models = () => {
  const ov=D.overall.map(x=>Object.assign({Model:MN[x.model]},x)), rf=ov[2], pe=ov[0], mc=MC(), sc=SC();
  kpis('kpi-models',[["Best model","Random Forest","80 trees · depth 14 · leaf 20"],["Test MAE",fmt(rf.test_MAE,2)+" min",fmt(rf.test_MAE_gain_vs_persistence_pct,2)+" % lower than persistence"],
    ["Test RMSE",fmt(rf.test_RMSE,2)+" min",fmt(rf.test_RMSE_gain_vs_persistence_pct,2)+" % lower than persistence"],["Test MSE",fmt(rf.test_MSE)+" min²",fmt(rf.test_MSE_gain_vs_persistence_pct,2)+" % lower than persistence"],
    ["Test R²",fmt(rf.test_R2,4),"persistence "+fmt(pe.test_R2,4)]]);
  const draw=m=>{
    const sp=[["Train","train_"],["Validation","val_"],["Test","test_"]];
    if(m==='R²'){
      plot('c-overall',sp.map(([n,p])=>({type:'scatter',mode:'markers',name:n,x:ov.map(x=>x.Model),y:ov.map(x=>x[p+'R2']),marker:{size:13,color:sc[n],line:{width:2,color:css('--surface')}},hovertemplate:n+' · %{x}<br>R² %{y:.4f}<extra></extra>'})),
        {yaxis:{title:{text:'R²'}}});
    } else {
      plot('c-overall',sp.map(([n,p])=>({type:'bar',name:n,x:ov.map(x=>x.Model),y:ov.map(x=>x[p+m]),marker:{color:sc[n]},hovertemplate:n+' · %{x}<br>'+m+' %{y:.3f}<extra></extra>'})),
        {barmode:'group',yaxis:{title:{text:m+(m==='MSE'?' (min²)':' (min)')}}});
    }
  };
  const g=seg('s-metric',["MAE","RMSE","MSE","R²"],"MAE",draw); draw(g());
  const bk=Object.keys(D.bands);
  plot('c-bands',bk.map(k=>({type:'bar',name:MN[k],x:['±5 min','±10 min','±15 min'],y:[D.bands[k].pct_within_5m,D.bands[k].pct_within_10m,D.bands[k].pct_within_15m],marker:{color:mc[MN[k]]},hovertemplate:MN[k]+' · %{x}<br>%{y:.2f} %<extra></extra>'})),
    {barmode:'group',yaxis:{range:[0,100],title:{text:'%'}}});
  const sevL={"normal":"Normal ≤ 167.65","high_risk":"High-risk 167.65–233.68","critical":"Critical > 233.68"};
  plot('c-sev',MODELS.map(mn=>{const r=D.severity.filter(x=>MN[x.model]===mn);return {type:'bar',name:mn,x:r.map(x=>sevL[x.target_severity]),y:r.map(x=>x.MAE),marker:{color:mc[mn]},customdata:r.map(x=>[x.RMSE,x.n_test_transitions]),hovertemplate:mn+'<br>MAE %{y:.2f} · RMSE %{customdata[0]:.2f}<br>%{customdata[1]:,} transitions<extra></extra>'};}),
    {barmode:'group',yaxis:{title:{text:'MAE (min)'}}});
  plot('c-ctxm',MODELS.map(mn=>{const r=D.context.filter(x=>MN[x.model]===mn);return {type:'bar',name:mn,x:r.map(x=>x.stop_context.startsWith('Terminal')?'Terminal destination':'Intermediate stop'),y:r.map(x=>x.MAE),marker:{color:mc[mn]},customdata:r.map(x=>[x.Mean_Bias,x.RMSE]),hovertemplate:mn+'<br>MAE %{y:.2f} · RMSE %{customdata[1]:.2f}<br>bias %{customdata[0]:+.2f} min<extra></extra>'};}),
    {barmode:'group',yaxis:{title:{text:'MAE (min)'}}});
  if(D.curves){
    const cv=D.curves, s=cv.summary;
    const drawC=sc_=>{plot('c-curve',Object.entries(cv.scenarios[sc_]).map(([k,ys])=>({type:'scatter',mode:'lines',name:MN[k],x:cv.curr_delay_min,y:ys,line:{color:mc[MN[k]],width:MN[k]==='XGBoost'?3:2,dash:MN[k]==='Persistence'?'dot':'solid'},hovertemplate:MN[k]+'<br>%{x} → %{y:.1f} min<extra></extra>'})),
      {xaxis:{title:{text:'current-stop delay (min)'},showgrid:true},yaxis:{title:{text:'forecast next-stop delay (min)'}},
       shapes:[{type:'rect',xref:'x',yref:'paper',x0:T3,x1:1440,y0:0,y1:1,fillcolor:css('--red'),opacity:0.07,line:{width:0},layer:'below'}],
       annotations:[{x:T3+10,y:1,xref:'x',yref:'paper',text:'critical regime (> 3σ)',showarrow:false,xanchor:'left',yanchor:'top',font:{color:css('--ink2')}}]});};
    const gc=seg('s-scen',["intermediate","terminal"],"intermediate",drawC); drawC(gc());
    document.getElementById('curve-note').innerHTML=`XGBoost's forecast is <b>constant at ${fmt(s.xgb_plateau_value_min,0)} min</b> for every current delay ≥ ${fmt(s.xgb_forecast_constant_above_curr_delay_min,0)} min — the histogram split finder has only a few bins in the sparse tail. At 1,000 min it is ${fmt(s["4_XGBoost_abs_gap_vs_identity_at_1000min"],0)} min off the identity line (Ridge ${fmt(s["2_Linear_Regression_abs_gap_vs_identity_at_1000min"],0)}, RF ${fmt(s["3_Random_Forest_abs_gap_vs_identity_at_1000min"],0)}). Fix: predict the residual next − current.`;
  }
  const cats=[...new Set(D.cat_eval.map(x=>x.primary_type_code))];
  const pers=Object.fromEntries(D.cat_eval.filter(x=>MN[x.model]==='Persistence').map(x=>[x.primary_type_code,x.MAE]));
  cats.sort((a,b)=>pers[a]-pers[b]);
  const z=cats.map(c=>MODELS.map(mn=>{const r=D.cat_eval.find(x=>x.primary_type_code===c&&MN[x.model]===mn);return r?r.MAE:null;}));
  plot('c-heat',[{type:'heatmap',x:MODELS,y:cats.map(c=>CAT[c]),z,text:z.map(r=>r.map(v=>fmt(v,2))),texttemplate:'%{text}',
    colorscale:[[0,css('--seq1')],[0.25,css('--seq2')],[0.5,css('--seq3')],[0.75,css('--seq4')],[1,css('--seq5')]],colorbar:{title:{text:'MAE'},thickness:10},hovertemplate:'%{y} · %{x}<br>MAE %{z:.2f} min<extra></extra>'}],
    {xaxis:{side:'top'}});
  const drawI=w=>{const k=w==='Random Forest'?'rf_importance_pct':'xgb_gain_importance_pct'; const fi=[...D.importance].sort((a,b)=>a[k]-b[k]);
    plot('c-imp',[{type:'scatter',mode:'markers+text',x:fi.map(x=>x[k]),y:fi.map(x=>x.feature_name),text:fi.map(x=>fmt(x[k],2)+' %'),textposition:'middle right',textfont:{color:css('--ink2'),size:11},
      marker:{size:11,color:w==='Random Forest'?mc['Random Forest']:mc['XGBoost']},hovertemplate:'%{y}<br>%{x:.2f} %<extra></extra>'}],
      {showlegend:false,xaxis:{type:'log',range:[-1.8,2.85],title:{text:'importance (%, log scale)'},showgrid:true},yaxis:{tickfont:{size:11}}});};
  const gi=seg('s-imp',["Random Forest","XGBoost gain"],"Random Forest",drawI); drawI(gi());
  tableEl('tb-overall',ov,[{k:'Model',h:'Model',t:1},{k:'selected_config',h:'Config',t:1},{k:'val_MSE',h:'Val MSE',f:v=>fmt(v,2)},{k:'val_RMSE',h:'Val RMSE',f:v=>fmt(v,2)},{k:'val_MAE',h:'Val MAE',f:v=>fmt(v,2)},
    {k:'val_R2',h:'Val R²',f:v=>fmt(v,4)},{k:'test_MSE',h:'Test MSE',f:v=>fmt(v,2)},{k:'test_RMSE',h:'Test RMSE',f:v=>fmt(v,2)},{k:'test_MAE',h:'Test MAE',f:v=>fmt(v,2)},{k:'test_R2',h:'Test R²',f:v=>fmt(v,4)},
    {k:'test_MAE_gain_vs_persistence_pct',h:'MAE gain %',f:v=>fmt(v,2)},{k:'test_MSE_gain_vs_persistence_pct',h:'MSE gain %',f:v=>fmt(v,2)}]);
};
RENDER.pred = () => {
  if(!D.curves){document.getElementById('pred-form').innerHTML='<p>Run analysis/step05_04_model_response_audit.py to export the Ridge equation.</p>';return;}
  const eq=D.curves.ridge_raw_unit_equation, sc=D.scaler, pr=D.priors, m=D.m;
  const shrink=(x,n,p)=>(n*x+m*p)/(n+m);
  const presets=[["Typical segment (training medians)",null],...D.trunk.map(r=>[r.from_station+' → '+r.to_station+' (+'+fmt(r.train_mean_delay_delta_min)+' min/run)',r]),...D.edges.map(r=>[r.from_station+' → '+r.to_station+' (+'+fmt(r.train_mean_delay_delta_min)+' min/run)',r])];
  const hubs=Object.fromEntries([...D.hubs].map(h=>[h.station_name,h]));
  const cats=D.cat_priors.map(c=>[(CAT[c.primary_type_code]||c.primary_type_code)+(c.is_premium_special?' · special':''),c.train_category_delay_prior_min]);
  const F=[["seg","Segment preset","select"],["curr","Delay at current stop (min)",45],["prev","Delay at previous stop (min)",30],["cat","Train category","selectc"],
    ["km","Segment distance (km)",pr.median_train_seg_dist_km],["run","Scheduled running time (min)",pr.median_train_seg_run_min],["hour","Scheduled arrival hour (0–23)",12],
    ["prog","Route progress of next stop (0–1)",0.55],["term","Next stop is the terminus","bool"],["edge","Edge amplification prior (min)",+sc.transition_hist_delay_delta_min.median.toFixed(2)],
    ["hist","Next-station mean arrival delay prior (min)",+sc.next_station_hist_arr_delay_min.median.toFixed(2)],["btw","Next-station betweenness",sc.next_station_betweenness.median]];
  const form=document.getElementById('pred-form');
  form.innerHTML=F.map(([k,l,v])=>{
    if(v==='select') return `<div><label for="p-${k}">${l}</label><select id="p-${k}">${presets.map((p,i)=>`<option value="${i}">${p[0]}</option>`).join('')}</select></div>`;
    if(v==='selectc') return `<div><label for="p-${k}">${l}</label><select id="p-${k}">${cats.map((c,i)=>`<option value="${i}" ${i===1?'selected':''}>${c[0]}</option>`).join('')}</select></div>`;
    if(v==='bool') return `<div><label for="p-${k}">${l}</label><select id="p-${k}"><option value="0">No</option><option value="1">Yes</option></select></div>`;
    return `<div><label for="p-${k}">${l}</label><input id="p-${k}" type="number" step="any" value="${v}"></div>`;}).join('');
  const val=k=>parseFloat(document.getElementById('p-'+k).value)||0;
  const ridge=f=>eq.intercept+Object.entries(eq.coefficients).reduce((s,[k,c])=>s+c*f[k],0);
  const clip=v=>Math.max(-120,Math.min(1440,v));
  const feats=(curr,prev)=>({curr_delay_min:curr,prev_delay_min:prev,prev_delay_delta_min:curr-prev,train_category_delay_prior_min:cats[val('cat')][1],
    route_progress_ratio:val('term')?1:val('prog'),is_terminal_arrival:val('term'),segment_distance_km:val('km'),segment_sched_running_min:val('run'),
    next_sched_arrival_hour:val('hour'),next_station_betweenness:val('btw'),next_station_hist_arr_delay_min:val('hist'),transition_hist_delay_delta_min:val('edge')});
  const update=()=>{
    const curr=val('curr'),prev=val('prev'), y=clip(ridge(feats(curr,prev)));
    const reg=y>T3?'critical (> 3σ)':y>T2?'high-risk (2σ–3σ)':'normal';
    kpis('kpi-pred',[["Persistence forecast",fmt(curr)+" min","= current delay"],["Ridge forecast",fmt(y)+" min",(y-curr>=0?'+':'')+fmt(y-curr)+" min vs current"],
      ["Regime",reg,"thresholds 167.65 / 233.68 min"],["Ridge test MAE","7.17 min","Random Forest 6.68 min (Streamlit app)"]]);
    const xs=[];for(let x=-30;x<=720;x+=10)xs.push(x);
    const mc=MC();
    plot('c-whatif',[{type:'scatter',mode:'lines',name:'Persistence (no change)',x:xs,y:xs.map(()=>0),line:{color:mc.Persistence,dash:'dot',width:2},hoverinfo:'skip'},
      {type:'scatter',mode:'lines',name:'Ridge adjustment',x:xs,y:xs.map(x=>clip(ridge(feats(x,x-(curr-prev))))-x),line:{color:mc.Ridge,width:2},hovertemplate:'current %{x} min → %{y:+.1f} min change<extra></extra>'}],
      {xaxis:{title:{text:'current-stop delay (min)'},showgrid:true},yaxis:{title:{text:'forecast − current (min)'}},shapes:[{type:'line',x0:curr,x1:curr,yref:'paper',y0:0,y1:1,line:{color:css('--axis'),width:1}}]});
  };
  document.getElementById('p-seg').onchange=()=>{const r=presets[val('seg')][1];
    document.getElementById('p-edge').value=r?shrink(r.train_mean_delay_delta_min,r.train_obs_transition_count,pr.global_train_delta_mean).toFixed(2):sc.transition_hist_delay_delta_min.median.toFixed(2);
    const h=r&&hubs[r.to_station];
    document.getElementById('p-hist').value=h?shrink(h.train_arr_mean_delay_min,h.train_obs_stops,pr.global_train_arr_mean_delay).toFixed(2):sc.next_station_hist_arr_delay_min.median.toFixed(2);
    document.getElementById('p-btw').value=h?h.betweenness_centrality:sc.next_station_betweenness.median; update();};
  form.querySelectorAll('input,select').forEach(el=>{if(el.id!=='p-seg')el.oninput=update;});
  update();
};

/* ---------------- theme ---------------- */
document.getElementById('themeBtn').onclick=()=>{
  const r=document.documentElement, cur=r.getAttribute('data-theme')||(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');
  r.setAttribute('data-theme',cur==='dark'?'light':'dark');
  const active=document.querySelector('nav button[aria-selected="true"]'); if(active) RENDER[active.dataset.t]();
};
const start=(location.hash||'#overview').slice(1);
show(TABS.some(t=>t[0]===start)?start:'overview');
</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inline", action="store_true", help="embed Plotly.js so the file works offline")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    data = json.dumps(build_data(), separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    if args.inline:
        import plotly
        js = (Path(plotly.__file__).parent / "package_data" / "plotly.min.js").read_text(encoding="utf-8")
        plotly_tag = "<script>" + js + "</script>"
    else:
        plotly_tag = f'<script src="{PLOTLY_CDN}" charset="utf-8"></script>'
    html = HTML.replace("__PLOTLY__", plotly_tag).replace("__DATA__", data)
    Path(args.out).write_text(html, encoding="utf-8")
    print(f"Wrote {args.out} ({len(html)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
