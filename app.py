"""
Supply Chain Intelligence — Interactive Dashboard
AI-Enhanced Logistics & Supply Chain Optimization — ZF Group
Team: Syntax Squad | INNOVITUS 1.0
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Supply Chain Intelligence",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0f1117;
    color: #e2e8f0;
    font-family: 'Inter', sans-serif;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
    background-color: #161b27;
    border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stMetric"] {
    background: #161b27;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 14px 18px;
}
[data-testid="stMetric"] label { color: #94a3b8 !important; font-size: 12px !important; }
[data-testid="stMetricValue"]  { color: #f1f5f9 !important; font-size: 22px !important; font-weight: 600 !important; }
.stTabs [data-baseweb="tab-list"] { background: #161b27; border-radius: 8px; padding: 4px; gap: 4px; }
.stTabs [data-baseweb="tab"]      { color: #64748b; border-radius: 6px; padding: 8px 20px; font-size: 13px; }
.stTabs [aria-selected="true"]    { background: #1e40af !important; color: #fff !important; }
.stButton > button {
    background: #1e40af; color: #fff; border: none;
    border-radius: 6px; padding: 10px 28px;
    font-size: 14px; font-weight: 500; transition: background 0.2s;
}
.stButton > button:hover { background: #2563eb; }
.section-title {
    font-size: 15px; font-weight: 600; color: #94a3b8;
    text-transform: uppercase; letter-spacing: 0.08em;
    margin: 28px 0 12px 0; padding-bottom: 6px;
    border-bottom: 1px solid #1e293b;
}
.card {
    background: #161b27; border: 1px solid #1e293b;
    border-radius: 8px; padding: 16px 20px; margin-bottom: 10px;
    font-size: 14px; line-height: 1.6;
}
.card.critical { border-left: 3px solid #ef4444; }
.card.warning  { border-left: 3px solid #f59e0b; }
.card.ok       { border-left: 3px solid #10b981; }
.card.info     { border-left: 3px solid #3b82f6; }
.badge-critical { background:#ef44441a; color:#ef4444; font-size:11px; padding:2px 8px; border-radius:4px; font-weight:600; }
.badge-warning  { background:#f59e0b1a; color:#f59e0b; font-size:11px; padding:2px 8px; border-radius:4px; font-weight:600; }
.badge-ok       { background:#10b9811a; color:#10b981; font-size:11px; padding:2px 8px; border-radius:4px; font-weight:600; }
.badge-info     { background:#3b82f61a; color:#3b82f6; font-size:11px; padding:2px 8px; border-radius:4px; font-weight:600; }
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── Data & model loading (cached) ──────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading dataset and training models (XGBoost-style HGBR + GBM ensemble)…")
def load_and_train():
    from backend.data   import load_data, preprocess
    from backend.models import train_models

    df_raw   = load_data()
    df_clean = preprocess(df_raw)
    demand_model, delay_model, d_cols, l_cols = train_models(df_clean)
    return df_raw, df_clean, demand_model, delay_model, d_cols, l_cols


df_raw, df_clean, demand_model, delay_model, d_cols, l_cols = load_and_train()

from backend.services import (
    reorder_point, inventory_status, select_best_supplier,
    route_info, calculate_emission, sustainability_score,
    detect_spike, get_weather_risk, make_decision,
    resilience_scenarios, kpi_summary,
)
from backend.models import predict_demand, predict_delay
from config import (
    PLANT_COORDS, SUPPLIER_COORDS,
    SHIPPING_MODES, CUSTOMER_SEGMENTS, TRAFFIC_OPTIONS,
    DISRUPTION_TYPES, SUPPLIER_LOCS, OPT_MODES,
    SHIPPING_MODE_MAP, CUSTOMER_SEG_MAP, TRAFFIC_MAP, DISRUPTION_MAP,
    SUPPLIER_LOC_MAP,
)


# ── Build input row from user selections ───────────────────────────────────────

def build_input(params: dict) -> pd.DataFrame:
    now = datetime.now()
    return pd.DataFrame([{
        # time
        "order_day":   now.day,
        "order_month": now.month,
        "order_hour":  now.hour,
        # user-supplied
        "sales_per_customer":           params["sales_per_customer"],
        "category_id":                  params["category_id"],
        "days_for_shipment_(scheduled)": params["lead_time"],
        "days_for_shipping_(real)":      params["lead_time"],
        "lead_time_days":                params["lead_time"],
        "distance_km":                   params["distance_km"],
        "supplier_lead_time":            params["supplier_lead_time"],
        "disruption_severity":           params["disruption_severity"],
        "lag_1":                         params["lag_1"],
        "rolling_mean_3":                params["rolling_mean_3"],
        "traffic_condition":             TRAFFIC_MAP.get(params["traffic_condition"], 1),
        "disruption_type":               max(DISRUPTION_MAP.get(params["disruption_type"], -1), 0),
        "shipping_mode":                 SHIPPING_MODE_MAP.get(params["shipping_mode"], 0),
        "customer_segment":              CUSTOMER_SEG_MAP.get(params["customer_segment"], 0),
        "late_delivery_risk":            0,
        "supplier_reliability_score":    params["supplier_reliability"],
        "supplier_location":             SUPPLIER_LOC_MAP.get(params["supplier_location"], 0),
        "vehicle_type":                  0,
    }])


# ── Plot helpers ───────────────────────────────────────────────────────────────

PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#94a3b8", size=12),
    margin=dict(l=10, r=10, t=30, b=10),
)
GRID = dict(showgrid=True, gridcolor="#1e293b", zeroline=False)


def demand_chart(forecast):
    days = [f"Day {i+1}" for i in range(len(forecast))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=days, y=forecast,
        mode="lines+markers",
        line=dict(color="#3b82f6", width=2),
        marker=dict(size=6, color="#60a5fa"),
        fill="tozeroy",
        fillcolor="rgba(59,130,246,0.08)",
        name="Demand",
    ))
    fig.update_layout(**PLOTLY_BASE, height=260, showlegend=False,
                      xaxis=dict(GRID, title=""), yaxis=dict(GRID, title="Units"))
    return fig


def delay_gauge(prob_pct):
    color = "#ef4444" if prob_pct > 70 else "#f59e0b" if prob_pct > 40 else "#10b981"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_pct,
        number={"suffix": "%", "font": {"color": color, "size": 32}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#334155", "tickfont": {"color": "#64748b"}},
            "bar":  {"color": color},
            "bgcolor": "#161b27",
            "steps": [
                {"range": [0,  40], "color": "#0f2a1a"},
                {"range": [40, 70], "color": "#2a1f0a"},
                {"range": [70, 100], "color": "#2a0a0a"},
            ],
            "threshold": {"line": {"color": color, "width": 2}, "value": prob_pct},
        },
    ))
    fig.update_layout(**PLOTLY_BASE, height=220)
    return fig


def co2_bar(emission, score):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=["CO2 Emission"], y=[emission],
                         marker_color="#f59e0b", text=[f"{emission:.1f} kg"], textposition="auto"))
    fig.add_trace(go.Bar(x=["Eco Score"], y=[score],
                         marker_color="#10b981", text=[f"{score}/100"], textposition="auto"))
    fig.update_layout(**PLOTLY_BASE, height=220, showlegend=False,
                      xaxis=dict(GRID), yaxis=dict(GRID))
    return fig


def route_map(supplier_name, slat, slon, plat, plon):
    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lat=[slat, plat], lon=[slon, plon],
        mode="lines+markers+text",
        line=dict(width=2, color="#3b82f6"),
        marker=dict(size=[12, 12], color=["#3b82f6", "#10b981"], symbol=["circle", "diamond"]),
        text=[supplier_name, "Bangalore Plant"],
        textposition="top center",
        textfont=dict(color="#e2e8f0", size=11),
    ))
    fig.update_layout(
        **PLOTLY_BASE, height=300,
        geo=dict(
            scope="asia",
            showland=True, landcolor="#161b27",
            showocean=True, oceancolor="#0f1117",
            showcountries=True, countrycolor="#1e293b",
            showcoastlines=True, coastlinecolor="#1e293b",
            bgcolor="rgba(0,0,0,0)",
            center=dict(lat=(slat + plat) / 2, lon=(slon + plon) / 2),
            projection_scale=4,
        ),
    )
    return fig


def resilience_chart(scenarios):
    names  = [s["scenario"] for s in scenarios]
    delays = [s["avg_delay_days"] for s in scenarios]
    orders = [s["affected_orders"] for s in scenarios]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Avg Delay (days)", x=names, y=delays,
                         marker_color="#f59e0b", yaxis="y"))
    fig.add_trace(go.Bar(name="Affected Orders",  x=names, y=orders,
                         marker_color="#3b82f6",  yaxis="y2", opacity=0.6))
    fig.update_layout(
        **PLOTLY_BASE, height=280, barmode="group",
        legend=dict(orientation="h", y=1.08, x=0),
        xaxis=dict(GRID),
        yaxis=dict(GRID, title="Avg Delay (days)"),
        yaxis2=dict(overlaying="y", side="right", title="Orders", showgrid=False),
    )
    return fig


def supplier_radar(grp_data):
    """Radar chart comparing suppliers across KPIs."""
    categories = ["Reliability", "Cost Score", "Distance Score", "Speed Score"]
    fig = go.Figure()
    colors = {"Chennai": "#3b82f6", "Mumbai": "#f59e0b", "Delhi": "#10b981"}
    for _, row in grp_data.iterrows():
        name = row["Supplier"]
        # Normalise: higher = better for all axes
        rel   = row["raw_reliability"]
        cost  = 1 - min(row["raw_cost"] / 100000, 1)
        dist  = 1 - min(row["raw_dist"] / 2000, 1)
        speed = 1 - min(row["raw_dist"] / 3000, 1)
        vals  = [rel, cost, dist, speed]
        vals  += vals[:1]
        cats  = categories + categories[:1]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats, fill="toself", name=name,
            line_color=colors.get(name, "#94a3b8"),
            opacity=0.75,
        ))
    fig.update_layout(
        **PLOTLY_BASE, height=300,
        polar=dict(
            bgcolor="#161b27",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#1e293b", tickfont_color="#64748b"),
            angularaxis=dict(gridcolor="#1e293b", tickfont_color="#94a3b8"),
        ),
        legend=dict(orientation="h", y=-0.1, x=0.2),
    )
    return fig


# ════════════════════════════════════════════════════════════════════════════════
# SIDEBAR — FULL INTERACTIVE CONTROLS
# ════════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🚚 Supply Chain Intelligence")
    st.caption("ZF Group | AI Logistics Platform")
    st.divider()

    # ── Section 1: Shipment & Lead Time ───────────────────────────────────────
    st.markdown("#### 📦 Shipment Parameters")
    lead_time       = st.slider("Lead Time (days)", 1, 14, 3)
    safety_stock    = st.slider("Safety Stock (units)", 0, 500, 50, step=10)
    current_stock   = st.number_input("Current Stock (units)",
                                      min_value=0, max_value=20000, value=500, step=50)
    shipping_mode   = st.selectbox("Shipping Mode", SHIPPING_MODES,
                                   index=SHIPPING_MODES.index("Standard Class"))
    opt_mode        = st.selectbox("Optimisation Mode", OPT_MODES)

    st.divider()

    # ── Section 2: Supplier & Location ───────────────────────────────────────
    st.markdown("#### 🏭 Supplier & Location")
    supplier_pref   = st.selectbox("Preferred Supplier", SUPPLIER_LOCS)
    supplier_lead   = st.slider("Supplier Lead Time (days)", 1, 21, 5)
    supplier_rel    = st.slider("Supplier Reliability Score", 0.0, 1.0, 0.85, step=0.01)

    # Distance auto-fills from supplier; user can override
    _auto_dist = {
        "Chennai": 350, "Mumbai": 980, "Delhi": 1740,
    }
    distance_km = st.number_input(
        "Route Distance (km)",
        min_value=50, max_value=5000,
        value=_auto_dist.get(supplier_pref, 350), step=10,
    )

    st.divider()

    # ── Section 3: Demand Inputs ──────────────────────────────────────────────
    st.markdown("#### 📊 Demand Parameters")
    sales_per_cust  = st.number_input("Sales per Customer (₹)", min_value=10,
                                      max_value=10000, value=150, step=10)
    category_id     = st.selectbox("Product Category ID", list(range(1, 16)), index=0)
    lag_1           = st.number_input("Previous Day Demand (lag-1)", min_value=0,
                                      max_value=5000, value=200, step=10)
    rolling_mean    = st.number_input("3-Day Rolling Avg Demand", min_value=0,
                                      max_value=5000, value=200, step=10)

    st.divider()

    # ── Section 4: Risk Inputs ────────────────────────────────────────────────
    st.markdown("#### ⚠️ Risk & Environment")
    customer_seg    = st.selectbox("Customer Segment", CUSTOMER_SEGMENTS)
    traffic_cond    = st.selectbox("Traffic Condition", TRAFFIC_OPTIONS,
                                   index=TRAFFIC_OPTIONS.index("Medium"))
    disruption_type = st.selectbox("Active Disruption", DISRUPTION_TYPES)
    disruption_sev  = st.slider("Disruption Severity (0 = none, 5 = severe)", 0, 5, 2)

    st.divider()
    run = st.button("▶  Run AI Analysis", use_container_width=True)
    st.caption(f"Dataset: {len(df_raw):,} orders loaded")
    st.caption(f"Model: XGBoost-style HGBR + GBM Ensemble")


# ── Collect all params ─────────────────────────────────────────────────────────

params = dict(
    lead_time           = lead_time,
    safety_stock        = safety_stock,
    current_stock       = current_stock,
    shipping_mode       = shipping_mode,
    opt_mode            = opt_mode,
    supplier_location   = supplier_pref,
    supplier_lead_time  = supplier_lead,
    supplier_reliability= supplier_rel,
    distance_km         = distance_km,
    sales_per_customer  = sales_per_cust,
    category_id         = category_id,
    lag_1               = lag_1,
    rolling_mean_3      = rolling_mean,
    customer_segment    = customer_seg,
    traffic_condition   = traffic_cond,
    disruption_type     = disruption_type,
    disruption_severity = disruption_sev,
)

df_input = build_input(params)


# ── Build supplier comparison table (for tab 2) ────────────────────────────────

def build_supplier_table():
    rows = []
    for name, (lat, lon) in SUPPLIER_COORDS.items():
        from backend.services import haversine
        dist = haversine(lat, lon, PLANT_COORDS[0], PLANT_COORDS[1])
        # Mask over df_clean supplier data
        code = SUPPLIER_LOC_MAP.get(name, 0)
        sub  = df_clean[df_clean["supplier_location"] == code] if "supplier_location" in df_clean.columns else df_clean
        rel  = sub["supplier_reliability_score"].mean() if "supplier_reliability_score" in sub.columns else 0.85
        cost = sub["supplier_cost"].mean() if "supplier_cost" in sub.columns else dist * 10
        rows.append({"Supplier": name, "raw_reliability": rel, "raw_cost": cost, "raw_dist": dist,
                     "Reliability": f"{rel*100:.1f}%",
                     "Distance (km)": f"{dist:,.0f}",
                     "Avg Cost (₹)": f"₹{cost:,.0f}"})
    return pd.DataFrame(rows)


# ── Compute results ────────────────────────────────────────────────────────────

def compute_results(params, df_input):
    demand_forecast = predict_demand(demand_model, d_cols, df_input)
    delay_prob      = predict_delay(delay_model, l_cols, df_input)
    reorder         = reorder_point(demand_forecast, params["lead_time"], params["safety_stock"])
    inv_stat        = inventory_status(params["current_stock"], reorder)

    # Use user-selected supplier for routing (override best_supplier location)
    best_supplier   = select_best_supplier(df_clean, mode=params["opt_mode"])
    # Override with user preference if they explicitly selected one
    user_lat, user_lon = SUPPLIER_COORDS[params["supplier_location"]]
    from backend.services import haversine as _hav
    user_dist = _hav(user_lat, user_lon, PLANT_COORDS[0], PLANT_COORDS[1])
    user_supplier = {
        "supplier_location": params["supplier_location"],
        "distance_km":       round(user_dist, 2),
        "co2_per_km":        best_supplier["co2_per_km"],
        "reliability":       params["supplier_reliability"],
        "lat":               user_lat,
        "lon":               user_lon,
    }

    route       = route_info(user_supplier)
    emission    = calculate_emission(user_supplier["distance_km"], user_supplier["co2_per_km"])
    eco_score   = sustainability_score(emission)
    weather     = get_weather_risk(params["supplier_location"])
    traffic     = detect_spike(df_clean)
    decisions   = make_decision(demand_forecast, delay_prob, params["current_stock"], reorder, params["opt_mode"])
    scenarios   = resilience_scenarios(df_clean)
    kpis        = kpi_summary(df_raw)

    return dict(
        demand_forecast = demand_forecast,
        delay_prob      = delay_prob,
        reorder         = reorder,
        inv_status      = inv_stat,
        best_supplier   = user_supplier,
        route           = route,
        emission        = emission,
        eco_score       = eco_score,
        weather         = weather,
        traffic         = traffic,
        decisions       = decisions,
        scenarios       = scenarios,
        kpis            = kpis,
    )


# Always recompute on sidebar change; cache only on explicit run button
if run or "results" not in st.session_state:
    with st.spinner("Running XGBoost-style HGBR + GBM ensemble inference…"):
        st.session_state["results"] = compute_results(params, df_input)
        st.session_state["params"]  = params.copy()

R = st.session_state["results"]


# ── Header ─────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="display:flex;align-items:center;gap:14px;margin-bottom:4px;">
  <div>
    <div style="font-size:24px;font-weight:700;color:#f1f5f9;">🚚 Supply Chain Intelligence</div>
    <div style="font-size:13px;color:#64748b;">AI-Enhanced Logistics &amp; Supply Chain Optimization — ZF Group | XGBoost-style HGBR + GBM Ensemble</div>
  </div>
</div>
""", unsafe_allow_html=True)
st.divider()

# ── KPI Strip ──────────────────────────────────────────────────────────────────

k = R["kpis"]
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total Orders",   f"{k['total_orders']:,}")
c2.metric("Service Level",  f"{k['service_level_pct']}%")
c3.metric("Late Delivery",  f"{k['late_delivery_pct']}%")
c4.metric("Avg Demand",     f"{k['avg_demand']:,.0f}")
c5.metric("Avg CO2 (kg)",   f"{k['avg_co2_kg']}")
c6.metric("Avg Route Cost", f"₹{k['avg_route_cost']:,.0f}")

st.markdown("")

# ── Tabs ───────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Forecast & Delay",
    "🏪 Inventory & Supplier",
    "🗺️ Routing & Sustainability",
    "⚡ Resilience",
    "🤖 Decision Engine",
])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Forecast & Delay
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    col_l, col_r = st.columns([3, 2], gap="large")

    with col_l:
        st.markdown('<div class="section-title">7-Day Demand Forecast</div>', unsafe_allow_html=True)
        st.plotly_chart(demand_chart(R["demand_forecast"]), use_container_width=True,
                        config={"displayModeBar": False})

        avg_d = sum(R["demand_forecast"]) / 7
        d_min = min(R["demand_forecast"])
        d_max = max(R["demand_forecast"])
        a, b, c_ = st.columns(3)
        a.metric("Avg Daily Demand", f"{avg_d:,.0f} units")
        b.metric("Forecast Min",     f"{d_min:,.0f} units")
        c_.metric("Forecast Max",    f"{d_max:,.0f} units")

        # Inline parameter summary
        st.markdown('<div class="section-title">Forecast Inputs</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="card info">
            <strong>Model Inputs Used:</strong><br>
            Sales/Customer: <strong>₹{params['sales_per_customer']:,}</strong> &nbsp;|&nbsp;
            Category: <strong>#{params['category_id']}</strong> &nbsp;|&nbsp;
            Lag-1 Demand: <strong>{params['lag_1']:,}</strong> &nbsp;|&nbsp;
            Rolling Avg: <strong>{params['rolling_mean_3']:,}</strong><br>
            Shipping Mode: <strong>{params['shipping_mode']}</strong> &nbsp;|&nbsp;
            Segment: <strong>{params['customer_segment']}</strong> &nbsp;|&nbsp;
            Traffic: <strong>{params['traffic_condition']}</strong>
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="section-title">Delay Risk Probability</div>', unsafe_allow_html=True)
        delay_pct  = round(R["delay_prob"] * 100, 1)
        st.plotly_chart(delay_gauge(delay_pct), use_container_width=True,
                        config={"displayModeBar": False})

        risk_label = "Critical" if delay_pct > 70 else "Moderate" if delay_pct > 40 else "Low"
        badge_cls  = "critical" if delay_pct > 70 else "warning" if delay_pct > 40 else "ok"
        st.markdown(f"""
        <div class="card {badge_cls}">
            <span class="badge-{badge_cls}">{risk_label.upper()}</span>
            &nbsp; Delay probability is <strong>{delay_pct}%</strong> for a <strong>{lead_time}-day</strong>
            lead-time shipment via <strong>{params['shipping_mode']}</strong>.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">Disruption Impact</div>', unsafe_allow_html=True)
        dis_cls = "critical" if params["disruption_severity"] >= 4 \
                  else "warning" if params["disruption_severity"] >= 2 else "ok"
        st.markdown(f"""
        <div class="card {dis_cls}">
            <span class="badge-{dis_cls}">SEVERITY {params['disruption_severity']}/5</span>
            &nbsp; Active disruption: <strong>{params['disruption_type']}</strong><br>
            <span style="font-size:12px;color:#94a3b8;">
                {'No active disruption.' if params['disruption_type'] == 'None'
                 else 'Factor reflected in delay model prediction.'}
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">Traffic Conditions</div>', unsafe_allow_html=True)
        t = R["traffic"]
        tcls = "critical" if t["level"] == "High" else "warning" if t["level"] == "Moderate" else "ok"
        st.markdown(f"""
        <div class="card {tcls}">
            <span class="badge-{tcls}">{t['level'].upper()}</span>
            &nbsp; High-traffic routes: <strong>{t['high_pct']}%</strong> of network
            &nbsp;|&nbsp; Selected: <strong>{params['traffic_condition']}</strong>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Inventory & Supplier
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown('<div class="section-title">Inventory Status</div>', unsafe_allow_html=True)

        inv_cls = {
            "SAFE":     "ok", "ADEQUATE": "ok",
            "LOW":      "warning", "CRITICAL": "critical",
        }.get(R["inv_status"], "info")

        fill_ratio = min(params["current_stock"] / max(R["reorder"], 1), 1.5)
        st.progress(min(fill_ratio / 1.5, 1.0))

        i1, i2 = st.columns(2)
        i1.metric("Current Stock", f"{params['current_stock']:,} units")
        i2.metric("Reorder Point", f"{R['reorder']:,.0f} units")
        cover_days = round(params["current_stock"] / max(R["reorder"] / max(params["lead_time"], 1), 1), 1)
        st.metric("Coverage Estimate", f"~{cover_days} days")

        st.markdown(f"""
        <div class="card {inv_cls}">
            <span class="badge-{inv_cls}">{R['inv_status']}</span>
            &nbsp; {'Stock is sufficient.' if inv_cls == 'ok'
                    else 'Review stock levels and consider replenishment.'}
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">Safety Stock Policy</div>', unsafe_allow_html=True)
        avg_fc = sum(R["demand_forecast"]) / 7
        st.markdown(f"""
        <div class="card info">
            <strong>Reorder Point Formula:</strong><br>
            ROP = Avg Demand × Lead Time + Safety Stock<br>
            &nbsp;&nbsp;&nbsp; = {avg_fc:,.1f} × {params['lead_time']} + {params['safety_stock']}
            &nbsp; = <strong>{R['reorder']:,.0f} units</strong><br><br>
            <span style="font-size:12px;color:#64748b;">
                Safety stock set to <strong>{params['safety_stock']} units</strong>.
                Adjust via sidebar slider.
            </span>
        </div>
        """, unsafe_allow_html=True)

        # Inventory sensitivity mini-table
        st.markdown('<div class="section-title">Sensitivity: Safety Stock vs ROP</div>', unsafe_allow_html=True)
        ss_range = [0, 50, 100, 150, 200, params["safety_stock"]]
        ss_df = pd.DataFrame({
            "Safety Stock": [f"{s} units" for s in sorted(set(ss_range))],
            "Reorder Point": [f"{round(avg_fc * params['lead_time'] + s):,}" for s in sorted(set(ss_range))],
        })
        st.dataframe(ss_df, use_container_width=True, hide_index=True)

    with col_r:
        st.markdown('<div class="section-title">Selected Supplier</div>', unsafe_allow_html=True)
        s = R["best_supplier"]

        s1, s2 = st.columns(2)
        s1.metric("Supplier",    s["supplier_location"])
        s2.metric("Reliability", f"{s['reliability']*100:.1f}%")
        s3, s4 = st.columns(2)
        s3.metric("Distance",    f"{s['distance_km']} km")
        s4.metric("CO2/km",      f"{s['co2_per_km']} kg")

        w    = R["weather"]
        wcls = "critical" if w["risk"] == "High" else "warning" if w["risk"] == "Medium" else "ok"
        st.markdown(f"""
        <div class="card {wcls}" style="margin-top:16px;">
            <strong>☁️ Weather at {w['location']}</strong><br>
            Condition: {w['condition']} &nbsp;|&nbsp; Risk: <span class="badge-{wcls}">{w['risk'].upper()}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">All Suppliers — Comparison</div>', unsafe_allow_html=True)
        sup_df = build_supplier_table()

        # Highlight selected
        display_df = sup_df[["Supplier", "Reliability", "Distance (km)", "Avg Cost (₹)"]].copy()
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.markdown('<div class="section-title">Supplier Capability Radar</div>', unsafe_allow_html=True)
        st.plotly_chart(supplier_radar(sup_df), use_container_width=True,
                        config={"displayModeBar": False})


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — Routing & Sustainability
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    col_l, col_r = st.columns([3, 2], gap="large")

    with col_l:
        st.markdown('<div class="section-title">Route Map</div>', unsafe_allow_html=True)
        s = R["best_supplier"]
        fig_map = route_map(s["supplier_location"], s["lat"], s["lon"],
                            PLANT_COORDS[0], PLANT_COORDS[1])
        st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})

        r = R["route"]
        r1, r2, r3 = st.columns(3)
        r1.metric("Origin",       r["origin"])
        r2.metric("Distance",     f"{r['distance_km']} km")
        r3.metric("Est. Transit", f"{r['duration_hrs']} hrs")

        st.markdown('<div class="section-title">Route Details</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="card info">
            <strong>{r['origin']}</strong> → <strong>{r['destination']}</strong><br>
            Distance: <strong>{r['distance_km']} km</strong> &nbsp;|&nbsp;
            Avg Speed: <strong>60 km/h</strong> &nbsp;|&nbsp;
            Transit: <strong>{r['duration_hrs']} hrs</strong><br>
            Vehicle: <strong>Road Freight</strong> &nbsp;|&nbsp;
            Mode: <strong>{opt_mode}</strong>
        </div>
        """, unsafe_allow_html=True)

        # All-routes comparison
        st.markdown('<div class="section-title">All Routes at a Glance</div>', unsafe_allow_html=True)
        from backend.services import haversine as _hav, calculate_emission as _ce, sustainability_score as _ss
        route_rows = []
        for nm, (la, lo) in SUPPLIER_COORDS.items():
            d = round(_hav(la, lo, PLANT_COORDS[0], PLANT_COORDS[1]), 1)
            em = _ce(d, R["best_supplier"]["co2_per_km"])
            sc = _ss(em)
            route_rows.append({
                "Supplier": nm,
                "Distance (km)": d,
                "Transit (hrs)": round(d / 60, 1),
                "CO2 (kg)": em,
                "Eco Score": sc,
            })
        st.dataframe(pd.DataFrame(route_rows), use_container_width=True, hide_index=True)

    with col_r:
        st.markdown('<div class="section-title">Emissions & Eco Score</div>', unsafe_allow_html=True)
        st.plotly_chart(co2_bar(R["emission"], R["eco_score"]),
                        use_container_width=True, config={"displayModeBar": False})

        e1, e2 = st.columns(2)
        e1.metric("CO2 Emission", f"{R['emission']} kg")
        e2.metric("Eco Score",    f"{R['eco_score']} / 100")

        eco_cls = "ok" if R["eco_score"] > 70 else "warning" if R["eco_score"] > 40 else "critical"
        eco_txt = ("This route has a low carbon footprint." if R["eco_score"] > 70
                   else "Consider consolidating loads or shifting to rail." if R["eco_score"] > 40
                   else "High emissions detected. Switch to eco-friendly routing.")
        st.markdown(f"""
        <div class="card {eco_cls}">
            <span class="badge-{eco_cls}">
                {'ECO ✓' if R['eco_score']>70 else 'REVIEW' if R['eco_score']>40 else 'HIGH EMISSION'}
            </span>
            &nbsp; {eco_txt}
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">Optimisation Mode</div>', unsafe_allow_html=True)
        mode_desc = {
            "Cost Efficient": "Batch shipments using standard road transport to minimise per-unit cost.",
            "Fast Delivery":  "Air freight or express road for time-critical SKUs.",
            "Eco Friendly":   "Rail and water transport preferred. Shipments consolidated to reduce CO2.",
        }
        st.markdown(f"""
        <div class="card info">
            <strong>{opt_mode}</strong><br>{mode_desc[opt_mode]}
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">CO2 by Shipping Mode</div>', unsafe_allow_html=True)
        mode_mult = {"Standard Class": 1.0, "Second Class": 1.1, "First Class": 1.4, "Same Day": 2.0}
        mode_rows = [{"Shipping Mode": m, "CO2 Multiplier": f"{v}×",
                      "Est. Emission (kg)": round(R["emission"] * v, 1)}
                     for m, v in mode_mult.items()]
        st.dataframe(pd.DataFrame(mode_rows), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — Resilience
# ════════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">Disruption Scenario Analysis</div>', unsafe_allow_html=True)
    st.plotly_chart(resilience_chart(R["scenarios"]), use_container_width=True,
                    config={"displayModeBar": False})

    for sc in R["scenarios"]:
        sev = sc["avg_delay_days"]
        cls = "critical" if sev > 5 else "warning" if sev > 3 else "info"
        active = "🔴 ACTIVE" if sc["scenario"] == params["disruption_type"] else ""
        st.markdown(f"""
        <div class="card {cls}">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                <strong style="font-size:14px;">{sc['scenario']} {active}</strong>
                <span style="font-size:12px;color:#64748b;">
                    {sc['affected_orders']:,} orders &nbsp;|&nbsp; Avg delay: {sc['avg_delay_days']} days
                </span>
            </div>
            <div style="color:#94a3b8;font-size:13px;">{sc['recommendation']}</div>
        </div>
        """, unsafe_allow_html=True)

    col_kl, col_kr = st.columns(2, gap="large")
    with col_kl:
        st.markdown('<div class="section-title">KPI Baseline vs AI Model</div>', unsafe_allow_html=True)
        baseline_data = {
            "Metric":         ["Service Level", "Late Delivery", "Avg Demand", "Avg CO2 (kg)", "Avg Route Cost"],
            "Baseline (ROP)": ["82%", "18%", "180 units", "95 kg", "₹4,200"],
            "AI Model":       [
                f"{R['kpis']['service_level_pct']}%",
                f"{R['kpis']['late_delivery_pct']}%",
                f"{R['kpis']['avg_demand']:,.0f} units",
                f"{R['kpis']['avg_co2_kg']} kg",
                f"₹{R['kpis']['avg_route_cost']:,.0f}",
            ],
        }
        st.dataframe(pd.DataFrame(baseline_data), use_container_width=True, hide_index=True)

    with col_kr:
        st.markdown('<div class="section-title">Your Session — Risk Summary</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="card {'critical' if disruption_sev >= 4 else 'warning' if disruption_sev >= 2 else 'ok'}">
            <strong>Current Disruption:</strong> {params['disruption_type']} (Severity {disruption_sev}/5)<br>
            <strong>Traffic:</strong> {params['traffic_condition']}<br>
            <strong>Supplier Reliability:</strong> {params['supplier_reliability']*100:.0f}%<br>
            <strong>Delay Probability:</strong> {round(R['delay_prob']*100,1)}%<br>
            <strong>Inventory Status:</strong> {R['inv_status']}
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — Decision Engine
# ════════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-title">AI Prescriptive Recommendations</div>', unsafe_allow_html=True)

    for d in R["decisions"]:
        label_map = {"critical": "Action Required", "warning": "Warning", "ok": "OK", "info": "Info"}
        icon_map  = {"critical": "🔴", "warning": "🟡", "ok": "🟢", "info": "🔵"}
        st.markdown(f"""
        <div class="card {d['level']}">
            <span class="badge-{d['level']}">{icon_map.get(d['level'],'')} {label_map.get(d['level'],'').upper()}</span>
            &nbsp; {d['text']}
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    col_dl, col_dr = st.columns(2, gap="large")

    with col_dl:
        st.markdown('<div class="section-title">Full Model Inputs Summary</div>', unsafe_allow_html=True)
        summary = pd.DataFrame({
            "Parameter": [
                "Lead Time", "Safety Stock", "Current Stock",
                "Shipping Mode", "Optimisation Mode",
                "Supplier", "Supplier Lead Time", "Supplier Reliability",
                "Route Distance", "Sales per Customer",
                "Category ID", "Customer Segment",
                "Traffic Condition", "Disruption Type", "Disruption Severity",
                "Lag-1 Demand", "Rolling Avg (3d)",
                "Demand (avg forecast)", "Delay Probability",
                "Reorder Point", "Inventory Status",
            ],
            "Value": [
                f"{params['lead_time']} days",
                f"{params['safety_stock']} units",
                f"{params['current_stock']:,} units",
                params["shipping_mode"],
                params["opt_mode"],
                params["supplier_location"],
                f"{params['supplier_lead_time']} days",
                f"{params['supplier_reliability']*100:.0f}%",
                f"{params['distance_km']:,} km",
                f"₹{params['sales_per_customer']:,}",
                f"#{params['category_id']}",
                params["customer_segment"],
                params["traffic_condition"],
                params["disruption_type"],
                f"{params['disruption_severity']}/5",
                f"{params['lag_1']:,} units",
                f"{params['rolling_mean_3']:,} units",
                f"{sum(R['demand_forecast'])/7:,.0f} units/day",
                f"{R['delay_prob']*100:.1f}%",
                f"{R['reorder']:,.0f} units",
                R["inv_status"],
            ],
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

    with col_dr:
        st.markdown('<div class="section-title">7-Day Demand Forecast Detail</div>', unsafe_allow_html=True)
        forecast_df = pd.DataFrame({
            "Day":              [f"Day {i+1}" for i in range(7)],
            "Forecast (units)": [f"{v:,.0f}" for v in R["demand_forecast"]],
            "vs Avg":           [f"{'▲' if v > sum(R['demand_forecast'])/7 else '▼'} {abs(v - sum(R['demand_forecast'])/7):,.0f}"
                                 for v in R["demand_forecast"]],
        })
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)

        st.markdown('<div class="section-title">Model Architecture</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card info">
            <strong>Demand Model</strong><br>
            VotingRegressor ensemble:<br>
            &nbsp;• HistGradientBoostingRegressor (XGBoost-style, 300 iters)<br>
            &nbsp;• GradientBoostingRegressor (150 estimators)<br><br>
            <strong>Delay Model</strong><br>
            &nbsp;• HistGradientBoostingClassifier (XGBoost-style, 300 iters)<br><br>
            Both models use histogram-based gradient boosting with L2 regularisation,
            equivalent to XGBoost's tree method.
        </div>
        """, unsafe_allow_html=True)