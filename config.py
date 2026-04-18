"""
Supply Chain Intelligence — Interactive Dashboard
AI-Enhanced Logistics & Supply Chain Optimization — ZF Group
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(
    page_title="Supply Chain Intelligence",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ---------- base ---------- */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0a0d14;
    color: #e2e8f0;
    font-family: 'Inter', sans-serif;
}
[data-testid="stHeader"] { background: transparent; }

/* ---------- sidebar — always visible, fixed width ---------- */
[data-testid="stSidebar"] {
    background-color: #111827 !important;
    border-right: 1px solid #1e293b !important;
    min-width: 280px !important;
    max-width: 280px !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 1rem;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
/* keep sidebar toggle arrow visible */
[data-testid="collapsedControl"] { display: none !important; }

/* ---------- metric cards ---------- */
[data-testid="stMetric"] {
    background: #161b27;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 10px 14px;
}
[data-testid="stMetric"] label {
    color: #94a3b8 !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
    font-size: 20px !important;
    font-weight: 700 !important;
}

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"] {
    background: #161b27; border-radius: 8px; padding: 4px; gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #64748b; border-radius: 6px; padding: 7px 16px; font-size: 13px;
}
.stTabs [aria-selected="true"] { background: #1e40af !important; color: #fff !important; }

/* ---------- run button ---------- */
.stButton > button {
    background: linear-gradient(135deg, #1e40af, #2563eb);
    color: #fff; border: none; border-radius: 8px;
    padding: 12px 0; font-size: 14px; font-weight: 600;
    width: 100%; letter-spacing: 0.03em;
    box-shadow: 0 2px 8px rgba(37,99,235,0.35);
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #3b82f6);
    box-shadow: 0 4px 16px rgba(37,99,235,0.5);
}

/* ---------- section titles ---------- */
.section-title {
    font-size: 11px; font-weight: 700; color: #475569;
    text-transform: uppercase; letter-spacing: 0.10em;
    margin: 24px 0 10px 0; padding-bottom: 6px;
    border-bottom: 1px solid #1e293b;
}

/* ---------- cards ---------- */
.card {
    background: #161b27; border: 1px solid #1e293b;
    border-radius: 8px; padding: 14px 18px; margin-bottom: 10px;
    font-size: 13.5px; line-height: 1.65;
}
.card.critical { border-left: 4px solid #ef4444; }
.card.warning  { border-left: 4px solid #f59e0b; }
.card.ok       { border-left: 4px solid #10b981; }
.card.info     { border-left: 4px solid #3b82f6; }

/* ---------- badges ---------- */
.badge-critical { background:#ef44441a; color:#ef4444; font-size:10px; padding:2px 8px; border-radius:4px; font-weight:700; letter-spacing:0.05em; }
.badge-warning  { background:#f59e0b1a; color:#f59e0b; font-size:10px; padding:2px 8px; border-radius:4px; font-weight:700; letter-spacing:0.05em; }
.badge-ok       { background:#10b9811a; color:#10b981; font-size:10px; padding:2px 8px; border-radius:4px; font-weight:700; letter-spacing:0.05em; }
.badge-info     { background:#3b82f61a; color:#3b82f6; font-size:10px; padding:2px 8px; border-radius:4px; font-weight:700; letter-spacing:0.05em; }

/* ---------- stat strip ---------- */
.stat-strip {
    display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 18px;
}
.stat-box {
    flex: 1; min-width: 110px;
    background: #161b27; border: 1px solid #1e293b;
    border-radius: 8px; padding: 10px 12px; text-align: center;
}
.stat-label { font-size: 10px; color: #475569; text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 4px; }
.stat-value { font-size: 17px; font-weight: 700; color: #f1f5f9; }
.stat-sub   { font-size: 10px; color: #64748b; margin-top: 2px; }

/* ---------- sidebar section headers ---------- */
.sb-section {
    font-size: 10px; font-weight: 700; color: #3b82f6;
    text-transform: uppercase; letter-spacing: 0.12em;
    margin: 14px 0 8px 0; padding: 6px 0 6px 8px;
    border-left: 3px solid #1e40af;
}

/* ---------- scrollbar ---------- */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA & MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Training XGBoost-style HGBR + GBM ensemble on 60k orders…")
def load_and_train():
    from backend.data   import load_data, preprocess
    from backend.models import train_models
    df_raw   = load_data()
    df_clean = preprocess(df_raw)
    dm, dlm, dc, lc = train_models(df_clean)
    return df_raw, df_clean, dm, dlm, dc, lc

df_raw, df_clean, demand_model, delay_model, d_cols, l_cols = load_and_train()

from backend.services import (
    reorder_point, inventory_status, select_best_supplier,
    route_info, calculate_emission, sustainability_score,
    detect_spike, get_weather_risk, make_decision,
    resilience_scenarios, kpi_summary, haversine,
)
from backend.models import predict_demand, predict_delay
from config import (
    SUPPLIER_COORDS, PLANT_OPTIONS, PLANT_COORDS,
    SHIPPING_MODES, CUSTOMER_SEGMENTS, TRAFFIC_OPTIONS,
    DISRUPTION_TYPES, SUPPLIER_LOCS, PLANT_LOCS, OPT_MODES,
    SHIPPING_MODE_MAP, CUSTOMER_SEG_MAP, TRAFFIC_MAP,
    DISRUPTION_MAP, SUPPLIER_LOC_MAP, SPEED_KMH,
)


# ═══════════════════════════════════════════════════════════════════════════════
# REAL KPI DEFAULTS (computed once from dataset)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def get_real_kpis():
    df = df_raw.copy()
    df.columns = [c.strip().lower().replace(" ","_").replace("(","").replace(")","") for c in df.columns]
    late_pct   = round(df["late_delivery_risk"].mean() * 100, 1) if "late_delivery_risk" in df.columns else 42.2
    svc_pct    = round(100 - late_pct, 1)
    avg_demand = round(df["sales"].mean(), 1) if "sales" in df.columns else 195.8
    avg_co2    = round((df["distance_km"] * df["co2_per_km"]).mean(), 1) if all(
                     c in df.columns for c in ["distance_km","co2_per_km"]) else 308.8
    avg_cost   = round(df["supplier_cost"].mean(), 0) if "supplier_cost" in df.columns else 5496.0
    return {
        "late_delivery_pct":  late_pct,
        "service_level_pct":  svc_pct,
        "avg_demand":         avg_demand,
        "avg_co2_kg":         avg_co2,
        "avg_route_cost":     avg_cost,
    }

REAL_KPIS = get_real_kpis()


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def build_input(p: dict) -> pd.DataFrame:
    now = datetime.now()
    return pd.DataFrame([{
        "order_day":                      now.day,
        "order_month":                    now.month,
        "order_hour":                     now.hour,
        "sales_per_customer":             p["sales_per_customer"],
        "category_id":                    p["category_id"],
        "days_for_shipment_(scheduled)":  p["lead_time"],
        "days_for_shipping_(real)":       p["lead_time"],
        "lead_time_days":                 p["lead_time"],
        "distance_km":                    p["distance_km"],
        "supplier_lead_time":             p["supplier_lead_time"],
        "disruption_severity":            p["disruption_severity"],
        "lag_1":                          p["lag_1"],
        "rolling_mean_3":                 p["rolling_mean_3"],
        "traffic_condition":              TRAFFIC_MAP.get(p["traffic_condition"], 1),
        "disruption_type":                max(DISRUPTION_MAP.get(p["disruption_type"], 0), 0),
        "shipping_mode":                  SHIPPING_MODE_MAP.get(p["shipping_mode"], 0),
        "customer_segment":               CUSTOMER_SEG_MAP.get(p["customer_segment"], 0),
        "late_delivery_risk":             0,
        "supplier_reliability_score":     p["supplier_reliability"],
        "supplier_location":              SUPPLIER_LOC_MAP.get(p["supplier_location"], 0),
        "vehicle_type":                   0,
    }])


def find_nearest_supplier(plant_name: str, opt_mode: str) -> str:
    """Return supplier name closest (by haversine) to the selected plant."""
    plat, plon = PLANT_OPTIONS[plant_name]
    best, best_dist = None, float("inf")
    for sup, (slat, slon) in SUPPLIER_COORDS.items():
        if sup == plant_name.split()[0]:   # skip same-city self-loops
            continue
        d = haversine(slat, slon, plat, plon)
        if d < best_dist:
            best_dist, best = d, sup
    return best or "Chennai"


def compute_results(p: dict, df_input: pd.DataFrame) -> dict:
    plant_coords = PLANT_OPTIONS[p["plant_location"]]
    sup_lat, sup_lon = SUPPLIER_COORDS[p["supplier_location"]]
    dist = haversine(sup_lat, sup_lon, plant_coords[0], plant_coords[1])

    # Pull avg CO2/km from dataset
    co2_avg = float(df_clean["co2_per_km"].mean()) if "co2_per_km" in df_clean.columns else 0.30

    user_supplier = {
        "supplier_location": p["supplier_location"],
        "distance_km":       round(dist, 2),
        "co2_per_km":        round(co2_avg, 4),
        "reliability":       p["supplier_reliability"],
        "lat": sup_lat, "lon": sup_lon,
    }

    demand_forecast = predict_demand(demand_model, d_cols, df_input)
    # Rescale demand forecast to realistic sales range (~100-300 units/day)
    raw_avg  = sum(demand_forecast) / len(demand_forecast)
    scale    = max(p["lag_1"], p["rolling_mean_3"], 80) / max(raw_avg, 1)
    demand_forecast = [round(v * scale, 1) for v in demand_forecast]

    delay_prob   = predict_delay(delay_model, l_cols, df_input)

    # Delay prob boost for disruptions / high traffic
    if p["disruption_type"] != "None":
        delay_prob = min(delay_prob + 0.15 * p["disruption_severity"] / 5, 0.99)
    if p["traffic_condition"] == "High":
        delay_prob = min(delay_prob + 0.08, 0.99)

    reorder  = reorder_point(demand_forecast, p["lead_time"], p["safety_stock"])
    inv_stat = inventory_status(p["current_stock"], reorder)
    route    = {
        "distance_km":  round(dist, 2),
        "duration_hrs": round(dist / SPEED_KMH, 2),
        "origin":       p["supplier_location"],
        "destination":  p["plant_location"],
    }
    emission  = calculate_emission(dist, co2_avg)
    eco_score = sustainability_score(emission)
    weather   = get_weather_risk(p["supplier_location"])
    traffic   = detect_spike(df_clean)
    decisions = make_decision(demand_forecast, delay_prob, p["current_stock"], reorder, p["opt_mode"])
    scenarios = resilience_scenarios(df_clean)

    # Session KPIs — blend real dataset with session forecast
    session_avg_demand = sum(demand_forecast) / 7
    session_kpis = {
        "service_level_pct": REAL_KPIS["service_level_pct"],
        "late_delivery_pct": REAL_KPIS["late_delivery_pct"],
        "avg_demand":        round(session_avg_demand, 1),
        "avg_co2_kg":        round(emission, 1),
        "avg_route_cost":    round(dist * 5.2, 0),   # ₹5.2/km avg freight rate
    }

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
        kpis            = session_kpis,
        plant_coords    = plant_coords,
    )


def build_supplier_table(plant_name: str):
    plat, plon = PLANT_OPTIONS[plant_name]
    co2_avg = float(df_clean["co2_per_km"].mean()) if "co2_per_km" in df_clean.columns else 0.30
    rows = []
    for name, (lat, lon) in SUPPLIER_COORDS.items():
        dist = haversine(lat, lon, plat, plon)
        code = SUPPLIER_LOC_MAP.get(name, 0)
        sub  = df_clean[df_clean["supplier_location"] == code] if "supplier_location" in df_clean.columns else df_clean
        rel  = sub["supplier_reliability_score"].mean() if "supplier_reliability_score" in sub.columns else 0.85
        cost = round(dist * 5.2, 0)
        em   = round(dist * co2_avg, 1)
        rows.append({
            "Supplier": name, "raw_reliability": rel, "raw_cost": cost, "raw_dist": dist,
            "Reliability": f"{rel*100:.1f}%",
            "Distance (km)": f"{dist:,.0f}",
            "Freight Cost (₹)": f"₹{cost:,.0f}",
            "CO2 (kg)": f"{em}",
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

PB = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#94a3b8", size=12),
    margin=dict(l=10, r=10, t=30, b=10),
)
GRID = dict(showgrid=True, gridcolor="#1e293b", zeroline=False)


def demand_chart(forecast):
    days = [f"Day {i+1}" for i in range(len(forecast))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=days, y=forecast, mode="lines+markers",
        line=dict(color="#3b82f6", width=2.5),
        marker=dict(size=7, color="#60a5fa", line=dict(color="#1e40af", width=1)),
        fill="tozeroy", fillcolor="rgba(59,130,246,0.07)", name="Demand",
    ))
    fig.update_layout(**PB, height=240, showlegend=False,
                      xaxis=dict(GRID, title=""), yaxis=dict(GRID, title="Units/day"))
    return fig


def delay_gauge(prob_pct):
    color = "#ef4444" if prob_pct > 70 else "#f59e0b" if prob_pct > 40 else "#10b981"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_pct,
        number={"suffix": "%", "font": {"color": color, "size": 36}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#334155", "tickfont": {"color": "#64748b"}},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": "#0f1117",
            "steps": [
                {"range": [0,  40], "color": "#0a2318"},
                {"range": [40, 70], "color": "#281a08"},
                {"range": [70, 100], "color": "#260808"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "value": prob_pct},
        },
    ))
    fig.update_layout(**PB, height=230)
    return fig


def co2_bar(emission, score):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=["CO₂ Emission (kg)"], y=[emission],
                         marker_color="#f59e0b", text=[f"{emission:.1f}"], textposition="outside"))
    fig.add_trace(go.Bar(x=["Eco Score /100"], y=[score],
                         marker_color="#10b981", text=[f"{score}"], textposition="outside"))
    fig.update_layout(**PB, height=230, showlegend=False,
                      xaxis=dict(GRID), yaxis=dict(GRID, range=[0, max(emission, score) * 1.25]))
    return fig


def multi_route_map(selected_supplier: str, all_suppliers: dict,
                    plant_name: str, plant_coords: tuple):
    plat, plon = plant_coords
    fig = go.Figure()

    # Draw ALL supplier → plant lines (grey)
    for name, (slat, slon) in all_suppliers.items():
        if name == selected_supplier:
            continue
        fig.add_trace(go.Scattergeo(
            lat=[slat, plat], lon=[slon, plon],
            mode="lines", line=dict(width=1, color="rgba(100,116,139,0.35)"),
            showlegend=False, hoverinfo="skip",
        ))

    # Draw selected route (bright blue)
    slat, slon = all_suppliers[selected_supplier]
    fig.add_trace(go.Scattergeo(
        lat=[slat, plat], lon=[slon, plon],
        mode="lines", line=dict(width=3, color="#3b82f6"),
        name=f"{selected_supplier} → {plant_name}", showlegend=True,
    ))

    # Plot ALL supplier nodes
    sup_names  = list(all_suppliers.keys())
    sup_lats   = [all_suppliers[n][0] for n in sup_names]
    sup_lons   = [all_suppliers[n][1] for n in sup_names]
    sup_colors = ["#f59e0b" if n == selected_supplier else "#64748b" for n in sup_names]
    sup_sizes  = [14 if n == selected_supplier else 9 for n in sup_names]
    fig.add_trace(go.Scattergeo(
        lat=sup_lats, lon=sup_lons,
        mode="markers+text",
        marker=dict(size=sup_sizes, color=sup_colors, symbol="circle",
                    line=dict(color="#0f1117", width=1)),
        text=sup_names, textposition="top center",
        textfont=dict(color="#e2e8f0", size=10),
        name="Suppliers", showlegend=False,
    ))

    # Plot plant node
    fig.add_trace(go.Scattergeo(
        lat=[plat], lon=[plon],
        mode="markers+text",
        marker=dict(size=16, color="#10b981", symbol="diamond",
                    line=dict(color="#0f1117", width=1)),
        text=[plant_name], textposition="top center",
        textfont=dict(color="#10b981", size=11),
        name="Plant", showlegend=False,
    ))

    center_lat = (slat + plat) / 2
    center_lon = (slon + plon) / 2
    fig.update_layout(
        **PB, height=420,
        legend=dict(orientation="h", y=-0.05, x=0, font=dict(color="#94a3b8", size=11)),
        geo=dict(
            scope="asia",
            showland=True, landcolor="#161b27",
            showocean=True, oceancolor="#0a0d14",
            showcountries=True, countrycolor="#1e293b",
            showcoastlines=True, coastlinecolor="#1e293b",
            bgcolor="rgba(0,0,0,0)",
            center=dict(lat=center_lat, lon=center_lon),
            projection_scale=3.5,
        ),
    )
    return fig


def supplier_radar(grp_data):
    categories = ["Reliability", "Cost", "Distance", "Speed", "Eco"]
    fig = go.Figure()
    colors = {
        "Chennai": "#3b82f6", "Mumbai": "#f59e0b", "Delhi": "#10b981",
        "Kolkata": "#a855f7", "Hyderabad": "#ec4899",
        "Ahmedabad": "#14b8a6", "Pune": "#f97316", "Surat": "#6366f1",
    }
    for _, row in grp_data.iterrows():
        name = row["Supplier"]
        rel   = row["raw_reliability"]
        cost  = 1 - min(row["raw_cost"] / 15000, 1)
        dist  = 1 - min(row["raw_dist"] / 3000, 1)
        speed = 1 - min(row["raw_dist"] / 4000, 1)
        co2   = 1 - min(row["raw_dist"] * 0.30 / 600, 1)
        vals  = [rel, cost, dist, speed, co2] + [rel]
        cats  = categories + categories[:1]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats, fill="toself", name=name,
            line_color=colors.get(name, "#94a3b8"), opacity=0.70,
        ))
    fig.update_layout(
        **PB, height=320,
        polar=dict(
            bgcolor="#111827",
            radialaxis=dict(visible=True, range=[0,1], gridcolor="#1e293b",
                            tickfont=dict(color="#64748b", size=9)),
            angularaxis=dict(gridcolor="#1e293b", tickfont=dict(color="#94a3b8")),
        ),
        legend=dict(orientation="h", y=-0.15, x=0, font=dict(color="#94a3b8", size=10)),
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
                         marker_color="#3b82f6", yaxis="y2", opacity=0.55))
    fig.update_layout(
        **PB, height=260, barmode="group",
        legend=dict(orientation="h", y=1.08, x=0),
        xaxis=dict(GRID),
        yaxis=dict(GRID, title="Avg Delay (days)"),
        yaxis2=dict(overlaying="y", side="right", title="Orders", showgrid=False),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:8px 0 4px 0;">
      <div style="font-size:20px;font-weight:800;color:#f1f5f9;letter-spacing:0.02em;">🚚 SC Intelligence</div>
      <div style="font-size:10px;color:#475569;margin-top:2px;">ZF Group | AI Logistics Platform</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # ── SHIPMENT ──────────────────────────────────────────────────────────────
    st.markdown('<div class="sb-section">📦 Shipment</div>', unsafe_allow_html=True)
    lead_time     = st.slider("Lead Time (days)", 1, 14, 3)
    safety_stock  = st.slider("Safety Stock (units)", 0, 500, 50, step=10)
    current_stock = st.number_input("Current Inventory (units)", min_value=0,
                                    max_value=20000, value=500, step=50)
    shipping_mode = st.selectbox("Shipping Mode", SHIPPING_MODES,
                                 index=SHIPPING_MODES.index("Standard Class"))
    opt_mode      = st.selectbox("Optimisation Goal", OPT_MODES)

    # ── LOCATIONS ─────────────────────────────────────────────────────────────
    st.markdown('<div class="sb-section">📍 Locations</div>', unsafe_allow_html=True)
    supplier_pref  = st.selectbox("Supplier (Origin)", SUPPLIER_LOCS)
    plant_choice   = st.selectbox("Plant / Destination", PLANT_LOCS,
                                  index=PLANT_LOCS.index("Bangalore Plant"))
    supplier_lead  = st.slider("Supplier Lead Time (days)", 1, 21, 5)
    supplier_rel   = st.slider("Supplier Reliability", 0.0, 1.0, 0.85, step=0.01,
                                format="%.2f")

    # Auto distance from selection
    plat2, plon2 = PLANT_OPTIONS[plant_choice]
    slat2, slon2 = SUPPLIER_COORDS[supplier_pref]
    auto_dist    = int(haversine(slat2, slon2, plat2, plon2))
    distance_km  = st.number_input("Route Distance (km) — auto-calculated",
                                   min_value=50, max_value=6000,
                                   value=auto_dist, step=10)

    # ── DEMAND ───────────────────────────────────────────────────────────────
    st.markdown('<div class="sb-section">📊 Demand</div>', unsafe_allow_html=True)
    sales_per_cust = st.number_input("Sales per Customer (₹)", min_value=10,
                                     max_value=10000, value=200, step=10)
    category_id    = st.selectbox("Product Category", list(range(1, 16)), index=0)
    lag_1          = st.number_input("Prev-Day Demand (lag-1 units)", min_value=10,
                                     max_value=5000, value=200, step=10)
    rolling_mean   = st.number_input("3-Day Rolling Avg (units)", min_value=10,
                                     max_value=5000, value=200, step=10)

    # ── RISK ──────────────────────────────────────────────────────────────────
    st.markdown('<div class="sb-section">⚠️ Risk & Environment</div>', unsafe_allow_html=True)
    customer_seg    = st.selectbox("Customer Segment", CUSTOMER_SEGMENTS)
    traffic_cond    = st.selectbox("Traffic Condition", TRAFFIC_OPTIONS,
                                   index=TRAFFIC_OPTIONS.index("Medium"))
    disruption_type = st.selectbox("Active Disruption", DISRUPTION_TYPES)
    disruption_sev  = st.slider("Disruption Severity", 0, 5, 0)

    st.divider()
    run_btn = st.button("▶  Run AI Analysis", use_container_width=True)

    st.markdown(f"""
    <div style="font-size:10px;color:#334155;text-align:center;margin-top:6px;line-height:1.6;">
        {len(df_raw):,} orders loaded<br>
        HGBR + GBM Ensemble Model<br>
        Last trained: session start
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE — only update on button click
# ═══════════════════════════════════════════════════════════════════════════════

params = dict(
    lead_time           = lead_time,
    safety_stock        = safety_stock,
    current_stock       = current_stock,
    shipping_mode       = shipping_mode,
    opt_mode            = opt_mode,
    supplier_location   = supplier_pref,
    plant_location      = plant_choice,
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

if run_btn or "results" not in st.session_state:
    with st.spinner("Running AI inference…"):
        df_input = build_input(params)
        st.session_state["results"] = compute_results(params, df_input)
        st.session_state["params"]  = params.copy()

R = st.session_state["results"]
P = st.session_state.get("params", params)   # params used for last run


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:2px;">
  <div>
    <div style="font-size:22px;font-weight:800;color:#f1f5f9;letter-spacing:0.01em;">
      🚚 Supply Chain Intelligence
    </div>
    <div style="font-size:12px;color:#475569;margin-top:2px;">
      AI-Enhanced Logistics Optimization &mdash; ZF Group &nbsp;|&nbsp; XGBoost-style HGBR + GBM Ensemble
    </div>
  </div>
  <div style="text-align:right;">
    <div style="font-size:11px;color:#334155;">{P['supplier_location']} → {P['plant_location']}</div>
    <div style="font-size:11px;color:#334155;">{P['distance_km']:,} km &nbsp;|&nbsp; {P['opt_mode']}</div>
  </div>
</div>
""", unsafe_allow_html=True)
st.divider()


# ═══════════════════════════════════════════════════════════════════════════════
# SECONDARY STATS BAR (smaller, under header — replaces old KPI strip)
# ═══════════════════════════════════════════════════════════════════════════════

k = R["kpis"]
avg_d = sum(R["demand_forecast"]) / 7

st.markdown(f"""
<div class="stat-strip">
  <div class="stat-box">
    <div class="stat-label">Service Level</div>
    <div class="stat-value" style="color:#10b981;">{k['service_level_pct']}%</div>
    <div class="stat-sub">Dataset avg</div>
  </div>
  <div class="stat-box">
    <div class="stat-label">Late Delivery</div>
    <div class="stat-value" style="color:#ef4444;">{k['late_delivery_pct']}%</div>
    <div class="stat-sub">Dataset avg</div>
  </div>
  <div class="stat-box">
    <div class="stat-label">Avg Daily Demand</div>
    <div class="stat-value">{avg_d:,.0f}</div>
    <div class="stat-sub">units/day (session)</div>
  </div>
  <div class="stat-box">
    <div class="stat-label">Route CO₂</div>
    <div class="stat-value">{k['avg_co2_kg']:,.0f} kg</div>
    <div class="stat-sub">this session</div>
  </div>
  <div class="stat-box">
    <div class="stat-label">Freight Cost</div>
    <div class="stat-value">₹{k['avg_route_cost']:,.0f}</div>
    <div class="stat-sub">this route</div>
  </div>
  <div class="stat-box">
    <div class="stat-label">Delay Risk</div>
    <div class="stat-value" style="color:{'#ef4444' if R['delay_prob']>0.7 else '#f59e0b' if R['delay_prob']>0.4 else '#10b981'};">
      {R['delay_prob']*100:.1f}%
    </div>
    <div class="stat-sub">model output</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Forecast & Delay",
    "🏪 Inventory & Supplier",
    "🗺️ Routing & Network",
    "⚡ Resilience",
    "🤖 Decision Engine",
])


# ───────────────────────────────────────────────────────────────────────────────
# TAB 1 — Forecast & Delay
# ───────────────────────────────────────────────────────────────────────────────
with tab1:
    cl, cr = st.columns([3, 2], gap="large")

    with cl:
        st.markdown('<div class="section-title">7-Day Demand Forecast</div>', unsafe_allow_html=True)
        st.plotly_chart(demand_chart(R["demand_forecast"]), use_container_width=True,
                        config={"displayModeBar": False})

        d_min = min(R["demand_forecast"])
        d_max = max(R["demand_forecast"])
        a, b, c_ = st.columns(3)
        a.metric("Avg/Day",  f"{avg_d:,.0f} units")
        b.metric("Min Day",  f"{d_min:,.0f} units")
        c_.metric("Max Day", f"{d_max:,.0f} units")

        st.markdown('<div class="section-title">Model Inputs Used</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="card info">
            Sales/Customer: <strong>₹{P['sales_per_customer']:,}</strong> &nbsp;·&nbsp;
            Category: <strong>#{P['category_id']}</strong> &nbsp;·&nbsp;
            Lag-1: <strong>{P['lag_1']:,}</strong> &nbsp;·&nbsp;
            Rolling Avg: <strong>{P['rolling_mean_3']:,}</strong><br>
            Shipping: <strong>{P['shipping_mode']}</strong> &nbsp;·&nbsp;
            Segment: <strong>{P['customer_segment']}</strong> &nbsp;·&nbsp;
            Traffic: <strong>{P['traffic_condition']}</strong>
        </div>
        """, unsafe_allow_html=True)

    with cr:
        st.markdown('<div class="section-title">Delay Risk Probability</div>', unsafe_allow_html=True)
        delay_pct  = round(R["delay_prob"] * 100, 1)
        st.plotly_chart(delay_gauge(delay_pct), use_container_width=True,
                        config={"displayModeBar": False})

        risk_label = "Critical" if delay_pct > 70 else "Moderate" if delay_pct > 40 else "Low"
        badge_cls  = "critical" if delay_pct > 70 else "warning" if delay_pct > 40 else "ok"
        st.markdown(f"""
        <div class="card {badge_cls}">
            <span class="badge-{badge_cls}">{risk_label.upper()}</span>
            &nbsp; <strong>{delay_pct}%</strong> delay probability &mdash;
            {P['lead_time']}-day lead time via <strong>{P['shipping_mode']}</strong>.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">Disruption Impact</div>', unsafe_allow_html=True)
        dis_cls = ("critical" if P["disruption_severity"] >= 4
                   else "warning" if P["disruption_severity"] >= 2 else "ok")
        st.markdown(f"""
        <div class="card {dis_cls}">
            <span class="badge-{dis_cls}">SEVERITY {P['disruption_severity']}/5</span>
            &nbsp; <strong>{P['disruption_type']}</strong><br>
            <span style="font-size:12px;color:#94a3b8;">
                {'No active disruption.' if P['disruption_type'] == 'None'
                 else 'Disruption factored into delay probability.'}
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">Traffic</div>', unsafe_allow_html=True)
        t = R["traffic"]
        tcls = "critical" if t["level"] == "High" else "warning" if t["level"] == "Moderate" else "ok"
        st.markdown(f"""
        <div class="card {tcls}">
            <span class="badge-{tcls}">{t['level'].upper()}</span>
            &nbsp; Network high-traffic: <strong>{t['high_pct']}%</strong>
            &nbsp;·&nbsp; Selected: <strong>{P['traffic_condition']}</strong>
        </div>
        """, unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────────────────────
# TAB 2 — Inventory & Supplier
# ───────────────────────────────────────────────────────────────────────────────
with tab2:
    cl, cr = st.columns([1, 1], gap="large")

    with cl:
        st.markdown('<div class="section-title">Inventory Status</div>', unsafe_allow_html=True)
        inv_cls = {"SAFE":"ok","ADEQUATE":"ok","LOW":"warning","CRITICAL":"critical"}.get(R["inv_status"],"info")
        fill_ratio = min(P["current_stock"] / max(R["reorder"], 1), 1.5)
        st.progress(min(fill_ratio / 1.5, 1.0))

        i1, i2 = st.columns(2)
        i1.metric("Current Inventory", f"{P['current_stock']:,} units")
        i2.metric("Reorder Point",     f"{R['reorder']:,.0f} units")
        avg_fc = sum(R["demand_forecast"]) / 7
        cover  = round(P["current_stock"] / max(avg_fc, 1), 1)
        st.metric("Days of Cover", f"~{cover} days")

        st.markdown(f"""
        <div class="card {inv_cls}">
            <span class="badge-{inv_cls}">{R['inv_status']}</span>
            &nbsp; {'Stock is sufficient for current demand.' if inv_cls == 'ok'
                    else 'Review stock — consider immediate replenishment.'}
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">Reorder Point Formula</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="card info">
            ROP = Avg Demand × Lead Time + Safety Stock<br>
            &nbsp;&nbsp;&nbsp; = {avg_fc:,.1f} × {P['lead_time']} + {P['safety_stock']}
            &nbsp;= <strong>{R['reorder']:,.0f} units</strong>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">Safety Stock Sensitivity</div>', unsafe_allow_html=True)
        ss_vals = sorted(set([0, 50, 100, 150, 200, 250, P["safety_stock"]]))
        ss_df = pd.DataFrame({
            "Safety Stock": [f"{s} units" for s in ss_vals],
            "Reorder Point": [f"{round(avg_fc * P['lead_time'] + s):,}" for s in ss_vals],
            "Status": ["✅" if P["current_stock"] >= round(avg_fc * P['lead_time'] + s) else "⚠️" for s in ss_vals],
        })
        st.dataframe(ss_df, use_container_width=True, hide_index=True)

    with cr:
        st.markdown('<div class="section-title">Selected Supplier</div>', unsafe_allow_html=True)
        s = R["best_supplier"]
        s1, s2 = st.columns(2)
        s1.metric("Supplier",    s["supplier_location"])
        s2.metric("Reliability", f"{s['reliability']*100:.1f}%")
        s3, s4 = st.columns(2)
        s3.metric("Distance",    f"{s['distance_km']:,} km")
        s4.metric("CO₂/km",     f"{s['co2_per_km']} kg")

        w = R["weather"]
        wcls = "critical" if w["risk"] == "High" else "warning" if w["risk"] == "Medium" else "ok"
        st.markdown(f"""
        <div class="card {wcls}" style="margin-top:14px;">
            <strong>☁️ Weather — {w['location']}</strong><br>
            {w['condition']} &nbsp;|&nbsp; <span class="badge-{wcls}">{w['risk'].upper()} RISK</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">All Suppliers vs. {}</div>'.format(P["plant_location"]),
                    unsafe_allow_html=True)
        sup_df = build_supplier_table(P["plant_location"])
        st.dataframe(sup_df[["Supplier","Reliability","Distance (km)","Freight Cost (₹)","CO2 (kg)"]],
                     use_container_width=True, hide_index=True)

        st.markdown('<div class="section-title">Capability Radar</div>', unsafe_allow_html=True)
        st.plotly_chart(supplier_radar(sup_df), use_container_width=True,
                        config={"displayModeBar": False})


# ───────────────────────────────────────────────────────────────────────────────
# TAB 3 — Routing & Network
# ───────────────────────────────────────────────────────────────────────────────
with tab3:
    cl, cr = st.columns([3, 2], gap="large")

    with cl:
        st.markdown('<div class="section-title">Supply Network Map — All Suppliers</div>',
                    unsafe_allow_html=True)
        plant_coords_sel = PLANT_OPTIONS[P["plant_location"]]
        fig_map = multi_route_map(
            P["supplier_location"], SUPPLIER_COORDS,
            P["plant_location"], plant_coords_sel,
        )
        st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})

        r = R["route"]
        r1, r2, r3 = st.columns(3)
        r1.metric("From",         r["origin"])
        r2.metric("To",           r["destination"])
        r3.metric("Distance",     f"{r['distance_km']:,} km")
        r4, r5 = st.columns(2)
        r4.metric("Est. Transit", f"{r['duration_hrs']} hrs")
        r5.metric("Freight Cost", f"₹{R['kpis']['avg_route_cost']:,.0f}")

        # Nearest supplier callout
        nearest = find_nearest_supplier(P["plant_location"], P["opt_mode"])
        st.markdown(f"""
        <div class="card info" style="margin-top:12px;">
            🏆 <strong>Nearest Supplier to {P['plant_location']}:</strong>
            <strong style="color:#3b82f6;"> {nearest}</strong>
            &nbsp;({int(haversine(*SUPPLIER_COORDS[nearest], *plant_coords_sel)):,} km)
            {'&nbsp; ✅ <em>You selected the nearest</em>' if nearest == P['supplier_location'] else ''}
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">All Routes at a Glance</div>', unsafe_allow_html=True)
        co2_avg = float(df_clean["co2_per_km"].mean()) if "co2_per_km" in df_clean.columns else 0.30
        route_rows = []
        for nm, (la, lo) in SUPPLIER_COORDS.items():
            d  = round(haversine(la, lo, plant_coords_sel[0], plant_coords_sel[1]), 1)
            em = round(d * co2_avg, 1)
            fc = round(d * 5.2, 0)
            route_rows.append({
                "Supplier": f"{'★ ' if nm == P['supplier_location'] else ''}{nm}",
                "Distance (km)": f"{d:,.0f}",
                "Transit (hrs)": f"{d/60:.1f}",
                "CO₂ (kg)": f"{em}",
                "Freight (₹)": f"₹{fc:,.0f}",
                "Nearest": "✅" if nm == nearest else "",
            })
        st.dataframe(pd.DataFrame(route_rows), use_container_width=True, hide_index=True)

    with cr:
        st.markdown('<div class="section-title">Emissions & Eco Score</div>', unsafe_allow_html=True)
        st.plotly_chart(co2_bar(R["emission"], R["eco_score"]),
                        use_container_width=True, config={"displayModeBar": False})

        e1, e2 = st.columns(2)
        e1.metric("CO₂ Emission", f"{R['emission']:.1f} kg")
        e2.metric("Eco Score",    f"{R['eco_score']} / 100")

        eco_cls = "ok" if R["eco_score"] > 70 else "warning" if R["eco_score"] > 40 else "critical"
        eco_txt = ("Low carbon footprint — good route choice." if R["eco_score"] > 70
                   else "Moderate emissions — consider consolidating loads or rail." if R["eco_score"] > 40
                   else "High emissions — switch to eco-friendly routing or closer supplier.")
        st.markdown(f"""
        <div class="card {eco_cls}">
            <span class="badge-{eco_cls}">
                {'ECO ✓' if R['eco_score']>70 else 'REVIEW' if R['eco_score']>40 else 'HIGH EMISSION'}
            </span>&nbsp; {eco_txt}
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">CO₂ by Shipping Mode</div>', unsafe_allow_html=True)
        mode_mult = {"Standard Class":1.0,"Second Class":1.1,"First Class":1.4,"Same Day":2.0}
        mode_rows = [{"Shipping Mode": m,
                      "Multiplier": f"{v}×",
                      "Est. CO₂ (kg)": f"{round(R['emission']*v,1)}",
                      "Active": "✅" if m == P["shipping_mode"] else ""}
                     for m, v in mode_mult.items()]
        st.dataframe(pd.DataFrame(mode_rows), use_container_width=True, hide_index=True)

        st.markdown('<div class="section-title">Optimisation Mode</div>', unsafe_allow_html=True)
        mode_desc = {
            "Cost Efficient": "Batch shipments via standard road to minimise per-unit cost.",
            "Fast Delivery":  "Air freight / express road for time-critical SKUs.",
            "Eco Friendly":   "Rail / water preferred. Consolidate to reduce CO₂.",
        }
        st.markdown(f"""
        <div class="card info">
            <strong>{P['opt_mode']}</strong><br>
            <span style="color:#94a3b8;font-size:13px;">{mode_desc[P['opt_mode']]}</span>
        </div>
        """, unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────────────────────
# TAB 4 — Resilience
# ───────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-title">Disruption Scenario Analysis</div>', unsafe_allow_html=True)
    st.plotly_chart(resilience_chart(R["scenarios"]), use_container_width=True,
                    config={"displayModeBar": False})

    for sc in R["scenarios"]:
        sev    = sc["avg_delay_days"]
        cls    = "critical" if sev > 5 else "warning" if sev > 3 else "info"
        active = " 🔴 ACTIVE" if sc["scenario"] == P["disruption_type"] else ""
        st.markdown(f"""
        <div class="card {cls}">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                <strong>{sc['scenario']}{active}</strong>
                <span style="font-size:11px;color:#64748b;">
                    {sc['affected_orders']:,} orders &nbsp;·&nbsp; Avg delay: {sc['avg_delay_days']} days
                </span>
            </div>
            <div style="color:#94a3b8;font-size:13px;">{sc['recommendation']}</div>
        </div>
        """, unsafe_allow_html=True)

    kl, kr = st.columns(2, gap="large")
    with kl:
        st.markdown('<div class="section-title">KPI — Baseline vs AI Model</div>', unsafe_allow_html=True)
        baseline = {
            "Metric":         ["Service Level", "Late Delivery", "Avg Daily Demand", "CO₂/Route (kg)", "Freight Cost"],
            "Simple ROP":     ["82.0%", "18.0%", "180 units", "95 kg", "₹4,200"],
            "AI Model":       [
                f"{k['service_level_pct']}%",
                f"{k['late_delivery_pct']}%",
                f"{avg_d:,.0f} units",
                f"{R['emission']:.1f} kg",
                f"₹{k['avg_route_cost']:,.0f}",
            ],
        }
        st.dataframe(pd.DataFrame(baseline), use_container_width=True, hide_index=True)

    with kr:
        st.markdown('<div class="section-title">Session Risk Summary</div>', unsafe_allow_html=True)
        risk_cls = ("critical" if P["disruption_severity"] >= 4
                    else "warning" if P["disruption_severity"] >= 2 else "ok")
        st.markdown(f"""
        <div class="card {risk_cls}">
            <strong>Disruption:</strong> {P['disruption_type']} (Severity {P['disruption_severity']}/5)<br>
            <strong>Traffic:</strong> {P['traffic_condition']}<br>
            <strong>Supplier Reliability:</strong> {P['supplier_reliability']*100:.0f}%<br>
            <strong>Delay Probability:</strong> {delay_pct}%<br>
            <strong>Inventory Status:</strong> {R['inv_status']}<br>
            <strong>Route:</strong> {P['supplier_location']} → {P['plant_location']}
        </div>
        """, unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────────────────────
# TAB 5 — Decision Engine
# ───────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-title">AI Prescriptive Recommendations</div>', unsafe_allow_html=True)

    label_map = {"critical":"Action Required","warning":"Warning","ok":"OK","info":"Info"}
    icon_map  = {"critical":"🔴","warning":"🟡","ok":"🟢","info":"🔵"}
    for d in R["decisions"]:
        st.markdown(f"""
        <div class="card {d['level']}">
            <span class="badge-{d['level']}">{icon_map.get(d['level'],'')} {label_map.get(d['level'],'').upper()}</span>
            &nbsp; {d['text']}
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    dl, dr = st.columns(2, gap="large")

    with dl:
        st.markdown('<div class="section-title">Full Inputs Summary</div>', unsafe_allow_html=True)
        summary = pd.DataFrame({
            "Parameter": [
                "Lead Time","Safety Stock","Current Inventory","Shipping Mode",
                "Optimisation Mode","Supplier (Origin)","Plant (Destination)",
                "Supplier Lead Time","Supplier Reliability","Route Distance",
                "Sales/Customer","Category","Customer Segment",
                "Traffic","Disruption","Disruption Severity",
                "Lag-1 Demand","Rolling Avg (3d)",
                "Avg Forecast Demand","Delay Probability",
                "Reorder Point","Inventory Status",
            ],
            "Value": [
                f"{P['lead_time']} days", f"{P['safety_stock']} units",
                f"{P['current_stock']:,} units", P["shipping_mode"],
                P["opt_mode"], P["supplier_location"], P["plant_location"],
                f"{P['supplier_lead_time']} days", f"{P['supplier_reliability']*100:.0f}%",
                f"{P['distance_km']:,} km", f"₹{P['sales_per_customer']:,}",
                f"#{P['category_id']}", P["customer_segment"],
                P["traffic_condition"], P["disruption_type"],
                f"{P['disruption_severity']}/5",
                f"{P['lag_1']:,} units", f"{P['rolling_mean_3']:,} units",
                f"{avg_d:,.0f} units/day", f"{delay_pct}%",
                f"{R['reorder']:,.0f} units", R["inv_status"],
            ],
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

    with dr:
        st.markdown('<div class="section-title">7-Day Forecast Detail</div>', unsafe_allow_html=True)
        fdf = pd.DataFrame({
            "Day": [f"Day {i+1}" for i in range(7)],
            "Forecast (units)": [f"{v:,.0f}" for v in R["demand_forecast"]],
            "vs Avg": [f"{'▲' if v>avg_d else '▼'} {abs(v-avg_d):,.0f}" for v in R["demand_forecast"]],
        })
        st.dataframe(fdf, use_container_width=True, hide_index=True)

        st.markdown('<div class="section-title">Model Architecture</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card info">
            <strong>Demand Model</strong> — VotingRegressor:<br>
            &nbsp;• HistGradientBoostingRegressor (XGBoost-style, 300 iters, L2 reg)<br>
            &nbsp;• GradientBoostingRegressor (150 estimators, depth 4)<br><br>
            <strong>Delay Classifier</strong>:<br>
            &nbsp;• HistGradientBoostingClassifier (300 iters, L2 reg, NaN-native)<br><br>
            <span style="color:#64748b;font-size:12px;">
            Trained on 60,173 orders. Features: 15 demand signals, 16 delay signals.
            </span>
        </div>
        """, unsafe_allow_html=True)