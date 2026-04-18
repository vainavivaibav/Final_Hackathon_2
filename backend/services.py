import math
import numpy as np
import pandas as pd
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SUPPLIER_COORDS, PLANT_COORDS, SPEED_KMH, CO2_BASE


# ── Geo ────────────────────────────────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── Inventory ──────────────────────────────────────────────────────────────────

def reorder_point(demand_forecast: list, lead_time: float, safety_stock: float) -> float:
    avg = sum(demand_forecast) / len(demand_forecast)
    return avg * lead_time + safety_stock


def inventory_status(current: float, reorder: float) -> str:
    if current > reorder * 1.5:
        return "SAFE"
    elif current > reorder:
        return "ADEQUATE"
    elif current > reorder * 0.7:
        return "LOW"
    else:
        return "CRITICAL"


# ── Supplier optimisation ──────────────────────────────────────────────────────

def select_best_supplier(df: pd.DataFrame, mode: str = "Cost Efficient") -> dict:
    if "supplier_location" not in df.columns:
        return _fallback_supplier()

    # ── Ensure required columns exist (fallback defaults) ──
    df = df.copy()

    if "distance_km" not in df.columns:
        df["distance_km"] = 300  # default distance

    if "co2_per_km" not in df.columns:
        df["co2_per_km"] = 0.21  # default CO2 from config

    if "supplier_cost" not in df.columns:
        df["supplier_cost"] = df["distance_km"] * 10  # dummy cost logic

    if "supplier_reliability_score" not in df.columns:
        df["supplier_reliability_score"] = 0.8  # default reliability

    # ── Grouping ──
    grp = df.groupby("supplier_location").agg(
        avg_distance=("distance_km", "mean"),
        avg_co2=("co2_per_km", "mean"),
        avg_cost=("supplier_cost", "mean"),
        avg_reliability=("supplier_reliability_score", "mean"),
    ).reset_index()

    supplier_names = {0: "Chennai", 1: "Mumbai", 2: "Delhi"}

    # ── Selection logic ──
    if mode == "Fast Delivery":
        best_row = grp.loc[grp["avg_distance"].idxmin()]
    elif mode == "Eco Friendly":
        best_row = grp.loc[grp["avg_co2"].idxmin()]
    else:
        best_row = grp.loc[grp["avg_cost"].idxmin()]

    # ── Map supplier code → name ──
    loc_code = int(best_row["supplier_location"])
    loc_name = supplier_names.get(loc_code, "Chennai")

    lat, lon = SUPPLIER_COORDS.get(loc_name, SUPPLIER_COORDS["Chennai"])

    # ── Recalculate real distance using haversine ──
    dist = haversine(lat, lon, PLANT_COORDS[0], PLANT_COORDS[1])

    return {
        "supplier_location": loc_name,
        "distance_km": round(dist, 2),
        "co2_per_km": round(float(best_row["avg_co2"]), 4),
        "reliability": round(float(best_row["avg_reliability"]), 3),
        "lat": lat,
        "lon": lon,
    }


def _fallback_supplier():
    return {
        "supplier_location": "Chennai",
        "distance_km": 350.0,
        "co2_per_km": CO2_BASE,
        "reliability": 0.85,
        "lat": 13.08,
        "lon": 80.27,
    }


# ── Routing ────────────────────────────────────────────────────────────────────

def route_info(supplier: dict) -> dict:
    dist = supplier.get("distance_km", 350.0)
    return {
        "distance_km": round(dist, 2),
        "duration_hrs": round(dist / SPEED_KMH, 2),
        "origin": supplier.get("supplier_location", "Chennai"),
        "destination": "Bangalore Plant",
    }


# ── Sustainability ─────────────────────────────────────────────────────────────

def calculate_emission(distance_km: float, co2_per_km: float) -> float:
    return round(distance_km * co2_per_km, 2)


def sustainability_score(emission: float, max_emission: float = 500.0) -> int:
    return max(0, 100 - int((emission / max_emission) * 100))


# ── Traffic ────────────────────────────────────────────────────────────────────

def detect_spike(df: pd.DataFrame) -> dict:
    if "traffic_condition" not in df.columns:
        return {"level": "Unknown", "high_pct": 0.0, "message": "No traffic data available."}
    counts = df["traffic_condition"].value_counts()
    total = max(len(df), 1)
    high_pct = counts.get(0, 0) / total * 100
    if high_pct > 40:
        level = "High"
    elif high_pct > 20:
        level = "Moderate"
    else:
        level = "Normal"
    return {"level": level, "high_pct": round(high_pct, 1)}


# ── Weather (mock) ─────────────────────────────────────────────────────────────

def get_weather_risk(location: str = "Chennai") -> dict:
    risks = {
        "Chennai": {"condition": "Partly Cloudy", "risk": "Low",    "risk_score": 0.15},
        "Mumbai":  {"condition": "Monsoon Risk",   "risk": "High",   "risk_score": 0.75},
        "Delhi":   {"condition": "Clear",           "risk": "Low",    "risk_score": 0.10},
    }
    data = risks.get(location, {"condition": "Unknown", "risk": "Medium", "risk_score": 0.40})
    data["location"] = location
    return data


# ── Decision engine ────────────────────────────────────────────────────────────

def make_decision(demand_forecast: list, delay_prob: float,
                  inventory: float, reorder: float, mode: str = "Cost Efficient") -> list:
    decisions = []
    avg_demand = sum(demand_forecast) / len(demand_forecast) if demand_forecast else 0

    if delay_prob > 0.7:
        decisions.append({"level": "critical", "text": "High delay probability detected. Switch to express shipping or activate alternate supplier."})
    elif delay_prob > 0.4:
        decisions.append({"level": "warning", "text": "Moderate delay risk. Monitor shipment closely and pre-alert warehouse team."})
    else:
        decisions.append({"level": "ok", "text": "Delay risk is within acceptable range."})

    if inventory < reorder:
        decisions.append({"level": "critical", "text": f"Inventory ({inventory:,.0f} units) is below reorder point ({reorder:,.0f} units). Trigger replenishment immediately."})
    elif inventory < reorder * 1.2:
        decisions.append({"level": "warning", "text": "Stock is close to reorder threshold. Consider early replenishment."})
    else:
        decisions.append({"level": "ok", "text": "Inventory levels are adequate."})

    if avg_demand > 250:
        decisions.append({"level": "warning", "text": "Demand spike detected. Increase safety stock by 20% for the next cycle."})
    elif avg_demand < 100:
        decisions.append({"level": "info", "text": "Low demand period. Reduce incoming volume to minimise holding costs."})
    else:
        decisions.append({"level": "ok", "text": f"Average demand is stable at {avg_demand:,.0f} units/day."})

    mode_msgs = {
        "Eco Friendly":   {"level": "info", "text": "Eco mode active. Prefer rail or water transport. Consolidate shipments to reduce CO2 emissions."},
        "Fast Delivery":  {"level": "info", "text": "Fast mode active. Use air freight or express road for critical SKUs."},
        "Cost Efficient": {"level": "info", "text": "Cost mode active. Batch shipments and use standard road transport."},
    }
    decisions.append(mode_msgs.get(mode, mode_msgs["Cost Efficient"]))

    return decisions


# ── Resilience playbooks ───────────────────────────────────────────────────────

PLAYBOOKS = {
    "Strike":     {
        "avg_delay_recommendation": "Pre-position safety stock. Activate secondary supplier. Use air freight for critical parts.",
        "icon": "strike",
    },
    "Weather":    {
        "avg_delay_recommendation": "Monitor forecasts 48h ahead. Shift to inland routes. Pre-alert customers of potential delays.",
        "icon": "weather",
    },
    "Port Delay": {
        "avg_delay_recommendation": "Route via alternative ports. Use road transport for regional fulfilment. Adjust safety stock +25%.",
        "icon": "port",
    },
}


def resilience_scenarios(df: pd.DataFrame) -> list:
    scenarios = []

    # ── Create safe copy ──
    df = df.copy()

    # ── Fix missing columns ──
    if "days_for_shipping_(real)" not in df.columns:
        df["days_for_shipping_(real)"] = 4.0  # default delay

    if "disruption_type" not in df.columns:
        df["disruption_type"] = 0  # default category

    # Mapping (encoded values)
    code_map = {
        "Port Delay": 0,
        "Strike": 1,
        "Weather": 2
    }

    for disruption, meta in PLAYBOOKS.items():

        code = code_map.get(disruption, -1)

        # ── Filter subset safely ──
        if code >= 0:
            subset = df[df["disruption_type"] == code]
        else:
            subset = df

        # ── Safe delay calculation ──
        if len(subset) > 0:
            avg_delay = subset["days_for_shipping_(real)"].mean()
        else:
            avg_delay = 4.0  # fallback

        scenarios.append({
            "scenario": disruption,
            "avg_delay_days": round(float(avg_delay), 1),
            "affected_orders": int(len(subset)),
            "recommendation": meta["avg_delay_recommendation"],
        })

    return scenarios


# ── KPI summary ────────────────────────────────────────────────────────────────

def kpi_summary(df: pd.DataFrame) -> dict:
    total = len(df)
    late_pct = round(df["late_delivery_risk"].mean() * 100, 1) if "late_delivery_risk" in df.columns else 0.0
    avg_demand = round(df["demand"].mean(), 1) if "demand" in df.columns else 0.0
    avg_co2 = round((df["distance_km"] * df["co2_per_km"]).mean(), 2) \
        if all(c in df.columns for c in ["distance_km", "co2_per_km"]) else 0.0
    avg_cost = round(df["route_cost"].mean(), 2) if "route_cost" in df.columns else 0.0
    return {
        "total_orders":      total,
        "late_delivery_pct": late_pct,
        "service_level_pct": round(100 - late_pct, 1),
        "avg_demand":        avg_demand,
        "avg_co2_kg":        avg_co2,
        "avg_route_cost":    avg_cost,
    }