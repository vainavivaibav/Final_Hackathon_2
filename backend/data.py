import pandas as pd
import numpy as np
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_PATH


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, low_memory=False)
    # Normalise column names to lowercase with underscores
    df.columns = [c.strip().lower().replace(" ", "_").replace("(", "").replace(")", "") for c in df.columns]
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── Rename common columns to standard names ────────────────────────────────
    rename = {
        "days_for_shipping_real":       "days_for_shipping_(real)",
        "days_for_shipment_scheduled":  "days_for_shipment_(scheduled)",
        "order_date_dateorders":        "order_date_(dateorders)",
        "shipping_date_dateorders":     "shipping_date_(dateorders)",
        "late_delivery_risk":           "late_delivery_risk",
        "category_id":                  "category_id",
        "sales_per_customer":           "sales_per_customer",
        "co2_per_km":                   "co2_per_km",
        "order_item_quantity":          "demand",
    }
    df.rename(columns=rename, inplace=True)

    # ── Parse dates ───────────────────────────────────────────────────────────
    for col in ["order_date_(dateorders)", "shipping_date_(dateorders)"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "order_date_(dateorders)" in df.columns:
        df["order_day"]   = df["order_date_(dateorders)"].dt.day.fillna(1).astype(int)
        df["order_month"] = df["order_date_(dateorders)"].dt.month.fillna(1).astype(int)
        df["order_hour"]  = df["order_date_(dateorders)"].dt.hour.fillna(0).astype(int)
    else:
        df["order_day"]   = 1
        df["order_month"] = 1
        df["order_hour"]  = 0

    # ── Encode categoricals ───────────────────────────────────────────────────
    for col in ["shipping_mode", "customer_segment", "order_status",
                "supplier_location", "traffic_condition", "disruption_type",
                "vehicle_type", "market", "order_region", "type",
                "delivery_status"]:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes

    # ── Demand proxy if missing ───────────────────────────────────────────────
    if "demand" not in df.columns:
        if "sales" in df.columns:
            df["demand"] = df["sales"]
        else:
            df["demand"] = 100.0

    # ── Lag / rolling features ────────────────────────────────────────────────
    if "demand" in df.columns:
        df = df.sort_values("order_day").reset_index(drop=True)
        df["lag_1"]          = df["demand"].shift(1).fillna(df["demand"].median())
        df["rolling_mean_3"] = df["demand"].rolling(3, min_periods=1).mean()
    else:
        df["lag_1"]          = 200
        df["rolling_mean_3"] = 200

    # ── Fill numerics ─────────────────────────────────────────────────────────
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # ── Ensure late_delivery_risk is int ─────────────────────────────────────
    if "late_delivery_risk" in df.columns:
        df["late_delivery_risk"] = df["late_delivery_risk"].astype(int)

    return df