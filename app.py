#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import pickle
import math
from datetime import datetime, time
from distance_utils import get_distance_miles, AIRPORT_COORDS

# ——— Page config & theming ———
st.set_page_config(
    page_title="Flight Delay Predictor",
    page_icon="✈️",
    layout="wide"
)
st.markdown("""
    <style>
      .css-18e3th9 { background-color: #f7f9fb; }
      .stButton>button { border-radius: 8px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }
      .stSelectbox>div, .stTimeInput>div, .stNumberInput>div { border-radius: 6px; }
      section[data-testid=\"stSidebar\"] { width: 300px; min-width: 300px; }
    </style>
""", unsafe_allow_html=True)

# ——— Persist departure time across reruns ———
# Use Streamlit's built-in state for time_input with key
DEFAULT_DEP_TIME = time(hour=12, minute=0)

# ——— Load models ———
@st.cache_resource
def load_models():
    paths = {
        "binary":    "models/binary_model.pkl",
        "cause":     "models/cause_model_pre.pkl",
        "encoders":  "models/encoders.pkl",
        "target_le": "models/target_le.pkl",
        "reg_hgb":   "models/reg_model_hgb.pkl",
    }
    return {k: pickle.load(open(p, "rb")) for k, p in paths.items()}

models = load_models()
binary_model    = models["binary"]
cause_model_pre = models["cause"]
encoders        = models["encoders"]
target_le       = models["target_le"]
reg_model_hgb   = models["reg_hgb"]

# ——— Feature lists ———
num_feats   = ["DepTime","CRSElapsedTime","Distance","DayOfWeek","Month","Day"]
cat_feats   = ["UniqueCarrier","Origin","Dest"]
feats_cause = ["UniqueCarrier","DayOfWeek","DepTime","CRSElapsedTime",
               "Distance","Origin","Dest","Month","Day"]

# ——— Sidebar form ———
st.sidebar.header("✈️ Flight Details")
with st.sidebar.form("flight_form"):
    date    = st.date_input("Date of Flight", datetime.today())
    carrier = st.selectbox("Airline", encoders["UniqueCarrier"].classes_)
    # Use hard-coded airport list
    valid_iata = sorted(set(encoders["Origin"].classes_) & set(AIRPORT_COORDS.keys()))
    origin  = st.selectbox("Origin", valid_iata)
    dest    = st.selectbox("Destination", valid_iata)

    dow     = date.weekday() + 1
    month   = date.month
    day     = date.day

    # Use key-only time_input to avoid double-setting default via session_state
    dep_time_obj = st.time_input(
        "Departure Time", 
        DEFAULT_DEP_TIME,
        key="dep_time"
    )
    dep_time = dep_time_obj.hour * 100 + dep_time_obj.minute

    distance = get_distance_miles(origin, dest)
    if distance is None:
        st.error(f"Distance not available for {origin}→{dest}.")
    else:
        st.write(f"Distance: {distance:.1f} miles")

    crs_etime = st.number_input("Scheduled Elapsed Time (min)", min_value=0, max_value=1000, value=120)
    submit    = st.form_submit_button("Predict Delay")

# ——— Main & Tabs ———
st.title("✈️Flight Delay Predictor")
tab1, tab2 = st.tabs(["Flight Delay Predictor","Model Info"])

with tab1:
    if not submit:
        st.info("Enter flight details in the sidebar and click **Predict Delay**.")
    else:
        with st.spinner("Analyzing flight..."):
            dfb = pd.DataFrame([{"DepTime": dep_time,
                                  "DayOfWeek": dow,
                                  "Month": month,
                                  "UniqueCarrier": carrier,
                                  "Origin": origin,
                                  "Dest": dest}])
            dfb["DepHour"] = dfb["DepTime"] // 100
            dfb["IsWeekend"] = dfb["DayOfWeek"].isin([6,7]).astype(int)
            dfb["IsMorningFlight"] = dfb["DepHour"].between(5,12).astype(int)
            keep = ["DepHour","DayOfWeek","Month","IsWeekend","IsMorningFlight",
                    "UniqueCarrier","Origin","Dest"]
            dfb_ohe = (pd.get_dummies(dfb[keep])
                       .reindex(columns=binary_model.get_booster().feature_names, fill_value=0))
            delayed = binary_model.predict(dfb_ohe)[0]
            proba   = binary_model.predict_proba(dfb_ohe)[0,1]
        c1, c2 = st.columns(2)
        if delayed == 0:
            c1.success("✅ On‑time flight")
        else:
            c1.error("⏰ Delayed flight")
        c2.metric("Delay Probability", f"{proba:.1%}")

        if delayed == 1:
            dfr = pd.DataFrame([{"DepTime": dep_time,
                                  "CRSElapsedTime": crs_etime,
                                  "Distance": distance,
                                  "DayOfWeek": dow,
                                  "Month": month,
                                  "Day": day,
                                  "UniqueCarrier": carrier,
                                  "Origin": origin,
                                  "Dest": dest}])
            est = reg_model_hgb.predict(dfr)[0]
            st.metric("⏱ Estimated Delay (min)", f"{est:.0f}")
            dfc = pd.DataFrame([{"UniqueCarrier": carrier,
                                  "Origin": origin,
                                  "Dest": dest,
                                  "DayOfWeek": dow,
                                  "Month": month,
                                  "Day": day,
                                  "DepTime": dep_time,
                                  "CRSElapsedTime": crs_etime,
                                  "Distance": distance}])
            for col in ["UniqueCarrier","Origin","Dest"]:
                dfc[col] = encoders[col].transform(dfc[col].astype(str))
            enc_preds = cause_model_pre.predict(dfc[feats_cause])
            raw_cause = target_le.inverse_transform(enc_preds)[0]
            friendly_map = {"CarrierDelay":"Carrier Delay",
                            "LateAircraftDelay":"Late Aircraft Delay",
                            "NASDelay":"National Airspace System Delay",
                            "SecurityDelay":"Security Security Delay",
                            "WeatherDelay":"Weather Delay"}
            st.info(f"🔍 Likely Cause of Delay: **{friendly_map.get(raw_cause, raw_cause)}**")

with tab2:
    st.subheader("Binary Delay Classifier (XGBoost)")
    st.write("- Accuracy: 0.59  ")
    st.write("- ROC AUC: 0.67")
    booster = binary_model.get_booster()
    imps = booster.get_score(importance_type="weight")
    imp_ser = pd.Series(imps).sort_values(ascending=False)
    st.bar_chart(imp_ser.head(10))

    st.subheader("Cause Predictor")
    st.write("- Accuracy: 0.50  ")
    st.write("- ROC AUC: 0.82")
    if hasattr(cause_model_pre, "feature_importances_"):
        ci = pd.Series(cause_model_pre.feature_importances_, index=feats_cause).sort_values(ascending=False)
        st.bar_chart(ci.head(10))
    else:
        st.write("Feature importances not available.")

    st.subheader("Delay Duration Regressor (HGB)")
    st.write("- RMSE: 47.65 min")
    st.write("- R²: 0.209")
    reg = reg_model_hgb.named_steps["reg"]
    if hasattr(reg, "feature_importances_"):
        pre = reg_model_hgb.named_steps["preproc"]
        ohe_feats = pre.named_transformers_["cat"].get_feature_names_out(cat_feats)
        feat_names = num_feats + list(ohe_feats)
        ri = pd.Series(reg.feature_importances_, index=feat_names).sort_values(ascending=False)
        st.bar_chart(ri.head(10))
    else:
        st.write("Feature importances not available.")
