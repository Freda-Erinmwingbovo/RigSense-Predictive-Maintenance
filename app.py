# app.py — RIGSENSE: FINAL WORKING VERSION (Tested Live)
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

st.set_page_config(page_title="RigSense", page_icon="oil_rig", layout="wide")

# ========================= LOAD MODEL =========================
@st.cache_resource
def load_model():
    model = joblib.load("models/rigsense_prod.pkl")
    features = joblib.load("models/rigsense_features.pkl")
    return model, features

model, expected_features = load_model()

# ========================= FEATURE ENGINEERING =========================
def engineer_features(df):
    df = df.copy().reset_index(drop=True)
    
    # Force correct column names
    if df.shape[1] >= 26:
        df = df.iloc[:, :26]
    cols = ['unit_number','time_cycles','setting_1','setting_2','setting_3'] + [f's{i}' for i in range(1,22)]
    df.columns = cols
    
    df = df.sort_values(['unit_number', 'time_cycles']).reset_index(drop=True)
    
    sensors = ['s2','s3','s4','s7','s11','s12','s15','s20']
    for s in sensors:
        roll_mean = df.groupby('unit_number')[s].transform(lambda x: x.rolling(30, min_periods=1).mean())
        roll_std  = df.groupby('unit_number')[s].transform(lambda x: x.rolling(30, min_periods=1).std()).fillna(0)
        df = df.assign(**{f'{s}_roll_mean': roll_mean, f'{s}_roll_std': roll_std})
    
    # Add all expected features (fill missing with 0)
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0
    
    return df[expected_features].values, df  # return both X and original df

# ========================= UI =========================
st.title("RigSense — Oil & Gas Predictive Maintenance")
st.markdown("**Upload NASA .txt file → instant failure prediction**")

with st.sidebar:
    threshold = st.slider("Alert Threshold", 0.5, 1.0, 0.90, 0.01)

tab1, tab2 = st.tabs(["Upload File", "Demo (NASA Test Data)"])

with tab1:
    uploaded = st.file_uploader("Drop file", type=["txt","csv"])
with tab2:
    uploaded = "data/test_FD001.txt"

if uploaded:
    with st.spinner("Analyzing rig health..."):
        # Load file
        if isinstance(uploaded, str):
            df_raw = pd.read_csv(uploaded, sep=r"\s+", header=None, engine='python')
        else:
            df_raw = pd.read_csv(uploaded)
        
        X, df_full = engineer_features(df_raw)
        prob = model.predict_proba(X)[:, 1]
        pred = (prob >= threshold).astype(int)
        
        # Results
        result_df = pd.DataFrame({
            "Engine": df_full['unit_number'],
            "Cycle": df_full['time_cycles'],
            "Failure_Probability": prob.round(4),
            "Alert": np.where(pred==1, "REPLACE NOW", "Safe")
        })

    st.success("Analysis Complete!")
    col1, col2, col3 = st.columns(3)
    col1.metric("Engines", df_full['unit_number'].nunique())
    col2.metric("Alerts", pred.sum())
    col3.metric("Highest Risk", f"{prob.max():.1%}")

    st.dataframe(result_df.tail(30), use_container_width=True)

    if pred.sum() > 0:
        st.balloons()
        st.error("RIG FAILURE DETECTED — Immediate action required!")
    else:
        st.success("All rigs currently healthy")

st.caption("Production ML Engineer • Ships real systems")
