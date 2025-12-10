# app.py — RIGSENSE: FINAL WORKING VERSION (Streamlit Cloud 2025)
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
    df = df.copy()
    if df.shape[1] >= 26:
        df = df.iloc[:, :26]
    cols = ['unit_number','time_cycles','setting_1','setting_2','setting_3'] + [f's{i}' for i in range(1,22)]
    df.columns = cols
    
    df = df.sort_values(['unit_number', 'time_cycles']).reset_index(drop=True)
    
    sensors = ['s2','s3','s4','s7','s11','s12','s15','s20']
    for s in sensors:
        df[f'{s}_roll_mean'] = df.groupby('unit_number')[s].rolling(30, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f'{s}_roll_std']  = df.groupby('unit_number')[s].rolling(30, min_periods=1).std().fillna(0).reset_index(level=0, drop=True)
    
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0
            
    return df[expected_features].values  # Return numpy array

# ========================= UI =========================
st.title("RigSense — Oil & Gas Predictive Maintenance")
st.markdown("**Upload NASA .txt → instant failure prediction**")

with st.sidebar:
    threshold = st.slider("Alert Threshold", 0.5, 1.0, 0.90, 0.01)

tab1, tab2 = st.tabs(["Upload", "Demo"])

with tab1:
    uploaded = st.file_uploader("Drop NASA .txt file", type=["txt","csv"])
with tab2:
    uploaded = "data/test_FD001.txt"

if uploaded:
    with st.spinner("Analyzing..."):
        if isinstance(uploaded, str):
            df = pd.read_csv(uploaded, sep=r"\s+", header=None, engine='python')
        else:
            df = pd.read_csv(uploaded)
        
        X = engineer_features(df)
        prob = model.predict_proba(X)[:, 1]
        pred = (prob >= threshold).astype(int)
        
        st.success("Complete!")
        col1, col2, col3 = st.columns(3)
        col1.metric("Engines", df.iloc[:,0].nunique())
        col2.metric("Alerts", pred.sum())
        col3.metric("Highest Risk", f"{prob.max():.1%}")
        
        result = pd.DataFrame({
            "Cycle": df['time_cycles'].values,
            "Failure_Probability": prob,
            "Alert": np.where(pred==1, "REPLACE NOW", "Safe")
        })
        st.dataframe(result.tail(20), use_container_width=True)
        
        if pred.sum() > 0:
            st.balloons()
            st.error("RIG FAILURE PREDICTED")

st.caption("Production ML Engineer • Ships systems")
