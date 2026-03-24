import pandas as pd
import streamlit as st
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from sklearn.preprocessing import StandardScaler


@st.cache_data(show_spinner="Detecting anomalies...")
def detect_retail_anomalies(df):

    df = df.copy()  # avoid modifying original

    # ---------------------------
    # Feature Engineering
    # ---------------------------
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week.astype("int16")

    features = ["Weekly_Sales", "Store", "Dept", "Month", "Week"]
    X = df[features]

    # ---------------------------
    # Scaling (IMPORTANT for KNN)
    # ---------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---------------------------
    # Isolation Forest
    # ---------------------------
    iforest = IForest(contamination=0.02)
    iforest.fit(X_scaled)
    df["IForest_Anomaly"] = iforest.predict(X_scaled)

    # ---------------------------
    # KNN Outlier Detection
    # ---------------------------
    knn = KNN(contamination=0.02)
    knn.fit(X_scaled)
    df["KNN_Anomaly"] = knn.predict(X_scaled)

    # ---------------------------
    # Ensemble Logic
    # ---------------------------
    df["Final_Anomaly"] = (
        (df["IForest_Anomaly"] == 1) |
        (df["KNN_Anomaly"] == 1)
    ).astype(int)

    anomalies = df[df["Final_Anomaly"] == 1]

    # ---------------------------
    # Sort by impact
    # ---------------------------
    anomalies = anomalies.sort_values("Weekly_Sales", ascending=False)

    return anomalies.head(20)
