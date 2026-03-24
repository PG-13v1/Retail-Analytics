import streamlit as st
import pandas as pd
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from sklearn.preprocessing import StandardScaler


@st.cache_data(show_spinner="Detecting anomalies...")
def detect_anomalies(df):

    if "Revenue" not in df.columns:
        raise ValueError(
            "'Revenue' column not found. Available columns: " + str(list(df.columns))
        )

    df = df.copy()

    # Feature Engineering
    features = ["Revenue"]
    if "Quantity" in df.columns:
        features.append("Quantity")
    if "UnitPrice" in df.columns:
        features.append("UnitPrice")

    X = df[features]

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Models
    iforest = IForest(contamination=0.02)
    knn = KNN(contamination=0.02)

    iforest.fit(X_scaled)
    knn.fit(X_scaled)

    # Predictions
    df["IForest_Label"] = iforest.predict(X_scaled)
    df["KNN_Label"] = knn.predict(X_scaled)

    # Ensemble
    df["Final_Anomaly"] = (
        (df["IForest_Label"] == 1) &
        (df["KNN_Label"] == 1)
    ).astype(int)

    # ✅ Add score BEFORE filtering
    df["Anomaly_Score"] = iforest.decision_scores_

    anomalies = df[df["Final_Anomaly"] == 1]

    # Sort anomalies
    anomalies = anomalies.sort_values("Anomaly_Score", ascending=False)

    return anomalies.head(20)
