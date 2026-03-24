import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


@st.cache_data(show_spinner="Performing RFM segmentation...")
def rfm_segmentation(df):

    df = df.copy()

    today = df["InvoiceDate"].max()

    # ---------------------------
    # RFM Calculation
    # ---------------------------
    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (today - x.max()).days,
        "InvoiceNo": "nunique",
        "Revenue": "sum"
    })

    rfm.columns = ["Recency", "Frequency", "Monetary"]

    # ---------------------------
    # Log Transform (IMPORTANT)
    # ---------------------------
    rfm["Frequency"] = np.log1p(rfm["Frequency"])
    rfm["Monetary"] = np.log1p(rfm["Monetary"])

    # ---------------------------
    # Scaling
    # ---------------------------
    scaler = StandardScaler()
    X = scaler.fit_transform(rfm)

    # ---------------------------
    # KMeans
    # ---------------------------
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm["Cluster"] = kmeans.fit_predict(X)

    # ---------------------------
    # Label Clusters (VERY IMPORTANT)
    # ---------------------------
    cluster_summary = rfm.groupby("Cluster").mean()

    def label_cluster(row):
        if row["Monetary"] > cluster_summary["Monetary"].mean() and row["Frequency"] > cluster_summary["Frequency"].mean():
            return "High Value"
        elif row["Recency"] < cluster_summary["Recency"].mean():
            return "Recent Customers"
        else:
            return "Low Value"

    rfm["Segment"] = rfm.apply(label_cluster, axis=1)

    return rfm.reset_index()
