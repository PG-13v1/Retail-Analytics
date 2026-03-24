import streamlit as st
import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data


@st.cache_data(show_spinner="Training CLV model...")
def clv_model(df):

    df = df.copy()

    today = df["InvoiceDate"].max()

    # ---------------------------
    # Summary Table
    # ---------------------------
    summary = summary_data_from_transaction_data(
        df,
        customer_id_col="CustomerID",
        datetime_col="InvoiceDate",
        monetary_value_col="Revenue",
        observation_period_end=today
    )

    # ---------------------------
    # Filter customers (IMPORTANT)
    # ---------------------------
    summary = summary[summary["frequency"] > 0]

    # ---------------------------
    # BG/NBD Model
    # ---------------------------
    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(summary["frequency"], summary["recency"], summary["T"])

    summary["Predicted_30_Day_Purchases"] = bgf.conditional_expected_number_of_purchases_up_to_time(
        30,
        summary["frequency"],
        summary["recency"],
        summary["T"]
    )

    # ---------------------------
    # Gamma-Gamma Model (Monetary)
    # ---------------------------
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(summary["frequency"], summary["monetary_value"])

    summary["Predicted_Avg_Value"] = ggf.conditional_expected_average_profit(
        summary["frequency"],
        summary["monetary_value"]
    )

    # ---------------------------
    # CLV Calculation
    # ---------------------------
    summary["CLV_30_Days"] = (
        summary["Predicted_30_Day_Purchases"] *
        summary["Predicted_Avg_Value"]
    )

    # ---------------------------
    # Sort by value
    # ---------------------------
    summary = summary.sort_values("CLV_30_Days", ascending=False)

    return summary.reset_index()
