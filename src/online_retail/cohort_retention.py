import pandas as pd
import streamlit as st


@st.cache_data(show_spinner="Computing cohort analysis...")
def cohort_matrix(df):

    df = df.copy()

    # ---------------------------
    # Create Cohort
    # ---------------------------
    df["CohortMonth"] = df.groupby("CustomerID")["InvoiceDate"] \
                           .transform("min") \
                           .dt.to_period("M")

    df["OrderMonth"] = df["InvoiceDate"].dt.to_period("M")

    # ---------------------------
    # Cohort Index (months since first purchase)
    # ---------------------------
    df["CohortIndex"] = (
        (df["OrderMonth"].dt.year - df["CohortMonth"].dt.year) * 12 +
        (df["OrderMonth"].dt.month - df["CohortMonth"].dt.month)
    )

    # ---------------------------
    # Count customers
    # ---------------------------
    cohort_data = df.groupby(
        ["CohortMonth", "CohortIndex"]
    )["CustomerID"].nunique().reset_index()

    cohort_pivot = cohort_data.pivot_table(
        index="CohortMonth",
        columns="CohortIndex",
        values="CustomerID"
    )

    # ---------------------------
    # Retention Rate (IMPORTANT)
    # ---------------------------
    cohort_size = cohort_pivot.iloc[:, 0]
    retention = cohort_pivot.divide(cohort_size, axis=0)

    return retention.fillna(0)
