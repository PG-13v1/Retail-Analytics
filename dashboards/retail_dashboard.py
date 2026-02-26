import streamlit as st
import plotly.express as px

from src.online_retail.loader import load_online_retail
from src.online_retail.kpis import compute_kpis
from src.online_retail.segmentation import rfm_segmentation
from src.online_retail.cohort_retention import cohort_matrix
from src.online_retail.anomaly import detect_anomalies
from src.online_retail.hypothesis_testing import revenue_ttest
from src.online_retail.churn_clv import clv_model


def run():
    st.header("🛒 Online Retail Customer Intelligence")

    df = load_online_retail()

    # KPIs
    kpis = compute_kpis(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Revenue", f"${kpis['Revenue']:,.0f}")
    c2.metric("Orders", kpis["Orders"])
    c3.metric("Customers", kpis["Customers"])
    c4.metric("Avg Order Value", f"${kpis['AOV']:,.2f}")

    # Segmentation
    st.subheader("Customer Segmentation (RFM Clusters)")
    rfm = rfm_segmentation(df)

    fig = px.scatter(
        rfm,
        x="Recency",
        y="Monetary",
        color="Cluster",
        title="Customer Segments"
    )
    st.plotly_chart(fig, width='stretch')

    # Cohort Retention
    st.subheader("Retention Cohort Matrix")
    cohort = cohort_matrix(df)
    st.dataframe(cohort)

    # Hypothesis Testing
    st.subheader("Revenue Distribution Hypothesis Test")
    test_res = revenue_ttest(df)
    st.write(f"T-statistic: {test_res['t_statistic']:.3f}")
    st.write(f"P-value: {test_res['p_value']:.3e}")
    if test_res["significant"]:
        st.success("Significant difference between high and low revenue groups.")
    else:
        st.info("No significant difference between revenue groups.")

    # plot revenue groups
    median_rev = df["Revenue"].median()
    df_groups = df.copy()
    df_groups["Group"] = df_groups["Revenue"].apply(
        lambda x: "High" if x > median_rev else "Low"
    )
    fig_ht = px.histogram(
        df_groups,
        x="Revenue",
        color="Group",
        nbins=50,
        title="Revenue Distribution by Group (median split)",
        barmode="overlay",
        opacity=0.7
    )
    st.plotly_chart(fig_ht, width='stretch')

    # Anomaly Detection
    st.subheader("Revenue Anomaly Detection")
    anomalies = detect_anomalies(df)
    st.write(f"Total anomalies detected: {len(anomalies)}")

    # mark anomalies in main dataframe for plotting
    df = df.copy()
    df["Anomaly"] = False
    df.loc[anomalies.index, "Anomaly"] = True

    fig_anom = px.scatter(
        df,
        x="InvoiceDate",
        y="Revenue",
        color="Anomaly",
        title="Revenue over Time with Anomalies Highlighted",
        color_discrete_map={False: "blue", True: "red"}
    )
    st.plotly_chart(fig_anom, width='stretch')

    # optionally show anomalies table
    if not anomalies.empty:
        with st.expander("View anomaly records"):
            st.dataframe(anomalies)

    summary= clv_model(df)
    
