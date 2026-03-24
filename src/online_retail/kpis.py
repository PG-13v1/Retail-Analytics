import streamlit as st


@st.cache_data(show_spinner="Computing KPIs...")
def compute_kpis(df):

    required_cols = ["Revenue", "InvoiceNo", "CustomerID"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"{col} column missing")

    df = df.copy()

    # ---------------------------
    # Core KPIs
    # ---------------------------
    revenue = df["Revenue"].sum()
    orders = df["InvoiceNo"].nunique()
    customers = df["CustomerID"].nunique()

    # ---------------------------
    # AOV (Average Order Value)
    # ---------------------------
    order_values = df.groupby("InvoiceNo")["Revenue"].sum()
    aov = order_values.mean()

    # ---------------------------
    # Avg Basket Size
    # ---------------------------
    basket_size = df.groupby("InvoiceNo").size().mean()

    # ---------------------------
    # Revenue per Customer
    # ---------------------------
    revenue_per_customer = revenue / customers if customers > 0 else 0

    # ---------------------------
    # Repeat Purchase Rate
    # ---------------------------
    customer_orders = df.groupby("CustomerID")["InvoiceNo"].nunique()
    repeat_customers = (customer_orders > 1).sum()
    repeat_rate = repeat_customers / customers if customers > 0 else 0

    return {
        "Revenue": revenue,
        "Orders": orders,
        "Customers": customers,
        "AOV": aov,
        "Avg_Basket_Size": basket_size,
        "Revenue_per_Customer": revenue_per_customer,
        "Repeat_Rate": repeat_rate
    }
