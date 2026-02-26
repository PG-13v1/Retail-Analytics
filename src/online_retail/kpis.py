def compute_kpis(df):
    return {
        "Revenue": df["Revenue"].sum(),
        "Orders": df["InvoiceNo"].nunique(),
        "Customers": df["CustomerID"].nunique(),
        "AOV": df.groupby("InvoiceNo")["Revenue"].sum().mean()
    }
