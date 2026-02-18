def compute_kpis(df):
    return {
        "Revenue": df["Revenue"].sum(),
        "Orders": df["Invoice"].nunique(),
        "Customers": df["CustomerID"].nunique(),
        "AOV": df.groupby("Invoice")["Revenue"].sum().mean()
    }
