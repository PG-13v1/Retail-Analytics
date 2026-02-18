import pandas as pd


def load_online_retail():
    df = pd.read_csv("data/online_retail/online_retail.csv")

    df = df.dropna(subset=["Customer ID"])
    df = df[df["Quantity"] > 0]

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Revenue"] = df["Quantity"] * df["Price"]
    df = df.rename(columns={"Customer ID": "CustomerID"})

    return df
