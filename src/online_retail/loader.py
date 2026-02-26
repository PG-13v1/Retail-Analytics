import pandas as pd


def load_online_retail():
    df = pd.read_csv("data/online_retail/OnlineRetail.csv")

    df = df.dropna(subset=["CustomerID"])
    df = df[df["Quantity"] > 0]

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    df = df.rename(columns={"Customer ID": "CustomerID"})

    return df
