import pandas as pd
import os

def load_online_retail():
    HF_TOKEN = os.getenv("HF_TOKEN")
    url = "https://huggingface.co/datasets/PratulG/orders/resolve/main/OnlineRetail.csv"

    df = pd.read_csv(
    url,
    storage_options={
        "Authorization": f"Bearer {HF_TOKEN}"
    })

    df = df.dropna(subset=["CustomerID"])
    df = df[df["Quantity"] > 0]

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    df = df.rename(columns={"Customer ID": "CustomerID"})

    return df

