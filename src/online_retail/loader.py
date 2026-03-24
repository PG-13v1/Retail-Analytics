import pandas as pd
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

@st.cache_data(ttl=3600, show_spinner="Loading Online Retail data...")
def load_online_retail():
    HF_TOKEN = os.getenv("HF_TOKEN")
    url = "https://huggingface.co/datasets/PratulG/orders/resolve/main/OnlineRetail.csv"

    df = pd.read_csv(
    url,
    storage_options={
        "Authorization": f"Bearer {HF_TOKEN}"
    })

    HF_TOKEN = os.getenv("HF_TOKEN")

    url = "https://huggingface.co/datasets/PratulG/orders/resolve/main/OnlineRetail.csv"

    df = pd.read_csv(
        url,
        storage_options={
            "Authorization": f"Bearer {HF_TOKEN}"
        }
    )

    # Data Cleaning
    df = df.dropna(subset=["CustomerID"])
    df = df[df["Quantity"] > 0]

    # Feature Engineering
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    print(df.head())

    return df


load_online_retail()
