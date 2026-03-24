import pandas as pd
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

@st.cache_data(ttl=3600, show_spinner="Loading Walmart sales data...")
def load_walmart_sales():
    HF_TOKEN = os.getenv("HF_TOKEN")

    url = "https://huggingface.co/datasets/PratulG/orders/resolve/main/train.csv"

    df = pd.read_csv(
        url,
        parse_dates=["Date"],   # faster than converting later
        dtype={
            "Store": "int16",
            "Dept": "int16",
            "Weekly_Sales": "float32",
            "IsHoliday": "bool"
        },
        storage_options={
            "Authorization": f"Bearer {HF_TOKEN}"
        }
    )

    return df
