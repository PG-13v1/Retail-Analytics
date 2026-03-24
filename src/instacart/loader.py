


import pandas as pd
import os

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

@st.cache_data(ttl=3600, show_spinner="Loading Instacart data...")
def load_instacart():

    HF_TOKEN = os.getenv("HF_TOKEN")

    url_1 = "https://huggingface.co/datasets/PratulG/orders/resolve/main/orders.csv"
    orders = pd.read_csv(
    url_1,
    storage_options={
        "Authorization": f"Bearer {HF_TOKEN}"
    })

    url_2 = "https://huggingface.co/datasets/PratulG/orders/resolve/main/products.csv"
    products = pd.read_csv(
    url_2,
    storage_options={
        "Authorization": f"Bearer {HF_TOKEN}"
    })

    url_3 = "https://huggingface.co/datasets/PratulG/orders/resolve/main/order_products__train.csv"
    order_products = pd.read_csv(
    url_3,
    storage_options={
        "Authorization": f"Bearer {HF_TOKEN}"
    })


    merged = order_products.merge(products, on="product_id")
    print(merged.head())

