import streamlit as st

from src.instacart.loader import load_instacart
from src.instacart.top_products import most_popular_products


def run():
    st.header("🥦 Instacart Market Basket Analytics")

    merged = load_instacart()

    st.metric("Total Purchases", merged.shape[0])

    st.subheader("Top Purchased Products")
    top = most_popular_products(merged)

    st.dataframe(top)
