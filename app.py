import streamlit as st
from streamlit_option_menu import option_menu

from dashboards.retail_dashboard import run as retail_run
from dashboards.walmart_dashboard import run as walmart_run
from dashboards.instacart_dashboard import run as instacart_run

st.set_page_config(page_title="Retail Analytics Suite", layout="wide")

st.title("🧠 Retail Analytics Intelligence Suite")

with st.sidebar:
    selected = option_menu(
        "Choose Dashboard",
        ["Online Retail", "Walmart Forecasting", "Instacart Basket"],
        icons=["cart", "graph-up", "basket"],
        menu_icon="layers",
        default_index=0
    )

if selected == "Online Retail":
    retail_run()

elif selected == "Walmart Forecasting":
    walmart_run()

elif selected == "Instacart Basket":
    instacart_run()
