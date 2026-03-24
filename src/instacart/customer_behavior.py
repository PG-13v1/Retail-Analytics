import pandas as pd
import streamlit as st


@st.cache_data(show_spinner="Analyzing customer behavior...")
def customer_reorder_behavior(orders, order_products):

    merged = order_products.merge(
        orders[["order_id", "user_id"]],
        on="order_id"
    )

    # -------------------------------
    # Basket Size
    # -------------------------------
    basket_size = merged.groupby("order_id").size().rename("basket_size")
    avg_basket = basket_size.mean()

    # -------------------------------
    # Orders per User
    # -------------------------------
    user_orders = merged.groupby("user_id")["order_id"].nunique()

    # -------------------------------
    # Reorder Rate per User
    # -------------------------------
    user_reorder = merged.groupby("user_id")["reordered"].mean()

    user_stats = pd.concat([user_orders, user_reorder], axis=1)
    user_stats.columns = ["total_orders", "reorder_rate"]

    # -------------------------------
    # Filter meaningful users
    # -------------------------------
    user_stats = user_stats[user_stats["total_orders"] > 3]

    # -------------------------------
    # Customer Segmentation
    # -------------------------------
    def segment(row):
        if row["reorder_rate"] > 0.7:
            return "Gold"
        elif row["reorder_rate"] > 0.4:
            return "Silver"
        else:
            return "Bronze"

    user_stats["segment"] = user_stats.apply(segment, axis=1)

    # -------------------------------
    # Top loyal users
    # -------------------------------
    loyal_users = user_stats.sort_values(
        "reorder_rate", ascending=False
    ).head(10)

    return {
        "avg_basket_size": avg_basket,
        "top_loyal_customers": loyal_users.reset_index(),
        "user_reorder_table": user_stats.reset_index()
    }


@st.cache_data(show_spinner="Analyzing product reorder patterns...")
def top_reordered_products(products, order_products):

    merged = order_products.merge(products, on="product_id")

    reorder_stats = merged.groupby("product_name").agg(
        reorder_rate=("reordered", "mean"),
        total_orders=("reordered", "count")
    )

    # Filter noisy products
    reorder_stats = reorder_stats[reorder_stats["total_orders"] > 50]

    reorder_stats = reorder_stats.sort_values(
        "reorder_rate", ascending=False
    ).head(15)

    return reorder_stats.reset_index()
