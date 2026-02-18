import pandas as pd


def customer_reorder_behavior(orders, order_products):
    """
    Analyze customer reorder behavior:
    - reorder ratio
    - average basket size
    - loyal customers
    """

    # Merge order_products with orders to get user_id
    merged = order_products.merge(
        orders[["order_id", "user_id"]],
        on="order_id"
    )

    # -------------------------------
    # Reorder Rate per User
    # -------------------------------
    user_reorder = merged.groupby("user_id")["reordered"].mean().reset_index()
    user_reorder.columns = ["user_id", "reorder_rate"]

    # -------------------------------
    # Basket Size per Order
    # -------------------------------
    basket_size = merged.groupby("order_id").size().reset_index()
    basket_size.columns = ["order_id", "basket_size"]

    avg_basket = basket_size["basket_size"].mean()

    # -------------------------------
    # Loyal Customers (High reorder)
    # -------------------------------
    loyal_users = user_reorder.sort_values(
        "reorder_rate", ascending=False
    ).head(10)

    return {
        "avg_basket_size": avg_basket,
        "top_loyal_customers": loyal_users,
        "user_reorder_table": user_reorder
    }


def top_reordered_products(products, order_products):
    """
    Find products with highest reorder probability
    """

    merged = order_products.merge(products, on="product_id")

    reorder_stats = merged.groupby("product_name")["reordered"].mean()
    reorder_stats = reorder_stats.sort_values(ascending=False).head(15)

    return reorder_stats.reset_index()
