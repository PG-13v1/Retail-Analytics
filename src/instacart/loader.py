import pandas as pd


def load_instacart():
    orders = pd.read_csv("data/instacart/orders.csv")
    products = pd.read_csv("data/instacart/products.csv")
    order_products = pd.read_csv("data/instacart/order_products__train.csv")

    merged = order_products.merge(products, on="product_id")
    return merged
