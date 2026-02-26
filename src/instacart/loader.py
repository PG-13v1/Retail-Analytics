import pandas as pd


def load_instacart():
    orders = pd.read_csv("https://drive.google.com/file/d/18EJoLq1tl0gwpHkn4Iv099ctaczZWlYK/view?usp=sharing")
    products = pd.read_csv("https://drive.google.com/file/d/1UDJocoMhzZZFYEdq61pii7epE1g0j5Fi/view?usp=sharing")
    order_products = pd.read_csv("https://drive.google.com/file/d/19DUKQWkLh6f4BqsgoBoTbwiUF5w7nscp/view?usp=sharing")

    merged = order_products.merge(products, on="product_id")
    return merged
