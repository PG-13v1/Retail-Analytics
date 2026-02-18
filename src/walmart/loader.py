import pandas as pd


def load_walmart_sales():
    df = pd.read_csv("data/walmart/train.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df
