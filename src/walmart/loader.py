import pandas as pd


def load_walmart_sales():
    df = pd.read_csv("https://drive.google.com/file/d/1qElZAoD8f78OcpVoKiz_YAot-Ft93jtj/view?usp=sharing")
    df["Date"] = pd.to_datetime(df["Date"])
    return df
