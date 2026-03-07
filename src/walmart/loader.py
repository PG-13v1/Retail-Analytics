import pandas as pd
import os


def load_walmart_sales():
    HF_TOKEN = os.getenv("HF_TOKEN")
    url = "https://huggingface.co/datasets/PratulG/orders/resolve/main/train.csv"

    df = pd.read_csv(
    url,
    storage_options={
        "Authorization": f"Bearer {HF_TOKEN}"
    })

    df["Date"] = pd.to_datetime(df["Date"])
    return df
