import pandas as pd
import streamlit as st
from prophet import Prophet


@st.cache_data(show_spinner="Training forecasting model...")
def prophet_forecast(weekly_df):

    df = weekly_df.copy()

    # -------------------------------
    # Basic validation
    # -------------------------------
    if "Date" not in df.columns or "Weekly_Sales" not in df.columns:
        raise ValueError("Required columns missing")

    # -------------------------------
    # Preprocessing
    # -------------------------------
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    # Fill missing weeks
    df = (
        df.set_index("Date")
        .asfreq("W")
        .ffill()
        .bfill()
        .reset_index()
    )

    df = df.rename(columns={
        "Date": "ds",
        "Weekly_Sales": "y"
    })

    # Drop NaNs
    df = df.dropna(subset=["y"])

    # -------------------------------
    # SAFETY CHECK
    # -------------------------------
    if df.shape[0] < 10:
        return fallback_forecast(df)

    # -------------------------------
    # Prophet Model
    # -------------------------------
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False
    )

    model.add_seasonality(name="monthly", period=30.5, fourier_order=5)

    model.fit(df)

    # -------------------------------
    # Future
    # -------------------------------
    future = model.make_future_dataframe(periods=12, freq="W")
    forecast = model.predict(future)

    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper", "trend"]]


def fallback_forecast(df):

    df = df.copy()

    # ❌ If no valid dates → stop
    if df.empty or df["ds"].isna().all():
        raise ValueError("Fallback failed: No valid dates available")

    # Moving average
    df["yhat"] = df["y"].rolling(3, min_periods=1).mean()

    last_date = df["ds"].dropna().max()

    if pd.isna(last_date):
        raise ValueError("Invalid last date for forecasting")

    # ✅ FIX: ensure start is valid
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(weeks=1),
        periods=12,
        freq="W"
    )

    future_df = pd.DataFrame({
        "ds": future_dates,
        "yhat": df["yhat"].iloc[-1]
    })

    future_df["yhat_lower"] = future_df["yhat"] * 0.9
    future_df["yhat_upper"] = future_df["yhat"] * 1.1
    future_df["trend"] = future_df["yhat"]

    return pd.concat([
        df[["ds", "yhat"]],
        future_df
    ], ignore_index=True)
