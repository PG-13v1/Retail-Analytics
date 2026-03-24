import pandas as pd
import streamlit as st
import pmdarima as pm


@st.cache_data(show_spinner="Training ARIMA model...")
def arima_forecast(weekly_df):

    df = weekly_df.copy()

    # -------------------------------
    # Ensure datetime + sorting
    # -------------------------------
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    # -------------------------------
    # Handle missing weeks (IMPORTANT)
    # -------------------------------
    df = df.set_index("Date").asfreq("W").fillna(method="ffill").reset_index()

    # -------------------------------
    # Rename for consistency
    # -------------------------------
    df = df.rename(columns={
        "Date": "ds",
        "Weekly_Sales": "y"
    })

    y = df["y"].values

    # -------------------------------
    # Train model (safe config)
    # -------------------------------
    try:
        model = pm.auto_arima(
            y,
            seasonal=True,
            m=52,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore"
        )
    except:
        # fallback if seasonal fails
        model = pm.auto_arima(
            y,
            seasonal=False,
            stepwise=True
        )

    # -------------------------------
    # Forecast
    # -------------------------------
    n_periods = 12
    forecast, conf_int = model.predict(
        n_periods=n_periods,
        return_conf_int=True
    )

    # Future dates
    last_date = df["ds"].max()
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(weeks=1),
        periods=n_periods,
        freq="W"
    )

    forecast_df = pd.DataFrame({
        "ds": future_dates,
        "yhat": forecast,
        "yhat_lower": conf_int[:, 0],
        "yhat_upper": conf_int[:, 1]
    })

    # -------------------------------
    # Historical
    # -------------------------------
    history_df = df.copy()
    history_df["yhat"] = history_df["y"]
    history_df["yhat_lower"] = None
    history_df["yhat_upper"] = None

    # -------------------------------
    # Combine
    # -------------------------------
    full_df = pd.concat(
        [
            history_df[["ds", "yhat", "yhat_lower", "yhat_upper"]],
            forecast_df
        ],
        ignore_index=True
    )

    return full_df

