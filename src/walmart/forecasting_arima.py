import pandas as pd
import pmdarima as pm

import pandas as pd
import pmdarima as pm

def arima_forecast(weekly_df):
    """
    AutoARIMA forecast + historical data combined (Prophet-style output).
    Returns one dataframe for plotting full series.
    """

    # Ensure datetime format
    weekly_df["Date"] = pd.to_datetime(weekly_df["Date"])
    weekly_df = weekly_df.sort_values("Date")

    # -------------------------------
    # 1. Historical Data
    # -------------------------------
    history_df = weekly_df.rename(columns={
        "Date": "ds",
        "Weekly_Sales": "y"
    })

    # Time series values
    y = history_df["y"].values

    # -------------------------------
    # 2. Fit AutoARIMA Model
    # -------------------------------
    model = pm.auto_arima(
        y,
        seasonal=True,
        m=52,
        stepwise=True,
        suppress_warnings=True,
        trace=False
    )

    # -------------------------------
    # 3. Forecast Next 12 Weeks
    # -------------------------------
    n_periods = 12
    forecast, conf_int = model.predict(
        n_periods=n_periods,
        return_conf_int=True
    )

    # Future dates
    last_date = history_df["ds"].max()

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
    # 4. Add Historical as yhat too
    # -------------------------------
    history_plot_df = history_df.copy()
    history_plot_df["yhat"] = history_plot_df["y"]

    # No confidence interval for history
    history_plot_df["yhat_lower"] = None
    history_plot_df["yhat_upper"] = None

    # -------------------------------
    # 5. Combine History + Forecast
    # -------------------------------
    full_df = pd.concat(
        [history_plot_df[["ds", "yhat", "yhat_lower", "yhat_upper"]],
         forecast_df],
        ignore_index=True
    )

    return full_df

