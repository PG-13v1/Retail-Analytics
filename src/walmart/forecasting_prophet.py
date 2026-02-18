from prophet import Prophet


def prophet_forecast(weekly_df):
    prophet_df = weekly_df.rename(columns={"Date": "ds", "Weekly_Sales": "y"})

    model = Prophet()
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=12, freq="W")
    forecast = model.predict(future)

    return forecast
