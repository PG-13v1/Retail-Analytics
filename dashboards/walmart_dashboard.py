import streamlit as st
import plotly.express as px

from src.walmart.loader import load_walmart_sales
from src.walmart.forecasting_prophet import prophet_forecast
from src.walmart.holiday_impact import holiday_lift
from src.walmart.forecasting_arima import arima_forecast
from src.walmart.anomaly import detect_retail_anomalies


def run():
    st.header("🏬 Walmart Demand Forecasting")

    df = load_walmart_sales()

    st.metric("Total Sales", f"${df['Weekly_Sales'].sum():,.0f}")

    # Holiday Impact
    lift = holiday_lift(df)
    st.write(f"🎄 Holiday Sales Lift: {lift:.2f}%")

    # Trend
    weekly = df.groupby("Date")["Weekly_Sales"].sum().reset_index()

    fig = px.line(weekly, x="Date", y="Weekly_Sales", title="Weekly Sales Trend")
    st.plotly_chart(fig, use_container_width=True)

    # Forecast
    st.subheader("Next 12 Weeks Forecast")
    forecast = prophet_forecast(weekly)

    fig2 = px.line(forecast, x="ds", y="yhat", title="Forecasted Sales")
    st.plotly_chart(fig2, use_container_width=True)

    forecast2 = arima_forecast(weekly)
    forecast2 = forecast2.dropna(subset=["yhat"])

    fig3 = px.line(
        forecast2,
        x="ds",
        y="yhat",
        title="Sales Forecast (History + Next 12 Weeks)")

    st.plotly_chart(fig3, use_container_width=True)

    # Anomaly Detection
    st.subheader("Anomaly Detection")
    anomalies = detect_retail_anomalies(df)
    st.write(f"Detected {len(anomalies)} anomalies in the data")
    if len(anomalies) > 0:
        st.dataframe(anomalies)
        
        # Plot anomalies
        fig_anomaly = px.scatter(
            anomalies,
            x="Date",
            y="Weekly_Sales",
            title="Detected Anomalies in Sales",
            color_discrete_sequence=["red"]
        )
        st.plotly_chart(fig_anomaly, use_container_width=True)


     


    
