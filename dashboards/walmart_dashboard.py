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
    lift_data = holiday_lift(df)

    st.metric(
    "🎄 Holiday Sales Lift",
    f"{lift_data['lift_percent']:.2f}%",
    delta=None
)

    st.write(f"📊 Holiday Avg Sales: {lift_data['holiday_avg_sales']:.2f}")
    st.write(f"📊 Normal Avg Sales: {lift_data['normal_avg_sales']:.2f}")
    st.write(f"📉 P-value: {lift_data['p_value']:.5f}")

    if lift_data["significant"]:
     st.success("✅ Statistically Significant")
    else:
     st.warning("⚠️ Not Statistically Significant")

    # Trend
    weekly = df.groupby("Date")["Weekly_Sales"].sum().reset_index()

    fig = px.line(weekly, x="Date", y="Weekly_Sales", title="Weekly Sales Trend")
    st.plotly_chart(fig, width='stretch')

    # Forecast
    #st.subheader("Next 12 Weeks Forecast")
    #forecast = prophet_forecast(weekly)

    #fig2 = px.line(forecast, x="ds", y="yhat", title="Forecasted Sales")
    #st.plotly_chart(fig2, width='stretch')

    #forecast2 = arima_forecast(weekly)
    #forecast2 = forecast2.dropna(subset=["yhat"])

    '''fig3 = px.line(
        forecast2,
        x="ds",
        y="yhat",
        title="Sales Forecast (History + Next 12 Weeks)")

    st.plotly_chart(fig3, width='stretch')'''

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
        st.plotly_chart(fig_anomaly, width='stretch')


     


    


     


    
