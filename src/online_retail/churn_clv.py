from lifetimes import BetaGeoFitter
from lifetimes.utils import summary_data_from_transaction_data

def clv_model(df):
    today = df["InvoiceDate"].max()

    summary = summary_data_from_transaction_data(
        df,
        customer_id_col="CustomerID",
        datetime_col="InvoiceDate",
        monetary_value_col="Revenue",
        observation_period_end=today
    )

    bgf = BetaGeoFitter()
    bgf.fit(summary["frequency"], summary["recency"], summary["T"])

    summary["Predicted_30_Day_Purchases"] = bgf.conditional_expected_number_of_purchases_up_to_time(
        30,
        summary["frequency"],
        summary["recency"],
        summary["T"]
    )

    return summary.sort_values("Predicted_30_Day_Purchases", ascending=False)
