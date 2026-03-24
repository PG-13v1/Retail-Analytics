import streamlit as st
import scipy.stats as stats


@st.cache_data(show_spinner="Computing holiday lift...")
def holiday_lift(train_df):

    df = train_df.copy()

    if "Weekly_Sales" not in df.columns or "IsHoliday" not in df.columns:
        raise ValueError("Required columns missing")

    holiday = df[df["IsHoliday"] == True]["Weekly_Sales"].dropna()
    normal = df[df["IsHoliday"] == False]["Weekly_Sales"].dropna()

    # ---------------------------
    # Mean sales
    # ---------------------------
    holiday_sales = holiday.mean()
    normal_sales = normal.mean()

    # ---------------------------
    # Lift calculation
    # ---------------------------
    lift = (
        (holiday_sales - normal_sales) / normal_sales * 100
        if normal_sales != 0 else 0
    )

    # ---------------------------
    # Statistical significance
    # ---------------------------
    t_stat, p_val = stats.ttest_ind(holiday, normal, equal_var=False)

    return {
        "holiday_avg_sales": holiday_sales,
        "normal_avg_sales": normal_sales,
        "lift_percent": lift,
        "p_value": p_val,
        "significant": p_val < 0.05,
        "holiday_samples": len(holiday),
        "normal_samples": len(normal)
    }
