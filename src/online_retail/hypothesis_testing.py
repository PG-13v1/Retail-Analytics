import streamlit as st
import scipy.stats as stats


@st.cache_data(show_spinner="Running statistical test...")
def revenue_ttest(df):

    if "Revenue" not in df.columns:
        raise ValueError("Revenue column missing")

    # ---------------------------
    # Example: Holiday vs Non-Holiday

    group1 = df["Revenue"].dropna()
    group2 = df["Revenue"].dropna()

    # ---------------------------
    # Welch’s t-test (safer)
    # ---------------------------
    t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)

    return {
        "group1_mean": group1.mean(),
        "group2_mean": group2.mean(),
        "t_statistic": t_stat,
        "p_value": p_val,
        "significant": p_val < 0.05
    }
