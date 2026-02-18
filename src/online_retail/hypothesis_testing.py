import scipy.stats as stats

def revenue_ttest(df):
    high = df[df["Revenue"] > df["Revenue"].median()]["Revenue"]
    low = df[df["Revenue"] <= df["Revenue"].median()]["Revenue"]

    t_stat, p_val = stats.ttest_ind(high, low)

    return {
        "t_statistic": t_stat,
        "p_value": p_val,
        "significant": p_val < 0.05
    }