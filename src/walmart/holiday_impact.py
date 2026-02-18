def holiday_lift(train_df):
    holiday_sales = train_df[train_df["IsHoliday"] == True]["Weekly_Sales"].mean()
    normal_sales = train_df[train_df["IsHoliday"] == False]["Weekly_Sales"].mean()

    lift = (holiday_sales - normal_sales) / normal_sales * 100

    return lift
