def cohort_matrix(df):
    df["CohortMonth"] = df.groupby("CustomerID")["InvoiceDate"].transform("min").dt.to_period("M")
    df["OrderMonth"] = df["InvoiceDate"].dt.to_period("M")

    cohort = df.groupby(["CohortMonth", "OrderMonth"])["CustomerID"].nunique().reset_index()

    matrix = cohort.pivot_table(
        index="CohortMonth",
        columns="OrderMonth",
        values="CustomerID"
    )

    return matrix.fillna(0)
