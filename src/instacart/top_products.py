def most_popular_products(merged, top_n=15):
    """
    Returns the top N most frequently purchased products.
    """

    top = (
        merged["product_name"]
        .value_counts()
        .head(top_n)
        .reset_index()
    )

    top.columns = ["Product", "Count"]

    return top