from mlxtend.frequent_patterns import apriori, association_rules

def apriori_rules(df):
    basket = df.pivot_table(
        index="order_id",
        columns="product_name",
        values="add_to_cart_order",
        aggfunc="count"
    ).fillna(0)

    basket = basket.applymap(lambda x: 1 if x > 0 else 0)

    frequent = apriori(basket, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent, metric="lift", min_threshold=1.3)

    return rules.sort_values("lift", ascending=False)
