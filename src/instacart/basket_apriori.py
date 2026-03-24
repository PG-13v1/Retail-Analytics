import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules


@st.cache_data(show_spinner="Running market basket analysis...")
def apriori_rules(df):

    # ---------------------------
    # Reduce dimensionality (IMPORTANT)
    # ---------------------------
    top_products = df["product_name"].value_counts().head(100).index
    df = df[df["product_name"].isin(top_products)]

    # ---------------------------
    # Basket creation
    # ---------------------------
    basket = df.pivot_table(
        index="order_id",
        columns="product_name",
        values="add_to_cart_order",
        aggfunc="count"
    ).fillna(0)

    # ---------------------------
    # FAST binarization
    # ---------------------------
    basket = (basket > 0).astype("int8")

    # ---------------------------
    # Apriori
    # ---------------------------
    frequent = apriori(basket, min_support=0.01, use_colnames=True)

    # ---------------------------
    # Association Rules
    # ---------------------------
    rules = association_rules(frequent, metric="lift", min_threshold=1.3)

    # ---------------------------
    # Clean output (important)
    # ---------------------------
    rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
    rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))

    # ---------------------------
    # Sort by strongest rules
    # ---------------------------
    rules = rules.sort_values("lift", ascending=False)

    return rules[["antecedents", "consequents", "support", "confidence", "lift"]]
