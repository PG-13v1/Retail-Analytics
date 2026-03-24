"""
recommendation.py

Instacart Product Recommendation Engine
(Compiler-free, Windows-safe)

Includes:
1. Co-purchase recommendations
2. Item-based collaborative filtering (cosine similarity)
3. Basket-based association rules recommender
"""

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from mlxtend.frequent_patterns import apriori, association_rules
import streamlit as st

@st.cache_data(show_spinner="Generating co-purchase recommendations...")
def copurchase_recommender(order_products, products, product_name, top_n=10):

    merged = order_products.merge(products, on="product_id")
    merged["product_name"] = merged["product_name"].str.lower().str.strip()
    product_name = product_name.lower().strip()

    target_orders = merged.loc[
        merged["product_name"] == product_name, "order_id"
    ]

    recommendations = (
        merged.loc[merged["order_id"].isin(target_orders), "product_name"]
        .value_counts()
        .drop(product_name, errors="ignore")
        .head(top_n)
        .reset_index()
    )

    recommendations.columns = ["Recommended Product", "Count"]

    return recommendations


# ============================================================
# 2. Item-Based Collaborative Filtering (Cosine Similarity)
# ============================================================

@st.cache_resource(show_spinner="Building similarity matrix...")
def build_similarity_matrix(order_products, products):

    merged = order_products.merge(products, on="product_id")

    # Reduce dimensionality (IMPORTANT)
    top_products = merged["product_name"].value_counts().head(200).index
    merged = merged[merged["product_name"].isin(top_products)]

    basket = merged.pivot_table(
        index="order_id",
        columns="product_name",
        values="add_to_cart_order",
        aggfunc="count"
    ).fillna(0)

    # FAST binarization
    basket = (basket > 0).astype("int8")

    similarity = cosine_similarity(basket.T)

    similarity_df = pd.DataFrame(
        similarity,
        index=basket.columns,
        columns=basket.columns
    )

    return similarity_df


def similar_products(similarity_df, product_name, top_n=10):

    product_name = product_name.lower().strip()

    if product_name not in similarity_df.columns:
        suggestions = [
            p for p in similarity_df.columns if product_name in p
        ][:5]

        return {
            "error": f"'{product_name}' not found",
            "suggestions": suggestions
        }

    sims = similarity_df[product_name].sort_values(ascending=False)

    sims = sims.iloc[1:top_n+1].reset_index()
    sims.columns = ["Similar Product", "Similarity Score"]

    return sims

# ============================================================
# 3. Association Rule-Based Recommendation (Apriori)
# ============================================================

from mlxtend.frequent_patterns import apriori, association_rules

from mlxtend.frequent_patterns import apriori, association_rules

@st.cache_data(show_spinner="Running Apriori...")
def apriori_recommender(order_products, products, min_support=0.01):

    merged = order_products.merge(products, on="product_id")

    # Reduce size
    top_products = merged["product_name"].value_counts().head(100).index
    merged = merged[merged["product_name"].isin(top_products)]

    basket = merged.pivot_table(
        index="order_id",
        columns="product_name",
        values="add_to_cart_order",
        aggfunc="count"
    ).fillna(0)

    # FAST binarization
    basket = (basket > 0).astype("int8")

    frequent_itemsets = apriori(
        basket,
        min_support=min_support,
        use_colnames=True
    )

    rules = association_rules(
        frequent_itemsets,
        metric="lift",
        min_threshold=1.2
    )

    # Clean output
    rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
    rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))

    return rules.sort_values("lift", ascending=False)[
        ["antecedents", "consequents", "support", "confidence", "lift"]
    ]
