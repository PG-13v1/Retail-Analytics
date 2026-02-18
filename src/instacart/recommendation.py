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


# ============================================================
# 1. Co-Purchase Recommendation (Simple + Powerful)
# ============================================================

def copurchase_recommender(order_products, products, product_name, top_n=10):
    """
    Recommend products frequently bought together with a given product.
    """

    merged = order_products.merge(products, on="product_id")

    # Orders containing chosen product
    target_orders = merged[merged["product_name"] == product_name]["order_id"]

    # Other products in same orders
    recommendations = (
        merged[merged["order_id"].isin(target_orders)]
        ["product_name"]
        .value_counts()
        .drop(product_name, errors="ignore")
        .head(top_n)
    )

    return recommendations.reset_index().rename(
        columns={"index": "Recommended Product", "product_name": "Count"}
    )


# ============================================================
# 2. Item-Based Collaborative Filtering (Cosine Similarity)
# ============================================================

def build_similarity_matrix(order_products, products):
    """
    Build product-product similarity matrix using cosine similarity.
    """

    merged = order_products.merge(products, on="product_id")

    # Create order-product matrix
    basket = merged.pivot_table(
        index="order_id",
        columns="product_name",
        values="add_to_cart_order",
        aggfunc="count"
    ).fillna(0)

    # Convert to binary purchase matrix
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)

    # Compute cosine similarity between products
    similarity = cosine_similarity(basket.T)

    similarity_df = pd.DataFrame(
        similarity,
        index=basket.columns,
        columns=basket.columns
    )

    return similarity_df


def similar_products(similarity_df, product_name, top_n=10):
    """
    Recommend similar products using similarity matrix.
    """

    if product_name not in similarity_df.columns:
        return f"Product '{product_name}' not found."

    sims = (
        similarity_df[product_name]
        .sort_values(ascending=False)
        .iloc[1:top_n+1]
    )

    return sims.reset_index().rename(
        columns={"index": "Similar Product", product_name: "Similarity Score"}
    )


# ============================================================
# 3. Association Rule-Based Recommendation (Apriori)
# ============================================================

from mlxtend.frequent_patterns import apriori, association_rules

def apriori_recommender(order_products, products, min_support=0.01):
    """
    Build Apriori association rules for basket recommendation.
    """

    merged = order_products.merge(products, on="product_id")

    basket = merged.pivot_table(
        index="order_id",
        columns="product_name",
        values="add_to_cart_order",
        aggfunc="count"
    ).fillna(0)

    basket = basket.applymap(lambda x: 1 if x > 0 else 0)

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

    return rules.sort_values("lift", ascending=False)
