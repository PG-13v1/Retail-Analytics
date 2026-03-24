import streamlit as st
from gensim.models import Word2Vec


@st.cache_resource(show_spinner="Training product embeddings...")
def train_product_embeddings(order_products, products):

    merged = order_products.merge(products, on="product_id")

    # ---------------------------
    # Clean product names
    # ---------------------------
    merged["product_name"] = merged["product_name"].str.lower().str.strip()

    # ---------------------------
    # Build sentences
    # ---------------------------
    orders = (
        merged.groupby("order_id")["product_name"]
        .apply(list)
        .tolist()
    )

    # ---------------------------
    # Train Word2Vec
    # ---------------------------
    model = Word2Vec(
        sentences=orders,
        vector_size=64,
        window=5,
        min_count=5,
        workers=4,
        sg=1  # Skip-gram (better for recommendations)
    )

    return model


def similar_products(model, product_name, top_n=10):

    product_name = product_name.lower().strip()

    if product_name not in model.wv:
        # Suggest closest match
        vocab = list(model.wv.index_to_key)
        suggestions = [p for p in vocab if product_name in p][:5]

        return {
            "error": f"'{product_name}' not found",
            "suggestions": suggestions
        }

    sims = model.wv.most_similar(product_name, topn=top_n)

    return [
        {"product": prod, "similarity": score}
        for prod, score in sims
    ]
