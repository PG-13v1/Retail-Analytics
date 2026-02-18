from gensim.models import Word2Vec


def train_product_embeddings(order_products, products):
    """
    Train product embeddings based on co-purchase behavior.
    Each order = a sentence, each product = a word.
    """

    # Merge product names
    merged = order_products.merge(products, on="product_id")

    # Build "sentences": list of products per order
    orders = (
        merged.groupby("order_id")["product_name"]
        .apply(list)
        .tolist()
    )

    # Train Word2Vec model
    model = Word2Vec(
        sentences=orders,
        vector_size=50,
        window=5,
        min_count=5,
        workers=4
    )

    return model


def similar_products(model, product_name, top_n=10):
    """
    Recommend similar products based on embedding similarity.
    """

    if product_name not in model.wv:
        return f"Product '{product_name}' not in vocabulary."

    sims = model.wv.most_similar(product_name, topn=top_n)

    return sims
