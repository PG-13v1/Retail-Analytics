from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def rfm_segmentation(df):
    today = df["InvoiceDate"].max()

    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (today - x.max()).days,
        "InvoiceNo": "nunique",
        "Revenue": "sum"
    })

    rfm.columns = ["Recency", "Frequency", "Monetary"]

    scaler = StandardScaler()
    X = scaler.fit_transform(rfm)

    model = KMeans(n_clusters=4, random_state=42)
    rfm["Cluster"] = model.fit_predict(X)

    return rfm.reset_index()
