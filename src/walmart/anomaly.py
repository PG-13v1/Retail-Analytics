import pandas as pd
from pyod.models.iforest import IForest
from pyod.models.knn import KNN


def detect_retail_anomalies(df):
    """
    Detect anomalies in retail transactions using:
    - Isolation Forest
    - KNN Outlier Detection

    Returns top suspicious transactions.
    """

    X = df[["Weekly_Sales"]]

    # ---------------------------
    # Model 1: Isolation Forest
    # ---------------------------
    iforest = IForest(contamination=0.02)
    iforest.fit(X)

    df["IForest_Anomaly"] = iforest.predict(X)

    # ---------------------------
    # Model 2: KNN Outlier
    # ---------------------------
    knn = KNN(contamination=0.02)
    knn.fit(X)

    df["KNN_Anomaly"] = knn.predict(X)

    # ---------------------------
    # Combine anomaly flags
    # ---------------------------
    anomalies = df[
        (df["IForest_Anomaly"] == 1) |
        (df["KNN_Anomaly"] == 1)
    ]

    anomalies = anomalies.sort_values("Weekly_Sales", ascending=False)

    return anomalies.head(20)
