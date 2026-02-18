from pyod.models.iforest import IForest
from pyod.models.knn import KNN

def detect_anomalies(df):
    if "Revenue" not in df.columns:
        raise ValueError("'Revenue' column not found in dataframe. Available columns: " + str(list(df.columns)))
    
    X = df[["Revenue"]].values.reshape(-1, 1)

    iforest = IForest(contamination=0.02)
    knn = KNN(contamination=0.02)

    df["IForest_Score"] = iforest.fit_predict(X)
    df["KNN_Score"] = knn.fit_predict(X)

    anomalies = df[(df["IForest_Score"] == 1) & (df["KNN_Score"] == 1)]
    return anomalies
