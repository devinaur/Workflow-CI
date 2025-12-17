import os
os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"

import pandas as pd
import mlflow
import mlflow.sklearn

print("Tracking URI:", mlflow.get_tracking_uri())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


df = pd.read_csv("phoneprice_preprocessing.csv")

X = df.drop(columns=["price_range"])
y = df["price_range"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run():

    model = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    mlflow.log_param("C", 1.0)
    mlflow.log_param("solver", "lbfgs")

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(model, "model")

