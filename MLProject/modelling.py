import argparse
import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="phoneprice_preprocessing.csv",
        help="Path ke dataset"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("Tracking URI:", mlflow.get_tracking_uri())

    active_run = mlflow.active_run()
    parent_run_context = None

    if not active_run:
        parent_run_context = mlflow.start_run(run_name="PhonePriceClassification")

    if parent_run_context:
        parent_run_context.__enter__()

    try:
        if not os.path.exists(args.data_path):
            raise FileNotFoundError(f"Dataset tidak ditemukan: {args.data_path}")

        df = pd.read_csv(args.data_path)

        X = df.drop(columns=["price_range"])
        y = df["price_range"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Model
        model = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=1000
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted")
        rec = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        # Log params
        mlflow.log_param("C", 1.0)
        mlflow.log_param("solver", "lbfgs")
        mlflow.log_param("max_iter", 1000)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # Log model
        mlflow.sklearn.log_model(
            model,
            artifact_path="model"
        )

        print("berhasil")

    finally:
        if parent_run_context:
            parent_run_context.__exit__(None, None, None)

if __name__ == "__main__":
    main()