import os
from pathlib import Path

import mlflow
import pandas as pd
import matplotlib.pyplot as plt


def main():
    inference_log_path = os.getenv(
        "INFERENCE_LOG_PATH", os.path.join("Artifacts", "inference_logs.csv")
    )
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    experiment_name = os.getenv(
        "MLFLOW_EXPERIMENT_NAME", "student-performance-prediction"
    )
    run_name = os.getenv("MLFLOW_RUN_NAME", "inference-log-analysis")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    if not Path(inference_log_path).exists():
        raise FileNotFoundError(
            f"Inference log not found at {inference_log_path}. Run the app to create logs."
        )

    df = pd.read_csv(inference_log_path)
    if df.empty:
        raise ValueError("Inference log is empty; submit predictions first.")

    summary = df.describe(include="all").fillna("")

    def log_histogram(column, bins=10):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df[column].dropna(), bins=bins, color="#2f6f5e", alpha=0.85)
        ax.set_title(f"{column.replace('_', ' ').title()} Distribution")
        ax.set_xlabel(column.replace("_", " ").title())
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.2)
        mlflow.log_figure(fig, f"plots/{column}_hist.png")
        plt.close(fig)

    def log_category_counts(column):
        counts = df[column].fillna("Unknown").value_counts()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(counts.index.astype(str), counts.values, color="#8c5a2b", alpha=0.9)
        ax.set_title(f"{column.replace('_', ' ').title()} Counts")
        ax.set_xlabel(column.replace("_", " ").title())
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        mlflow.log_figure(fig, f"plots/{column}_counts.png")
        plt.close(fig)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_metric("inference_count", len(df))
        mlflow.log_metric("predicted_math_score_mean", df["predicted_math_score"].mean())
        mlflow.log_metric("predicted_math_score_min", df["predicted_math_score"].min())
        mlflow.log_metric("predicted_math_score_max", df["predicted_math_score"].max())

        mlflow.log_dict(summary.to_dict(), "inference_summary.json")
        mlflow.log_artifact(inference_log_path)

        for column in ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]:
            counts = df[column].value_counts().to_dict()
            mlflow.log_dict(counts, f"inference_{column}_counts.json")

        for column in ["reading_score", "writing_score", "predicted_math_score"]:
            log_histogram(column, bins=12)

        for column in ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]:
            log_category_counts(column)


if __name__ == "__main__":
    main()
