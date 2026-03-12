#!/usr/bin/env python3
"""
Version 2 evaluation:
- combine existing + new data
- split again into train/test
- evaluate the existing trained model on the new test set
- keep training code and model unchanged
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

FEATURES = [
    "trip_distance",
    "trip_duration_min",
    "passenger_count",
    "RatecodeID",
    "pickup_hour",
]
TARGET = "fare_amount"


def load_parquet_files(paths: list[str]) -> pd.DataFrame:
    dfs = [pd.read_parquet(path) for path in paths]
    return pd.concat(dfs, ignore_index=True)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    if not {"lpep_pickup_datetime", "lpep_dropoff_datetime"}.issubset(df.columns):
        raise ValueError("Missing pickup/dropoff datetime columns.")

    df = df.copy()
    df["lpep_pickup_datetime"] = pd.to_datetime(
        df["lpep_pickup_datetime"], errors="coerce"
    )
    df["lpep_dropoff_datetime"] = pd.to_datetime(
        df["lpep_dropoff_datetime"], errors="coerce"
    )

    df["trip_duration_min"] = (
        df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]
    ).dt.total_seconds() / 60.0

    df["pickup_hour"] = df["lpep_pickup_datetime"].dt.hour
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    needed = set(FEATURES + [TARGET])
    missing_cols = needed - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing expected columns: {sorted(missing_cols)}")

    df = df.dropna(subset=[TARGET, "trip_distance", "trip_duration_min"])

    df = df[df[TARGET].between(0.5, 300)]
    df = df[df["trip_distance"].between(0.01, 100)]
    df = df[df["trip_duration_min"].between(0.1, 300)]

    if "passenger_count" in df.columns:
        df = df[df["passenger_count"].between(1, 8)]

    if "RatecodeID" in df.columns:
        df = df[df["RatecodeID"].between(1, 6)]

    if "pickup_hour" in df.columns:
        df = df[df["pickup_hour"].between(0, 23)]

    return df


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-files",
        nargs="+",
        default=[
            "data/green_tripdata_2021-01.parquet",
            "data/green_tripdata_2021-02.parquet",
        ],
        help="List of parquet files to combine",
    )
    parser.add_argument(
        "--model-path",
        default="models/regression_model.joblib",
        help="Path to existing trained model",
    )
    parser.add_argument(
        "--outdir",
        default="models",
        help="Directory to save evaluation results",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = load_parquet_files(args.data_files)
    rows_before_cleaning = len(df)

    df = add_derived_features(df)
    df = basic_clean(df)
    rows_after_cleaning = len(df)

    X = df[FEATURES].astype(float)
    y = df[TARGET].astype(float)

    # New split for version 2
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    model = joblib.load(args.model_path)
    metrics = evaluate_model(model, X_test, y_test)

    print("Version 2 evaluation")
    print("Features:", FEATURES)
    print(f"Rows before cleaning: {rows_before_cleaning:,}")
    print(f"Rows after cleaning : {rows_after_cleaning:,}")
    print(f"Test rows           : {len(X_test):,}")
    print(f"MAE                 : {metrics['mae']:.3f}")
    print(f"RMSE                : {metrics['rmse']:.3f}")
    print(f"R^2                 : {metrics['r2']:.3f}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    result = {
        "version": 2,
        "data_files": args.data_files,
        "model_path": args.model_path,
        "features": FEATURES,
        "target": TARGET,
        "rows_before_cleaning": rows_before_cleaning,
        "rows_after_cleaning": rows_after_cleaning,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "metrics": metrics,
        "notes": [
            "Existing trained model reused without retraining.",
            "Training code unchanged.",
            "Data changed from January-only to January+February combined.",
            "Evaluation code changed to build a new split from combined data.",
        ],
    }

    output_path = outdir / "regression_v2_evaluation.json"
    output_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved evaluation results to: {output_path}")


if __name__ == "__main__":
    main()
