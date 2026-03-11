#!/usr/bin/env python3
"""
Train a regression model to predict fare_amount from 4+1 features.

Features used:
- trip_distance
- trip_duration_min (derived)
- passenger_count
- RatecodeID
+ pickup_hour

Target:
- fare_amount
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

import joblib


FEATURES = [
    "trip_distance",
    "trip_duration_min",
    "passenger_count",
    "RatecodeID",
    "pickup_hour",
]
TARGET = "fare_amount"


def load_parquet(path: str) -> pd.DataFrame:
    # Pandas needs pyarrow installed to read parquet.
    return pd.read_parquet(path)


def add_trip_duration_minutes(df: pd.DataFrame) -> pd.DataFrame:
    if not {"lpep_pickup_datetime", "lpep_dropoff_datetime"}.issubset(df.columns):
        raise ValueError(
            "Missing pickup/dropoff datetime columns needed for trip_duration_min."
        )

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

    # ✅ NEW FEATURE
    df["pickup_hour"] = df["lpep_pickup_datetime"].dt.hour

    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop rows missing core columns
    needed = set(FEATURES + [TARGET])
    missing_cols = needed - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing expected columns: {sorted(missing_cols)}")

    df = df.dropna(subset=[TARGET, "trip_distance", "trip_duration_min"])

    # Filter obvious invalid/outlier rows (reasonable EDA-friendly bounds)
    df = df[df[TARGET].between(0.5, 300)]
    df = df[df["trip_distance"].between(0.01, 100)]
    df = df[df["trip_duration_min"].between(0.1, 300)]
    if "passenger_count" in df.columns:
        df = df[df["passenger_count"].between(1, 8)]
    if "RatecodeID" in df.columns:
        df = df[df["RatecodeID"].between(1, 6)]

    return df


def build_model(random_state: int = 42) -> Pipeline:
    # All selected features are numeric (or numeric-coded), so we keep preprocessing simple.
    # Impute missing values just in case.
    pre = SimpleImputer(strategy="median")

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
        max_depth=None,
        min_samples_leaf=2,
    )

    return Pipeline(
        steps=[
            ("imputer", pre),
            ("model", model),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="data/green_tripdata_2021-01.parquet",
        help="Path to parquet file",
    )
    parser.add_argument(
        "--outdir", default="models", help="Directory to save trained model"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Test split fraction"
    )
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = load_parquet(args.data)
    df = add_trip_duration_minutes(df)
    df = basic_clean(df)

    X = df[FEATURES].astype(float)
    y = df[TARGET].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    pipe = build_model(random_state=args.random_state)
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print("Regression model: RandomForestRegressor")
    print("Features:", FEATURES)
    print(f"Rows after cleaning: {len(df):,}")
    print(f"MAE : {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R^2 : {r2:.3f}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model_path = outdir / "regression_model.joblib"
    joblib.dump(pipe, model_path)

    # Save metadata for your report/grading
    meta = {
        "target": TARGET,
        "features": FEATURES,
        "model": "RandomForestRegressor",
        "metrics": {"mae": mae, "rmse": rmse, "r2": r2},
        "rows_after_cleaning": int(len(df)),
        "test_size": args.test_size,
        "random_state": args.random_state,
    }
    (outdir / "regression_model_metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"\nSaved model to: {model_path}")
    print(f"Saved metadata to: {outdir / 'regression_model_metadata.json'}")


if __name__ == "__main__":
    main()
