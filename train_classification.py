#!/usr/bin/env python3
"""
Train a multiclass classifier to predict payment_type.

Features (4):
- trip_distance
- trip_duration_min (derived from pickup/dropoff timestamps)
- passenger_count
- RatecodeID

Target:
- payment_type

Outputs:
- prints metrics (accuracy, macro precision/recall/F1) and confusion matrix
- saves metadata JSON to models/classification_model_metadata.json
- optionally saves model pipeline to models/classification_model.joblib (recommended to ignore in git)
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
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier

import joblib


FEATURES = ["trip_distance", "trip_duration_min", "passenger_count", "RatecodeID"]
TARGET = "payment_type"


def load_parquet(path: str) -> pd.DataFrame:
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
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    needed = set(FEATURES + [TARGET])
    missing_cols = needed - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing expected columns: {sorted(missing_cols)}")

    # Drop missing core values
    df = df.dropna(subset=["trip_distance", "trip_duration_min", TARGET])

    # Reasonable bounds to remove obvious junk/outliers
    df = df[df["trip_distance"].between(0.01, 100)]
    df = df[df["trip_duration_min"].between(0.1, 300)]

    if "passenger_count" in df.columns:
        df = df[df["passenger_count"].between(1, 8)]
    if "RatecodeID" in df.columns:
        # typical TLC codes are small ints; keep sensible range
        df = df[df["RatecodeID"].between(1, 6)]

    # payment_type is usually small integer codes.
    # Common TLC codes:
    # 1=Credit card, 2=Cash, 3=No charge, 4=Dispute, 5=Unknown, 6=Voided trip
    # We'll keep 2 or up to 5 by default (drop 6), unless user overrides.
    return df


def build_model(random_state: int = 42) -> Pipeline:
    pre = SimpleImputer(strategy="median")

    clf = RandomForestClassifier(
        n_estimators=400,
        random_state=random_state,
        n_jobs=-1,
        min_samples_leaf=2,
        class_weight="balanced",
    )

    return Pipeline(
        steps=[
            ("imputer", pre),
            ("model", clf),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default="green_tripdata_2021-01.parquet", help="Path to parquet file"
    )
    parser.add_argument(
        "--outdir", default="models", help="Directory to save metadata/model"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Test split fraction"
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--allowed-payment-types",
        default="1,2",
        help="Comma-separated list of allowed payment_type codes to keep (default: 1,2)",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="If set, save joblib model to outdir (keep ignored by git).",
    )
    args = parser.parse_args()

    allowed = [
        int(x.strip()) for x in args.allowed_payment_types.split(",") if x.strip()
    ]
    allowed_set = set(allowed)

    df = load_parquet(args.data)
    df = add_trip_duration_minutes(df)
    df = basic_clean(df)

    # Keep only allowed payment types
    df = df[df[TARGET].astype(int).isin(allowed_set)]

    # Basic class distribution (useful for your write-up)
    class_counts = df[TARGET].astype(int).value_counts().sort_index()
    print("Class distribution (payment_type -> count):")
    for k, v in class_counts.items():
        print(f"  {k}: {v:,}")

    # Prepare X/y
    X = df[FEATURES].astype(float)
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    pipe = build_model(random_state=args.random_state)
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, preds, average="macro", zero_division=0
    )

    labels = sorted(list(allowed_set))
    cm = confusion_matrix(y_test, preds, labels=labels)

    print("\nClassification model: RandomForestClassifier")
    print("Target:", TARGET)
    print("Features:", FEATURES)
    print(f"Rows after cleaning/filtering: {len(df):,}")
    print(f"Accuracy      : {acc:.4f}")
    print(f"Macro Precision: {prec:.4f}")
    print(f"Macro Recall   : {rec:.4f}")
    print(f"Macro F1       : {f1:.4f}")

    print("\nConfusion matrix (rows=true, cols=pred), labels =", labels)
    print(cm)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    meta = {
        "target": TARGET,
        "features": FEATURES,
        "allowed_payment_types": labels,
        "model": "RandomForestClassifier",
        "metrics": {
            "accuracy": float(acc),
            "macro_precision": float(prec),
            "macro_recall": float(rec),
            "macro_f1": float(f1),
        },
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm.tolist(),
        },
        "class_distribution": {str(int(k)): int(v) for k, v in class_counts.items()},
        "rows_after_cleaning": int(len(df)),
        "test_size": args.test_size,
        "random_state": args.random_state,
    }

    (outdir / "classification_model_metadata.json").write_text(
        json.dumps(meta, indent=2)
    )
    print(f"\nSaved metadata to: {outdir / 'classification_model_metadata.json'}")

    if args.save_model:
        model_path = outdir / "classification_model.joblib"
        joblib.dump(pipe, model_path)
        print(f"Saved model to: {model_path} (keep ignored by git)")


if __name__ == "__main__":
    main()
