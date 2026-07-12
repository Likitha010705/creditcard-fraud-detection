"""
data_loader.py
---------------
Helpers for loading, validating, and preprocessing the credit card
transactions dataset (works with the real Kaggle file or the bundled
synthetic sample, since both share the same schema).
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

REQUIRED_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount", "Class"]

SAMPLE_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sample_creditcard.csv")


def load_sample_data() -> pd.DataFrame:
    """Load the bundled synthetic sample dataset."""
    return pd.read_csv(SAMPLE_DATA_PATH)


def load_uploaded_data(uploaded_file) -> pd.DataFrame:
    """Load a user-uploaded CSV (e.g. the real Kaggle creditcard.csv)."""
    return pd.read_csv(uploaded_file)


def validate_schema(df: pd.DataFrame) -> Tuple[bool, str]:
    """Check that a dataframe has the columns this app expects."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return False, f"Missing expected columns: {', '.join(missing)}"
    return True, "OK"


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows and rows with missing values in key columns."""
    df = df.drop_duplicates()
    df = df.dropna(subset=REQUIRED_COLUMNS)
    df["Class"] = df["Class"].astype(int)
    return df


def add_scaled_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add scaled versions of Time and Amount (scaled_time, scaled_amount).
    V1-V28 are already PCA-transformed/standardized in the original dataset,
    so only Time and Amount need scaling before modeling.
    """
    df = df.copy()
    scaler = StandardScaler()
    df[["scaled_time", "scaled_amount"]] = scaler.fit_transform(df[["Time", "Amount"]])
    return df


def get_feature_target_split(df: pd.DataFrame):
    """Return (X, y) using V1-V28 + scaled_time + scaled_amount as features."""
    df = add_scaled_features(df)
    feature_cols = [f"V{i}" for i in range(1, 29)] + ["scaled_time", "scaled_amount"]
    X = df[feature_cols]
    y = df["Class"]
    return X, y, feature_cols


def train_test_split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """Stratified train/test split -- stratification matters a lot here
    since fraud cases are extremely rare and we want both splits to
    contain a representative share of them."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def class_balance_summary(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["Class"].value_counts().rename({0: "Legitimate", 1: "Fraud"})
    pct = (df["Class"].value_counts(normalize=True) * 100).rename({0: "Legitimate", 1: "Fraud"})
    summary = pd.DataFrame({"Count": counts, "Percent": pct.round(4)})
    return summary
