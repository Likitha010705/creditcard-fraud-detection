"""
generate_sample_data.py
------------------------
Generates a small SYNTHETIC sample dataset that mirrors the schema and
statistical shape of the real Kaggle "Credit Card Fraud Detection" dataset
(https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

This is NOT the real dataset. The real dataset (284,807 transactions,
~150 MB) requires a free Kaggle account to download and is licensed for
use through Kaggle directly, so it is not redistributed here.

This script exists so the app has something to run against out of the box.
Once you download the real creditcard.csv from Kaggle, drop it into this
`data/` folder (or upload it through the app's sidebar) and the app will
use the real data automatically instead.

Usage:
    python generate_sample_data.py
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

N_ROWS = 6000
FRAUD_RATE = 0.0017  # matches the real dataset's ~0.172% fraud rate


def generate_sample(n_rows: int = N_ROWS, fraud_rate: float = FRAUD_RATE) -> pd.DataFrame:
    n_fraud = max(8, int(n_rows * fraud_rate))
    n_normal = n_rows - n_fraud

    # Time: seconds elapsed, dataset spans ~2 days (0 to 172792 seconds)
    time_normal = np.sort(RNG.uniform(0, 172792, n_normal))
    time_fraud = np.sort(RNG.uniform(0, 172792, n_fraud))

    # V1-V28: PCA components. In the real data these are roughly standardized
    # (mean ~0, std ~1) for normal transactions, with fraud cases showing
    # shifted means on several components -- that shift is what makes fraud
    # detectable, and we replicate that pattern here.
    n_components = 28
    normal_means = np.zeros(n_components)
    fraud_means = RNG.normal(0, 2.5, n_components)  # fraud has a different "signature"

    cov = np.eye(n_components)  # independent components, close enough for demo

    v_normal = RNG.multivariate_normal(normal_means, cov, n_normal)
    v_fraud = RNG.multivariate_normal(fraud_means, cov * 1.8, n_fraud)

    # Amount: real data is right-skewed, most transactions are small
    amount_normal = RNG.gamma(shape=1.2, scale=60, size=n_normal)
    amount_fraud = RNG.gamma(shape=1.0, scale=110, size=n_fraud)  # fraud skews slightly higher

    df_normal = pd.DataFrame(v_normal, columns=[f"V{i}" for i in range(1, n_components + 1)])
    df_normal.insert(0, "Time", time_normal)
    df_normal["Amount"] = np.round(amount_normal, 2)
    df_normal["Class"] = 0

    df_fraud = pd.DataFrame(v_fraud, columns=[f"V{i}" for i in range(1, n_components + 1)])
    df_fraud.insert(0, "Time", time_fraud)
    df_fraud["Amount"] = np.round(amount_fraud, 2)
    df_fraud["Class"] = 1

    df = pd.concat([df_normal, df_fraud], ignore_index=True)
    df = df.sort_values("Time").reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate_sample()
    out_path = "sample_creditcard.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows ({df['Class'].sum()} fraud) to {out_path}")
