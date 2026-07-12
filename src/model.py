"""
model.py
--------
Model training, imbalance handling, and evaluation utilities for the
credit card fraud classifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

MODEL_REGISTRY = {
    "Logistic Regression": lambda **kw: LogisticRegression(max_iter=1000, **kw),
    "Random Forest": lambda **kw: RandomForestClassifier(
        n_estimators=200, max_depth=12, n_jobs=-1, random_state=42, **kw
    ),
    "Gradient Boosting": lambda **kw: GradientBoostingClassifier(random_state=42, **kw),
}


@dataclass
class TrainResult:
    model: object
    feature_cols: list
    metrics: dict
    y_test: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray
    confusion: np.ndarray
    fpr: np.ndarray
    tpr: np.ndarray
    precision_curve: np.ndarray
    recall_curve: np.ndarray


def resample_training_data(X_train, y_train, strategy: str):
    """
    Apply an imbalance-handling strategy to the TRAINING split only.
    (Never resample the test set -- it must reflect real-world class
    proportions so evaluation metrics stay meaningful.)
    """
    if strategy == "None":
        return X_train, y_train

    if strategy == "SMOTE (oversample minority)":
        from imblearn.over_sampling import SMOTE

        sm = SMOTE(random_state=42)
        return sm.fit_resample(X_train, y_train)

    if strategy == "Random undersampling":
        from imblearn.under_sampling import RandomUnderSampler

        rus = RandomUnderSampler(random_state=42)
        return rus.fit_resample(X_train, y_train)

    raise ValueError(f"Unknown resampling strategy: {strategy}")


def build_model(algo_name: str, use_class_weight: bool):
    kwargs = {}
    if use_class_weight and algo_name in ("Logistic Regression", "Random Forest"):
        kwargs["class_weight"] = "balanced"
    return MODEL_REGISTRY[algo_name](**kwargs)


def train_and_evaluate(
    X_train, X_test, y_train, y_test,
    algo_name: str,
    resample_strategy: str,
    use_class_weight: bool,
    feature_cols: list,
) -> TrainResult:
    X_train_final, y_train_final = resample_training_data(X_train, y_train, resample_strategy)

    model = build_model(algo_name, use_class_weight)
    model.fit(X_train_final, y_train_final)

    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else float("nan"),
        "PR-AUC (Avg Precision)": average_precision_score(y_test, y_proba)
        if len(np.unique(y_test)) > 1
        else float("nan"),
    }

    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba) if len(np.unique(y_test)) > 1 else (np.array([0, 1]), np.array([0, 1]), None)
    prec_curve, rec_curve, _ = (
        precision_recall_curve(y_test, y_proba)
        if len(np.unique(y_test)) > 1
        else (np.array([1, 0]), np.array([0, 1]), None)
    )

    return TrainResult(
        model=model,
        feature_cols=feature_cols,
        metrics=metrics,
        y_test=np.asarray(y_test),
        y_pred=y_pred,
        y_proba=y_proba,
        confusion=cm,
        fpr=fpr,
        tpr=tpr,
        precision_curve=prec_curve,
        recall_curve=rec_curve,
    )


def get_feature_importance(model, feature_cols: list) -> Optional[pd.DataFrame]:
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        label = "Importance"
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_[0])
        label = "|Coefficient|"
    else:
        return None
    df = pd.DataFrame({"Feature": feature_cols, label: imp})
    return df.sort_values(label, ascending=False).reset_index(drop=True)


def save_model(model, path: str):
    joblib.dump(model, path)


def load_model(path: str):
    return joblib.load(path)
