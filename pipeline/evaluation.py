"""Model evaluation — metrics, reports, and comparison."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
)

logger = logging.getLogger(__name__)


@dataclass
class ClassificationMetrics:
    """Evaluation metrics for classification."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    confusion_matrix: list[list[int]]
    report: str


@dataclass
class RegressionMetrics:
    """Evaluation metrics for regression."""

    mse: float
    rmse: float
    mae: float
    r2: float
    mape: float | None


def evaluate_classifier(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> ClassificationMetrics:
    """Evaluate a classification model."""
    y_pred = model.predict(X_test)

    try:
        y_proba = model.predict_proba(X_test)
        if y_proba.shape[1] == 2:
            auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
    except (AttributeError, ValueError):
        auc = None

    metrics = ClassificationMetrics(
        accuracy=float(accuracy_score(y_test, y_pred)),
        precision=float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        recall=float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        f1=float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        roc_auc=float(auc) if auc is not None else None,
        confusion_matrix=confusion_matrix(y_test, y_pred).tolist(),
        report=classification_report(y_test, y_pred, zero_division=0),
    )

    logger.info(f"Classification: accuracy={metrics.accuracy:.4f}, f1={metrics.f1:.4f}, auc={metrics.roc_auc}")
    return metrics


def evaluate_regressor(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> RegressionMetrics:
    """Evaluate a regression model."""
    y_pred = model.predict(X_test)

    mse = float(mean_squared_error(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    # MAPE (avoid division by zero)
    nonzero_mask = y_test != 0
    if nonzero_mask.any():
        mape = float(np.mean(np.abs((y_test[nonzero_mask] - y_pred[nonzero_mask]) / y_test[nonzero_mask])) * 100)
    else:
        mape = None

    metrics = RegressionMetrics(mse=mse, rmse=float(np.sqrt(mse)), mae=mae, r2=r2, mape=mape)
    logger.info(f"Regression: RMSE={metrics.rmse:.4f}, MAE={metrics.mae:.4f}, R2={metrics.r2:.4f}")
    return metrics


def compare_models(
    results: list[dict[str, Any]],
    metric: str = "f1",
) -> list[dict[str, Any]]:
    """Compare multiple model results by a metric."""
    sorted_results = sorted(results, key=lambda r: r.get(metric, 0), reverse=True)
    for i, r in enumerate(sorted_results, 1):
        logger.info(f"#{i} {r['model_name']}: {metric}={r.get(metric, 'N/A')}")
    return sorted_results
