"""Model training — train, tune, and persist ML models."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline as SkPipeline

logger = logging.getLogger(__name__)

CLASSIFIERS = {
    "logistic_regression": (LogisticRegression, {"C": [0.1, 1, 10], "max_iter": [200]}),
    "random_forest": (RandomForestClassifier, {"n_estimators": [100, 200], "max_depth": [10, 20, None], "min_samples_split": [2, 5]}),
    "gradient_boosting": (GradientBoostingClassifier, {"n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]}),
    "svm": (SVC, {"C": [0.1, 1, 10], "kernel": ["rbf"], "probability": [True]}),
}

REGRESSORS = {
    "ridge": (Ridge, {"alpha": [0.1, 1.0, 10.0]}),
    "random_forest": (RandomForestRegressor, {"n_estimators": [100, 200], "max_depth": [10, 20, None]}),
    "gradient_boosting": (GradientBoostingRegressor, {"n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]}),
    "svr": (SVR, {"C": [0.1, 1, 10], "kernel": ["rbf"]}),
}


@dataclass
class TrainConfig:
    """Configuration for model training."""

    task: str = "classification"
    model_name: str = "random_forest"
    hyperparameter_search: bool = True
    cv_folds: int = 5
    scoring: Optional[str] = None
    random_state: int = 42


@dataclass
class TrainResult:
    """Result of model training."""

    model: Any
    model_name: str
    best_params: dict[str, Any]
    cv_score: float
    cv_std: float


def get_model_registry(task: str) -> dict:
    """Get available models for the task."""
    return CLASSIFIERS if task == "classification" else REGRESSORS


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: TrainConfig,
    preprocessor: Optional[Any] = None,
) -> TrainResult:
    """Train a model with optional hyperparameter search."""
    registry = get_model_registry(config.task)

    if config.model_name not in registry:
        raise ValueError(f"Unknown model: {config.model_name}. Available: {list(registry.keys())}")

    model_class, param_grid = registry[config.model_name]
    scoring = config.scoring or ("accuracy" if config.task == "classification" else "r2")

    logger.info(f"Training {config.model_name} ({config.task})")

    if config.hyperparameter_search and param_grid:
        search = GridSearchCV(
            model_class(random_state=config.random_state) if "random_state" in model_class().get_params() else model_class(),
            param_grid,
            cv=config.cv_folds,
            scoring=scoring,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        best_params = search.best_params_
        cv_score = search.best_score_
        cv_std = search.cv_results_["std_test_score"][search.best_index_]
        logger.info(f"Best params: {best_params}, CV score: {cv_score:.4f} (+/- {cv_std:.4f})")
    else:
        model = model_class(random_state=config.random_state) if "random_state" in model_class().get_params() else model_class()
        model.fit(X_train, y_train)
        scores = cross_val_score(model, X_train, y_train, cv=config.cv_folds, scoring=scoring)
        best_params = model.get_params()
        cv_score = float(scores.mean())
        cv_std = float(scores.std())
        logger.info(f"CV score: {cv_score:.4f} (+/- {cv_std:.4f})")

    # Wrap with preprocessor if provided
    if preprocessor is not None:
        pipeline = SkPipeline([("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)
    else:
        pipeline = model

    return TrainResult(model=pipeline, model_name=config.model_name, best_params=best_params, cv_score=cv_score, cv_std=cv_std)


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task: str = "classification",
    cv_folds: int = 5,
) -> list[TrainResult]:
    """Train all available models and return sorted results."""
    registry = get_model_registry(task)
    results: list[TrainResult] = []

    for name in registry:
        try:
            config = TrainConfig(task=task, model_name=name, hyperparameter_search=True, cv_folds=cv_folds)
            result = train_model(X_train, y_train, config)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to train {name}: {e}")

    results.sort(key=lambda r: r.cv_score, reverse=True)
    logger.info(f"Trained {len(results)} models. Best: {results[0].model_name} ({results[0].cv_score:.4f})")
    return results


def save_model(model: Any, path: str | Path) -> Path:
    """Save a trained model to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")
    return path


def load_model(path: str | Path) -> Any:
    """Load a trained model from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    model = joblib.load(path)
    logger.info(f"Model loaded from {path}")
    return model
