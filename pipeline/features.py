"""Feature engineering — transformations, encoding, scaling, feature selection."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, f_regression

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""

    numeric_strategy: str = "median"  # mean, median, most_frequent
    categorical_strategy: str = "most_frequent"
    scale_numeric: bool = True
    encode_categorical: bool = True
    select_k_best: Optional[int] = None
    task: str = "classification"  # classification or regression
    custom_features: list[str] = field(default_factory=list)


def detect_column_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Auto-detect numeric and categorical columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    logger.info(f"Detected {len(numeric_cols)} numeric, {len(categorical_cols)} categorical columns")
    return numeric_cols, categorical_cols


def build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
    config: FeatureConfig,
) -> ColumnTransformer:
    """Build a scikit-learn ColumnTransformer for preprocessing."""
    transformers: list[tuple] = []

    if numeric_cols:
        numeric_steps: list[tuple] = [("imputer", SimpleImputer(strategy=config.numeric_strategy))]
        if config.scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", SkPipeline(numeric_steps), numeric_cols))

    if categorical_cols and config.encode_categorical:
        cat_steps: list[tuple] = [
            ("imputer", SimpleImputer(strategy=config.categorical_strategy)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
        transformers.append(("categorical", SkPipeline(cat_steps), categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    logger.info(f"Built preprocessor: {len(transformers)} transformer(s)")
    return preprocessor


def add_interaction_features(df: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Add interaction (product) features for specified column pairs."""
    df = df.copy()
    for col_a, col_b in pairs:
        if col_a in df.columns and col_b in df.columns:
            name = f"{col_a}_x_{col_b}"
            df[name] = df[col_a] * df[col_b]
            logger.info(f"Added interaction feature: {name}")
    return df


def add_polynomial_features(df: pd.DataFrame, columns: list[str], degree: int = 2) -> pd.DataFrame:
    """Add polynomial features for specified columns."""
    df = df.copy()
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            for d in range(2, degree + 1):
                name = f"{col}_pow{d}"
                df[name] = df[col] ** d
                logger.info(f"Added polynomial feature: {name}")
    return df


def add_binned_features(df: pd.DataFrame, columns: list[str], n_bins: int = 5) -> pd.DataFrame:
    """Add binned (discretized) versions of numeric columns."""
    df = df.copy()
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            name = f"{col}_binned"
            df[name] = pd.qcut(df[col], q=n_bins, labels=False, duplicates="drop")
            logger.info(f"Added binned feature: {name} ({n_bins} bins)")
    return df


def select_features(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    task: str = "classification",
    feature_names: Optional[list[str]] = None,
) -> tuple[np.ndarray, list[str], SelectKBest]:
    """Select top-k features using univariate statistics."""
    score_func = f_classif if task == "classification" else f_regression
    selector = SelectKBest(score_func=score_func, k=min(k, X.shape[1]))
    X_selected = selector.fit_transform(X, y)

    if feature_names:
        mask = selector.get_support()
        selected_names = [n for n, m in zip(feature_names, mask) if m]
        logger.info(f"Selected {len(selected_names)} features: {selected_names[:10]}...")
    else:
        selected_names = [f"feature_{i}" for i in range(X_selected.shape[1])]

    return X_selected, selected_names, selector
