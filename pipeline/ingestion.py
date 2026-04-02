"""Data ingestion — load, validate, and split datasets."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for data loading and splitting."""

    target_column: str = "target"
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    drop_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    numeric_columns: list[str] = field(default_factory=list)


@dataclass
class DataSplit:
    """Container for train/val/test splits."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    feature_names: list[str]
    target_name: str


def load_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    """Load a CSV file with basic validation."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path, **kwargs)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns from {path.name}")
    return df


def generate_synthetic_data(
    n_samples: int = 1000,
    n_features: int = 10,
    task: str = "classification",
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate synthetic data for testing the pipeline."""
    rng = np.random.default_rng(random_state)

    data: dict[str, np.ndarray] = {}
    for i in range(n_features):
        if i < n_features // 3:
            data[f"feature_{i}"] = rng.normal(0, 1, n_samples)
        elif i < 2 * n_features // 3:
            data[f"feature_{i}"] = rng.uniform(0, 100, n_samples)
        else:
            data[f"feature_{i}"] = rng.choice(["A", "B", "C", "D"], n_samples)

    df = pd.DataFrame(data)

    if task == "classification":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        scores = df[numeric_cols].sum(axis=1)
        df["target"] = (scores > scores.median()).astype(int)
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        weights = rng.normal(0, 1, len(numeric_cols))
        df["target"] = df[numeric_cols].values @ weights + rng.normal(0, 0.5, n_samples)

    logger.info(f"Generated synthetic {task} dataset: {n_samples} samples, {n_features} features")
    return df


def validate_dataframe(df: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    """Validate and clean a DataFrame."""
    if config.target_column not in df.columns:
        raise ValueError(f"Target column '{config.target_column}' not found in data")

    if config.drop_columns:
        df = df.drop(columns=[c for c in config.drop_columns if c in df.columns])

    # Report missing values
    missing = df.isnull().sum()
    if missing.any():
        logger.warning(f"Missing values:\n{missing[missing > 0]}")

    # Report duplicates
    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        logger.warning(f"Found {n_dupes} duplicate rows")
        df = df.drop_duplicates()

    logger.info(f"Validated: {len(df)} rows, {len(df.columns)} columns")
    return df


def split_data(df: pd.DataFrame, config: DataConfig) -> DataSplit:
    """Split data into train/val/test sets."""
    X = df.drop(columns=[config.target_column])
    y = df[config.target_column]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state, stratify=y if y.nunique() < 20 else None,
    )

    val_ratio = config.val_size / (1 - config.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_ratio, random_state=config.random_state,
    )

    logger.info(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    return DataSplit(
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        feature_names=list(X.columns), target_name=config.target_column,
    )
