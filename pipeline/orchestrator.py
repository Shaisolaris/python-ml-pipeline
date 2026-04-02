"""Pipeline orchestrator — end-to-end ML workflow."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pipeline.ingestion import DataConfig, generate_synthetic_data, validate_dataframe, split_data, load_csv
from pipeline.features import FeatureConfig, detect_column_types, build_preprocessor
from pipeline.training import TrainConfig, TrainResult, train_model, train_all_models, save_model
from pipeline.evaluation import evaluate_classifier, evaluate_regressor

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Full pipeline configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    training: TrainConfig = field(default_factory=TrainConfig)
    output_dir: str = "outputs"
    experiment_name: str = "experiment"


def run_pipeline(config: PipelineConfig, data_path: str | None = None) -> dict[str, Any]:
    """Run the full ML pipeline end-to-end."""
    logger.info(f"Starting pipeline: {config.experiment_name}")

    # 1. Data Ingestion
    if data_path:
        df = load_csv(data_path)
    else:
        df = generate_synthetic_data(n_samples=2000, task=config.training.task)

    df = validate_dataframe(df, config.data)
    data = split_data(df, config.data)

    # 2. Feature Engineering
    numeric_cols, categorical_cols = detect_column_types(data.X_train)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols, config.features)

    X_train_processed = preprocessor.fit_transform(data.X_train)
    X_val_processed = preprocessor.transform(data.X_val)
    X_test_processed = preprocessor.transform(data.X_test)

    # 3. Model Training
    result = train_model(X_train_processed, data.y_train.values, config.training)

    # 4. Evaluation
    if config.training.task == "classification":
        val_metrics = evaluate_classifier(result.model, X_val_processed, data.y_val.values)
        test_metrics = evaluate_classifier(result.model, X_test_processed, data.y_test.values)
        metrics_dict = {"accuracy": test_metrics.accuracy, "f1": test_metrics.f1, "roc_auc": test_metrics.roc_auc}
    else:
        val_metrics = evaluate_regressor(result.model, X_val_processed, data.y_val.values)
        test_metrics = evaluate_regressor(result.model, X_test_processed, data.y_test.values)
        metrics_dict = {"rmse": test_metrics.rmse, "mae": test_metrics.mae, "r2": test_metrics.r2}

    # 5. Save Model
    output_dir = Path(config.output_dir)
    model_path = save_model(result.model, output_dir / f"{config.experiment_name}_model.joblib")

    summary = {
        "experiment": config.experiment_name,
        "model": result.model_name,
        "best_params": result.best_params,
        "cv_score": result.cv_score,
        "test_metrics": metrics_dict,
        "model_path": str(model_path),
        "data_shape": {"train": len(data.X_train), "val": len(data.X_val), "test": len(data.X_test)},
        "features": data.feature_names,
    }

    logger.info(f"Pipeline complete. Test metrics: {metrics_dict}")
    return summary


def run_model_comparison(config: PipelineConfig, data_path: str | None = None) -> list[dict[str, Any]]:
    """Run all models and compare performance."""
    logger.info(f"Starting model comparison: {config.experiment_name}")

    if data_path:
        df = load_csv(data_path)
    else:
        df = generate_synthetic_data(n_samples=2000, task=config.training.task)

    df = validate_dataframe(df, config.data)
    data = split_data(df, config.data)

    numeric_cols, categorical_cols = detect_column_types(data.X_train)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols, config.features)
    X_train_processed = preprocessor.fit_transform(data.X_train)
    X_test_processed = preprocessor.transform(data.X_test)

    results = train_all_models(X_train_processed, data.y_train.values, task=config.training.task)

    comparison = []
    for result in results:
        if config.training.task == "classification":
            metrics = evaluate_classifier(result.model, X_test_processed, data.y_test.values)
            comparison.append({"model_name": result.model_name, "cv_score": result.cv_score, "accuracy": metrics.accuracy, "f1": metrics.f1, "roc_auc": metrics.roc_auc})
        else:
            metrics = evaluate_regressor(result.model, X_test_processed, data.y_test.values)
            comparison.append({"model_name": result.model_name, "cv_score": result.cv_score, "rmse": metrics.rmse, "r2": metrics.r2})

    logger.info(f"Comparison complete. {len(comparison)} models evaluated.")
    return comparison
