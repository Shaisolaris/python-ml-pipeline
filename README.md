# python-ml-pipeline

Production ML pipeline with data ingestion, feature engineering, model training with hyperparameter search, evaluation with classification/regression metrics, model comparison, and serving via FastAPI. Supports synthetic data generation, CSV loading, multiple scikit-learn models, and batch prediction.

## Stack

- **ML:** scikit-learn (RandomForest, GradientBoosting, Logistic Regression, SVM, Ridge)
- **Data:** pandas, numpy
- **API:** FastAPI + uvicorn
- **Serialization:** joblib

## Pipeline Stages

```
Data Ingestion → Feature Engineering → Model Training → Evaluation → Serving
     │                  │                    │               │           │
  CSV/synthetic    auto-detect types     GridSearchCV    accuracy    FastAPI
  validate         impute + scale       cross-validate   f1/AUC     /predict
  split            encode categoricals  save model       RMSE/R2    /predict/batch
```

## Usage

```bash
# Train a model
python main.py train

# Compare all models
python main.py compare

# Serve model via API
python main.py serve [model_path]
```

## Pipeline Modules

### `pipeline/ingestion.py`
- Load CSV or generate synthetic classification/regression data
- Auto-validate: missing values, duplicates, target column presence
- Stratified train/val/test split with configurable ratios
- DataConfig for target column, drop columns, split sizes

### `pipeline/features.py`
- Auto-detect numeric and categorical columns
- ColumnTransformer with SimpleImputer + StandardScaler (numeric) and OneHotEncoder (categorical)
- Interaction features (product of column pairs)
- Polynomial features (configurable degree)
- Binned features (quantile-based discretization)
- SelectKBest feature selection (f_classif / f_regression)

### `pipeline/training.py`
- 4 classifiers: Logistic Regression, Random Forest, Gradient Boosting, SVM
- 4 regressors: Ridge, Random Forest, Gradient Boosting, SVR
- GridSearchCV with configurable parameter grids
- Cross-validation scoring
- `train_all_models()` for automatic comparison
- Model persistence via joblib

### `pipeline/evaluation.py`
- Classification: accuracy, precision, recall, F1, ROC-AUC, confusion matrix, report
- Regression: MSE, RMSE, MAE, R2, MAPE
- Model comparison by any metric

### `pipeline/orchestrator.py`
- `run_pipeline()`: end-to-end single model training
- `run_model_comparison()`: train all models, evaluate, and rank

### `api/serving.py`
- `GET /health` — API + model status
- `POST /predict` — Single prediction with probability output
- `POST /predict/batch` — Batch predictions (up to 1000)
- `GET /model/info` — Model parameters and feature importances
- Pydantic input/output validation

## Architecture

```
python-ml-pipeline/
├── main.py                    # CLI: train, compare, serve
├── pipeline/
│   ├── ingestion.py           # Data loading, validation, splitting
│   ├── features.py            # Preprocessing, encoding, feature selection
│   ├── training.py            # Model registry, training, GridSearchCV, persistence
│   ├── evaluation.py          # Classification and regression metrics
│   └── orchestrator.py        # End-to-end pipeline runner
├── api/
│   └── serving.py             # FastAPI prediction endpoints
├── requirements.txt
└── pyproject.toml
```

## Setup

```bash
git clone https://github.com/Shaisolaris/python-ml-pipeline.git
cd python-ml-pipeline
pip install -r requirements.txt

# Run training
python main.py train

# Start API
python main.py serve
# → http://localhost:8000/docs
```

## Key Design Decisions

**Dataclass configs everywhere.** DataConfig, FeatureConfig, TrainConfig, PipelineConfig are all dataclasses with defaults. This makes the pipeline fully configurable without modifying code. The orchestrator composes them into a single PipelineConfig.

**Model registry pattern.** Available models and their hyperparameter grids are stored in dictionaries (CLASSIFIERS, REGRESSORS). Adding a new model means adding one dict entry. `train_all_models()` iterates the registry automatically.

**Preprocessor as ColumnTransformer.** The feature pipeline uses scikit-learn's ColumnTransformer with separate pipelines for numeric and categorical columns. This ensures consistent transformation between training and serving.

**Synthetic data for testing.** `generate_synthetic_data()` creates realistic datasets without external dependencies. This enables pipeline testing and demos without requiring real data files.

**FastAPI serving with global model.** The serving module loads the model into a global reference at startup. This avoids re-loading on every request while keeping the API stateless from the client's perspective.

## License

MIT
