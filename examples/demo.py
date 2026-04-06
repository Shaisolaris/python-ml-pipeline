"""
Demo: Run the full ML pipeline on synthetic data.
Run: python examples/demo.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pipeline.ingestion import generate_synthetic_data, DataConfig, validate_dataframe, split_data
from pipeline.features import detect_column_types, build_preprocessor, FeatureConfig
from pipeline.training import train_all_models
from pipeline.evaluation import evaluate_classifier

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

def main():
    print("🧪 ML Pipeline Demo")
    print("=" * 50)

    # Generate dataset
    print("\n📊 Generating synthetic dataset...")
    df = generate_synthetic_data(n_samples=1000, n_features=8, task="classification")
    print(f"   Shape: {df.shape}")

    # Validate and split
    config = DataConfig(target_column="target", test_size=0.2, val_size=0.1)
    df = validate_dataframe(df, config)
    data = split_data(df, config)
    print(f"   Train: {len(data.X_train)}, Val: {len(data.X_val)}, Test: {len(data.X_test)}")

    # Feature engineering
    print("\n🔧 Feature engineering...")
    numeric_cols, categorical_cols = detect_column_types(data.X_train)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols, FeatureConfig())
    X_train = preprocessor.fit_transform(data.X_train)
    X_test = preprocessor.transform(data.X_test)

    # Train all models
    print("\n🏋️ Training 4 models with GridSearchCV...")
    results = train_all_models(X_train, data.y_train.values, task="classification")

    # Evaluate
    print("\n📈 Results:")
    print(f"{'Model':<25} {'CV Score':>10} {'Test Acc':>10} {'Test F1':>10} {'AUC':>10}")
    print("-" * 65)
    for r in results:
        metrics = evaluate_classifier(r.model, X_test, data.y_test.values)
        print(f"{r.model_name:<25} {r.cv_score:>10.4f} {metrics.accuracy:>10.4f} {metrics.f1:>10.4f} {metrics.roc_auc or 0:>10.4f}")

    print(f"\n🏆 Best model: {results[0].model_name} (CV: {results[0].cv_score:.4f})")

if __name__ == "__main__":
    main()
