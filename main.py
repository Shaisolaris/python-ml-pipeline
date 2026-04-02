"""Main entry point — run the ML pipeline or start the API server."""

import logging
import sys

from pipeline.orchestrator import PipelineConfig, run_pipeline, run_model_comparison
from pipeline.ingestion import DataConfig
from pipeline.features import FeatureConfig
from pipeline.training import TrainConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"

    if mode == "train":
        config = PipelineConfig(
            data=DataConfig(target_column="target"),
            features=FeatureConfig(task="classification"),
            training=TrainConfig(task="classification", model_name="random_forest"),
            experiment_name="demo_classification",
        )
        summary = run_pipeline(config)
        print(f"\nPipeline Summary:\n{'-' * 40}")
        for key, value in summary.items():
            if key != "features":
                print(f"  {key}: {value}")

    elif mode == "compare":
        config = PipelineConfig(
            data=DataConfig(target_column="target"),
            features=FeatureConfig(task="classification"),
            training=TrainConfig(task="classification"),
            experiment_name="model_comparison",
        )
        results = run_model_comparison(config)
        print(f"\nModel Comparison:\n{'-' * 60}")
        for r in results:
            print(f"  {r['model_name']:25s} | F1: {r.get('f1', 'N/A'):>7} | AUC: {r.get('roc_auc', 'N/A'):>7}")

    elif mode == "serve":
        import uvicorn
        from api.serving import app, load_serving_model

        model_path = sys.argv[2] if len(sys.argv) > 2 else "outputs/demo_classification_model.joblib"
        load_serving_model(model_path, "demo_model")
        uvicorn.run(app, host="0.0.0.0", port=8000)

    else:
        print(f"Usage: python main.py [train|compare|serve]")
        sys.exit(1)


if __name__ == "__main__":
    main()
