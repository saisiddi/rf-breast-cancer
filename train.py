"""Train the Random Forest, save model.pkl, metrics.json and the two plots."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier

from data_utils import RANDOM_STATE, TEST_SIZE, Dataset, load_data, split_data
from evaluate import METRICS_PATH, compute_metrics, print_metrics, save_plots

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "model.pkl"

# Baseline hyperparameters fixed by the project spec.
N_ESTIMATORS = 200


def build_model() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)


def train(dataset: Dataset | None = None, save: bool = True) -> tuple[RandomForestClassifier, dict]:
    """Fit the model on the stratified train split and evaluate on the test split.

    Returns the fitted model and the metrics dict. When ``save`` is True the
    model, metrics.json and both PNG plots are written to disk.
    """
    dataset = dataset or load_data()
    X_train, X_test, y_train, y_test = split_data(dataset)

    model = build_model()
    model.fit(X_train, y_train)

    metrics = compute_metrics(model, X_test, y_test, dataset.target_names)
    metrics.update(
        {
            "model": "RandomForestClassifier",
            "params": {
                "n_estimators": N_ESTIMATORS,
                "random_state": RANDOM_STATE,
                "test_size": TEST_SIZE,
                "stratify": True,
            },
            "n_train": int(len(X_train)),
            "n_features": dataset.n_features,
            "feature_names": dataset.feature_names,
            "target_names": dataset.target_names,
            "feature_importances": {
                name: float(imp)
                for name, imp in zip(dataset.feature_names, model.feature_importances_)
            },
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    )

    if save:
        joblib.dump(model, MODEL_PATH)
        METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        save_plots(metrics, dataset.target_names)

    return model, metrics


def load_or_train() -> tuple[RandomForestClassifier, dict]:
    """Load model.pkl + metrics.json if both exist, otherwise train from scratch."""
    if MODEL_PATH.exists() and METRICS_PATH.exists():
        model = joblib.load(MODEL_PATH)
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        return model, metrics
    return train()


if __name__ == "__main__":
    dataset = load_data()
    print(
        f"Training RandomForestClassifier(n_estimators={N_ESTIMATORS}, "
        f"random_state={RANDOM_STATE}) on {dataset.n_samples} samples / "
        f"{dataset.n_features} features..."
    )
    model, metrics = train(dataset)
    print(f"Saved model -> {MODEL_PATH.name}\n")
    print_metrics(metrics)
    print(f"\nSaved {METRICS_PATH.name}, confusion_matrix.png, roc_curve.png")
