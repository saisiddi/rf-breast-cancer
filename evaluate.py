"""Metric computation and plotting for the Random Forest breast cancer classifier.

Positive class convention
-------------------------
`sklearn.datasets.load_breast_cancer` encodes target_names as
``['malignant', 'benign']``, i.e. class 0 = malignant, class 1 = benign.
The headline precision / recall / F1 / ROC-AUC use class 1 (benign) as the
positive class, which is scikit-learn's default. Because the clinically
important number is how many malignant cases we catch, per-class figures for
BOTH classes are also stored under ``per_class`` and in the
``classification_report``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; safe for scripts and Streamlit

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).parent
METRICS_PATH = ROOT / "metrics.json"
CM_PLOT_PATH = ROOT / "confusion_matrix.png"
ROC_PLOT_PATH = ROOT / "roc_curve.png"

POSITIVE_CLASS = 1  # benign


def compute_metrics(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_names: Sequence[str],
) -> dict:
    """Compute every metric required by the spec from a fitted model."""
    y_pred = model.predict(X_test)
    # column for the positive class, looked up rather than assumed
    pos_col = list(model.classes_).index(POSITIVE_CLASS)
    y_proba = model.predict_proba(X_test)[:, pos_col]

    labels = [0, 1]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fpr, tpr, _ = roc_curve(y_test, y_proba, pos_label=POSITIVE_CLASS)

    report_dict = classification_report(
        y_test, y_pred, labels=labels, target_names=list(target_names), output_dict=True
    )
    report_text = classification_report(
        y_test, y_pred, labels=labels, target_names=list(target_names), digits=4
    )

    per_class = {
        name: {
            "precision": float(precision_score(y_test, y_pred, pos_label=cls, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, pos_label=cls, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, pos_label=cls, zero_division=0)),
            "support": int((np.asarray(y_test) == cls).sum()),
        }
        for cls, name in zip(labels, target_names)
    }

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, pos_label=POSITIVE_CLASS, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, pos_label=POSITIVE_CLASS, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, pos_label=POSITIVE_CLASS, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": list(target_names),
        "positive_class": str(target_names[POSITIVE_CLASS]),
        "per_class": per_class,
        "classification_report": report_text,
        "classification_report_dict": report_dict,
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "n_test": int(len(y_test)),
    }


def plot_confusion_matrix(cm, target_names: Sequence[str]):
    """Seaborn heatmap of the confusion matrix. Returns the matplotlib Figure."""
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        square=True,
        xticklabels=list(target_names),
        yticklabels=list(target_names),
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix (test set)")
    fig.tight_layout()
    return fig


def plot_roc_curve(fpr: Sequence[float], tpr: Sequence[float], roc_auc: float):
    """ROC curve with the chance diagonal. Returns the matplotlib Figure."""
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"Random Forest (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--", label="Chance (AUC = 0.5)")
    ax.set_xlim(-0.01, 1.0)
    ax.set_ylim(0.0, 1.01)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (positive class: benign)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def save_plots(metrics: dict, target_names: Sequence[str]) -> tuple[Path, Path]:
    """Write confusion_matrix.png and roc_curve.png next to this file."""
    fig_cm = plot_confusion_matrix(metrics["confusion_matrix"], target_names)
    fig_cm.savefig(CM_PLOT_PATH, dpi=150)
    plt.close(fig_cm)

    fig_roc = plot_roc_curve(
        metrics["roc_curve"]["fpr"], metrics["roc_curve"]["tpr"], metrics["roc_auc"]
    )
    fig_roc.savefig(ROC_PLOT_PATH, dpi=150)
    plt.close(fig_roc)

    return CM_PLOT_PATH, ROC_PLOT_PATH


def load_metrics(path: Path = METRICS_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def print_metrics(metrics: dict) -> None:
    print("=== Evaluation on held-out test set "
          f"(n={metrics['n_test']}, positive class = {metrics['positive_class']}) ===")
    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1-score  : {metrics['f1_score']:.4f}")
    print(f"ROC-AUC   : {metrics['roc_auc']:.4f}")
    print("\nConfusion matrix (rows = true, cols = predicted, order "
          f"{metrics['confusion_matrix_labels']}):")
    for row in metrics["confusion_matrix"]:
        print(f"  {row}")
    print("\nClassification report:")
    print(metrics["classification_report"])


if __name__ == "__main__":
    # Standalone evaluation of the saved model. Recomputes metrics and plots.
    import joblib

    from data_utils import load_data, split_data
    from train import MODEL_PATH

    if not MODEL_PATH.exists():
        raise SystemExit("model.pkl not found. Run `python train.py` first.")

    dataset = load_data()
    _, X_test, _, y_test = split_data(dataset)
    model = joblib.load(MODEL_PATH)

    metrics = compute_metrics(model, X_test, y_test, dataset.target_names)
    print_metrics(metrics)

    cm_path, roc_path = save_plots(metrics, dataset.target_names)
    print(f"\nSaved {cm_path.name} and {roc_path.name}")
