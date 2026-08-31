"""Milestone self-tests. Each check asserts loudly and exits non-zero on failure.

Usage:
    python selftest.py m1        # data sanity
    python selftest.py m2        # baseline train + artifacts
    python selftest.py m3        # metrics + plot files
    python selftest.py m4        # accuracy / roc-auc thresholds
    python selftest.py m5        # streamlit app imports + builds figures
    python selftest.py all
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "model.pkl"
METRICS_PATH = ROOT / "metrics.json"
CM_PATH = ROOT / "confusion_matrix.png"
ROC_PATH = ROOT / "roc_curve.png"

ACC_THRESHOLD = 0.90
AUC_THRESHOLD = 0.95


def _load_metrics() -> dict:
    assert METRICS_PATH.exists(), f"metrics.json missing at {METRICS_PATH}"
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def m1() -> None:
    from data_utils import load_data, sanity_report, split_data

    ds = load_data()
    rep = sanity_report(ds)

    assert rep["shape"] == (569, 30), f"expected shape (569, 30), got {rep['shape']}"
    assert rep["missing_values"] == 0, f"found {rep['missing_values']} NaNs"
    assert rep["non_finite_values"] == 0, f"found {rep['non_finite_values']} non-finite values"
    assert rep["target_names"] == ["malignant", "benign"], rep["target_names"]
    assert rep["class_balance"] == {"malignant": 212, "benign": 357}, rep["class_balance"]
    assert set(ds.y.unique()) == {0, 1}, f"unexpected labels {set(ds.y.unique())}"

    X_train, X_test, y_train, y_test = split_data(ds)
    assert len(X_train) == 455 and len(X_test) == 114, (len(X_train), len(X_test))
    # stratification: class ratio preserved within 2 percentage points
    full_ratio = ds.y.mean()
    for name, part in (("train", y_train), ("test", y_test)):
        assert abs(part.mean() - full_ratio) < 0.02, f"{name} not stratified: {part.mean()} vs {full_ratio}"

    print("M1 OK: shape=(569, 30), 0 missing, 0 non-finite, balance=212/357, stratified 455/114 split")


def m2() -> None:
    import joblib
    from sklearn.ensemble import RandomForestClassifier

    assert MODEL_PATH.exists(), "model.pkl missing - run train.py first"
    model = joblib.load(MODEL_PATH)
    assert isinstance(model, RandomForestClassifier), f"expected RandomForestClassifier, got {type(model)}"
    assert model.n_estimators == 200, f"n_estimators should be 200, got {model.n_estimators}"
    assert model.random_state == 42, f"random_state should be 42, got {model.random_state}"
    assert len(model.estimators_) == 200, f"model not fitted with 200 trees: {len(model.estimators_)}"
    assert model.n_features_in_ == 30, f"expected 30 input features, got {model.n_features_in_}"
    assert hasattr(model, "predict_proba"), "model lacks predict_proba"

    m = _load_metrics()
    assert m["n_train"] == 455 and m["n_test"] == 114, (m["n_train"], m["n_test"])
    assert m["params"]["n_estimators"] == 200 and m["params"]["random_state"] == 42, m["params"]

    print(
        f"M2 OK: fitted RandomForestClassifier(n_estimators={model.n_estimators}, "
        f"random_state={model.random_state}), trees={len(model.estimators_)}, "
        f"train/test={m['n_train']}/{m['n_test']}"
    )


def m3() -> None:
    import numpy as np

    m = _load_metrics()

    required = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "confusion_matrix"]
    for key in required:
        assert key in m, f"metrics.json missing required key '{key}'"

    for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        val = m[key]
        assert isinstance(val, (int, float)), f"{key} is not numeric: {val!r}"
        assert 0.0 <= val <= 1.0, f"{key} out of range: {val}"

    cm = np.array(m["confusion_matrix"])
    assert cm.shape == (2, 2), f"confusion matrix shape {cm.shape}, expected (2, 2)"
    assert cm.sum() == m["n_test"] == 114, f"cm total {cm.sum()} != n_test {m['n_test']}"

    # accuracy must be internally consistent with the confusion matrix
    acc_from_cm = float(np.trace(cm) / cm.sum())
    assert abs(acc_from_cm - m["accuracy"]) < 1e-9, (
        f"accuracy {m['accuracy']} disagrees with confusion matrix ({acc_from_cm})"
    )

    # precision/recall/f1 must be internally consistent (positive class = benign = 1)
    tn, fp, fn, tp = cm.ravel()
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = 2 * prec * rec / (prec + rec)
    assert abs(prec - m["precision"]) < 1e-9, f"precision {m['precision']} != cm-derived {prec}"
    assert abs(rec - m["recall"]) < 1e-9, f"recall {m['recall']} != cm-derived {rec}"
    assert abs(f1 - m["f1_score"]) < 1e-9, f"f1 {m['f1_score']} != cm-derived {f1}"

    assert "classification_report" in m and "malignant" in m["classification_report"], (
        "classification_report missing or malformed"
    )

    for path in (CM_PATH, ROC_PATH):
        assert path.exists(), f"plot missing: {path.name}"
        size = path.stat().st_size
        assert size > 5_000, f"{path.name} suspiciously small ({size} bytes)"

    print(
        f"M3 OK: acc={m['accuracy']:.4f} prec={m['precision']:.4f} rec={m['recall']:.4f} "
        f"f1={m['f1_score']:.4f} auc={m['roc_auc']:.4f}; cm={cm.tolist()} consistent; "
        f"confusion_matrix.png={CM_PATH.stat().st_size}B roc_curve.png={ROC_PATH.stat().st_size}B"
    )


def m4() -> None:
    m = _load_metrics()
    acc, auc = m["accuracy"], m["roc_auc"]
    assert acc >= ACC_THRESHOLD, f"accuracy {acc:.4f} < required {ACC_THRESHOLD}"
    assert auc >= AUC_THRESHOLD, f"roc_auc {auc:.4f} < required {AUC_THRESHOLD}"
    print(f"M4 OK: accuracy {acc:.4f} >= {ACC_THRESHOLD} and roc_auc {auc:.4f} >= {AUC_THRESHOLD}")


def m5() -> None:
    """Exercise the app's logic layer without a browser: figures + prediction path."""
    import matplotlib

    matplotlib.use("Agg")

    import app as streamlit_app
    from data_utils import load_data
    from evaluate import plot_confusion_matrix, plot_roc_curve

    for name in ("main", "render_metrics", "render_prediction_form"):
        assert hasattr(streamlit_app, name), f"app.py missing {name}()"

    ds = load_data()
    bounds = ds.feature_bounds()
    assert list(bounds.index) == ds.feature_names, "feature bounds misaligned with feature names"
    assert (bounds["max"] > bounds["min"]).all(), "some feature has max <= min; sliders would break"

    m = _load_metrics()
    fig_cm = plot_confusion_matrix(m["confusion_matrix"], ds.target_names)
    fig_roc = plot_roc_curve(m["roc_curve"]["fpr"], m["roc_curve"]["tpr"], m["roc_auc"])
    assert fig_cm is not None and fig_roc is not None, "figure builders returned None"

    # live prediction path with mean feature values must return a valid class + probability
    import joblib
    import pandas as pd

    model = joblib.load(MODEL_PATH)
    row = pd.DataFrame([bounds["mean"].to_dict()])[ds.feature_names]
    pred = int(model.predict(row)[0])
    proba = model.predict_proba(row)[0]
    assert pred in (0, 1), f"bad prediction {pred}"
    assert abs(proba.sum() - 1.0) < 1e-9, f"probabilities do not sum to 1: {proba}"

    print(
        f"M5 OK: app.py imports, 30 slider bounds valid, both figures built, "
        f"mean-row prediction={ds.target_names[pred]} p={proba.max():.4f}"
    )


CHECKS = {"m1": m1, "m2": m2, "m3": m3, "m4": m4, "m5": m5}


if __name__ == "__main__":
    which = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    targets = list(CHECKS) if which == "all" else [which]
    for name in targets:
        if name not in CHECKS:
            raise SystemExit(f"unknown check '{name}'; choose from {list(CHECKS)} or 'all'")
        CHECKS[name]()
    print(f"\nALL REQUESTED SELF-TESTS PASSED: {targets}")
