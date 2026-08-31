"""Streamlit UI for the Random Forest breast cancer classifier.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import streamlit as st

from data_utils import load_data
from evaluate import plot_confusion_matrix, plot_roc_curve
from train import MODEL_PATH, load_or_train, train
from evaluate import METRICS_PATH

st.set_page_config(page_title="RF Breast Cancer Classifier", layout="wide")


@st.cache_data(show_spinner=False)
def get_dataset():
    return load_data()


def _artifacts_exist() -> bool:
    return MODEL_PATH.exists() and METRICS_PATH.exists()


def render_metrics(metrics: dict, target_names: list[str]) -> None:
    """Render the five headline metrics, the confusion matrix and the ROC curve."""
    st.subheader("Evaluation metrics (held-out test set)")
    st.caption(
        f"n_test = {metrics['n_test']} samples · positive class = "
        f"{metrics.get('positive_class', target_names[1])} · trained at "
        f"{metrics.get('trained_at', 'unknown')}"
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    c2.metric("Precision", f"{metrics['precision']:.4f}")
    c3.metric("Recall", f"{metrics['recall']:.4f}")
    c4.metric("F1-score", f"{metrics['f1_score']:.4f}")
    c5.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        st.pyplot(plot_confusion_matrix(metrics["confusion_matrix"], target_names))
    with right:
        st.markdown("**ROC curve**")
        st.pyplot(
            plot_roc_curve(
                metrics["roc_curve"]["fpr"], metrics["roc_curve"]["tpr"], metrics["roc_auc"]
            )
        )

    st.markdown("**Classification report**")
    st.code(metrics["classification_report"], language="text")

    if "per_class" in metrics:
        st.markdown("**Per-class breakdown**")
        st.dataframe(pd.DataFrame(metrics["per_class"]).T, width="stretch")

    with st.expander("Raw confusion matrix values"):
        cm_df = pd.DataFrame(
            metrics["confusion_matrix"],
            index=[f"true: {n}" for n in target_names],
            columns=[f"pred: {n}" for n in target_names],
        )
        st.dataframe(cm_df, width="stretch")


def render_feature_importances(metrics: dict, top_n: int = 10) -> None:
    importances = metrics.get("feature_importances")
    if not importances:
        return
    st.subheader(f"Top {top_n} most important features")
    series = pd.Series(importances).sort_values(ascending=False).head(top_n)
    st.bar_chart(series)


def render_prediction_form(model, dataset, metrics: dict) -> None:
    """Manual feature entry bounded by the real min/max of each feature."""
    st.subheader("Try a live prediction")
    st.caption(
        "Sliders are bounded by each feature's actual min and max in the dataset "
        "and default to the dataset mean. Adjust any values, then predict."
    )

    bounds = dataset.feature_bounds()

    preset = st.radio(
        "Start from",
        ["Dataset mean", "A real malignant sample", "A real benign sample"],
        horizontal=True,
    )
    if preset == "A real malignant sample":
        defaults = dataset.X[dataset.y == 0].iloc[0]
    elif preset == "A real benign sample":
        defaults = dataset.X[dataset.y == 1].iloc[0]
    else:
        defaults = bounds["mean"]

    with st.form("prediction_form"):
        values: dict[str, float] = {}
        cols = st.columns(3)
        for i, name in enumerate(dataset.feature_names):
            lo = float(bounds.loc[name, "min"])
            hi = float(bounds.loc[name, "max"])
            default = float(min(max(float(defaults[name]), lo), hi))
            step = (hi - lo) / 100.0 or 0.01
            values[name] = cols[i % 3].slider(
                name,
                min_value=lo,
                max_value=hi,
                value=default,
                step=step,
                key=f"slider_{preset}_{name}",
            )
        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        row = pd.DataFrame([values])[dataset.feature_names]
        pred = int(model.predict(row)[0])
        proba = model.predict_proba(row)[0]
        label = dataset.target_names[pred]

        if label == "malignant":
            st.error(f"Prediction: **{label.upper()}**")
        else:
            st.success(f"Prediction: **{label.upper()}**")

        pcols = st.columns(len(dataset.target_names))
        for col, cls_idx in zip(pcols, model.classes_):
            name = dataset.target_names[int(cls_idx)]
            col.metric(f"P({name})", f"{proba[list(model.classes_).index(cls_idx)]:.4f}")


def main() -> None:
    st.title("Random Forest — Breast Cancer Wisconsin Classifier")
    st.write(
        "Predicts **malignant** vs **benign** from 30 cell-nucleus measurements "
        "using a `RandomForestClassifier` (scikit-learn)."
    )

    dataset = get_dataset()

    with st.sidebar:
        st.header("Model")
        st.write(
            f"Dataset: **{dataset.n_samples}** samples, **{dataset.n_features}** features"
        )
        st.write(f"Class balance: `{dataset.class_balance()}`")
        st.divider()

        if _artifacts_exist():
            st.success("Pre-trained model found (`model.pkl`)")
        else:
            st.warning("No saved model yet — click Train to build one.")

        retrain = st.button("Train / retrain model", type="primary")
        st.caption(
            "Training refits RandomForestClassifier(n_estimators=200, random_state=42) "
            "and rewrites model.pkl, metrics.json and both plots."
        )

    if retrain:
        with st.spinner("Training Random Forest..."):
            model, metrics = train(dataset)
        st.session_state["model"] = model
        st.session_state["metrics"] = metrics
        st.success("Training complete — metrics below are freshly computed.")
    elif "model" in st.session_state:
        model, metrics = st.session_state["model"], st.session_state["metrics"]
    else:
        if not _artifacts_exist():
            st.info("No saved model found. Training one now...")
        with st.spinner("Loading model..."):
            model, metrics = load_or_train()
        st.session_state["model"] = model
        st.session_state["metrics"] = metrics

    with st.sidebar:
        st.divider()
        st.header("Sanity thresholds")
        st.write(
            f"Accuracy ≥ 0.90 → {'PASS' if metrics['accuracy'] >= 0.90 else 'FAIL'} "
            f"({metrics['accuracy']:.4f})"
        )
        st.write(
            f"ROC-AUC ≥ 0.95 → {'PASS' if metrics['roc_auc'] >= 0.95 else 'FAIL'} "
            f"({metrics['roc_auc']:.4f})"
        )

    render_metrics(metrics, dataset.target_names)
    st.divider()
    render_feature_importances(metrics)
    st.divider()
    render_prediction_form(model, dataset, metrics)


if __name__ == "__main__":
    main()
