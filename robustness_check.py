"""Robustness check: is the reported single-split result a real result or a lucky draw?

The spec fixes random_state=42 for both the split and the forest, which makes the
headline numbers reproducible but says nothing about whether they generalise. This
script answers that separately, WITHOUT touching the shipped model or metrics.json:

  1. Stratified 5-fold cross-validation over all 569 samples.
  2. The same evaluation repeated across 10 different split seeds.
  3. Per-class recall across folds, since malignant recall is the number that matters.

Read-only with respect to model.pkl / metrics.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from data_utils import load_data
from train import N_ESTIMATORS, build_model

ROOT = Path(__file__).parent
ACC_THRESHOLD = 0.90
AUC_THRESHOLD = 0.95


def main() -> None:
    ds = load_data()
    X, y = ds.X, ds.y

    # --- 1. stratified 5-fold CV -------------------------------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc_cv = cross_val_score(build_model(), X, y, cv=cv, scoring="accuracy")
    auc_cv = cross_val_score(build_model(), X, y, cv=cv, scoring="roc_auc")

    print(f"[1/3] Stratified 5-fold CV (n_estimators={N_ESTIMATORS}, all 569 samples)")
    print(f"      accuracy per fold : {np.round(acc_cv, 4).tolist()}")
    print(f"      accuracy          : {acc_cv.mean():.4f} +/- {acc_cv.std():.4f}")
    print(f"      roc_auc per fold  : {np.round(auc_cv, 4).tolist()}")
    print(f"      roc_auc           : {auc_cv.mean():.4f} +/- {auc_cv.std():.4f}")

    assert acc_cv.mean() >= ACC_THRESHOLD, f"CV accuracy {acc_cv.mean():.4f} below {ACC_THRESHOLD}"
    assert auc_cv.mean() >= AUC_THRESHOLD, f"CV roc_auc {auc_cv.mean():.4f} below {AUC_THRESHOLD}"
    assert acc_cv.min() >= ACC_THRESHOLD, (
        f"worst fold accuracy {acc_cv.min():.4f} below {ACC_THRESHOLD} - result is split-dependent"
    )

    # --- 2. same protocol, 10 different split seeds ------------------------------
    accs, aucs, mal_recalls, ben_recalls = [], [], [], []
    for seed in range(10):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=seed
        )
        model = build_model()
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        pos_col = list(model.classes_).index(1)
        y_proba = model.predict_proba(X_te)[:, pos_col]

        accs.append(accuracy_score(y_te, y_pred))
        aucs.append(roc_auc_score(y_te, y_proba))
        mal_recalls.append(recall_score(y_te, y_pred, pos_label=0))
        ben_recalls.append(recall_score(y_te, y_pred, pos_label=1))

    accs, aucs = np.array(accs), np.array(aucs)
    mal_recalls, ben_recalls = np.array(mal_recalls), np.array(ben_recalls)

    print("\n[2/3] Same 80/20 stratified protocol across 10 split seeds (0-9)")
    print(f"      accuracy : mean {accs.mean():.4f}  min {accs.min():.4f}  max {accs.max():.4f}")
    print(f"      roc_auc  : mean {aucs.mean():.4f}  min {aucs.min():.4f}  max {aucs.max():.4f}")

    assert accs.min() >= ACC_THRESHOLD, (
        f"seed {int(accs.argmin())} gave accuracy {accs.min():.4f}, below {ACC_THRESHOLD}"
    )
    assert aucs.min() >= AUC_THRESHOLD, (
        f"seed {int(aucs.argmin())} gave roc_auc {aucs.min():.4f}, below {AUC_THRESHOLD}"
    )

    # --- 3. is the shipped seed-42 result representative or cherry-picked? --------
    shipped = json.loads((ROOT / "metrics.json").read_text(encoding="utf-8"))
    acc_pct = float((accs < shipped["accuracy"]).mean() * 100)

    print("\n[3/3] Per-class recall across the 10 seeds (malignant is the clinically "
          "important one)")
    print(f"      malignant recall : mean {mal_recalls.mean():.4f}  "
          f"min {mal_recalls.min():.4f}  max {mal_recalls.max():.4f}")
    print(f"      benign    recall : mean {ben_recalls.mean():.4f}  "
          f"min {ben_recalls.min():.4f}  max {ben_recalls.max():.4f}")
    print(f"\n      shipped (seed 42) accuracy {shipped['accuracy']:.4f} sits at the "
          f"{acc_pct:.0f}th percentile of the 10 seeds")
    print(f"      shipped malignant recall   {shipped['per_class']['malignant']['recall']:.4f} "
          f"vs {mal_recalls.mean():.4f} mean across seeds")

    verdict = "representative, not cherry-picked" if acc_pct <= 90 else (
        "OPTIMISTIC - shipped seed is in the top decile"
    )
    print(f"\nROBUSTNESS OK: result holds across 5-fold CV and 10 split seeds; "
          f"shipped seed is {verdict}")


if __name__ == "__main__":
    main()
