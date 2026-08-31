# Random Forest — Breast Cancer Wisconsin Classifier

A Random Forest classifier that predicts **malignant** vs **benign** from the 30
cell-nucleus measurements in the Breast Cancer Wisconsin dataset
(`sklearn.datasets.load_breast_cancer`), with a Streamlit UI for metrics and live
predictions.

## Setup

Developed and tested on **Python 3.13.7**. **Python 3.14 will not work** —
`pyarrow`, a hard Streamlit dependency, has no cp314 wheel yet and its source
build fails without CMake. Earlier 3.x versions back to 3.11 should be fine
(all pinned packages publish wheels for them) but were not tested here.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On macOS / Linux use `source .venv/bin/activate` instead.

## Run

One command — the app trains on first launch if no saved model exists:

```powershell
streamlit run app.py
```

Then open http://localhost:8501.

To train from the command line instead (writes `model.pkl`, `metrics.json`,
`confusion_matrix.png`, `roc_curve.png`):

```powershell
python train.py          # train + evaluate + save everything
python evaluate.py       # re-evaluate the saved model, regenerate both plots
python data_utils.py     # print dataset shape, class balance, missing-value check
python selftest.py all   # run every milestone assertion
python e2e_test.py       # drive the UI headless and verify on-screen numbers
python clean_run_test.py # fresh-clone path (delete the 4 artifacts first)
python robustness_check.py  # 5-fold CV + 10 split seeds; confirms the result generalises
```

## What each file does

| File | Purpose |
| --- | --- |
| `data_utils.py` | Loads the dataset into a DataFrame, does the stratified train/test split, exposes per-feature min/max/mean used to bound the UI sliders, and reports missing/non-finite values. |
| `train.py` | Fits `RandomForestClassifier(n_estimators=200, random_state=42)`, saves `model.pkl`, writes `metrics.json`, renders both plots. Also provides `load_or_train()` used by the app. |
| `evaluate.py` | Computes every metric (accuracy, confusion matrix, precision, recall, F1, `classification_report`, ROC-AUC from `predict_proba`) and builds the confusion-matrix heatmap and ROC curve figures. Runnable standalone against the saved model. |
| `app.py` | Streamlit UI. |
| `selftest.py` | Per-milestone assertions (M1–M5). Fails loudly and exits non-zero. |
| `e2e_test.py` | End-to-end UI test via Streamlit's `AppTest`: asserts the rendered numbers equal `metrics.json` and that the prediction form classifies real samples correctly. |
| `clean_run_test.py` | Verifies the fresh-clone path: with no artifacts on disk, launching the app trains, writes all four artifacts, and reproduces the documented metrics. |
| `robustness_check.py` | Read-only generalisation check: stratified 5-fold CV plus the same protocol across 10 split seeds, so the headline numbers can be shown not to be a lucky split. |
| `requirements.txt` | Pinned dependencies. |
| `PROGRESS_LOG.md` | Milestone-by-milestone build log with the self-test output for each. |
| `model.pkl`, `metrics.json`, `confusion_matrix.png`, `roc_curve.png` | Generated artifacts. |

## Model configuration

- `RandomForestClassifier(n_estimators=200, random_state=42)`
- `train_test_split(test_size=0.2, stratify=y, random_state=42)` → 455 train / 114 test
- 569 samples, 30 features, class balance 212 malignant / 357 benign, zero missing values

**Positive class:** scikit-learn encodes this dataset as class 0 = malignant,
class 1 = benign. The headline precision / recall / F1 / ROC-AUC therefore use
**benign** as the positive class, which is scikit-learn's default. Because the
clinically important figure is how many malignant cases are caught, per-class
numbers for both classes are stored in `metrics.json` under `per_class` and shown
in the UI's per-class table.

## Sample metrics

From the committed `metrics.json` (test set, n = 114):

| Metric | Value |
| --- | --- |
| Accuracy | 0.9561 |
| Precision (benign) | 0.9589 |
| Recall (benign) | 0.9722 |
| F1-score (benign) | 0.9655 |
| ROC-AUC | 0.9931 |

Confusion matrix (rows = true, columns = predicted):

|  | pred: malignant | pred: benign |
| --- | --- | --- |
| **true: malignant** | 39 | 3 |
| **true: benign** | 2 | 70 |

Classification report:

```
              precision    recall  f1-score   support

   malignant     0.9512    0.9286    0.9398        42
      benign     0.9589    0.9722    0.9655        72

    accuracy                         0.9561       114
   macro avg     0.9551    0.9504    0.9526       114
weighted avg     0.9561    0.9561    0.9560       114
```

Both sanity thresholds pass: accuracy 0.9561 ≥ 0.90, ROC-AUC 0.9931 ≥ 0.95.

These numbers are reproducible — `random_state=42` is fixed on both the split and
the forest, so `python train.py` regenerates them exactly.

### Do these numbers generalise?

A single fixed split is reproducible but says nothing about generalisation, so
`robustness_check.py` re-tests the same model two other ways:

| Protocol | Accuracy | ROC-AUC |
| --- | --- | --- |
| Shipped single split (seed 42) | 0.9561 | 0.9931 |
| Stratified 5-fold CV, all 569 samples | 0.9543 ± 0.0102 | 0.9896 ± 0.0077 |
| 80/20 split repeated over seeds 0–9 | 0.9649 (min 0.9386, max 0.9825) | 0.9909 (min 0.9727, max 0.9987) |

Every fold and every seed clears both sanity thresholds, so the result is not
split-dependent. The shipped seed-42 accuracy sits at the **10th percentile** of
the ten seeds — it is a slightly pessimistic draw, not a cherry-picked one, so the
documented 0.9561 understates typical performance rather than flattering it.

**Malignant recall is the weak spot.** Across the ten seeds it averages 0.9452
(min 0.9286, max 0.9762) against benign recall of 0.9764. The shipped split's
0.9286 is the worst case observed: 3 of 42 malignant cases classified as benign.
For any real screening use that false-negative rate is the number to attack —
lowering the decision threshold below 0.5 would trade benign precision for
malignant recall. The spec fixes the default 0.5 threshold, so no such tuning is
applied here.

## What the UI shows

**Sidebar.** Dataset summary (569 samples, 30 features, class balance), a note on
whether a pre-trained `model.pkl` was found, a **Train / retrain model** button,
and a live PASS/FAIL readout of the two sanity thresholds.

**Main pane, top to bottom.**

1. Title and a one-line description of the task.
2. Five metric cards in a row: Accuracy, Precision, Recall, F1-score, ROC-AUC,
   each to four decimal places, with a caption giving test-set size, positive
   class, and training timestamp.
3. Two side-by-side plots. Left: the confusion-matrix heatmap (blue, annotated
   with the raw counts 39 / 3 / 2 / 70, malignant and benign on both axes).
   Right: the ROC curve, a blue line hugging the top-left corner with the dashed
   grey chance diagonal for reference and `AUC = 0.9931` in the legend.
4. The full `classification_report` as monospace text, a per-class
   precision/recall/F1/support table, and a collapsed expander with the raw
   confusion-matrix values as a labelled table.
5. A bar chart of the top 10 most important features by Gini importance.
6. **Try a live prediction** — a preset selector (dataset mean / a real malignant
   sample / a real benign sample) followed by 30 sliders in three columns, one
   per feature, each bounded by that feature's true min and max in the data.
   Clicking **Predict** shows a red MALIGNANT or green BENIGN verdict plus
   `P(malignant)` and `P(benign)` from `predict_proba`.

## Note on serving

Streamlit runs without authentication and also binds a LAN-visible URL. That's
fine locally, but don't expose this app on an untrusted network as-is.
