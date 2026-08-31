# PROGRESS LOG — Random Forest Breast Cancer Classifier

Autonomous milestone log. Every milestone runs a self-test that fails loudly
(`selftest.py`) before being marked PASS.

Environment: Windows / PowerShell, project-local virtualenv at `.venv`.

## M1 — Environment & data sanity check
Status: FAIL→FIXED
What I did: Created project venv and installed scikit-learn, numpy, pandas, matplotlib, seaborn, streamlit, joblib. Wrote `data_utils.py` (load, stratified split, sanity report) and `selftest.py` with an asserting check per milestone.
Deviation logged: The system default interpreter is Python 3.14.3, and `pip install -r requirements.txt` FAILED there — `pyarrow` (a hard Streamlit dependency) has no cp314 wheel and its source build aborted with `error: command 'cmake' failed: None`. Fix: rebuilt the venv on the already-installed Python 3.13.7 (`py -3.13 -m venv .venv`), where all wheels resolve. Installed clean: scikit-learn 1.7.2, numpy 2.3.4, pandas 2.3.3, matplotlib 3.10.7, seaborn 0.13.2, streamlit 1.51.0, pyarrow 21.0.0, joblib 1.5.2.
Self-test run: `.\.venv\Scripts\python.exe data_utils.py` then `.\.venv\Scripts\python.exe selftest.py m1`
Result:
```
Feature matrix shape : (569, 30)
Target classes       : ['malignant', 'benign']
Class balance        : {'malignant': 212, 'benign': 357}
Missing values (NaN) : 0
Non-finite values    : 0
Train / test split   : 455 train / 114 test

M1 OK: shape=(569, 30), 0 missing, 0 non-finite, balance=212/357, stratified 455/114 split
ALL REQUESTED SELF-TESTS PASSED: ['m1']
```
Acceptance met? yes — deps installed, dataset loads, shape and class balance printed, and asserts confirm zero missing and zero non-finite values. The 3.14 install failure was a real blocker and was resolved without user input, so status is FAIL→FIXED rather than PASS.

## M2 — Baseline train
Status: PASS
What I did: Wrote `evaluate.py` (metric computation + plot builders) and `train.py` (fit, persist `model.pkl`, write `metrics.json`, render both PNGs). Model is `RandomForestClassifier(n_estimators=200, random_state=42)` on a `train_test_split(test_size=0.2, stratify=y, random_state=42)`.
Self-test run: `.\.venv\Scripts\python.exe train.py` then `.\.venv\Scripts\python.exe selftest.py m2`
Result:
```
Training RandomForestClassifier(n_estimators=200, random_state=42) on 569 samples / 30 features...
Saved model -> model.pkl

M2 OK: fitted RandomForestClassifier(n_estimators=200, random_state=42), trees=200, train/test=455/114
ALL REQUESTED SELF-TESTS PASSED: ['m2']
```
Acceptance met? yes — the self-test unpickles `model.pkl`, asserts the type is `RandomForestClassifier`, asserts `n_estimators == 200`, `random_state == 42`, `len(estimators_) == 200` (proves it is actually fitted), `n_features_in_ == 30`, and cross-checks the recorded split sizes 455/114.

## M3 — Metrics
Status: PASS
What I did: `evaluate.compute_metrics` computes accuracy, confusion matrix, precision, recall, F1, full `classification_report`, and ROC-AUC from `predict_proba` (positive class = benign, class index looked up via `model.classes_` rather than assumed). Per-class precision/recall/F1 for malignant are also stored, since malignant recall is the clinically meaningful number. `save_plots` writes `confusion_matrix.png` (seaborn heatmap) and `roc_curve.png` (ROC + chance diagonal).
Self-test run: deleted both PNGs, then `.\.venv\Scripts\python.exe evaluate.py` (standalone, reloads `model.pkl` from disk and recomputes) then `.\.venv\Scripts\python.exe selftest.py m3`
Result:
```
Accuracy  : 0.9561
Precision : 0.9589
Recall    : 0.9722
F1-score  : 0.9655
ROC-AUC   : 0.9931
Confusion matrix (rows = true, cols = predicted, ['malignant', 'benign']):
  [39, 3]
  [2, 70]
              precision    recall  f1-score   support
   malignant     0.9512    0.9286    0.9398        42
      benign     0.9589    0.9722    0.9655        72
    accuracy                         0.9561       114

M3 OK: acc=0.9561 prec=0.9589 rec=0.9722 f1=0.9655 auc=0.9931; cm=[[39, 3], [2, 70]] consistent; confusion_matrix.png=22836B roc_curve.png=42978B
ALL REQUESTED SELF-TESTS PASSED: ['m3']
```
Acceptance met? yes — and the self-test is stronger than a presence check: it re-derives accuracy from `trace(cm)/sum(cm)` and re-derives precision/recall/F1 from the raw `tn, fp, fn, tp` cells, asserting each matches the stored value to within 1e-9. So the stored numbers cannot be stale or fabricated without the test failing. It also asserts each metric is numeric and in [0, 1], that the matrix is 2x2 summing to 114, and that both PNGs exist and exceed 5 KB. The regenerated plots were also visually inspected: the heatmap shows 39/3/2/70 and the ROC curve hugs the top-left corner with AUC 0.9931 in the legend.

## M4 — Sanity thresholds
Status: PASS
What I did: Gated the run on accuracy >= 0.90 and ROC-AUC >= 0.95. No debugging was needed — the baseline cleared both on the first attempt. Also ran a negative control to prove the gate is not vacuous.
Self-test run: `.\.venv\Scripts\python.exe selftest.py m4`, plus negative control `.\.venv\Scripts\python.exe -c "import selftest; selftest.ACC_THRESHOLD=0.999; selftest.m4()"`
Result:
```
M4 OK: accuracy 0.9561 >= 0.9 and roc_auc 0.9931 >= 0.95
ALL REQUESTED SELF-TESTS PASSED: ['m4']

# negative control (threshold raised to 0.999):
AssertionError: accuracy 0.9561 < required 0.999   [exit code 1]
```
Acceptance met? yes — accuracy 0.9561 clears 0.90 with 5.6 points of margin and ROC-AUC 0.9931 clears 0.95 with 4.3 points. The negative control confirms the assertion genuinely fails and exits non-zero when the threshold is not met, so the PASS is meaningful rather than an assert that can never trip.

## M5 — UI
Status: PASS
What I did: Built `app.py` (Streamlit). Sidebar shows dataset facts, a "Train / retrain" button, and live PASS/FAIL against the M4 thresholds; it reports whether a pre-trained `model.pkl` was found and falls back to training if not. Main pane shows the five headline metrics as `st.metric` cards, the confusion-matrix heatmap, the ROC curve, the full classification report, a per-class breakdown table, and top-10 feature importances. The live-prediction form has one slider per feature, each bounded by that feature's real `min`/`max` from the data and defaulting to the dataset mean, plus presets to load a real malignant or benign sample.
Self-test run: `.\.venv\Scripts\python.exe selftest.py m5`
Result:
```
M5 OK: app.py imports, 30 slider bounds valid, both figures built, mean-row prediction=benign p=0.5900
ALL REQUESTED SELF-TESTS PASSED: ['m5']
```
Acceptance met? yes — the check imports `app.py`, asserts the render functions exist, asserts the 30 slider bounds are aligned with the feature names and that every feature has `max > min` (a zero-width slider would raise at render time), builds both figures, and runs a prediction through the model to confirm the output class is valid and the probabilities sum to 1.

## M6 — End-to-end run test
Status: FAIL→FIXED
What I did: Two independent end-to-end checks. (a) Launched the real server with `streamlit run app.py --server.headless true --server.port 8502` and hit it over HTTP. (b) Wrote `e2e_test.py`, which drives `app.py` through Streamlit's own `AppTest` harness so it can read the values actually rendered on screen and compare them to `metrics.json`.
Bug found and fixed: the first `e2e_test.py` run died with `AttributeError: 'AppTest' object has no attribute 'pyplot'` — there is no `.pyplot` accessor in Streamlit 1.51. I probed the rendered element tree, found `st.pyplot` output surfaces as the `"imgs"` element type, and switched to `at.get("imgs")`. I also made the Predict button lookup select by label instead of by index, so a future sidebar button can't shift the index out from under the test.
Self-test run: `.\.venv\Scripts\python.exe e2e_test.py`, plus the live server probes
Result:
```
[1/7] app.py ran with 0 uncaught exceptions
[2/7] title rendered: 'Random Forest — Breast Cancer Wisconsin Classifier'
[3/7] UI metrics match metrics.json exactly: Accuracy=0.9561, Precision=0.9589, Recall=0.9722, F1-score=0.9655, ROC-AUC=0.9931
[4/7] no placeholder values among the five headline metrics
[5/7] 2 figures rendered (confusion matrix + ROC) and classification report present
[6/7] prediction form works: 30 sliders, {'P(malignant)': '0.4100', 'P(benign)': '0.5900'}, sum=1.0000
[7/7] preset 'A real malignant sample' -> UI predicted MALIGNANT (correct)
[7/7] preset 'A real benign sample' -> UI predicted BENIGN (correct)

M6 OK: app runs headless with no errors, on-screen metrics == metrics.json, and real
samples of both classes are predicted correctly through the UI

# live headless server:
GET /                  -> HTTP 200, 1522 bytes
GET /_stcore/health    -> HTTP 200, body='ok'
# server stdout contained no tracebacks:
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8502
```
Acceptance met? yes — the app launches headless and serves HTTP 200 with a healthy `/_stcore/health`, the script run produced zero uncaught exceptions, and step [3/7] does exactly what the milestone demands: it reads the five metric values off the rendered UI and asserts string equality against `metrics.json`, so any drift between the two fails the test. Steps [6/7] and [7/7] additionally prove the prediction form is wired to the real model rather than a stub.
Note: Streamlit serves with no authentication and also binds a LAN/external URL. That is fine for local use, but this app should not be exposed on an untrusted network as-is.

## M7 — Docs
Status: PASS
What I did: Wrote `README.md` covering setup (including the Python 3.14 / pyarrow caveat from M1), the single run command, a table of what every file does, the model configuration, the positive-class convention, the sample metrics table with the confusion matrix and classification report, and a written description of exactly what the UI renders top-to-bottom. Also wrote `clean_run_test.py` to prove the README's central claim — that a fresh clone works from one command.
Self-test run: moved all four generated artifacts out of the directory, then `.\.venv\Scripts\python.exe clean_run_test.py`; then re-ran `selftest.py all` and `e2e_test.py` and checked their exit codes; then grepped the production modules for hardcoded metric literals.
Result:
```
[1/5] precondition OK: none of ['model.pkl', 'metrics.json', 'confusion_matrix.png', 'roc_curve.png'] exist on disk
[2/5] app.py ran from a clean state with 0 uncaught exceptions
[3/5] all artifacts created by the clean run: {'model.pkl': 651337, 'metrics.json': 5267, 'confusion_matrix.png': 22836, 'roc_curve.png': 42978}
[4/5] UI numbers on clean run match the freshly written metrics.json: Accuracy=0.9561, Precision=0.9589, Recall=0.9722, F1-score=0.9655, ROC-AUC=0.9931
[5/5] clean run reproduced the documented metrics exactly (random_state=42 holds)
CLEAN RUN OK: fresh-clone path trains, writes all artifacts, and shows real numbers

selftest all EXIT=0
e2e EXIT=0
```
Acceptance met? yes — every documented number was verified against the regenerated `metrics.json` rather than copied from memory, and the byte-identical artifact sizes plus the exact metric match confirm the documented results are reproducible. Docs were corrected once during this step: `clean_run_test.py` was missing from the README file table, so it was added.

---

# FINAL AUTONOMOUS AUDIT

**Did every milestone log a PASS?**
Yes — all seven milestones are green. Two are recorded as FAIL→FIXED rather than plain PASS, both honestly:
- **M1** — `pip install` failed on the default Python 3.14.3 because `pyarrow` has no cp314 wheel and its source build died on a missing CMake. Fixed by rebuilding the venv on the already-installed Python 3.13.7. Documented in the README setup section so a user doesn't hit it.
- **M6** — the first `e2e_test.py` run crashed with `AttributeError: 'AppTest' object has no attribute 'pyplot'`. Fixed by discovering the correct accessor (`at.get("imgs")`) from the live element tree instead of guessing.
No milestone was skipped or reordered, and nothing was marked complete on a failing test.

**Does `metrics.json` actually exist and contain all 5 required metrics?**
Yes. Verified programmatically, not by eye:
`{'accuracy': True, 'confusion_matrix': True, 'precision': True, 'recall': True, 'f1_score': True, 'roc_auc': True}` → all present. Values: accuracy 0.9561, precision 0.9589, recall 0.9722, F1 0.9655, ROC-AUC 0.9931, confusion matrix [[39, 3], [2, 70]]. The file also carries per-class figures, the classification report, the ROC curve points, feature importances, and the training params. `selftest.py m3` additionally re-derives accuracy, precision, recall and F1 from the raw confusion-matrix cells and asserts agreement to 1e-9, so the stored values are proven internally consistent rather than merely present.

**Does the UI, when launched fresh in a clean run, load without error and show real (not placeholder) numbers?**
Yes, and this was tested as the actual fresh-clone path: all four artifacts were deleted, then `clean_run_test.py` asserted their absence, ran `app.py`, and confirmed zero uncaught exceptions, all four artifacts recreated, and the on-screen values equal to the newly written `metrics.json`. Separately the real server was launched with `streamlit run app.py --server.headless true`, returning HTTP 200 on `/` and `ok` on `/_stcore/health` with no tracebacks in its output. `e2e_test.py` reads the five values off the rendered UI and asserts string equality with `metrics.json`, explicitly rejects placeholder strings, and confirms a real malignant sample renders MALIGNANT and a real benign sample renders BENIGN through the form.

**Is there any hardcoded/fake metric anywhere in the code?**
No fake metrics. I grepped the four production modules (`app.py`, `train.py`, `evaluate.py`, `data_utils.py`) for metric-shaped literals and assignments. Every value the UI displays traces back to the data:
- All five `st.metric` calls read `metrics['accuracy' | 'precision' | 'recall' | 'f1_score' | 'roc_auc']`.
- The live prediction shows `model.predict_proba(...)` output.
Three numeric literals survive the grep, all legitimate and disclosed:
1. `0.90` and `0.95` in the sidebar — the spec's own sanity thresholds, used only in `>=` comparisons; the numbers shown next to them come from `metrics[...]`.
2. `step = (hi - lo) / 100.0 or 0.01` — slider granularity, not a metric.
3. `selftest.py` holds `ACC_THRESHOLD = 0.90` / `AUC_THRESHOLD = 0.95` (spec-mandated), and `clean_run_test.py` holds the five expected values as an intentional reproducibility oracle. These are in test files asserting *against* the pipeline, not values fed *into* the reported output. Deleting them cannot change what the app displays.

**Deviations from spec**
1. Interpreter: Python 3.13.7 instead of the system default 3.14.3, forced by the pyarrow wheel gap (M1). No effect on results.
2. Positive class: the spec asks for precision/recall/F1/ROC-AUC without naming a positive class. scikit-learn encodes class 1 = benign, so the headline figures use benign as positive (scikit-learn's default). Since malignant recall is the clinically meaningful number, I did not silently drop it — both classes are reported in `per_class`, in the classification report, and in a per-class table in the UI. Malignant: precision 0.9512, recall 0.9286, F1 0.9398.
3. Extra files beyond the requested structure: `selftest.py`, `e2e_test.py`, `clean_run_test.py`. These implement the mandated self-verification loop; all requested files exist as specified.

**Final verdict: PROJECT COMPLETE**
All seven milestones pass their own acceptance criteria, verified by executed assertions with checked exit codes rather than inspection. Accuracy 0.9561 (threshold 0.90) and ROC-AUC 0.9931 (threshold 0.95) both clear with margin, and the threshold gate was negative-controlled to prove it can fail. The app runs from a clean checkout with a single command, and the numbers on screen are provably the numbers in `metrics.json`.

One caveat, not a defect: Streamlit serves without authentication and binds a LAN-visible URL by default. Fine for local use; don't expose this on an untrusted network as-is.

---

# POST-COMPLETION: GENERALISATION CHECK (beyond spec)

Status: PASS
Why I did this: the spec's fixed `random_state=42` makes the headline numbers
reproducible, but reproducible is not the same as generalisable — a single split
can flatter a model. Nothing in M1–M7 ruled out the possibility that 0.9561 was a
lucky draw, so I closed that gap. `robustness_check.py` is read-only with respect
to `model.pkl` and `metrics.json`; the shipped model and reported metrics are
unchanged.

What I did: re-tested the same `RandomForestClassifier(n_estimators=200)` under
stratified 5-fold CV across all 569 samples, then re-ran the exact 80/20 stratified
protocol across 10 different split seeds, tracking per-class recall throughout.

Self-test run: `.\.venv\Scripts\python.exe robustness_check.py`
Result:
```
[1/3] Stratified 5-fold CV (n_estimators=200, all 569 samples)
      accuracy per fold : [0.9649, 0.9386, 0.9561, 0.9474, 0.9646]
      accuracy          : 0.9543 +/- 0.0102
      roc_auc per fold  : [0.9987, 0.9766, 0.9858, 0.994, 0.9928]
      roc_auc           : 0.9896 +/- 0.0077
[2/3] Same 80/20 stratified protocol across 10 split seeds (0-9)
      accuracy : mean 0.9649  min 0.9386  max 0.9825
      roc_auc  : mean 0.9909  min 0.9727  max 0.9987
[3/3] Per-class recall across the 10 seeds
      malignant recall : mean 0.9452  min 0.9286  max 0.9762
      benign    recall : mean 0.9764  min 0.9444  max 1.0000
      shipped (seed 42) accuracy 0.9561 sits at the 10th percentile of the 10 seeds
      shipped malignant recall   0.9286 vs 0.9452 mean across seeds

ROBUSTNESS OK: result holds across 5-fold CV and 10 split seeds; shipped seed is
representative, not cherry-picked
```
Acceptance met? yes. The script asserts the CV mean, the *worst individual fold*,
and the worst of 10 seeds all clear accuracy >= 0.90 and ROC-AUC >= 0.95 — so a
single bad fold would fail it, not just a bad average. All pass.

Findings worth recording:
1. **The shipped number is conservative, not lucky.** Seed 42's 0.9561 lands at the
   10th percentile of ten seeds (mean 0.9649). The documented result understates
   typical performance rather than flattering it, which is the safe direction to be
   wrong in. CV agrees closely at 0.9543 +/- 0.0102.
2. **Tight variance.** CV accuracy std is 0.0102 and every one of the 15
   fold/seed evaluations cleared both thresholds. The M4 pass was not split luck.
3. **Malignant recall is the real weakness, and the shipped split is its worst
   case.** Mean malignant recall across seeds is 0.9452 versus 0.9764 for benign;
   the shipped split's 0.9286 is the minimum observed (3 of 42 malignant cases
   missed). The class asymmetry is consistent, not a one-split artifact — the model
   is systematically more willing to call a case benign. For a screening context
   that false-negative rate is the number to attack, most directly by lowering the
   decision threshold below 0.5 to trade benign precision for malignant recall.
   I did not apply that tuning: the spec fixes the model configuration and the
   default 0.5 threshold, and silently re-tuning would have invalidated the
   documented, reproducible metrics. Flagging it as the top follow-up instead.

Verdict unchanged: **PROJECT COMPLETE**, now with generalisation evidence behind
the headline numbers rather than a single split.
