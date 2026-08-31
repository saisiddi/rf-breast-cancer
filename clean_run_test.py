"""Clean-run test: with NO artifacts on disk, launching the app must train from
scratch, create every artifact, and render real numbers.

This is the "fresh clone" path a new user hits after `pip install -r requirements.txt`.
"""

from __future__ import annotations

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).parent
ARTIFACTS = ["model.pkl", "metrics.json", "confusion_matrix.png", "roc_curve.png"]

# Precondition: this test is only meaningful if the artifacts are absent.
present = [n for n in ARTIFACTS if (ROOT / n).exists()]
assert not present, f"clean-run precondition violated, these already exist: {present}"
print(f"[1/5] precondition OK: none of {ARTIFACTS} exist on disk")

at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=600)
at.run()

assert not at.exception, "app raised on clean run: " + "; ".join(
    str(e.value) for e in at.exception
)
print("[2/5] app.py ran from a clean state with 0 uncaught exceptions")

missing = [n for n in ARTIFACTS if not (ROOT / n).exists()]
assert not missing, f"clean run did not create: {missing}"
sizes = {n: (ROOT / n).stat().st_size for n in ARTIFACTS}
assert all(v > 1000 for v in sizes.values()), f"an artifact is suspiciously small: {sizes}"
print(f"[3/5] all artifacts created by the clean run: {sizes}")

metrics = json.loads((ROOT / "metrics.json").read_text(encoding="utf-8"))
rendered = {m.label: m.value for m in at.metric}
for label, key in [
    ("Accuracy", "accuracy"),
    ("Precision", "precision"),
    ("Recall", "recall"),
    ("F1-score", "f1_score"),
    ("ROC-AUC", "roc_auc"),
]:
    assert label in rendered, f"{label} not rendered on clean run; got {list(rendered)}"
    assert rendered[label] == f"{metrics[key]:.4f}", (
        f"UI {label}={rendered[label]} != metrics.json {metrics[key]:.4f}"
    )
print("[4/5] UI numbers on clean run match the freshly written metrics.json: "
      + ", ".join(f"{k}={rendered[k]}" for k in
                  ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]))

# Reproducibility: the freshly trained run must reproduce the documented numbers.
EXPECTED = {
    "accuracy": 0.956140350877193,
    "precision": 0.958904109589041,
    "recall": 0.9722222222222222,
    "f1_score": 0.9655172413793104,
    "roc_auc": 0.9930555555555556,
}
for key, want in EXPECTED.items():
    got = metrics[key]
    assert abs(got - want) < 1e-12, f"{key} not reproducible: got {got}, documented {want}"
assert metrics["confusion_matrix"] == [[39, 3], [2, 70]], metrics["confusion_matrix"]
print("[5/5] clean run reproduced the documented metrics exactly (random_state=42 holds)")

print("\nCLEAN RUN OK: fresh-clone path trains, writes all artifacts, and shows real numbers")
