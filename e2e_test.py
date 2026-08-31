"""M6 end-to-end test: run app.py through Streamlit's AppTest harness and assert
that the numbers actually rendered on screen match metrics.json exactly.

This catches runtime errors AND placeholder/stale values, which a plain
"does the server start" check cannot.
"""

from __future__ import annotations

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).parent
METRICS = json.loads((ROOT / "metrics.json").read_text(encoding="utf-8"))

EXPECTED = {
    "Accuracy": f"{METRICS['accuracy']:.4f}",
    "Precision": f"{METRICS['precision']:.4f}",
    "Recall": f"{METRICS['recall']:.4f}",
    "F1-score": f"{METRICS['f1_score']:.4f}",
    "ROC-AUC": f"{METRICS['roc_auc']:.4f}",
}


def main() -> None:
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
    at.run()

    # 1. no uncaught exceptions anywhere in the script run
    assert not at.exception, "app raised exception(s): " + "; ".join(
        str(e.value) for e in at.exception
    )
    print("[1/7] app.py ran with 0 uncaught exceptions")

    # 2. title rendered
    titles = [t.value for t in at.title]
    assert any("Random Forest" in t for t in titles), f"title missing, got {titles}"
    print(f"[2/7] title rendered: {titles[0]!r}")

    # 3. all five headline metrics present on screen AND equal to metrics.json
    rendered = {m.label: m.value for m in at.metric}
    for label, expected_value in EXPECTED.items():
        assert label in rendered, f"metric '{label}' not rendered. rendered={list(rendered)}"
        assert rendered[label] == expected_value, (
            f"UI shows {label}={rendered[label]!r} but metrics.json says {expected_value!r}"
        )
    print("[3/7] UI metrics match metrics.json exactly: " + ", ".join(
        f"{k}={rendered[k]}" for k in EXPECTED
    ))

    # 4. values are real, not placeholders
    for label, value in ((k, rendered[k]) for k in EXPECTED):
        assert value not in ("0.0000", "1.0000", "N/A", "--", ""), (
            f"{label} looks like a placeholder: {value!r}"
        )
    print("[4/7] no placeholder values among the five headline metrics")

    # 5. both plots rendered as images (st.pyplot -> "imgs" element), and the
    #    classification report is on screen
    figures = at.get("imgs")
    assert len(figures) >= 2, f"expected >=2 matplotlib figures (cm + roc), got {len(figures)}"
    report_blocks = [c.value for c in at.code]
    assert any("malignant" in b and "benign" in b for b in report_blocks), (
        "classification report not rendered"
    )
    print(f"[5/7] {len(figures)} figures rendered (confusion matrix + ROC) "
          "and classification report present")

    # 6. exercise the live-prediction form end to end through the widget layer
    n_sliders = len(at.slider)
    assert n_sliders == 30, f"expected 30 feature sliders, got {n_sliders}"
    predict_buttons = [b for b in at.button if b.label == "Predict"]
    assert predict_buttons, f"no Predict button found; labels={[b.label for b in at.button]}"
    predict_buttons[0].click().run()
    assert not at.exception, "exception after clicking Predict: " + "; ".join(
        str(e.value) for e in at.exception
    )
    post = {m.label: m.value for m in at.metric}
    prob_labels = [k for k in post if k.startswith("P(")]
    assert len(prob_labels) == 2, f"expected 2 class-probability metrics, got {prob_labels}"
    probs = [float(post[k]) for k in prob_labels]
    assert abs(sum(probs) - 1.0) < 1e-4, f"probabilities do not sum to 1: {post}"
    verdicts = [s.value for s in at.success] + [e.value for e in at.error]
    assert any("Prediction:" in v for v in verdicts), f"no prediction verdict shown: {verdicts}"
    print(f"[6/7] prediction form works: 30 sliders, "
          f"{ {k: post[k] for k in prob_labels} }, sum={sum(probs):.4f}")

    # 7. behavioural check: real samples of each class must be classified correctly
    #    through the UI, proving the form is wired to the real model.
    for preset, expected in (
        ("A real malignant sample", "MALIGNANT"),
        ("A real benign sample", "BENIGN"),
    ):
        at2 = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
        at2.run()
        at2.radio[0].set_value(preset).run()
        [b for b in at2.button if b.label == "Predict"][0].click().run()
        assert not at2.exception, f"exception on preset {preset!r}: " + "; ".join(
            str(e.value) for e in at2.exception
        )
        verdict_text = " ".join([s.value for s in at2.success] + [e.value for e in at2.error])
        assert expected in verdict_text, (
            f"preset {preset!r} should predict {expected}, UI said: {verdict_text!r}"
        )
        print(f"[7/7] preset {preset!r} -> UI predicted {expected} (correct)")

    print("\nM6 OK: app runs headless with no errors, on-screen metrics == metrics.json, "
          "and real samples of both classes are predicted correctly through the UI")


if __name__ == "__main__":
    main()
