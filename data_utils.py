"""Data loading and splitting utilities for the Breast Cancer Wisconsin dataset."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.2


@dataclass(frozen=True)
class Dataset:
    """Container for the loaded dataset plus metadata used by the UI."""

    X: pd.DataFrame
    y: pd.Series
    feature_names: list[str]
    # target_names[i] is the label for class index i -> ['malignant', 'benign']
    target_names: list[str]

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    def class_balance(self) -> dict[str, int]:
        counts = self.y.value_counts().sort_index()
        return {self.target_names[int(idx)]: int(cnt) for idx, cnt in counts.items()}

    def feature_bounds(self) -> pd.DataFrame:
        """Min / max / mean per feature. Used to bound the UI sliders."""
        return pd.DataFrame(
            {
                "min": self.X.min(),
                "max": self.X.max(),
                "mean": self.X.mean(),
            }
        )


def load_data() -> Dataset:
    """Load the Breast Cancer Wisconsin dataset as a DataFrame/Series pair."""
    bunch = load_breast_cancer()
    feature_names = [str(name) for name in bunch.feature_names]
    X = pd.DataFrame(bunch.data, columns=feature_names)
    y = pd.Series(bunch.target, name="target")
    return Dataset(
        X=X,
        y=y,
        feature_names=feature_names,
        target_names=[str(name) for name in bunch.target_names],
    )


def split_data(
    dataset: Dataset,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split, as specified in the project spec."""
    return train_test_split(
        dataset.X,
        dataset.y,
        test_size=test_size,
        stratify=dataset.y,
        random_state=random_state,
    )


def sanity_report(dataset: Dataset) -> dict[str, object]:
    """Facts about the dataset, used by the M1 self-test and printed by __main__."""
    missing = int(dataset.X.isna().sum().sum())
    non_finite = int((~np.isfinite(dataset.X.to_numpy())).sum())
    return {
        "shape": tuple(dataset.X.shape),
        "n_samples": dataset.n_samples,
        "n_features": dataset.n_features,
        "class_balance": dataset.class_balance(),
        "target_names": dataset.target_names,
        "missing_values": missing,
        "non_finite_values": non_finite,
    }


if __name__ == "__main__":
    ds = load_data()
    report = sanity_report(ds)

    print("=== M1: Breast Cancer Wisconsin dataset sanity check ===")
    print(f"Feature matrix shape : {report['shape']}")
    print(f"Samples              : {report['n_samples']}")
    print(f"Features             : {report['n_features']}")
    print(f"Target classes       : {report['target_names']}")
    print(f"Class balance        : {report['class_balance']}")
    print(f"Missing values (NaN) : {report['missing_values']}")
    print(f"Non-finite values    : {report['non_finite_values']}")

    X_train, X_test, y_train, y_test = split_data(ds)
    print(f"Train / test split   : {X_train.shape[0]} train / {X_test.shape[0]} test")
    print(
        "Train class balance  : "
        f"{ {ds.target_names[int(i)]: int(c) for i, c in y_train.value_counts().sort_index().items()} }"
    )
    print(
        "Test class balance   : "
        f"{ {ds.target_names[int(i)]: int(c) for i, c in y_test.value_counts().sort_index().items()} }"
    )
