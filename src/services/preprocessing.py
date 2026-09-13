from __future__ import annotations

import pandas as pd
from meerax.data.split import stratified_split
from sklearn.preprocessing import PowerTransformer


def power_transform(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit PowerTransformer on train only, apply to both — matches the notebook's approach."""
    transformer = PowerTransformer()
    train_out = pd.DataFrame(
        transformer.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    test_out = pd.DataFrame(
        transformer.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    return train_out, test_out


def split_fraud_data(
    df: pd.DataFrame,
    target_col: str = "Class",
    train_ratio: float = 0.7,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified 70/30 train/test split via meerax's 3-way split utility.

    stratified_split rejects a 0.0 val or test ratio (its second internal
    train_test_split call would receive test_size=1.0, which scikit-learn
    rejects). A 15/15 val/test split is requested instead and merged back
    into one held-out frame, so the outcome is a plain 70/30 split with no
    rows set aside and never used.
    """
    train, val, test = stratified_split(
        df, target_col, train_ratio=train_ratio, val_ratio=0.15, random_state=random_state
    )
    held_out = pd.concat([val, test], ignore_index=True)
    return train.reset_index(drop=True), held_out
