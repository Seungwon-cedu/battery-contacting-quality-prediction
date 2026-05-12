from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.preprocessing import LabelEncoder


def split_features_and_target(
    frame: pd.DataFrame,
    target_column: str = "target",
    drop_columns: tuple[str, ...] = ("sample_id", "component"),
) -> tuple[pd.DataFrame, pd.Series]:
    """Return numeric feature columns and target labels."""

    y = frame[target_column]
    excluded = set(drop_columns) | {target_column}
    x = frame.drop(columns=[col for col in excluded if col in frame.columns])
    x = x.select_dtypes(include="number")
    return x, y


def evaluate_random_forest(
    frame: pd.DataFrame,
    target_column: str = "target",
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, object]:
    """Evaluate a class-balanced Random Forest using stratified cross-validation."""

    x, y = split_features_and_target(frame, target_column)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scores = {
        "accuracy": cross_val_score(model, x, y_encoded, cv=cv, scoring="accuracy"),
        "f1_macro": cross_val_score(model, x, y_encoded, cv=cv, scoring="f1_macro"),
    }
    predictions = cross_val_predict(model, x, y_encoded, cv=cv)

    return {
        "model": model,
        "label_encoder": label_encoder,
        "scores": scores,
        "classification_report": classification_report(
            y_encoded,
            predictions,
            target_names=label_encoder.classes_,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_encoded, predictions),
    }


def tune_random_forest(
    frame: pd.DataFrame,
    target_column: str = "target",
    random_state: int = 42,
) -> GridSearchCV:
    """Tune Random Forest hyperparameters for macro-F1."""

    x, y = split_features_and_target(frame, target_column)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    param_grid = {
        "max_depth": [None, 5, 10, 15],
        "min_samples_split": [2, 5, 10],
        "max_features": ["sqrt", "log2", None],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
    )
    search.fit(x, y_encoded)
    return search


def top_feature_importances(
    model: RandomForestClassifier,
    feature_names: list[str],
    n: int = 20,
) -> pd.Series:
    """Return the top feature importances from a fitted Random Forest."""

    importances = pd.Series(model.feature_importances_, index=feature_names)
    return importances.sort_values(ascending=False).head(n)
