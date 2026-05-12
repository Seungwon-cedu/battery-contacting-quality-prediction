from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from battery_quality.modeling import evaluate_random_forest, split_features_and_target
from battery_quality.modeling import top_feature_importances


def make_synthetic_feature_table(
    n_samples: int = 180,
    n_channels: int = 12,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create non-confidential demo data with similar shape to engineered sensor features."""

    rng = np.random.default_rng(random_state)
    targets = np.array(["reference", "gap_like", "alignment_like", "contamination_like"])
    target_idx = rng.integers(0, len(targets), size=n_samples)

    rows = []
    for sample_id, class_id in enumerate(target_idx):
        row: dict[str, float | str] = {
            "sample_id": f"synthetic_{sample_id:03d}",
            "component": "component_a" if sample_id % 2 == 0 else "component_b",
            "target": targets[class_id],
            "temp": 22.0 + rng.normal(0, 0.7),
            "humidity": 40.0 + rng.normal(0, 2.0),
            "pressure": 1013.0 + rng.normal(0, 3.0),
        }

        for channel in range(n_channels):
            class_shift = class_id * 0.45
            channel_bias = channel * 0.03
            signal_level = rng.normal(class_shift + channel_bias, 0.25)
            row[f"chan{channel}_mean"] = signal_level
            row[f"chan{channel}_std"] = abs(rng.normal(0.12 + class_id * 0.02, 0.03))
            row[f"chan{channel}_iqr"] = abs(rng.normal(0.20 + class_id * 0.03, 0.04))
            row[f"chan{channel}_energy"] = abs(rng.normal(10 + class_id * 2 + channel, 1.5))

        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    features = make_synthetic_feature_table()
    evaluation = evaluate_random_forest(features, n_splits=5)

    print("Cross-validation summary")
    for metric, scores in evaluation["scores"].items():
        print(f"- {metric}: mean={scores.mean():.3f}, folds={np.round(scores, 3)}")

    print("\nClassification report")
    print(evaluation["classification_report"])

    x, y = split_features_and_target(features)
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ).fit(x, y)

    print("Top synthetic feature importances")
    print(top_feature_importances(model, list(x.columns), n=10))


if __name__ == "__main__":
    main()
