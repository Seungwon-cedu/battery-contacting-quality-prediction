from __future__ import annotations

"""Local-only feature extraction entry point for authorized private data.

Generated feature tables should stay outside Git unless they are explicitly
cleared for publication.
"""

import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from battery_quality import MeasurementConfig, build_feature_table
from battery_quality.data_loading import load_private_measurements


def main() -> None:
    data_root = os.environ.get("PRIVATE_DATA_ROOT")
    label_table = os.environ.get("PRIVATE_LABEL_TABLE")

    if not data_root:
        raise SystemExit(
            "Set PRIVATE_DATA_ROOT to an authorized local data folder. "
            "Do not commit private data to this repository."
        )

    config = MeasurementConfig(
        data_root=Path(data_root),
        label_table=Path(label_table) if label_table else None,
        output_dir=PROJECT_ROOT / "output_data",
        # Keep project-specific folder prefixes in your local environment if needed.
        component_prefixes={
            "component_a": os.environ.get("COMPONENT_A_PREFIX", "A"),
            "component_b": os.environ.get("COMPONENT_B_PREFIX", "B"),
        },
    )

    measurements = load_private_measurements(config)
    features = build_feature_table(measurements, list(config.sensor_channels))

    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.output_dir / "features_private.csv"
    features.to_csv(output_path, index=False)
    print(f"Wrote private feature table to {output_path}")
    print("Review confidentiality requirements before sharing generated features.")


if __name__ == "__main__":
    main()
