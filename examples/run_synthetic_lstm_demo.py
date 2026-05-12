from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from battery_quality.lstm import LSTMTrainingConfig, cross_validate_lstm


def make_synthetic_sequences(
    n_samples: int = 120,
    timesteps: int = 40,
    n_channels: int = 8,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a small non-confidential sequence dataset for LSTM review."""

    rng = np.random.default_rng(random_state)
    labels = np.array(["reference", "gap_like", "alignment_like", "contamination_like"])
    y_idx = rng.integers(0, len(labels), size=n_samples)
    x = np.zeros((n_samples, timesteps, n_channels), dtype=np.float32)
    t = np.linspace(0, 1, timesteps)

    for i, class_id in enumerate(y_idx):
        base_frequency = 1.0 + class_id * 0.45
        for channel in range(n_channels):
            amplitude = 0.6 + class_id * 0.18 + channel * 0.03
            phase = channel * 0.15
            trend = class_id * 0.2 * t
            signal = amplitude * np.sin(2 * np.pi * base_frequency * t + phase) + trend
            x[i, :, channel] = signal + rng.normal(0, 0.05, size=timesteps)

    return x, labels[y_idx]


def main() -> None:
    sequences, labels = make_synthetic_sequences()
    config = LSTMTrainingConfig(
        hidden_size=32,
        num_layers=2,
        dropout=0.20,
        learning_rate=1e-3,
        batch_size=16,
        epochs=8,
        patience=3,
        random_state=42,
    )

    result = cross_validate_lstm(
        sequences,
        labels,
        n_splits=3,
        augmentation="oversampling_jitter",
        config=config,
    )

    print("Synthetic LSTM cross-validation")
    print(f"- accuracy: mean={result['accuracy'].mean():.3f}, folds={np.round(result['accuracy'], 3)}")
    print(f"- f1_macro: mean={result['f1_macro'].mean():.3f}, folds={np.round(result['f1_macro'], 3)}")
    print("Note: this demo uses synthetic data only; metrics are for code smoke testing.")


if __name__ == "__main__":
    main()
