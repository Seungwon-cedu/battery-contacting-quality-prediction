from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import iqr, kurtosis, skew, zscore


def trim_signal(
    frame: pd.DataFrame,
    channel: str,
    start_ms: float = 30.0,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Remove pre-trigger samples and optionally stop after signal decay."""

    if "time_ms" not in frame.columns:
        raise ValueError("Expected a 'time_ms' column. Run add_relative_time first.")
    if channel not in frame.columns:
        raise ValueError(f"Trim channel not found: {channel}")

    trimmed = frame[frame["time_ms"] >= start_ms].copy()
    if threshold is None or trimmed.empty:
        return trimmed

    valid = trimmed[channel] > threshold
    if not valid.any():
        return trimmed

    last_valid_position = np.flatnonzero(valid.to_numpy())[-1]
    return trimmed.iloc[: last_valid_position + 1].copy()


def smooth_channels(
    frame: pd.DataFrame,
    channels: list[str],
    window_length: int = 11,
    polyorder: int = 3,
) -> pd.DataFrame:
    """Apply Savitzky-Golay smoothing to selected sensor channels."""

    result = frame.copy()
    for channel in channels:
        if channel in result.columns and len(result[channel]) >= window_length:
            result[channel] = savgol_filter(result[channel].astype(float), window_length, polyorder)
    return result


def extract_signal_features(signal: np.ndarray) -> dict[str, float]:
    """Compute robust statistical features from one sensor signal."""

    values = np.asarray(signal, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "var": np.nan,
            "iqr": np.nan,
            "skewness": np.nan,
            "kurtosis": np.nan,
            "energy": np.nan,
            "peaks": 0.0,
            "outliers": 0.0,
        }

    z_scores = zscore(finite)
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "var": float(np.var(finite)),
        "iqr": float(iqr(finite)),
        "skewness": float(skew(finite)),
        "kurtosis": float(kurtosis(finite)),
        "energy": float(np.sum(finite**2)),
        "peaks": float(len(find_peaks(finite)[0])),
        "outliers": float(np.sum(np.abs(z_scores) > 3)),
    }


def extract_channel_features(frame: pd.DataFrame, channels: list[str]) -> dict[str, float]:
    """Extract feature columns for each selected sensor channel."""

    features: dict[str, float] = {}
    for channel in channels:
        if channel not in frame.columns:
            continue

        channel_features = extract_signal_features(frame[channel].to_numpy())
        for name, value in channel_features.items():
            features[f"{channel}_{name}"] = value
    return features


def build_feature_table(
    measurements: pd.DataFrame,
    channels: list[str],
    group_columns: tuple[str, ...] = ("sample_id", "component", "target"),
    smooth: bool = True,
) -> pd.DataFrame:
    """Convert long-form measurements into one feature row per experiment."""

    rows: list[dict[str, float | str]] = []
    for group_values, group in measurements.groupby(list(group_columns), dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        prepared = smooth_channels(group, channels) if smooth else group
        row = dict(zip(group_columns, group_values))
        row.update(extract_channel_features(prepared, channels))

        for column in ("temp", "humidity", "pressure"):
            if column in group.columns:
                row[column] = float(group[column].iloc[0])

        rows.append(row)

    return pd.DataFrame(rows)
