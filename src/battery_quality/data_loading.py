from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from .config import MeasurementConfig


def read_sqlite_table(db_path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read the first table from a SQLite measurement file."""

    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Measurement database not found: {db_path}")

    with sqlite3.connect(db_path) as connection:
        table_names = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table';",
            connection,
        )["name"].tolist()
        if not table_names:
            raise ValueError(f"No tables found in {db_path}")

        table_name = table_names[0]
        if columns:
            column_sql = ", ".join(columns)
            query = f"SELECT {column_sql} FROM {table_name}"
        else:
            query = f"SELECT * FROM {table_name}"

        return pd.read_sql_query(query, connection)


def add_relative_time(
    frame: pd.DataFrame,
    time_column: str = "time",
    output_column: str = "time_ms",
) -> pd.DataFrame:
    """Convert an absolute timestamp column to milliseconds from measurement start."""

    result = frame.copy()
    result[time_column] = pd.to_datetime(result[time_column])
    start_time = result[time_column].iloc[0]
    result[output_column] = (result[time_column] - start_time).dt.total_seconds() * 1000
    return result


def load_label_table(label_table: Path | None) -> pd.DataFrame | None:
    """Load a private label table if the caller provides one."""

    if label_table is None:
        return None

    label_table = Path(label_table)
    if not label_table.exists():
        raise FileNotFoundError(f"Label table not found: {label_table}")

    if label_table.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(label_table)
    if label_table.suffix.lower() == ".csv":
        return pd.read_csv(label_table)

    raise ValueError("Supported label formats are .xlsx, .xls, and .csv")


def load_measurement_folder(
    folder: Path,
    component: str,
    config: MeasurementConfig,
    target: str | None = None,
) -> pd.DataFrame:
    """Load one measurement folder and attach minimal metadata."""

    db_files = sorted(Path(folder).glob(f"*{config.file_suffix}"))
    if not db_files:
        raise FileNotFoundError(f"No measurement file ending in {config.file_suffix} in {folder}")

    frame = read_sqlite_table(db_files[0], config.selected_columns())
    frame = add_relative_time(frame, config.time_column)
    frame.insert(0, "sample_id", Path(folder).name)
    frame.insert(1, "component", component)
    frame.insert(2, "target", target or "unknown")
    return frame


def discover_measurement_folders(config: MeasurementConfig) -> list[tuple[Path, str]]:
    """Find measurement folders using the configured component prefixes."""

    folders: list[tuple[Path, str]] = []
    for component, prefix in config.component_prefixes.items():
        for folder in sorted(config.data_root.glob(f"{prefix}*")):
            if folder.is_dir():
                folders.append((folder, component))
    return folders


def load_private_measurements(config: MeasurementConfig) -> pd.DataFrame:
    """Load private measurements into one DataFrame.

    The label-joining step is intentionally left project-specific. For public review, this function
    demonstrates the ingestion pattern without embedding any private label schema.
    """

    frames = [
        load_measurement_folder(folder, component, config)
        for folder, component in discover_measurement_folders(config)
    ]
    if not frames:
        raise ValueError(f"No measurement folders found under {config.data_root}")

    return pd.concat(frames, ignore_index=True)
