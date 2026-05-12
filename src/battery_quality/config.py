from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class MeasurementConfig:
    """Configuration for reading private multi-channel sensor measurements."""

    data_root: Path
    label_table: Path | None = None
    output_dir: Path = Path("output_data")
    component_prefixes: dict[str, str] = field(
        default_factory=lambda: {
            "component_a": "A",
            "component_b": "B",
        }
    )
    file_suffix: str = "_ddc0.db"
    time_column: str = "time"
    metadata_columns: tuple[str, ...] = ("sample_id", "component", "target")
    sensor_channels: tuple[str, ...] = tuple(
        [f"chan{i}" for i in range(16, 32)] + [f"chan{i}" for i in range(48, 64)]
    )
    environment_columns: tuple[str, ...] = ("temp", "humidity", "pressure")

    def selected_columns(self) -> list[str]:
        return [self.time_column, *self.environment_columns, *self.sensor_channels]
