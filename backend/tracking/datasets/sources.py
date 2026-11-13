"""Load and validate `sources.yaml`, the registry of external training sets."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from tracking.datasets import taxonomy

SOURCES_PATH = Path(__file__).resolve().parent / "sources.yaml"
SLUG = r"^[A-Za-z0-9][A-Za-z0-9_.-]*$"


class RoboflowCoordinates(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    workspace: str = Field(pattern=SLUG)
    project: str = Field(pattern=SLUG)
    version: int = Field(ge=1)
    export_format: str = "yolov8"
    group_pattern: str | None = None


class SourceEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    source: Literal["roboflow", "huggingface-archive"]
    license: str  # human-readable verdict, recorded verbatim in dataset manifests
    # permissive: may train shipped weights | restricted: internal experiments only |
    # unknown: not downloaded at all (the license filter runs before download).
    license_status: Literal["permissive", "restricted", "unknown"]
    version: str | int
    remap: str
    classes: tuple[str, ...]
    domain: Literal["broadcast", "tripod", "mixed"]
    roboflow: RoboflowCoordinates | None = None
    repo: str | None = None
    url: str | None = None
    revision: str | None = None
    archive: str | None = None
    archive_sha256: str | None = None
    citation: str | None = None
    prepared: tuple[str, ...] = ()
    notes: str | None = None

    @model_validator(mode="after")
    def _consistent(self):
        taxonomy.validate_source_classes(self.remap, self.classes)
        if (self.source == "roboflow") != (self.roboflow is not None):
            raise ValueError(f"{self.name}: `roboflow:` coordinates go with source: roboflow")
        return self

    @property
    def shippable(self) -> bool:
        """True only if weights trained on this source may ship."""
        return self.license_status == "permissive"


def load_sources(path: Path = SOURCES_PATH) -> dict[str, SourceEntry]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    entries = [SourceEntry.model_validate(item) for item in data.get("sources") or []]
    names = [entry.name for entry in entries]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate source names in {path}: {duplicates}")
    return {entry.name: entry for entry in entries}
