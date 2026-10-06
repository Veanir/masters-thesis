"""Versioned, JSON-serializable data and provenance contracts."""

from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from masters_rgbd.contracts.geometry import (
    AffineTransform,
)

SCHEMA_VERSION = "2.0.0"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_OCI_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class DatasetMembership(StrEnum):
    BASE = "base"
    GENERATED = "generated"


class SplitName(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


def _required_text(value: str, *, name: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{name} cannot be empty")


def _sha256(value: str, *, name: str) -> None:
    if not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class ArtifactRef:
    uri: str
    sha256: str
    size_bytes: int
    media_type: str

    def __post_init__(self) -> None:
        _required_text(self.uri, name="artifact uri")
        _required_text(self.media_type, name="artifact media_type")
        _sha256(self.sha256, name="artifact sha256")
        if self.size_bytes <= 0:
            raise ValueError("artifact size_bytes must be positive")


@dataclass(frozen=True)
class GeneratorProvenance:
    family: str
    version: str
    model_digest: str
    preprocessing_version: str
    prompt: str | None = None
    negative_prompt: str | None = None
    seed: int | None = None
    input_sha256: tuple[str, ...] = ()
    missing_fields_reason: str | None = None

    def __post_init__(self) -> None:
        _required_text(self.family, name="generator family")
        _required_text(self.version, name="generator version")
        _required_text(self.preprocessing_version, name="preprocessing version")
        if not _OCI_DIGEST_RE.fullmatch(self.model_digest):
            raise ValueError("generator model_digest must be an immutable sha256 digest")
        if self.seed is not None and self.seed < 0:
            raise ValueError("generator seed cannot be negative")
        for digest in self.input_sha256:
            _sha256(digest, name="generator input_sha256")


@dataclass(frozen=True)
class AssetManifest:
    asset_id: str
    source_asset_id: str
    source_uri: str
    source_license: str
    source_units: str
    geometry_sha256: str
    family_id: str
    lineage_root_id: str
    split_group_id: str
    category_id: str
    membership: DatasetMembership
    preprocessing_version: str
    original_mesh: ArtifactRef
    processed_mesh: ArtifactRef
    generator: GeneratorProvenance | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "asset_id",
            "source_asset_id",
            "source_uri",
            "source_license",
            "source_units",
            "family_id",
            "lineage_root_id",
            "split_group_id",
            "category_id",
            "preprocessing_version",
        ):
            _required_text(getattr(self, name), name=name)
        _sha256(self.geometry_sha256, name="geometry_sha256")
        if self.membership == DatasetMembership.GENERATED and self.generator is None:
            raise ValueError("generated assets require generator provenance")
        if self.membership == DatasetMembership.BASE and self.generator is not None:
            raise ValueError("base assets cannot carry generated-asset provenance")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported asset schema version: {self.schema_version}")


def to_json_dict(value: object) -> Any:
    """Convert a validated contract object into canonical JSON-compatible data."""

    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, AffineTransform):
        return {
            "source": value.source.value,
            "target": value.target.value,
            "matrix": value.to_json(),
        }
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: to_json_dict(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [to_json_dict(item) for item in value]
    if isinstance(value, list):
        return [to_json_dict(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_json_dict(item) for key, item in value.items()}
    return value


def canonical_json_bytes(value: object) -> bytes:
    """Serialize semantic content deterministically for content addressing."""

    return json.dumps(
        to_json_dict(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
