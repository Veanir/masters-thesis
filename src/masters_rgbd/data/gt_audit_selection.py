"""Deterministic freeze of the twenty-asset M1 ground-truth audit.

Feature extraction and quantization happen upstream. A zero score means that an
asset did not pass the frozen eligibility threshold for that feature; positive
integer scores rank eligible assets. The selector never observes audit outcomes,
so a frozen rejection cannot cause a result-dependent replacement.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import StrEnum
from numbers import Integral

from masters_rgbd.contracts.manifests import SplitName

_SHA256_RE = re.compile(r"[0-9a-f]{64}")

_COMMIT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")


class AuditBucket(StrEnum):
    """Frozen source-geometry strata of the M1 audit."""

    OPEN_BOUNDARY = "open_boundary"
    THIN_SLENDER = "thin_slender"
    MULTI_COMPONENT = "multi_component"
    CONCAVE_HOLLOW = "concave_hollow"
    FINE_DETAIL = "fine_detail"


class AuditRole(StrEnum):
    """Whether an audit item may calibrate policy or is a held-out sentinel."""

    DEVELOPMENT = "development"
    TEST_SENTINEL = "test_sentinel"


_BUCKETS = tuple(AuditBucket)

_ROLES = (AuditRole.TEST_SENTINEL, AuditRole.DEVELOPMENT)

_REQUIRED_BY_ROLE = {
    AuditRole.DEVELOPMENT: 3,
    AuditRole.TEST_SENTINEL: 1,
}


def _require_nonempty(value: str, *, field: str) -> None:
    if not value or value.strip() != value:
        raise ValueError(f"{field} must be a non-empty, trimmed string")


def _require_sha256(value: str, *, field: str) -> None:
    if _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase 64-character SHA-256")


@dataclass(frozen=True)
class QuantizedFeatureScores:
    """Integer-only source-geometry feature scores; zero means ineligible."""

    open_boundary: int = 0
    thin_slender: int = 0
    multi_component: int = 0
    concave_hollow: int = 0
    fine_detail: int = 0

    def __post_init__(self) -> None:
        for bucket in _BUCKETS:
            value = self.for_bucket(bucket)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"{bucket.value} score must be a non-negative integer")

    def for_bucket(self, bucket: AuditBucket) -> int:
        """Return the quantized score associated with ``bucket``."""

        return getattr(self, bucket.value)


@dataclass(frozen=True)
class AuditCandidate:
    """Outcome-free source asset considered for the frozen audit."""

    asset_id: str
    geometry_sha256: str
    category: str
    split: SplitName
    feature_scores: QuantizedFeatureScores

    def __post_init__(self) -> None:
        _require_nonempty(self.asset_id, field="asset_id")
        _require_sha256(self.geometry_sha256, field="geometry_sha256")
        _require_nonempty(self.category, field="category")
        if not isinstance(self.split, SplitName):
            raise ValueError("split must be a SplitName")

    def qualifies_for(self, bucket: AuditBucket) -> bool:
        """Whether the frozen upstream feature score admits this bucket."""

        return self.feature_scores.for_bucket(bucket) > 0


@dataclass(frozen=True)
class AuditSelectionFreeze:
    """Immutable provenance inputs that bind one selector execution."""

    source_manifest_sha256: str
    selector_version: str
    source_commit: str

    def __post_init__(self) -> None:
        _require_sha256(self.source_manifest_sha256, field="source_manifest_sha256")
        _require_nonempty(self.selector_version, field="selector_version")
        if _COMMIT_RE.fullmatch(self.source_commit) is None:
            raise ValueError("source_commit must be a full lowercase Git object id")


@dataclass(frozen=True)
class AuditSelectionItem:
    """One asset's immutable bucket and audit-role assignment."""

    bucket: AuditBucket
    role: AuditRole
    asset_id: str
    geometry_sha256: str
    category: str
    split: SplitName
    feature_score: int
    tie_sha256: str


@dataclass(frozen=True)
class FrozenAuditSelection:
    """Complete, outcome-independent 20-asset M1 audit freeze."""

    freeze: AuditSelectionFreeze
    max_category_items_per_bucket: int
    items: tuple[AuditSelectionItem, ...]

    def __post_init__(self) -> None:
        if len(self.items) != 20:
            raise ValueError("a frozen M1 audit must contain exactly 20 items")
        asset_ids = [item.asset_id for item in self.items]
        geometry_hashes = [item.geometry_sha256 for item in self.items]
        if len(set(asset_ids)) != len(asset_ids):
            raise ValueError("a frozen M1 audit cannot repeat an asset_id")
        if len(set(geometry_hashes)) != len(geometry_hashes):
            raise ValueError("a frozen M1 audit cannot repeat geometry")

    @property
    def selection_sha256(self) -> str:
        """Canonical digest binding provenance, relaxation, and assignments."""

        payload = {
            "freeze": {
                "selector_version": self.freeze.selector_version,
                "source_commit": self.freeze.source_commit,
                "source_manifest_sha256": self.freeze.source_manifest_sha256,
            },
            "items": [
                {
                    "asset_id": item.asset_id,
                    "bucket": item.bucket.value,
                    "category": item.category,
                    "feature_score": item.feature_score,
                    "geometry_sha256": item.geometry_sha256,
                    "role": item.role.value,
                    "split": item.split.value,
                    "tie_sha256": item.tie_sha256,
                }
                for item in self.items
            ],
            "max_category_items_per_bucket": self.max_category_items_per_bucket,
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class AuditPoolCount:
    """Raw unique-geometry supply for one bucket and role."""

    bucket: AuditBucket
    role: AuditRole
    required: int
    eligible: int


@dataclass(frozen=True)
class AuditSelectionShortage:
    """Typed failure returned instead of mutating or replacing a frozen list."""

    freeze: AuditSelectionFreeze
    pool_counts: tuple[AuditPoolCount, ...]
    detail: str


AuditSelectionResult = FrozenAuditSelection | AuditSelectionShortage
