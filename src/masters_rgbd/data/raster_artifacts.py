"""Content-addressed packed boolean volumes for M1 raster evidence."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from numbers import Integral

import numpy as np
from numpy.typing import NDArray

from masters_rgbd.contracts.manifests import ArtifactRef

BoolArray = NDArray[np.bool_]

PACKED_BOOL_VOLUME_MEDIA_TYPE = "application/vnd.masters-rgbd.packed-bool-volume"

PACKING_ORDER = "C"

PACKING_BITORDER = "little"

_ARTIFACT_DIRECTORY = "masks"

_ARTIFACT_SUFFIX = ".bin"


class M1RasterArtifactError(ValueError):
    """A packed raster artifact violates its immutable storage contract."""


@dataclass(frozen=True, slots=True)
class PackedBoolVolumeRef:
    """A content-bound reference to a C-order, little-endian boolean volume."""

    artifact: ArtifactRef
    shape_xyz: tuple[int, int, int]
    packing_order: str = PACKING_ORDER
    packing_bitorder: str = PACKING_BITORDER

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, ArtifactRef):
            raise M1RasterArtifactError("packed volume artifact must be an ArtifactRef")
        if self.artifact.media_type != PACKED_BOOL_VOLUME_MEDIA_TYPE:
            raise M1RasterArtifactError("packed volume media type differs")
        if not isinstance(self.shape_xyz, tuple) or len(self.shape_xyz) != 3:
            raise M1RasterArtifactError("shape_xyz must contain three positive integers")
        normalized_shape: list[int] = []
        for value in self.shape_xyz:
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise M1RasterArtifactError("shape_xyz must contain three positive integers")
            normalized_shape.append(int(value))
        object.__setattr__(self, "shape_xyz", tuple(normalized_shape))
        if self.packing_order != PACKING_ORDER:
            raise M1RasterArtifactError("packed volume order must be C")
        if self.packing_bitorder != PACKING_BITORDER:
            raise M1RasterArtifactError("packed volume bit order must be little")
        if self.artifact.size_bytes != self.expected_size_bytes:
            raise M1RasterArtifactError("packed volume size differs from shape")

    @property
    def logical_bit_count(self) -> int:
        return prod(self.shape_xyz)

    @property
    def expected_size_bytes(self) -> int:
        return (self.logical_bit_count + 7) // 8
