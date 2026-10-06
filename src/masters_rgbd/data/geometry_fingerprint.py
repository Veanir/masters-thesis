"""Canonical fingerprints of processed triangle geometry expressed in metres."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

DEFAULT_QUANTIZATION_M = 1e-6

FINGERPRINT_ALGORITHM = "sha256"

FINGERPRINT_PREPROCESSING_VERSION = "processed-triangle-geometry-m-v1"

_DOMAIN_SEPARATOR = b"masters-rgbd\x00processed-triangle-geometry\x00v1\x00"

_INT64_EXCLUSIVE_BOUND = float(2**63)


@dataclass(frozen=True, slots=True)
class GeometryFingerprint:
    """Digest and the complete metadata needed to interpret it."""

    sha256: str
    algorithm: str
    preprocessing_version: str
    quantization_m: float
    triangle_count: int


def _length_prefixed(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack(">I", len(encoded)) + encoded


def _validated_vertices(vertices_m: ArrayLike) -> NDArray[np.float64]:
    try:
        vertices = np.asarray(vertices_m)
    except (TypeError, ValueError) as error:
        raise ValueError("vertices_m must be a rectangular numeric array") from error
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or vertices.shape[0] == 0:
        raise ValueError("vertices_m must have non-empty shape [vertex, xyz]")
    if not np.issubdtype(vertices.dtype, np.number) or np.issubdtype(
        vertices.dtype, np.complexfloating
    ):
        raise TypeError("vertices_m must contain real numeric values")
    vertices_float = np.asarray(vertices, dtype=np.float64)
    if not np.isfinite(vertices_float).all():
        raise ValueError("vertices_m must contain only finite values")
    return vertices_float


def _validated_faces(faces: ArrayLike, *, vertex_count: int) -> NDArray[np.int64]:
    try:
        face_array = np.asarray(faces)
    except (TypeError, ValueError) as error:
        raise ValueError("faces must be a rectangular integer array") from error
    if face_array.ndim != 2 or face_array.shape[1:] != (3,) or face_array.shape[0] == 0:
        raise ValueError("faces must have non-empty triangulated shape [face, 3]")
    if not np.issubdtype(face_array.dtype, np.integer):
        raise TypeError("faces must contain integer vertex indices")
    if face_array.min() < 0 or face_array.max() >= vertex_count:
        raise ValueError("faces contain a vertex index outside vertices_m")
    return np.asarray(face_array, dtype=np.int64)


def _quantized_vertices(
    vertices_m: NDArray[np.float64],
    *,
    quantization_m: float,
) -> NDArray[np.int64]:
    scaled = vertices_m / quantization_m
    if not np.isfinite(scaled).all():
        raise ValueError("vertices_m exceed the quantized coordinate range")
    rounded = np.rint(scaled)
    if (rounded < -_INT64_EXCLUSIVE_BOUND).any() or (rounded >= _INT64_EXCLUSIVE_BOUND).any():
        raise ValueError("vertices_m exceed the signed 64-bit quantized coordinate range")
    # Integer quantization canonicalizes both +0.0 and -0.0 to the same value.
    return np.asarray(rounded, dtype=np.int64)


def _canonical_triangles(
    quantized_vertices: NDArray[np.int64],
    faces: NDArray[np.int64],
) -> NDArray[np.int64]:
    triangles = quantized_vertices[faces]

    vertex_order = np.lexsort(
        (triangles[:, :, 2], triangles[:, :, 1], triangles[:, :, 0]),
        axis=1,
    )
    ordered_vertices = np.take_along_axis(triangles, vertex_order[:, :, None], axis=1)
    flattened = ordered_vertices.reshape((-1, 9))

    triangle_order = np.lexsort(tuple(flattened[:, column] for column in range(8, -1, -1)))
    ordered_triangles = flattened[triangle_order]

    # Duplicate faces do not change the geometric triangle set. Collapse them after
    # winding and ordering canonicalization so exporter artefacts do not affect SHA.
    keep = np.ones(ordered_triangles.shape[0], dtype=np.bool_)
    keep[1:] = np.any(ordered_triangles[1:] != ordered_triangles[:-1], axis=1)
    return np.ascontiguousarray(ordered_triangles[keep], dtype=np.dtype(">i8"))


def fingerprint_processed_triangle_geometry(
    vertices_m: ArrayLike,
    faces: ArrayLike,
    *,
    quantization_m: float = DEFAULT_QUANTIZATION_M,
    preprocessing_version: str = FINGERPRINT_PREPROCESSING_VERSION,
) -> GeometryFingerprint:
    """Fingerprint a triangle set after metric, permutation-invariant preprocessing.

    Coordinates are divided by ``quantization_m`` and rounded to the nearest integer
    using IEEE-754 round-to-even semantics. Triangle winding, input vertex numbering,
    face ordering, duplicate faces, and signed zero do not affect the result.
    """

    if isinstance(quantization_m, bool) or not isinstance(
        quantization_m, (int, float, np.integer, np.floating)
    ):
        raise TypeError("quantization_m must be a real scalar")
    quantization = float(quantization_m)
    if not np.isfinite(quantization) or quantization <= 0.0:
        raise ValueError("quantization_m must be finite and positive")
    if not isinstance(preprocessing_version, str) or not preprocessing_version:
        raise ValueError("preprocessing_version must be a non-empty string")

    vertices = _validated_vertices(vertices_m)
    face_array = _validated_faces(faces, vertex_count=vertices.shape[0])
    canonical = _canonical_triangles(
        _quantized_vertices(vertices, quantization_m=quantization),
        face_array,
    )

    digest = hashlib.sha256()
    digest.update(_DOMAIN_SEPARATOR)
    digest.update(_length_prefixed(FINGERPRINT_ALGORITHM))
    digest.update(_length_prefixed(preprocessing_version))
    digest.update(struct.pack(">d", quantization))
    digest.update(struct.pack(">Q", canonical.shape[0]))
    digest.update(canonical.tobytes(order="C"))
    return GeometryFingerprint(
        sha256=digest.hexdigest(),
        algorithm=FINGERPRINT_ALGORITHM,
        preprocessing_version=preprocessing_version,
        quantization_m=quantization,
        triangle_count=canonical.shape[0],
    )
