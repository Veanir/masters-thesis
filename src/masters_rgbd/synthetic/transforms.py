"""Rigid transform and camera math utilities."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def as_vector3(value: list[float] | tuple[float, float, float] | FloatArray) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"Expected vector shape (3,), got {array.shape}")
    return array


def normalize(vector: FloatArray, *, name: str = "vector") -> FloatArray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError(f"Cannot normalize near-zero {name}")
    return vector / norm


def quat_wxyz_to_matrix(
    quat: list[float] | tuple[float, float, float, float] | FloatArray,
) -> FloatArray:
    """Convert a MuJoCo quaternion in wxyz order to a 3x3 rotation matrix."""

    q = np.asarray(quat, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError(f"Expected quaternion shape (4,), got {q.shape}")
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quat_wxyz(rotation: FloatArray) -> FloatArray:
    """Convert a 3x3 rotation matrix to a normalized MuJoCo wxyz quaternion."""

    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError(f"Expected rotation shape (3, 3), got {rotation.shape}")

    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = float(np.sqrt(trace + 1.0) * 2.0)
        quat = np.array(
            [
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ],
            dtype=np.float64,
        )
        return quat / np.linalg.norm(quat)

    diagonal = np.diag(rotation)
    index = int(np.argmax(diagonal))
    if index == 0:
        scale = float(np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0)
        quat = np.array(
            [
                (rotation[2, 1] - rotation[1, 2]) / scale,
                0.25 * scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
            ],
            dtype=np.float64,
        )
    elif index == 1:
        scale = float(np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0)
        quat = np.array(
            [
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                0.25 * scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
            ],
            dtype=np.float64,
        )
    else:
        scale = float(np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0)
        quat = np.array(
            [
                (rotation[1, 0] - rotation[0, 1]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                0.25 * scale,
            ],
            dtype=np.float64,
        )
    return quat / np.linalg.norm(quat)


def random_quat_wxyz(rng: np.random.Generator) -> FloatArray:
    """Sample a uniformly random unit quaternion in wxyz order."""

    u1, u2, u3 = rng.random(3)
    q_xyzw = np.array(
        [
            math.sqrt(1.0 - u1) * math.sin(2.0 * math.pi * u2),
            math.sqrt(1.0 - u1) * math.cos(2.0 * math.pi * u2),
            math.sqrt(u1) * math.sin(2.0 * math.pi * u3),
            math.sqrt(u1) * math.cos(2.0 * math.pi * u3),
        ],
        dtype=np.float64,
    )
    x, y, z, w = q_xyzw
    return np.array([w, x, y, z], dtype=np.float64)


def make_transform(rotation: FloatArray, translation: FloatArray) -> FloatArray:
    if rotation.shape != (3, 3):
        raise ValueError(f"Expected rotation shape (3, 3), got {rotation.shape}")
    if translation.shape != (3,):
        raise ValueError(f"Expected translation shape (3,), got {translation.shape}")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def invert_transform(transform: FloatArray) -> FloatArray:
    if transform.shape != (4, 4):
        raise ValueError(f"Expected transform shape (4, 4), got {transform.shape}")
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def transform_from_pos_quat(position: FloatArray, quat_wxyz: FloatArray) -> FloatArray:
    return make_transform(quat_wxyz_to_matrix(quat_wxyz), position)


def matrix_to_json(transform: FloatArray) -> list[list[float]]:
    if transform.shape != (4, 4):
        raise ValueError(f"Expected transform shape (4, 4), got {transform.shape}")
    return [[float(value) for value in row] for row in transform]


def validate_matrix4(value: object, *, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 matrix, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def look_at_camera(
    position: FloatArray, target: FloatArray, world_up: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Return world_T_camera rotation and MuJoCo xyaxes for a look-at camera.

    MuJoCo cameras follow the OpenGL convention: the camera looks along its
    negative local Z axis. The returned rotation matrix has camera axes as
    columns in world coordinates.
    """

    position = as_vector3(position)
    target = as_vector3(target)
    world_up = normalize(as_vector3(world_up), name="world_up")

    forward = normalize(target - position, name="camera forward")
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) <= 1e-8:
        fallback_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(forward, fallback_up)
    right = normalize(right, name="camera right")
    z_axis = -forward
    y_axis = normalize(np.cross(z_axis, right), name="camera up")

    rotation = np.column_stack([right, y_axis, z_axis])
    xyaxes = np.concatenate([right, y_axis])
    return rotation, xyaxes
