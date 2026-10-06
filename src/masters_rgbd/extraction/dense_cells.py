"""Small deterministic dense-cell extractor for the P0 contract fixture.

This is intentionally not the D0 marching-cubes or MISE implementation. It
proves the field/extractor boundary and exact query-budget accounting before
those independent research streams start.
"""

from __future__ import annotations

import numpy as np

_CUBE_CORNERS = np.asarray(
    (
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
    ),
    dtype=np.float64,
)

_TETRAHEDRA = (
    (0, 5, 1, 6),
    (0, 1, 2, 6),
    (0, 2, 3, 6),
    (0, 3, 7, 6),
    (0, 7, 4, 6),
    (0, 4, 5, 6),
)

_TETRA_EDGES = (
    (0, 1),
    (1, 2),
    (2, 0),
    (0, 3),
    (1, 3),
    (2, 3),
)

_TETRA_TRIANGLES = (
    (),
    (0, 3, 2),
    (0, 1, 4),
    (1, 4, 2, 2, 4, 3),
    (1, 2, 5),
    (0, 3, 5, 0, 5, 1),
    (0, 2, 5, 0, 5, 4),
    (5, 4, 3),
    (3, 4, 5),
    (4, 5, 0, 5, 2, 0),
    (1, 5, 0, 5, 3, 0),
    (5, 2, 1),
    (3, 4, 2, 2, 4, 1),
    (4, 1, 0),
    (2, 3, 0),
    (),
)


def _triangulate_tetrahedron(
    corner_points: np.ndarray,
    corner_values: np.ndarray,
    tetrahedron: tuple[int, int, int, int],
) -> tuple[np.ndarray, ...]:
    points = corner_points[np.asarray(tetrahedron)]
    values = corner_values[np.asarray(tetrahedron)]
    case = sum((1 << index) for index, value in enumerate(values) if value <= 0.0)
    triangle_edges = _TETRA_TRIANGLES[case]
    edge_points: dict[int, np.ndarray] = {}
    for edge_index in set(triangle_edges):
        start, end = _TETRA_EDGES[edge_index]
        start_value = values[start]
        end_value = values[end]
        denominator = start_value - end_value
        fraction = 0.5 if denominator == 0.0 else start_value / denominator
        edge_points[edge_index] = points[start] + fraction * (points[end] - points[start])
    return tuple(
        np.stack(tuple(edge_points[index] for index in triangle_edges[offset : offset + 3]))
        for offset in range(0, len(triangle_edges), 3)
    )
