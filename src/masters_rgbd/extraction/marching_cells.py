"""Deterministic, provider-neutral extraction kernels for the D0 study.

The implementations in this module deliberately depend only on the frozen P0
field contracts.  A historical MCC adapter and a current learned field can
therefore be compared without either extractor knowing how the model is
hosted.  ``QueryBudget`` always counts unique points actually sent to
``FieldModel.query``; repeated requests are retained as cache hits in the exact
ledger.
"""

from __future__ import annotations

import numpy as np

from masters_rgbd.contracts.fields import (
    FieldModel,
    FieldPrediction,
    QueryBatch,
    QueryBudget,
    ROIBounds,
)


def _point_key(point: np.ndarray) -> bytes:
    canonical = np.asarray(point, dtype="<f8").copy()
    canonical[canonical == 0.0] = 0.0
    return canonical.tobytes()


class ExactQueryLedger:
    """Fail-closed point cache with exact physical field-query accounting."""

    def __init__(
        self,
        field: FieldModel,
        roi: ROIBounds,
        budget: QueryBudget,
        *,
        batch_size: int = 131_072,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("query batch size must be positive")
        self._field = field
        self._frame = roi.frame
        self._maximum = budget.max_field_evaluations
        self._batch_size = batch_size
        self._cache: dict[bytes, tuple[float, float]] = {}
        self._requested_point_count = 0
        self._field_evaluation_count = 0
        self._poisoned = False

    @property
    def requested_point_count(self) -> int:
        return self._requested_point_count

    @property
    def field_evaluation_count(self) -> int:
        return self._field_evaluation_count

    @property
    def cache_hit_count(self) -> int:
        return self._requested_point_count - self._field_evaluation_count

    @property
    def remaining(self) -> int:
        return self._maximum - self._field_evaluation_count

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    def contains(self, point: np.ndarray) -> bool:
        return _point_key(point) in self._cache

    def missing_count(self, points: object) -> int:
        array = _points_array(points)
        known = set(self._cache)
        missing = 0
        for point in array:
            key = _point_key(point)
            if key not in known:
                known.add(key)
                missing += 1
        return missing

    def query(self, points: object) -> FieldPrediction:
        if self._poisoned:
            raise RuntimeError("query ledger is poisoned; retry with a new ledger")
        array = _points_array(points)
        keys = [_point_key(point) for point in array]
        occurrences: dict[bytes, int] = {}
        for key in keys:
            occurrences[key] = occurrences.get(key, 0) + 1
        missing_keys: list[bytes] = []
        missing_points: list[np.ndarray] = []
        planned = set(self._cache)
        for point, key in zip(array, keys, strict=True):
            if key not in planned:
                planned.add(key)
                missing_keys.append(key)
                missing_points.append(point)
        if len(missing_keys) > self.remaining:
            raise ValueError(
                f"query budget {self._maximum} would be exceeded: "
                f"{self._field_evaluation_count} evaluated + {len(missing_keys)} new"
            )

        self._requested_point_count += sum(
            count for key, count in occurrences.items() if key in self._cache
        )
        for offset in range(0, len(missing_points), self._batch_size):
            point_chunk = np.asarray(
                missing_points[offset : offset + self._batch_size],
                dtype=np.float64,
            ).reshape(-1, 3)
            key_chunk = missing_keys[offset : offset + self._batch_size]
            # A submitted point consumes physical query budget even if the field
            # raises before returning a usable prediction.
            self._requested_point_count += len(point_chunk)
            self._field_evaluation_count += len(point_chunk)
            try:
                prediction = self._field.query(QueryBatch(point_chunk, self._frame))
                if len(prediction.udf_m) != len(point_chunk):
                    raise ValueError(
                        "field returned a prediction count different from the query count"
                    )
                chunk_cache = {
                    key: (float(udf_m), float(occupancy_logit))
                    for key, udf_m, occupancy_logit in zip(
                        key_chunk,
                        prediction.udf_m,
                        prediction.occupancy_logit,
                        strict=True,
                    )
                }
            except Exception:
                self._poisoned = True
                raise
            self._cache.update(chunk_cache)
            self._requested_point_count += sum(occurrences[key] - 1 for key in key_chunk)

        udf_m = np.asarray([self._cache[key][0] for key in keys], dtype=np.float64)
        occupancy_logit = np.asarray([self._cache[key][1] for key in keys], dtype=np.float64)
        return FieldPrediction(udf_m=udf_m, occupancy_logit=occupancy_logit)


def _points_array(points: object) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"query points must have shape [N, 3], got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("query points must be finite")
    return array
