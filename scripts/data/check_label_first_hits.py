"""Independent positive and foreground-omission checks against actual triangles."""

import hashlib

import numpy as np
from scipy.ndimage import binary_erosion


def verify_view(mesh, camera, k, encoded, sample_key, tolerance=0.00025):
    assert encoded.ndim == 2 and encoded.dtype == np.uint16
    h, w = encoded.shape
    mask = encoded > 0
    positive = np.flatnonzero(mask)
    chosen = (
        positive[
            np.floor(
                (np.arange(min(64, len(positive))) + 0.5) * len(positive) / min(64, len(positive))
            ).astype(int)
        ]
        if len(positive)
        else np.array([], dtype=int)
    )
    # Stratify the full image, including background. Do not choose using GT.
    rng = np.random.default_rng(
        int.from_bytes(hashlib.sha256(sample_key.encode()).digest()[:8], "big")
    )
    yy, xx = np.mgrid[:8, :8]
    u = np.minimum(w - 1, ((xx + rng.uniform(0.1, 0.9, (8, 8))) * w / 8).astype(int)).ravel()
    v = np.minimum(h - 1, ((yy + rng.uniform(0.1, 0.9, (8, 8))) * h / 8).astype(int)).ravel()
    interior = binary_erosion(mask, border_value=1)
    exterior = binary_erosion(~mask, border_value=1)
    stable = interior[v, u] | exterior[v, u]
    # A one-pixel mask boundary can differ under independent float32/float64
    # ray arithmetic. Exclude only those uniformly sampled boundary pixels;
    # keep the deterministic positive samples and report the exclusion count.
    uniform = v[stable] * w + u[stable]
    indices = np.unique(np.concatenate([chosen, uniform]))
    v, u = indices // w, indices % w
    directions = (
        np.column_stack(((u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1], np.ones(len(indices))))
        @ camera[:3, :3].T
    )
    origins = np.broadcast_to(camera[:3, 3], directions.shape).copy()
    hits, ids, _ = mesh.ray.intersects_location(origins, directions, multiple_hits=False)
    axial = np.zeros(len(indices), dtype=np.float64)
    axial[ids] = ((hits - origins[ids]) * directions[ids]).sum(1) / (directions[ids] ** 2).sum(1)
    truth = (axial > 0) & (axial < 10) & (np.rint(axial * 65535 / 10) > 0)
    predicted = encoded[v, u] > 0
    decoded = encoded[v, u].astype(float) * 10 / 65535
    errors = np.abs(axial - decoded)
    missed = int((truth & ~predicted).sum())
    spurious = int((predicted & ~truth).sum())
    # Preserve the v1 strict center-ray test everywhere except positive pixels
    # on the already-defined one-pixel silhouette boundary. A grazing triangle
    # can amplify a micrometre surface error into millimetres of axial depth.
    # Such a pixel must independently pass BOTH the unchanged metric surface
    # tolerance and the first-hit interval across its own half-pixel footprint.
    boundary = predicted & ~interior[v, u]
    ambiguous = boundary & ((~truth) | (errors >= tolerance))
    footprints = []
    for i in np.flatnonzero(ambiguous):
        points = origins[i : i + 1] + directions[i : i + 1] * decoded[i]
        _, distance, _ = mesh.nearest.on_surface(points)
        surface_error = float(distance[0])
        offsets = np.array([(x, y) for y in [-0.5, 0, 0.5] for x in [-0.5, 0, 0.5]])
        pixel = offsets + np.array([u[i], v[i]])
        cone = (
            np.column_stack(
                ((pixel[:, 0] - k[0, 2]) / k[0, 0], (pixel[:, 1] - k[1, 2]) / k[1, 1], np.ones(9))
            )
            @ camera[:3, :3].T
        )
        org = np.broadcast_to(camera[:3, 3], cone.shape).copy()
        locations, ray_ids, _ = mesh.ray.intersects_location(org, cone, multiple_hits=False)
        depths = ((locations - org[ray_ids]) * cone[ray_ids]).sum(1) / (cone[ray_ids] ** 2).sum(1)
        depths = depths[(depths > 0) & (depths < 10)]
        passed = bool(
            len(depths)
            and surface_error < tolerance
            and depths.min() - tolerance <= decoded[i] <= depths.max() + tolerance
        )
        record = {
            "pixel_uv": [int(u[i]), int(v[i])],
            "center_axial_error_m": float(errors[i]),
            "surface_error_m": surface_error,
            "first_hit_interval_m": [float(depths.min()), float(depths.max())]
            if len(depths)
            else None,
            "passed": passed,
        }
        footprints.append(record)
        assert passed, {"sample_key": sample_key, "boundary_footprint_failed": record}
    strict = ~ambiguous
    assert not np.any((truth != predicted) & strict), {
        "missed_first_hits": missed,
        "spurious_first_hits": spurious,
        "sample_key": sample_key,
    }
    error = float(errors[truth & strict].max()) if (truth & strict).any() else 0.0
    assert error < tolerance, (sample_key, error)
    return {
        "rays": len(indices),
        "positive_rays": int(predicted.sum()),
        "background_rays": int((~predicted).sum()),
        "uniform_boundary_pixels_excluded": int((~stable).sum()),
        "max_axial_error_m": error,
        "max_unadjusted_center_axial_error_m": float(errors[predicted | truth].max())
        if (predicted | truth).any()
        else 0.0,
        "boundary_footprint_checks": footprints,
        "strict_center_ray_count": int(strict.sum()),
        "missed_first_hits": int(((truth & ~predicted) & strict).sum()),
        "spurious_first_hits": int(((predicted & ~truth) & strict).sum()),
    }


def checks():
    import trimesh

    mesh = trimesh.creation.box(extents=[0.03, 0.03, 0.002])
    mesh.apply_translation([0, 0, 0.07])
    rear = trimesh.creation.box(extents=[0.03, 0.03, 0.002])
    rear.apply_translation([0, 0, 0.3])
    mesh = trimesh.util.concatenate([mesh, rear])
    k = np.array([[200, 0, 159.5], [0, 200, 119.5], [0, 0, 1]], float)
    camera = np.eye(4)
    v, u = np.mgrid[:240, :320]
    front = 0.069
    mask = (np.abs((u - k[0, 2]) / 200 * front) <= 0.015) & (
        np.abs((v - k[1, 2]) / 200 * front) <= 0.015
    )
    encoded = np.where(mask, np.rint(front * 65535 / 10), 0).astype(np.uint16)
    result = verify_view(mesh, camera, k, encoded, "analytic-check")
    assert result["positive_rays"] > 0 and result["background_rays"] > 0
    for defect in ["clipped_foreground", "back_surface", "invented_background"]:
        bad = encoded.copy()
        if defect == "clipped_foreground":
            bad[:] = 0
        elif defect == "back_surface":
            bad[mask] = np.rint(0.299 * 65535 / 10)
        else:
            bad[~mask] = np.rint(0.3 * 65535 / 10)
        try:
            verify_view(mesh, camera, k, bad, "analytic-check")
        except AssertionError:
            pass
        else:
            raise AssertionError("Missed defect: " + defect)
    print(
        "Independent ray checker catches clipped foreground, wrong first surface "
        "and invented background",
        flush=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    checks()
