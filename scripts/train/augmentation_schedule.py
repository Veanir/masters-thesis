"""Outcome-independent matched exposures and spatially unchanged CLASSIC RGB."""

import hashlib

import numpy as np
from PIL import Image, ImageEnhance

from scripts.common.paths import script_help

script_help(__doc__, __name__)


def schedule(sample_ids, accepted_photo, seed, epochs=4, novel_views=21):
    assert len(sample_ids) == len(set(sample_ids)) and epochs > 0 and epochs % 2 == 0
    assert set(accepted_photo) <= set(sample_ids)
    rng = np.random.default_rng(seed)
    rows = []
    for epoch in range(epochs):
        for ordinal in rng.permutation(len(sample_ids)):
            sid = sample_ids[int(ordinal)]
            attempt = bool((epoch + int(ordinal)) % 2)
            rows.append(
                {
                    "step": len(rows) + 1,
                    "epoch": epoch,
                    "sample_id": sid,
                    "ordinal": int(ordinal),
                    "views": rng.choice(novel_views, 2, replace=False).tolist(),
                    "augmentation_attempt": attempt,
                    "effective_augmentation": bool(attempt and sid in accepted_photo),
                    "augmentation_seed": int.from_bytes(
                        hashlib.sha256(
                            f"evolution-classic-v1/{seed}/{len(rows) + 1}".encode()
                        ).digest()[:8],
                        "big",
                    ),
                }
            )
    return rows


def classic_rgb(rgb, seed):
    """Apply fixed photometric transforms to full RGB; context belongs to wrapper."""
    assert rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3
    rng = np.random.default_rng(seed)
    image = Image.fromarray(rgb)
    for enhancer in [ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color]:
        image = enhancer(image).enhance(float(rng.uniform(0.8, 1.2)))
    value = np.asarray(image, dtype=np.float32) / 255
    return np.rint(255 * value ** rng.uniform(0.9, 1.1)).clip(0, 255).astype(np.uint8)


def checks():
    ids = [f"synthetic-{i:03}" for i in range(40)]
    accepted = set(ids[::3])
    first = None
    for seed in [0, 1, 2]:
        rows = schedule(ids, accepted, seed)
        assert len(rows) == 160
        for sid in ids:
            values = [r for r in rows if r["sample_id"] == sid]
            assert len(values) == 4 and sum(r["augmentation_attempt"] for r in values) == 2
            assert sum(r["effective_augmentation"] for r in values) == (2 if sid in accepted else 0)
            assert all(
                len(set(r["views"])) == 2 and all(0 <= v < 21 for v in r["views"]) for r in values
            )
        assert rows == schedule(ids, accepted, seed)
        if first is not None:
            assert rows != first
        first = rows
        flags = {
            arm: [r["effective_augmentation"] and arm != "BASE" for r in rows]
            for arm in ["BASE", "CLASSIC", "PHOTO"]
        }
        assert flags["CLASSIC"] == flags["PHOTO"] and not any(flags["BASE"])
    rgb = np.tile(np.array([90, 130, 180], np.uint8), (24, 32, 1))
    previous = rgb.copy()
    edited = classic_rgb(rgb, 1701)
    assert edited.dtype == np.uint8 and edited.shape == rgb.shape and np.array_equal(rgb, previous)
    assert (
        np.all(edited == edited[0, 0])
        and np.all(edited[0, 0] > 0)
        and not np.array_equal(edited, rgb)
    )
    assert np.array_equal(edited, classic_rgb(rgb, 1701))
    print(
        "Pairing verified:4 exposures/object, exactly2 attempts, "
        "identical CLASSIC/PHOTO fallback masks; photometric fullRGB "
        "retains spatial arrangement."
    )


if __name__ == "__main__":
    checks()
