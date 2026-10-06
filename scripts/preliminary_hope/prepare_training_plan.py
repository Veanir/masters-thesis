"""Outcome-independent pairing for the frozen RaySt3R adaptation protocol."""

import hashlib
import json

import numpy as np
from PIL import Image, ImageEnhance

from scripts.common.paths import script_help

script_help(__doc__, __name__)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def schedule(seed, count=36, epochs=60, views=21):
    rng = np.random.default_rng(seed)
    rows = []
    for epoch in range(epochs):
        for ordinal in rng.permutation(count):
            rows.append(
                {
                    "step": len(rows) + 1,
                    "epoch": epoch,
                    "ordinal": int(ordinal),
                    "views": rng.choice(views, 2, replace=False).tolist(),
                    "augment": bool((epoch + int(ordinal)) % 2),
                    "variant": (epoch // 2) % 2,
                    "augmentation_seed": int.from_bytes(
                        hashlib.sha256(f"ray-rgb-v1/{seed}/{len(rows)}".encode()).digest()[:8],
                        "big",
                    ),
                }
            )
    return rows


def classic_rgb(rgb, mask, seed):
    rng = np.random.default_rng(seed)
    image = Image.fromarray(rgb)
    for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        image = enhancer(image).enhance(float(rng.uniform(0.8, 1.2)))
    value = np.asarray(image, dtype=np.float32) / 255
    value = np.rint(255 * value ** rng.uniform(0.9, 1.1)).clip(0, 255).astype(np.uint8)
    value[~mask.astype(bool)] = 0
    return value


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)
