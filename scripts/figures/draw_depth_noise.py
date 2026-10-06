"""Illustrate frozen YCB-Video depth statistics using the original 42 measurements.

CPU-only post-study illustration; never modifies experimental artifacts.
preview: contact sheet of source RGB crops. render: verifies all measurements,
writes the manuscript figure and extends its evidence record.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from PIL import Image, ImageDraw
from scipy import ndimage

from scripts.common.paths import DATASET_ROOT
from scripts.common.paths import ROOT as WORKSPACE_ROOT

ROOT = WORKSPACE_ROOT

matplotlib.use("Agg")

DATA = DATASET_ROOT / "bop-ycbv-real-v1/ycbv"
SOURCE = ROOT / "runs/ray-ycbv-observations-20260906"
STATS = ROOT / "runs/research-evolution-depth-model-20260907/development-statistics-v2.json"
WORK = ROOT / "build/thesis-depth-estimation"
ASSETS = ROOT / "artifacts/figures/assets"
EVIDENCE = ROOT / "artifacts/figures/evidence/depth-estimation-figure.json"
FILES = {}
INK, BLUE, GOLD, RED, GREEN = "#243444", "#246899", "#ba7619", "#bd393c", "#309777"


def read(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rel(p):
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return "../" + p.resolve().relative_to(ROOT.parent).as_posix()


def verify(p, expected=None):
    digest = sha(p)
    if expected is not None:
        assert digest == expected, (p, digest, expected)
    FILES[rel(p)] = {"sha256": digest}
    return p


def load(row, bindings):
    sid = row["sample_id"]
    meta = read(verify(SOURCE / (sid + ".json")))
    prefix = f"test/{meta['scene_id']:06d}"
    paths = {
        kind: f"{prefix}/{folder}/{meta['image_id']:06d}{suffix}"
        for kind, folder, suffix in [
            ("depth", "depth", ".png"),
            ("mask", "mask_visib", f"_{meta['gt_id']:06d}.png"),
            ("rgb", "rgb", ".png"),
        ]
    }
    camera = f"{prefix}/scene_camera.json"
    for p in [*paths.values(), camera]:
        verify(DATA / p, bindings[p])
    scale = read(DATA / camera)[str(meta["image_id"])]["depth_scale"] * 0.001
    depth = np.asarray(Image.open(DATA / paths["depth"]), dtype=np.float64) * scale
    mask = np.asarray(Image.open(DATA / paths["mask"])) > 0
    rgb = np.asarray(Image.open(DATA / paths["rgb"]).convert("RGB"))
    assert sha(DATA / paths["depth"]) == row["raw_depth_sha256"]
    assert sha(DATA / paths["mask"]) == row["raw_visible_mask_sha256"]
    valid = np.isfinite(depth) & (depth > 0)
    interior = ndimage.binary_erosion(mask, iterations=3)
    boundary = mask & ~interior
    patch = ndimage.binary_erosion(interior & valid, iterations=2)
    safe = np.where(valid, depth, 0.0)
    variation = ndimage.maximum_filter(safe, size=5) - ndimage.minimum_filter(safe, size=5)
    patch &= variation < 0.01
    return rgb, depth, mask, valid, interior, boundary, patch, scale


def bbox(mask):
    yy, xx = np.where(mask)
    side = min(max(np.ptp(xx) + 1, np.ptp(yy) + 1) * 1.25, min(mask.shape))
    x = int(np.clip((xx.min() + xx.max() + 1 - side) / 2, 0, mask.shape[1] - side))
    y = int(np.clip((yy.min() + yy.max() + 1 - side) / 2, 0, mask.shape[0] - side))
    return x, y, x + int(side), y + int(side)


def crop(a, box):
    x, y, r, b = box
    return a[y:b, x:r]


def check_row(row, arrays):
    _, depth, mask, valid, interior, boundary, patch, scale = arrays
    inverse = np.divide(1.0, depth, out=np.zeros_like(depth), where=valid)
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=float)
    residual = ndimage.convolve(inverse, kernel, mode="nearest")[patch]
    mad = float(np.median(abs(residual - np.median(residual))))
    values = np.unique(depth[mask & valid])
    steps = np.diff(values)
    steps = steps[steps > 1e-7]
    actual = {
        "source_depth_step_m": scale,
        "valid_target_pixels": int((mask & valid).sum()),
        "mask_pixels": int(mask.sum()),
        "median_depth_m": float(np.median(depth[mask & valid])),
        "interior_invalid_rate": float((interior & ~valid).sum() / interior.sum()),
        "boundary_invalid_rate": float((boundary & ~valid).sum() / boundary.sum()),
        "inverse_depth_noise_coefficient_per_m": mad / (0.6744897501960817 * np.sqrt(20)),
        "plane_patch_pixels": int(patch.sum()),
        "observed_lower_step_m": float(np.median(steps[steps <= np.quantile(steps, 0.1) * 1.5])),
    }
    for key, value in actual.items():
        assert abs(value - row[key]) <= 1e-12, (row["sample_id"], key, value, row[key])
    return actual


def preview(stats, bindings):
    sheet = Image.new("RGB", (7 * 205, 6 * 228), "white")
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(stats["rows"]):
        a = load(row, bindings)
        thumb = Image.fromarray(crop(a[0], bbox(a[2]))).resize((195, 195))
        x, y = i % 7 * 205, i // 7 * 228
        sheet.paste(thumb, (x, y + 23))
        draw.text((x + 2, y + 4), f"{i:02d} | object {row['obj_id']}", fill="black")
    sheet.save(WORK / "contact.png")
    print(WORK / "contact.png")


def dots(values):
    # Deterministic spreading of tied/nearby values; x has no numerical meaning.
    xs = np.zeros(len(values))
    occupied = []
    for i in np.argsort(values, kind="stable"):
        for offset in [0, 0.08, -0.08, 0.16, -0.16, 0.24, -0.24, 0.32, -0.32]:
            if all(
                abs(values[i] - v) > (max(values) - min(values)) * 0.032 or abs(offset - x) >= 0.079
                for v, x in occupied
            ):
                break
        xs[i] = offset
        occupied.append((values[i], offset))
    return xs


def render(stats, bindings, index):
    chosen = None
    recomputed = []
    for i, row in enumerate(stats["rows"]):
        arrays = load(row, bindings)
        recomputed.append(check_row(row, arrays))
        if i == index:
            chosen = arrays
    assert len(recomputed) == 42 and chosen is not None
    for key, expected in stats["medians_across_observations"].items():
        assert abs(np.median([r[key] for r in recomputed]) - expected) <= 1e-12
    row = stats["rows"][index]
    rgb, depth, mask, valid, interior, boundary, patch, _ = chosen
    box = bbox(mask)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.6,
            "axes.titlesize": 9.6,
            "axes.labelsize": 9.6,
            "xtick.labelsize": 9.6,
            "ytick.labelsize": 9.6,
            "pdf.fonttype": 42,
            "text.color": INK,
            "axes.labelcolor": INK,
        }
    )
    fig = plt.figure(figsize=(160 / 25.4, 163 / 25.4), facecolor="white")
    top_y, top_h = 0.635, 0.29
    axes = [fig.add_axes([x, top_y, 0.292, top_h]) for x in [0.01, 0.353, 0.696]]
    for ax, title in zip(
        axes, ["a  RGB i kontur celu", "b  Głębia [m]", "c  Obszary analizy"], strict=False
    ):
        ax.set_title(title, loc="left", pad=8)
        ax.set_axis_off()
    axes[0].imshow(crop(rgb, box))
    axes[0].contour(crop(mask, box), [0.5], colors=GOLD, linewidths=0.8)
    d = crop(depth, box)
    m = crop(mask, box)
    v = crop(valid, box)
    cmap = matplotlib.colormaps["viridis"].copy()
    cmap.set_bad("#edf0f2")
    lo, hi = (
        np.floor(depth[mask & valid].min() * 100) / 100,
        np.ceil(depth[mask & valid].max() * 100) / 100,
    )
    im = axes[1].imshow(
        np.ma.masked_where(~(m & v), d), cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest"
    )
    overlay = np.zeros((*m.shape, 4))
    overlay[m & ~v] = matplotlib.colors.to_rgba(RED)
    axes[1].imshow(overlay, interpolation="nearest")
    categories = np.zeros(mask.shape, dtype=int)
    categories[interior] = 1
    categories[boundary] = 2
    categories[patch] = 3
    categories[mask & ~valid] = 4
    cc = matplotlib.colors.ListedColormap(["#edf0f2", "#b7c4ce", GOLD, GREEN, RED])
    axes[2].imshow(crop(categories, box), cmap=cc, vmin=0, vmax=4, interpolation="nearest")
    cax = fig.add_axes([0.37, 0.613, 0.26, 0.012])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[lo, (lo + hi) / 2, hi])
    cb.ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.2f}".replace(".", ",")))
    cb.outline.set_visible(False)
    handles = [
        Patch(color=RED, label="brak pomiaru"),
        Patch(color=GOLD, label="pas graniczny"),
        Patch(color="#b7c4ce", label="pozostałe wnętrze"),
        Patch(color=GREEN, label="fragmenty do estymacji szumu"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.558),
        ncol=2,
        frameon=False,
        fontsize=9.6,
        handlelength=1.1,
        columnspacing=1.6,
        labelspacing=0.6,
    )
    left = fig.add_axes([0.11, 0.104, 0.39, 0.31])
    right = fig.add_axes([0.65, 0.104, 0.32, 0.31])
    left.set_title("d  Braki w 42 obserwacjach", loc="left", pad=11)
    right.set_title("e  Oszacowany szum", loc="left", pad=11)
    for ax in [left, right]:
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e3e7eb", linewidth=0.6)
    median_values = stats["medians_across_observations"]
    for j, (key, color) in enumerate(
        [("interior_invalid_rate", BLUE), ("boundary_invalid_rate", GOLD)]
    ):
        vals = np.array([r[key] * 100 for r in stats["rows"]])
        med = median_values[key] * 100
        left.scatter(
            j + dots(vals) * 0.7, vals, s=14, color=color, alpha=0.78, linewidths=0, zorder=3
        )
        left.plot([j - 0.33, j + 0.33], [med, med], color=INK, lw=1.8, zorder=4)
        left.text(j + 0.35, med, f"{med:.1f}%".replace(".", ","), va="center", fontsize=9.6)
    left.set(
        xticks=[0, 1],
        xticklabels=["Wnętrze", "Brzeg"],
        xlim=(-0.45, 1.85),
        ylim=(0, 80),
        ylabel="Brakujące pomiary [%]",
    )
    key = "inverse_depth_noise_coefficient_per_m"
    vals = np.array([r[key] * 1e4 for r in stats["rows"]])
    med = median_values[key] * 1e4
    right.scatter(dots(vals), vals, s=14, color=BLUE, alpha=0.78, linewidths=0, zorder=3)
    right.plot([-0.34, 0.34], [med, med], color=INK, lw=1.8)
    right.text(0.39, med, f"{med:.2f}".replace(".", ","), va="center", fontsize=9.6)
    right.set(
        xlim=(-0.65, 1.0),
        ylim=(-0.3, 7),
        yticks=range(8),
        xticks=[],
        ylabel=r"$c$ [$10^{-4}$ m$^{-1}$]",
    )
    fig.text(0.65, 0.063, "Punkt = obserwacja", fontsize=9.6)
    fig.text(
        0.11,
        0.023,
        "Czarne kreski: mediany. Krok głębi: 1 mm we wszystkich 42 obserwacjach.",
        fontsize=9.6,
    )
    files = {}
    for ext in ["pdf", "png"]:
        p = ASSETS / ("data-depth-estimation." + ext)
        fig.savefig(
            p,
            dpi=350,
            metadata={"Creator": "figures/draw_depth_noise.py"} if ext == "pdf" else None,
        )
        files[rel(p)] = sha(p)
    plt.close(fig)
    ledger = {
        "evidence_class": "poststudy_illustration",
        "script": rel(Path(__file__)),
        "script_sha256": sha(Path(__file__)),
        "source_files": FILES,
        "population": {
            "observations": 42,
            "objects": len({r["obj_id"] for r in stats["rows"]}),
            "dataset": "YCB-Video",
        },
        "selection": {
            "index": index,
            "sample_id": row["sample_id"],
            "rule": (
                "One source RGB crop selected visually for object "
                "recognizability, a readable mask and depth structure without an "
                "extreme range obscuring the surface. All 42 observations appear "
                "in the plots; no reconstruction outcome used."
            ),
            "bbox_xyxy": box,
            "example_statistics": row,
        },
        "figure": {"id": "data-depth-estimation", "files": files, "depth_range_m": [lo, hi]},
        "verification": {
            "status": "passed",
            "recomputed_observations": len(recomputed),
            "checked_statistics_per_observation": len(recomputed[0]),
            "absolute_tolerance": 1e-12,
            "medians": median_values,
            "raw_source_hashes_verified": True,
        },
        "limitations": (
            "Illustration of development statistics, not a physical sensor "
            "calibration. The local estimator also includes quantization, "
            "curvature and spatial correlation; no clean real-world GT is "
            "assumed."
        ),
    }
    EVIDENCE.write_text(
        json.dumps(ledger, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(ledger["verification"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["preview", "render"])
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    plan = read(
        verify(ROOT / "runs/research-evolution-data-20260907/source-final-plan-v1/plan.json")
    )
    stats = read(verify(STATS, plan["statistics_sha256"]))
    verify(ROOT / "scripts/data/estimate_depth_noise_ycbv.py", stats["script_sha256"])
    binding = read(verify(SOURCE / "binding.json"))["source_sha256"]
    if args.operation == "preview":
        preview(stats, binding)
    else:
        if args.index is None:
            parser.error("--index is required for render")
        render(stats, binding, args.index)


if __name__ == "__main__":
    main()
