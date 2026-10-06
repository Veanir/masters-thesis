"""Draw original didactic figures; no measured data or experimental predictions.

Run with .venv/Scripts/python.exe scripts/figures/draw_surface_and_rgbd_theory.py.
Both PDFs have a physical width of 160 mm and embedded vector text.
The hand-defined 2D geometry is an explanatory schematic, without metric scale.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle

from scripts.common.paths import ROOT as WORKSPACE_ROOT
from scripts.common.paths import script_help

script_help(__doc__, __name__)


ROOT = WORKSPACE_ROOT

matplotlib.use("Agg")

OUT = ROOT / "artifacts/figures/assets"
EVIDENCE = ROOT / "artifacts/figures/evidence/theory-figures.json"
WIDTH = 160 / 25.4
INK = "#243444"
BLUE = "#246899"
GOLD = "#a86512"
GREY = "#74818c"
GREEN = "#237c5b"
RED = "#b03b3e"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "text.color": INK,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "mathtext.fontset": "dejavusans",
    }
)


def text(ax, x, y, value, **kwargs):
    return ax.text(x, y, value, fontsize=9, va="center", **kwargs)


def setup(ax, xlim, ylim):
    ax.set(xlim=xlim, ylim=ylim)
    ax.set_aspect("equal")
    ax.axis("off")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(fig, name):
    # Fixed figure size is intentional: tight bounding boxes would change width.
    fig.savefig(
        OUT / f"{name}.pdf",
        metadata={
            "Title": name,
            "Author": "Opracowanie własne",
            "Subject": "Autorski schemat dydaktyczny, bez danych eksperymentalnych",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    return {
        ext: {
            "path": (OUT / f"{name}.{ext}").relative_to(ROOT).as_posix(),
            "sha256": digest(OUT / f"{name}.{ext}"),
        }
        for ext in ("pdf", "png")
    }


def visible_front(ax, x, y0, y1):
    yy = np.linspace(y0, y1, 7)
    ax.plot([x, x], [y0, y1], color=BLUE, lw=2.8, zorder=5)
    ax.scatter(
        np.full_like(yy, x), yy, s=17, color=BLUE, edgecolors="white", linewidths=0.6, zorder=6
    )


def rgbd_figure():
    fig = plt.figure(figsize=(WIDTH, 72 / 25.4))
    left = fig.add_axes([0.028, 0.12, 0.452, 0.86])
    right = fig.add_axes([0.523, 0.12, 0.452, 0.86])
    for ax in (left, right):
        setup(ax, (0, 6), (0.15, 4.65))
    left.text(0, 4.45, "a) Obserwacja RGB-D", fontsize=10, weight="bold")
    right.text(0, 4.45, "b) Możliwe kształty", fontsize=10, weight="bold")

    left.add_patch(Rectangle((2.62, 1.5), 2.95, 2.1, facecolor="#f2f4f5", edgecolor="none"))
    left.plot([2.62, 5.57, 5.57, 2.62], [3.6, 3.6, 1.5, 1.5], color=GREY, lw=1.3, ls=(0, (3, 3)))
    # Rays terminate at the first surface: this is the information measured.
    camera = np.array([0.68, 2.55])
    for yy in np.linspace(1.5, 3.6, 7):
        left.plot([camera[0], 2.62], [camera[1], yy], lw=0.85, color="#b1c5d4", zorder=1)
    left.add_patch(
        Rectangle((0.08, 2.35), 0.48, 0.40, facecolor=INK, edgecolor=INK, lw=0.7, zorder=4)
    )
    left.add_patch(
        Polygon(
            [[0.50, 2.39], [0.77, 2.25], [0.77, 2.85], [0.50, 2.71]],
            facecolor=INK,
            edgecolor=INK,
            zorder=4,
        )
    )
    text(left, 0.44, 1.97, "kamera", ha="center")
    visible_front(left, 2.62, 1.5, 3.6)
    text(left, 3.68, 4.02, "pierwsza powierzchnia", ha="center", color=BLUE)
    left.annotate(
        "",
        xy=(2.66, 3.58),
        xytext=(3.15, 3.92),
        arrowprops={"arrowstyle": "-", "lw": 0.8, "color": BLUE},
    )
    text(left, 4.16, 2.57, "część niewidoczna\nw pomiarze", ha="center", linespacing=1.45)
    text(
        left,
        3.0,
        0.74,
        "RGB opisuje wygląd; głębia\nokreśla położenie punktów.",
        ha="center",
        linespacing=1.45,
    )

    for y0, y1, x1, label in [(2.55, 3.6, 3.45, "Pudełko A"), (1.00, 2.05, 5.35, "Pudełko B")]:
        right.add_patch(
            Rectangle((1.05, y0), x1 - 1.05, y1 - y0, facecolor="#fbf3e7", edgecolor="none")
        )
        right.plot([1.05, x1, x1, 1.05], [y1, y1, y0, y0], color=GOLD, lw=1.65, ls=(0, (5, 2)))
        visible_front(right, 1.05, y0, y1)
        text(right, (x1 + 1.05) / 2, (y0 + y1) / 2, label, ha="center", color=GOLD)
    text(right, 3.0, 0.48, "Ten sam przód, inna głębokość.", ha="center")
    fig.text(
        0.5,
        0.034,
        "Przekrój z góry. Tyłu obiektu nie da się odczytać z tego pomiaru.",
        ha="center",
        fontsize=9,
        color=GREY,
    )
    return save(fig, "theory-rgbd")


def nearest(source, target):
    matrix = np.linalg.norm(source[:, None, :] - target[None, :, :], axis=2)
    indices = matrix.argmin(axis=1)
    return indices, matrix[np.arange(len(source)), indices]


def metric_panel(ax, gt, pred, tau, reverse=False):
    setup(ax, (0, 5.55), (0.0, 4.18))
    source, target = (gt, pred) if reverse else (pred, gt)
    title = "b) Pokrycie powierzchni" if reverse else "a) Precyzja punktów"
    direction = "odniesienie → wynik" if reverse else "wynik → odniesienie"
    ax.text(0, 4.0, title, fontsize=10, weight="bold")
    text(ax, 2.7, 3.55, direction, ha="center")
    for q in target:
        ax.add_patch(
            Circle(
                q, tau, facecolor="#eef1f3", edgecolor="#a4b2bc", lw=0.7, ls=(0, (2, 2)), zorder=0
            )
        )
    indices, distances = nearest(source, target)
    for point, idx, distance in zip(source, indices, distances, strict=False):
        ax.annotate(
            "",
            xy=target[idx],
            xytext=point,
            arrowprops={
                "arrowstyle": "-|>",
                "mutation_scale": 8,
                "lw": 1.3,
                "shrinkA": 2.5,
                "shrinkB": 2.5,
                "color": GREEN if distance <= tau else RED,
            },
            zorder=3,
        )
    ax.scatter(
        gt[:, 0], gt[:, 1], marker="s", s=24, c=INK, edgecolor="white", linewidth=0.5, zorder=4
    )
    ax.scatter(
        pred[:, 0],
        pred[:, 1],
        marker="o",
        s=31,
        c=BLUE,
        edgecolor="white",
        linewidth=0.65,
        zorder=5,
    )
    if reverse:
        text(
            ax,
            2.65,
            0.13,
            "Odsetek punktów odniesienia\npokrytych przez wynik.",
            ha="center",
            linespacing=1.45,
        )
        ax.annotate(
            "brak\npokrycia",
            xy=gt[-1],
            xytext=(4.82, 2.64),
            fontsize=9,
            ha="center",
            va="center",
            color=RED,
            arrowprops={"arrowstyle": "-", "lw": 0.8, "color": RED},
        )
    else:
        text(
            ax,
            2.65,
            0.13,
            "Odsetek punktów wyniku\nw tolerancji τ od odniesienia.",
            ha="center",
            linespacing=1.45,
        )
        ax.annotate(
            "punkt błędny",
            xy=pred[3],
            xytext=(4.52, 3.07),
            fontsize=9,
            ha="center",
            va="center",
            color=RED,
            arrowprops={"arrowstyle": "-", "lw": 0.8, "color": RED},
        )
    return {
        "nearest_indices": indices.tolist(),
        "distances": distances.tolist(),
        "within_tolerance": (distances <= tau).tolist(),
    }


def metrics_figure():
    fig = plt.figure(figsize=(WIDTH, 86 / 25.4))
    left = fig.add_axes([0.028, 0.24, 0.452, 0.74])
    right = fig.add_axes([0.523, 0.24, 0.452, 0.74])
    # Deliberately hand-defined, dimensionless point sets. These are not results.
    gt = np.array(
        [[0.65, 1.45], [1.45, 1.75], [2.25, 1.85], [3.05, 1.8], [3.85, 1.55], [4.8, 0.87]]
    )
    pred = np.array([[0.64, 1.87], [1.47, 2.18], [2.48, 2.22], [3.18, 2.91], [4.1, 1.9]])
    tau = 0.46
    forward = metric_panel(left, gt, pred, tau)
    backward = metric_panel(right, gt, pred, tau, reverse=True)

    legend = fig.add_axes([0.035, 0.025, 0.93, 0.15])
    legend.set(xlim=(0, 1), ylim=(0, 1))
    legend.axis("off")
    legend.scatter([0.04], [0.78], s=29, color=BLUE, marker="o")
    text(legend, 0.069, 0.78, "wynik modelu")
    legend.scatter([0.27], [0.78], s=23, color=INK, marker="s")
    text(legend, 0.3, 0.78, "odniesienie")
    legend.plot([0.615, 0.66], [0.78, 0.78], color="#a4b2bc", lw=1, ls=(0, (2, 2)))
    text(legend, 0.68, 0.78, "granica tolerancji τ")
    legend.plot([0.17, 0.21], [0.18, 0.18], color=GREEN, lw=1.5)
    text(legend, 0.23, 0.18, "odległość ≤ τ")
    legend.plot([0.57, 0.61], [0.18, 0.18], color=RED, lw=1.5)
    text(legend, 0.63, 0.18, "odległość > τ")

    outputs = save(fig, "theory-metrics")
    geometry = {
        "type": "dimensionless_hand_defined_2d_points",
        "gt": gt.tolist(),
        "prediction": pred.tolist(),
        "tolerance": tau,
        "prediction_to_gt": forward,
        "gt_to_prediction": backward,
        "warning": "Synthetic schematic coordinates; never use as experimental results.",
    }
    return outputs, geometry


def overview_figure():
    fig = plt.figure(figsize=(WIDTH, 95 / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 160), ylim=(109, 0))
    ax.axis("off")

    def box(x, y, w, h, value, color=GREY, fill="#f4f6f8", bold=False):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0,rounding_size=1.2",
                linewidth=0.85,
                edgecolor=color,
                facecolor=fill,
            )
        )
        ax.text(
            x + w / 2,
            y + h / 2,
            value,
            ha="center",
            va="center",
            fontsize=9,
            weight="bold" if bold else "normal",
            linespacing=1.35,
        )

    def arrow(x, y1, y2):
        ax.annotate(
            "",
            xy=(x, y2),
            xytext=(x, y1),
            arrowprops={"arrowstyle": "-|>", "lw": 0.9, "color": GREY},
        )

    ax.text(4, 4, "UCZENIE NA DANYCH SYNTETYCZNYCH", fontsize=9, weight="bold", va="center")
    box(
        4,
        9,
        152,
        17,
        (
            "Modele 3D przedmiotów → obrazy RGB-D i dane nadzorujące\nGłębia i "
            "maski są wspólne dla trzech wariantów."
        ),
    )
    colors = [BLUE, GOLD, "#88559c"]
    fills = ["#edf4f8", "#fbf4e8", "#f3eef7"]
    for x, name, description, color, fill in zip(
        [4, 57, 110],
        ["BASE", "CLASSIC", "PHOTO"],
        [
            "Oryginalne RGB",
            "Zmiany barw\nlub oryginalne RGB",
            "Generatywne edycje\nlub oryginalne RGB",
        ],
        colors,
        fills,
        strict=False,
    ):
        arrow(x + 23, 26, 32)
        box(x, 32, 46, 20, name + "\n" + description, color, fill, bold=True)
        arrow(x + 23, 52, 58)
        box(x, 58, 46, 14, "Osobne uczenie\nRaySt3R", color)
        arrow(x + 23, 72, 91)
    ax.text(
        80,
        81,
        "OCENA NA RZECZYWISTYCH POMIARACH",
        ha="center",
        va="center",
        fontsize=9,
        weight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 3},
    )
    box(
        4,
        91,
        152,
        15,
        (
            "Te same pomiary RGB-D dla wszystkich gotowych modeli\nPorównanie "
            "odtworzonych punktów z pełnym kształtem odniesienia"
        ),
        BLUE,
        "#edf4f8",
    )
    return save(fig, "study-overview")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    overview = overview_figure()
    rgbd = rgbd_figure()
    metrics, geometry = metrics_figure()
    ledger = {
        "date": "2026-09-11",
        "authorship": "Opracowanie własne",
        "purpose": "Didactic vector figures for the rebuilt thesis; no research results.",
        "data_sources": [],
        "source_type": "original_schematics",
        "script": Path(__file__).relative_to(ROOT).as_posix(),
        "script_sha256": digest(__file__),
        "width_mm": 160,
        "font": "DejaVu Sans",
        "font_size_pt": {"body": 9, "panel_titles": 10},
        "pdf_font_type": 42,
        "png_dpi": 300,
        "figures": {
            "study-overview": {
                "height_mm": 95,
                "outputs": overview,
                "concept": (
                    "Shared synthetic training data, three RGB variants, separate "
                    "learned models and common real RGB-D evaluation."
                ),
                "limitations": (
                    "Conceptual overview; PHOTO admission includes edits without "
                    "individual visual review. It is not an extra experiment or a "
                    "claim of full edit correctness."
                ),
            },
            "theory-rgbd": {
                "height_mm": 72,
                "outputs": rgbd,
                "concept": (
                    "First surface along a camera ray and two different object depths "
                    "consistent with the same measured front."
                ),
                "projection": (
                    "2D top-view cross section; illustrative coordinates, no physical scale."
                ),
                "measured_color": BLUE,
                "hypothesis_color": GOLD,
                "limitations": (
                    "Occlusion by another object and sensor holes are discussed in "
                    "prose, not represented by these simple boxes."
                ),
            },
            "theory-metrics": {
                "height_mm": 86,
                "outputs": metrics,
                "concept": (
                    "Opposite nearest-neighbour distance directions underlying "
                    "precision and recall."
                ),
                "geometry": geometry,
                "limitations": (
                    "Discrete 2D schematic. Dotted disks indicate the tolerance "
                    "neighbourhood of destination points; not object volume or "
                    "measurement confidence."
                ),
            },
        },
        "visual_review": {
            "status": "pending",
            "method": "Inspect Poppler renders of all three generated PDFs before integration.",
        },
    }
    EVIDENCE.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Created study-overview, theory-rgbd and theory-metrics as 160 mm vector PDF + PNG.")


if __name__ == "__main__":
    main()
