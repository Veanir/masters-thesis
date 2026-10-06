"""Draw the camera geometry schematic without regenerating existing figures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.common.paths import script_help
from scripts.figures.draw_surface_and_rgbd_theory import (
    BLUE,
    EVIDENCE,
    GOLD,
    GREY,
    INK,
    WIDTH,
    Polygon,
    digest,
    plt,
    save,
)

script_help(__doc__, __name__)


def arrow(ax, start, end, color=INK, style="-|>", lw=1.0, **kwargs):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": style,
            "color": color,
            "lw": lw,
            "shrinkA": 0,
            "shrinkB": 0,
            **kwargs,
        },
    )


def draw():
    fig = plt.figure(figsize=(WIDTH, 86 / 25.4))
    a = fig.add_axes([0.012, 0.115, 0.57, 0.82])
    b = fig.add_axes([0.62, 0.115, 0.37, 0.82])
    for ax, xmax in [(a, 100), (b, 74)]:
        ax.set(xlim=(0, xmax), ylim=(0, 76), aspect="equal")
        ax.axis("off")
    fig.text(0.025, 0.945, "a) Od piksela do punktu", weight="bold", fontsize=10)
    fig.text(0.62, 0.945, "b) Dwie odległości", weight="bold", fontsize=10)

    # Affine display projection: +x rises slightly right, +y is up, +z left.
    # The virtual image plane is in front of C, so there is no image inversion.
    c = np.array([12.0, 31.0])
    principal = np.array([50.0, 31.0])
    image_x = np.array([15.0, 6.0])
    image_y = np.array([0.0, 18.0])
    corners = [
        principal - image_x - image_y,
        principal + image_x - image_y,
        principal + image_x + image_y,
        principal - image_x + image_y,
    ]
    a.add_patch(Polygon(corners, facecolor="#f1f5f8", edgecolor=GREY, lw=0.9))
    # A light grid and a highlighted cell make the image plane recognizable.
    for t in np.linspace(-0.75, 0.75, 7):
        for start, end in [
            (principal + t * image_x - image_y, principal + t * image_x + image_y),
            (principal - image_x + t * image_y, principal + image_x + t * image_y),
        ]:
            a.plot([start[0], end[0]], [start[1], end[1]], color="#d8e2e9", lw=0.45)
    arrow(a, c, (95, 31), GREY, lw=0.8, linestyle=(0, (4, 3)))
    a.text(82, 25, "oś optyczna\n$-z_C$", ha="center", va="top", fontsize=8.5, color=GREY)
    arrow(a, c, c + [12, 4.8], lw=0.9)
    arrow(a, c, c + [0, 18], lw=0.9)
    arrow(a, c, c + [-10, 0], lw=0.9)
    a.text(23, 39, "$x_C$", fontsize=9)
    a.text(12, 52, "$y_C$", ha="center", fontsize=9)
    a.text(2, 35, "$z_C$", fontsize=9)
    a.scatter(*c, s=25, color=INK, zorder=6)
    a.text(12, 23, "$C$", ha="center", fontsize=10)
    a.text(12, 15, "środek\nkamery", ha="center", va="top", fontsize=8.5)

    pixel = principal + [7.5, 12.0]
    point = c + 1.75 * (pixel - c)
    a.plot([c[0], point[0]], [c[1], point[1]], color=BLUE, lw=1.7, zorder=4)
    dx, dy = image_x / 8, image_y / 8
    a.add_patch(
        Polygon(
            [pixel - dx - dy, pixel + dx - dy, pixel + dx + dy, pixel - dx + dy],
            facecolor="#c8e0f2",
            edgecolor=BLUE,
            lw=1.2,
            zorder=5,
        )
    )
    a.scatter(*pixel, color=BLUE, s=15, zorder=6)
    a.scatter(*point, color=BLUE, s=32, zorder=6)
    a.text(point[0], point[1] + 6, "$\\mathbf{p}_C$", ha="center", fontsize=11, color=BLUE)
    a.annotate(
        "piksel $(u,v)$",
        xy=pixel + [0.5, 2.5],
        xytext=(72, 64),
        fontsize=9,
        ha="center",
        color=BLUE,
        arrowprops={"arrowstyle": "-", "color": BLUE, "lw": 0.65},
    )
    a.scatter(*principal, color=GOLD, marker="+", s=75, lw=1.6, zorder=6)
    a.annotate(
        "punkt główny\n$(c_x,c_y)$",
        xy=principal,
        xytext=(58, 7),
        fontsize=9,
        ha="center",
        va="top",
        color=GOLD,
        arrowprops={"arrowstyle": "-", "color": GOLD, "lw": 0.65, "shrinkA": 3, "shrinkB": 4},
    )
    a.text(41, 69, "płaszczyzna\nobrazu", ha="center", va="center", fontsize=9)

    origin_uv = np.array(corners[3])
    arrow(a, origin_uv, origin_uv + [12, 4.8], GREY, lw=0.9)
    arrow(a, origin_uv, origin_uv + [0, -13], GREY, lw=0.9)
    a.text(origin_uv[0] + 10, origin_uv[1] + 8, "$u$", fontsize=9, color=GREY)
    a.text(origin_uv[0] - 5, origin_uv[1] - 12, "$v$", fontsize=9, color=GREY)

    c2 = np.array([5.0, 24.0])
    p2 = np.array([57.0, 59.0])
    foot = np.array([57.0, 24.0])
    b.add_patch(Polygon([c2, foot, p2], facecolor="#f5f8fb", edgecolor="none"))
    arrow(b, foot, (71, 24), GREY, lw=0.9)
    b.text(70, 28, "$-z_C$", fontsize=9, color=GREY, ha="center")
    b.plot([foot[0], p2[0]], [foot[1], p2[1]], color=GREY, lw=0.9, ls=(0, (3, 3)))
    b.plot([c2[0], foot[0]], [c2[1], foot[1]], color=GOLD, lw=2.1)
    b.plot([c2[0], p2[0]], [c2[1], p2[1]], color=BLUE, lw=2.1)
    b.scatter(*c2, s=25, color=INK, zorder=5)
    b.scatter(*p2, s=32, color=BLUE, zorder=5)
    b.text(c2[0] - 1, c2[1] + 5, "$C$", fontsize=10)
    b.text(p2[0] + 3, p2[1] + 3, "$\\mathbf{p}_C$", fontsize=11, color=BLUE)
    b.plot([53, 53, 57], [24, 28, 28], color=GREY, lw=0.8)
    b.text(31, 17, "głębia osiowa\n$d=-z_C$", ha="center", va="top", color=GOLD, fontsize=10)
    b.text(
        28,
        51,
        "odległość wzdłuż promienia\n$r=\\|\\mathbf{p}_C\\|_2$",
        ha="center",
        va="center",
        fontsize=9,
        linespacing=1.5,
        color=BLUE,
        rotation=np.degrees(np.arctan2(35, 52)),
        rotation_mode="anchor",
    )
    fig.text(
        0.5, 0.035, "Poza osią optyczną: $r>d$. Na osi optycznej: $r=d$.", ha="center", fontsize=9
    )

    # Geometry and sign checks use an illustrative point, not research data.
    fx, fy, cx, cy = 400.0, 420.0, 320.0, 240.0
    u, v, depth = 420.0, 180.0, 0.8
    p = depth * np.array([(u - cx) / fx, -(v - cy) / fy, -1.0])
    assert p[0] > 0 and p[1] > 0 and p[2] < 0
    assert np.isclose(-p[2], depth) and np.linalg.norm(p) > depth
    assert np.isclose(fx * p[0] / (-p[2]) + cx, u)
    assert np.isclose(cy - fy * p[1] / (-p[2]), v)
    assert np.isclose(np.linalg.norm([0, 0, -depth]), depth)
    assert np.allclose((pixel - c) * 1.75, point - c)
    assert np.allclose(pixel - principal, 0.5 * image_x + 0.5 * image_y)
    outputs = save(fig, "theory-camera")
    ledger = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    ledger["figures"]["theory-camera"] = {
        "height_mm": 86,
        "outputs": outputs,
        "script": Path(__file__).relative_to(EVIDENCE.parents[4]).as_posix(),
        "script_sha256": digest(__file__),
        "concept": "Camera unprojection; pixel ray; axial depth versus Euclidean distance.",
        "coordinates": (
            "+x right, +y up, camera looks along -z; image u right, v down; "
            "virtual image plane in front of camera."
        ),
        "mathematical_checks": {
            "pixel_roundtrip": True,
            "axis_signs": True,
            "off_axis_range_exceeds_depth": True,
            "on_axis_equality": True,
            "collinear_pixel_and_point": True,
        },
        "limitations": "Hand-drawn schematic without physical scale or empirical observations.",
        "visual_review": {"status": "pending", "date": "2026-09-14"},
    }
    EVIDENCE.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Created theory-camera; mathematical checks passed; existing figures preserved.")


if __name__ == "__main__":
    draw()
