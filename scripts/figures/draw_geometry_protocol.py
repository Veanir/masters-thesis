"""Post-study illustrations of two preselected, frozen HB observations.

No fitting, smoothing, mesh reconstruction or score selection is performed.
"""

import hashlib
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from scipy.spatial import cKDTree

from scripts.common.paths import ROOT as WORKSPACE_ROOT
from scripts.common.paths import script_help
from scripts.eval.surface_metrics import select_points

script_help(__doc__, __name__)


ROOT = WORKSPACE_ROOT

matplotlib.use("Agg")


OUT = ROOT / "artifacts/figures"
E = ROOT / "runs/research-evolution-evaluation-20260907"


def read(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def bound(ref):
    p = ROOT / ref["path"]
    assert sha(p) == ref["sha256"]
    return p


plt.rcParams.update(
    {"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "pdf.fonttype": 42}
)
job = read(E / "final-scoring-job-20260909-v6/job.json")
proto = read(bound(job["protocol"]))
gp = bound(proto["references"]["GT_manifest"])
gtrows = {r["sample_id"]: r for r in read(gp)["rows"]}
cp = bound(job["cache"])
cr = {r["sample_id"]: r for r in read(cp)["rows"]}
scorep = E / "final-scores-20260909-v6/per-observation.json"
assert sha(scorep) == read(scorep.parent / "complete.json")["per_observation_sha256"]
sr = {r["sample_id"]: r for r in read(scorep)}
cmap = LinearSegmentedColormap.from_list(
    "distance", ["#224b77", "#5c9eb4", "#f0dfab", "#dc8538", "#a8332e"]
)
norm = Normalize(0, 10, clip=True)
ledger = {
    "script_sha256": sha(Path(__file__)),
    "selection": read(OUT / "evidence/reconstruction-selection.json"),
    "sources": {
        "job": sha(E / "final-scoring-job-20260909-v6/job.json"),
        "scores": sha(scorep),
        "GT_manifest": sha(gp),
        "cache_manifest": sha(cp),
    },
    "observations": [],
}


def basis(az, el):
    az, el = np.radians([az, el])
    toward = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    right = np.array([-np.sin(az), np.cos(az), 0])
    up = np.cross(toward, right)
    return np.array([right, up, toward]).T


def xyz(p):
    return p[:, [0, 2, 1]] * np.array([1, -1, 1])


views = [basis(-65, 18), basis(25, 18)]

for obj in [1, 33, 2]:
    sid = (
        next(
            s
            for s in proto["sample_ids"]
            if gtrows[s]["visibility_bin"] == "high" and gtrows[s]["obj_id"] not in [1, 33]
        )
        if obj == 2
        else next(s for s in proto["sample_ids"] if s.startswith(f"hb-obj{obj:06d}-"))
    )
    gfile = gp.parent / (sid + ".npz")
    cfile = cp.parent / (sid + ".npz")
    assert sha(gfile) == gtrows[sid]["output_sha256"]
    assert sha(cfile) == cr[sid]["cache_sha256"]
    with np.load(gfile) as z:
        g = {
            k: z[k]
            for k in ["rgb", "mask", "depth_sensor_m", "mesh_vertices_camera_m", "mesh_faces"]
        }
    with np.load(cfile) as z:
        truth = z["truth_points_camera_m"]
        dense = z["input_dense_points_camera_m"]
    native = {"INPUT": dense}
    hashes = {"GT_observation": sha(gfile), "cache": sha(cfile)}
    for a in ["BASE", "PHOTO"]:
        pp = bound(job["prediction_manifests"][f"ray-{a}-seed0"])
        rows = {r["sample_id"]: r for r in read(pp)["rows"]}
        p = pp.parent / (sid + ".npz")
        assert sha(p) == rows[sid]["sha256"]
        hashes[a] = sha(p)
        with np.load(p) as z:
            native[a] = z["points_camera_m"]
    points = {a: select_points(p, 16384, sample_id=sid) for a, p in native.items()}
    distances = {a: cKDTree(truth).query(p, workers=-1)[0] * 1000 for a, p in points.items()}
    metrics = {}
    for a, p in points.items():
        pr = float((distances[a] <= 5).mean())
        rec = float((cKDTree(p).query(truth, workers=-1)[0] <= 0.005).mean())
        f = 2 * pr * rec / (pr + rec) if pr + rec else 0
        mid = "INPUT" if a == "INPUT" else f"ray-{a}-seed0"
        stored = sr[sid]["methods"][mid]["scores"]["cap16384"]["surface"]["0.005"]["fscore"]
        assert abs(f - stored) < 1e-12, (a, f, stored)
        metrics[a] = {
            "F5": stored,
            "count": len(p),
            "native_count": len(native[a]),
            "above10mm": int((distances[a] > 10).sum()),
        }
    vertices = xyz(g["mesh_vertices_camera_m"])
    faces = g["mesh_faces"]
    center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
    vertices -= center
    display = {a: xyz(p) - center for a, p in points.items()}
    projected = [np.vstack([vertices] + list(display.values())) @ b for b in views]
    lo = np.min([p[:, :2].min(axis=0) for p in projected], axis=0)
    hi = np.max([p[:, :2].max(axis=0) for p in projected], axis=0)
    middle = (lo + hi) / 2
    span = max(hi - lo) * 1.10
    limits = [
        middle[0] - span / 2,
        middle[0] + span / 2,
        middle[1] - span / 2,
        middle[1] + span / 2,
    ]
    fig = plt.figure(figsize=(6.3, 6.5))
    grid = fig.add_gridspec(3, 4, height_ratios=[1.1, 1.25, 1.25], hspace=0.30, wspace=0.05)
    # The full RGB and a separate magnified input preserve scene context and target identity.
    rgb = g["rgb"]
    mask = g["mask"].astype(bool)
    yy, xx = np.where(mask)
    x0, x1 = max(0, xx.min() - 25), min(rgb.shape[1], xx.max() + 26)
    y0, y1 = max(0, yy.min() - 25), min(rgb.shape[0], yy.max() + 26)
    ax = fig.add_subplot(grid[0, :2])
    ax.imshow(rgb)
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#f6d958", lw=1.1))
    ax.set_title("(a) Rzeczywisty pomiar RGB", loc="left")
    ax.axis("off")
    ax = fig.add_subplot(grid[0, 2])
    ax.imshow(rgb[y0:y1, x0:x1] * mask[y0:y1, x0:x1, None])
    ax.set_title("RGB celu")
    ax.axis("off")
    ax = fig.add_subplot(grid[0, 3])
    dep = np.where(mask, g["depth_sensor_m"], np.nan)
    dep = np.where(dep > 0, dep, np.nan)
    ax.imshow(dep[y0:y1, x0:x1], cmap="viridis")
    ax.set_title("Głębia celu")
    ax.axis("off")
    face_xyz = vertices[faces]
    normal = np.cross(face_xyz[:, 1] - face_xyz[:, 0], face_xyz[:, 2] - face_xyz[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    for vi, b in enumerate(views):
        for col, a in enumerate(["GT", "INPUT", "BASE", "PHOTO"]):
            ax = fig.add_subplot(grid[vi + 1, col])
            ax.set_aspect("equal")
            ax.set_xlim(limits[:2])
            ax.set_ylim(limits[2:])
            ax.axis("off")
            if a == "GT":
                projected_faces = face_xyz @ b
                order = np.argsort(projected_faces[:, :, 2].mean(axis=1))
                light = b[:, 2] + 0.5 * b[:, 1] - 0.2 * b[:, 0]
                light /= np.linalg.norm(light)
                intensity = 0.27 + 0.73 * np.maximum(normal @ light, 0)
                color = np.clip(
                    np.array([0.62, 0.75, 0.82])[None, :] * intensity[:, None] + 0.10, 0, 1
                )
                ax.add_collection(
                    PolyCollection(
                        projected_faces[order, :, :2],
                        facecolors=color[order],
                        edgecolors="none",
                        rasterized=True,
                    )
                )
                ax.set_title(f"GT · rzut {vi + 1}")
                ax.plot(
                    [limits[0] + 0.012, limits[0] + 0.062],
                    [limits[2] + 0.013] * 2,
                    color="#222",
                    lw=1.6,
                )
                ax.text(limits[0] + 0.037, limits[2] + 0.020, "5 cm", ha="center", fontsize=8)
            else:
                p = display[a] @ b
                order = np.argsort(p[:, 2])
                ax.scatter(
                    p[order, 0],
                    p[order, 1],
                    c=distances[a][order],
                    s=0.30,
                    cmap=cmap,
                    norm=norm,
                    lw=0,
                    rasterized=True,
                )
                f = f"{100 * metrics[a]['F5']:.2f}".replace(".", ",")
                ax.set_title(f"{a}\nF5 = {f}%", linespacing=1.3)
    fig.subplots_adjust(left=0.018, right=0.985, top=0.958, bottom=0.115)
    cax = fig.add_axes([0.34, 0.058, 0.49, 0.018])
    bar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
        orientation="horizontal",
        ticks=[0, 5, 10],
    )
    bar.ax.set_xticklabels(["0", "5", "≥10"])
    bar.set_label("Odległość punktu od GT [mm]", labelpad=2)
    name = f"results-geometry-hb{obj:02d}"
    for ext in ["pdf", "png"]:
        fig.savefig(OUT / f"assets/{name}.{ext}", dpi=300)
    plt.close(fig)
    ledger["observations"].append(
        {
            "sample_id": sid,
            "object_id": obj,
            "hashes": hashes,
            "metrics": metrics,
            "camera_azimuth_elevation_degrees": [[-65, 18], [25, 18]],
            "projection": "orthographic, shared full bounding square across methods and views",
            "coordinate_display": (
                "(x,-z,y) from GL camera; center from GT bounding box shared by all panels"
            ),
            "limits_m": limits,
            "no_points_clipped": True,
            "artifacts": {ext: sha(OUT / f"assets/{name}.{ext}") for ext in ["pdf", "png"]},
        }
    )
    print(name, metrics)
# Separate failure panel: the missing output is a defined input-eligibility case.
sid = next(s for s in proto["sample_ids"] if s.startswith("hb-obj000017-"))
gfile = gp.parent / (sid + ".npz")
assert sha(gfile) == gtrows[sid]["output_sha256"]
with np.load(gfile) as z:
    rgb = z["rgb"]
    mask = z["mask"].astype(bool)
assert not gtrows[sid]["input_eligible"]
for a in ["BASE", "CLASSIC", "PHOTO"]:
    assert sr[sid]["methods"][f"ray-{a}-seed0"]["native_count"] == 0
yy, xx = np.where(mask)
x0, x1 = max(0, xx.min() - 12), min(rgb.shape[1], xx.max() + 13)
y0, y1 = max(0, yy.min() - 12), min(rgb.shape[0], yy.max() + 13)
fig = plt.figure(figsize=(6.3, 2.15))
grid = fig.add_gridspec(1, 3, width_ratios=[1.6, 0.85, 1.3], wspace=0.15)
ax = fig.add_subplot(grid[0, 0])
ax.imshow(rgb)
ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#f6d958", lw=1))
ax.set_title("Scena RGB", loc="left")
ax.axis("off")
ax = fig.add_subplot(grid[0, 1])
ax.imshow(rgb[y0:y1, x0:x1] * mask[y0:y1, x0:x1, None])
ax.set_title("RGB wejścia")
ax.axis("off")
ax = fig.add_subplot(grid[0, 2])
ax.axis("off")
ax.text(0.04, 0.70, "Brak predykcji", fontsize=12, weight="bold")
ax.text(
    0.04,
    0.47,
    f"{gtrows[sid]['input_pixels']} punktów wejścia\nPróg modelu: 512",
    fontsize=10,
    linespacing=1.6,
)
ax.text(0.04, 0.14, "BASE · CLASSIC · PHOTO\nF5 = 0%", fontsize=10, linespacing=1.6)
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.86)
for ext in ["pdf", "png"]:
    fig.savefig(OUT / f"assets/results-insufficient.{ext}", dpi=300)
plt.close(fig)
ledger["insufficient_input"] = {
    "sample_id": sid,
    "GT_observation_sha256": sha(gfile),
    "input_pixels": gtrows[sid]["input_pixels"],
    "model_minimum": 512,
    "prediction_count": 0,
    "F5": 0,
    "selection": (
        "First observation of object17 in frozen protocol, seed0; rule "
        "recorded with initial selection."
    ),
    "artifacts": {ext: sha(OUT / f"assets/results-insufficient.{ext}") for ext in ["pdf", "png"]},
}
(OUT / "evidence/geometry-figures.json").write_text(
    json.dumps(ledger, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
