"""Post-study galleries from frozen observations and predictions; CPU only.

Inventory/preview do not load PHOTO or public-model predictions. Selection is
recorded before comparative rendering. No training, inference or data repair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from scipy.spatial import cKDTree

from scripts.common.paths import ROOT as WORKSPACE_ROOT
from scripts.eval.surface_metrics import select_points

ROOT = WORKSPACE_ROOT

matplotlib.use("Agg")


WORK = ROOT / "build/thesis-qualitative-supplement"
EVAL = ROOT / "runs/research-evolution-evaluation-20260907"
INK, BLUE = "#243444", "#246899"
VIEWS = [(-65, 18), (25, 18)]
VIS_PL = {"low": "mała", "medium": "średnia", "high": "duża"}
CMAP = LinearSegmentedColormap.from_list(
    "surface_distance", ["#224b77", "#5c9eb4", "#f0dfab", "#dc8538", "#a8332e"]
)
NORM = Normalize(0, 10, clip=True)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 9,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "text.color": INK,
        "savefig.facecolor": "white",
    }
)


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def put(p, data):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def rel(p):
    return Path(p).resolve().relative_to(ROOT).as_posix()


def basis(az, el):
    az, el = np.radians([az, el])
    toward = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    right = np.array([-np.sin(az), np.cos(az), 0])
    return np.array([right, np.cross(toward, right), toward]).T


def xyz(points, cv=False):
    """One rigid display transform: GL -> (x,-z,y), CV -> (x,z,-y)."""
    return np.asarray(points)[:, [0, 2, 1]] * np.array([1, 1, -1] if cv else [1, -1, 1])


def crop_box(mask, pad=0.22):
    y, x = np.where(mask)
    assert len(x)
    side = int(np.ceil(max(x.max() - x.min() + 1, y.max() - y.min() + 1) * (1 + 2 * pad)))
    side = min(side, *mask.shape)
    left = int(np.clip((x.min() + x.max() + 1 - side) / 2, 0, mask.shape[1] - side))
    top = int(np.clip((y.min() + y.max() + 1 - side) / 2, 0, mask.shape[0] - side))
    return [left, top, left + side, top + side]


def crop(a, box):
    x0, y0, x1, y1 = box
    return a[y0:y1, x0:x1]


def off(ax):
    ax.set_axis_off()


def rgb_panel(ax, rgb, mask=None):
    ax.imshow(rgb, interpolation="nearest")
    if mask is not None:
        y, x = np.where(mask)
        ax.add_patch(
            Rectangle(
                (x.min(), y.min()),
                x.max() - x.min() + 1,
                y.max() - y.min() + 1,
                fill=False,
                edgecolor="#f7bd36",
                lw=1.1,
            )
        )
    off(ax)


def depth_panel(ax, depth, mask, *, label=True):
    valid = mask & np.isfinite(depth) & (depth > 0)
    assert valid.any()
    lo, hi = float(depth[valid].min()), float(depth[valid].max())
    cm = plt.get_cmap("viridis").copy()
    cm.set_bad("#edf0f3")
    ax.imshow(np.ma.array(depth, mask=~valid), cmap=cm, vmin=lo, vmax=hi, interpolation="nearest")
    off(ax)
    if label:
        ax.text(
            0.5,
            -0.07,
            f"{lo:.2f}–{hi:.2f} m".replace(".", ","),
            transform=ax.transAxes,
            ha="center",
            fontsize=9,
        )
    return [lo, hi]


def mesh_panel(ax, vertices, faces, *, view=VIEWS[0], limits=None, center=None, cv=False):
    v = xyz(vertices, cv=cv)
    if center is None:
        center = (v.min(0) + v.max(0)) / 2
    v = v - center
    b = basis(*view)
    tri = v[faces]
    norm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm /= np.maximum(np.linalg.norm(norm, axis=1, keepdims=True), 1e-12)
    light = b[:, 2] + 0.5 * b[:, 1] - 0.2 * b[:, 0]
    light /= np.linalg.norm(light)
    intensity = 0.30 + 0.70 * np.abs(norm @ light)
    color = np.clip(np.array([0.62, 0.75, 0.82])[None, :] * intensity[:, None] + 0.10, 0, 1)
    projected = tri @ b
    order = np.argsort(projected[:, :, 2].mean(1))
    ax.add_collection(
        PolyCollection(
            projected[order, :, :2], facecolors=color[order], edgecolors="none", rasterized=True
        )
    )
    if limits is None:
        p = v @ b
        middle = (p[:, :2].min(0) + p[:, :2].max(0)) / 2
        half = max(np.ptp(p[:, :2], axis=0)) * 0.56
        limits = [middle[0] - half, middle[0] + half, middle[1] - half, middle[1] + half]
    ax.set_xlim(limits[:2])
    ax.set_ylim(limits[2:])
    ax.set_aspect("equal")
    off(ax)


class Sources:
    def __init__(self):
        self.verified = {}
        self.job = read(self.check(EVAL / "final-scoring-job-20260909-v6/job.json"))
        self.protocol = read(self.bound(self.job["protocol"]))
        self.gp = self.bound(self.protocol["references"]["GT_manifest"])
        self.cp = self.bound(self.job["cache"])
        self.gt = {r["sample_id"]: r for r in read(self.gp)["rows"]}
        self.cache = {r["sample_id"]: r for r in read(self.cp)["rows"]}
        sp = EVAL / "final-scores-20260909-v6/per-observation.json"
        self.scores = {
            r["sample_id"]: r
            for r in read(
                self.check(sp, read(sp.parent / "complete.json")["per_observation_sha256"])
            )
        }
        self.pred_rows = {}
        self.training = None

    def check(self, p, expected=None):
        p = Path(p)
        key = rel(p)
        h = self.verified.get(key, {}).get("sha256") or sha(p)
        assert expected is None or h == expected, (p, h, expected)
        self.verified[key] = {"sha256": h, "expected_sha256": expected}
        return p

    def bound(self, ref):
        return self.check(ROOT / ref["path"], ref["sha256"])

    def hb(self, sid, cache=False):
        row = self.gt[sid]
        p = self.check(self.gp.parent / (sid + ".npz"), row["output_sha256"])
        with np.load(p) as z:
            a = {k: z[k] for k in z.files}
        assert int(a["mask"].sum()) == row["input_pixels"]
        if cache:
            p = self.check(self.cp.parent / (sid + ".npz"), self.cache[sid]["cache_sha256"])
            with np.load(p) as z:
                c = {k: z[k] for k in z.files}
            y, x = np.where(a["mask"])
            d = a["depth_m"][y, x]
            fx, fy, cx, cy = a["intrinsics"]
            expected = np.column_stack(((x - cx) * d / fx, -(y - cy) * d / fy, -d))
            assert np.array_equal(expected, c["input_dense_points_camera_m"])
            assert len(expected) == row["input_pixels"]
            return a, c
        return a

    def synthetic_index(self):
        if self.training is not None:
            return
        cp = ROOT / "runs/research-evolution-campaign-20260907/frozen-campaign-v3/campaign.json"
        campaign = read(self.check(cp))
        refs = campaign["references"]
        self.training = {
            r["sample_id"]: r for r in read(self.bound(refs["training_manifest"]))["rows"]
        }
        source = read(self.bound(refs["source_complete"]))
        self.index = {}
        for item in source["sources"]:
            folder = ROOT / item["source"].replace("\\", "/")
            for row in read(self.check(folder / "complete.json", item["manifest_sha256"]))["rows"]:
                sid = row["sample_id"]
                if sid in self.training:
                    assert (
                        row["files"]["observation.npz"]
                        == self.training[sid]["source_observation_sha256"]
                    )
                    self.index[sid] = (folder, row)
        assert len(self.index) == len(self.training) == 6400

    def synthetic(self, sid):
        self.synthetic_index()
        folder, row = self.index[sid]
        p = self.check(folder / sid / "observation.npz", row["files"]["observation.npz"])
        with np.load(p) as z:
            a = {k: z[k] for k in z.files}
        return a

    def prediction(self, method, sid):
        if method not in self.pred_rows:
            p = self.bound(self.job["prediction_manifests"][method])
            self.pred_rows[method] = (p.parent, {r["sample_id"]: r for r in read(p)["rows"]})
        parent, rows = self.pred_rows[method]
        p = self.check(parent / (sid + ".npz"), rows[sid]["sha256"])
        with np.load(p) as z:
            a = z["points_camera_m"]
        assert a.ndim == 2 and a.shape[1] == 3 and np.isfinite(a).all()
        return a


def inventory(src, out):
    eligible = [
        r
        for r in src.scores.values()
        if r["input_eligible"] and r["source_registration_status"] == "passed"
    ]
    assert len(eligible) == 125
    slots = []
    for vis in ("low", "medium", "high"):
        pool = []
        for r in eligible:
            if r["visibility_bin"] != vis:
                continue
            f = statistics.mean(
                r["methods"][f"ray-BASE-seed{k}"]["scores"]["cap16384"]["surface"]["0.005"][
                    "fscore"
                ]
                for k in range(3)
            )
            pool.append({"sample_id": r["sample_id"], "obj_id": r["obj_id"], "BASE_F5_mean": f})
        pool.sort(key=lambda r: (r["BASE_F5_mean"], r["sample_id"]))
        for i, r in enumerate(pool):
            r["rank_fraction"] = i / (len(pool) - 1)
        for q in (0.25, 0.75):
            ranked = sorted(pool, key=lambda r: (abs(r["rank_fraction"] - q), r["sample_id"]))
            slots.append(
                {
                    "slot": f"{vis}-q{int(q * 100)}",
                    "visibility_bin": vis,
                    "target": q,
                    "ranked": ranked,
                }
            )
    src.synthetic_index()
    models = defaultdict(list)
    for sid, r in src.training.items():
        models[r["key"]].append(
            {
                "sample_id": sid,
                "family_id": r["family_id"],
                "source_mask_pixels": src.index[sid][1]["source_frame"]["mask_pixels"],
            }
        )
    for rows in models.values():
        median = statistics.median(r["source_mask_pixels"] for r in rows)
        rows.sort(key=lambda r: (abs(r["source_mask_pixels"] - median), r["sample_id"]))
    result = {
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "candidates_not_final_selection",
        "HB_slots": slots,
        "synthetic_models": dict(models),
        "verified_files": src.verified,
        "selection_uses": (
            "BASE mean over three runs only; no PHOTO/public performance or prediction appearance"
        ),
    }
    put(out / "inventory.json", result)
    print("Inventoried 125 eligible HB observations and 320 training models.", flush=True)
    return result


def preview(src, out, inv, group, offset=0, only_slot=None):
    out.mkdir(parents=True, exist_ok=True)
    if group == "hb":
        for slot in inv["HB_slots"]:
            if only_slot and slot["slot"] != only_slot:
                continue
            fig, axs = plt.subplots(4, 4, figsize=(9, 10.8))
            for i, r in enumerate(slot["ranked"][offset : offset + 4]):
                a = src.hb(r["sample_id"])
                box = crop_box(a["mask"], 0.35)
                rgb_panel(axs[i, 0], crop(a["rgb"], box))
                rgb_panel(axs[i, 1], crop(a["rgb"] * a["mask"][:, :, None], box))
                depth_panel(axs[i, 2], crop(a["depth_m"], box), crop(a["mask"], box))
                mesh_panel(axs[i, 3], a["mesh_vertices_camera_m"], a["mesh_faces"])
                axs[i, 0].set_title(
                    f"#{offset + i + 1} HB {r['obj_id']} | {r['sample_id'].split('-scene')[1]}",
                    fontsize=8,
                    loc="left",
                )
            fig.suptitle(slot["slot"] + " | RGB / RGB maskowane / głębia / GT", fontsize=12)
            fig.subplots_adjust(
                left=0.02, right=0.99, top=0.94, bottom=0.03, hspace=0.30, wspace=0.06
            )
            fig.savefig(
                out / f"preview-{slot['slot']}{'-offset' + str(offset) if offset else ''}.png",
                dpi=150,
            )
            plt.close(fig)
    else:
        keys = [
            "gso/Cole_Hardware_Hammer_Black",
            "gso/SCHOOL_BUS",
            "gso/Nintendo_2DS_Crimson_Red",
            "gso/Dino_3",
        ]
        for category in ("02876657", "03797390"):
            keys += sorted(
                k for k in inv["synthetic_models"] if k.startswith("shapenet/" + category)
            )[:2]
        for start in range(0, len(keys), 4):
            fig, axs = plt.subplots(4, 4, figsize=(9, 10.8))
            for i, key in enumerate(keys[start : start + 4]):
                sid = inv["synthetic_models"][key][0]["sample_id"]
                a = src.synthetic(sid)
                box = crop_box(a["source_target_mask"])
                mesh_panel(axs[i, 0], a["mesh_vertices_camera_cv_m"], a["mesh_faces"], cv=True)
                rgb_panel(axs[i, 1], a["rgb"], a["source_target_mask"])
                rgb_panel(axs[i, 2], crop(a["rgb"] * a["input_mask"][:, :, None], box))
                depth = a["input_depth_uint16"].astype(float) * 10 / 65535
                depth_panel(axs[i, 3], crop(depth, box), crop(a["input_mask"], box))
                axs[i, 0].set_title(
                    key.replace("gso/", "").replace("shapenet/", "")[:43], fontsize=8, loc="left"
                )
            fig.suptitle("Syntetyczne: GT / scena / RGB wejścia / głębia wejścia", fontsize=12)
            fig.subplots_adjust(
                left=0.02, right=0.99, top=0.94, bottom=0.03, hspace=0.30, wspace=0.06
            )
            fig.savefig(out / f"preview-synthetic-{start // 4 + 1}.png", dpi=150)
            plt.close(fig)
    print("Saved input-only previews:", group, flush=True)


def select(src, out, inv):
    """Record the input-only review, before any comparative render."""
    path = out / "selection.json"
    if path.exists():
        raise FileExistsError("Selection already recorded; do not silently replace it.")
    choices = [
        (
            4,
            "Telefon",
            [
                "Widać dwa odłączone fragmenty opakowania za innymi przedmiotami.",
                "Wąski bok misia nie pozwala rozpoznać jego kształtu.",
                "Mały czerwony fragment pojazdu jest nierozpoznawalny.",
                "Mały niebieski fragment samochodu nie pokazuje wyraźnego obrysu ani kół.",
            ],
        ),
        (
            2,
            "Samochód wyścigowy",
            [
                "Figurka niemal całkowicie zasłonięta psem; dwa niewielkie fragmenty.",
                "Ciemny, wąski fragment za psem nie pozwala rozpoznać części.",
            ],
        ),
        (1, "Część z otworem", ["Widok od tyłu i z góry ukrywa charakterystyczne cechy figurki."]),
        (0, "Królik", []),
        (0, "Element profilowany", []),
        (0, "Wiertarko-wkrętarka", []),
    ]
    hb = []
    used = set()
    for letter, slot, (index, name, reasons) in zip(
        "ABCDEF", inv["HB_slots"], choices, strict=False
    ):
        assert index == len(reasons)
        chosen = dict(slot["ranked"][index])
        assert chosen["obj_id"] not in used
        used.add(chosen["obj_id"])
        src.hb(chosen["sample_id"])
        chosen.update(
            letter=letter,
            name=name,
            slot=slot["slot"],
            visibility_bin=slot["visibility_bin"],
            target=slot["target"],
            candidate_position=index + 1,
            rejected=[
                dict(r, reason=reason)
                for r, reason in zip(slot["ranked"][:index], reasons, strict=False)
            ],
        )
        hb.append(chosen)
    keys = [
        ("gso/Cole_Hardware_Hammer_Black", "Młotek", 0),
        ("gso/SCHOOL_BUS", "Autobus", 0),
        ("shapenet/02876657-20c5dff3b09282035b76194468418a0e", "Butelka", 0),
        ("shapenet/03797390-46955fddcc83a50f79b586547e543694", "Kubek", 2),
    ]
    syn = []
    for key, name, index in keys:
        r = dict(inv["synthetic_models"][key][index])
        src.synthetic(r["sample_id"])
        r.update(
            key=key,
            name=name,
            source="GSO" if key.startswith("gso/") else "ShapeNet",
            candidate_position=index + 1,
            inspected_family_observations=20,
            selection_reason="Pierwszy czytelny widok w kolejności od mediany pola maski.",
            rejected=[
                dict(x, reason="Ucho kubka jest ukryte za korpusem w tym ujęciu.")
                for x in inv["synthetic_models"][key][:index]
            ],
        )
        syn.append(r)
    put(
        path,
        {
            "recorded_utc": datetime.now(UTC).isoformat(),
            "baseline_commit": "ee67e54",
            "status": "fixed_before_comparative_prediction_render",
            "HB": hb,
            "synthetic": syn,
            "performance_ranking_used": True,
            "ranking": (
                "BASE cap16384 F5 mean over three runs; nearest relative ranks "
                ".25/.75 within visibility; unique objects; input-only "
                "readability review."
            ),
            "PHOTO_or_public_performance_used_for_selection": False,
            "training_seed_for_display": 0,
            "views_azimuth_elevation_degrees": VIEWS,
            "verified_files_at_selection": src.verified,
        },
    )
    print("Fixed selection:", ", ".join(r["letter"] + " " + r["name"] for r in hb), flush=True)


def metric(src, sid, method, points, truth, budget):
    forward = cKDTree(truth).query(points, workers=2)[0] if len(points) else np.empty(0)
    backward = (
        cKDTree(points).query(truth, workers=2)[0] if len(points) else np.full(len(truth), np.inf)
    )
    p = float(np.mean(forward <= 0.005)) if len(points) else 0.0
    r = float(np.mean(backward <= 0.005))
    f = 2 * p * r / (p + r) if p + r else 0.0
    actual = dict(precision=p, recall=r, fscore=f)
    stored = src.scores[sid]["methods"][method]["scores"][budget]
    assert len(points) == stored["count"], (sid, method, budget, len(points), stored["count"])
    for k, v in actual.items():
        assert abs(v - stored["surface"]["0.005"][k]) <= 1e-12, (sid, method, budget, k, v)
    return dict(
        actual,
        count=len(points),
        native_count=src.scores[sid]["methods"][method]["native_count"],
        score_locator=f"sample_id={sid}/methods/{method}/scores/{budget}/surface/0.005",
    ), forward * 1000


def panel(fig, x, top, w, h):
    """Physical millimetres measured from the top-left of a 150-mm figure."""
    fw, fh = fig.get_size_inches() * 25.4
    return fig.add_axes([x / fw, 1 - (top + h) / fh, w / fw, h / fh])


def note(fig, x, top, text, **kwargs):
    fw, fh = fig.get_size_inches() * 25.4
    return fig.text(
        x / fw, 1 - top / fh, text, va="top", fontsize=kwargs.pop("fontsize", 9), **kwargs
    )


def depth_key(ax, limits):
    bar = ax.inset_axes([0.13, -0.055, 0.74, 0.045])
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(*limits), cmap="viridis"),
        cax=bar,
        orientation="horizontal",
    )
    cb.set_ticks(limits)
    cb.set_ticklabels([f"{v:.2f}".replace(".", ",") for v in limits])
    cb.ax.tick_params(length=1.5, pad=1, labelsize=9)
    cb.outline.set_visible(False)


def save_figure(fig, name, out, record):
    files = {}
    for ext in ("pdf", "png"):
        p = out / (name + "." + ext)
        fig.savefig(p, dpi=320)
        files[rel(p)] = sha(p)
    plt.close(fig)
    record.update(
        files=files, physical_size_mm=list(fig.get_size_inches() * 25.4), minimum_font_pt=9
    )
    return record


def data_gallery(src, selection, out, synthetic):
    rows = selection["synthetic"] if synthetic else [selection["HB"][i] for i in (1, 2, 5)]
    height = 4 + len(rows) * 49 + 7
    fig = plt.figure(figsize=(150 / 25.4, height / 25.4))
    records = []
    for i, r in enumerate(rows):
        top = 2 + i * 49
        sid = r["sample_id"]
        a = src.synthetic(sid) if synthetic else src.hb(sid)
        note(
            fig,
            1,
            top,
            (r["name"] + " · " + r["source"])
            if synthetic
            else f"{r['letter']}  {r['name']} · {VIS_PL[r['visibility_bin']]} widoczność",
            color=BLUE,
            weight="bold",
            fontsize=10,
        )
        titles = (
            ["Pełny kształt GT", "Scena RGB", "RGB wejścia", "Głębia [m]"]
            if synthetic
            else ["Scena RGB", "RGB celu", "Głębia [m]", "Pełny kształt GT"]
        )
        axs = []
        for j, title in enumerate(titles):
            note(fig, j * 38 + 18, top + 6, title, ha="center")
            axs.append(panel(fig, j * 38, top + 11, 35, 32))
        mask = a["source_target_mask"] if synthetic else a["mask"]
        box = crop_box(mask)
        depth = a["input_depth_uint16"].astype(float) * 10 / 65535 if synthetic else a["depth_m"]
        valid = a["input_mask"] if synthetic else a["mask"]
        mesh_panel(
            axs[0 if synthetic else 3],
            a["mesh_vertices_camera_cv_m" if synthetic else "mesh_vertices_camera_m"],
            a["mesh_faces"],
            cv=synthetic,
        )
        rgb_panel(axs[1 if synthetic else 0], a["rgb"], mask)
        rgb_panel(
            axs[2 if synthetic else 1],
            crop(a["rgb"] * valid[:, :, None] if synthetic else a["rgb"], box),
        )
        ax = axs[3 if synthetic else 2]
        lim = depth_panel(ax, crop(depth, box), crop(valid, box), label=False)
        depth_key(ax, lim)
        records.append(
            dict(
                sample_id=sid,
                crop_xyxy=box,
                depth_range_m=lim,
                valid_input_pixels=int(valid.sum()),
                label=r.get("letter", r["name"]),
            )
        )
    note(fig, 1, height - 4, "Głębia: szary — brak wartości.  Żółta ramka wskazuje cel w scenie.")
    return save_figure(
        fig,
        "data-synthetic-gallery" if synthetic else "data-hb-gallery",
        out,
        {"observations": records},
    )


def comparison(src, rows, out, name, public=False):
    methods = (
        [
            ("ray-pretrained", "RaySt3R"),
            ("octmae-original", "OctMAE"),
            ("ray-BASE-seed0", "BASE"),
            ("ray-PHOTO-seed0", "PHOTO"),
        ]
        if public
        else [
            ("INPUT", "INPUT"),
            ("ray-BASE-seed0", "BASE"),
            ("ray-CLASSIC-seed0", "CLASSIC"),
            ("ray-PHOTO-seed0", "PHOTO"),
        ]
    )
    budget = "shared_nonfailed_cap" if public else "cap16384"
    fig = plt.figure(figsize=(150 / 25.4, 221 / 25.4))
    records = []
    for i, r in enumerate(rows):
        top = 1 + i * 102
        sid = r["sample_id"]
        a, cache = src.hb(sid, cache=True)
        cap = src.scores[sid]["shared_nonfailed_point_cap"] if public else 16384
        clouds = {}
        metrics = {}
        colors = {}
        for mid, label in methods:
            native = (
                cache["input_dense_points_camera_m"] if mid == "INPUT" else src.prediction(mid, sid)
            )
            clouds[label] = select_points(native, cap, sample_id=sid)
            metrics[label], colors[label] = metric(
                src, sid, mid, clouds[label], cache["truth_points_camera_m"], budget
            )
            metrics[label]["method"] = mid
        center = (
            xyz(a["mesh_vertices_camera_m"]).min(0) + xyz(a["mesh_vertices_camera_m"]).max(0)
        ) / 2
        allpts = (
            np.concatenate([xyz(a["mesh_vertices_camera_m"])] + [xyz(v) for v in clouds.values()])
            - center
        )
        projected = [(allpts @ basis(*v))[:, :2] for v in VIEWS]
        aspect = 28 / 20
        half = max(max(np.ptp(p, axis=0) / [aspect, 1]) for p in projected) * 0.54
        viewport_centers = [(p.min(0) + p.max(0)) / 2 for p in projected]
        view_limits = [
            [c[0] - half * aspect, c[0] + half * aspect, c[1] - half, c[1] + half]
            for c in viewport_centers
        ]
        box = crop_box(a["mask"], 0.28)
        note(
            fig,
            1,
            top,
            f"{r['letter']}  {r['name']} · {VIS_PL[r['visibility_bin']]} widoczność",
            weight="bold",
            color=BLUE,
            fontsize=10,
        )
        for j, title in enumerate(["RGB celu", "RGB po maskowaniu", "Głębia wejścia [m]"]):
            note(fig, 25 + j * 50, top + 5.5, title, ha="center")
            ax = panel(fig, 2 + j * 50, top + 10, 46, 31)
            if j == 2:
                lim = depth_panel(ax, crop(a["depth_m"], box), crop(a["mask"], box), label=False)
                depth_key(ax, lim)
            else:
                rgb_panel(ax, crop(a["rgb"] if j == 0 else a["rgb"] * a["mask"][:, :, None], box))
        for j, label in enumerate([x[1] for x in methods] + ["GT"]):
            note(fig, j * 30 + 15, top + 47, label, ha="center", weight="bold")
            if label != "GT":
                note(
                    fig,
                    j * 30 + 15,
                    top + 51,
                    "F5 " + f"{metrics[label]['fscore'] * 100:.1f}".replace(".", ",") + "%",
                    ha="center",
                )
            else:
                length = next(v for v in [0.05, 0.02, 0.01, 0.005, 0.002, 0.001] if v <= half * 0.7)
                note(
                    fig,
                    j * 30 + 15,
                    top + 50.9,
                    f"{length * 100:g} cm".replace(".", ","),
                    ha="center",
                )
                bar = panel(fig, j * 30 + 1, top + 54.3, 28, 0.5)
                length_mm = length / (2 * half / 20)
                bar.plot([14 - length_mm / 2, 14 + length_mm / 2], [0.5, 0.5], color=INK, lw=1.4)
                bar.set_xlim(0, 28)
                bar.set_ylim(0, 1)
                off(bar)
            for k, view in enumerate(VIEWS):
                limits = view_limits[k]
                ax = panel(fig, j * 30 + 1, top + 55 + k * 20, 28, 20)
                if label == "GT":
                    mesh_panel(
                        ax,
                        a["mesh_vertices_camera_m"],
                        a["mesh_faces"],
                        view=view,
                        limits=limits,
                        center=center,
                    )
                else:
                    pts = (xyz(clouds[label]) - center) @ basis(*view)
                    order = np.argsort(pts[:, 2])
                    ax.scatter(
                        pts[order, 0],
                        pts[order, 1],
                        c=colors[label][order],
                        cmap=CMAP,
                        norm=NORM,
                        s=0.15,
                        edgecolors="none",
                        rasterized=True,
                    )
                    ax.set_xlim(limits[:2])
                    ax.set_ylim(limits[2:])
                    ax.set_aspect("equal")
                    off(ax)
                    if not len(pts):
                        ax.text(
                            0.5,
                            0.5,
                            "Brak\npredykcji",
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                        )
        counts = " · ".join(
            label + " " + f"{m['count']:,}".replace(",", " ") for label, m in metrics.items()
        )
        note(fig, 1, top + 96, "Punkty: " + counts, fontsize=9)
        records.append(
            dict(
                sample_id=sid,
                label=r["letter"],
                name=r["name"],
                budget=budget,
                cap=cap,
                metrics=metrics,
                crop_xyxy=box,
                depth_range_m=lim,
                center_display_m=center.tolist(),
                limits_per_view_m=view_limits,
                common_metres_per_mm=2 * half / 20,
                scale_bar_m=length,
                all_selected_points_inside_bounds=True,
            )
        )
    note(fig, 1, 206, "Odległość punktu od GT [mm]")
    bar = panel(fig, 91, 206, 53, 2.5)
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=NORM, cmap=CMAP), cax=bar, orientation="horizontal"
    )
    cb.set_ticks([0, 5, 10])
    cb.set_ticklabels(["0", "5", "≥10"])
    cb.ax.tick_params(length=2, pad=1, labelsize=9)
    cb.outline.set_visible(False)
    note(fig, 1, 214, "Dwa rzuty tych samych punktów.  Głębia: szary — brak wartości.")
    return save_figure(fig, name, out, {"observations": records, "point_rule": budget})


def render(src, out, selection, only=None):
    folder = out / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    figures = {}
    if only is None:
        figures["data-synthetic-gallery"] = data_gallery(src, selection, folder, True)
        figures["data-hb-gallery"] = data_gallery(src, selection, folder, False)
    for i in range(3):
        name = f"results-hb-matrix-{i + 1:02}"
        if only is None or only == name:
            figures[name] = comparison(src, selection["HB"][i * 2 : i * 2 + 2], folder, name)
    if only is None or only == "results-public-models":
        figures["results-public-models"] = comparison(
            src, [selection["HB"][i] for i in (2, 5)], folder, "results-public-models", public=True
        )
    put(
        out / ("trial-ledger.json" if only else "qualitative-figures.json"),
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "script": rel(Path(__file__)),
            "script_sha256": sha(__file__),
            "selection_path": rel(out / "selection.json"),
            "selection_sha256": sha(out / "selection.json"),
            "figures": figures,
            "source_files": src.verified,
            "rendering": {
                "views_degrees": VIEWS,
                "GL_display_transform": "(x,-z,y)",
                "CV_display_transform": "(x,z,-y)",
                "projection": (
                    "orthographic; one GT-based 3D translation; equal physical scale "
                    "for all methods and both views; viewport bounds include every "
                    "selected point and GT; each view framed around its joint bounds, "
                    "no per-method alignment"
                ),
                "mesh_lighting": (
                    "double-sided Lambert shading; original vertices and triangles; "
                    "no geometry alteration"
                ),
                "point_sampling": (
                    "scripts.eval.surface_metrics.select_points; frozen sample-id keyed subset"
                ),
                "point_error_colors_mm": [0, 10],
                "prediction_smoothing_or_trimming": False,
                "depth": (
                    "actual filtered/quantized input in metres; missing values gray; "
                    "RGB missing black"
                ),
                "metrics": (
                    "independent bidirectional nearest-neighbour F5, precision and "
                    "recall checked against frozen cap-specific scores, abs tolerance "
                    "1e-12"
                ),
            },
        },
    )
    print("Rendered and independently checked:", ", ".join(figures), flush=True)


def verify(src, out):
    ledger = read(out / "qualitative-figures.json")
    selection = read(out / "selection.json")
    assert len({r["obj_id"] for r in selection["HB"]}) == 6
    assert Counter(r["source"] for r in selection["synthetic"]) == {"GSO": 2, "ShapeNet": 2}
    assert sha(out / "selection.json") == ledger["selection_sha256"]
    assert sha(__file__) == ledger["script_sha256"]
    for path, rec in ledger["source_files"].items():
        assert sha(ROOT / path) == rec["sha256"], path
    checks = 0
    for figure in ledger["figures"].values():
        for path, digest in figure["files"].items():
            assert sha(ROOT / path) == digest, path
        for r in figure["observations"]:
            if "metrics" not in r:
                continue
            a, cache = src.hb(r["sample_id"], cache=True)
            for m in r["metrics"].values():
                native = (
                    cache["input_dense_points_camera_m"]
                    if m["method"] == "INPUT"
                    else src.prediction(m["method"], r["sample_id"])
                )
                pts = select_points(native, r["cap"], sample_id=r["sample_id"])
                actual, _ = metric(
                    src,
                    r["sample_id"],
                    m["method"],
                    pts,
                    cache["truth_points_camera_m"],
                    r["budget"],
                )
                assert all(actual[k] == m[k] for k in actual)
                checks += 1
    put(
        out / "verification.json",
        {
            "status": "passed",
            "figures": len(ledger["figures"]),
            "independently_recomputed_cloud_metrics": checks,
            "metric_components_per_cloud": 3,
            "source_files_checked": len(ledger["source_files"]),
            "ledger_sha256": sha(out / "qualitative-figures.json"),
        },
    )
    print(
        "Verified",
        checks,
        "clouds, all figure/source hashes and selection constraints.",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["inventory", "preview", "select", "render", "verify"])
    parser.add_argument("--output-directory", type=Path, default=WORK)
    parser.add_argument("--group", choices=["hb", "synthetic"], default="hb")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--slot")
    parser.add_argument("--figure")
    args = parser.parse_args()
    out = args.output_directory.resolve()
    src = Sources()
    if args.operation == "inventory":
        inventory(src, out)
    elif args.operation == "preview":
        preview(src, out, read(out / "inventory.json"), args.group, args.offset, args.slot)
    elif args.operation == "select":
        select(src, out, read(out / "inventory.json"))
    elif args.operation == "render":
        render(src, out, read(out / "selection.json"), args.figure)
    else:
        verify(src, out)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
