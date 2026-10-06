"""Read-only, post-study illustrations of frozen thesis data (CPU only).

Selection rules were written to reviews/data-figures-notes.md before composition.
Run with .venv/Scripts/python.exe scripts/figures/draw_dataset_examples.py.
The PDF labels remain vector text; PNG companions are rendered at 350 dpi.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from PIL import Image

from scripts.common.paths import ROOT as WORKSPACE_ROOT

ROOT = WORKSPACE_ROOT

matplotlib.use("Agg")

OUT = ROOT / "artifacts/figures/assets"
EVIDENCE = ROOT / "artifacts/figures/evidence/data-figures.json"
DATA = ROOT / "runs/research-evolution-data-20260907"
PHOTO = ROOT / "runs/research-evolution-photo-20260907/final-admission-v2"
ABO = ROOT.parent / "output/abo-discovery-360-v1"
TRAIN_ABO = ROOT / "runs/ray-adaptation-training-data-v5-20260905"
CTX_ABO = ROOT / "runs/photoreal-training-contexts-20260906"
PHOTO_ABO = ROOT / "runs/ray-photo-training-20260906"
WIDTH = 160 / 25.4
INK, BLUE, GOLD, RED = "#243444", "#246899", "#a86512", "#a63537"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": "#aab5be",
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.facecolor": "white",
    }
)
LEDGER = {
    "purpose": "Post-study descriptive and didactic figures; no outcome-based selection.",
    "selection_record": "artifacts/figures/reviews/data-figures-notes.md",
    "verified_files": {},
    "figures": {},
    "metrics": [],
}


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def digest(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        while b := f.read(8 * 1024 * 1024):
            h.update(b)
    return h.hexdigest()


def rel(p):
    try:
        return Path(p).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return "../" + Path(p).resolve().relative_to(ROOT.parent).as_posix()


def verify(p, expected=None, origin=None):
    p = Path(p)
    got = digest(p)
    if expected is not None:
        assert got == expected, (p, got, expected)
    LEDGER["verified_files"][rel(p)] = {
        "sha256": got,
        "expected_sha256": expected,
        "binding": origin,
    }
    return p


def load_sources():
    p = ROOT / "runs/research-evolution-campaign-20260907/frozen-campaign-v3/campaign.json"
    campaign = read(verify(p))
    refs = campaign["references"]
    complete = read(
        verify(ROOT / refs["source_complete"]["path"], refs["source_complete"]["sha256"], rel(p))
    )
    for name in ("source_descriptive_report", "photo_admission", "training_manifest"):
        verify(ROOT / refs[name]["path"], refs[name]["sha256"], rel(p))
    train = {r["sample_id"]: r for r in read(ROOT / refs["training_manifest"]["path"])["rows"]}
    index = {}
    for source in complete["sources"]:
        folder = ROOT / source["source"]
        manifest = verify(
            folder / "complete.json",
            source["manifest_sha256"],
            rel(ROOT / refs["source_complete"]["path"]),
        )
        for r in read(manifest)["rows"]:
            sid = r["sample_id"]
            if sid in train:
                assert r["files"]["observation.npz"] == train[sid]["source_observation_sha256"], sid
            index[sid] = (folder, r)
    assert len(index) == 7200 and len(train) == 6400
    return index, campaign


def observation(index, sid):
    folder, row = index[sid]
    p = verify(
        folder / sid / "observation.npz",
        row["files"]["observation.npz"],
        rel(folder / "complete.json"),
    )
    with np.load(p) as z:
        arrays = {k: z[k] for k in z.files}
    return arrays, row


def bbox(mask, pad=0.18):
    yy, xx = np.where(mask)
    x0, x1, y0, y1 = xx.min(), xx.max() + 1, yy.min(), yy.max() + 1
    side = max(x1 - x0, y1 - y0) * (1 + 2 * pad)
    side = min(side, mask.shape[0], mask.shape[1])
    x0 = int(np.clip((x0 + x1 - side) / 2, 0, mask.shape[1] - side))
    y0 = int(np.clip((y0 + y1 - side) / 2, 0, mask.shape[0] - side))
    return (x0, y0, min(mask.shape[1], x0 + int(side)), min(mask.shape[0], y0 + int(side)))


def crop(a, box):
    x0, y0, x1, y1 = box
    return a[y0:y1, x0:x1]


def image_panel(fig, rect, a, title=None, cmap=None, vmin=None, vmax=None):
    ax = fig.add_axes(rect)
    ax.imshow(a, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, pad=6, loc="left", fontweight="medium")
    return ax


def depth_image(d, mask=None):
    valid = np.isfinite(d) & (d > 0)
    if mask is not None:
        valid &= mask
    return np.ma.array(d, mask=~valid)


def depth_cmap():
    cm = plt.get_cmap("viridis").copy()
    cm.set_bad("#f0f2f4")
    return cm


def mesh_panel(fig, rect, vertices, faces, title, camera_cv=False):
    # Rigid display rotation for source meshes only; vertices/faces stay intact.
    v = vertices.copy()
    if not camera_cv:
        az, el = np.deg2rad([25, 25])
        rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
        rx = np.array([[1, 0, 0], [0, np.cos(el), -np.sin(el)], [0, np.sin(el), np.cos(el)]])
        v = (v @ rz.T) @ rx.T
        v = v[:, [0, 2, 1]]
        v[:, 1] *= -1
    tri = v[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-15)
    light = np.array([-0.5, -0.7, -1.0])
    light /= np.linalg.norm(light)
    illum = 0.30 + 0.65 * np.abs(normals @ light)
    color = np.clip(np.array([0.32, 0.55, 0.68])[None, :] * illum[:, None] + 0.14, 0, 1)
    order = np.argsort(tri[:, :, 2].mean(1))[::-1]
    ax = fig.add_axes(rect)
    ax.add_collection(
        PolyCollection(
            tri[order, :, :2], facecolors=color[order], edgecolors="none", rasterized=True
        )
    )
    mn = v[:, :2].min(0)
    mx = v[:, :2].max(0)
    center = (mn + mx) / 2
    half = max(mx - mn) * 0.57
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] + half, center[1] - half)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title, loc="left", pad=6)
    return ax


def footer(fig, text, y=0.025, color=INK):
    fig.text(0.02, y, text, ha="left", va="bottom", fontsize=9, color=color, linespacing=1.45)


def save(fig, name, info, *, crop_bottom=0):
    OUT.mkdir(parents=True, exist_ok=True)
    files = {}
    bounds = (
        matplotlib.transforms.Bbox.from_bounds(
            0, crop_bottom, fig.get_figwidth(), fig.get_figheight() - crop_bottom
        )
        if crop_bottom
        else None
    )
    for ext in ("pdf", "png", "svg"):
        p = OUT / f"{name}.{ext}"
        fig.savefig(p, dpi=350, bbox_inches=bounds, pad_inches=0)
        files[rel(p)] = digest(p)
    plt.close(fig)
    LEDGER["figures"][name] = {
        **info,
        "files": files,
        "width_mm": 160,
        "minimum_font_pt": 9,
        "png_dpi": 350,
        "geometry_edited": False,
    }
    print("saved", name, flush=True)


def abo_record(sid):
    protocol_path = ROOT / "runs/ray-adaptation-20260906/photoreal-protocol.json"
    protocol = read(verify(protocol_path))
    manifest = read(
        verify(TRAIN_ABO / "complete.json", protocol["training_sha256"], rel(protocol_path))
    )
    row = next(r for r in manifest["rows"] if r["sample_id"] == sid)
    z = np.load(
        verify(TRAIN_ABO / f"{sid}.npz", row["output_sha256"], rel(TRAIN_ABO / "complete.json"))
    )
    s = read(verify(ABO / f"renders/train/{sid}/sample.json"))
    for r in s["artifacts"].values():
        verify(ABO / r["uri"], r["sha256"], rel(ABO / f"renders/train/{sid}/sample.json"))
    return {k: z[k] for k in z.files}, row, s


def abo_pipeline():
    # Author chose scene 3 from the SAPIEN gallery; see selection_record.
    sid = "000620-B074VMBGT4-medium-0"
    a, row, s = abo_record(sid)
    population = read(TRAIN_ABO / "complete.json")["rows"]
    pixel_counts = []
    for r in population:
        with np.load(
            verify(
                TRAIN_ABO / f"{r['sample_id']}.npz",
                r["output_sha256"],
                rel(TRAIN_ABO / "complete.json"),
            )
        ) as z:
            pixel_counts.append(
                {"sample_id": r["sample_id"], "input_pixels": int(z["input_mask"].sum())}
            )
    pixel_counts.sort(key=lambda r: (r["input_pixels"], r["sample_id"]))
    selection = {
        "reason_pl": (
            "Autor wybrał zieloną poduszkę, scenę 3 z galerii ośmiu "
            "zapisanych scen SAPIEN. Celowy wybór dydaktyczny po badaniu, bez "
            "wykorzystania wyników rekonstrukcji."
        ),
        "author_choice_gallery_number": 3,
        "metric": (
            "Count of input_mask pixels in the full frame; not an occlusion "
            "ratio or full-surface visibility."
        ),
        "population_size": len(pixel_counts),
        "population_median_input_pixels": float(
            np.median([r["input_pixels"] for r in pixel_counts])
        ),
        "selected_input_pixels": int(a["input_mask"].sum()),
        "selected_rank_ascending": next(
            i + 1 for i, r in enumerate(pixel_counts) if r["sample_id"] == sid
        ),
        "population_input_pixels": pixel_counts,
        "previous_sample_id": "001400-B07ML7NZR7-medium-0",
        "initial_sample_id": "000029-B00R3MZITG-medium-0",
    }
    inv = read(verify(ABO / "package/inventory.json"))
    asset = next(r for r in inv["assets"] if r["asset_id"] == row["asset_id"])
    surface = read(
        verify(
            ABO / asset["processed_mesh"]["uri"], asset["processed_mesh"]["sha256"], "ABO inventory"
        )
    )
    assert surface["geometry_sha256"] == row["geometry_sha256"]
    v = np.load(
        verify(ABO / surface["vertices"]["uri"], surface["vertices"]["sha256"], "surface.json")
    )
    f = np.load(verify(ABO / surface["faces"]["uri"], surface["faces"]["sha256"], "surface.json"))
    full = np.asarray(Image.open(ABO / s["artifacts"]["rgb_clean"]["uri"]).convert("RGB"))
    clean = np.load(ABO / s["artifacts"]["depth_clean_m"]["uri"])
    original_mask = np.asarray(Image.open(ABO / s["artifacts"]["target_mask"]["uri"])) > 0
    novel = a["novel_depth_uint16"][0].astype(float) * 10 / 65535
    valid_clean = clean[np.isfinite(clean) & (clean > 0)]
    depth_max = float(
        np.ceil(float(max(valid_clean.max(), novel[a["novel_masks"][0]].max())) * 5) / 5
    )
    b = bbox(a["input_mask"], 0.18)
    cv = np.asarray(s["camera_T_mesh"])
    vc = v @ cv[:3, :3].T + cv[:3, 3]
    vc[:, 1:] *= -1
    fig = plt.figure(figsize=(WIDTH, 5.15))
    x = [0.02, 0.355, 0.69]
    w = 0.29
    mesh_panel(fig, [x[0], 0.56, w, 0.35], vc, f, "a  Siatka celu (ABO)", True)
    ax = image_panel(fig, [x[1], 0.56, w, 0.35], full, "b  Scena w SAPIEN")
    yy, xx = np.where(original_mask)
    ax.add_patch(
        Rectangle(
            (xx.min() - 7, yy.min() - 7),
            np.ptp(xx) + 14,
            np.ptp(yy) + 14,
            fill=False,
            ec="#e5bc54",
            lw=1.1,
        )
    )
    image_panel(
        fig,
        [x[2], 0.56, w, 0.35],
        depth_image(clean),
        "c  Głębia czysta",
        depth_cmap(),
        0,
        depth_max,
    )
    image_panel(
        fig, [x[0], 0.12, w, 0.33], crop(a["input_mask"], b), "d  Maska wejściowa", "gray", 0, 1
    )
    masked = a["rgb"].copy()
    masked[~a["input_mask"]] = 0
    image_panel(fig, [x[1], 0.12, w, 0.33], crop(masked, b), "e  RGB wejścia (zbliżenie)")
    nb = bbox(a["novel_masks"][0], 0.16)
    image_panel(
        fig,
        [x[2], 0.12, w, 0.33],
        crop(depth_image(novel, a["novel_masks"][0]), nb),
        "f  Głębia do przewidzenia",
        depth_cmap(),
        0,
        depth_max,
    )
    # Flow keeps the input and supervision branches separate.
    fig.text(0.334, 0.76, "→", fontsize=15, ha="center")
    fig.text(0.665, 0.76, "→", fontsize=15, ha="center")
    fig.text(0.02, 0.495, "OBSERWACJA  →  maskowanie RGB i głębi", fontsize=9, color=BLUE)
    fig.text(0.69, 0.495, "SIATKA  →  21 kamer", fontsize=9, color=BLUE)
    ticks = [0, depth_max / 2, depth_max]
    cax = fig.add_axes([0.77, 0.082, 0.19, 0.017])
    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(0, depth_max), cmap=depth_cmap()),
        cax=cax,
        orientation="horizontal",
        ticks=ticks,
    )
    cb.ax.set_xticklabels(
        [f"{v:g}".replace(".", ",") + (" m" if i == 2 else "") for i, v in enumerate(ticks)]
    )
    cb.ax.tick_params(length=2, pad=2)
    footer(
        fig,
        (
            "Cel: zielona poduszka. Żółta ramka wskazuje ją w scenie.\nPanele "
            "d–f są zbliżeniami; kolor obrazuje głębię, szary — brak wartości."
        ),
        0.017,
    )
    save(
        fig,
        "data-abo-pipeline",
        {
            "sample_ids": [sid],
            "selection": selection,
            "caption_pl": (
                "Powstawanie przykładu ABO w badaniu wstępnym: zielona poduszka "
                "częściowo zasłonięta przez naczynie. Siatka celu uczestniczy w "
                "scenie SAPIEN, z której pochodzą RGB, czysta głębia i maska; "
                "maskowanie wyznacza wejście rekonstruktora. Ta sama geometria "
                "dostarcza głębi z 21 nowych kamer, z których pokazano pierwszą. "
                "Panele d–f powiększono niezależnie od pełnego kadru. Nowa "
                "wizualizacja istniejących artefaktów, bez ponownego generowania "
                "danych."
            ),
            "source_geometry_sha256": row["geometry_sha256"],
            "crop_input_xyxy": b,
            "novel_camera_index": 0,
            "display_depth_range_m": [0, depth_max],
        },
    )


def main_pipeline(index):
    ids = [
        "3D_Dollhouse_TablePurple-a00-v0",
        "shapenet-02876657-2f4ec01bad6cd5ac488017d48a7f7eb4-a00-v0",
    ]
    titles = ["GSO: miniaturowy stolik", "Przykład ShapeNet"]
    fig = plt.figure(figsize=(WIDTH, 7.8))
    crops = []
    for j, (sid, title) in enumerate(zip(ids, titles, strict=False)):
        a, row = observation(index, sid)
        b = bbox(a["source_target_mask"], 0.16)
        crops.append(b)
        top = 0.78 - j * 0.485
        bot = 0.575 - j * 0.485
        w = 0.29
        h = 0.17
        x = [0.02, 0.355, 0.69]
        fig.text(0.02, top + 0.204, title, fontsize=10.5, fontweight="bold")
        mesh_panel(
            fig,
            [x[0], top, w, h],
            a["mesh_vertices_camera_cv_m"],
            a["mesh_faces"],
            "Siatka celu",
            True,
        )
        ax = image_panel(fig, [x[1], top, w, h], a["rgb"], "Końcowa scena RGB")
        bx, by, bxx, byy = b
        ax.add_patch(Rectangle((bx, by), bxx - bx, byy - by, fill=False, ec="#e5bc54", lw=0.9))
        rgb = a["rgb"].copy()
        rgb[~a["input_mask"]] = 0
        image_panel(fig, [x[2], top, w, h], crop(rgb, b), "RGB wejścia")
        dr = a["depth_clean_m"][a["source_target_mask"]]
        lo = np.floor(np.min(dr) * 10) / 10
        hi = np.ceil(np.max(dr) * 10) / 10
        if hi - lo < 0.1:
            hi = lo + 0.1
        image_panel(
            fig,
            [x[0], bot, w, h],
            crop(depth_image(a["depth_clean_m"], a["source_target_mask"]), b),
            "Czysta głębia celu",
            depth_cmap(),
            lo,
            hi,
        )
        image_panel(
            fig,
            [x[1], bot, w, h],
            crop(depth_image(a["depth_sensor_m"], a["source_target_mask"]), b),
            "Głębia wejściowa",
            depth_cmap(),
            lo,
            hi,
        )
        image_panel(
            fig, [x[2], bot, w, h], crop(a["input_mask"], b), "Maska wejściowa", "gray", 0, 1
        )
        cax = fig.add_axes([0.035, bot - 0.021, 0.245, 0.009])
        cb = fig.colorbar(
            plt.cm.ScalarMappable(norm=Normalize(lo, hi), cmap=depth_cmap()),
            cax=cax,
            orientation="horizontal",
            ticks=[lo, hi],
        )
        cb.ax.set_xticklabels([f"{lo:.1f}".replace(".", ","), f"{hi:.1f} m".replace(".", ",")])
        cb.ax.tick_params(length=2, pad=2)
        fig.text(0.36, bot - 0.025, "Wspólna skala obu map głębi", fontsize=9)
        fig.text(0.334, top + 0.09, "→", fontsize=14, ha="center")
        fig.text(0.664, top + 0.09, "→", fontsize=14, ha="center")
    footer(fig, "Szary: brak głębi; biały: maska wejścia. Pełny kadr i wspólne zbliżenia.", 0.009)
    save(
        fig,
        "data-main-pipeline",
        {
            "sample_ids": ids,
            "crop_xyxy": crops,
            "caption_pl": (
                "Dwa przykłady końcowych danych eksperymentu głównego: geometria "
                "GSO i ShapeNet. Pełne RGB przedstawia zachowaną scenę po "
                "kontroli stanu końcowego; ramka wyznacza wspólny obszar zbliżeń. "
                "Czysta głębia celu pochodzi z renderera, a głębia wejściowa "
                "zawiera zaburzenia i braki po modelu sensora oraz filtrze "
                "mediany. Maska wejściowa określa dostępne punkty celu. Siatka "
                "pozostaje źródłem nadzoru nowych kamer. W każdym bloku użyto "
                "jednej obserwacji z manifestu końcowej kampanii; nie pokazano "
                "historycznej trajektorii ruchu."
            ),
        },
    )


def photo_pilot():
    protocol_path = ROOT / "runs/ray-adaptation-20260906/photoreal-protocol.json"
    protocol = read(verify(protocol_path))
    comp = read(verify(PHOTO_ABO / "complete.json", protocol["photo_sha256"], rel(protocol_path)))
    review = read(
        verify(PHOTO_ABO / "review.json", comp["review_sha256"], rel(PHOTO_ABO / "complete.json"))
    )
    archive = verify(
        ROOT / "runs/klein-candidates-20260906/results.zip",
        comp["candidate_archive_sha256"],
        rel(PHOTO_ABO / "complete.json"),
    )
    context = read(verify(CTX_ABO / "complete.json"))
    ctx = {r["sample_id"]: r for r in context["rows"]}
    assert context["training_sha256"] == protocol["training_sha256"]
    chosen = [
        sorted(
            (r for r in comp["rows"] if r["decision"] == d and r["variant"] == 0),
            key=lambda r: r["sample_id"],
        )[0]
        for d in ("accepted", "fallback_original")
    ]
    fig = plt.figure(figsize=(WIDTH, 4.85))
    records = []
    for j, r in enumerate(chosen):
        sid = r["sample_id"]
        a, _, _ = abo_record(sid)
        b = bbox(a["input_mask"], 0.23)
        source = np.asarray(
            Image.open(
                verify(
                    CTX_ABO / sid / "rgb.png",
                    ctx[sid]["files"]["rgb.png"],
                    rel(CTX_ABO / "complete.json"),
                )
            ).convert("RGB")
        )
        with zipfile.ZipFile(archive) as z:
            raw = z.read(f"{sid}-v0-raw.png")
        assert hashlib.sha256(raw).hexdigest() == r["source_raw_sha256"]
        photo = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
        final = np.asarray(
            Image.open(
                verify(PHOTO_ABO / r["filename"], r["sha256"], rel(PHOTO_ABO / "complete.json"))
            ).convert("RGB")
        )
        assert source.shape == photo.shape == a["rgb"].shape
        top = 0.53 - j * 0.44
        title = "a  Zaakceptowana edycja: stojak" if j == 0 else "b  Odrzucona edycja: hantel"
        fig.text(
            0.02, top + 0.37, title, fontsize=10.5, fontweight="bold", color=BLUE if j == 0 else RED
        )
        for x, im, t in zip(
            [0.02, 0.355, 0.69],
            [source, photo, final],
            ["Źródłowe RGB", "Edycja PHOTO", "Wejście PHOTO"],
            strict=False,
        ):
            image_panel(fig, [x, top, 0.29, 0.305], crop(im, b), t)
        reason = next(
            v["reason"] for v in review["rows"] if v["sample_id"] == sid and v["variant"] == 0
        )
        records.append(
            {
                **r,
                "crop_xyxy": b,
                "review_reason": reason,
                "raw_archive_member": f"{sid}-v0-raw.png",
            }
        )
    footer(
        fig,
        (
            "Zachowano profil stojaka; hantel przypomina butelkę.\nPo "
            "odrzuceniu edycji model otrzymuje źródłowe RGB."
        ),
        0.012,
    )
    save(
        fig,
        "data-photo-pilot",
        {
            "samples": records,
            "caption_pl": (
                "Receptura PHOTO w badaniu wstępnym: powiększenia dwóch par "
                "SOURCE–edycja oraz obrazu faktycznie użytego w ramieniu PHOTO. "
                "Wariant 0 stojaka B010QZD6I6 zaakceptowano po ocenie zgodności "
                "profilu i granic przesłonięcia. Wariant 0 hantla B00R3MZITG "
                "odrzucono, ponieważ kandydat sugerował korpus i szyjkę butelki; "
                "użyto pierwotnego RGB. Statusy pochodzą z historycznego "
                "przeglądu. Wybrano pierwszą alfabetycznie obserwację w każdej z "
                "dwóch kategorii decyzji. Dla każdej pary zachowano wspólny kadr "
                "wyznaczony z maski źródłowej."
            ),
        },
    )


def photo_main(index):
    admission = read(PHOTO / "complete.json")
    # Two different observations of the same target: appearance versus added parts.
    ids = [
        "Sapota_Threshold_4_Ceramic_Round_Planter_Red-a00-v2",
        "Sapota_Threshold_4_Ceramic_Round_Planter_Red-a02-v1",
    ]
    chosen = [next(r for r in admission["rows"] if r["sample_id"] == sid) for sid in ids]
    review_path = ROOT / "runs/research-evolution-photo-20260907/full-visual-progress-v19.json"
    reviews = {r["sample_id"]: r for r in read(verify(review_path))["rows"]}
    fig = plt.figure(figsize=(WIDTH, 5.0))
    records = []
    for j, r in enumerate(chosen):
        review = reviews[r["sample_id"]]
        assert r["admission_reason"] == ["visual_pass", "visual_reject"][j]
        assert (
            r["accepted"] == (j == 0) and r["SAM_eligible"] and r["individually_visually_reviewed"]
        )
        assert review["decision"] == ["pass", "reject"][j] and review["PHOTO_sha256"] == r["sha256"]
        for item in review["evidence"]:
            verify(ROOT / item["path"], item["sha256"], rel(review_path))
        a, row = observation(index, r["sample_id"])
        b = bbox(a["source_target_mask"], 0.32 if j == 1 else 0.20)
        photo = np.asarray(
            Image.open(
                verify(PHOTO / r["filename"], r["sha256"], rel(PHOTO / "complete.json"))
            ).convert("RGB")
        )
        # The admission binds an ndarray hash, while source manifests bind the NPZ.
        assert hashlib.sha256(a["rgb"].tobytes()).hexdigest() == r["source_rgb_array_sha256"]
        final = (photo if r["accepted"] else a["rgb"]).copy()
        final[~a["input_mask"]] = 0
        y = 0.57 - j * 0.45
        names = ["a  Dopuszczona po przeglądzie", "b  Odrzucona po przeglądzie"]
        fig.text(0.02, y + 0.385, names[j], fontsize=10, fontweight="bold", color=[BLUE, RED][j])
        for x, im, t in zip(
            [0.02, 0.355, 0.69],
            [a["rgb"], photo, final],
            ["Źródłowe RGB", "Edycja PHOTO", "Wejście PHOTO"],
            strict=False,
        ):
            image_panel(fig, [x, y, 0.29, 0.32], crop(im, b), t)
        records.append(
            {
                **r,
                "crop_xyxy": b,
                "source_observation_sha256": row["files"]["observation.npz"],
                "review_record": review,
                "selection": [
                    (
                        "Visible gloss and directional lighting with the recorded visible "
                        "structure retained."
                    ),
                    "Two conspicuous added handles on the reconstruction target.",
                ][j],
            }
        )
    footer(
        fig,
        (
            "a: zmiana połysku i oświetlenia.  b: dodane uchwyty.\nPo "
            "odrzuceniu model otrzymuje źródłowe RGB z maską."
        ),
        0.025,
    )
    save(
        fig,
        "data-photo-main",
        {
            "samples": records,
            "selection": (
                "Purposeful didactic pairing of two different observations of the "
                "same GSO planter, both historically reviewed and SAM-eligible. "
                "Accepted example chosen after visual inspection of 42 candidates "
                "from distinct objects, shortlisted from the 866 final visual "
                "passes by recorded target RGB change and target pixel count. Not "
                "a representative sample; no reconstruction metrics used. The "
                "unreviewed-status row was removed at the author's request."
            ),
            "caption_pl": (
                "Dopuszczona (a) i odrzucona (b) edycja dwóch różnych obserwacji "
                "tej samej doniczki. W (a) zmieniono połysk i oświetlenie, a w "
                "(b) dodano dwa uchwyty. Obie pary przeszły kontrolę masek SAM. "
                "Każdy wiersz zachowuje wspólny kadr źródła, edycji i wejścia "
                "rekonstruktora."
            ),
        },
    )


def sofa_example(index):
    sid = "3D_Dollhouse_Sofa-a00-v2"
    a, row = observation(index, sid)
    b = bbox(a["source_target_mask"], 0.20)
    fig = plt.figure(figsize=(WIDTH, 2.05))
    x = [0.02, 0.265, 0.51, 0.755]
    w = 0.225
    y = 0.23
    h = 0.63
    mesh_panel(
        fig, [x[0], y, w, h], a["mesh_vertices_camera_cv_m"], a["mesh_faces"], "a  Model 3D", True
    )
    image_panel(fig, [x[1], y, w, h], crop(a["rgb"], b), "b  RGB")
    valid = a["depth_clean_m"][a["source_target_mask"]]
    lo = float(np.floor(valid.min() * 10) / 10)
    hi = float(np.ceil(valid.max() * 10) / 10)
    if hi - lo < 0.1:
        hi = lo + 0.1
    image_panel(
        fig,
        [x[2], y, w, h],
        crop(depth_image(a["depth_clean_m"], a["source_target_mask"]), b),
        "c  Głębia celu",
        depth_cmap(),
        lo,
        hi,
    )
    image_panel(
        fig, [x[3], y, w, h], crop(a["source_target_mask"], b), "d  Maska celu", "gray", 0, 1
    )
    cax = fig.add_axes([x[2] + 0.012, 0.15, w - 0.024, 0.035])
    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(lo, hi), cmap=depth_cmap()),
        cax=cax,
        orientation="horizontal",
        ticks=[lo, hi],
    )
    cb.ax.set_xticklabels([f"{lo:.1f}".replace(".", ","), f"{hi:.1f} m".replace(".", ",")])
    cb.ax.tick_params(length=2, pad=2)
    save(
        fig,
        "data-sofa-example",
        {
            "sample_ids": [sid],
            "crop_xyxy": b,
            "source_observation_sha256": row["files"]["observation.npz"],
            "display_depth_range_m": [lo, hi],
            "selection": (
                "The same GSO sofa observation already used in PHOTO panel a, "
                "chosen to explain the data beside its first mention in section "
                "3.1."
            ),
            "caption_pl": (
                "Pełna siatka sofy GSO oraz RGB, czysta głębia i maska widocznej "
                "części celu z jednej obserwacji. Panele b-d mają wspólny kadr; "
                "źródłowa maska i czysta głębia poprzedzają model sensora. "
                "Istniejące zamrożone dane; nowa kompozycja ilustracyjna."
            ),
        },
    )


def strategies():
    from scripts.preliminary_hope.prepare_training_plan import classic_rgb, schedule

    sid = "000059-B010QZD6I6-medium-0"
    a, row, _ = abo_record(sid)
    protocol_path = ROOT / "runs/ray-adaptation-20260906/photoreal-protocol.json"
    protocol = read(verify(protocol_path))
    function_path = verify(
        ROOT / "scripts/preliminary_hope/prepare_training_plan.py",
        protocol["common_training_contract"]["plan_script_sha256"],
        rel(protocol_path),
    )
    manifest = read(TRAIN_ABO / "complete.json")
    ordinal = next(i for i, r in enumerate(manifest["rows"]) if r["sample_id"] == sid)
    exposure = next(r for r in schedule(0) if r["ordinal"] == ordinal and r["augment"])
    seed = exposure["augmentation_seed"]
    variant = exposure["variant"]
    comp = read(verify(PHOTO_ABO / "complete.json", protocol["photo_sha256"], rel(protocol_path)))
    photo_row = next(r for r in comp["rows"] if r["sample_id"] == sid and r["variant"] == variant)
    assert photo_row["decision"] == "accepted"
    photo = np.asarray(
        Image.open(
            verify(
                PHOTO_ABO / photo_row["filename"],
                photo_row["sha256"],
                rel(PHOTO_ABO / "complete.json"),
            )
        ).convert("RGB")
    )
    original = a["rgb"].copy()
    original[~a["input_mask"]] = 0
    classic = classic_rgb(a["rgb"], a["input_mask"], seed)
    assert np.array_equal(classic, classic_rgb(a["rgb"], a["input_mask"], seed))
    assert np.all(classic[~a["input_mask"]] == 0)
    b = bbox(a["input_mask"], 0.14)
    fig = plt.figure(figsize=(WIDTH, 2.42))
    for x, im, title, color in zip(
        [0.02, 0.355, 0.69],
        [original, classic, photo],
        ["BASE: render źródłowy", "CLASSIC", "PHOTO"],
        ["#326b8b", "#b37b2a", "#88559c"],
        strict=False,
    ):
        ax = image_panel(fig, [x, 0.17, 0.29, 0.72], crop(im, b), title)
        ax.title.set_color(color)
    footer(fig, "Ta sama obserwacja, maska, głębia, kalibracja i etykiety nowych widoków.", 0.035)
    rng = np.random.default_rng(seed)
    factors = [float(rng.uniform(0.8, 1.2)) for _ in range(3)] + [float(rng.uniform(0.9, 1.1))]
    save(
        fig,
        "data-strategies",
        {
            "sample_ids": [sid],
            "training_seed": 0,
            "selection_rule": (
                "First scheduled augmented exposure of fixed sample in "
                "schedule(0), no visual or outcome selection of transform seed."
            ),
            "exposure": exposure,
            "classic_function": rel(function_path),
            "classic_augmentation_seed": seed,
            "classic_parameters": dict(
                zip(["brightness", "contrast", "saturation", "gamma"], factors, strict=False)
            ),
            "classic_array_sha256": hashlib.sha256(classic.tobytes()).hexdigest(),
            "source_array_sha256": hashlib.sha256(original.tobytes()).hexdigest(),
            "photo_row": photo_row,
            "crop_xyxy": b,
            "caption_pl": (
                "Trzy strategie RGB na tej samej obserwacji uczącej ABO: BASE, "
                "dokładna transformacja CLASSIC oraz dopuszczona edycja PHOTO. "
                "CLASSIC odtworzono funkcją używaną w treningu dla pierwszej "
                "augmentowanej ekspozycji tej obserwacji w harmonogramie z "
                "ziarnem treningu 0; PHOTO odpowiada tej samej ekspozycji. "
                "Wszystkie panele pokazują wspólne zbliżenie maskowanego RGB. "
                "Głębia, maska, kalibracja oraz nadzór nowych widoków pozostają "
                "identyczne. Wybór ilustracji po badaniu, bez dobierania "
                "transformacji według wyniku rekonstrukcji."
            ),
        },
    )


def compute_metrics(index, reuse):
    if reuse and EVIDENCE.exists():
        old = read(EVIDENCE)
        if len(old.get("metrics", [])) == 7200:
            LEDGER["metrics"] = old["metrics"]
            report_rows = {
                r["sample_id"]: r
                for r in read(DATA / "source-input-descriptive-report-full-v1.json")["rows"]
            }
            for r in LEDGER["metrics"]:
                r["missing_depth_before_adapter_pct"] = (
                    100 * report_rows[r["sample_id"]]["sensor_target_invalid_rate"]
                )
            for path, record in old["verified_files"].items():
                if path.endswith("observation.npz"):
                    LEDGER["verified_files"][path] = record
            return old["metrics"]
    report = read(DATA / "source-input-descriptive-report-full-v1.json")

    def one(r):
        sid = r["sample_id"]
        folder, row = index[sid]
        p = folder / sid / "observation.npz"
        got = digest(p)
        assert got == row["files"]["observation.npz"], sid
        with np.load(p) as z:
            m = z["source_target_mask"]
            d = z["depth_sensor_m"]
            mi = z["input_mask"]
        valid = np.isfinite(d) & (d > 0)
        return {
            "sample_id": sid,
            "key": r["key"],
            "source": r["asset_source"],
            "role": r["role"],
            "source_mask_occupancy_pct": 100 * float(m.mean()),
            "input_mask_occupancy_pct": 100 * float(mi.mean()),
            "missing_depth_in_source_mask_pct": 100 * float(np.count_nonzero(m & ~valid) / m.sum()),
            "missing_depth_before_adapter_pct": 100 * r["sensor_target_invalid_rate"],
            "npz_path": rel(p),
            "npz_sha256": got,
        }

    print("computing all 7200 descriptive mask/depth records", flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(one, report["rows"]))
    assert len(rows) == 7200
    LEDGER["metrics"] = rows
    for r in rows:
        LEDGER["verified_files"][r["npz_path"]] = {
            "sha256": r["npz_sha256"],
            "expected_sha256": r["npz_sha256"],
            "binding": "source_complete -> source manifest -> row files",
        }
    return rows


def distributions(index, reuse=False):
    rows = compute_metrics(index, reuse)
    packaged = read(verify(PHOTO / "packaged-distribution-v1.json"))
    verify(
        PHOTO / "complete.json",
        packaged["references"]["admission"]["sha256"],
        rel(PHOTO / "packaged-distribution-v1.json"),
    )
    fig = plt.figure(figsize=(WIDTH, 6.75))
    # Diversity is explicit fixed-ID illustration, not representative sampling.
    ids = [
        "3D_Dollhouse_TablePurple-a00-v0",
        "Chelsea_lo_fl_rdheel_zAQrnhlEfw8-a00-v0",
        "Razer_Kraken_Pro_headset_Full_size_Black-a00-v0",
        "shapenet-02876657-2f4ec01bad6cd5ac488017d48a7f7eb4-a00-v0",
    ]
    names = ["GSO: stolik", "GSO: obuwie", "GSO: słuchawki", "ShapeNet"]
    for i, (sid, name) in enumerate(zip(ids, names, strict=False)):
        a, _ = observation(index, sid)
        rgb = a["rgb"].copy()
        rgb[~a["source_target_mask"]] = 245
        image_panel(
            fig,
            [0.02 + i * 0.25, 0.775, 0.215, 0.165],
            crop(rgb, bbox(a["source_target_mask"], 0.13)),
            name,
        )
    groups = [
        ("TRAIN", "gso", "GSO uczący", BLUE, "-"),
        ("TRAIN", "shapenet", "ShapeNet uczący", GOLD, "-"),
        ("VAL_holdout", None, "Dane do oceny", INK, "--"),
    ]
    specs = [
        (
            "source_mask_occupancy_pct",
            "a  Rozmiar widocznej maski celu",
            "Udział pikseli kadru [%]",
            [0.095, 0.455, 0.37, 0.24],
        ),
        (
            "missing_depth_in_source_mask_pct",
            "b  Braki głębi wejścia w masce",
            "Brakujące wartości [%]",
            [0.605, 0.455, 0.37, 0.24],
        ),
    ]
    stats = {}
    for key, title, xlabel, pos in specs:
        ax = fig.add_axes(pos)
        for role, source, label, color, ls in groups:
            v = np.sort(
                [
                    r[key]
                    for r in rows
                    if r["role"] == role and (source is None or r["source"] == source)
                ]
            )
            ax.step(
                v,
                100 * np.arange(1, len(v) + 1) / len(v),
                where="post",
                color=color,
                ls=ls,
                lw=1.6,
                label=label,
            )
            stats[f"{role}/{source}/{key}"] = {
                "n": len(v),
                "minimum": float(v.min()),
                "median": float(np.median(v)),
                "maximum": float(v.max()),
            }
        ax.set_ylim(0, 102)
        ax.set_yticks([0, 50, 100])
        ax.set_xlim(left=0)
        ax.grid(alpha=0.2)
        ax.set_axisbelow(True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Odsetek obserwacji [%]")
        ax.set_title(title, loc="left", pad=9)
    fig.legend(
        *fig.axes[-1].get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.52, 0.779),
        ncol=3,
        frameon=False,
        columnspacing=1.3,
        handlelength=1.8,
    )
    ax = fig.add_axes([0.095, 0.12, 0.88, 0.215])
    counts = Counter(r["accepted"] for r in packaged["rows"])
    ax.bar(np.arange(21), [counts[i] for i in range(21)], color=BLUE, width=0.85)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xlim(-0.75, 20.75)
    ax.set_xlabel("Dopuszczone edycje / 20 obserwacji")
    ax.set_ylabel("Liczba modeli 3D")
    ax.set_title("c  Dopuszczone edycje PHOTO na model 3D", loc="left", pad=9)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    # The mask interpretation belongs to panel (a) in the manuscript caption.
    save(
        fig,
        "data-distributions",
        {
            "diversity_sample_ids": ids,
            "statistics": stats,
            "bottom_crop_inches": 0.20,
            "caption_pl": (
                "Przykłady różnorodności i opis całego końcowego zbioru "
                "syntetycznego. Górny pas zawiera celowo wskazane geometrie GSO i "
                "ShapeNet. Dystrybuanty empiryczne obejmują wszystkie 6400 "
                "obserwacji uczących i 720 obserwacji syntetycznego holdoutu: "
                "udział źródłowej maski celu w pełnym kadrze oraz odsetek głębi "
                "niedostępnej na wejściu w tej masce, po modelu sensora i filtrze "
                "mediany. Udział pikseli maski w panelu (a) nie określa odsetka "
                "zasłonięcia przedmiotu. Obserwacje rozwojowe wyłączono z krzywej "
                "holdoutu. Histogram (c) przedstawia liczbę dopuszczonych edycji "
                "PHOTO spośród 20 obserwacji każdej z 320 geometrii uczących. "
                "Opis wyliczono po badaniu z końcowych artefaktów; dwadzieścia "
                "widoków i aranżacji jednej geometrii nie tworzy dwudziestu "
                "niezależnych obiektów."
            ),
        },
        crop_bottom=0.20,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-metrics", action="store_true")
    parser.add_argument(
        "--only",
        choices=["abo", "main", "photo-pilot", "photo-main", "sofa", "distributions", "strategies"],
    )
    args = parser.parse_args()
    if EVIDENCE.exists():
        old = read(EVIDENCE)
        LEDGER["figures"] = old.get("figures", {})
        LEDGER["metrics"] = old.get("metrics", [])
        LEDGER["verified_files"] = old.get("verified_files", {})
    index, campaign = load_sources()
    tasks = {
        "abo": abo_pipeline,
        "main": lambda: main_pipeline(index),
        "photo-pilot": photo_pilot,
        "photo-main": lambda: photo_main(index),
        "sofa": lambda: sofa_example(index),
        "distributions": lambda: distributions(index, args.reuse_metrics),
        "strategies": strategies,
    }
    for name, fn in tasks.items():
        if args.only is None or args.only == name:
            fn()
    summaries = {}
    for role in ("TRAIN", "VAL_development", "VAL_holdout"):
        rr = [r for r in LEDGER["metrics"] if r["role"] == role]
        if not rr:
            continue
        summaries[role] = {
            "observations": len(rr),
            "geometries": len({r["key"] for r in rr}),
            "sources": dict(Counter(r["source"] for r in rr)),
        }
        for key in (
            "source_mask_occupancy_pct",
            "input_mask_occupancy_pct",
            "missing_depth_in_source_mask_pct",
            "missing_depth_before_adapter_pct",
        ):
            if key not in rr[0]:
                continue
            values = np.array([r[key] for r in rr])
            summaries[role][key] = dict(
                zip(
                    ("minimum", "p05", "median", "p95", "maximum"),
                    np.quantile(values, [0, 0.05, 0.5, 0.95, 1]).tolist(),
                    strict=False,
                )
            )
    LEDGER["global_descriptive_summaries"] = summaries
    LEDGER["metric_definitions"] = {
        "source_mask_occupancy_pct": (
            "100 * mean(source_target_mask); visible rendered target "
            "silhouette before sensor missingness, in the 640x480 full frame; "
            "not an occlusion ratio"
        ),
        "input_mask_occupancy_pct": (
            "100 * mean(input_mask); target pixels available to the model "
            "after depth validity selection, in the 640x480 full frame"
        ),
        "missing_depth_in_source_mask_pct": (
            "100 * count(source_target_mask & ~(isfinite(depth_sensor_m) & "
            "(depth_sensor_m > 0))) / count(source_target_mask); actual "
            "unavailable input depth AFTER sensor corruption and median "
            "+/-0.25m filtering, before uint16 encoding"
        ),
        "missing_depth_before_adapter_pct": (
            "100 * sensor_target_invalid_rate from frozen source descriptive "
            "report; measured missingness returned by sensor_candidate before "
            "median-depth filtering, not a nominal dropout parameter"
        ),
    }
    LEDGER["script_sha256"] = digest(__file__)
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps(LEDGER, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
