"""Illustrate one archived FLUX -> RMBG -> TRELLIS -> SAPIEN training example.

Post-study explanatory selection, CPU only. No generation, inference, geometry
repair, scoring, or source-data mutation is performed. Run with a Python
environment containing numpy, Pillow, matplotlib, and trimesh. The local project
venv is a fallback for trimesh when the desktop Python has plotting libraries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from PIL import Image

from masters_rgbd.data.rgbd_inventory import (
    preprocess_legacy_diagnostic_geometry,
)
from scripts.common.paths import CODE_ROOT, DATASET_ROOT
from scripts.common.paths import ROOT as WORKSPACE_ROOT

matplotlib.use("Agg")

ROOT = WORKSPACE_ROOT
sys.path.insert(0, str(CODE_ROOT / "src"))
try:
    import trimesh
except ModuleNotFoundError:
    sys.path.append(str(ROOT / ".venv/Lib/site-packages"))
    import trimesh


ASSET_ID = "apple_001"
SAMPLE_ID = "scene_004597_i009"
RUN = ROOT / "runs/submission-diversity-mix-s0-20260904"
WIDTH_MM, HEIGHT_MM, DPI = 145, 115, 350
INK, BLUE = "#243444", "#246899"
CAPTION = (
    "Historyczny pipeline FLUX.2 → RMBG-2.0 → TRELLIS.2 → SAPIEN dla obiektu "
    "apple_001 z dawnej puli treningowej. Panele d–e pochodzą z tej samej "
    "zachowanej obserwacji; pokazano pełny RGB i głębię w masce celu. Siatka ma "
    "uproszczone oświetlenie, bez tekstury. Szachownica oznacza przezroczystość, "
    "szarość — obszar poza celem. Dobór ilustracyjny po badaniu. Opracowanie własne."
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def relative(path: Path) -> str:
    path = path.resolve()
    if path.is_relative_to(ROOT):
        return path.relative_to(ROOT).as_posix()
    if path.is_relative_to(ROOT.parent):
        return "../" + path.relative_to(ROOT.parent).as_posix()
    return path.as_posix()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def image_panel(fig, rect, data, title, **kwargs):
    ax = fig.add_axes(rect)
    ax.imshow(data, interpolation="nearest", **kwargs)
    ax.set_axis_off()
    ax.set_title(title, loc="left", pad=7)
    return ax


def checkerboard_rgba(rgba: np.ndarray) -> np.ndarray:
    yy, xx = np.indices(rgba.shape[:2])
    tile = ((yy // 48 + xx // 48) % 2).astype(bool)
    background = np.where(tile[:, :, None], 0.91, 0.98)
    alpha = rgba[:, :, 3:4].astype(float) / 255
    return rgba[:, :, :3].astype(float) / 255 * alpha + background * (1 - alpha)


def mesh_panel(fig, rect, mesh, camera_t_mesh, intrinsics):
    """Full source triangles, saved camera pose, approximate flat shading.

    Painter ordering is a display-only approximation. Source coordinates,
    triangle connectivity, and materials in the preserved GLB are not changed.
    The display omits texture/PBR and zooms independently of the full RGB frame.
    """
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vc = vertices @ camera_t_mesh[:3, :3].T + camera_t_mesh[:3, 3]
    depth = -vc[:, 2]
    assert (depth > 0).all(), "Selected mesh must lie in front of saved camera"
    projected = np.column_stack(
        [
            intrinsics["fx"] * vc[:, 0] / depth + intrinsics["cx"],
            intrinsics["cy"] - intrinsics["fy"] * vc[:, 1] / depth,
        ]
    )
    triangles = vc[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-15)
    light = np.array([-0.5, 0.7, 1.0])
    light /= np.linalg.norm(light)
    illumination = 0.30 + 0.65 * np.abs(normals @ light)
    colors = np.clip(np.array([0.32, 0.55, 0.68])[None, :] * illumination[:, None] + 0.14, 0, 1)
    order = np.argsort(depth[faces].mean(axis=1), kind="stable")[::-1]
    ax = fig.add_axes(rect)
    ax.add_collection(
        PolyCollection(
            projected[faces][order],
            facecolors=colors[order],
            edgecolors="none",
            antialiaseds=False,
            rasterized=True,
        )
    )
    lower, upper = projected.min(axis=0), projected.max(axis=0)
    center = (lower + upper) / 2
    half = float(np.max(upper - lower) * 0.65)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] + half, center[1] - half)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title("c  Siatka TRELLIS.2", loc="left", pad=7)
    return {
        "display_crop_image_xyxy": [*(center - half).tolist(), *(center + half).tolist()],
        "light_camera_xyz": light.tolist(),
        "source_triangle_count": len(faces),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    args = parser.parse_args()
    dataset = args.dataset_root.resolve()
    output = ROOT / "artifacts/figures/assets"
    evidence = ROOT / "artifacts/figures/evidence/trellis-figure.json"
    sources: list[dict] = []

    def source(path: Path, role: str, **details) -> Path:
        assert path.is_file(), path
        sources.append({"path": relative(path), "sha256": digest(path), "role": role, **details})
        return path

    config_path = source(
        RUN / "resolved-config.json", "Completed historical MIX0 training configuration"
    )
    config = read_json(config_path)
    selected = next(row for row in config["cohort"] if row["asset"]["asset_id"] == ASSET_ID)
    assert selected["sample_id"] == SAMPLE_ID and selected["asset"]["split"] == "train"
    assert ASSET_ID in config["active_train_asset_ids"]
    exposures = config["training_asset_schedule"].count(ASSET_ID)
    assert exposures == 15
    complete = read_json(source(RUN / "complete.json", "Completion and checkpoint binding"))
    checkpoint = source(RUN / "pixel_knn/best-checkpoint.pt", "Preserved completed MIX0 checkpoint")
    assert digest(checkpoint) == complete["checkpoint_sha256"]

    rendered = dataset / "generated_sapien_trellis_train_5k"
    manifest = source(
        rendered / "training_manifest_target.jsonl",
        "Historical RGB-D observation manifest",
        line=4598,
    )
    matches = [
        (number, json.loads(line))
        for number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1)
        if f'"{SAMPLE_ID}"' in line
    ]
    assert len(matches) == 1
    manifest_line, row = matches[0]
    assert manifest_line == 4598 and row["object_id"] == ASSET_ID and row["instance_id"] == 9
    assert row["is_target"] is True
    camera_t_mesh = np.asarray(row["label"]["camera_T_mesh"], dtype=float)
    np.testing.assert_array_equal(camera_t_mesh, selected["legacy_camera_T_mesh"])

    mesh_path = source((rendered / row["paths"]["mesh"]).resolve(), "GLB imported into SAPIEN")
    asset_meta_path = source(
        mesh_path.with_name("metadata.json"), "Imported asset to source GLB binding"
    )
    asset_meta = read_json(asset_meta_path)
    assert asset_meta["object_id"] == ASSET_ID and asset_meta["source_format"] == "trellis_glb"
    original_glb = source(
        dataset / asset_meta["source_path"].replace("\\", "/"), "Original TRELLIS GLB"
    )
    assert digest(mesh_path) == digest(original_glb), (
        "Imported and source GLB must be byte-identical"
    )
    batch = original_glb.parent.parent
    raw_path = source(batch / "references_raw" / f"{ASSET_ID}.png", "Original FLUX image; panel a")
    raw_meta = read_json(
        source(raw_path.with_suffix(".json"), "FLUX model, prompt, seed and image settings")
    )
    rgba_path = source(batch / "references_rgba" / f"{ASSET_ID}.png", "Original RMBG RGBA; panel b")
    bg_meta = read_json(
        source(rgba_path.with_suffix(".background.json"), "RMBG identity and raw-image binding")
    )
    assert raw_meta["id"] == bg_meta["id"] == ASSET_ID
    assert raw_meta["model"] == "diffusers/FLUX.2-dev-bnb-4bit"
    assert bg_meta["method"] == "rmbg2" and bg_meta["model"] == "briaai/RMBG-2.0"
    assert Path(bg_meta["source"]).name == raw_path.name
    log_path = source(batch / "continue_trellis2.log", "Actual TRELLIS model loaded", line=18)
    assert "loading TRELLIS model microsoft/TRELLIS.2-4B" in log_path.read_text(encoding="utf-8")

    render_paths = {
        key: source(rendered / row["paths"][key], f"Saved SAPIEN {key}")
        for key in ["rgb", "depth", "instance_mask", "camera", "scene"]
    }
    scene = read_json(render_paths["scene"])
    camera = read_json(render_paths["camera"])
    assert scene["target_instance_id"] == 9
    scene_target = next(obj for obj in scene["objects"] if obj["instance_id"] == 9)
    assert (
        scene_target["object_id"] == ASSET_ID and scene_target["mesh_path"] == row["paths"]["mesh"]
    )
    assert camera["intrinsics"] == row["camera"]["intrinsics"] == selected["camera"]["intrinsics"]

    rgba = np.asarray(Image.open(rgba_path).convert("RGBA"))
    raw = np.asarray(Image.open(raw_path).convert("RGB"))
    rgb = np.asarray(Image.open(render_paths["rgb"]).convert("RGB"))
    depth = np.load(render_paths["depth"], allow_pickle=False)
    target = np.asarray(Image.open(render_paths["instance_mask"])) == 9
    assert raw.shape[:2] == rgba.shape[:2] == (768, 768)
    assert depth.shape == target.shape == rgb.shape[:2] == (360, 640)
    assert (
        int(target.sum())
        == row["visibility"]["visible_pixel_count"]
        == selected["visible_pixel_count"]
        == 49746
    )
    assert (np.isfinite(depth[target]) & (depth[target] > 0)).all()
    mesh = trimesh.load(mesh_path, force="scene", process=False).to_geometry()
    processed = preprocess_legacy_diagnostic_geometry(
        vertices=mesh.vertices,
        faces=mesh.faces,
        uniform_scale_m_per_source_unit=selected["asset"]["uniform_scale_m_per_source_unit"],
    )
    assert processed.fingerprint.sha256 == selected["asset"]["geometry_sha256"]
    source(Path(__file__), "Figure builder; display-only processing")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "savefig.facecolor": "white",
        }
    )
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4))
    top_h = 0.29 * WIDTH_MM / HEIGHT_MM
    top = [[x, 0.565, 0.29, top_h] for x in [0.02, 0.355, 0.69]]
    image_panel(fig, top[0], raw, "a  Obraz FLUX.2")
    image_panel(fig, top[1], checkerboard_rgba(rgba), "b  RGBA po RMBG")
    mesh_info = mesh_panel(fig, top[2], mesh, camera_t_mesh, camera["intrinsics"])
    for x in [0.337, 0.672]:
        fig.text(x, 0.748, "→", ha="center", va="center", fontsize=13, color=BLUE)
    bottom_h = 0.455 * WIDTH_MM / HEIGHT_MM * 360 / 640
    image_panel(fig, [0.02, 0.15, 0.455, bottom_h], rgb, "d  RGB w SAPIEN")
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#eceff1")
    shown_depth = np.ma.array(depth, mask=~target)
    image_panel(
        fig,
        [0.525, 0.15, 0.455, bottom_h],
        shown_depth,
        "e  Głębia celu w SAPIEN",
        cmap=cmap,
        vmin=0.16,
        vmax=0.26,
    )
    cax = fig.add_axes([0.58, 0.090, 0.34, 0.018])
    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(0.16, 0.26), cmap=cmap),
        cax=cax,
        orientation="horizontal",
        ticks=[0.16, 0.21, 0.26],
    )
    cb.ax.set_xticklabels(["0,16", "0,21", "0,26 m"])
    cb.ax.tick_params(length=2, pad=2)
    cb.outline.set_linewidth(0.5)
    output.mkdir(parents=True, exist_ok=True)
    outputs = []
    for ext in ["pdf", "png"]:
        path = output / f"trellis-pipeline.{ext}"
        metadata = {"CreationDate": None, "ModDate": None} if ext == "pdf" else None
        fig.savefig(path, dpi=DPI, metadata=metadata)
        outputs.append({"path": relative(path), "sha256": digest(path)})
    plt.close(fig)
    ledger = {
        "schema_version": 1,
        "figure_id": "trellis-pipeline",
        "created_date": "2026-09-10",
        "purpose": "Post-study illustration of the historical generated-geometry data pipeline.",
        "selection": {
            "asset_id": ASSET_ID,
            "sample_id": SAMPLE_ID,
            "rule": (
                "Editorial post-study selection of apple_001 for a complete "
                "source chain and a clear illustration, without reference to "
                "reconstruction outcomes; the view is the already selected "
                "historical training observation."
            ),
            "outcome_based_selection": False,
            "representative_sample_claim": False,
            "historical_training_run": relative(RUN),
            "split": "train",
            "active_in_training_schedule": True,
            "training_schedule_occurrences": exposures,
            "manifest_line": manifest_line,
            "instance_id": 9,
        },
        "sources": sources,
        "outputs": outputs,
        "bindings": {
            "original_glb_equals_imported_glb_bytes": True,
            "geometry_sha256_after_historical_preprocessing": processed.fingerprint.sha256,
            "raw_glb_triangle_count": len(mesh.faces),
            "historical_processed_triangle_count": processed.fingerprint.triangle_count,
            "uniform_scale_m_per_source_unit": selected["asset"]["uniform_scale_m_per_source_unit"],
            "target_pixel_count": int(target.sum()),
            "camera_T_mesh": camera_t_mesh.tolist(),
            "camera_intrinsics": camera["intrinsics"],
            "simulator": scene["simulator"],
            "source_image_metadata": raw_meta,
            "background_removal_metadata": bg_meta,
        },
        "display": {
            "width_mm": WIDTH_MM,
            "height_mm": HEIGHT_MM,
            "minimum_font_pt": 9,
            "png_dpi": DPI,
            "panels": [
                "a: original FLUX RGB",
                "b: original RMBG RGBA composited on checkerboard",
                "c: preserved GLB, display-only mesh view without texture",
                "d: preserved full-frame SAPIEN RGB",
                "e: preserved SAPIEN depth restricted to the saved target-instance mask",
            ],
            "mesh": {
                **mesh_info,
                "texture_used": False,
                "pbr_render": False,
                "geometry_edited": False,
                "camera_pose": "saved observation camera; independent display zoom",
                "visibility": "triangle painter ordering, approximate",
                "shading": "two-sided flat directional display shading",
            },
            "rgb_cropped": False,
            "depth_cropped": False,
            "depth_range_m": [0.16, 0.26],
            "target_depth_min_m": float(depth[target].min()),
            "target_depth_max_m": float(depth[target].max()),
            "checkerboard": "Display-only alpha compositing; archived RGBA unchanged.",
            "gray_in_depth": "Pixels outside target instance 9, not missing target depth.",
        },
        "limitations": [
            (
                "Illustrative historical example, not evidence of improvement "
                "from generating geometries or of current PHOTO/HB results."
            ),
            (
                "Model names are known, but all immutable generator weight "
                "revisions and per-object TRELLIS seeds are not established."
            ),
            (
                "The original GLB and RGB-D files are reused without "
                "regeneration. The mesh panel is a new technical visualization "
                "without PBR materials."
            ),
        ],
        "caption_pl": CAPTION,
    }
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "outputs": outputs,
                "evidence": relative(evidence),
                "asset_id": ASSET_ID,
                "sample_id": SAMPLE_ID,
                "geometry_sha256": processed.fingerprint.sha256,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
