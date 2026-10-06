"""Draw the used RaySt3R architecture; inspect frozen source, never run a model."""

import hashlib
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

from scripts.common.paths import ROOT as WORKSPACE_ROOT
from scripts.common.paths import script_help

script_help(__doc__, __name__)

ROOT = WORKSPACE_ROOT

matplotlib.use("Agg")

OUT = ROOT / "artifacts/figures/assets"
LEDGER = ROOT / "artifacts/figures/evidence/method-protocol-map.json"
BUNDLE = "runs/research-evolution-campaign-20260907/matrix-bundle-v3/matrix-training-bundle-v2.zip"
CONFIG = "runs/ray-adaptation-20260906/checkpoint-model-configuration.json"
PILOT = "runs/ray-adaptation-20260906/base-seed0/trainable-parameters.json"
MAIN = (
    "runs/research-evolution-campaign-20260907/MAIN-independent-owner-"
    "v1/seed0/unpacked/BASE/binding.json"
)
TRAINER = "scripts/train/train_ray_variants.py"
NAME = "method-architecture"
INK, BLUE, GOLD, GREY = "#243444", "#246899", "#a86512", "#74818c"
STYLES = {"frozen": (BLUE, "#edf4f8"), "trained": (GOLD, "#fbf1df"), "operation": (GREY, "#f6f7f8")}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def source_checks():
    trainable = read(PILOT)
    assert trainable == read(MAIN)["trainable"]
    counts = defaultdict(int)
    for name, count in trainable.items():
        group = (
            ".".join(name.split(".")[:2])
            if name.startswith("decoder_blocks.")
            else name.split(".")[0]
        )
        counts[group] += count
    assert dict(counts) == {
        "decoder_blocks.10": 9448704,
        "decoder_blocks.11": 9448704,
        "pts_head": 20176834,
        "classifier_head": 20176834,
    }
    assert sum(counts.values()) == 59251076
    config = read(CONFIG)["model"]
    assert all(
        s in config
        for s in [
            "decoder_depth=12",
            "depth=4",
            "dpt_depth",
            "dpt_mask",
            "dino_layers=[4, 11, 17, 23]",
        ]
    )
    members = [
        "models/rayquery.py",
        "models/blocks.py",
        "models/heads/dpt_head.py",
        "models/heads/postprocess.py",
        "utils/batch_prep.py",
        "utils/geometry.py",
        "engine.py",
        "eval_wrapper/eval.py",
    ]
    archive_sources = []
    with zipfile.ZipFile(ROOT / BUNDLE) as z:
        sources = {m: z.read("work/vendor/rayst3r/" + m) for m in members}
        rq = sources["models/rayquery.py"].decode()
        assert "torch.cat([pointmaps,dino_features],dim=1)" in rq
        assert "self.pts_head(rays, self.imshape)" in rq
        assert "self.classifier_head(rays, self.imshape)" in rq
        dpt = sources["models/heads/dpt_head.py"].decode()
        assert "hooks_idx=[0, l2*2//4, l2*3//4, l2]" in dpt
        hooks = [0, 11 * 2 // 4, 11 * 3 // 4, 11]
        assert hooks == [0, 5, 8, 11]
        for member, data in sources.items():
            archive_sources.append(
                {
                    "path": BUNDLE,
                    "archive_member": "work/vendor/rayst3r/" + member,
                    "member_sha256": hashlib.sha256(data).hexdigest(),
                }
            )
    return dict(counts), archive_sources


def draw():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "text.color": INK,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    fig = plt.figure(figsize=(160 / 25.4, 157 / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 160), ylim=(157, 0))
    ax.axis("off")

    def txt(x, y, text, **kwargs):
        return ax.text(x, y, text, ha="center", va="center", fontsize=9, linespacing=1.3, **kwargs)

    def box(x, y, w, h, text, style="operation", bold=False):
        edge, fill = STYLES[style]
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0,rounding_size=1.1",
                edgecolor=edge,
                facecolor=fill,
                linewidth=1.1 if style == "trained" else 0.8,
            )
        )
        txt(x + w / 2, y + h / 2, text, weight="bold" if bold else "normal")

    def line(points, color=GREY):
        ax.plot(*zip(*points, strict=False), color=color, lw=0.85, solid_capstyle="round")

    def arrow(points, color=GREY):
        if len(points) > 2:
            line(points[:-1], color)
        ax.annotate(
            "",
            xy=points[-1],
            xytext=points[-2],
            arrowprops={
                "arrowstyle": "-|>",
                "lw": 0.85,
                "color": color,
                "shrinkA": 0,
                "shrinkB": 0.2,
            },
        )

    for x, label, style in [
        (4, "Zamrożone wagi", "frozen"),
        (62, "Dostrajane wagi (59,3 mln)", "trained"),
    ]:
        edge, fill = STYLES[style]
        ax.add_patch(Rectangle((x, 2), 5, 3.5, edgecolor=edge, facecolor=fill, linewidth=1))
        ax.text(x + 7, 3.75, label, va="center", fontsize=9)

    box(4, 12, 46, 14, "Maskowane RGB\nBASE / CLASSIC / PHOTO")
    box(57, 12, 46, 14, "Głębia → punkty 3D\nMaska i parametry kamer")
    box(110, 12, 46, 14, "Promienie\nnowej kamery")
    arrow([(27, 26), (27, 32)])
    arrow([(80, 26), (80, 32)])
    arrow([(133, 26), (133, 32)])
    box(4, 32, 46, 20, "DINOv2\n+ projekcja cech RGB", "frozen")
    box(57, 32, 46, 20, "Enkoder mapy\npunktów 3D\n4 bloki", "frozen")
    box(110, 32, 46, 20, "Enkoder promieni\n4 bloki", "frozen")
    arrow([(27, 52), (27, 59)])
    arrow([(80, 52), (80, 59)])
    box(4, 59, 99, 13, "Połączenie cech RGB i geometrii\nKontekst dla każdego bloku dekodera")
    arrow([(133, 52), (133, 59)])
    box(110, 59, 46, 16, "Dekoder\nbloki 1-10", "frozen")
    arrow([(133, 75), (133, 81)])
    box(110, 81, 46, 16, "Dekoder\nbloki 11-12", "trained", bold=True)
    arrow([(103, 66), (110, 66)])
    arrow([(103, 66), (106.5, 66), (106.5, 89), (110, 89)])
    txt(
        53.5,
        86.5,
        "Każdy blok dekodera:\nuwaga między promieniami, uwaga\ndo cech RGB i geometrii oraz MLP.",
    )

    # Both DPT heads use outputs 0, 5, 8 and 11, not only the last block.
    line([(156, 71), (158.5, 71), (158.5, 106), (40, 106)])
    line([(156, 93), (158.5, 93)])
    txt(
        81,
        106,
        "Cechy bloków 1, 6, 9 i 12",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 2.2},
    )
    arrow([(40, 106), (40, 113)])
    arrow([(120, 106), (120, 113)])
    box(4, 113, 72, 13, "Głowa DPT\ngłębia i pewność", "trained", bold=True)
    box(84, 113, 72, 13, "Głowa DPT\nmaska obiektu", "trained", bold=True)
    arrow([(40, 126), (40, 131)])
    arrow([(120, 126), (120, 131)])
    txt(40, 134, "Głębia i pewność nowego widoku")
    txt(120, 134, "Maska nowego widoku")
    arrow([(40, 137), (40, 142)])
    arrow([(120, 137), (120, 142)])
    box(
        4,
        142,
        152,
        12,
        (
            "Odtworzenie punktów 3D, filtracja i połączenie widoków\nChmura "
            "punktów rekonstruowanego przedmiotu"
        ),
    )
    outputs = {}
    for ext in ["pdf", "svg", "png"]:
        path = OUT / f"{NAME}.{ext}"
        kwargs = {"dpi": 350} if ext == "png" else {}
        if ext == "pdf":
            kwargs["metadata"] = {
                "Title": "RaySt3R: architektura i zakres dostrajania",
                "Author": "Opracowanie własne na podstawie publicznego RaySt3R",
                "CreationDate": None,
                "ModDate": None,
            }
        if ext == "svg":
            kwargs["metadata"] = {"Date": None}
        fig.savefig(path, **kwargs)
        outputs[ext] = {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path)}
    plt.close(fig)
    return outputs


def main():
    counts, archive_sources = source_checks()
    outputs = draw()
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    ledger.setdefault("figures", {})[NAME] = {
        "date": "2026-09-12",
        "authorship": "Author-drawn schematic of the external RaySt3R architecture",
        "scope": (
            "Module-level data flow of the configuration used in both "
            "experiments; no new experiment"
        ),
        "script": Path(__file__).relative_to(ROOT).as_posix(),
        "script_sha256": sha(__file__),
        "outputs": outputs,
        "width_mm": 160,
        "height_mm": 157,
        "font_size_pt": 9,
        "source_files": [
            {"path": p, "sha256": sha(ROOT / p)} for p in [BUNDLE, CONFIG, PILOT, MAIN, TRAINER]
        ],
        "archive_sources": archive_sources,
        "trainable_parameter_counts": counts,
        "trainable_total": sum(counts.values()),
        "decoder_block_numbering": "One-based in the figure; zero-based in code",
        "dpt_decoder_hooks_zero_based": [0, 5, 8, 11],
        "dpt_decoder_hooks_figure": [1, 6, 9, 12],
        "frozen": ["DINOv2", "dino_proj", "pointmap_enc", "ray_enc", "decoder_blocks.0-9"],
        "simplifications": [
            (
                "Module-level figure omits internal patch embeddings, positional "
                "encodings, normalization and DPT convolutional detail."
            ),
            (
                "RGB has four DINO layers concatenated before the frozen "
                "projection; figure groups these in one box."
            ),
            (
                "Mask head also emits conf_classifier, unused by the configured "
                "loss and reconstruction filtering; only used outputs are drawn."
            ),
            (
                "Final point-cloud operation summarizes multi-view inference, not "
                "the training loss or supervision path."
            ),
            (
                "Network architecture belongs to RaySt3R authors; chosen "
                "adaptation scope belongs to this thesis."
            ),
        ],
        "semantic_review": {
            "status": "verified",
            "method": (
                "Read frozen forward functions, DecoderBlock, DPT head hooks, "
                "postprocess, batch preparation and trainability configuration; "
                "sum pilot and main parameter ledgers."
            ),
        },
        "visual_review": {"status": "pending"},
    }
    LEDGER.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"figure": NAME, "trainable": sum(counts.values()), "outputs": outputs}, indent=2
        )
    )


if __name__ == "__main__":
    main()
