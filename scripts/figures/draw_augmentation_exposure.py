"""Make thesis F6 from frozen contracts and the completed RGB exposure audit.

CPU-only, post-study vector diagram. No training, generation, or score selection.
The full canvas is 160 mm wide; PDF/SVG preserve text and vector primitives.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from scripts.common.paths import ROOT as WORKSPACE_ROOT
from scripts.common.paths import script_help

script_help(__doc__, __name__)


ROOT = WORKSPACE_ROOT

matplotlib.use("Agg")

OUT = ROOT / "artifacts/figures/assets"
LEDGER = ROOT / "artifacts/figures/evidence/exposure-figure.json"
INK, EDGE = "#243444", "#8f9da8"
BLUE, GOLD, PURPLE = "#326b8b", "#b37b2a", "#88559c"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: str):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def main():
    sources = {
        "campaign": "runs/research-evolution-campaign-20260907/frozen-campaign-v3/campaign.json",
        "exposure": (
            "runs/research-evolution-campaign-20260907/frozen-campaign-v3/expo"
            "sure-audit/complete.json"
        ),
        "pilot": "runs/ray-adaptation-20260906/photoreal-protocol.json",
        "method": "artifacts/figures/evidence/method-protocol-map.json",
    }
    campaign, exposure, pilot = (load(sources[x]) for x in ("campaign", "exposure", "pilot"))
    n, g, steps = (
        campaign[k] for k in ("training_observations", "training_geometries", "steps_per_run")
    )
    admitted = exposure["accepted_count"]
    assert (n, g, steps, admitted) == (6400, 320, 25600, 3344)
    assert n // g == 20 and steps // n == 4
    for seed in ("0", "1", "2"):
        for arm in ("CLASSIC", "PHOTO"):
            row = exposure["totals"][seed][arm]
            assert (row["exposures"], row["augmentation_attempts"], row["model_RGB_changed"]) == (
                25600,
                12800,
                6688,
            )
            assert row["model_changed_exposure_fraction"] == 0.26125
        assert exposure["totals"][seed]["BASE"]["model_RGB_changed"] == 0
    assert pilot["accepted_photo_images"] == 54
    assert pilot["common_training_contract"]["steps"] == 2160
    changed = admitted * 2
    fallback = (n - admitted) * 2
    assert changed == 6688 and fallback == 6112 and changed + fallback == n * 2

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.2,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "text.color": INK,
            "savefig.facecolor": "white",
        }
    )
    fig = plt.figure(figsize=(160 / 25.4, 116 / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 160), ylim=(116, 0))
    ax.axis("off")

    def label(x, y, text, size=10.2, weight="normal", ha="center", color=INK):
        return ax.text(
            x,
            y,
            text,
            ha=ha,
            va="center",
            fontsize=size,
            fontweight=weight,
            color=color,
            linespacing=1.35,
        )

    def box(x, y, w, h, text, edge=EDGE, fill="#f5f7f9", size=10.2, weight="normal"):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0,rounding_size=1.5",
                linewidth=0.85,
                edgecolor=edge,
                facecolor=fill,
            )
        )
        label(x + w / 2, y + h / 2, text, size=size, weight=weight)

    def arrow(x1, y1, x2, y2, color=EDGE):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, linewidth=0.9, color=color
            )
        )

    label(2, 5, "Częstość użycia zmienionego RGB", 10.5, "bold", "left")
    box(
        2,
        12,
        156,
        16,
        "Wspólne 320 geometrii × 20 obserwacji = 6400 wejść\n"
        "BASE, CLASSIC i PHOTO: po trzy powtórzenia uczenia",
    )
    label(80, 36, "6400 obserwacji × 4 użycia = 25 600 aktualizacji na trening", weight="bold")
    box(2, 44, 61, 17, "2 zwykłe użycia źródła\n6400 × 2 = 12 800")
    box(70, 44, 88, 17, "2 próby zmiany RGB w CLASSIC i PHOTO\n6400 × 2 = 12 800 prób")
    arrow(90, 61, 70, 70)
    arrow(137, 61, 137, 70)
    box(
        2,
        70,
        99,
        17,
        "3344 obserwacje z edycją × 2 = 6688\nzmian RGB widocznych dla modelu",
        PURPLE,
        "#f3eef7",
    )
    box(108, 70, 50, 17, "3056 × 2 = 6112\npowrotów do źródła")
    label(80, 95, "BASE: 0%     CLASSIC = PHOTO: 6688 / 25 600 ≈ 26,1%", 10.5, "bold")
    label(
        80,
        107,
        "Badanie wstępne: 36 × 60 = 2160 aktualizacji; CLASSIC: 50%\n"
        "PHOTO: 54 edycje × 15 użyć / 2160 = 37,5% (rzadziej niż CLASSIC)",
    )

    OUT.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for suffix in ("pdf", "svg", "png"):
        path = OUT / f"method-exposure.{suffix}"
        kwargs = {"dpi": 350} if suffix == "png" else {}
        fig.savefig(path, **kwargs)
        outputs[path.relative_to(ROOT).as_posix()] = digest(path)
    plt.close(fig)
    ledger = {
        "purpose": (
            "Post-study didactic F6 diagram of existing method and exact RGB "
            "exposure; no new model computation."
        ),
        "source_files": {
            name: {"path": path, "sha256": digest(ROOT / path)} for name, path in sources.items()
        },
        "source_fields": {
            "campaign": ["/training_geometries", "/training_observations", "/steps_per_run"],
            "exposure": [
                "/accepted_count",
                "/totals/{0,1,2}/{BASE,CLASSIC,PHOTO}/exposures",
                "/totals/{0,1,2}/{CLASSIC,PHOTO}/augmentation_attempts",
                "/totals/{0,1,2}/{BASE,CLASSIC,PHOTO}/model_RGB_changed",
                "/totals/{0,1,2}/{CLASSIC,PHOTO}/model_changed_exposure_fraction",
            ],
            "pilot": ["/accepted_photo_images", "/common_training_contract/steps"],
            "model_and_historical_schedule": (
                "Source selectors in method-protocol-map.json; same method as manuscript04/05."
            ),
        },
        "main": {
            "geometries": g,
            "observations_per_geometry": n // g,
            "observations": n,
            "uses_per_observation": steps // n,
            "updates_per_run": steps,
            "planned_attempts_CLASSIC_and_PHOTO": 12800,
            "admitted_observations": admitted,
            "fallback_observations": n - admitted,
            "changed_exposures_each_arm": changed,
            "fallback_attempts_each_arm": fallback,
            "fraction_changed_CLASSIC_and_PHOTO": changed / steps,
            "fraction_changed_BASE": 0.0,
        },
        "pilot": {
            "geometries_and_observations": 36,
            "uses_per_observation": 60,
            "updates_per_run": 2160,
            "accepted_edits": 54,
            "uses_per_edit": 15,
            "CLASSIC_changed_fraction": 0.5,
            "PHOTO_changed_fraction": 54 * 15 / 2160,
        },
        "interpretation": [
            "Counts and fractions are per one training run.",
            "The admission mask is shared by main CLASSIC and PHOTO.",
            (
                "The header explicitly identifies three repetitions per variant; "
                "counts below apply to one run. The displayed percentage is "
                "rounded; the exact fraction remains in this ledger."
            ),
            "The diagram does not assert statistical independence of observations or seeds.",
        ],
        "layout": {
            "width_mm": 160,
            "height_mm": 116,
            "minimum_font_pt": 10.2,
            "minimum_font_pt_at_150mm_inclusion": 10.2 * 150 / 160,
            "vector_text": True,
            "png_dpi": 350,
        },
        "script_sha256": digest(Path(__file__)),
        "outputs": outputs,
    }
    LEDGER.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"outputs": list(outputs), "ledger": str(LEDGER.relative_to(ROOT))}))


if __name__ == "__main__":
    main()
