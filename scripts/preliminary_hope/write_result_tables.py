"""Create tables and plots only from the complete, frozen three-cohort evaluation."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from scripts.common.paths import digest, read, save
from scripts.common.reconstruction_protocol import (
    CAMPAIGN,
    COHORTS,
    PROTOCOL,
    ROOT,
    validate_protocol,
    validate_run,
)
from scripts.preliminary_hope.bootstrap_objects import balanced_mean, object_ids

ARMS = ("base", "classic", "photo")
MODELS = ("input512", "input_dense", "pretrained", *ARMS)
NAMES = {
    "input512": "INPUT512",
    "input_dense": "INPUT gęste",
    "pretrained": "Bez adaptacji",
    **{a: a.upper() for a in ARMS},
}
PRIMARY = "photo-seed-mean minus base-seed-mean"


def members(model):
    return [f"{model}-seed{s}" for s in range(3)] if model in ARMS else [model]


def aggregate(result, model, budget, metric):
    values = []
    for name in members(model):
        cells = [r["models"][name][budget] for r in result["rows"]]
        if metric in ("f2", "f5", "f10", "p5", "r5"):
            threshold = {"f2": "0.002", "f10": "0.010"}.get(metric, "0.005")
            key = {"p5": "precision", "r5": "recall"}.get(metric, "fscore")
            data = [c["surface"][threshold][key] for c in cells]
        elif metric in ("occluded", "uncovered"):
            subset = "occluded" if metric == "occluded" else "uncovered_by_input512"
            data = [c["gt_subsets"][subset]["recall5"] for c in cells]
        elif metric == "free":
            data = [c["known_free_rate_all"] for c in cells]
        else:
            raise ValueError(metric)
        values.append(balanced_mean(data, result["rows"]))
    # Unavailable GT-subset scores are not zero. Identical GT makes eligibility common to seeds.
    if any(v is None for v in values):
        assert all(v is None for v in values)
        return None
    return float(np.mean(values))


def percent(value):
    return "—" if value is None else f"{100 * value:.2f}".replace(".", ",")


def interval(record):
    return f"{percent(record['mean'])} [{percent(record['ci95'][0])}; {percent(record['ci95'][1])}]"


def subset_coverage(result, subset):
    """Count unique eligible observations/objects, never duplicate them across seeds."""
    eligible = []
    for row in result["rows"]:
        cells = [model["512"]["gt_subsets"][subset] for model in row["models"].values()]
        assert len({c["count"] for c in cells}) == 1, "GT subset must be shared across models"
        assert all((c["recall5"] is not None) == (c["count"] > 0) for c in cells)
        if cells[0]["count"] > 0:
            eligible.append(row)
    return {"observations": len(eligible), "objects": len(set(object_ids(eligible)))}


def count_range(result, model, budget):
    counts = [
        row["models"][name][budget]["count"] for row in result["rows"] for name in members(model)
    ]
    return f"{min(counts)}–{max(counts)}"


def distance_summary(result, model, budget):
    """Descriptive distance conditions on nonempty predictions, with explicit coverage."""
    values, rows = [], []
    for name in members(model):
        for row in result["rows"]:
            cell = row["models"][name][budget]
            value = cell["surface"]["0.005"]["chamfer_mean_m"]
            assert (value is None) == (cell["count"] == 0)
            if value is not None:
                values.append(value)
                rows.append(row)
    mean = balanced_mean(values, rows) if rows else None
    return {
        "mean_mm": None if mean is None else 1000 * mean,
        "eligible": len(rows),
        "total": len(result["rows"]) * len(members(model)),
        "objects": len(set(object_ids(rows))),
    }


def load_complete(portable=False):
    protocol = validate_protocol()
    frozen_path = CAMPAIGN / "frozen-checkpoints.json"
    frozen = read(frozen_path)
    assert frozen["status"] == "frozen" and frozen["protocol_sha256"] == digest(PROTOCOL)
    expected = {f"{a}-seed{s}" for a in ARMS for s in range(3)}
    assert len(frozen["runs"]) == 9 and {r["run"] for r in frozen["runs"]} == expected
    for row in frozen["runs"]:
        assert validate_run(row["run"], protocol) == row
    assert (
        digest(Path(__file__).parents[2].joinpath("scripts/preliminary_hope/bootstrap_objects.py"))
        == frozen["analysis"]["statistics_sha256"]
    )
    results, sources = {}, {}
    amendment = None
    if portable:
        amendment_path = ROOT / "runs/photoreal-portable-20260906/amendment.json"
        amendment = read(amendment_path)
        assert amendment["status"] == "frozen_before_portable_final_inference"
        assert amendment["original_protocol_sha256"] == digest(PROTOCOL)
        assert amendment["frozen_checkpoints_sha256"] == digest(frozen_path)
    for cohort, (_, count, role) in COHORTS.items():
        prefix = "ray-photoreal-portable" if portable else "ray-photoreal"
        path = ROOT / "runs" / f"{prefix}-scores-{cohort}-20260906/run/complete.json"
        result = read(path)
        assert result["status"] == "complete" and result["cohort"] == cohort
        assert result["evaluation_role"] == role and len(result["rows"]) == count
        assert len({r["sample_id"] for r in result["rows"]}) == count
        assert result["frozen_checkpoints_sha256"] == digest(frozen_path)
        assert result["script_sha256"] == frozen["analysis"]["script_sha256"]
        assert result["statistics_sha256"] == frozen["analysis"]["statistics_sha256"]
        assert PRIMARY in result["paired_f5_comparisons"]
        if portable:
            for model in ["pretrained", *[r["run"] for r in frozen["runs"]]]:
                manifest_path = (
                    ROOT
                    / "runs"
                    / f"{prefix}-predictions-{cohort}-20260906"
                    / model
                    / "complete.json"
                )
                manifest = read(manifest_path)
                for key in ("name", "profile_sha256", "wrapper_sha256"):
                    assert (
                        manifest["contract"]["portable_profile"][key] == amendment["profile"][key]
                    )
                records = {r["sample_id"]: r for r in manifest["records"]}
                assert all(
                    r["source_hashes"][model] == records[r["sample_id"]]["output_sha256"]
                    for r in result["rows"]
                )
        results[cohort], sources[cohort] = result, {"path": str(path), "sha256": digest(path)}
        if portable:
            sources[cohort]["amendment_sha256"] = digest(amendment_path)
    return protocol, frozen, results, sources


def tables(results):
    text = [
        "## Końcowe porównanie rekonstrukcji",
        "",
        "Wartości F-score, precyzji i recall w tabelach podano w procentach. "
        "Każdy obiekt ma równą wagę; ramiona adaptacji są średnią trzech "
        "seedów. Różnice i ich przedziały są w punktach procentowych. "
        "INPUT gęste w tabeli z limitem 512 również podlega podpróbkowaniu. "
        "Kolumna natywna zawsze zachowuje pełną liczność; przy limitach "
        "mniejsze chmury także pozostają bez zmiany liczby punktów.",
        "",
    ]
    records = {}
    for cohort, result in results.items():
        role = {
            "hope146": "HOPE — wynik główny",
            "ycbv42": "YCB-Video — diagnostyka",
            "abo24": "ABO — wynik syntetyczny",
        }[cohort]
        text += [
            f"### {role}",
            "",
            "| Model | F5 | Precision5 | Recall5 | F2 | F10 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        records[cohort] = {}
        for model in MODELS:
            record = {
                key: aggregate(result, model, "512", key) for key in ("f5", "p5", "r5", "f2", "f10")
            }
            records[cohort][model] = record
            text.append(
                "| " + NAMES[model] + " | " + " | ".join(percent(v) for v in record.values()) + " |"
            )
        text += [
            "",
            "| Sparowana różnica F5 | Średnia i 95% przedział obiektowy [pp] |",
            "|---|---:|",
        ]
        for key in (
            PRIMARY,
            "photo-seed-mean minus classic-seed-mean",
            "photo-seed-mean minus input512",
            "photo-seed-mean minus pretrained",
        ):
            label = (
                key.replace("-seed-mean", "")
                .replace("minus", "–")
                .upper()
                .replace("PRETRAINED", "bez adaptacji")
            )
            text.append(f"| {label} | {interval(result['paired_f5_comparisons'][key])} |")
        text += [
            "",
            "| Seed | BASE F5 | CLASSIC F5 | PHOTO F5 | PHOTO–BASE [pp] |",
            "|---|---:|---:|---:|---:|",
        ]
        for seed in range(3):
            values = [aggregate(result, f"{a}-seed{seed}", "512", "f5") for a in ARMS]
            text.append(
                f"| {seed} | "
                + " | ".join(percent(v) for v in [*values, values[2] - values[0]])
                + " |"
            )
        text += [
            "",
            "| Model | F5 przy limicie 16384 | F5 natywne | Puste wyjścia / obserwacje modeli |",
            "|---|---:|---:|---:|",
        ]
        for model in MODELS:
            empty = sum(result["summary"][name]["native"]["empty"] for name in members(model))
            total = len(result["rows"]) * len(members(model))
            text.append(
                f"| {NAMES[model]} | {percent(aggregate(result, model, '16384', 'f5'))} | "
                f"{percent(aggregate(result, model, 'native', 'f5'))} | {empty}/{total} |"
            )
        text += [
            "",
            "| Model | Liczba punktów: limit 512 | Limit 16384 | Natywna |",
            "|---|---:|---:|---:|",
        ]
        for model in MODELS:
            text.append(
                "| "
                + NAMES[model]
                + " | "
                + " | ".join(count_range(result, model, b) for b in ("512", "16384", "native"))
                + " |"
            )
        text += [
            "",
            (
                "Zakresy obejmują wszystkie obserwacje i wszystkie seedy danego "
                "ramienia, w tym puste wyjścia."
            ),
            "",
            (
                "| Model | Chamfer 512 [mm] | Chamfer 16384 [mm] | Chamfer "
                "natywny [mm] | Dostępne / wszystkie |"
            ),
            "|---|---:|---:|---:|---:|",
        ]
        for model in MODELS:
            distances = [distance_summary(result, model, b) for b in ("512", "16384", "native")]
            assert len({(d["eligible"], d["objects"]) for d in distances}) == 1
            records[cohort][model]["chamfer"] = dict(
                zip(("512", "16384", "native"), distances, strict=True)
            )
            formatted = [
                "—" if d["mean_mm"] is None else f"{d['mean_mm']:.2f}".replace(".", ",")
                for d in distances
            ]
            d = distances[0]
            text.append(
                "| "
                + NAMES[model]
                + " | "
                + " | ".join(formatted)
                + f" | {d['eligible']}/{d['total']} |"
            )
        text += [
            "",
            "Chamfer jest średnią dwóch kierunkowych średnich odległości, bez kwadratowania. "
            "Podano go warunkowo dla niepustych predykcji: najpierw uśrednia się dostępne "
            "obserwacje i seedy danego obiektu, następnie nadaje równą wagę obiektom. "
            "Kolumna dostępności liczy pary obserwacja–model, więc dla ramion adaptacji "
            "obejmuje trzy seedy. Puste wyjście pozostaje porażką w F-score i tabeli liczebności; "
            "nie otrzymuje zerowej odległości. Odległość warunkowa nie zastępuje głównego F5.",
            "",
            (
                "| Model | Recall5 poza INPUT512 | Recall5 części zasłoniętej | "
                "Punkty w znanej wolnej przestrzeni |"
            ),
            "|---|---:|---:|---:|",
        ]
        for model in MODELS:
            text.append(
                "| "
                + NAMES[model]
                + " | "
                + " | ".join(
                    percent(aggregate(result, model, "512", metric))
                    for metric in ("uncovered", "occluded", "free")
                )
                + " |"
            )
        text += [
            "",
            "Wszystkie trzy miary w powyższej tabeli dotyczą limitu 512 punktów. "
            "Poza INPUT512 oznacza odległość referencji większą niż 5 mm od tych punktów, "
            "więc obejmuje również luki próbkowania widocznej powierzchni. "
            "Odsetek punktów w wolnej przestrzeni ma w mianowniku całą predykcję; "
            "dla pustej predykcji wynosi zero i wymaga interpretacji razem z liczbą pustych wyjść.",
            "",
        ]
        for subset, label in (
            ("uncovered_by_input512", "Poza INPUT512"),
            ("occluded", "Część zasłonięta"),
        ):
            coverage = subset_coverage(result, subset)
            objects = len(set(object_ids(result["rows"])))
            text += [
                (
                    f"{label}: metryka dostępna dla "
                    f"{coverage['observations']}/{len(result['rows'])} "
                    f"obserwacji i {coverage['objects']}/{objects} "
                    f"obiektów. "
                    "Liczebności są wspólne dla modeli i nie są mnożone przez liczbę seedów. "
                    "Braki podzbioru pozostają poza średnią; zero recall z pustej "
                    "predykcji pozostaje w średniej."
                ),
                "",
            ]
        primary = result["paired_f5_comparisons"][PRIMARY]
        text += [
            "",
            f"Dodatnia średnia różnica wystąpiła dla {primary['objects_positive']} "
            f"z {primary['objects']} obiektów, ujemna dla {primary['objects_negative']}. "
            "Przedział warunkuje na trzech ustalonych seedach treningu.",
            "",
        ]
        if "scene_sensitivity" in primary:
            scene = primary["scene_sensitivity"]
            leave = [v["mean"] for v in scene["leave_one_scene_out"].values()]
            text += [
                f"Analiza wrażliwości po scenach ({scene['clusters']} grup) daje zakres "
                f"95% [{percent(scene['ci95'][0])}; {percent(scene['ci95'][1])}] pp. "
                f"Po pominięciu kolejno jednej sceny średnia wynosi od {percent(min(leave))} "
                f"do {percent(max(leave))} pp. Nie jest to łączny przedział populacyjny po "
                "obiektach i scenach.",
                "",
            ]
    return "\n".join(text), records


def plot(results, output):
    sys.path.insert(0, str(ROOT / "build/thesis-figures/deps"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    primary = results["hope146"]["paired_f5_comparisons"][PRIMARY]
    objects = primary["object_deltas"]
    labels = sorted(objects, key=int)
    values = np.array([objects[k] for k in labels]) * 100
    fig, ax = plt.subplots(figsize=(9.2, 4.4), layout="constrained")
    ax.bar(np.arange(len(labels)), values, color=np.where(values >= 0, "#157a6e", "#b54545"))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(primary["mean"] * 100, color="#173d77", linestyle="--", label="Średnia po obiektach")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=90)
    ax.set_xlabel("Identyfikator obiektu HOPE (wszystkie 28)")
    ax.set_ylabel("PHOTO – BASE: F5 [pp]")
    ax.legend()
    fig.savefig(output / "hope-object-effects.png", dpi=220)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), layout="constrained", sharey=True)
    for ax, (cohort, result) in zip(axes, results.items(), strict=True):
        for a, color in zip(ARMS, ("#64748b", "#dc8c26", "#157a6e"), strict=True):
            ax.plot(
                range(3),
                [100 * aggregate(result, f"{a}-seed{s}", "512", "f5") for s in range(3)],
                "o-",
                label=a.upper(),
                color=color,
            )
        ax.axhline(
            100 * aggregate(result, "pretrained", "512", "f5"),
            color="#173d77",
            linestyle=":",
            label="Bez adaptacji",
        )
        ax.axhline(
            100 * aggregate(result, "input512", "512", "f5"),
            color="black",
            linestyle="--",
            label="INPUT512",
        )
        ax.set_title(cohort.upper())
        ax.set_xticks(range(3))
        ax.set_xlabel("Seed treningu")
    axes[0].set_ylabel("F5 przy 512 punktach [%]")
    axes[-1].legend(fontsize=7)
    fig.savefig(output / "all-cohort-seeds.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "runs/ray-photoreal-report-20260906")
    parser.add_argument("--portable", action="store_true")
    args = parser.parse_args()
    assert not args.output.exists(), "Do not overwrite a reviewed report"
    protocol, frozen, results, sources = load_complete(args.portable)
    text, records = tables(results)
    args.output.mkdir()
    (args.output / "results-fragment.md").write_text(text, encoding="utf-8")
    plot(results, args.output)
    durations = {
        r["run"]: read(CAMPAIGN / r["run"] / "complete.json")["seconds_this_attempt"]
        for r in frozen["runs"]
    }
    save(
        args.output / "complete.json",
        {
            "status": "complete_report_requires_interpretation",
            "protocol_sha256": digest(PROTOCOL),
            "source_scores": sources,
            "script_sha256": digest(__file__),
            "table_values": records,
            "primary": results["hope146"]["paired_f5_comparisons"][PRIMARY],
            "training_loop_seconds": durations,
            "training_loop_total_hours": sum(durations.values()) / 3600,
            "timing_scope": (
                "Recorded training attempts only; excludes process/model startup, "
                "generation, inference, scoring and electricity cost."
            ),
            "photo_images": protocol["accepted_photo_images"],
            "artifacts": {p.name: digest(p) for p in args.output.iterdir() if p.is_file()},
        },
    )
    print(json.dumps({"report": str(args.output), "sources": sources}))


if __name__ == "__main__":
    main()
