"""Export complete supplemental measurements, without selecting outcomes.

Run after analysis-v1/complete.json exists. The report preserves both point
budgets, every model and the two predefined HB populations. It performs no
new scoring or inference and writes no manuscript source.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.common.paths import ROOT, digest, read, save, script_help

script_help(__doc__, __name__)


BASE = ROOT / "runs/thesis-supplement-20260910"
ANALYSIS = BASE / "analysis-v1"
OUT = BASE / "numeric-report-v1"
ARMS = ("BASE", "CLASSIC", "PHOTO")
MODELS = tuple(f"ray-{arm}-seed{seed}" for arm in ARMS for seed in range(3))
BUDGETS = ("cap16384", "cap512")
CELLS = []


def ptr(*parts):
    return "/" + "/".join(str(x).replace("~", "~0").replace("/", "~1") for x in parts)


def at(document, path):
    value = document
    for encoded in path[1:].split("/"):
        token = encoded.replace("~1", "/").replace("~0", "~")
        value = value[int(token)] if isinstance(value, list) else value[token]
    return value


def fmt(value, factor=1, places=2, signed=False):
    if value is None:
        return "N/D"
    number = value * factor
    return format(number, ("+" if signed else "") + f".{places}f").replace(".", ",")


def cell(data, table, row, column, path, *, factor=1, places=2, signed=False, interval=False):
    value = at(data, path)
    if interval:
        display = "[" + "; ".join(fmt(x, factor, places, signed) for x in value) + "]"
    else:
        display = fmt(value, factor, places, signed)
    CELLS.append(
        {
            "table": table,
            "row": row,
            "column": column,
            "source": "summaries.json",
            "json_pointer": path,
            "source_value": value,
            "displayed": display,
            "factor": factor,
            "decimal_places": places,
            "signed": signed,
            "interval": interval,
        }
    )
    return display


def table(lines, header, rows):
    lines += [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    lines.append("")


def model_name(mid):
    if mid == "INPUT":
        return "INPUT"
    if mid == "ray-pretrained":
        return "RaySt3R publiczny"
    return mid.removeprefix("ray-").replace("-seed", ", ziarno ")


def export_full_metrics(data):
    """Export existing summaries and denominators, without recomputing scores."""
    fields = (
        "condition",
        "population",
        "point_budget",
        "comparison",
        "method_id",
        "metric",
        "mean",
        "defined_observations",
        "total_observations",
        "defined_objects",
        "total_objects",
        "json_pointer",
    )
    rows, counts = [], []

    def add(condition, label, budget, comparison, mid, metric, path):
        value = at(data, ptr(*path))
        rows.append(
            dict(zip(fields[:6], (condition, label, budget, comparison, mid, metric), strict=False))
            | {key: value[key] for key in fields[6:11]}
            | {"json_pointer": ptr(*path)}
        )

    for condition, populations in data.items():
        for label, population in populations.items():
            for budget, scores in population["scores"].items():
                for mid, method in scores["methods"].items():
                    base = (condition, label, "scores", budget, "methods", mid)
                    for metric in method["metrics"]:
                        add(
                            condition,
                            label,
                            budget,
                            "absolute",
                            mid,
                            metric,
                            (*base, "metrics", metric),
                        )
                    for key in ("empty", "under512"):
                        counts.append(
                            [
                                condition,
                                label,
                                budget,
                                mid,
                                key,
                                method[key],
                                scores["observations"],
                                ptr(*base, key),
                            ]
                        )
                    for status, count in method["status_counts"].items():
                        counts.append(
                            [
                                condition,
                                label,
                                budget,
                                mid,
                                "status:" + status,
                                count,
                                scores["observations"],
                                ptr(*base, "status_counts", status),
                            ]
                        )
            for budget, paired in population.get("paired_RGB", {}).items():
                for mid, method in paired["methods"].items():
                    base = (
                        condition,
                        label,
                        "paired_RGB",
                        budget,
                        "methods",
                        mid,
                        "neutral_minus_original",
                    )
                    for metric in method["neutral_minus_original"]:
                        add(
                            condition,
                            label,
                            budget,
                            "neutral_minus_original",
                            mid,
                            metric,
                            (*base, metric),
                        )
            for mid, changes in population.get("native_changes", {}).items():
                for metric in changes["metrics"]:
                    add(
                        condition,
                        label,
                        "native",
                        "neutral_minus_original_cloud",
                        mid,
                        metric,
                        (
                            condition,
                            label,
                            "native_changes",
                            mid,
                            "metrics",
                            metric,
                            "object_balanced",
                        ),
                    )
    with (OUT / "metrics-per-model.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (OUT / "failure-counts.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "condition",
                "population",
                "point_budget",
                "method_id",
                "counter",
                "count",
                "observations",
                "json_pointer",
            )
        )
        writer.writerows(counts)
    return {"metric_rows": len(rows), "failure_counter_rows": len(counts)}


def rgb_quality_rows(data, base, ident):
    rows = []
    for mid in MODELS:
        row = [model_name(mid)]
        for metric in (
            "precision@0.005",
            "recall@0.005",
            "GT_occluded_recall5",
            "GT_uncovered_by_dense_input_recall5",
        ):
            path = (*base, "methods", mid, "neutral_minus_original", metric)
            display = cell(data, ident, mid, metric, ptr(*path, "mean"), factor=100, signed=True)
            if metric.startswith("GT_"):
                n = cell(
                    data,
                    ident,
                    mid,
                    metric + " określone obserwacje",
                    ptr(*path, "defined_observations"),
                    places=0,
                )
                total = cell(
                    data,
                    ident,
                    mid,
                    metric + " wszystkie obserwacje",
                    ptr(*path, "total_observations"),
                    places=0,
                )
                display += f" ({n}/{total})"
            row.append(display)
        rows.append(row)
    return rows


def f5_contrast_rows(data, base, ident, names):
    rows = []
    for name in names:
        path = (*base, name)
        rows.append(
            [
                name.replace("ray-pretrained", "model publiczny"),
                cell(
                    data,
                    ident,
                    name,
                    "różnica pp",
                    ptr(*path, "paired_objects", "difference"),
                    factor=100,
                    signed=True,
                ),
                cell(
                    data,
                    ident,
                    name,
                    "CI obiekty pp",
                    ptr(*path, "paired_objects", "object_bootstrap95"),
                    factor=100,
                    signed=True,
                    interval=True,
                ),
                cell(
                    data,
                    ident,
                    name,
                    "CI obiekty+treningi pp",
                    ptr(*path, "object_and_seed_bootstrap95"),
                    factor=100,
                    signed=True,
                    interval=True,
                ),
                "; ".join(
                    cell(
                        data,
                        ident,
                        name,
                        f"ziarno {i} pp",
                        ptr(*path, "per_seed_object_mean", i),
                        factor=100,
                        signed=True,
                    )
                    for i in range(3)
                ),
            ]
        )
    return rows


def main():
    assert not OUT.exists(), "Preserve an existing report; use a new version for revisions."
    proof = read(ANALYSIS / "complete.json")
    assert proof["status"] == "complete_fixed_supplement_summaries_no_test_driven_selection"
    source = ROOT / proof["summaries"]["path"]
    assert source.resolve() == (ANALYSIS / "summaries.json").resolve()
    assert digest(source) == proof["summaries"]["sha256"]
    assert digest(ROOT / proof["F5_csv"]["path"]) == proof["F5_csv"]["sha256"]
    assert len(proof["all30_parity_checks"]) == 30 and all(
        x["passed"] for x in proof["all30_parity_checks"]
    )
    data = read(source)
    lines = [
        "# Kontrole RGB i holdoutu — kompletne wyniki",
        "",
        (
            "Analiza uzupełniająca po poznaniu wyniku głównego HB. Zachowano "
            "wszystkie dziewięć końcowych modeli, oba limity punktów i "
            "wszystkie obserwacje. Nie dobierano checkpointów, ziaren ani "
            "podzbiorów według nowych wyników."
        ),
        "",
        (
            "Wartości F5, precyzji i pokrycia są w procentach, różnice w "
            "punktach procentowych, odległości w milimetrach. Średnie "
            "najpierw równoważą obserwacje danego obiektu, następnie obiekty. "
            "Przedziały z losowania obiektów warunkują na zachowanych "
            "treningach; przedziały z losowania obiektów i treningów przy "
            "zaledwie trzech ziarnach mają ograniczoną zdolność opisu "
            "przyszłego uczenia. Nie są przedziałami wyniku pojedynczego "
            "nowego treningu."
        ),
        "",
        "## Holdout: 36 geometrii, 720 obserwacji",
        "",
        (
            "Każdy model porównano z INPUT i publicznym RaySt3R. Wszystkie "
            "720 wejść spełniają próg 512 punktów. Holdout jest rozłączny z "
            "własnym TRAIN; nie wykazano pełnej niezależności od publicznego "
            "pretreningu."
        ),
        "",
    ]
    contrasts = (
        "PHOTO-BASE",
        "CLASSIC-BASE",
        "PHOTO-CLASSIC",
        "BASE-ray-pretrained",
        "CLASSIC-ray-pretrained",
        "PHOTO-ray-pretrained",
        "BASE-INPUT",
        "CLASSIC-INPUT",
        "PHOTO-INPUT",
    )
    holdout_models = ("INPUT", "ray-pretrained", *MODELS)
    for budget in BUDGETS:
        base = ("synthetic-holdout720", "all", "scores", budget)
        population = at(data, ptr(*base))
        assert population["observations"] == 720 and population["objects"] == 36
        assert set(population["methods"]) == set(holdout_models)
        ident = "holdout-models-" + budget
        lines += ["### Limit " + ("16 384" if budget == "cap16384" else "512") + " punktów", ""]
        rows = []
        for mid in holdout_models:
            path = (*base, "methods", mid)
            row = [model_name(mid)]
            for metric in ("fscore@0.005", "precision@0.005", "recall@0.005"):
                row.append(
                    cell(
                        data, ident, mid, metric, ptr(*path, "metrics", metric, "mean"), factor=100
                    )
                )
            row.append(cell(data, ident, mid, "puste", ptr(*path, "empty"), places=0))
            rows.append(row)
        table(
            lines,
            ["Model", "F5 [%]", "Precyzja 5 mm [%]", "Pokrycie 5 mm [%]", "Puste / 720"],
            rows,
        )
        ident = "holdout-arms-" + budget
        table(
            lines,
            [
                "Ramię",
                "F5 po trzech treningach [%]",
                "Ziarno 0 [%]",
                "Ziarno 1 [%]",
                "Ziarno 2 [%]",
            ],
            [
                [
                    arm,
                    cell(data, ident, arm, "F5", ptr(*base, "arms", arm, "F5", "mean"), factor=100),
                ]
                + [
                    cell(
                        data,
                        ident,
                        arm,
                        f"ziarno {i}",
                        ptr(*base, "arms", arm, "per_seed_F5", i),
                        factor=100,
                    )
                    for i in range(3)
                ]
                for arm in ARMS
            ],
        )
        table(
            lines,
            [
                "Kontrast F5",
                "Różnica [pp]",
                "CI95 obiekty [pp]",
                "CI95 obiekty i treningi [pp]",
                "Różnice ziaren 0; 1; 2 [pp]",
            ],
            f5_contrast_rows(
                data, (*base, "F5_contrasts"), "holdout-contrasts-" + budget, contrasts
            ),
        )
    lines += [
        "## HB: neutralne RGB minus oryginalne RGB",
        "",
        (
            "Zmieniono kolor w istniejącej masce na (128,128,128), przy "
            "zachowaniu głębi, maski i kalibracji. Reguła kamer zapytań jest "
            "wspólna; ich konkretne położenia mogą zależeć od zmienionej "
            "predykcji wejściowej. Jest to reakcja całego rekonstruktora na "
            "interwencję spoza rozkładu, a nie porównanie modeli uczonych z "
            "RGB i bez RGB."
        ),
        "",
    ]
    for label, expected in (("all", 198), ("eligible", 182)):
        lines += [
            "### "
            + (
                "Cała kohorta: 198 obserwacji"
                if label == "all"
                else "Wejścia kwalifikujące się do inferencji: 182 obserwacje"
            ),
            "",
        ]
        for budget in BUDGETS:
            base = ("HB198-neutral128", label, "paired_RGB", budget)
            paired = at(data, ptr(*base))
            assert paired["observations"] == expected and set(paired["methods"]) == set(MODELS)
            ident = "RGB-models-" + label + "-" + budget
            lines += [
                "#### Limit " + ("16 384" if budget == "cap16384" else "512") + " punktów",
                "",
            ]
            rows = []
            for mid in MODELS:
                path = (*base, "methods", mid)
                effect = (*path, "neutral_minus_original", "fscore@0.005")
                rows.append(
                    [
                        model_name(mid),
                        cell(
                            data,
                            ident,
                            mid,
                            "oryginalne F5",
                            ptr(*path, "original_F5", "mean"),
                            factor=100,
                        ),
                        cell(
                            data,
                            ident,
                            mid,
                            "neutralne F5",
                            ptr(*path, "neutral_F5", "mean"),
                            factor=100,
                        ),
                        cell(
                            data,
                            ident,
                            mid,
                            "zmiana pp",
                            ptr(*effect, "mean"),
                            factor=100,
                            signed=True,
                        ),
                        cell(
                            data,
                            ident,
                            mid,
                            "CI obiekty pp",
                            ptr(*effect, "paired_objects", "object_bootstrap95"),
                            factor=100,
                            signed=True,
                            interval=True,
                        ),
                    ]
                )
            table(
                lines,
                [
                    "Model",
                    "F5 oryginalne [%]",
                    "F5 neutralne [%]",
                    "Zmiana [pp]",
                    "CI95 obiekty [pp]",
                ],
                rows,
            )
            lines += [
                (
                    "Zmiany jakości geometrycznej są sparowane w obrębie obserwacji. "
                    "W nawiasach podano liczbę par z określoną miarą względem "
                    "wszystkich obserwacji; mianowniki obiektowe i pozostałe metryki "
                    "zawiera `metrics-per-model.csv`."
                ),
                "",
            ]
            table(
                lines,
                [
                    "Model",
                    "Zmiana precyzji [pp]",
                    "Zmiana pokrycia [pp]",
                    "Zmiana pokrycia zasłoniętego GT [pp] (pary)",
                    "Zmiana pokrycia GT poza gęstym wejściem [pp] (pary)",
                ],
                rgb_quality_rows(data, base, "RGB-quality-" + label + "-" + budget),
            )
            table(
                lines,
                [
                    "Ramię",
                    "Zmiana F5 [pp]",
                    "CI95 obiekty [pp]",
                    "CI95 obiekty i treningi [pp]",
                    "Zmiany ziaren 0; 1; 2 [pp]",
                ],
                f5_contrast_rows(
                    data,
                    (*base, "arm_neutral_minus_original_F5"),
                    "RGB-arms-" + label + "-" + budget,
                    ARMS,
                ),
            )
            path = (*base, "change_in_PHOTO_minus_BASE_F5")
            ident = "RGB-change-of-effect-" + label + "-" + budget
            row = [
                "Zmiana kontrastu PHOTO–BASE",
                cell(
                    data,
                    ident,
                    "PHOTO-BASE",
                    "różnica pp",
                    ptr(*path, "paired_objects", "difference"),
                    factor=100,
                    signed=True,
                ),
                cell(
                    data,
                    ident,
                    "PHOTO-BASE",
                    "CI obiekty pp",
                    ptr(*path, "paired_objects", "object_bootstrap95"),
                    factor=100,
                    signed=True,
                    interval=True,
                ),
                cell(
                    data,
                    ident,
                    "PHOTO-BASE",
                    "CI obiekty+treningi pp",
                    ptr(*path, "object_and_seed_bootstrap95"),
                    factor=100,
                    signed=True,
                    interval=True,
                ),
            ]
            table(
                lines,
                ["Wielkość", "Zmiana [pp]", "CI95 obiekty [pp]", "CI95 obiekty i treningi [pp]"],
                [row],
            )
        lines += [
            "#### Zmiana natywnych chmur",
            "",
            (
                "To porównanie geometrii przed ograniczeniem liczby punktów, bez "
                "użycia GT. Dokładna równość tablic uwzględnia również ich "
                "kolejność; zerowa odległość zbiorów jest osobnym pojęciem. "
                "Odległość poniżej jest obiektową średnią z dwukierunkowej "
                "średniej najbliższych sąsiadów dla par niepustych chmur."
            ),
            "",
        ]
        rows = []
        for mid in MODELS:
            base = ("HB198-neutral128", label, "native_changes", mid)
            ident = "RGB-native-" + label
            rows.append(
                [
                    model_name(mid),
                    cell(
                        data,
                        ident,
                        mid,
                        "identyczne tablice",
                        ptr(*base, "exact_array_equal_count"),
                        places=0,
                    ),
                    cell(
                        data,
                        ident,
                        mid,
                        "dwie niepuste",
                        ptr(*base, "both_nonempty_count"),
                        places=0,
                    ),
                    cell(
                        data,
                        ident,
                        mid,
                        "średnia odległość mm",
                        ptr(
                            *base, "metrics", "symmetric_mean_distance_m", "object_balanced", "mean"
                        ),
                        factor=1000,
                        places=4,
                    ),
                ]
            )
        table(
            lines,
            [
                "Model",
                "Identyczne tablice",
                "Pary dwóch niepustych chmur",
                "Średnia odległość [mm]",
            ],
            rows,
        )
    lines += [
        "## Zakres wnioskowania i artefakty",
        "",
        (
            "Zestawienia opisują ukończoną macierz; interpretacja naukowa "
            "musi uwzględniać audyt PHOTO, oryginalny wynik HB, tylko trzy "
            "seedy i niejednolite zastosowanie historycznej poprawki "
            "treningu. Żadna z tych kontroli nie zastępuje nowej jednolitej "
            "macierzy treningów ani nie izoluje przyczyny efektu PHOTO."
        ),
        "",
        (
            "Pełne metryki (także inne progi, Chamfer, pokrycie podzbiorów GT "
            "i wolna przestrzeń) wraz z licznościami określonych par "
            "pozostają w `analysis-v1/summaries.json`. Ich mianowniki mogą "
            "się różnić. Nie utożsamia się wartości nieokreślonej z zerem."
        ),
        "",
        (
            "Plik `metrics-per-model.csv` udostępnia wszystkie średnie i ich "
            "mianowniki dla każdego modelu, populacji oraz limitu punktów, a "
            "`failure-counts.csv` zawiera liczby pustych i zbyt małych chmur "
            "oraz wszystkie zapisane statusy. Eksport zachowuje źródłowe "
            "jednostki: udziały 0–1 i odległości w metrach. Pusta wartość "
            "oznacza miarę nieokreśloną. Kolumna `json_pointer` prowadzi do "
            "dokładnego źródła w `analysis-v1/summaries.json`; eksport nie "
            "wykonuje nowej agregacji."
        ),
        "",
        (
            "Źródła: `analysis-v1/complete.json`, "
            "`analysis-v1/summaries.json`, "
            "`analysis-v1/F5-per-observation.csv`. Każda komórka tego raportu "
            "ma selektor i regułę wyświetlania w `table-cells.json`. Raport "
            "nie nadpisuje wyników HB ani źródeł pracy."
        ),
        "",
    ]
    OUT.mkdir()
    export_counts = export_full_metrics(data)
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    save(OUT / "table-cells.json", {"source": proof["summaries"], "cells": CELLS})
    save(
        OUT / "complete.json",
        {
            "status": "complete_numeric_control_report_exported",
            "analysis_complete_sha256": digest(ANALYSIS / "complete.json"),
            "summaries": proof["summaries"],
            "script_sha256": digest(Path(__file__)),
            "cell_count": len(CELLS),
            "export_counts": export_counts,
            "files": {
                name: digest(OUT / name)
                for name in (
                    "report.md",
                    "table-cells.json",
                    "metrics-per-model.csv",
                    "failure-counts.csv",
                )
            },
            "scientific_interpretation_pending": True,
        },
    )
    print(json.dumps({"status": "exported", "cells": len(CELLS), "report": str(OUT / "report.md")}))


if __name__ == "__main__":
    main()
