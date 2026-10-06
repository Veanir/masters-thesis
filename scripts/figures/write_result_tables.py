"""Write the current result tables from verified frozen scores."""

import hashlib
import json
from pathlib import Path

from scripts.common.paths import ROOT as WORKSPACE_ROOT
from scripts.common.paths import script_help

script_help(__doc__, __name__)


ROOT = WORKSPACE_ROOT
OUT = ROOT / "artifacts/figures"
SCORES = ROOT / "runs/research-evolution-evaluation-20260907/final-scores-20260909-v6"
PILOT = ROOT / "runs/ray-photoreal-portable-report-20260906-v2/complete.json"


def read(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


complete = read(SCORES / "complete.json")
assert sha(SCORES / "summaries.json") == complete["summaries_sha256"]
s = read(SCORES / "summaries.json")
p = read(PILOT)
recorded = read(OUT / "evidence/results.json")
assert sha(PILOT) == recorded["source_pilot_report_sha256"]


def metric(method, key, budget="cap16384"):
    return s["all"]["budgets"][budget]["methods"][method]["metrics"][key]["mean"]


def arm(name, key):
    values = [metric(f"ray-{name}-seed{i}", key) for i in range(3)]
    assert all(v is not None for v in values)
    return sum(values) / 3


def fmt(value):
    return "N/D" if value is None else f"{100 * value:.2f}".replace(".", ",")


def name(mid):
    if mid == "INPUT":
        return mid
    if mid == "ray-pretrained":
        return "RaySt3R przed dostrajaniem"
    if mid.startswith("octmae-"):
        return "OctMAE " + ("oryginalny" if mid.endswith("original") else "kadr 1,4")
    if mid.startswith("ray-historical-"):
        return "BASE wstępne, " + str(int(mid[-1]) + 1)
    return mid.split("-")[1] + ", " + str(int(mid[-1]) + 1)


def table(headers, rows, caption):
    return (
        "\n".join(
            ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
            + ["| " + " | ".join(row) + " |" for row in rows]
        )
        + "\n\nTable: "
        + caption
        + "\n"
    )


methods = list(s["all"]["budgets"]["cap16384"]["methods"])
assert len(methods) == 16
budgets = ["cap512", "shared_nonfailed_cap", "cap16384", "native"]
rows = [[name(mid)] + [fmt(metric(mid, "fscore@0.005", b)) for b in budgets] for mid in methods]
parts = [
    r"""# Uzupełniające wyniki liczbowe
\label{app:wyniki}

Zestawienia uzupełniają rozdział \ref{sec:wyniki}. Pokazują każdy model osobno oraz zmiany
wyników przy różnych regułach liczby punktów. HB obejmuje 198 obserwacji 33 obiektów; każdy
obiekt ma jednakową wagę w średniej. Numery 1–3 oznaczają trzy powtórzenia uczenia. Wszystkie
miary w tabelach podano w procentach.

## F5 poszczególnych modeli na HB
""",
    table(
        ["Metoda, powtórzenie", "512", "Wspólna", "16 384", "Bez limitu"],
        rows,
        (
            "F5 na HB dla czterech reguł próbkowania. Wspólna liczność "
            "dotyczy chmur mających co najmniej 512 punktów; mniejsze i puste "
            "wyniki pozostają w ocenie. Źródło: zapisane wyniki eksperymentu. "
            "\\label{tab:hb-all-1}"
        ),
    ),
    r"""
Kolumny 512 i 16 384 oznaczają górne limity punktów. Wspólna liczność wyrównuje dostatecznie
duże chmury danej obserwacji, a wynik bez limitu pozostawia wyjście bez dodatkowego
ograniczenia. OctMAE pokazano z oryginalnym kadrem i z wycinkiem wokół maski, powiększonym
według współczynnika 1,4 i zachowującym proporcje 4:3 w granicach obrazu. Wycinek przeskalowano
do 640 × 480 i przeliczono parametry kamery.

Spośród 198 wejść 182 spełniają próg 512 punktów, 14 jest mniejszych i niepustych, a dwa są
puste. Dla 16 niewystarczających wejść modele otrzymują F5 równy zero. Główna różnica PHOTO–BASE
dotyczy limitu 16 384 punktów.

## Pokrycie braków i poprawność dodanych punktów
\label{sec:surface-results}

W tabeli R oznacza pokrycie, a P precyzję. Udział naruszeń wolnej przestrzeni odnosi się do
całej niepustej chmury.
""",
]
keys = [
    "GT_occluded_recall5",
    "GT_uncovered_by_dense_input_recall5",
    "added_precision5",
    "known_free_rate_all",
]
surface = [
    [a] + [fmt(metric("INPUT", k) if a == "INPUT" else arm(a, k)) for k in keys]
    for a in ["INPUT", "BASE", "CLASSIC", "PHOTO"]
]
parts += [
    table(
        ["Metoda", "R zasł.", "R poza INPUT", "P dod.", "Wolna przestrzeń"],
        surface,
        (
            "HB, limit 16 384: pokrycie zasłoniętego i niepokrytego GT, "
            "precyzja dodanych punktów oraz udział naruszeń wolnej "
            "przestrzeni. Warianty uśredniono po trzech treningach. N/D: "
            "miara nieokreślona. \\label{tab:hb-surface-all}"
        ),
    ),
    r"""
Pokrycie obu części GT obejmuje wszystkie 198 obserwacji. Precyzja dodanych punktów i udział
naruszeń mają po 182 określone wyniki każdego RaySt3R. Dla obu OctMAE liczby te wynoszą 181 i
182, a dla INPUT 0 i 196. INPUT nie dodaje punktów do siebie, stąd jego nieokreślona precyzja.
Udział naruszeń liczony tylko wśród punktów z wiarygodną projekcją ma 195 określonych wyników
INPUT; dla modeli pozostaje ich 182. Każda grupa z określonymi miarami obejmuje 33 obiekty.

Zwykły Chamfer obejmuje 182 obserwacje modeli lub 196 INPUT. Wersja z ograniczeniem odległości i
karą za pustą chmurę obejmuje wszystkie 198 przypadków każdej metody. Reguły wartości
nieokreślonych podaje punkt \ref{sec:failures}.

## Wyniki na trzech zbiorach badania wstępnego
""",
]
pilot_methods = [
    ("input512", "INPUT512"),
    ("input_dense", "INPUT gęste"),
    ("pretrained", "RaySt3R przed dostrajaniem"),
    ("base", "BASE"),
    ("classic", "CLASSIC"),
    ("photo", "PHOTO"),
]
pilot_rows = [
    [label]
    + [fmt(p["table_values"][cohort][key]["f5"]) for cohort in ["hope146", "ycbv42", "abo24"]]
    for key, label in pilot_methods
]
parts += [
    table(
        ["Metoda", "HOPE", "YCB-Video", "ABO"],
        pilot_rows,
        (
            "F5 przy limicie 512 punktów; dostrojone warianty uśredniono po "
            "trzech treningach. HOPE: 28 obiektów i 146 obserwacji; YCB-V: 21 "
            "i 42; ABO: 24 geometrie i 24 obserwacje. "
            "\\label{tab:pilot-cohorts}"
        ),
    ),
    r"""
INPUT512 przygotowano przed oceną, a gęsty INPUT ograniczano do 512 punktów dopiero w jej
trakcie. Inny wybór punktów wyjaśnia różne wyniki tych odniesień.
""",
]
destination = OUT / "appendices/02-wyniki.md"
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text("\n\n".join(parts), encoding="utf-8")
ledger = {
    "script_sha256": sha(Path(__file__)),
    "summary_sha256": sha(SCORES / "summaries.json"),
    "pilot_report_sha256": sha(PILOT),
    "output_sha256": sha(destination),
    "selection": (
        "All 16 HB methods and all four F5 rules; all three pilot cohorts "
        "and six methods; primary surface metrics of INPUT and all three "
        "arms. Appearance assessments remain in archived technical "
        "records."
    ),
    "appearance_assessment_status": (
        "Archived only; removed from the manuscript after the author "
        "clarification of manual verification."
    ),
    "HB_F5": {mid: {b: metric(mid, "fscore@0.005", b) for b in budgets} for mid in methods},
    "surface": {
        a: {k: metric("INPUT", k) if a == "INPUT" else arm(a, k) for k in keys}
        for a in ["INPUT", "BASE", "CLASSIC", "PHOTO"]
    },
}
(OUT / "evidence/compact-results.json").write_text(
    json.dumps(ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print("Compact appendix written; original frozen scores unchanged.")
