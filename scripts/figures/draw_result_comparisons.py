"""Create editorial summaries and figures from frozen scores; never score new predictions."""

import hashlib
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from scripts.common.paths import ROOT as WORKSPACE_ROOT
from scripts.common.paths import script_help

script_help(__doc__, __name__)


ROOT = WORKSPACE_ROOT

matplotlib.use("Agg")

OUT = ROOT / "artifacts/figures"
E = ROOT / "runs/research-evolution-evaluation-20260907"
COL = {"BASE": "#326b8b", "CLASSIC": "#b37b2a", "PHOTO": "#88559c", "INPUT": "#525c62"}
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def load(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fmt(v, scale=100):
    return "N/D" if v is None else f"{v * scale:.2f}".replace(".", ",")


scores = E / "final-scores-20260909-v6"
complete = load(scores / "complete.json")
assert sha(scores / "summaries.json") == complete["summaries_sha256"]
s = load(scores / "summaries.json")
pilotpath = ROOT / "runs/ray-photoreal-portable-report-20260906-v2/complete.json"
p = load(pilotpath)
for ref in p["source_scores"].values():
    assert sha(Path(ref["path"])) == ref["sha256"]
for filename, h in p["artifacts"].items():
    assert sha(pilotpath.parent / filename) == h
main = s["all"]["budgets"]["cap16384"]


def metric(mid, key, budget="cap16384", population="all"):
    return s[population]["budgets"][budget]["methods"][mid]["metrics"][key]["mean"]


def arm(arm, key, budget="cap16384", population="all"):
    v = [metric(f"ray-{arm}-seed{i}", key, budget, population) for i in range(3)]
    assert all(x is not None for x in v)
    return float(np.mean(v))


keys = [
    "fscore@0.002",
    "fscore@0.005",
    "fscore@0.010",
    "precision@0.005",
    "recall@0.005",
    "count",
    "chamfer_mean_m_success_only",
    "chamfer_capped_100mm_m",
    "GT_occluded_recall5",
    "GT_uncovered_by_dense_input_recall5",
    "added_precision5",
    "known_free_rate_all",
]
rows = {a: {k: arm(a, k) for k in keys} for a in ["BASE", "CLASSIC", "PHOTO"]}
for a in rows:
    assert abs(rows[a]["fscore@0.005"] - main["arm_F5_seed_averaged"][a]["mean"]) < 1e-12
contrasts = main["F5_contrasts"]
evidence = {
    "source_scores_sha256": sha(scores / "complete.json"),
    "summaries_sha256": sha(scores / "summaries.json"),
    "source_pilot_report_sha256": sha(pilotpath),
    "source_summary_path": str((scores / "summaries.json").relative_to(ROOT)),
    "main_population": {"observations": 198, "objects": 33, "scenes": 13, "seeds": 3},
    "main_arms": rows,
    "contrasts": contrasts,
}
evidence["density"] = {
    b: {a: arm(a, "fscore@0.005", b) for a in ["BASE", "CLASSIC", "PHOTO"]}
    | {"INPUT": metric("INPUT", "fscore@0.005", b)}
    for b in ["cap512", "shared_nonfailed_cap", "cap16384", "native"]
}
evidence["pilot"] = {"primary": p["primary"], "table_values": p["table_values"]}
evidence["secondary_population_effects"] = {
    pop: s[pop]["budgets"]["cap16384"]["F5_contrasts"]["PHOTO-BASE"]
    for pop in ["eligible", "registration_passed", "low", "medium", "high"]
}
(OUT / "evidence").mkdir(parents=True, exist_ok=True)
(OUT / "assets").mkdir(parents=True, exist_ok=True)
(OUT / "evidence/results.json").write_text(
    json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
# Main finding: all three seeds and both uncertainty levels in one visual.
fig, ax = plt.subplots(figsize=(6.3, 3.65))
x = contrasts["PHOTO-BASE"]
mean = x["paired_objects"]["difference"] * 100
vals = np.array(x["dependence_sensitivity"]["per_seed_object_mean"]) * 100
for yy, v in zip([4, 3, 2], vals, strict=False):
    ax.scatter(v, yy, s=54, color=COL["PHOTO"], zorder=3)
    ax.text(v + 0.04, yy + 0.08, f"{v:+.2f}".replace(".", ","), fontsize=9)
for yy, ci, c in [
    (1, x["paired_objects"]["object_bootstrap95"], COL["BASE"]),
    (0, x["dependence_sensitivity"]["object_and_seed_bootstrap95"], "#222222"),
]:
    lo, hi = np.array(ci) * 100
    ax.errorbar(
        mean, yy, xerr=[[mean - lo], [hi - mean]], fmt="o", color=c, capsize=5, lw=2, ms=6, zorder=3
    )
    ax.text(
        (lo + hi) / 2, yy - 0.30, f"[{lo:.2f}; {hi:.2f}]".replace(".", ","), ha="center", fontsize=9
    )
ax.axvline(0, color="#777777", lw=1, ls="--")
ax.set(
    yticks=[4, 3, 2, 1, 0],
    yticklabels=[
        "Powtórzenie 1",
        "Powtórzenie 2",
        "Powtórzenie 3",
        "Średnia: losowanie obiektów",
        "Średnia: obiektów i treningów",
    ],
    xlabel="Różnica PHOTO − BASE w F5 [pp]",
    xlim=(-1.53, 0.27),
    ylim=(-0.7, 4.6),
)

ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:g}".replace(".", ",")))
ax.grid(axis="x", color="#e4e4e4", lw=0.6)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(axis="y", length=0)
fig.subplots_adjust(left=0.385, right=0.98, bottom=0.18, top=0.98)
for ext in ["pdf", "png"]:
    fig.savefig(OUT / f"assets/results-effect.{ext}", dpi=300)
plt.close(fig)
# Categorical strategies, not continuous budgets.
fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.4), gridspec_kw={"width_ratios": [1.05, 1]})
budgets = ["cap512", "shared_nonfailed_cap", "cap16384", "native"]
xx = np.arange(4)
for a, off in [("INPUT", -0.23), ("BASE", 0), ("PHOTO", 0.23)]:
    yy = np.array([evidence["density"][b][a] for b in budgets]) * 100
    axes[0].bar(xx + off, yy, width=0.21, label=a, color=COL[a])
axes[0].set(
    xticks=xx,
    xticklabels=["512", "Wspólna\nliczność", "16 384", "Bez\nlimitu"],
    ylabel="F5 [%]",
    ylim=(0, 60),
)
axes[0].set_title("(a) Reguła liczby punktów", loc="left")
axes[0].legend(
    frameon=False,
    fontsize=8,
    ncol=3,
    loc="upper center",
    bbox_to_anchor=(0.52, 1.03),
    columnspacing=0.6,
    handlelength=0.9,
)
axes[0].tick_params(axis="x", labelsize=8)
for a, off in [("INPUT", -0.23), ("BASE", 0), ("PHOTO", 0.23)]:
    vals = [
        metric("INPUT", k) if a == "INPUT" else rows[a][k]
        for k in ["precision@0.005", "recall@0.005", "GT_occluded_recall5"]
    ]
    axes[1].bar(np.arange(3) + off, np.array(vals) * 100, width=0.21, color=COL[a])
axes[1].set(
    xticks=np.arange(3),
    xticklabels=["Precyzja", "Pokrycie\ncałości", "Pokrycie\nzasłoniętej"],
    ylabel="Wartość [%]",
    ylim=(0, 100),
)
axes[1].set_title("(b) Precyzja i pokrycie", loc="left")
axes[1].tick_params(axis="x", labelsize=8)
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e4e4e4", lw=0.6)
fig.tight_layout(w_pad=1.4)
for ext in ["pdf", "png"]:
    fig.savefig(OUT / f"assets/results-density.{ext}", dpi=300)
plt.close(fig)
print("Created result evidence and two figures.")
