"""Freeze evaluation definitions and the unchanged HB membership before inference."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time

from scripts.common.paths import ROOT as WORKSPACE_ROOT
from scripts.common.paths import script_help

script_help(__doc__, __name__)

ROOT = WORKSPACE_ROOT


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    references = {
        "protocol": "configs/evaluation.json",
        "metric_core": "scripts/eval/surface_metrics.py",
        "metric_checks": "scripts/eval/check_surface_metrics.py",
        "reservation": "runs/research-evolution-hb-20260906/cohort-reservation-v1.json",
        "input_manifest": "runs/research-evolution-hb-inputs-v2/complete.json",
        "GT_manifest": "runs/research-evolution-hb-observations-v2/complete.json",
        "calibration": "runs/research-evolution-hb-20260906/calibration-v1/complete.json",
        "export_verification": "runs/research-evolution-hb-20260906/export-v2-verification.json",
    }
    inputs = json.loads((ROOT / references["input_manifest"]).read_text())
    gt = json.loads((ROOT / references["GT_manifest"]).read_text())
    assert not inputs["gt_exported"] and not inputs["contract"]["model_predictions_computed"]
    assert len(inputs["rows"]) == len(gt["rows"]) == 198
    assert [r["sample_id"] for r in inputs["rows"]] == [r["sample_id"] for r in gt["rows"]]
    assert sum(r["input_eligible"] for r in inputs["rows"]) == 182
    assert sum(r["source_registration_status"] == "passed" for r in inputs["rows"]) == 125
    checks = subprocess.run(
        [sys.executable, str(ROOT / references["metric_checks"])],
        capture_output=True,
        text=True,
        check=True,
    )
    output = ROOT / "runs/research-evolution-evaluation-20260907"
    output.mkdir(exist_ok=False)
    record = {
        "status": "evaluation_contract_frozen_before_final_predictions",
        "created_unix": time.time(),
        "primary_endpoint": (
            "object-mean F5 at16384 point cap, PHOTO minus BASE, "
            "paired3seeds; all198 including failures"
        ),
        "secondary_point_budgets": [512, "shared_nonfailed_cap", "native"],
        "thresholds_m": [0.002, 0.005, 0.010],
        "reference_points": 262144,
        "truth_seed_purpose": "evolution-eval-GT-v1",
        "prediction_selection_seed_purpose": "evolution-eval-point-selection-v1",
        "chamfer_cap_m": 0.1,
        "visibility_tolerance_m": 0.005,
        "visibility_patch_span_m": 0.010,
        "bootstrap_replicates": 10000,
        "object_bootstrap_seed": 2026090702,
        "sensitivity_seed": 2026090703,
        "cohort_observations": 198,
        "cohort_objects": 33,
        "cohort_scenes": 13,
        "eligible_observations": 182,
        "registration_passed_observations": 125,
        "sample_ids": [r["sample_id"] for r in inputs["rows"]],
        "references": {
            key: {"path": path, "sha256": sha(ROOT / path)} for key, path in references.items()
        },
        "checks": json.loads(checks.stdout),
        "scope": (
            "Evaluation component only. Final data, augmentation and "
            "adaptation protocol must be frozen and bound separately before "
            "any final-cohort predictions."
        ),
    }
    (output / "protocol-v1.json").write_text(json.dumps(record, indent=2))
    print("Frozen evaluation component", sha(output / "protocol-v1.json"))


if __name__ == "__main__":
    main()
