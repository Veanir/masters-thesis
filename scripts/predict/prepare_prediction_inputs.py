"""Bind all final HB jobs only after the complete nine-training download audit.

Writes jobs and a local artifact plan; never calls a model or a cloud API.
Deployment and its fresh EVAL budget are separate from this preparation.
"""

import argparse
import copy
import json
from pathlib import Path

from scripts.common.artifact_bindings import bound, reference
from scripts.common.paths import CODE_ROOT, ROOT, digest, read, save

EVAL = ROOT / "runs/research-evolution-evaluation-20260907"
ARMS = ["BASE", "CLASSIC", "PHOTO"]


def method_inventory(campaign):
    methods = campaign["evaluation_methods"]
    expected = {"INPUT", "ray-pretrained", "octmae-original", "octmae-crop1p4"}
    expected |= {f"ray-historical-BASE-seed{s}" for s in range(3)}
    expected |= {f"ray-{arm}-seed{s}" for s in range(3) for arm in ARMS}
    assert len(methods) == 16 and {m["id"] for m in methods} == expected
    adapted = [m for m in methods if m["role"] == "adapted"]
    assert len(adapted) == 9
    assert {(m["arm"], m["seed"]) for m in adapted} == {(a, s) for a in ARMS for s in range(3)}
    for method in adapted:
        assert method["id"] == f"ray-{method['arm']}-seed{method['seed']}"
        assert method["checkpoint_step"] == 25600 and method["context"] == "masked"
        assert (
            method["training_job_sha256"]
            == campaign["training_jobs"][str(method["seed"])]["sha256"]
        )
    return methods


def completed_matrix(matrix_path, campaign_path):
    """Verify identities of downloaded artifacts already tensor-audited by MAIN."""
    matrix = read(matrix_path)
    assert matrix["status"] == "all_nine_frozen_trainings_downloaded_verified_GPUs_stopped"
    assert matrix["campaign_sha256"] == digest(campaign_path)
    assert matrix["total_updates"] == 230400 and matrix["HB_evaluation_performed"] is False
    assert len(matrix["seeds"]) == 3 and {s["seed"] for s in matrix["seeds"]} == {0, 1, 2}
    campaign = read(campaign_path)
    checkpoints = {}
    for seed in range(3):
        folder = matrix_path.parent / f"seed{seed}"
        proof = read(folder / "complete.json")
        assert proof == next(s for s in matrix["seeds"] if s["seed"] == seed)
        assert proof["status"] == "downloaded_seed_verified_GPU_stopped"
        assert proof["GPU_stopped_confirmed"] is True
        assert read(folder / "resource-status.json")["stopped_or_absent_confirmed"] is True
        target = folder / "unpacked"
        assert digest(target / "complete.json") == proof["complete_sha256"]
        remote = read(target / "complete.json")
        assert remote["status"] == "all_three_frozen_seed_arms_complete"
        assert remote["seed"] == seed and remote["total_updates"] == 76800
        assert remote["binding"]["campaign_sha256"] == digest(campaign_path)
        assert remote["arms"] == proof["arms"]
        assert len(proof["arms"]) == 3 and {r["arm"] for r in proof["arms"]} == set(ARMS)
        for item in proof["arms"]:
            arm = item["arm"]
            run = target / arm
            assert item["steps"] == 25600 and item["final_tensors_checked"] is True
            for filename, key in [
                ("complete.json", "complete_sha256"),
                ("binding.json", "binding_sha256"),
                ("steps.jsonl", "log_sha256"),
                ("schedule.json", "schedule_sha256"),
                ("final-delta.pt", "checkpoint_sha256"),
                ("recovery.pt", "recovery_sha256"),
            ]:
                assert digest(run / filename) == item[key], (
                    f"Changed MAIN artifact: {run / filename}"
                )
            complete = read(run / "complete.json")
            binding = read(run / "binding.json")
            assert (
                complete["status"] == "paired_evolution_matrix_training_complete"
                and complete["steps"] == 25600
            )
            assert (
                complete["binding"] == binding
                and binding["arm"] == arm
                and binding["context"] == "masked"
            )
            assert binding["job_sha256"] == campaign["training_jobs"][str(seed)]["sha256"]
            assert (
                complete["checkpoint"] == "final-delta.pt"
                and complete["checkpoint_sha256"] == item["checkpoint_sha256"]
            )
            checkpoints[f"ray-{arm}-seed{seed}"] = reference(run / "final-delta.pt")
    assert len(checkpoints) == 9
    return checkpoints


def make_job(method, template, campaign_sha, input_sha, checkpoints):
    job = copy.deepcopy(template)
    assert job["purpose"] == "final_inference_development_canary"
    job.update(
        purpose="frozen_final_HB_inference",
        method_id=method["id"],
        campaign_sha256=campaign_sha,
        input_manifest_sha256=input_sha,
    )
    if method["role"] == "adapted":
        assert template["method_id"] == "ray-pretrained" and template["checkpoint_sha256"] is None
        job.update(
            checkpoint_kind="evolution",
            checkpoint_step=25600,
            checkpoint_sha256=checkpoints[method["id"]]["sha256"],
            training_job_sha256=method["training_job_sha256"],
        )
    else:
        assert template["method_id"] == method["id"]
    return job


def prepare(campaign_path, matrix_path, out):
    assert not out.exists()
    campaign = read(campaign_path)
    assert campaign["status"] == "frozen_evolution_campaign"
    assert campaign["runs"] == 9 and campaign["steps_per_run"] == 25600
    methods = method_inventory(campaign)
    # This guard precedes writes and HB input reads. No incomplete matrix jobs.
    checkpoints = completed_matrix(matrix_path, campaign_path)
    protocol_path = bound(campaign["references"]["evaluation_protocol"])
    protocol = read(protocol_path)
    assert digest(protocol_path) == campaign["evaluation_protocol_sha256"]
    parity_path = bound(campaign["references"]["adapter_parity"])
    parity = read(parity_path)
    assert parity["status"] == "downloaded_adapter_canaries_recomputed_locally"
    assert (
        parity["all_passed"] is True
        and len(parity["checks"]) == 18
        and all(r["passed"] for r in parity["checks"])
    )
    refs_path = bound(campaign["references"]["reference_systems"])
    refs = read(refs_path)
    assert [m for m in methods if m["role"] != "adapted"] == refs["methods"]
    canary = EVAL / "adapter-canary-v1/bundle"
    canary_plan = read(canary / "plan.json")
    inputs = EVAL / "HB-inference-inputs-v1"
    input_complete = read(inputs / "complete.json")
    assert input_complete["protocol_sha256"] == digest(protocol_path)
    input_refs = {}
    for name in ["ray", "octmae-original", "octmae-crop1p4"]:
        manifest_path = inputs / name / "manifest.json"
        archive = inputs / (name + ".zip")
        binding = input_complete["bindings"][name]
        assert digest(manifest_path) == binding["manifest_sha256"]
        assert (
            digest(archive) == binding["archive_sha256"]
            and archive.stat().st_size == binding["archive_bytes"]
        )
        manifest = read(manifest_path)
        assert manifest["gt_uploaded"] is False and len(manifest["rows"]) == 198
        assert [r["sample_id"] for r in manifest["rows"]] == protocol["sample_ids"]
        assert sum(r["input_eligible"] for r in manifest["rows"]) == 182
        input_refs[name] = {"manifest": reference(manifest_path), "archive": reference(archive)}
    historical = {r["run"]: r for r in refs["historical_runs"]}
    jobs = []
    for method in methods:
        if method["role"] == "input":
            continue
        model = method["model"]
        mid = method["id"]
        source_id = "ray-pretrained" if method["role"] == "adapted" else mid
        template_path = canary / "jobs" / (source_id + ".json")
        assert digest(template_path) == canary_plan["files_sha256"]["jobs/" + source_id + ".json"]
        source_entry = next(m for m in canary_plan["methods"] if m["id"] == source_id)
        assert digest(template_path) == source_entry["job_sha256"]
        runner = (
            "predict/predict_ray_homebrewed.py"
            if model == "RaySt3R"
            else "predict/predict_octmae_homebrewed.py"
        )
        assert digest(CODE_ROOT / "scripts" / runner) == campaign["source_files_sha256"][runner]
        input_name = "ray" if model == "RaySt3R" else mid
        job = make_job(
            method,
            read(template_path),
            digest(campaign_path),
            input_refs[input_name]["manifest"]["sha256"],
            checkpoints,
        )
        assert job["runner_sha256"] == digest(CODE_ROOT / "scripts" / runner)
        entry = {
            "method": method,
            "inputs": input_name,
            "runner": reference(CODE_ROOT / "scripts" / runner),
        }
        if method["role"] == "adapted":
            entry["checkpoint"] = checkpoints[mid]
        elif method["role"] == "historical":
            old = historical[method["checkpoint_model"]]
            entry.update(
                checkpoint=reference(bound(old["checkpoint"])),
                historical_complete=reference(bound(old["complete"])),
                historical_freeze=reference(bound(refs["historical_freeze"])),
            )
            assert job["checkpoint_sha256"] == entry["checkpoint"]["sha256"]
        jobs.append((mid, job, entry))
    assert len(jobs) == 15
    out.mkdir(parents=True)
    (out / "jobs").mkdir()
    entries = []
    for mid, job, entry in jobs:
        path = out / "jobs" / (mid + ".json")
        save(path, job)
        entries.append({**entry, "job": reference(path)})
    result = {
        "status": "all15_final_HB_inference_jobs_bound_after_complete_MAIN",
        "campaign": reference(campaign_path),
        "matrix_complete": reference(matrix_path),
        "protocol": reference(protocol_path),
        "adapter_parity": reference(parity_path),
        "inputs": input_refs,
        "jobs": entries,
        "methods": methods,
        "source_sha256": digest(Path(__file__)),
        "GPU_actions": False,
        "HB_predictions_computed": False,
        "scope": (
            "Artifact plan only. Reuse parity-tested environments with a "
            "fresh bounded EVAL lease plan. INPUT is computed only by the "
            "separate scorer. Never upload GT/cache to inference."
        ),
    }
    save(out / "complete.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--matrix-complete", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [args.campaign.resolve(), args.matrix_complete.resolve(), args.output.resolve()]
    for path in paths:
        path.relative_to(ROOT.resolve())
    result = prepare(*paths)
    print(
        json.dumps({"status": result["status"], "jobs": len(result["jobs"]), "GPU_actions": False})
    )


if __name__ == "__main__":
    main()
