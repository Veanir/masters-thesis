"""Validate the frozen pilot training protocol and completed runs."""

from pathlib import Path

from scripts.common.paths import ROOT, digest, read, script_help

script_help(__doc__, __name__)


CAMPAIGN = ROOT / "runs/ray-adaptation-20260906"

PROTOCOL = CAMPAIGN / "photoreal-protocol.json"

SNAPSHOT = ROOT / "build/ray-photoreal-code-v2"

TRAIN = ROOT / "runs/ray-adaptation-training-data-v5-20260905"

IMAGE = "0c6505708a4dfece840a2080089b41673c058c933241f5eacc7018a4f3ef7dc2"

COHORTS = {
    "hope146": ("ray-hope", 146, "project_heldout_real"),
    "ycbv42": ("ray-ycbv", 42, "exposed_real_diagnostic"),
    "abo24": ("ray-abo24-calibrated-v2", 24, "adaptation_holdout"),
}


def validate_protocol():
    protocol = read(PROTOCOL)
    assert protocol["status"] == "frozen_before_remaining_training"
    assert protocol["container_image_id"] == IMAGE
    for filename, sha in protocol["scripts"].items():
        assert digest(SNAPSHOT / filename) == sha
    assert {
        p.relative_to(SNAPSHOT / "src").as_posix() for p in (SNAPSHOT / "src").rglob("*.py")
    } == set(protocol["scoring_sources"])
    for filename, sha in protocol["scoring_sources"].items():
        assert digest(SNAPSHOT / "src" / filename) == sha
    assert digest(TRAIN / "complete.json") == protocol["training_sha256"]
    assert digest(Path(protocol["photo"]) / "complete.json") == protocol["photo_sha256"]
    return protocol


def validate_run(run, protocol):
    folder = CAMPAIGN / run
    result = read(folder / "complete.json")
    assert result["status"] == "complete" and result["step"] == 2160
    contract = result["contract"]
    arm, seed = run.split("-seed")
    assert contract["arm"] == arm.upper() and contract["seed"] == int(seed)
    for key, value in protocol["common_training_contract"].items():
        assert contract[key] == value, (run, key)
    assert digest(folder / "final-delta.pt") == result["checkpoint_sha256"]
    if run == "base-seed0":
        assert digest(folder / "complete.json") == protocol["legacy_base_seed0"]["complete_sha256"]
    else:
        assert contract["runner_sha256"] == protocol["scripts"]["preliminary_hope/train_ray.py"]
        assert contract["photo_sha256"] == (protocol["photo_sha256"] if arm == "photo" else None)
        assert contract["photo_validator_sha256"] == (
            protocol["scripts"]["preliminary_hope/training_data.py"] if arm == "photo" else None
        )
    return {
        "run": run,
        "arm": arm.upper(),
        "seed": int(seed),
        "complete_sha256": digest(folder / "complete.json"),
        "checkpoint_sha256": result["checkpoint_sha256"],
    }
