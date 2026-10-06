"""Bind complete generated images, SAM diagnostics and actual visual/author records."""

import argparse
import json
import os
import time
from pathlib import Path

from scripts.common.artifact_bindings import bound, canonical, indexed, local, reference
from scripts.common.paths import ROOT, digest, read, save
from scripts.photo.admission_policy import (
    QUALITY_STATUS,
    RULE,
    row_fields,
    summary,
    validate_admission,
    validate_author,
)


def finalize(production, sam_path, visual_path, author_path, rules_path, out):
    production = local(production)
    out = local(out)
    assert not out.exists(), "Never overwrite an admission or partial attempt"
    assert read(rules_path) == RULE
    generated_complete = read(production / "complete.json")
    assert (
        generated_complete["status"]
        == "all6400_PHOTO_generated_integrity_verified_require_QA_and_admission"
    )
    assert generated_complete["count"] == 6400 and generated_complete["reused_pilot_count"] == 48
    assert generated_complete["plan_sha256"] == digest(production / "plan.json")
    assert not (production / "failure.json").exists()
    input_plan_path = (
        ROOT / "runs/research-evolution-photo-20260907/photo-final-inputs-v1/plan.json"
    )
    plan = read(input_plan_path)
    recipe_sha = canonical(plan["recipe"])
    generated = {}
    origins = {}
    generation_refs = generated_complete["generation_manifests"]
    assert len(generation_refs) == 10
    for ref in generation_refs:
        manifest_path = bound(ref)
        manifest = read(manifest_path)
        verification = ROOT / ref["verification"]
        assert digest(verification) == ref["verification_sha256"]
        check = read(verification)
        assert check["all_passed"] and check["count"] == 640
        assert manifest["status"] == "all640_PHOTO_images_generated_require_admission"
        assert manifest["recipe_sha256"] == recipe_sha
        assert manifest["input_plan_sha256"] == digest(input_plan_path)
        assert (
            manifest["model"] == plan["recipe"]["model"]
            and manifest["revision"] == plan["recipe"]["revision"]
        )
        rows = indexed(manifest["rows"])
        assert len(rows) == 640 and not set(rows).intersection(generated)
        for sid, row in rows.items():
            assert row["condition"] == "reference" and row["masked_or_composited"] is False
            image = local(manifest_path.parent / row["filename"])
            assert image.is_relative_to(manifest_path.parent) and digest(image) == row["sha256"]
            origins[sid] = image
        generated.update(rows)
    assert len(generated) == 6400
    assert sum(row["reused_pilot"] for row in generated.values()) == 48
    sam = read(sam_path)
    assert sam["status"] == "all6400_source_edit_silhouette_diagnostics_recomputed"
    assert sam["pair_count"] == 6400 and sam["mask_count"] == 12800
    pairs = indexed(sam["pairs"])
    assert set(pairs) == set(generated)
    diagnostic_pairs = {}
    assert len(sam["diagnostics"]) == 10
    for ref in sam["diagnostics"]:
        diagnostic = read(bound(ref))
        assert diagnostic["status"] == "all640_source_edit_silhouette_diagnostics_recomputed"
        assert diagnostic["count"] == 1280 and diagnostic["pair_count"] == 640
        assert diagnostic["script_sha256"] == digest(ROOT / "scripts/photo/score_silhouettes.py")
        rows = indexed(diagnostic["pairs"])
        assert not set(rows).intersection(diagnostic_pairs)
        diagnostic_pairs.update(rows)
    assert diagnostic_pairs == pairs
    visual = read(visual_path)
    assert visual["status"] == "partial_individual_visual_reviews_missing_eligible_candidates"
    assert visual["recipe_sha256"] == recipe_sha
    author = read(author_path)
    validate_author(author, recipe_sha)
    assert visual["rules_sha256"] == author["references"]["prior_acceptance_rule"]["sha256"]
    reviews = indexed(visual["rows"])
    eligible = {sid for sid, pair in pairs.items() if pair["calibrated_silhouette_pass"]}
    assert set(reviews).issubset(generated)
    for sid, row in reviews.items():
        assert row["PHOTO_sha256"] == generated[sid]["sha256"]
        assert row["source_rgb_array_sha256"] == generated[sid]["source_rgb_array_sha256"]
        assert row["reviewed_full_and_shared_target_crop"] is True
        assert row["reviewer"] == "assistant_visual_review"
        assert isinstance(row["observation"], str) and row["observation"].strip()
        assert row["decision"] in ["pass", "reject", "uncertain"]
        assert row["evidence"], "Missing bound review evidence"
        for ref in row["evidence"]:
            bound(ref)
    rows = [
        {**image, "filename": "images/" + sid + ".png", **row_fields(pairs[sid], reviews.get(sid))}
        for sid, image in generated.items()
    ]
    refs = {
        name: reference(path)
        for name, path in {
            "production": production / "complete.json",
            "sam_verification": sam_path,
            "visual_review": visual_path,
            "author_control": author_path,
            "acceptance_rule": rules_path,
        }.items()
    }
    out.mkdir()
    (out / "images").mkdir()
    save(
        out / "preparation.json",
        {"status": "binding_verified_images_admission_not_yet_published", "references": refs},
    )
    for row in rows:
        target = out / row["filename"]
        os.link(origins[row["sample_id"]], target)
        assert digest(target) == row["sha256"]
    for ref in refs.values():
        bound(ref)
    admission = {
        "status": "frozen_PHOTO_admission",
        "created_unix": time.time(),
        "rows": rows,
        "generation_manifests": generation_refs,
        "admission_protocol_version": 2,
        "recipe_sha256": recipe_sha,
        "source_plan_sha256": plan["source_plan_sha256"],
        "author_control_sha256": digest(author_path),
        "references": refs,
        "script_sha256": digest(Path(__file__)),
        "software_canary": False,
    }
    quality = {
        "status": QUALITY_STATUS,
        "count": 6400,
        "accepted_count": sum(r["accepted"] for r in rows),
        "admission_rows_sha256": canonical(rows),
        "author_control_sha256": digest(author_path),
        "recipe_sha256": recipe_sha,
        "acceptance_rule": refs["acceptance_rule"],
        "sam_verification": refs["sam_verification"],
        "visual_review": refs["visual_review"],
        "SAM_eligible_count": len(eligible),
        "individually_visually_assessed_count": len(reviews),
        "all_eligible_individually_visually_assessed": False,
        "admission_counts": summary(rows),
        "scope": (
            "Complete integrity and SAM screen; partial individual visual "
            "review. Explicit author authorization admits unreviewed "
            "SAM-eligible candidates, marked separately. Known "
            "reject/uncertain and SAM failures use paired SOURCE fallback. No "
            "independent human ratings, validated photorealism, or measured "
            "PHOTO geometry."
        ),
    }
    validate_admission(admission, quality, author)
    save(out / "quality-control.json", quality)
    save(out / "complete.json", admission)
    return quality


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rules", type=Path, required=True)
    p.add_argument("--write-rules", action="store_true")
    for name in ["production", "sam", "visual", "author", "output"]:
        p.add_argument("--" + name, type=Path)
    a = p.parse_args()
    if a.write_rules:
        assert not a.rules.exists()
        local(a.rules)
        save(a.rules, RULE)
        print("Admission rule recorded; no image accepted or GPU action")
        return
    assert all([a.production, a.sam, a.visual, a.author, a.output])
    print(
        json.dumps(finalize(a.production, a.sam, a.visual, a.author, a.rules, a.output), indent=2)
    )


if __name__ == "__main__":
    main()
