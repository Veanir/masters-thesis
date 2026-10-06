"""Explicit author authorization for unreviewed SAM-eligible PHOTO candidates.

This records admission without inventing visual judgments or author ratings.
Historical v1 controls retain their original meaning.
"""

from scripts.common.artifact_bindings import bound, canonical, indexed, read
from scripts.common.paths import script_help

script_help(__doc__, __name__)

LEGACY_RULE = {
    "status": "PHOTO_admission_rule_before_MAIN_and_HB",
    "source_IoU_min": 0.98,
    "edited_IoU_min": 0.98,
    "max_direction_boundary_p95_pixels_at640width": 3.0,
    "acceptance": "calibrated_silhouette_pass AND visual_decision=pass",
    "visual_coverage": (
        "Every calibrated SAM passing candidate requires a bound "
        "individual visual decision. SAM failures fall back without "
        "requiring an extra visual acceptance judgment."
    ),
    "visual_pass": (
        "SOURCE and PHOTO full frames plus a shared target crop were "
        "inspected; no specific unresolved added, missing or displaced "
        "part, opening, edge, contact or occlusion cue was observed."
    ),
    "visual_uncertainty": (
        "A specific unresolved structural or depth-cue concern is not a "
        "pass. Clean SOURCE depth/normals may resolve a concern; bind "
        "that follow-up explicitly."
    ),
    "appearance": (
        "Record visible appearance changes separately. A visual pass is "
        "not a validated perceptual-realism score or a certificate of "
        "metric PHOTO geometry."
    ),
    "rejected_fallback": (
        "Use unchanged SOURCE in both PHOTO and CLASSIC on the same "
        "rejected observations. Keep all geometries, observations and "
        "generated images."
    ),
    "author_control": (
        "Actual responses for the fixed four source/edit pairs and "
        "confirmation of the unchanged recipe are required before final "
        "admission."
    ),
    "regeneration": (
        "None; preserve the fixed generated image and seed, including rejected pilot images."
    ),
    "test_quality_used": False,
}


def legacy_decision(pair, visual):
    assert all(
        type(pair[k]) is bool
        for k in ["source_calibrated", "edited_raw_screen_pass", "calibrated_silhouette_pass"]
    )
    assert pair["calibrated_silhouette_pass"] == (
        pair["source_calibrated"] and pair["edited_raw_screen_pass"]
    )
    if not pair["calibrated_silhouette_pass"]:
        return False, "source_or_edit_SAM_screen_failed"
    assert visual is not None, "Missing required individual visual decision"
    assert visual["decision"] in ["pass", "reject", "uncertain"]
    return visual["decision"] == "pass", "visual_" + visual["decision"]


AUTHOR_STATUS = "author_authorized_unreviewed_SAM_eligible_PHOTO_use"
QUALITY_STATUS = "all6400_PHOTO_integrity_SAM_and_author_scoped_admission_complete"
RULE = {
    **LEGACY_RULE,
    "protocol_version": 2,
    "acceptance": (
        "calibrated_silhouette_pass AND (visual_decision=pass OR "
        "unreviewed_with_explicit_author_authorization)"
    ),
    "visual_coverage": (
        "Preserve all existing individual reviews. Admit unreviewed "
        "SAM-eligible pairs explicitly as unreviewed; never relabel them "
        "pass. Known reject/uncertain and every SAM failure use SOURCE in "
        "both PHOTO and CLASSIC."
    ),
    "author_control": (
        "The explicit author authorization bound to this generated "
        "population supersedes the pending four-image review gate for use "
        "of the unchanged recipe. No four-image ratings or validated "
        "photorealism are inferred."
    ),
    "population_inference": (
        "The existing sequential partial review is not a random sample. "
        "Its rejection/uncertainty rates are not estimates for unreviewed "
        "candidates."
    ),
}
EXPECTED = {
    "accepted": 3344,
    "accepted_visual_pass": 866,
    "accepted_unreviewed": 2478,
    "SOURCE_fallback": 3056,
    "SAM_failed": 2846,
    "eligible_visual_reject": 90,
    "eligible_visual_uncertain": 120,
    "SAM_eligible": 3554,
    "reviewed_all": 1292,
    "reviewed_eligible": 1076,
}


def decision(pair, visual):
    # Keep the v1 calibrated-SAM boolean invariants and fail closed for
    # malformed reviews, even when SAM alone already causes fallback.
    if visual is not None:
        assert visual["decision"] in ["pass", "reject", "uncertain"]
    assert all(
        type(pair[k]) is bool
        for k in ["source_calibrated", "edited_raw_screen_pass", "calibrated_silhouette_pass"]
    )
    assert pair["calibrated_silhouette_pass"] == (
        pair["source_calibrated"] and pair["edited_raw_screen_pass"]
    )
    if pair["calibrated_silhouette_pass"] and visual is None:
        return True, "author_authorized_unreviewed_SAM_eligible"
    return legacy_decision(pair, visual)


def row_fields(pair, visual):
    accepted, reason = decision(pair, visual)
    status = visual["decision"] if visual else "unreviewed"
    return {
        "accepted": accepted,
        "admission_reason": reason,
        "SAM_eligible": pair["calibrated_silhouette_pass"],
        "visual_review_status": status,
        "individually_visually_reviewed": visual is not None,
        "author_unreviewed_waiver_applied": accepted and visual is None,
        "paired_augmentation_eligible": accepted,
    }


def summary(rows):
    return {
        "accepted": sum(r["accepted"] for r in rows),
        "accepted_visual_pass": sum(
            r["accepted"] and r["visual_review_status"] == "pass" for r in rows
        ),
        "accepted_unreviewed": sum(r["author_unreviewed_waiver_applied"] for r in rows),
        "SOURCE_fallback": sum(not r["accepted"] for r in rows),
        "SAM_failed": sum(not r["SAM_eligible"] for r in rows),
        "eligible_visual_reject": sum(
            r["SAM_eligible"] and r["visual_review_status"] == "reject" for r in rows
        ),
        "eligible_visual_uncertain": sum(
            r["SAM_eligible"] and r["visual_review_status"] == "uncertain" for r in rows
        ),
        "SAM_eligible": sum(r["SAM_eligible"] for r in rows),
        "reviewed_all": sum(r["individually_visually_reviewed"] for r in rows),
        "reviewed_eligible": sum(
            r["SAM_eligible"] and r["individually_visually_reviewed"] for r in rows
        ),
    }


def validate_author(author, recipe_sha):
    assert author["status"] == AUTHOR_STATUS
    assert author["author_response_recorded"] is True
    assert author["authorized_recipe_sha256"] == recipe_sha
    assert author["scope"] == "only_unreviewed_SAM_eligible_keep_known_reject_uncertain_SOURCE"
    assert author["source_author_message"].strip() and author["source_author_confirmation"].strip()
    assert author["expected_counts"] == EXPECTED
    assert author["fixed_sample_assessments"] == []
    assert author["per_image_author_ratings_recorded"] is False
    assert author["photorealism_validated"] is False
    assert author["test_quality_used"] is False and author["regeneration"] is False
    refs = author["references"]
    for ref in refs.values():
        bound(ref)
    assert read(bound(refs["prior_acceptance_rule"])) == LEGACY_RULE
    assert canonical(read(bound(refs["input_plan"]))["recipe"]) == recipe_sha


def validate_admission(admission, quality, author):
    validate_author(author, admission["recipe_sha256"])
    assert quality["status"] == QUALITY_STATUS and quality["count"] == 6400
    assert admission["admission_protocol_version"] == 2
    refs = admission["references"]
    for ref in refs.values():
        bound(ref)
    assert read(bound(refs["acceptance_rule"])) == RULE
    for key in ["production", "sam_verification", "visual_review"]:
        assert refs[key] == author["references"][key]
    for key in ["acceptance_rule", "sam_verification", "visual_review"]:
        assert refs[key] == quality[key]
    sam = read(bound(refs["sam_verification"]))
    assert sam["status"] == "all6400_source_edit_silhouette_diagnostics_recomputed"
    assert (sam["pair_count"], sam["mask_count"]) == (6400, 12800)
    pairs = indexed(sam["pairs"])
    visual = read(bound(refs["visual_review"]))
    assert visual["status"] == "partial_individual_visual_reviews_missing_eligible_candidates"
    assert visual["rules_sha256"] == author["references"]["prior_acceptance_rule"]["sha256"]
    reviews = indexed(visual["rows"])
    rows = indexed(admission["rows"])
    assert len(rows) == 6400 and set(rows) == set(pairs) and set(reviews).issubset(rows)
    for sid, row in rows.items():
        expected = row_fields(pairs[sid], reviews.get(sid))
        assert all(row[k] == value and type(row[k]) is type(value) for k, value in expected.items())
        if sid in reviews:
            review = reviews[sid]
            assert review["PHOTO_sha256"] == row["sha256"]
            assert review["source_rgb_array_sha256"] == row["source_rgb_array_sha256"]
    assert summary(admission["rows"]) == quality["admission_counts"] == EXPECTED
    assert quality["admission_rows_sha256"] == canonical(admission["rows"])
    assert quality["all_eligible_individually_visually_assessed"] is False
