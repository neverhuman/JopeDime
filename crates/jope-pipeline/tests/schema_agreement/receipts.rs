use super::*;

#[test]
fn v3_islands_records_are_checked() {
    let migration = |from: u64, to: u64, after: u64| {
        json!({"cut": 0, "after_generation": after, "from_island": from, "to_island": to,
               "program_sha256": "d".repeat(64), "rmse": 1.5, "admitted": true})
    };
    let islands = |count: u64, migrations: Vec<Value>| {
        let mut v = fixture();
        v["search"]["islands"] =
            json!({"count": count, "migration_interval": 2, "migrations": migrations});
        v
    };
    validate_value(&islands(3, vec![migration(0, 1, 1), migration(2, 0, 3)])).unwrap();
    let cases = [
        (
            islands(1, vec![migration(0, 1, 1)]),
            "1 island(s) with 1 migration(s)",
        ),
        (islands(0, vec![]), "0 island(s)"),
        (
            islands(3, vec![migration(0, 3, 1)]),
            "not a migration between",
        ),
        (
            islands(3, vec![migration(1, 1, 1)]),
            "not a migration between",
        ),
        (
            islands(3, vec![migration(0, 1, 2)]),
            "not a migration between",
        ),
    ];
    for (value, expected) in cases {
        let errors = validate_value(&value).unwrap_err();
        assert!(
            errors.iter().any(|e| e.contains(expected)),
            "wanted {expected:?}, got {errors:?}"
        );
    }
    let mut missing = fixture();
    missing["search"].as_object_mut().unwrap().remove("islands");
    assert!(validate_value(&missing).is_err());
}

#[test]
fn v2_receipts_still_verify_and_cannot_claim_v3_seeding() {
    let receipt = validate_value(&v2_fixture()).unwrap();
    assert_eq!(receipt.schema, V2_SCHEMA_ID);
    assert!(receipt.search.seeding.is_none());
    // A v2 receipt is checked by the v2 contract, which has no seeding field at all.
    let mut claimed = v2_fixture();
    claimed["search"]["seeding"] = fixture()["search"]["seeding"].clone();
    let errors = validate_value(&claimed).unwrap_err();
    assert!(errors.iter().any(|e| e.contains("seeding")), "{errors:?}");
    // And v2's own budget invariants still apply to it.
    let mut overspent = v2_fixture();
    overspent["search"]["evaluations"] = json!(4);
    let errors = validate_value(&overspent).unwrap_err();
    assert!(
        errors
            .iter()
            .any(|e| e.contains("total budget/phase accounting mismatch")),
        "{errors:?}"
    );
}

#[test]
fn v3_seeding_records_are_checked_against_digests_cap_and_coverage() {
    let sha = jope_pipeline::sha256_hex(b"(pipe X (ridge))");
    let seed_genome = "(pipe X (standardize) (ridge))";
    let seed_sha = jope_pipeline::sha256_hex(seed_genome.as_bytes());
    let seeded = |source: &str| {
        let mut v = fixture();
        v["search"]["seeding"] = json!({
            "offered": 1,
            "seeds": [{"genome": seed_genome, "program_sha256": seed_sha, "source": source}],
            "parent_receipt_sha256": "c".repeat(64),
            "seeds_file_sha256": null
        });
        // The seed was attempted on the one requested cut.
        v["search"]["coverage"][1]["program_sha256"] = json!(seed_sha);
        v
    };
    validate_value(&seeded("incumbent")).unwrap();

    let mut cases: Vec<(Value, &str)> = Vec::new();
    let mut v = fixture();
    v["search"].as_object_mut().unwrap().remove("seeding");
    cases.push((v, "seeding"));
    let mut v = seeded("incumbent");
    v["search"]["coverage"][1]["program_sha256"] = json!(format!("{:064x}", 1));
    cases.push((v, "not attempted on every requested cut"));
    let mut v = seeded("incumbent");
    v["search"]["seeding"]["seeds"][0]["program_sha256"] = json!(sha);
    cases.push((v, "digest, duplicate or bootstrap mismatch"));
    cases.push((seeded("seeds_file"), "without a seeds file digest"));
    cases.push((seeded("oracle"), "unknown"));
    let mut v = seeded("incumbent");
    v["search"]["seeding"]["offered"] = json!(0);
    cases.push((v, "exceeds the cap"));
    let mut v = seeded("incumbent");
    v["search"]["seeding"]["parent_receipt_sha256"] = json!("short");
    cases.push((v, "match"));
    for (value, expected) in cases {
        let errors = validate_value(&value).unwrap_err();
        assert!(
            errors.iter().any(|e| e.contains(expected)),
            "wanted {expected:?}, got {errors:?}"
        );
    }
}

#[test]
fn a_well_formed_receipt_validates() {
    validate_value(&fixture()).unwrap();
}

/// A run that emitted nothing must not be able to describe itself as one that emitted code.
/// The CLI already fails such a run, but the receipt is what a later reader trusts, so the
/// contradiction is refused here too rather than only at the moment of the run.
#[test]
fn the_validator_refuses_an_emit_stage_that_disagrees_with_the_artifacts() {
    // `passed` with nothing emitted.
    let mut receipt = fixture();
    receipt["artifacts"] = json!([]);
    receipt["emit_exclusions"] = json!([
        {"language": "rust", "reason": "the emitter does not cover head gpr"}
    ]);
    let errors = validate_value(&receipt).unwrap_err();
    assert!(
        errors
            .iter()
            .any(|e| e.contains("emit passed but the receipt names no artifact")),
        "{errors:?}"
    );

    // `failed` while artifacts are listed: the mirror image, equally a lie.
    let mut receipt = fixture();
    receipt["ok"] = json!(false);
    receipt["stages"][5]["status"] = json!("failed");
    receipt["stages"][5]["reason"] = json!("rust: the emitter does not cover head gpr");
    let errors = validate_value(&receipt).unwrap_err();
    assert!(
        errors
            .iter()
            .any(|e| e.contains("emit failed but the receipt names 1 artifact")),
        "{errors:?}"
    );

    // `ok` may not survive a failed stage.
    let mut receipt = fixture();
    receipt["artifacts"] = json!([]);
    receipt["stages"][5]["status"] = json!("failed");
    receipt["stages"][5]["reason"] = json!("rust: the emitter does not cover head gpr");
    let errors = validate_value(&receipt).unwrap_err();
    assert!(
        errors
            .iter()
            .any(|e| e.contains("ok is true but a stage failed")),
        "{errors:?}"
    );

    // And the honest shape of the same run validates.
    let mut receipt = fixture();
    receipt["ok"] = json!(false);
    receipt["artifacts"] = json!([]);
    receipt["stages"][5]["status"] = json!("failed");
    receipt["stages"][5]["reason"] = json!("rust: the emitter does not cover head gpr");
    receipt["emit_exclusions"] = json!([
        {"language": "rust", "reason": "the emitter does not cover head gpr"}
    ]);
    validate_value(&receipt).unwrap();
}

#[test]
fn the_validator_rejects_what_the_schema_and_the_invariants_forbid() {
    let cases = [
        ("schema", json!("jope.pipeline-receipt/v99"), "const"),
        ("policy", json!("maybe"), "policy"),
        ("observations_sha256", json!("short"), "match"),
    ];
    for (key, value, needle) in cases {
        let mut receipt = fixture();
        receipt[key] = value;
        let errors = validate_value(&receipt).unwrap_err();
        assert!(
            errors.iter().any(|e| e.contains(needle)),
            "{key}: {errors:?} has no {needle}"
        );
    }
    // An unknown key is refused rather than ignored.
    let mut receipt = fixture();
    receipt["surprise"] = json!(1);
    assert!(validate_value(&receipt).is_err());
    // A skipped stage without a reason is refused by the invariants.
    let mut receipt = fixture();
    receipt["stages"][2]
        .as_object_mut()
        .unwrap()
        .remove("reason");
    let errors = validate_value(&receipt).unwrap_err();
    assert!(
        errors.iter().any(|e| e.contains("must record a reason")),
        "{errors:?}"
    );
    // A missing stage is refused.
    let mut receipt = fixture();
    receipt["stages"] = json!([]);
    let errors = validate_value(&receipt).unwrap_err();
    assert!(
        errors
            .iter()
            .any(|e| e.contains("stage propose is missing")),
        "{errors:?}"
    );
    // A search without a best program must say it fell back.
    let mut receipt = fixture();
    receipt["search"].as_object_mut().unwrap().remove("best");
    receipt["search"]["cuts"][0]["successful_measurements"] = json!(0);
    receipt["search"]["cuts"][0]["failed_or_unavailable_measurements"] = json!(8);
    receipt["search"]["cuts"][0]
        .as_object_mut()
        .unwrap()
        .remove("best_program");
    let errors = validate_value(&receipt).unwrap_err();
    assert!(
        errors.iter().any(|e| e.contains("without a best program")),
        "{errors:?}"
    );

    // Measurements plus the unmeasured fallback is a contradiction.
    let mut receipt = fixture();
    receipt["search"]["selection"] = json!("unmeasured_fallback_bootstrap_genome");
    receipt["search"].as_object_mut().unwrap().remove("best");
    let errors = validate_value(&receipt).unwrap_err();
    assert!(
        errors.iter().any(|e| e.contains("unmeasured fallback")),
        "{errors:?}"
    );

    // A cut's evaluations must account for every candidate's outcome.
    let mut receipt = fixture();
    receipt["search"]["cuts"][0]["completed_evaluations"] = json!(99);
    let errors = validate_value(&receipt).unwrap_err();
    assert!(errors.iter().any(|e| e.contains("outcomes")), "{errors:?}");

    // A cut that measured something must name what it measured.
    let mut receipt = fixture();
    receipt["search"]["cuts"][0]
        .as_object_mut()
        .unwrap()
        .remove("best_program");
    let errors = validate_value(&receipt).unwrap_err();
    assert!(
        errors.iter().any(|e| e.contains("no best program")),
        "{errors:?}"
    );

    // `stopped_by` is closed: a cut cannot claim an invented bound.
    let mut receipt = fixture();
    receipt["search"]["cuts"][0]["stopped_by"] = json!("boredom");
    assert!(validate_value(&receipt).is_err());
}

#[test]
fn old_v1_receipts_remain_readable_without_claiming_v2_budget_guarantees() {
    let mut value = v2_fixture();
    value["schema"] = json!(jope_pipeline::receipt::LEGACY_SCHEMA_ID);
    for key in ["budget_policy", "compared_cuts", "coverage"] {
        value["search"].as_object_mut().unwrap().remove(key);
    }
    value["search"]["cuts"][0]
        .as_object_mut()
        .unwrap()
        .remove("accounting");
    value["search"]["selection"] = json!("min_mean_validation_rmse");
    value["search"]["evaluations"] = json!(4); // historical overspend remains historical
    let receipt = validate_value(&value).unwrap();
    assert_eq!(receipt.schema, jope_pipeline::receipt::LEGACY_SCHEMA_ID);
    assert!(receipt.search.cuts[0].accounting.is_none());
}

#[test]
fn v2_rejects_overspending_missing_accounting_and_incomparable_cut_sets() {
    let mut cases = Vec::new();
    let mut v = fixture();
    v["search"]["evaluations"] = json!(4);
    cases.push((v, "total budget/phase accounting mismatch"));
    let mut v = fixture();
    v["search"]["cuts"][0]
        .as_object_mut()
        .unwrap()
        .remove("accounting");
    cases.push((v, "accounting"));
    let mut v = fixture();
    v["search"]["compared_cuts"] = json!([1]);
    cases.push((v, "must compare all requested cuts"));
    let mut v = fixture();
    v["search"]["coverage"][0]["measured_cuts"] = json!([]);
    cases.push((v, "selected program lacks complete bound cut coverage"));
    let mut v = fixture();
    v["search"]["cuts"][0]["accounting"]["deadline_overrun"] = json!(true);
    cases.push((v, "total budget/phase accounting mismatch"));
    let mut v = fixture();
    v["search"]["cuts"][0]["accounting"]["comparison"]["completed_evaluations"] = json!(1);
    cases.push((v, "invalid phase outcomes or admission limit"));
    let mut v = fixture();
    v["search"]["cuts"][0]["accounting"]["comparison"]["stopped_by"] = json!("seconds");
    cases.push((v, "phase stop reason mismatch"));
    let mut v = fixture();
    v["search"]["cuts"][0]["accounting"]["exploration"]["stopped_by"] = json!("not_requested");
    cases.push((v, "exploration stop reason mismatch"));
    let mut v = fixture();
    v["search"]["cuts"][0]["accounting"]["exploration"]["successful_measurements"] =
        json!(u64::MAX);
    cases.push((v, "invalid phase outcomes or admission limit"));
    for (value, expected) in cases {
        let errors = validate_value(&value).unwrap_err();
        assert!(
            errors.iter().any(|e| e.contains(expected)),
            "wanted {expected:?}, got {errors:?}"
        );
    }
}
