use super::*;

#[test]
fn preserves_legacy_unpinned_behavior() {
    let fixture = prereg_fixture();
    let cli = write_cli_inputs(&fixture);
    let PreregFixture {
        identity,
        plan,
        parsed_committed: _,
        parent_map,
        ratchets,
        scores,
        parity,
        pinned_bytes,
        pinned_plan: _,
        ..
    } = fixture;
    // 7. Verify R3 A-M3 & R2 P3-3 backward compatibility on legacy no-`--prereg` calls:
    // (a) `kl_source = "caller_evaluated_leaf_log_probs"` remains accepted by `qualify_rounds` (no `--prereg`).
    let mut caller_kl_ratchets = ratchets.clone();
    caller_kl_ratchets[0]["rounds"][1]["kl_source"] = json!("caller_evaluated_leaf_log_probs");
    let rep_caller_kl_no_prereg = qualify_rounds(RoundQualInputs {
        plan: &plan,
        identity: &identity,
        ratchet_outputs: &caller_kl_ratchets,
        test_scores: &scores,
        parity: &parity,
        parent_of_context: &parent_map,
    })
    .expect("legacy qualify_rounds without --prereg still accepts caller_evaluated_leaf_log_probs");
    assert_eq!(rep_caller_kl_no_prereg.n_unmeasured_kl_contexts, 0);

    // (b) Legacy receipt omitting `ordinary_champion_genome`:
    //     - deserializes cleanly into `SelfPlayRoundReceipt` (`ordinary_champion_genome == ""`),
    //     - is rejected on `Endpoint::RatchetVsOrdinaryFinalRound` when missing on final round,
    //     - is accepted on `Endpoint::FinalVsRound0Incumbent` without `--prereg`.
    let mut legacy_no_ord_ratchets = ratchets.clone();
    for r in &mut legacy_no_ord_ratchets {
        for round_obj in r["rounds"].as_array_mut().unwrap() {
            round_obj
                .as_object_mut()
                .unwrap()
                .remove("ordinary_champion_genome");
        }
    }
    assert!(qualify_rounds(RoundQualInputs {
        plan: &plan,
        identity: &identity,
        ratchet_outputs: &legacy_no_ord_ratchets,
        test_scores: &scores,
        parity: &parity,
        parent_of_context: &parent_map,
    })
    .is_err());

    let mut primary_plan = plan.clone();
    primary_plan.endpoint = Endpoint::FinalVsRound0Incumbent;
    let mut primary_scores = Vec::new();
    let mut primary_parity = Vec::new();
    for ctx in ["ctx1", "ctx2", "ctx3", "ctx4"] {
        let g_r0 = format!("g_r0_{ctx}");
        let g_rat_final = format!("g_rat_final_{ctx}");
        let ref_s = make_score(ctx, &g_r0, 1.00, Vec::new());
        let cand_s = make_score(ctx, &g_rat_final, 0.72, vec![ref_s.program_sha256.clone()]);
        primary_scores.push(ref_s);
        primary_scores.push(cand_s);
        for g in [&g_r0, &g_rat_final] {
            primary_parity.push(ParityRecord {
                dataset_id: ctx.to_owned(),
                genome: (*g).clone(),
                passed: true,
                failure_reason: None,
            });
        }
    }
    let rep_legacy_primary = qualify_rounds(RoundQualInputs {
    plan: &primary_plan,
    identity: &identity,
    ratchet_outputs: &legacy_no_ord_ratchets,
    test_scores: &primary_scores,
    parity: &primary_parity,
    parent_of_context: &parent_map,
})
.expect("legacy receipt without ordinary_champion_genome succeeds on FinalVsRound0Incumbent without --prereg");
    assert_eq!(
        rep_legacy_primary.endpoint,
        Endpoint::FinalVsRound0Incumbent
    );

    // 8. CLI negative cases (`--prereg` missing file, hash mismatch, duplicate JSON key):
    let dir = &cli.dir;
    let plan_file = &cli.plan_file;
    let id_file = &cli.id_file;
    let ratchets_file = &cli.ratchets_file;
    let scores_file = &cli.scores_file;
    let parity_file = &cli.parity_file;
    let parents_file = &cli.parents_file;
    let bin = env!("CARGO_BIN_EXE_jope_pools");
    let dup_false_then_true = String::from_utf8(pinned_bytes.clone()).unwrap().replacen(
        "\"authorized\": true",
        "\"authorized\": false,\n    \"authorized\": true",
        1,
    );
    let missing_prereg_out = dir.path().join("report_missing_prereg.json");
    let cli_missing = Command::new(bin)
        .args([
            "qualify-rounds",
            "--plan",
            plan_file.to_str().unwrap(),
            "--identity",
            id_file.to_str().unwrap(),
            "--ratchets",
            ratchets_file.to_str().unwrap(),
            "--test-scores",
            scores_file.to_str().unwrap(),
            "--parity",
            parity_file.to_str().unwrap(),
            "--parent-map",
            parents_file.to_str().unwrap(),
            "--prereg",
            dir.path().join("does_not_exist.json").to_str().unwrap(),
            "--out",
            missing_prereg_out.to_str().unwrap(),
        ])
        .output()
        .expect("run cli with missing prereg");
    assert!(!cli_missing.status.success());

    let dup_prereg_file = dir.path().join("dup_prereg.json");
    std::fs::write(&dup_prereg_file, dup_false_then_true.as_bytes()).unwrap();
    let dup_out_file = dir.path().join("report_dup_prereg.json");
    let cli_dup = Command::new(bin)
        .args([
            "qualify-rounds",
            "--plan",
            plan_file.to_str().unwrap(),
            "--identity",
            id_file.to_str().unwrap(),
            "--ratchets",
            ratchets_file.to_str().unwrap(),
            "--test-scores",
            scores_file.to_str().unwrap(),
            "--parity",
            parity_file.to_str().unwrap(),
            "--parent-map",
            parents_file.to_str().unwrap(),
            "--prereg",
            dup_prereg_file.to_str().unwrap(),
            "--out",
            dup_out_file.to_str().unwrap(),
        ])
        .output()
        .expect("run cli with duplicate-key prereg");
    assert!(!cli_dup.status.success());
}
