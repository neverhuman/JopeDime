use super::*;

#[test]
fn secondary_endpoint_and_binding() {
    let fixture = prereg_fixture();
    let cli = write_cli_inputs(&fixture);
    let PreregFixture {
        identity,
        plan,
        prereg_path,
        committed_prereg_bytes,
        parsed_committed,
        parent_map,
        ratchets,
        scores,
        parity,
        pinned_prereg,
        pinned_bytes,
        pinned_plan,
        ..
    } = fixture;
    // 1. Verify Endpoint::RatchetVsOrdinaryFinalRound with committed UNPINNED prereg packet.
    let rep_unpinned = qualify_rounds_with_prereg(
        RoundQualInputs {
            plan: &plan,
            identity: &identity,
            ratchet_outputs: &ratchets,
            test_scores: &scores,
            parity: &parity,
            parent_of_context: &parent_map,
        },
        Some(&committed_prereg_bytes),
    )
    .expect("valid report with committed unpinned prereg");

    assert_eq!(rep_unpinned.endpoint, Endpoint::RatchetVsOrdinaryFinalRound);
    assert_eq!(
        rep_unpinned.contexts[0].candidate_genome.as_deref(),
        Some("g_rat_final_ctx1")
    );
    assert_eq!(
        rep_unpinned.contexts[0].reference_genome.as_deref(),
        Some("g_ord_final_ctx1")
    );
    assert_eq!(rep_unpinned.verdict, Verdict::Greater);
    assert!(rep_unpinned.margin_cleared);
    assert!(rep_unpinned.caller_asserted_diagnostic_checks_passed);
    assert_eq!(
        rep_unpinned.provenance_binding_status,
        PREREG_UNPINNED_BINDING_STATUS
    );
    assert!(!rep_unpinned.qualified_claim);
    assert!(!rep_unpinned.promotion);
    assert!(!rep_unpinned.l6_claim);
    assert_eq!(
        rep_unpinned.disqualification_reasons,
        vec![
            "prereg_unpinned_or_no_execution_grant".to_owned(),
            PREREG_CALLER_ASSERTED_REASON.to_owned(),
            "qualified_execution_mode_disabled_offline_verifier_only".to_owned(),
            "no_adopted_sealed_family_boundary".to_owned(),
        ]
    );

    // 2. Verify CLI `jope_pools qualify-rounds --prereg <PATH>` produces the exact same canonical digest
    // and includes provenance_binding_status in stdout.
    let plan_file = &cli.plan_file;
    let id_file = &cli.id_file;
    let ratchets_file = &cli.ratchets_file;
    let scores_file = &cli.scores_file;
    let parity_file = &cli.parity_file;
    let parents_file = &cli.parents_file;
    let out_file = &cli.out_file;

    let bin = env!("CARGO_BIN_EXE_jope_pools");
    let cli_out = Command::new(bin)
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
            prereg_path.to_str().unwrap(),
            "--out",
            out_file.to_str().unwrap(),
        ])
        .output()
        .expect("run jope_pools qualify-rounds --prereg");
    assert!(
        cli_out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&cli_out.stderr)
    );
    let stdout = String::from_utf8_lossy(&cli_out.stdout);
    assert!(stdout.contains(&rep_unpinned.canonical_sha256().unwrap()));
    assert!(stdout.contains(PREREG_UNPINNED_BINDING_STATUS));

    // 3. Construct a fully pinned SelfPlayPrereg packet and verify provenance_binding_status transitions
    // to PREREG_PINNED_BINDING_STATUS while qualified_claim/promotion/l6_claim stay false with
    // both caller-asserted and offline-verifier-only disqualification reasons.
    let rep_pinned = qualify_rounds_with_prereg(
        RoundQualInputs {
            plan: &pinned_plan,
            identity: &identity,
            ratchet_outputs: &ratchets,
            test_scores: &scores,
            parity: &parity,
            parent_of_context: &parent_map,
        },
        Some(&pinned_bytes),
    )
    .expect("valid report with pinned prereg");
    assert_eq!(
        rep_pinned.provenance_binding_status,
        PREREG_PINNED_BINDING_STATUS
    );
    assert!(rep_pinned.caller_asserted_diagnostic_checks_passed);
    assert!(!rep_pinned.qualified_claim);
    assert!(!rep_pinned.promotion);
    assert!(!rep_pinned.l6_claim);
    assert_eq!(
        rep_pinned.disqualification_reasons,
        vec![
            PREREG_CALLER_ASSERTED_REASON.to_owned(),
            "qualified_execution_mode_disabled_offline_verifier_only".to_owned(),
            "no_adopted_sealed_family_boundary".to_owned(),
        ]
    );

    // Also verify canonical SHA-256 binding when plan.prereg_sha256 is canonical_sha256(&pinned_prereg)
    // and the file is pretty-printed JSON.
    let mut canonical_pinned_plan = pinned_plan.clone();
    canonical_pinned_plan.prereg_sha256 = canonical_sha256(&pinned_prereg).unwrap();
    assert_ne!(
        canonical_pinned_plan.prereg_sha256,
        pinned_plan.prereg_sha256
    );
    let rep_canonical_pinned = qualify_rounds_with_prereg(
        RoundQualInputs {
            plan: &canonical_pinned_plan,
            identity: &identity,
            ratchet_outputs: &ratchets,
            test_scores: &scores,
            parity: &parity,
            parent_of_context: &parent_map,
        },
        Some(&pinned_bytes),
    )
    .expect("valid report with canonical prereg_sha256");
    assert_eq!(
        rep_canonical_pinned.prereg_sha256,
        canonical_pinned_plan.prereg_sha256
    );
    assert_eq!(
        rep_canonical_pinned.provenance_binding_status,
        PREREG_PINNED_BINDING_STATUS
    );
    assert_eq!(
        rep_canonical_pinned.disqualification_reasons,
        rep_pinned.disqualification_reasons
    );
}
