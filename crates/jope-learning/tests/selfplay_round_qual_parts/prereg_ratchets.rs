use super::*;

#[test]
fn rejects_invalid_ratchet_provenance() {
    let fixture = prereg_fixture();
    let PreregFixture {
        identity,
        plan: _,
        parent_map,
        ratchets,
        scores,
        parity,
        pinned_bytes,
        pinned_plan,
        ..
    } = fixture;
    // 6. Ratchet provenance cross-check failures when --prereg is supplied:
    type RatchetMutator = (&'static str, Box<dyn Fn(&mut Vec<Value>)>);
    let ratchet_mutations: Vec<RatchetMutator> = vec![
        (
            "seed_cap_b_mismatch",
            Box::new(|r| r[0]["seed_cap_b"] = json!(4)),
        ),
        (
            "max_conditional_base_kl_missing",
            Box::new(|r| {
                r[0].as_object_mut()
                    .unwrap()
                    .remove("max_conditional_base_kl");
            }),
        ),
        (
            "max_conditional_base_kl_mismatch",
            Box::new(|r| r[0]["max_conditional_base_kl"] = json!(0.19)),
        ),
        (
            "max_new_support_mass_missing",
            Box::new(|r| {
                r[0].as_object_mut().unwrap().remove("max_new_support_mass");
            }),
        ),
        (
            "max_new_support_mass_mismatch",
            Box::new(|r| r[0]["max_new_support_mass"] = json!(0.14)),
        ),
        (
            "measured_tilt_policy_missing",
            Box::new(|r| {
                r[0].as_object_mut().unwrap().remove("measured_tilt_policy");
            }),
        ),
        (
            "measured_tilt_policy_tau_mismatch",
            Box::new(|r| r[0]["measured_tilt_policy"]["temperature_tau"] = json!(0.50)),
        ),
        (
            "rounds_len_mismatch",
            Box::new(|r| {
                r[0]["rounds"].as_array_mut().unwrap().pop();
            }),
        ),
        (
            "round_kl_source_caller_evaluated_rejected_under_prereg",
            Box::new(|r| {
                r[0]["rounds"][1]["kl_source"] = json!("caller_evaluated_leaf_log_probs");
            }),
        ),
        (
            "ordinary_champion_genome_missing_under_prereg",
            Box::new(|r| {
                r[0]["rounds"][0]
                    .as_object_mut()
                    .unwrap()
                    .remove("ordinary_champion_genome");
            }),
        ),
        (
            "ordinary_champion_genome_sentinel_under_prereg",
            Box::new(|r| {
                r[0]["rounds"][0]["ordinary_champion_genome"] = json!(" UNPINNED ");
            }),
        ),
        (
            "ordinary_search_sha256_invalid",
            Box::new(|r| {
                r[0]["rounds"][2]["ordinary_search_sha256"] = json!("not-64-hex");
            }),
        ),
        (
            "ratchet_observations_sha256_invalid",
            Box::new(|r| {
                r[0]["rounds"][2]["ratchet_observations_sha256"] = json!("not-64-hex");
            }),
        ),
    ];

    for (label, mutate) in &ratchet_mutations {
        let mut bad_ratchets = ratchets.clone();
        mutate(&mut bad_ratchets);
        assert!(
            qualify_rounds_with_prereg(
                RoundQualInputs {
                    plan: &pinned_plan,
                    identity: &identity,
                    ratchet_outputs: &bad_ratchets,
                    test_scores: &scores,
                    parity: &parity,
                    parent_of_context: &parent_map,
                },
                Some(&pinned_bytes),
            )
            .is_err(),
            "expected ratchet provenance mutation `{label}` to be rejected under --prereg"
        );
    }
}
