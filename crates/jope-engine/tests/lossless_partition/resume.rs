use super::*;

#[test]
fn persisted_rank_state_and_committed_update_resume_exactly() {
    let b = batch(11, 0);
    let plan = Plan::new(3, 88, 16).unwrap();
    let state = plan
        .state(SEED, 0, update(&plan, &b, 5, 4).unwrap())
        .unwrap();
    let value = serde_json::to_value(&state).unwrap();
    let reloaded: State = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(reloaded, state);
    let next = reloaded.resume(&plan, SEED).unwrap();
    for (rank, slots) in next.iter().enumerate() {
        assert_eq!(*slots, plan.slots(rank, SEED, 5).unwrap());
        assert_eq!(
            state.ranks[rank].sampler_cursor,
            Some(5 * 88 + 8 * plan.shards()[rank].first_group as u64)
        );
    }
    assert_eq!(state.committed_update.empty_terms, Vec::<String>::new());
    assert!(!state.committed_update.clipped);
    // The committed update is consistent with the step and sampling origin.
    assert!(plan
        .state(SEED, 2, update(&plan, &b, 5, 4).unwrap())
        .is_err());
    assert!(plan
        .state(SEED, 2, update(&plan, &b, 5, 2).unwrap())
        .is_ok());
    // Any altered persisted field is refused on reload.
    type Edit = Box<dyn Fn(&mut Value)>;
    let edits: Vec<(&str, Edit)> = vec![
        (
            "sampler cursor",
            Box::new(|s| s["ranks"][1]["sampler_cursor"] = json!(5 * 88 + 8 * 4 + 8)),
        ),
        (
            "dropout seed",
            Box::new(|s| s["ranks"][2]["next_slots"][0]["dropout_seed"] = json!(1)),
        ),
        (
            "next local step",
            Box::new(|s| s["ranks"][0]["next_local_step"] = json!(6)),
        ),
        (
            "missing rank",
            Box::new(|s| drop(s["ranks"].as_array_mut().unwrap().pop())),
        ),
        (
            "partition",
            Box::new(|s| s["partition"]["world"] = json!(4)),
        ),
        (
            "partition digest",
            Box::new(|s| s["partition_sha256"] = json!("c".repeat(64))),
        ),
        (
            "committed step",
            Box::new(|s| s["committed_step"] = json!(6)),
        ),
        (
            "loss",
            Box::new(|s| {
                let loss = s["committed_update"]["loss"].as_f64().unwrap();
                s["committed_update"]["loss"] = json!(loss * 1.01);
            }),
        ),
        (
            "term",
            Box::new(|s| s["committed_update"]["term_values"][1] = json!(9.0)),
        ),
        (
            "clipping",
            Box::new(|s| s["committed_update"]["clipped"] = json!(true)),
        ),
        (
            "empty terms",
            Box::new(|s| s["committed_update"]["empty_terms"] = json!(["ranking_bt"])),
        ),
        (
            "coverage",
            Box::new(|s| s["committed_update"]["comparisons"] = json!(10)),
        ),
        (
            "digest",
            Box::new(|s| s["committed_update"]["gradient_sha256"] = json!("x")),
        ),
        (
            "schema",
            Box::new(|s| s["schema"] = json!("parallel.partition_state.v0")),
        ),
        (
            "unknown field",
            Box::new(|s| s["ranks"][0]["extra"] = json!(1)),
        ),
    ];
    for (what, edit) in edits {
        let mut altered = value.clone();
        edit(&mut altered);
        let refused = serde_json::from_value::<State>(altered)
            .map_err(anyhow::Error::from)
            .and_then(|s| s.resume(&plan, SEED));
        assert!(refused.is_err(), "altered {what} was accepted");
    }
    // A different rank count or microbatch, or seed, is not an exact resume.
    let error = state
        .resume(&Plan::new(4, 88, 16).unwrap(), SEED)
        .unwrap_err();
    assert!(error.to_string().contains("explicit transition"), "{error}");
    assert!(state.resume(&Plan::new(3, 88, 24).unwrap(), SEED).is_err());
    assert!(state.resume(&plan, SEED + 1).is_err());
    // An empty rank persists an explicit empty sampler state.
    let wide = Plan::new(12, 88, 8).unwrap();
    let state = wide
        .state(SEED, 0, update(&wide, &b, 1, 0).unwrap())
        .unwrap();
    assert_eq!(state.ranks[11].sampler_cursor, None);
    assert!(state.ranks[11].next_slots.is_empty());
    assert!(state.resume(&wide, SEED).is_ok());
    // A globally empty term is recorded, and a value for it is refused.
    let tied = batch(11, 11);
    let good = update(&plan, &tied, 5, 4).unwrap();
    assert_eq!(good.empty_terms, ["ranking_bt", "contextfree_bt"]);
    let mut values = good.term_values;
    values[1] = 0.25;
    assert!(CommittedUpdate::new(
        &plan,
        5,
        4,
        &denominators(&tied.rows).unwrap(),
        values,
        &Tensor::from_slice(&[0.0f32]),
        0.0,
        &"b".repeat(64),
    )
    .is_err());
}

const CORPUS: &str = "1111111111111111111111111111111111111111111111111111111111111111";
const VOCABULARY: &str = "3333333333333333333333333333333333333333333333333333333333333333";
const SAMPLER: &str = "family_uniform_champion_plus_seven_quantiles_v2";
