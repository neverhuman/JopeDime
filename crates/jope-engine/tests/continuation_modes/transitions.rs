use super::*;

#[test]
fn a_carry_transition_allows_only_declared_changes() {
    let mut request = Request::same();
    request.corpus = NEW_CORPUS;
    request.seed = 9;
    request.sampler = "lineage_fresh_program_champion_plus_seven_v2";
    let record = request
        .check(Mode::NewCorpusTransition, &parent(), None)
        .unwrap();
    assert_eq!(record["mode"], "new_corpus_transition");
    assert_eq!(record["exact_resume_claim"], false);
    assert_eq!(record["optimizer"], "carried");
    assert_eq!(record["sampler"], "new_window_from_zero");
    assert_eq!(
        record["corpus"],
        json!({"parent":CORPUS,"current":NEW_CORPUS})
    );
    assert_eq!(record["objective"]["continuity"], "identical");
    // An execution-partition change must be declared, and declared exactly.
    request.micro = 128;
    let undeclared = request
        .check(Mode::NewCorpusTransition, &parent(), None)
        .unwrap_err();
    assert!(
        undeclared.to_string().contains("undeclared"),
        "{undeclared}"
    );
    let declared = json!({"declared_changes":{
        "execution_partition":training_partition(8, 128, 2048).unwrap()}});
    let record = request
        .check(Mode::NewCorpusTransition, &parent(), Some(&declared))
        .unwrap();
    assert_eq!(record["execution_partition"]["continuity"], "changed");
    assert_eq!(
        record["execution_partition"]["declared"],
        declared["declared_changes"]["execution_partition"]
    );
    let wrong = json!({"declared_changes":{
        "execution_partition":training_partition(8, 64, 2048).unwrap()}});
    assert!(request
        .check(Mode::NewCorpusTransition, &parent(), Some(&wrong))
        .is_err());
    let other_field = json!({"declared_changes":{"objective_sha256":"c".repeat(64)}});
    assert!(Request::same()
        .check(Mode::NewCorpusTransition, &parent(), Some(&other_field))
        .is_err());
    // Objective, vocabulary and architecture changes are never carried.
    assert!(Request::same()
        .check(
            Mode::NewCorpusTransition,
            &parent_with_changed_objective(),
            None
        )
        .is_err());
    let mut other_vocabulary = Request::same();
    other_vocabulary.vocabulary = NEW_CORPUS;
    assert!(other_vocabulary
        .check(Mode::NewCorpusTransition, &parent(), None)
        .is_err());
    // A parent without an execution partition: recorded unknown.
    let mut state = parent();
    state.as_object_mut().unwrap().remove("execution_partition");
    let record = Request::same()
        .check(Mode::NewCorpusTransition, &state, None)
        .unwrap();
    assert_eq!(record["execution_partition"]["continuity"], "unknown");
    assert_eq!(record["objective"]["continuity"], "identical");
    // Without its objective identity the same parent never carries its
    // optimizer (owner decision 2026-09-26, OWNER-DECISIONS.md D12).
    legacy(&mut state);
    let error = Request::same()
        .check(Mode::NewCorpusTransition, &state, None)
        .unwrap_err();
    assert!(error.to_string().contains(LEGACY_REFUSAL), "{error}");
}

#[test]
fn a_weight_warm_start_resets_the_optimizer_and_declares_nothing() {
    let mut request = Request::same();
    request.corpus = NEW_CORPUS;
    request.micro = 128;
    let record = request
        .check(
            Mode::WeightWarmStart,
            &parent_with_changed_objective(),
            None,
        )
        .unwrap();
    assert_eq!(record["optimizer"], "reset");
    assert_eq!(record["sampler"], "new_window_from_zero");
    assert_eq!(record["exact_resume_claim"], false);
    assert_eq!(record["objective"]["continuity"], "changed");
    assert_eq!(record["execution_partition"]["continuity"], "changed");
    let declared = json!({"declared_changes":{
        "execution_partition":training_partition(8, 128, 2048).unwrap()}});
    assert!(request
        .check(Mode::WeightWarmStart, &parent(), Some(&declared))
        .is_err());
    let mut other_vocabulary = Request::same();
    other_vocabulary.vocabulary = NEW_CORPUS;
    assert!(other_vocabulary
        .check(Mode::WeightWarmStart, &parent(), None)
        .is_err());
    assert!(Request::same()
        .check(Mode::ColdInitialization, &parent(), None)
        .is_err());
}

#[test]
fn a_legacy_parent_continues_only_as_a_weight_warm_start() {
    let mut state = parent();
    legacy(&mut state);
    let before = state.clone();
    let mut new_corpus = Request::same();
    new_corpus.corpus = NEW_CORPUS;
    let declared = json!({"declared_changes":{
        "execution_partition":training_partition(8, 256, 2048).unwrap()}});
    for (mode, request, policy) in [
        (Mode::ExactResume, Request::same(), None),
        (Mode::NewCorpusTransition, Request::same(), None),
        (Mode::NewCorpusTransition, new_corpus, None),
        // A declared partition does not buy a legacy parent an optimizer carry.
        (Mode::NewCorpusTransition, Request::same(), Some(&declared)),
    ] {
        let error = request.check(mode, &state, policy).unwrap_err();
        assert!(
            error.to_string().contains(LEGACY_REFUSAL),
            "{mode:?}: {error}"
        );
        assert_eq!(state, before, "a refusal leaves the parent untouched");
    }
    // Weight warm start: the optimizer is reset, the sampler starts a new
    // window, and the objective is recorded as unverified, never identical.
    let record = Request::same()
        .check(Mode::WeightWarmStart, &state, None)
        .unwrap();
    assert_eq!(record["mode"], "weight_warm_start");
    assert_eq!(record["optimizer"], "reset");
    assert_eq!(record["sampler"], "new_window_from_zero");
    assert_eq!(record["exact_resume_claim"], false);
    assert_eq!(record["objective"]["continuity"], "legacy_ids_only");
    assert!(record["objective"]["parent_sha256"].is_null());
    // Positive control: the same parent with its objective identity still
    // resumes exactly and carries its optimizer.
    for mode in [Mode::ExactResume, Mode::NewCorpusTransition] {
        let record = Request::same().check(mode, &parent(), None).unwrap();
        assert_eq!(record["objective"]["continuity"], "identical");
    }
    // A legacy parent whose legacy identifiers differ is a changed objective:
    // refused for resume and carry with the existing wording, recorded
    // `changed` by a warm start.
    let mut other = state.clone();
    other["runtime_objective"] = json!("huber_seconds_v1");
    for mode in [Mode::ExactResume, Mode::NewCorpusTransition] {
        let error = Request::same().check(mode, &other, None).unwrap_err();
        assert!(error.to_string().contains("same objective"), "{error}");
    }
    let record = Request::same()
        .check(Mode::WeightWarmStart, &other, None)
        .unwrap();
    assert_eq!(record["objective"]["continuity"], "changed");
}

#[test]
fn the_accepted_step_60000_checkpoint_continues_as_a_weight_warm_start() {
    assert_eq!(
        hash(ACCEPTED_STEP_60000_STATE),
        ACCEPTED_STEP_60000_STATE_SHA256
    );
    let state: Value = serde_json::from_slice(ACCEPTED_STEP_60000_STATE).unwrap();
    assert_eq!(state["step"], 60000);
    assert!(
        state.get("objective").is_none() && state.get("objective_sha256").is_none(),
        "the accepted checkpoint predates the objective identity"
    );
    assert_eq!(state["ranking_backward"], "dense_pairwise_v2");
    assert_eq!(state["runtime_objective"], "huber_log1p_seconds_v2");
    // Its config.json is Config::regression(401) in the file layout.
    let config = config();
    assert_eq!(
        hash(&serde_json::to_vec_pretty(&Config::regression(401)).unwrap()),
        state["config_sha256"]
    );
    let objective = objective::identity_sha256();
    let vocabulary = state["vocabulary_sha256"].as_str().unwrap();
    let sampler = state["sampler_id"].as_str().unwrap();
    let seed = state["seed"].as_u64().unwrap();
    let current = |corpus| Current {
        corpus_sha256: corpus,
        vocabulary_sha256: vocabulary,
        world: 8,
        microbatch: 256,
        global_batch: 2048,
        seed,
        sampler_id: sampler,
        config: &config,
        objective_sha256: &objective,
    };
    let own_corpus = state["corpus_sha256"].as_str().unwrap();
    // The native loop's next block: parent weights with a carried optimizer
    // on a new corpus. Refused before CUDA.
    let carry = json!({"optimizer":"carry"});
    let mode = classify(false, true, Some(&carry)).unwrap();
    assert_eq!(mode, Mode::NewCorpusTransition);
    let error =
        check_parent(mode, &state, &config, &current(NEW_CORPUS), Some(&carry)).unwrap_err();
    assert!(error.to_string().contains(LEGACY_REFUSAL), "{error}");
    // Exact resume of the accepted checkpoint: refused.
    let error = check_parent(
        Mode::ExactResume,
        &state,
        &config,
        &current(own_corpus),
        None,
    )
    .unwrap_err();
    assert!(error.to_string().contains(LEGACY_REFUSAL), "{error}");
    // The same parent with the optimizer reset: a weight warm start that
    // continues from step 60000 with nothing carried.
    let reset = json!({"optimizer":"reset"});
    let mode = classify(false, true, Some(&reset)).unwrap();
    assert_eq!(mode, Mode::WeightWarmStart);
    let record = check_parent(mode, &state, &config, &current(NEW_CORPUS), Some(&reset)).unwrap();
    assert_eq!(record["mode"], "weight_warm_start");
    assert_eq!(record["optimizer"], "reset");
    assert_eq!(record["sampler"], "new_window_from_zero");
    assert_eq!(record["exact_resume_claim"], false);
    assert_eq!(record["parent_step"], 60000);
    assert_eq!(record["lineage_mode"], "unknown_legacy");
    assert_eq!(record["objective"]["continuity"], "legacy_ids_only");
    assert!(record["objective"]["parent_sha256"].is_null());
    assert_eq!(record["execution_partition"]["continuity"], "unknown");
    assert_eq!(
        record["corpus"],
        json!({"parent":own_corpus,"current":NEW_CORPUS})
    );
}

#[test]
fn the_native_loop_v2_first_block_is_a_warm_start_the_trainer_admits() {
    use jope_parallel::native_loop::warm_start;
    let state: Value = serde_json::from_slice(ACCEPTED_STEP_60000_STATE).unwrap();
    let config = config();
    let objective = objective::identity_sha256();
    let current = Current {
        corpus_sha256: NEW_CORPUS,
        vocabulary_sha256: state["vocabulary_sha256"].as_str().unwrap(),
        world: 8,
        microbatch: 256,
        global_batch: 2048,
        seed: state["seed"].as_u64().unwrap(),
        sampler_id: state["sampler_id"].as_str().unwrap(),
        config: &config,
        objective_sha256: &objective,
    };
    let parent = json!({"path":"parent/state.json","sha256":ACCEPTED_STEP_60000_STATE_SHA256});
    let corpus = json!({"path":"corpus/index.json","sha256":NEW_CORPUS});
    for schema in [warm_start::LOOP_V1, warm_start::LOOP_V2] {
        let window = warm_start::window(schema, 60000, 0).unwrap();
        let parameters = warm_start::train_parameters(&parent, &corpus, &window).unwrap();
        let carry = json!({"schema":"parallel.replay_policy.v2","optimizer":"carry",
            "eligibility_evidence":{"original_window":parameters}});
        // The policy the loop freezes for this request (`materialize::training_eligibility`).
        let policy = match warm_start::request_mode(&parameters, &state).unwrap() {
            None => carry,
            Some(step) => {
                assert_eq!(step, 60000);
                warm_start::warm_start_policy(
                    carry,
                    warm_start::binding(ACCEPTED_STEP_60000_STATE_SHA256, &state).unwrap(),
                )
            }
        };
        let mode = classify(false, true, Some(&policy)).unwrap();
        let checked = check_parent(mode, &state, &config, &current, Some(&policy));
        if schema == warm_start::LOOP_V1 {
            assert_eq!(mode, Mode::NewCorpusTransition);
            let error = checked.unwrap_err();
            assert!(error.to_string().contains(LEGACY_REFUSAL), "{error}");
            continue;
        }
        assert_eq!(mode, Mode::WeightWarmStart);
        let record = checked.unwrap();
        assert_eq!(record["optimizer"], "reset");
        assert_eq!(record["objective"]["continuity"], "legacy_ids_only");
        assert_eq!(record["parent_step"], 60000);
        let bound = warm_start::policy_mode(&policy, ACCEPTED_STEP_60000_STATE_SHA256, &state)
            .unwrap()
            .unwrap();
        assert_eq!(bound["parent_model_sha256"], state["model_sha256"]);
        assert_eq!(bound["optimizer_step_base"], 0);
        assert_eq!((window.initial, window.target), (0, 3000));
    }
}

#[test]
fn a_transition_lineage_never_inherits_an_exact_resume_claim() {
    // A checkpoint written inside a carry transition names that lineage mode;
    // resuming it exactly is an exact resume of the transition run, and the
    // claim applies to that attempt only.
    let mut state = parent();
    state["initialization"] = json!({"kind":"warm_start",
        "continuation_mode":"new_corpus_transition","exact_same_experiment_resume":false});
    let record = Request::same()
        .check(Mode::ExactResume, &state, None)
        .unwrap();
    assert_eq!(record["lineage_mode"], "new_corpus_transition");
    assert_eq!(record["mode"], "exact_resume");
    // A parent without a recorded lineage mode stays unknown, not inferred.
    let mut old = parent();
    old["initialization"] = json!({"kind":"fresh"});
    let record = Request::same()
        .check(Mode::NewCorpusTransition, &old, None)
        .unwrap();
    assert_eq!(record["lineage_mode"], "unknown_legacy");
    assert_eq!(record["exact_resume_claim"], false);
}
