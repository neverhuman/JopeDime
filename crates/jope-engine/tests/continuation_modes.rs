#![cfg(feature = "neural")]
//! W6-T06: the four continuation modes and the full objective identity (CPU,
//! no model). These are the pre-CUDA guards `training::rank_impl` applies; the
//! GPU path itself is not executed here.

use jope_parallel::{
    manifest::hash,
    neural::{self, Config},
    task_contract::{training_partition, verify_resume_metadata},
    training::{
        continuation::{check_parent, classify, Current, Mode},
        objective,
    },
};
use serde_json::{json, Value};

const CORPUS: &str = "1111111111111111111111111111111111111111111111111111111111111111";
const NEW_CORPUS: &str = "2222222222222222222222222222222222222222222222222222222222222222";
const VOCABULARY: &str = "3333333333333333333333333333333333333333333333333333333333333333";
const SAMPLER: &str = "family_uniform_champion_plus_seven_quantiles_v2";
/// The refusal a parent without an objective identity gets for exact resume and optimizer carry.
const LEGACY_REFUSAL: &str = "continues only as a weight warm start";
/// The accepted step-60000 `state.json`, byte for byte, as retained in the ASM manual's training
/// history. Its SHA-256 is the accepted state that `models/bundles/policy-step-60000.json` names.
const ACCEPTED_STEP_60000_STATE: &[u8] = include_bytes!(
    "../../../docs/internal/asm-manual/data/training-history/accepted60000-state.json"
);
const ACCEPTED_STEP_60000_STATE_SHA256: &str =
    "8ac80c269cc47c253b6f53fa3bc1eb75bb2fcda0bdace3ff8554be0175c400d6";

fn config() -> Value {
    serde_json::to_value(Config::regression(401)).unwrap()
}

fn parent() -> Value {
    json!({"schema":"parallel.checkpoint.v1","step":3000,"world":8,"global_batch":2048,
        "corpus_sha256":CORPUS,"vocabulary_sha256":VOCABULARY,"seed":7,"sampler_id":SAMPLER,
        "ranking_backward":"dense_pairwise_v2","runtime_objective":"huber_log1p_seconds_v2",
        "execution_partition":training_partition(8, 256, 2048).unwrap(),
        "objective":objective::identity(),"objective_sha256":objective::identity_sha256(),
        "initialization":{"kind":"fresh","continuation_mode":"cold_initialization"},
        "optimizer_sha256":"a".repeat(64)})
}

/// The same parent trained under a different objective (one coefficient),
/// with its own internally consistent identity and the same legacy IDs.
fn parent_with_changed_objective() -> Value {
    let mut state = parent();
    state["objective"]["coefficients"]["ranking_bt"] = json!(0.5);
    state["objective_sha256"] = json!(objective::identity_digest(&state["objective"]));
    state
}

fn legacy(state: &mut Value) {
    let fields = state.as_object_mut().unwrap();
    fields.remove("objective");
    fields.remove("objective_sha256");
}

struct Request {
    corpus: &'static str,
    vocabulary: &'static str,
    world: usize,
    micro: usize,
    seed: u64,
    sampler: &'static str,
    config: Value,
    objective: String,
}

impl Request {
    fn same() -> Self {
        Self {
            corpus: CORPUS,
            vocabulary: VOCABULARY,
            world: 8,
            micro: 256,
            seed: 7,
            sampler: SAMPLER,
            config: config(),
            objective: objective::identity_sha256(),
        }
    }
    fn current(&self) -> Current<'_> {
        Current {
            corpus_sha256: self.corpus,
            vocabulary_sha256: self.vocabulary,
            world: self.world,
            microbatch: self.micro,
            global_batch: 2048,
            seed: self.seed,
            sampler_id: self.sampler,
            config: &self.config,
            objective_sha256: &self.objective,
        }
    }
    fn check(&self, mode: Mode, state: &Value, policy: Option<&Value>) -> anyhow::Result<Value> {
        check_parent(mode, state, &config(), &self.current(), policy)
    }
}

#[test]
fn the_objective_identity_restates_every_declared_literal() {
    // Restated independently: any change to a coefficient, smoothing, weight,
    // temperature, boost, reduction or rule changes this value and its digest.
    let expected = json!({
        "schema": "jope.training_objective.v1",
        "coefficients": {"sequence_nll": 1.0, "ranking_bt": 0.25, "ranking_listwise": 0.10,
            "runtime_huber": 0.10, "contextfree_bt": 0.05, "stability": 0.0, "loss_log_vars": 0.0},
        "sequence": {
            "targets": "tokens_1_to_T_minus_1_bos_never_a_target",
            "legal_mask": "grammar_legal_set_neg_infinity",
            "label_smoothing": 0.05,
            "smoothing_support": "grammar_legal_set",
            "row_reduction": "mean_over_supervised_positions",
            "row_weight": {"floor": 0.05f32, "percentile_exponent": 4i32},
            "batch_reduction": "row_weighted_mean",
            "row_without_supervision": "refused"},
        "ranking": {
            "backward": "dense_pairwise_v2",
            "score": "mean_of_reward_heads",
            "reward_heads": 4i64,
            "group": "context_and_comparison_of_8",
            "pairs": "all_unordered_within_group",
            "pair_weight": "absolute_value_difference",
            "boost_threshold": 0.9f32,
            "boost_factor": 2.0,
            "ties": "skipped",
            "pair_reduction": "weighted_mean_over_microbatch",
            "empty_support": "connected_zero",
            "listwise_temperature": 0.25,
            "listwise_min_members": 3usize,
            "listwise_reduction": "mean_over_eligible_groups"},
        "runtime": {"objective": "huber_log1p_seconds_v2", "huber_beta": 1.0,
            "reduction": "mean_over_rows"},
        "ablation": {"included": true, "construction": "ranking_bt_on_context_free_head"},
        "inactive": ["stability", "loss_log_vars"],
        "reward_head_dropout": 0.05,
        "supervision": "validate_rows_v1",
        "nonfinite_objective": "refused",
        "normalization": "microbatch_weighted_means_rank_mean_v1"
    });
    assert_eq!(objective::identity(), expected);
    assert_eq!(
        objective::identity_sha256(),
        hash(&serde_json::to_vec(&expected).unwrap())
    );
    assert_eq!(
        (
            neural::LABEL_SMOOTHING,
            neural::LISTWISE_TEMPERATURE,
            neural::LISTWISE_MIN_MEMBERS,
            neural::BOOST_THRESHOLD,
            neural::BOOST_FACTOR,
            neural::REWARD_HEAD_DROPOUT
        ),
        (0.05, 0.25, 3, 0.9f32, 2.0, 0.05)
    );
    // The partition dependence is part of the identity's declared normalization
    // and is bound separately by the execution partition.
    assert_eq!(
        training_partition(8, 256, 2048).unwrap()["normalization"],
        expected["normalization"]
    );
}

#[test]
fn every_request_is_exactly_one_mode_and_ambiguity_is_refused() {
    let reset = json!({"optimizer":"reset"});
    let carry = json!({"optimizer":"carry"});
    for (resume, parent, policy, mode) in [
        (false, false, None, Mode::ColdInitialization),
        (false, true, None, Mode::WeightWarmStart),
        (false, true, Some(&reset), Mode::WeightWarmStart),
        (false, true, Some(&carry), Mode::NewCorpusTransition),
        (true, false, None, Mode::ExactResume),
        (true, false, Some(&carry), Mode::ExactResume),
    ] {
        assert_eq!(classify(resume, parent, policy).unwrap(), mode);
    }
    let unknown = json!({"optimizer":"partial"});
    let missing = json!({"replay":"broad"});
    for (resume, parent, policy) in [
        (true, true, None),
        (false, false, Some(&carry)),
        (false, true, Some(&unknown)),
        (false, true, Some(&missing)),
    ] {
        assert!(classify(resume, parent, policy).is_err());
    }
}

#[test]
fn the_two_identifier_guard_accepted_an_objective_change_that_the_full_identity_refuses() {
    // G1 reproduction: carry and resume compared only `ranking_backward` and
    // `runtime_objective`. A parent trained with a different coefficient
    // passes that guard unchanged.
    let changed = parent_with_changed_objective();
    let optimizer = json!({"step":3000,"sha256":"a".repeat(64)});
    let two_ids = json!({"ranking_backward":"dense_pairwise_v2","runtime_objective":"huber_log1p_seconds_v2"});
    verify_resume_metadata(&changed, &two_ids, &optimizer)
        .expect("the legacy guard cannot see the objective change");
    // The full identity refuses the same parent for carry and exact resume.
    let request = Request::same();
    for mode in [Mode::NewCorpusTransition, Mode::ExactResume] {
        let error = request.check(mode, &changed, None).unwrap_err();
        assert!(
            error.to_string().contains("same objective"),
            "{mode:?}: {error}"
        );
    }
    // A weight warm start may change the objective, and says so.
    let record = request
        .check(Mode::WeightWarmStart, &changed, None)
        .unwrap();
    assert_eq!(record["objective"]["continuity"], "changed");
    assert_eq!(record["optimizer"], "reset");
}

#[test]
fn exact_resume_refuses_every_undeclared_change() {
    let record = Request::same()
        .check(Mode::ExactResume, &parent(), None)
        .unwrap();
    assert_eq!(record["mode"], "exact_resume");
    assert_eq!(record["exact_resume_claim"], true);
    assert_eq!(record["objective"]["continuity"], "identical");
    assert_eq!(record["execution_partition"]["continuity"], "identical");
    assert_eq!(record["sampler"], "continued");
    type Change = Box<dyn Fn(&mut Request, &mut Value, &mut Option<Value>)>;
    let cases: Vec<(&str, Change)> = vec![
        (
            "objective",
            Box::new(|_, s, _| *s = parent_with_changed_objective()),
        ),
        (
            "objective digest",
            Box::new(|_, s, _| s["objective_sha256"] = json!("b".repeat(64))),
        ),
        ("microbatch", Box::new(|r, _, _| r.micro = 128)),
        ("world", Box::new(|r, _, _| r.world = 4)),
        ("corpus", Box::new(|r, _, _| r.corpus = NEW_CORPUS)),
        ("vocabulary", Box::new(|r, _, _| r.vocabulary = NEW_CORPUS)),
        ("seed", Box::new(|r, _, _| r.seed = 8)),
        ("sampler", Box::new(|r, _, _| r.sampler = "other_sampler")),
        (
            "architecture",
            Box::new(|r, _, _| {
                let mut c = Config::regression(401);
                c.layers = 5;
                r.config = serde_json::to_value(c).unwrap();
            }),
        ),
        (
            "declared change",
            Box::new(|_, _, p| {
                *p = Some(json!({"declared_changes":{
                    "execution_partition":training_partition(8, 256, 2048).unwrap()}}))
            }),
        ),
        (
            "partition absent",
            Box::new(|_, s, _| drop(s.as_object_mut().unwrap().remove("execution_partition"))),
        ),
        (
            "legacy with other runtime objective",
            Box::new(|_, s, _| {
                legacy(s);
                s["runtime_objective"] = json!("huber_seconds_v1");
            }),
        ),
    ];
    for (case, change) in cases {
        let mut request = Request::same();
        let mut state = parent();
        let mut policy = None;
        change(&mut request, &mut state, &mut policy);
        assert!(
            request
                .check(Mode::ExactResume, &state, policy.as_ref())
                .is_err(),
            "exact resume accepted a {case} change"
        );
    }
    // A legacy parent (no objective identity) is refused even when both legacy
    // IDs match: it continues only as a weight warm start (owner decision
    // 2026-09-26, OWNER-DECISIONS.md D12).
    let mut state = parent();
    legacy(&mut state);
    let error = Request::same()
        .check(Mode::ExactResume, &state, None)
        .unwrap_err();
    assert!(error.to_string().contains(LEGACY_REFUSAL), "{error}");
}

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

/// Owner decision 2026-09-26 (`docs/internal/core-ip/work-orders/OWNER-DECISIONS.md`, D12): a
/// parent without an objective identity cannot prove its objective, so it is refused for exact
/// resume and optimizer carry even when both legacy identifiers match. It continues only as a
/// weight warm start (optimizer reset, nothing carried), recorded as `legacy_ids_only`.
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

/// The accepted step-60000 checkpoint is a legacy parent. The native loop's replay policy
/// (`optimizer: carry`) classifies to an optimizer-carry transition, which the pre-CUDA guard now
/// refuses; the same parent with the optimizer reset is a weight warm start and continues.
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

/// D12, native loop: the first training block from the accepted step-60000 checkpoint, as each
/// native-loop protocol freezes it. Under `jope.native_loop.v1` it is an optimizer carry, and the
/// pre-CUDA guard still refuses it. Under `jope.native_loop.v2` it is a weight warm start that
/// binds the parent weights digest and that the guard admits with the optimizer reset.
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
