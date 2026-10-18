use feat_math::{evolve::FaultPolicy, kernel::profile::shipped_kernel2_profiles};
use jope_bench::methods::{
    shared_strategy, validate_shared_strategy_evidence, SearchConfig, SEARCH_METHOD_ID,
    SHARED_STRATEGY_METHOD_ID,
};
use jope_pipeline::{
    kernel2::search::{Options, Refinement},
    prepare::Cut,
    search::SearchOptions,
};

const SOURCE_COMMIT: &str = "85d7304a11462bb782607cbc780108408a930777";

fn fixture() -> (Cut, Options) {
    let cut = Cut {
        index: 0,
        train_x: vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]],
        train_y: vec![0.0, 1.0, 2.0, 3.0],
        hold_x: vec![vec![4.0], vec![5.0]],
        hold_y: vec![4.0, 5.0],
        hold_indices: vec![4, 5],
    };
    let profile = shipped_kernel2_profiles()
        .into_iter()
        .find(|p| p.id == "kernel2-baseline")
        .unwrap();
    let options = Options::new(
        SearchOptions::new(
            jope_pipeline::search::Budget {
                evaluations: 4,
                seconds: 30,
            },
            73,
        ),
        profile.id,
        FaultPolicy::RejectHard,
        Refinement {
            top_k: 0,
            trials_per_program: 0,
        },
    )
    .unwrap();
    (cut, options)
}

#[test]
fn actual_bench_entrypoint_and_product_route_share_exact_trace() {
    let (cut, options) = fixture();
    let bench = shared_strategy(&cut, &options, SOURCE_COMMIT, 4).unwrap();
    let product =
        jope_pipeline::kernel2::search::run_cut_shared_strategy(&cut, &options, SOURCE_COMMIT, 4)
            .unwrap();
    assert_eq!(bench.method.id, SHARED_STRATEGY_METHOD_ID);
    assert_ne!(bench.method.id, SEARCH_METHOD_ID);
    assert_eq!(bench.product_run.run.trace, product.run.trace);
    assert_eq!(bench.product_run.run.identity, product.run.identity);
    assert_eq!(bench.product_run.run.outcomes, product.run.outcomes);
    validate_shared_strategy_evidence(
        &bench.method.id,
        &bench.product_run,
        SOURCE_COMMIT,
        &bench.product_run.run.identity.profile_sha256,
        &bench.product_run.run.identity.evaluator_version,
    )
    .unwrap();
}

#[test]
fn historical_method_ids_cannot_be_relabelled_as_shared_evidence() {
    let (cut, options) = fixture();
    let run =
        jope_pipeline::kernel2::search::run_cut_shared_strategy(&cut, &options, SOURCE_COMMIT, 3)
            .unwrap();
    assert!(validate_shared_strategy_evidence(
        SEARCH_METHOD_ID,
        &run,
        SOURCE_COMMIT,
        &run.run.identity.profile_sha256,
        &run.run.identity.evaluator_version,
    )
    .is_err());
    assert!(validate_shared_strategy_evidence(
        SHARED_STRATEGY_METHOD_ID,
        &run,
        SOURCE_COMMIT,
        "stale-profile-sha256",
        &run.run.identity.evaluator_version,
    )
    .is_err());
    assert!(validate_shared_strategy_evidence(
        SHARED_STRATEGY_METHOD_ID,
        &run,
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        &run.run.identity.profile_sha256,
        &run.run.identity.evaluator_version,
    )
    .is_err());
}

#[test]
fn benchmark_config_identity_does_not_change_historical_search_config() {
    let old = SearchConfig {
        pop: 4,
        gens: 1,
        seed: 73,
        seconds: 0.0,
    };
    assert_eq!(old.pop, 4);
    assert_eq!(SEARCH_METHOD_ID, "afm.lite-race.v1");
    assert_eq!(SHARED_STRATEGY_METHOD_ID, "afm.shared-strategy.v1");
}
