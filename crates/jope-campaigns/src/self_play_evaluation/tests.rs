use super::helpers::choose_deployed_candidate;
use super::*;
use serde_json::json;
use std::collections::BTreeMap;
use std::path::Path;

#[test]
fn robust_score_clips_and_uses_population_fold_spread() {
    assert_eq!(robust_score(Some(-2.0), Some(&[0.0, 2.0])), Some(-1.25));
    assert_eq!(
        robust_score(Some(0.5), Some(&[1.0, 1.0, f64::NAN])),
        Some(0.5)
    );
    assert_eq!(robust_score(Some(f64::NAN), Some(&[1.0, 2.0])), None);
}

#[test]
fn selection_uses_robust_then_oof_and_keeps_first_full_tie() {
    let rows = vec![
        json!({"score_robust":0.7,"_oof":0.8,"rank":2}),
        json!({"score_robust":0.7,"_oof":0.9,"rank":1}),
        json!({"score_robust":0.7,"_oof":0.9,"rank":0}),
    ];
    assert_eq!(choose_deployed_candidate(&rows), Some(1));
    assert_eq!(
        choose_deployed_candidate(&[json!({"score_robust":null})]),
        None
    );
}

#[test]
fn gate_counts_ties_half_and_updates_best_monotonically() {
    let mut state = GateState {
        last_shipped_deployed: Some(BTreeMap::from([("a".into(), 0.5), ("b".into(), 0.2)])),
        ..GateState::default()
    };
    let deployed = BTreeMap::from([("a".into(), 0.5), ("b".into(), 0.4)]);
    let summary = gate_generation(&mut state, 3, &deployed, 0.7, &BTreeMap::new());
    assert_eq!(summary.beat_rate_vs_prev, Some(0.75));
    assert!(summary.shipped);
    assert_eq!(summary.new_bests, 2);
    let worse = BTreeMap::from([("b".into(), 0.1)]);
    let _ = gate_generation(&mut state, 4, &worse, 0.5, &BTreeMap::new());
    assert_eq!(state.self_best["b"], 0.4);
}

#[test]
fn explicit_and_elo_history_baselines_keep_python_precedence() {
    let mut state = GateState::default();
    state
        .gen_top1
        .insert(2, BTreeMap::from([("d".into(), 0.6)]));
    state.elo_history = vec![json!({"gen":2,"shipped":true})];
    assert_eq!(
        last_shipped_deployed_scores(&state, 3),
        (BTreeMap::from([("d".into(), 0.6)]), Some(2))
    );
    state.last_shipped_deployed = Some(BTreeMap::from([("x".into(), 0.9)]));
    state.last_shipped_generation = Some(1);
    assert_eq!(
        last_shipped_deployed_scores(&state, 3),
        (BTreeMap::from([("x".into(), 0.9)]), Some(1))
    );
}

#[test]
fn benchmark_prefers_probe_and_reports_both_regrets() {
    let report = json!({"modes":{
        "top1":{"elo":100.0,"field_rank":3,"double_holdout_elo":90.0},
        "cv_top6":{"elo":110.0,"field_rank":2,"double_holdout_elo":105.0,"battles_won":7,"battles_top3":8},
        "cv_select":{"elo":101.0},"best_k":{"elo":120.0}
    }});
    let summary = benchmark_summary(&report).unwrap();
    assert_eq!(summary["elo_deployed"], 110.0);
    assert_eq!(summary["top1_regret_vs_best_k"], 20.0);
    assert_eq!(summary["probe_regret_vs_best_k"], 10.0);
}

#[test]
fn evaluation_updates_candidate_scores_and_writes_training_rows() {
    let root = tempfile::tempdir().unwrap();
    let request = EvaluationRequest {
        generation: 4,
        run_name: "self_play".into(),
        artifact_dir: root.path().join("artifacts"),
        run_dir: root.path().join("run"),
        candidate_pool_dir: Some(root.path().join("candidate-pools")),
        eval_inputs: vec![EvaluationInput {
            dataset_hash: "dataset-a".into(),
            rank: 0,
            engine: "featmath".into(),
            genome: "(pipe X (ridge))".into(),
            candidate_pool_id: Some("pool-a".into()),
        }],
        eval_results: vec![(
            0,
            json!({
                "cv":{"oof_r2":0.8,"fold_r2":[0.7,0.9]},
                "timings":{"total_ms":12.5},
                "cache_hit":true
            }),
        )],
        picks: vec![DatasetCandidates {
            dataset_hash: "dataset-a".into(),
            rows: vec![json!({
                "rank":0,"genome":"(pipe X (ridge))","engine":"featmath",
                "candidate_pool_id":"pool-a","reward":0.6
            })],
        }],
        datasets: BTreeMap::from([(
            "dataset-a".into(),
            DatasetContext {
                embedding: vec![0.25, 0.5],
                target_embedding: vec![0.0; 3],
                former_best_score: Some(0.4),
            },
        )]),
        candidate_pool_rows: vec![json!({"candidate_pool_id":"pool-a"})],
        score_pool: BTreeMap::from([("dataset-a".into(), vec![0.2])]),
    };
    let summary = evaluate_and_append(request).unwrap();
    assert_eq!(summary.rows, 1);
    assert_eq!(summary.deployed_scores["dataset-a"], 0.775);
    assert_eq!(summary.rows_out[0]["score"], 0.8);
    assert_eq!(summary.rows_out[0]["metadata"]["rank_pct"], 1.0);
    assert!(summary.candidate_pool_rows[0]["deployed"]
        .as_bool()
        .unwrap());
    assert!(summary.scored_events[0].cache_hit);
    assert!(Path::new(&summary.self_rows_path).exists());
    assert!(summary
        .candidate_pool_path
        .as_ref()
        .is_some_and(|path| Path::new(path).exists()));
}
