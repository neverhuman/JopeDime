//! Reproducibility and gate regressions for the separate TRAIN-only utility run.
#[path = "../../examples/scalar_utility_v2/metrics.rs"]
mod utility_metrics;
#[path = "../../examples/scalar_utility_v2/worker.rs"]
mod worker;
use feat_math::evolve::FaultPolicy;
use feat_math::kernel::scalar_program::{ScalarCarrier, ScalarProducerKind, ScalarProducerProfile};
use feat_math::Rows;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::Path;

fn cohort() -> (Rows, Vec<f64>) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scalar_v2/utility");
    let manifest = std::fs::read(root.join("manifest.json")).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&manifest)),
        "20e247031fb140ebf1195374564be2c94d9afa4d7b7bd1c33e3635c969cbdd77"
    );
    let manifest: Value = serde_json::from_slice(&manifest).unwrap();
    assert_eq!(manifest["datasets"].as_array().unwrap().len(), 8);
    let mut first = None;
    for item in manifest["datasets"].as_array().unwrap() {
        let bytes = std::fs::read(root.join(item["path"].as_str().unwrap())).unwrap();
        assert_eq!(format!("{:x}", Sha256::digest(&bytes)), item["sha256"]);
        let data: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(data["labels_unchanged"], true);
        assert_eq!(data["rows_bits"].as_array().unwrap().len(), 96);
        assert_eq!(data["labels_bits"].as_array().unwrap().len(), 96);
        if first.is_none() {
            let bits = |v: &Value| {
                v.as_array()
                    .unwrap()
                    .iter()
                    .map(|x| f64::from_bits(x.as_u64().unwrap()))
                    .collect::<Vec<_>>()
            };
            first = Some((
                data["rows_bits"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(bits)
                    .collect(),
                bits(&data["labels_bits"]),
            ));
        }
    }
    first.unwrap()
}

#[test]
fn utility_replays_exact_work_and_keeps_reporting_outside_selection() {
    let (rows, labels) = cohort();
    let train = rows[..64].to_vec();
    let profile = ScalarProducerProfile::new(
        ScalarProducerKind::InverseLinks,
        ScalarCarrier::Row,
        rows[0].len(),
    )
    .unwrap();
    let policy = FaultPolicy::Penalize {
        lambda_domain: 0.05,
        lambda_saturation: 0.0,
        tie_break: true,
    };
    let a = worker::search(&profile, &train, &labels[..64], 3, 72, policy).unwrap();
    let state = a.model.to_json().unwrap();
    // Neither reporting rows nor reporting labels is an argument of search.
    let poison = vec![vec![f64::NAN; rows[0].len()]; 32];
    let _ = a.model.predict(&poison).unwrap();
    assert_eq!(state, a.model.to_json().unwrap());
    let b = worker::search(&profile, &train, &labels[..64], 3, 72, policy).unwrap();
    assert_eq!(a.deterministic, b.deterministic);
    assert_eq!(state, b.model.to_json().unwrap());
    let data = &a.deterministic;
    assert_eq!(data["attempted"], 72);
    assert_eq!(data["attempts"].as_array().unwrap().len(), 72);
    assert_eq!(
        data["completed"].as_u64().unwrap() + data["refused"].as_u64().unwrap(),
        72
    );
    assert_eq!(data["cell_accounting_complete"], true);
    assert_eq!(data["unobserved_internal_passes"], 0);
    assert_eq!(
        data["training_only_fold_key"],
        worker::training_fold_key(&train, &labels[..64])
    );
    let (models, controls) =
        worker::compare_existing_formulas(0, &profile, &train, &labels[..64], policy).unwrap();
    assert!(!models.is_empty());
    assert_eq!(
        controls["training_only_fold_key"],
        data["training_only_fold_key"]
    );
    assert_eq!(controls["producer_profile_unchanged"], true);
    assert_eq!(
        controls["controls_do_not_expand_generation_catalogues"],
        true
    );
    assert_eq!(controls["cell_accounting_complete"], true);
    assert_eq!(data["profile_sha256"], profile.sha256());
}

fn measurements() -> Vec<utility_metrics::Run> {
    let mut runs = vec![];
    for card in 0..3 {
        for dataset in 0..8 {
            for enabled in [false, true] {
                for seed in [3, 17, 29, 101, 701] {
                    runs.push(utility_metrics::Run {dataset:format!("dataset-{dataset}"),card,enabled,seed,wall:7.2,
            search:json!({"budget":72,"attempted":72,"completed":72,"refused":0,"seed":seed,
                "best_cv_bits":0.5_f64.to_bits(),"top_ten_median_bits":0.4_f64.to_bits(),
                "evals_to_95_final_best":36,"all_pass_faults":{"cells":100,"domain":0},
                "hard_rejected_attempts":0,"invalid_offspring_proposals":0,
                "cell_accounting_complete":true,"unobserved_internal_passes":0,
                "elite":[{"nodes":2,"depth":2,"card_token_occurrences":[0,0,0]}]}),
            receipt:json!({"fitted_source_bytes":100,"fitted_codec_bytes":10,"recipes":[
                {"language":"Rust","code_tokens":100,"doc_tokens":10,"bytes":1000},
                {"language":"Python","code_tokens":100,"doc_tokens":10,"bytes":1000},
                {"language":"Cpp17","code_tokens":100,"doc_tokens":10,"bytes":1000}]})});
                }
            }
        }
    }
    runs
}

#[test]
fn utility_gates_preserve_zero_noise_ties_missing_thresholds_and_missing_parity() {
    let mut runs = measurements();
    let result = utility_metrics::summarize(&runs).unwrap();
    for card in result["cards"].as_array().unwrap() {
        assert_eq!(card["gates"]["G-A"], false); // equal zero-noise ties are not two improvements
        assert_eq!(card["gates"]["G-C"], true);
        assert_eq!(card["gates"]["G-D"], true);
        assert_eq!(card["gates"]["G-E"], false);
        assert_eq!(card["decision"], "decline");
        assert_eq!(
            card["gates"]["G-B"],
            "pending compiled L1–L4 and fitted proof"
        );
    }
    for run in &mut runs {
        if run.enabled {
            run.search["evals_to_95_final_best"] = json!(30);
            run.search["elite"][0]["card_token_occurrences"][run.card] = json!(1);
        }
    }
    let result = utility_metrics::summarize(&runs).unwrap();
    assert_eq!(result["cards"][0]["gates"]["G-A"], true); // literal M3 speed path
    assert_eq!(result["cards"][0]["decision"], "keep-experimental"); // parity cannot be inferred
    runs[1].search["evals_to_95_final_best"] = Value::Null;
    let result = utility_metrics::summarize(&runs).unwrap();
    assert_eq!(result["cards"][0]["M3_complete"], false);
    assert_eq!(result["cards"][0]["gates"]["G-A"], false);
    // A single inferior dataset cannot be hidden by improvements elsewhere.
    for run in &mut runs {
        if run.card == 0 && run.enabled {
            run.search["best_cv_bits"] = json!(if run.dataset == "dataset-0" {
                0.49_f64.to_bits()
            } else {
                0.6_f64.to_bits()
            });
        }
    }
    assert_eq!(
        utility_metrics::summarize(&runs).unwrap()["cards"][0]["gates"]["G-A"],
        false
    );
    runs.pop();
    assert!(utility_metrics::summarize(&runs).is_err()); // no partial cohort qualification
}

#[test]
fn utility_soundness_and_cost_use_actual_work_and_best_artifacts() {
    let mut runs = measurements();
    for run in &mut runs {
        if run.enabled {
            run.search["all_pass_faults"]["domain"] = json!(2);
        }
    }
    assert_eq!(
        utility_metrics::summarize(&runs).unwrap()["cards"][0]["gates"]["G-C"],
        true
    );
    for run in &mut runs {
        if run.enabled {
            run.search["all_pass_faults"]["domain"] = json!(3);
            run.receipt["recipes"][0]["code_tokens"] = json!(120);
        }
    }
    let result = utility_metrics::summarize(&runs).unwrap();
    assert_eq!(result["cards"][0]["gates"]["G-C"], false);
    assert_eq!(result["cards"][0]["gates"]["G-D"], false);
    for run in &mut runs {
        if run.enabled {
            run.search["all_pass_faults"]["domain"] = json!(0);
            run.search["unobserved_internal_passes"] = json!(1);
            run.search["cell_accounting_complete"] = json!(false);
        }
    }
    assert_eq!(
        utility_metrics::summarize(&runs).unwrap()["cards"][0]["gates"]["G-C"],
        false
    );
}

#[test]
fn utility_quality_never_counts_fault_penalty_reduction_as_predictive_lift() {
    for (raw_b, expected_quality) in [(0.60_f64, false), (0.59, false), (0.61, true)] {
        let mut runs = measurements();
        for run in &mut runs {
            let raw = if run.enabled { raw_b } else { 0.60 };
            let objective = if run.enabled { raw_b } else { 0.58_f64 };
            run.search["best_cv_bits"] = json!(raw.to_bits());
            run.search["top_ten_median_bits"] = json!(raw.to_bits());
            run.search["best_objective_bits"] = json!(objective.to_bits());
            run.search["all_pass_faults"]["domain"] = json!(if run.enabled { 0 } else { 40 });
        }
        for card in utility_metrics::summarize(&runs).unwrap()["cards"]
            .as_array()
            .unwrap()
        {
            assert_eq!(card["gates"]["G-A"], expected_quality);
            for dataset in card["datasets"].as_array().unwrap() {
                assert_eq!(
                    dataset["A"]["M1_best_cv_mean"],
                    [0.60_f64; 5].iter().sum::<f64>() / 5.0
                );
                assert_eq!(
                    dataset["B"]["M1_best_cv_mean"],
                    [raw_b; 5].iter().sum::<f64>() / 5.0
                );
                assert_eq!(
                    dataset["A"]["M2_top10_cv_mean"],
                    [0.60_f64; 5].iter().sum::<f64>() / 5.0
                );
                assert_eq!(
                    dataset["B"]["M2_top10_cv_mean"],
                    [raw_b; 5].iter().sum::<f64>() / 5.0
                );
            }
        }
    }
}
