use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::os::unix::fs::PermissionsExt;

use feat_math::GenomeIr;
use jope_dime_regressor::jope_dlgp_best5::*;
use jope_dime_regressor::jope_generalized_gp::{
    raw_sha256, ArtifactStoreV1, GenomeV1, GpTaskV1, ProposalFreezeV1,
};

fn digest(label: &str) -> String {
    raw_sha256(label.as_bytes())
}

fn full_dataset_digest(id: &str) -> String {
    format!("{id}{}", &digest(id)[16..])
}

fn genome(index: usize) -> GenomeV1 {
    let head = match index % 4 {
        0 => format!("(ridge lambda_frac={})", (index + 1) as f64 / 100_000.0),
        1 => format!(
            "(elasticnet alpha={} l1_ratio=0.5)",
            (index + 1) as f64 / 100_000.0
        ),
        2 => format!("(krr m={})", 20 + index),
        _ => format!("(gpr kernel=linear m={} steps=10)", 20 + index),
    };
    let source = format!("(pipe X (standardize) {head})");
    let canonical = GenomeIr::parse(&source).unwrap().canonical();
    GenomeV1::from_canonical(GpTaskV1::Regression, &canonical).unwrap()
}

/// The real-corpus fixture reads the pinned SOTArena benchmark clone (ADR-0003
/// `SOTARENA_ROOT`). Absence is an unavailable input and remains a failure; it is
/// never replaced with synthetic rows under this test's name.
fn sotarena_fixture_root(dataset_id: &str) -> String {
    let root = jope_dime_regressor::paths::sotarena_root();
    let train = root.join(format!("data/tabular-regression-v1/{dataset_id}/train.csv"));
    assert!(
        train.is_file(),
        "unavailable: real-corpus input {} (set SOTARENA_ROOT to the pinned sotarena benchmark clone)",
        train.display()
    );
    root.display().to_string()
}

fn frozen_file(label: &str, sha256: String) -> FrozenFileV1 {
    FrozenFileV1 {
        path: format!("/frozen/{label}"),
        sha256,
        bytes: 1,
    }
}

fn campaign() -> CampaignFreezeV1 {
    let regression_heads = vec!["elasticnet", "gpr", "krr", "ridge"]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    let input: serde_json::Value =
        serde_json::from_str(include_str!("../manifests/input-manifest.v1.json")).unwrap();
    let mut datasets = input["datasets"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(index, row)| {
            let id = row["dataset_hash"].as_str().unwrap();
            FrozenDatasetV1 {
                dataset_id: id.into(),
                dataset_sha256: full_dataset_digest(id),
                train_relative_path: format!("data/{id}/train.csv"),
                test_relative_path: format!("data/{id}/test.csv"),
                metadata_relative_path: format!("splits/{id}/metadata.json"),
                train_sha256: row["train_sha256"].as_str().unwrap().into(),
                test_sha256: row["test_sha256"].as_str().unwrap().into(),
                metadata_sha256: row["meta_sha256"].as_str().unwrap().into(),
                train_rows: row["train_rows"].as_u64().unwrap() as u32,
                feature_count: row["features"].as_u64().unwrap() as u32,
                source_family: format!("family-{}", index % 7),
                stage: if index < 20 {
                    DatasetStageV1::Pilot
                } else {
                    DatasetStageV1::Confirmation
                },
                current_sotarena_win: index < 6,
            }
        })
        .collect::<Vec<_>>();
    // Freeze order is the input-manifest order, not a later score ordering.
    datasets.shrink_to_fit();
    CampaignFreezeV1 {
        schema: CampaignFreezeV1::SCHEMA.into(),
        campaign_id: CAMPAIGN_ID.into(),
        input_manifest_sha256: FROZEN_INPUT_MANIFEST_SHA256.into(),
        panel_sha256: FROZEN_PANEL_SHA256.into(),
        observation_snapshot_sha256: digest("observations"),
        generated_registry_sha256: digest("registry"),
        evaluator_sha256: digest("evaluator"),
        math_policy_sha256: digest("math"),
        backend_policy_sha256: digest("backend"),
        six_win_manifest_sha256: digest("six wins"),
        proposal_generator_sha256: digest("proposal generator"),
        proposal_runtime_sha256: digest("proposal runtime"),
        sotarena_root: "/frozen/sotarena".into(),
        input_manifest_artifact: frozen_file(
            "input-manifest.json",
            FROZEN_INPUT_MANIFEST_SHA256.into(),
        ),
        panel_artifact: frozen_file("panel.txt", FROZEN_PANEL_SHA256.into()),
        observation_snapshot_artifact: frozen_file("observations.json", digest("observations")),
        six_win_manifest_artifact: frozen_file("six-wins.json", digest("six wins")),
        registry_artifact: frozen_file("registry.json", digest("registry")),
        evaluator_artifact: frozen_file("evaluator", digest("evaluator")),
        math_policy_artifact: frozen_file("math.json", digest("math")),
        backend_policy_artifact: frozen_file("backend.json", digest("backend")),
        proposal_generator_artifact: frozen_file("generator", digest("proposal generator")),
        proposal_runtime_artifact: frozen_file("runtime", digest("proposal runtime")),
        checkpoint: FrozenCheckpointV1 {
            artifact_path: "/checkpoint/must-exist-at-launch".into(),
            artifact_sha256: FROZEN_CHECKPOINT_SHA256.into(),
            artifact_bytes: 100,
            regression_head_schema_sha256: jope_dime_regressor::search_v3::canonical_sha256(
                &regression_heads,
            )
            .unwrap(),
            regression_heads,
            inference_protocol_sha256: digest("inference"),
        },
        datasets,
        search_seeds: SEARCH_SEEDS.to_vec(),
        bootstrap_seed: BOOTSTRAP_SEED,
        budgets: Best5ArmV1::ALL
            .into_iter()
            .map(|arm| (arm, ArmBudgetV1::frozen()))
            .collect(),
        scq: ScqPolicyV1::frozen(),
        production_pointer_writes: false,
        production_learning_ingest: false,
        legacy_strategy_state_reused: false,
    }
}

fn proposal_for(dataset_sha256: &str) -> ProposalFreezeV1 {
    ProposalFreezeV1 {
        schema: ProposalFreezeV1::SCHEMA.into(),
        dataset_sha256: dataset_sha256.into(),
        checkpoint_sha256: FROZEN_CHECKPOINT_SHA256.into(),
        inference_request_sha256: digest(&format!("request-{dataset_sha256}")),
        inference_receipt_sha256: digest(&format!("receipt-{dataset_sha256}")),
        inference_ordinal: 1,
        proposals: (0..64).map(genome).collect(),
    }
}

fn search_result() -> SearchResultV1 {
    let archive = (0..20)
        .map(|index| {
            let candidate = genome(index);
            let predictor = FrozenPredictorV1::single(candidate.clone());
            let ancestry = AncestryV1 {
                predictor: predictor.clone(),
                founder_semantic_sha256: candidate.semantic_sha256.clone(),
                parent_semantic_sha256s: Vec::new(),
                model_seeded: true,
            };
            ArchiveMemberV1 {
                split_evidence: (0..8)
                    .map(|split_index| CandidateEvidenceV1 {
                        predictor: predictor.clone(),
                        ancestry: ancestry.clone(),
                        prediction_receipt_sha256: digest(&format!(
                            "archive-receipt-{index}-{split_index}"
                        )),
                        split_index,
                        r2_clipped: 0.2,
                        anchor_median_r2_clipped: 0.1,
                        paired_uplift: 0.1,
                        rmse: 1.0,
                        runtime_seconds: 1.0,
                        valid: true,
                        failure: None,
                    })
                    .collect(),
                predictor,
                ancestry,
            }
        })
        .collect::<Vec<_>>();
    let finalist_semantics = archive
        .iter()
        .take(10)
        .map(|member| member.predictor.semantic_sha256().unwrap())
        .collect::<Vec<_>>();
    let winner_semantic_sha256 = finalist_semantics.iter().min().unwrap().clone();
    let finalists = finalist_semantics
        .into_iter()
        .map(|semantic_sha256| {
            let predictor = archive
                .iter()
                .find(|member| member.predictor.semantic_sha256().unwrap() == semantic_sha256)
                .unwrap()
                .predictor
                .clone();
            Cv7EvidenceV1 {
                semantic_sha256: semantic_sha256.clone(),
                predictor: predictor.clone(),
                fold_prediction_receipt_sha256s: (0..7)
                    .map(|fold| digest(&format!("cv7-receipt-{semantic_sha256}-{fold}")))
                    .collect(),
                fold_r2: vec![0.2; 7],
                pooled_oof_r2: 0.2,
                rmse: 1.0,
                runtime_seconds: 1.0,
                complexity: predictor.complexity(),
                valid: true,
            }
        })
        .collect();
    SearchResultV1 {
        schema: SearchResultV1::SCHEMA.into(),
        campaign_freeze_sha256: digest("search-campaign"),
        proposal_freeze_sha256: digest("search-proposals"),
        arm: Best5ArmV1::ConfidenceGpControl,
        dataset_sha256: digest("search-dataset"),
        master_seed: 101,
        split_plan_sha256: digest("search-splits"),
        populations: (0..8)
            .map(|generation| GenerationPopulationV1 {
                generation,
                semantic_sha256s: (0..80)
                    .rev()
                    .map(|slot| digest(&format!("population-{generation}-{slot}")))
                    .collect(),
            })
            .collect(),
        archive,
        cv7_plan_sha256: digest("search-cv7"),
        finalists,
        winner_semantic_sha256,
        ledger: BudgetLedgerV1 {
            logical_operations: 950,
            predicted_cpu_seconds: 7_200,
            mandatory_anchor_evaluations: 80,
            completed_anchor_evaluations: 80,
            mandatory_search_evaluations: 640,
            completed_search_evaluations: 640,
            mandatory_archive_evaluations: 160,
            completed_archive_evaluations: 160,
            mandatory_cv7_evaluations: 70,
            completed_cv7_evaluations: 70,
            arm_specific_operations: 0,
        },
        test_data_accessed: false,
        ranking_state_shared: false,
    }
}

fn target_rows(count: usize) -> Vec<TargetRowV1> {
    (0..count)
        .map(|index| TargetRowV1 {
            stable_row_id: format!("row-{index:03}"),
            atomic_group: format!("group-{index:03}"),
            target: (index as f64 / 5.0).sin() + index as f64 / 100.0,
        })
        .collect()
}

fn descriptor(index: usize) -> DescriptorV1 {
    let mut counts = [0u8; 8];
    counts[index % 8] = (index % 6) as u8;
    DescriptorV1 {
        head_superfamily: (index % 4) as u8,
        transform_type_counts: counts,
        transform_count_normalized: (index % 7) as f64 / 6.0,
        complexity_normalized: (index % 20) as f64 / 19.0,
        normalized_hyperparameters: [(index % 11) as f64 / 10.0, (index % 13) as f64 / 12.0],
    }
}

fn fake_evaluator(path: &std::path::Path) -> FrozenFileV1 {
    let source = r#"#!/usr/bin/python3
import argparse, hashlib, json, pathlib, struct
p = argparse.ArgumentParser()
p.add_argument('--manifest', required=True)
p.add_argument('--output-dir', required=True)
p.add_argument('--threads')
p.add_argument('--timeout-seconds')
a = p.parse_args()
manifest_bytes = pathlib.Path(a.manifest).read_bytes()
m = json.loads(manifest_bytes)
out = pathlib.Path(a.output_dir)
out.mkdir(parents=True, exist_ok=True)
train = pathlib.Path(m['train_csv']).read_text().splitlines()
test = pathlib.Path(m['test_csv']).read_text().splitlines()
center = sum(float(row.split(',')[-1]) for row in train) / len(train)
rows = []
for candidate in m['candidates']:
    raw = b''.join(struct.pack('<d', center) for _ in test)
    name = candidate['semantic_sha256'] + '.bin'
    (out / name).write_bytes(raw)
    rows.append({
        'schema': 'JopeFlywheelCorpusEvalReceipt.v1',
        'task': m['task'],
        'dataset_hash': m['dataset_hash'],
        'semantic_sha256': candidate['semantic_sha256'],
        'predictions_sha256': hashlib.sha256(raw).hexdigest(),
        'predictions_path': name,
        'prediction_count': len(test),
        'valid': True,
        'failure': None,
        'failure_message': None,
        'runtime_seconds': 0.0,
        'train_sha256': m['train_sha256'],
        'test_sha256': m['test_sha256'],
        'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
    })
(out / (m['dataset_hash'] + '.receipts.jsonl')).write_text(
    ''.join(json.dumps(row, separators=(',', ':')) + '\n' for row in rows)
)
(out / (m['dataset_hash'] + '.done')).write_text(json.dumps({
    'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest()
}, separators=(',', ':')))
"#;
    fs::write(path, source).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
    let bytes = fs::read(path).unwrap();
    FrozenFileV1 {
        path: path.to_string_lossy().into_owned(),
        sha256: raw_sha256(&bytes),
        bytes: bytes.len() as u64,
    }
}

mod campaign_tests;
mod evaluation_tests;
mod search_tests;
mod worker_tests;
