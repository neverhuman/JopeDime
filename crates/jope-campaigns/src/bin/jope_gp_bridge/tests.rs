use super::*;
use tempfile::TempDir;

fn digest(value: &str) -> String {
    sha256_bytes(value.as_bytes())
}

fn proposal(slot: usize, checkpoint: &str) -> GpProposal {
    let genome = format!("(pipe X (rff_cos n=32 seed={}) (ridge))", slot + 1);
    GpProposal {
        task: "regression".to_owned(),
        dataset_hash: "dataset".to_owned(),
        canonical_genome: genome.clone(),
        semantic_sha256: digest(&genome),
        model_checkpoint_sha256: checkpoint.to_owned(),
        slot,
        score: Some(slot as f64),
        proposal_score: None,
    }
}

fn member(name: &str, score: f64) -> GpPopulationMember {
    GpPopulationMember {
        canonical_genome: name.to_owned(),
        semantic_sha256: digest(name),
        score: Some(score),
        dl_origin: false,
        injected_checkpoint_sha256: None,
        ancestry: None,
    }
}

fn proposal_fixture(temporary: &TempDir) -> (ProposalInputs, String) {
    let checkpoint_artifact = temporary.path().join("checkpoint.model");
    fs::write(&checkpoint_artifact, b"checkpoint").unwrap();
    let checkpoint = sha256_file(&checkpoint_artifact).unwrap();
    let checkpoint_pointer = temporary.path().join("checkpoint.json");
    atomic_json(
        &checkpoint_pointer,
        &json!({
            "checkpoint_sha256": checkpoint,
            "checkpoint": checkpoint_artifact,
        }),
    )
    .unwrap();
    let proposal_file = temporary.path().join("proposals.jsonl");
    let mut file = File::create(&proposal_file).unwrap();
    for row in (0..64).map(|slot| proposal(slot, &checkpoint)) {
        file.write_all(&canonical_json_bytes(&serde_json::to_value(row).unwrap()).unwrap())
            .unwrap();
        file.write_all(b"\n").unwrap();
    }
    file.sync_all().unwrap();
    let proposal_pointer = temporary.path().join("proposals.json");
    atomic_json(
        &proposal_pointer,
        &json!({
            "checkpoint_sha256": checkpoint,
            "proposals": proposal_file,
            "proposals_sha256": sha256_file(&proposal_file).unwrap(),
        }),
    )
    .unwrap();
    (
        ProposalInputs {
            checkpoint_pointer,
            proposals_pointer: proposal_pointer,
            task: Task::Regression,
            dataset: "dataset".to_owned(),
        },
        checkpoint,
    )
}

#[test]
fn export_verifies_checkpoint_artifact_pointer_and_all_64_slots() {
    let temporary = TempDir::new().unwrap();
    let (input, checkpoint) = proposal_fixture(&temporary);
    let verified = verify_proposals(&input).unwrap();
    assert_eq!(verified.checkpoint_sha256, checkpoint);
    assert_eq!(verified.proposals.len(), 64);
    assert_eq!(verified.proposals[0].slot, 0);
    assert_eq!(verified.proposals[63].slot, 63);
    assert_eq!(verified.task_invalid_skipped, 0);
    fs::write(&verified.checkpoint_artifact, b"changed").unwrap();
    assert!(verify_proposals(&input).is_err());
}

#[test]
fn task_invalid_proposals_are_skipped_and_counted() {
    let temporary = TempDir::new().unwrap();
    let (input, checkpoint) = proposal_fixture(&temporary);
    let genome = "(pipe X (ridge))";
    let invalid = GpProposal {
        task: "binary".to_owned(),
        dataset_hash: "other".to_owned(),
        canonical_genome: genome.to_owned(),
        semantic_sha256: digest(genome),
        model_checkpoint_sha256: checkpoint.clone(),
        slot: 0,
        score: None,
        proposal_score: None,
    };
    let proposal_file = temporary.path().join("proposals.jsonl");
    let mut file = OpenOptions::new()
        .append(true)
        .open(&proposal_file)
        .unwrap();
    file.write_all(&canonical_json_bytes(&serde_json::to_value(&invalid).unwrap()).unwrap())
        .unwrap();
    file.write_all(b"\n").unwrap();
    file.sync_all().unwrap();
    atomic_json(
        &input.proposals_pointer,
        &json!({
            "checkpoint_sha256": checkpoint,
            "proposals": proposal_file,
            "proposals_sha256": sha256_file(&proposal_file).unwrap(),
        }),
    )
    .unwrap();
    let verified = verify_proposals(&input).unwrap();
    assert_eq!(verified.proposals.len(), 64);
    assert_eq!(verified.task_invalid_skipped, 1);
}

#[test]
fn generation_zero_is_exact_budget_with_dl_and_robust_seeds() {
    let checkpoint = "a".repeat(64);
    let population = (0..80)
        .map(|slot| member(&format!("existing-{slot}"), slot as f64))
        .collect::<Vec<_>>();
    let proposals = (0..64)
        .map(|slot| proposal(slot, &checkpoint))
        .collect::<Vec<_>>();
    let robust = (0..20)
        .map(|slot| {
            let genome = format!("robust-{slot}");
            RobustSeed {
                task: "regression".to_owned(),
                semantic_sha256: digest(&genome),
                canonical_genome: genome,
                dataset_count: 3,
                mean_within_dataset_percentile: 0.9 - slot as f64 / 100.0,
                worst_within_dataset_percentile: 0.5,
                strict_win_count: 1,
            }
        })
        .collect::<Vec<_>>();
    let value = merge_generation_zero(
        &population,
        &proposals,
        &checkpoint,
        &robust,
        12,
        Task::Regression,
        "dataset",
    )
    .unwrap();
    assert_eq!(value["population"].as_array().unwrap().len(), 80);
    assert_eq!(value["composition"]["current_checkpoint_dl"], 64);
    assert_eq!(
        value["composition"]["repeated_cross_dataset_gp_winners"],
        12
    );
    assert_eq!(value["composition"]["retained_existing"], 4);
    assert!(value["population"]
        .as_array()
        .unwrap()
        .iter()
        .take(76)
        .all(|member| member["score"].is_null()));
}

#[test]
fn bridge_reads_protocol_scoped_robust_seeds_from_current_corpus() {
    let temporary = TempDir::new().unwrap();
    let corpus_path = temporary.path().join("corpus.sqlite");
    let connection = selfplay_ingest::open_corpus(&corpus_path).unwrap();
    let protocol = "9".repeat(64);
    let genome = "(pipe X (ridge))";
    let semantic = digest(genome);
    for (index, dataset) in ["dataset-a", "dataset-b"].into_iter().enumerate() {
        connection
            .execute(
                "INSERT INTO observations(
                   task,dataset_hash,semantic_sha256,canonical_genome,candidate_source,
                   metric,score,runtime_seconds,train_sha256,test_sha256,
                   receipt_sha256,receipt_path,manifest_sha256,frozen_seed_sha256,
                   model_checkpoint_sha256,hall_of_fame_sha256,created_at_ms,
                   authority_kind,protocol_sha256,origin_class
                 ) VALUES(
                   'regression',?1,?2,?3,'dataset_local_elite',
                   'r2',?4,1.0,?5,?6,?7,'receipt',?8,?9,?10,?11,?12,
                   'corrected',?13,'gp_origin'
                 )",
                rusqlite::params![
                    dataset,
                    semantic,
                    genome,
                    0.5 + index as f64 / 10.0,
                    "a".repeat(64),
                    "b".repeat(64),
                    "c".repeat(64),
                    "d".repeat(64),
                    "e".repeat(64),
                    "f".repeat(64),
                    "7".repeat(64),
                    index as i64,
                    protocol,
                ],
            )
            .unwrap();
    }
    drop(connection);
    let (first, first_hash, first_value) =
        read_dynamic_robust_bank(&corpus_path, Task::Regression, &protocol, 10, 2).unwrap();
    let (second, second_hash, second_value) =
        read_dynamic_robust_bank(&corpus_path, Task::Regression, &protocol, 10, 2).unwrap();
    assert_eq!(first.len(), 1);
    assert_eq!(first[0].semantic_sha256, semantic);
    assert_eq!(first[0].dataset_count, 2);
    assert_eq!(first_hash, second_hash);
    assert_eq!(first_value, second_value);
    assert_eq!(second.len(), 1);
}

#[test]
fn checkpoint_injection_replaces_exactly_eight_non_elites() {
    let current = "b".repeat(64);
    let previous = "a".repeat(64);
    let population = (0..20)
        .map(|slot| member(&format!("existing-{slot}"), slot as f64))
        .collect::<Vec<_>>();
    let proposals = (0..64)
        .map(|slot| proposal(slot, &current))
        .collect::<Vec<_>>();
    let value = injection_value(
        &population,
        &proposals,
        &current,
        &previous,
        4,
        Task::Regression,
        "dataset",
    )
    .unwrap();
    assert_eq!(value["injection"]["injected"], 8);
    assert_eq!(
        &value["population"].as_array().unwrap()[..4],
        &serde_json::to_value(&population)
            .unwrap()
            .as_array()
            .unwrap()[..4]
    );
    for replacement in value["injection"]["replacements"].as_array().unwrap() {
        let slot = replacement["slot"].as_u64().unwrap() as usize;
        assert!(value["population"][slot]["score"].is_null());
    }
}

#[test]
fn lineage_is_transitive_and_attributes_champion() {
    let temporary = TempDir::new().unwrap();
    let origin = digest("origin");
    let child = digest("child");
    let champion = digest("champion");
    let path = temporary.path().join("lineage.json");
    atomic_json(
        &path,
        &json!([
            {"semantic_sha256": origin, "dl_origin": true},
            {"semantic_sha256": child, "parents": [origin]},
            {"semantic_sha256": champion, "parents": [child], "strict_win": true}
        ]),
    )
    .unwrap();
    let value = lineage_value(&path, &champion, Task::Regression, "dataset", "run").unwrap();
    assert_eq!(value["dl_origin_parent_count"], 1);
    assert_eq!(value["dl_origin_descendant_count"], 2);
    assert_eq!(value["dl_origin_strict_win_count"], 1);
    assert_eq!(value["final_champion_has_dl_contribution"], true);
}

#[test]
fn receipts_are_idempotent_and_hash_chained() {
    let temporary = TempDir::new().unwrap();
    let output = temporary.path().join("output.json");
    let receipts = temporary.path().join("receipts.jsonl");
    let first = durable_write(
        "test",
        json!({"input": 1}),
        Value::Null,
        &output,
        &json!({"output": 1}),
        &receipts,
        "stable",
    )
    .unwrap();
    let second = durable_write(
        "test",
        json!({"input": 1}),
        Value::Null,
        &output,
        &json!({"output": 1}),
        &receipts,
        "stable",
    )
    .unwrap();
    assert_eq!(first, second);
    assert_eq!(validate_receipt_chain(&receipts).unwrap().len(), 1);
    assert!(durable_write(
        "test",
        json!({"input": 2}),
        Value::Null,
        &output,
        &json!({"output": 1}),
        &receipts,
        "stable",
    )
    .is_err());
}
