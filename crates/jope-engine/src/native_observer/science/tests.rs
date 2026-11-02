#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn a_fresh_source_does_not_hide_another_stale_scientific_row() {
        let rows = Rows {
            checkpoints: vec![
                json!({"identity":"old","status":"committed","scientific_verified_at_unix_ms":1000}),
                json!({"identity":"recent","status":"committed","scientific_verified_at_unix_ms":39000}),
            ],
            corpus: Some(json!({"sha256":"a".repeat(64),"scientific_verified_at_unix_ms":1000})),
            ..Default::default()
        };
        let reader = Reader {
            latest: Arc::new(Mutex::new(Observed {
                at: Some(40000),
                rows,
            })),
        };
        let mut snapshot = Snapshot {
            schema: "jain.jope_native_status.v1",
            observed_at_unix_ms: 40000,
            scope: "test".into(),
            resources: vec![],
            queues: vec![],
            continuations: vec![],
            corpus: None,
            checkpoints: vec![],
            comparisons: vec![],
            winners: vec![],
            notes: vec![],
            source_errors: vec![],
            scientific_sources: vec![],
            scientific_verified_at_unix_ms: None,
        };
        reader.enrich_science(&mut snapshot);
        assert!(snapshot.checkpoints[0]["status"]
            .as_str()
            .unwrap()
            .starts_with("STALE"));
        assert_eq!(snapshot.checkpoints[1]["status"], "committed");
        assert!(snapshot.corpus.is_none());
        assert_eq!(snapshot.observed_at_unix_ms, 40000);
    }
    #[test]
    fn descriptive_comparison_counts_cannot_become_independent_evidence() {
        let c = json!({"wins":1,"losses":1,"ties":0,"unavailable":0,"decisive_pairs":2,"win_rate":0.5,"nominal_unclustered_wilson_95_interval":[0.09453120573423074,0.9054687942657693],"independent_sampling_units_validated":false,"independent_final_test":false});
        let mut v = json!({"schema":"jope.native_matched_comparisons.v1","generation":0,"operational_progress_is_not_model_improvement":true,"model_comparisons":{"generic":c,"shuffled":c}});
        let rows = comparison_rows(&v, 0, &reference(), 1000).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["independent_model_improvement_claim"], false);
        v["model_comparisons"]["generic"]["win_rate"] = json!(0.9);
        assert!(comparison_rows(&v, 0, &reference(), 1000).is_err());
        v["model_comparisons"]["generic"]["win_rate"] = json!(0.5);
        v["model_comparisons"]["generic"]["independent_final_test"] = json!(true);
        assert!(comparison_rows(&v, 0, &reference(), 1000).is_err());
    }
    #[test]
    fn ordered_winner_joins_capture_identity_not_outbox_sequence() {
        let event = "a".repeat(64);
        let mut v = json!({"schema":"jain.datasets_service_receipt.v1","captures":[{"evaluation":format!("champion-event:{event}"),"producer":"original","sequence":71,"envelope_sha256":"b".repeat(64),"acknowledged":true}],"publications":[]});
        let status = publication_status(&event, &v).unwrap().unwrap();
        assert_eq!(status["capture_sequence"], 71);
        assert!(status["local_commit"].is_null());
        assert!(status["remote_verified_recorded"].is_null());
        assert!(publication_status(&"c".repeat(64), &v).unwrap().is_none());
        let duplicate = v["captures"][0].clone();
        v["captures"].as_array_mut().unwrap().push(duplicate);
        assert!(publication_status(&event, &v).is_err());
    }
    #[test]
    fn remote_record_must_bind_exact_original_chain_and_every_part() {
        let t = tempfile::tempdir().unwrap();
        let path = t.path().join("chain.json");
        let event = "a".repeat(64);
        let commit = "c".repeat(40);
        let chain = json!({"schema":"jain.publication.chain.v1","producer":"p","sequence":71,"envelope_sha256":"b".repeat(64),"final_event_commit":commit,"final_event_ref":"refs/events/71","batches":[{"commit":commit,"batch_sha256":"d".repeat(64)}]});
        let bytes = serde_json::to_vec(&chain).unwrap();
        fs::write(&path, &bytes).unwrap();
        let digest = hash(&bytes);
        let mut prepared = chain.clone();
        prepared["chain_manifest_path"] = json!(path);
        prepared["chain_manifest_sha256"] = json!(digest);
        let receipt = json!({"schema":"jain.datasets_service_receipt.v1","captures":[{"evaluation":format!("champion-event:{event}"),"producer":"p","sequence":71,"envelope_sha256":"b".repeat(64),"acknowledged":true}],"publications":[prepared]});
        assert_eq!(
            publication_status(&event, &receipt).unwrap().unwrap()["remote_verified_recorded"],
            false
        );
        let mut remote = json!({"schema":"jain.publication.chain_receipt.v1","chain_manifest_sha256":digest,"producer":"p","sequence":71,"envelope_sha256":"b".repeat(64),"final_event_commit":commit,"final_event_ref":"refs/events/71","verified_download":true,"parts":[{"schema":"jain.publication.receipt.v1","commit":commit,"batch_sha256":"d".repeat(64),"verified_download":true,"verified_at_unix_ms":1000}]});
        let p = t.path().join("published-chain.json");
        fs::write(&p, serde_json::to_vec(&remote).unwrap()).unwrap();
        assert_eq!(
            publication_status(&event, &receipt).unwrap().unwrap()["remote_verified_recorded"],
            true
        );
        remote["sequence"] = json!(72);
        fs::write(&p, serde_json::to_vec(&remote).unwrap()).unwrap();
        assert!(publication_status(&event, &receipt).is_err());
    }
    fn reference() -> Artifact {
        Artifact {
            path: "/immutable/qualification.json".into(),
            sha256: "a".repeat(64),
        }
    }
    #[test]
    fn qualification_checkpoint_is_never_production_or_promoted() {
        let v = json!({"schema":"jope.hardware_training_benchmark_verified.v1","production_learning":false,"numerical_result_verified":true,"initial_optimizer_step":60000,"target_optimizer_step":60010,"exact_updates":10,"checkpoint":{"path":"/original/state.json","sha256":"b".repeat(64)},"microbatch":256,"full_scheduler_lifecycle_passed":false});
        let r = qualification_rows("original", &reference(), &v, Some(1000), 2000).unwrap();
        assert_eq!(r.checkpoints[0]["qualification"], true);
        assert_eq!(r.checkpoints[0]["original_at_unix_ms"], 1000);
        assert!(r.comparisons.is_empty());
        let mut bad = v;
        bad["production_learning"] = json!(true);
        assert!(qualification_rows("bad", &reference(), &bad, None, 2000).is_err());
    }
    #[test]
    fn exact_eof_window_and_original_time_are_required() {
        let mut v = json!({"schema":"jope.actual_neural_eof_recovery_qualification.v1","production_learning":false,"passed":true,"unknown_final_tail":false,"original_initial_step":60000,"original_target_step":60100,"first_attempt_committed_updates":6,"retry_executed_updates":94,"total_updates":100,"final_checkpoint":{"path":"/old/state.json","sha256":"b".repeat(64)}});
        assert!(qualification_rows("q", &reference(), &v, None, 1000).is_ok());
        v["retry_executed_updates"] = json!(100);
        assert!(qualification_rows("q", &reference(), &v, None, 1000).is_err());
        assert!(original_time(Some(9000), 1000).is_err());
    }
    #[test]
    fn corrupt_and_nonregular_source_never_becomes_scientific_success() {
        let t = tempfile::tempdir().unwrap();
        let p = t.path().join("evidence");
        fs::write(&p, b"{}").unwrap();
        let a = Artifact {
            path: p.clone(),
            sha256: hash(b"{}"),
        };
        assert!(checked(None, &a, 1024).is_ok());
        fs::write(&p, b"[]").unwrap();
        assert!(checked(None, &a, 1024).is_err());
        fs::remove_file(&p).unwrap();
        std::os::unix::fs::symlink("/dev/zero", &p).unwrap();
        assert!(checked(None, &a, 1024).is_err());
    }
}
