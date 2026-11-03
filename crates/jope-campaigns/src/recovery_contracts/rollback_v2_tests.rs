#[cfg(test)]
mod rollback_v2_tests {
    use super::super::*;
    use serde_json::json;

    fn digest(label: &str) -> String {
        sha256_bytes(label.as_bytes())
    }

    fn write_json(root: &Path, name: &str, value: &Value) -> JsonEvidenceRefV2 {
        let path = root.join(name);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        let mut bytes = serde_json::to_vec(value).unwrap();
        bytes.push(b'\n');
        fs::write(&path, &bytes).unwrap();
        JsonEvidenceRefV2 {
            path: name.to_owned(),
            raw_file_sha256: sha256_bytes(&bytes),
            canonical_json_sha256: canonical_json_sha256(value).unwrap(),
        }
    }

    fn write_raw(root: &Path, name: &str, bytes: &[u8]) -> RawEvidenceRefV2 {
        let path = root.join(name);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, bytes).unwrap();
        RawEvidenceRefV2 {
            path: name.to_owned(),
            raw_file_sha256: sha256_bytes(bytes),
        }
    }

    fn fixture(root: &Path) -> RollbackCapsuleV2 {
        let mut closure = Vec::new();
        for index in 0..23 {
            let unit = format!("unit-{index:02}.service");
            let mut row = RollbackSystemdArtifactV2 {
                bundled_target: format!("runtime/systemd/{unit}"),
                unit,
                kind: "service".to_owned(),
                unit_file_sha256: digest(&format!("service-{index}")),
                resolved_effective_unit_sha256: String::new(),
                exec_start_argv: vec![format!(
                    "{runtime_current_root_str}/runtime/bin/service-{index}",
                    runtime_current_root_str = runtime_current_root_str()
                )],
                working_directory: Some(immutable_flywheel_root_str().to_owned()),
                drop_ins: Vec::new(),
            };
            row.resolved_effective_unit_sha256 = row.computed_effective_unit_sha256().unwrap();
            closure.push(row);
        }
        for index in 0..7 {
            let unit = format!("unit-{index:02}.timer");
            let mut row = RollbackSystemdArtifactV2 {
                bundled_target: format!("runtime/systemd/{unit}"),
                unit,
                kind: "timer".to_owned(),
                unit_file_sha256: digest(&format!("timer-{index}")),
                resolved_effective_unit_sha256: String::new(),
                exec_start_argv: Vec::new(),
                working_directory: None,
                drop_ins: Vec::new(),
            };
            row.resolved_effective_unit_sha256 = row.computed_effective_unit_sha256().unwrap();
            closure.push(row);
        }
        closure.sort_by(|left, right| left.unit.cmp(&right.unit));
        let mut files = closure
            .iter()
            .map(|row| RollbackBundleFileV2 {
                target: row.bundled_target.clone(),
                bytes: 1,
                sha256: row.unit_file_sha256.clone(),
            })
            .collect::<Vec<_>>();
        files.extend([
            RollbackBundleFileV2 {
                target: "runtime/bin/jope_selfplay_supply".to_owned(),
                bytes: 1,
                sha256: digest("rollback-supply"),
            },
            RollbackBundleFileV2 {
                target: "runtime/bin/jope_task_learning_controller".to_owned(),
                bytes: 1,
                sha256: digest("rollback-controller"),
            },
        ]);
        files.sort_by(|left, right| left.target.cmp(&right.target));
        let challenge = RollbackChallengeV1 {
            schema: ROLLBACK_CHALLENGE_SCHEMA.to_owned(),
            nonce_sha256: digest("challenge"),
            issued_at_unix_ms: 900,
            expires_at_unix_ms: 2_000,
            verifier_tool_sha256: digest("verifier"),
        };
        let challenge_ref = write_json(root, "challenge.json", &json!(challenge));
        let raw = write_raw(root, "raw/probe.txt", b"raw-probe-output");
        let manifest_file = digest("manifest-file");
        let manifest_canonical = digest("manifest-canonical");
        let mut node_probe_receipts = Vec::new();
        for node in REQUIRED_NODES {
            let receipt = RollbackNodeProbeV2 {
                schema: ROLLBACK_NODE_PROBE_SCHEMA.to_owned(),
                challenge_nonce_sha256: challenge.nonce_sha256.clone(),
                node: node.to_owned(),
                observed_hostname: format!("{node}.cluster"),
                runtime_bundle_sha256: ROLLBACK_RUNTIME_BUNDLE_SHA256.to_owned(),
                bundle_manifest_file_sha256: manifest_file.clone(),
                bundle_manifest_canonical_json_sha256: manifest_canonical.clone(),
                files: files.clone(),
                systemd_artifacts: closure.clone(),
                tool: RollbackProbeToolV2 {
                    executable_sha256: digest("probe-tool"),
                    version: "probe 1".to_owned(),
                    argv: vec!["probe".to_owned(), challenge.nonce_sha256.clone()],
                },
                started_at_unix_ms: 950,
                finished_at_unix_ms: 960,
                raw_output: raw.clone(),
            };
            node_probe_receipts.push(write_json(
                root,
                &format!("nodes/{node}.json"),
                &json!(receipt),
            ));
        }
        let artifact_names = closure.iter().map(|row| row.unit.clone()).collect();
        let path_audit = write_json(
            root,
            "path-audit.json",
            &json!(RollbackPathAuditV2 {
                schema: ROLLBACK_PATH_AUDIT_SCHEMA.to_owned(),
                challenge_nonce_sha256: challenge.nonce_sha256.clone(),
                runtime_bundle_sha256: ROLLBACK_RUNTIME_BUNDLE_SHA256.to_owned(),
                audited_systemd_artifacts: artifact_names,
                findings: Vec::new(),
                raw_output: raw.clone(),
            }),
        );
        let staged_tree_sha256 =
            rollback_isolated_staged_tree_sha256(ROLLBACK_RUNTIME_BUNDLE_SHA256, &closure).unwrap();
        let install = write_json(
            root,
            "install.json",
            &json!(RollbackInstallSimulationV2 {
                schema: ROLLBACK_INSTALL_SIMULATION_SCHEMA.to_owned(),
                challenge_nonce_sha256: challenge.nonce_sha256.clone(),
                runtime_bundle_sha256: ROLLBACK_RUNTIME_BUNDLE_SHA256.to_owned(),
                installer_argv: vec![
                    format!(
                        "{immutable_flywheel_root_str}/runtime-bundles/{ROLLBACK_RUNTIME_BUNDLE_SHA256}/runtime/bin/jope_selfplay_bundle", immutable_flywheel_root_str = immutable_flywheel_root_str()
                    ),
                    "--install-units".to_owned()
                ],
                installer_executable_sha256: digest("installer"),
                installer_version: "jope-selfplay-bundle 2".to_owned(),
                installer_exit_status: 0,
                systemd_analyze_argv: vec![
                    "systemd-analyze".to_owned(),
                    "--user".to_owned(),
                    "--root".to_owned(),
                    "/isolated".to_owned(),
                ],
                systemd_analyze_version: "systemd 257".to_owned(),
                systemd_analyze_exit_status: 0,
                staged_tree_sha256: staged_tree_sha256.clone(),
                expected_staged_tree_sha256: staged_tree_sha256,
                stdout: raw.clone(),
                stderr: raw.clone(),
            }),
        );
        let retention_index = write_json(
            root,
            "retention-index.json",
            &json!({
                "schema": "JopeEvaluatorRetentionIndex.v1",
                "active_bundle_sha256": ROLLBACK_RUNTIME_BUNDLE_SHA256,
                "prepared_for_bundle_sha256": ROLLBACK_RUNTIME_BUNDLE_SHA256,
                "retained": [{"evaluator_sha256": digest("eval")}],
                "referenced_evaluator_sha256": [digest("eval")],
                "unavailable_referenced_evaluator_sha256": [],
            }),
        );
        let retention = write_json(
            root,
            "retention.json",
            &json!(RollbackEvaluatorRetentionV2 {
                schema: ROLLBACK_EVALUATOR_RETENTION_SCHEMA.to_owned(),
                challenge_nonce_sha256: challenge.nonce_sha256.clone(),
                runtime_bundle_sha256: ROLLBACK_RUNTIME_BUNDLE_SHA256.to_owned(),
                retention_index,
            }),
        );
        let proposer = write_json(
            root,
            "states/proposer.json",
            &json!({"schema": "CohortSupplyState.v2", "proposal_job_id": "p"}),
        );
        let trainer = write_json(
            root,
            "states/trainer.json",
            &json!({"schema": "TaskLearningControllerState.v2", "intent": null}),
        );
        let compatibility = write_json(
            root,
            "compatibility.json",
            &json!(RollbackStateCompatibilityV2 {
                schema: ROLLBACK_STATE_COMPATIBILITY_SCHEMA.to_owned(),
                challenge_nonce_sha256: challenge.nonce_sha256,
                runtime_bundle_sha256: ROLLBACK_RUNTIME_BUNDLE_SHA256.to_owned(),
                rollback_binaries: vec![
                    RollbackStateBinaryV2 {
                        role: "cohort_proposer".to_owned(),
                        bundled_target: "runtime/bin/jope_selfplay_supply".to_owned(),
                        sha256: digest("rollback-supply"),
                    },
                    RollbackStateBinaryV2 {
                        role: "alternating_trainer".to_owned(),
                        bundled_target: "runtime/bin/jope_task_learning_controller".to_owned(),
                        sha256: digest("rollback-controller"),
                    },
                ],
                command: vec![
                    "runtime/bin/jope_selfplay_supply".to_owned(),
                    "runtime/bin/jope_task_learning_controller".to_owned(),
                    "--read-state".to_owned(),
                ],
                exit_status: 0,
                stdout: raw.clone(),
                stderr: raw,
                primary_documents: vec![
                    RollbackStateDocumentV2 {
                        role: "cohort_proposer".to_owned(),
                        schema: "CohortSupplyState.v2".to_owned(),
                        document: proposer,
                    },
                    RollbackStateDocumentV2 {
                        role: "alternating_trainer".to_owned(),
                        schema: "TaskLearningControllerState.v2".to_owned(),
                        document: trainer,
                    },
                ],
            }),
        );
        let mut capsule = RollbackCapsuleV2 {
            schema: ROLLBACK_CAPSULE_V2_SCHEMA.to_owned(),
            generated_at_unix_ms: 1_000,
            freshness_budget_seconds: 1,
            runtime_bundle_sha256: ROLLBACK_RUNTIME_BUNDLE_SHA256.to_owned(),
            runtime_bundle_manifest_file_sha256: manifest_file,
            runtime_bundle_manifest_canonical_json_sha256: manifest_canonical,
            embedded_service_manifest_file_sha256: digest("service-file"),
            embedded_service_manifest_canonical_json_sha256: digest("service-canonical"),
            files,
            systemd_closure: closure,
            systemd_closure_canonical_json_sha256: String::new(),
            challenge: challenge_ref,
            node_probe_receipts,
            complete_path_audit: path_audit,
            isolated_install_simulation: install,
            evaluator_retention: retention,
            rollback_state_compatibility: compatibility,
        };
        capsule.systemd_closure_canonical_json_sha256 =
            capsule.computed_systemd_closure_sha256().unwrap();
        capsule
    }

    #[test]
    fn v2_derives_outcomes_from_complete_content_addressed_evidence() {
        let directory = tempfile::tempdir().unwrap();
        fixture(directory.path())
            .validate_evidence_at(directory.path(), 1_000)
            .unwrap();
    }

    #[test]
    fn v1_and_fake_pass_boolean_cannot_authorize() {
        let historical = json!({"schema": ROLLBACK_CAPSULE_SCHEMA});
        assert_eq!(
            classify_rollback_capsule_schema(&historical).unwrap(),
            "historical-non-authorizing-v1"
        );
        assert!(serde_json::from_value::<RollbackCapsuleV2>(historical).is_err());
        let directory = tempfile::tempdir().unwrap();
        let mut value = serde_json::to_value(fixture(directory.path())).unwrap();
        value["simulated_install_passed"] = Value::Bool(true);
        assert!(serde_json::from_value::<RollbackCapsuleV2>(value).is_err());
    }

    #[test]
    fn missing_altered_stale_duplicate_and_incomplete_evidence_is_rejected() {
        let directory = tempfile::tempdir().unwrap();
        let capsule = fixture(directory.path());
        fs::remove_file(directory.path().join(&capsule.node_probe_receipts[0].path)).unwrap();
        assert!(capsule
            .validate_evidence_at(directory.path(), 1_000)
            .is_err());

        let directory = tempfile::tempdir().unwrap();
        let capsule = fixture(directory.path());
        fs::write(
            directory.path().join(&capsule.node_probe_receipts[0].path),
            b"{}\n",
        )
        .unwrap();
        assert!(capsule
            .validate_evidence_at(directory.path(), 1_000)
            .is_err());

        let directory = tempfile::tempdir().unwrap();
        let capsule = fixture(directory.path());
        assert!(capsule
            .validate_evidence_at(directory.path(), 2_000)
            .is_err());

        let directory = tempfile::tempdir().unwrap();
        let mut capsule = fixture(directory.path());
        let first: RollbackNodeProbeV2 = capsule.node_probe_receipts[0]
            .load(directory.path())
            .unwrap();
        let mut duplicate: RollbackNodeProbeV2 = capsule.node_probe_receipts[1]
            .load(directory.path())
            .unwrap();
        duplicate.node = first.node;
        duplicate.observed_hostname = first.observed_hostname;
        capsule.node_probe_receipts[1] =
            write_json(directory.path(), "nodes/duplicate.json", &json!(duplicate));
        assert!(capsule
            .validate_evidence_at(directory.path(), 1_000)
            .is_err());

        let directory = tempfile::tempdir().unwrap();
        let mut capsule = fixture(directory.path());
        let mut incomplete: RollbackNodeProbeV2 = capsule.node_probe_receipts[0]
            .load(directory.path())
            .unwrap();
        incomplete.files.pop();
        capsule.node_probe_receipts[0] = write_json(
            directory.path(),
            "nodes/incomplete.json",
            &json!(incomplete),
        );
        assert!(capsule
            .validate_evidence_at(directory.path(), 1_000)
            .is_err());
    }

    #[test]
    fn overlay_and_raw_vs_canonical_digest_substitution_are_rejected() {
        let directory = tempfile::tempdir().unwrap();
        let mut capsule = fixture(directory.path());
        capsule.systemd_closure[0].unit_file_sha256 = digest("altered-unit");
        capsule.systemd_closure_canonical_json_sha256 =
            capsule.computed_systemd_closure_sha256().unwrap();
        assert!(capsule
            .validate_evidence_at(directory.path(), 1_000)
            .is_err());

        let directory = tempfile::tempdir().unwrap();
        let mut capsule = fixture(directory.path());
        std::mem::swap(
            &mut capsule.challenge.raw_file_sha256,
            &mut capsule.challenge.canonical_json_sha256,
        );
        assert!(capsule
            .validate_evidence_at(directory.path(), 1_000)
            .is_err());
    }
}
