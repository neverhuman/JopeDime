use super::*;
use std::os::unix::fs::MetadataExt;

#[test]
fn synchronization_does_not_ssh_the_scheduler_local_node() {
    let hostname = "xbabe2";
    let local = GPU_SYNC_NODES
        .iter()
        .copied()
        .filter(|node| node_is_local(node, hostname))
        .collect::<Vec<_>>();
    assert_eq!(local, vec!["xbabe2"]);
    assert!(node_is_local("xbabe2", "xbabe2.cluster.local"));
    for remote in ["xbabe0", "xbabe1", "xbabe3", "xbabe4"] {
        assert!(!node_is_local(remote, hostname));
    }
}

#[test]
fn fresh_live_probe_records_rechallenged_hashes_for_all_nodes() {
    use std::os::unix::process::ExitStatusExt;

    fn fixture(root: &Path, bundle_sha256: &str) -> Value {
        let bundle = root.join(bundle_sha256);
        fs::create_dir_all(bundle.join("runtime/systemd")).unwrap();
        fs::write(bundle.join("bundle.json"), b"{}\n").unwrap();
        let target = "runtime/systemd/worker.service";
        let bytes = b"[Service]\nExecStart=/runtime/bin/worker\n";
        fs::write(bundle.join(target), bytes).unwrap();
        json!({"files": [{
            "target": target,
            "sha256": sha256_bytes(bytes),
            "bytes": bytes.len(),
        }]})
    }

    fn fake_node_probe(
        stale_node: Option<&'static str>,
    ) -> impl FnMut(&str, &str, &str, &[String]) -> Result<std::process::Output> {
        move |node, _local_hostname, program, arguments| {
            let stdout = match (program, arguments) {
                ("hostname", _) => format!("{node}.cluster.local\n"),
                ("sha256sum", [arg]) if arg == "--version" => {
                    "sha256sum (GNU coreutils) 9.1\n".to_owned()
                }
                ("sha256sum", args) => args
                    .iter()
                    .map(|arg| {
                        let digest = if Some(node) == stale_node && arg.contains("worker.service") {
                            "0".repeat(64)
                        } else {
                            sha256_file(Path::new(arg)).unwrap()
                        };
                        format!("{digest}  {arg}\n")
                    })
                    .collect(),
                _ => panic!("unexpected probe command {program} {arguments:?}"),
            };
            Ok(std::process::Output {
                status: std::process::ExitStatus::from_raw(0),
                stdout: stdout.into_bytes(),
                stderr: Vec::new(),
            })
        }
    }

    let bundle_sha256 = "a".repeat(64);
    let success_root = tempfile::tempdir().unwrap();
    let manifest = fixture(success_root.path(), &bundle_sha256);
    let receipt = fresh_live_bundle_probe_with(
        success_root.path(),
        &bundle_sha256,
        &manifest,
        || Ok("controller".to_owned()),
        fake_node_probe(None),
    )
    .unwrap();
    let receipt: Value =
        serde_json::from_slice(&fs::read(receipt["receipt"].as_str().unwrap()).unwrap()).unwrap();
    let node_receipts = receipt["receipts"].as_array().unwrap();
    assert_eq!(node_receipts.len(), GPU_SYNC_NODES.len());
    assert_eq!(
        node_receipts
            .iter()
            .map(|row| row["node"].as_str().unwrap())
            .collect::<Vec<_>>(),
        GPU_SYNC_NODES
    );
    assert!(node_receipts.iter().all(|row| {
        row["complete_file_inventory"][0]["sha256"] == manifest["files"][0]["sha256"]
    }));

    let stale_root = tempfile::tempdir().unwrap();
    let stale_manifest = fixture(stale_root.path(), &bundle_sha256);
    assert!(fresh_live_bundle_probe_with(
        stale_root.path(),
        &bundle_sha256,
        &stale_manifest,
        || Ok("controller".to_owned()),
        fake_node_probe(Some("xbabe3")),
    )
    .is_err());
    assert!(!stale_root.path().join("activation-probes").exists());
}

#[test]
fn operational_python_is_rejected() {
    let source = Path::new("/repo");
    let prime = Path::new("/prime");
    let encoding = Path::new("/encoding");
    let candidate = Candidate {
        source: source.join("scripts/queue.py"),
        target: "runtime/scripts/queue.py".to_owned(),
        python_role: None,
    };
    assert!(validate_python_candidate(&candidate, source, prime, encoding).is_err());
}

#[test]
fn exact_pytorch_allowlist_is_accepted() {
    let source = Path::new("/repo");
    let prime = Path::new("/prime");
    let encoding = Path::new("/encoding");
    let candidate = Candidate {
        source: source.join("scripts/test_exact_dl_generate.py"),
        target: "pytorch/proposals/test_exact_dl_generate.py".to_owned(),
        python_role: Some("pytorch_proposal_inference"),
    };
    validate_python_candidate(&candidate, source, prime, encoding).unwrap();
    let context_candidate = Candidate {
        source: source.join("scripts/test_exact_context_proposals.py"),
        target: "pytorch/proposals/test_exact_context_proposals.py".to_owned(),
        python_role: Some("pytorch_proposal_inference"),
    };
    validate_python_candidate(&context_candidate, source, prime, encoding).unwrap();
    let trainer = Candidate {
        source: source.join("scripts/test_exact_train.py"),
        target: "pytorch/training/test_exact_train.py".to_owned(),
        python_role: Some("pytorch_training"),
    };
    validate_python_candidate(&trainer, source, prime, encoding).unwrap();
    let dependency = Candidate {
        source: prime.join("jope/data/active.py"),
        target: "pytorch/jope/data/active.py".to_owned(),
        python_role: Some("pytorch_dependency_package"),
    };
    validate_python_candidate(&dependency, source, prime, encoding).unwrap();
    let encoder = Candidate {
        source: encoding.join("encode_batch.py"),
        target: "pytorch/encoding/encode_batch.py".to_owned(),
        python_role: Some("pytorch_dataset_encoding"),
    };
    validate_python_candidate(&encoder, source, prime, encoding).unwrap();
}

#[test]
fn python_systemd_execstart_is_rejected() {
    let unit = "[Service]\nExecStart=/usr/bin/python3 /runtime/queue.py\n";
    assert!(validate_systemd_unit(unit).is_err());
    validate_systemd_unit("[Service]\nExecStart=/runtime/bin/jope_selfplay_queue --submit\n")
        .unwrap();
    assert!(validate_systemd_unit(&format!(
        "[Service]\nExecStart={}/target-promotion/release/jope_selfplay_promotion\n",
        default_source_str()
    ))
    .is_err());
    assert!(validate_systemd_unit(&format!(
        "[Service]\nWorkingDirectory={}\nExecStart=/runtime/bin/tool\n",
        default_source_str()
    ))
    .is_err());
}

#[test]
fn systemd_accepts_only_bundle_local_pytorch_scripts() {
    let watchdog = rendered_selfplay_unit("jope-selfplay-trainer-watchdog.service");
    validate_systemd_unit(&watchdog).unwrap();

    let inside = format!(
            "[Service]\nExecStart=/runtime/bin/controller --trainer-script {active_bundle_pytorch_root_str}/training/test_exact_train.py\n", active_bundle_pytorch_root_str = active_bundle_pytorch_root_str()
        );
    validate_systemd_unit(&inside).unwrap();

    let outside =
        "[Service]\nExecStart=/runtime/bin/controller --trainer-script /tmp/training.py\n";
    assert!(validate_systemd_unit(outside).is_err());

    let mixed = format!(
            "[Service]\nExecStart=/runtime/bin/controller --trainer-script {active_bundle_pytorch_root_str}/training/test_exact_train.py --common-script /tmp/common.py\n", active_bundle_pytorch_root_str = active_bundle_pytorch_root_str()
        );
    assert!(validate_systemd_unit(&mixed).is_err());

    let traversal = format!(
            "[Service]\nExecStart=/runtime/bin/controller --trainer-script {active_bundle_pytorch_root_str}/../outside.py\n", active_bundle_pytorch_root_str = active_bundle_pytorch_root_str()
        );
    assert!(validate_systemd_unit(&traversal).is_err());

    let continued =
        "[Service]\nExecStart=/runtime/bin/controller \\\n+  --trainer-script /tmp/training.py\n";
    assert!(validate_systemd_unit(continued).is_err());

    for interpreter in ["/usr/bin/python3-dbg", "/usr/bin/pypy3"] {
        let unit = format!("[Service]\nExecStart={interpreter} --version\n");
        assert!(validate_systemd_unit(&unit).is_err());
    }

    let packed = format!(
            "[Service]\nExecStart=/runtime/bin/controller --pythonpath {active_bundle_pytorch_root_str}/training/test_exact_train.py:/tmp/outside.py\n", active_bundle_pytorch_root_str = active_bundle_pytorch_root_str()
        );
    assert!(validate_systemd_unit(&packed).is_err());

    let direct_script = format!(
        "[Service]\nExecStart={active_bundle_pytorch_root_str}/training/test_exact_train.py\n",
        active_bundle_pytorch_root_str = active_bundle_pytorch_root_str()
    );
    assert!(validate_systemd_unit(&direct_script).is_err());
}

fn selfplay_unit_templates() -> PathBuf {
    selfplay_unit_templates_in(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."))
}

/// The tracked unit is a template; render it the way the bundle does.
fn rendered_selfplay_unit(unit: &str) -> String {
    let template = selfplay_unit_templates().join(format!("{unit}.in"));
    let contents = fs::read_to_string(&template)
        .unwrap_or_else(|error| panic!("{}: {error}", template.display()));
    jope_paths::systemd::render(&contents, &unit_roots(), &template).unwrap()
}

#[test]
fn every_tracked_systemd_unit_satisfies_the_runtime_boundary() {
    let systemd = selfplay_unit_templates();
    let mut seen = 0;
    for entry in fs::read_dir(&systemd).unwrap() {
        let path = entry.unwrap().path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let Some(unit) = name.strip_suffix(".in") else {
            continue;
        };
        if !unit.ends_with(".service") || is_host_installed_unit(&path) {
            continue;
        }
        seen += 1;
        validate_systemd_unit(&rendered_selfplay_unit(unit))
            .unwrap_or_else(|error| panic!("{}: {error:#}", path.display()));
    }
    assert!(seen > 0, "no unit templates under {}", systemd.display());
}

/// The Python removal P1 (ADR-0006) archived the workspace-sync templates.
/// Rendering must not need them, and a copy restored with
/// `git checkout 8a55c4fef -- <path>` must still stay out of the bundle.
#[test]
fn host_installed_units_are_optional_and_never_rendered_into_the_bundle() {
    let temporary = tempfile::tempdir().unwrap();
    let source = temporary.path().join("source");
    let templates = selfplay_unit_templates_in(&source);
    fs::create_dir_all(&templates).unwrap();
    fs::write(
        templates.join("jope-selfplay-gc.service.in"),
        "[Service]\nExecStart=@JOPE_BUNDLE@/jope_selfplay_gc\n",
    )
    .unwrap();
    let rendered_names = |dir: &Path| {
        let mut names = fs::read_dir(dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        names.sort();
        names
    };

    let absent = render_unit_templates(&source, &temporary.path().join("absent")).unwrap();
    assert_eq!(rendered_names(&absent), ["jope-selfplay-gc.service"]);

    for unit in HOST_INSTALLED_UNITS {
        let text = if unit.ends_with(".timer") {
            "[Timer]\nUnit=jope-workspace-sync.service\n"
        } else {
            "[Service]\nExecStart=/usr/bin/python3 @JOPE_STATE@/scripts/workspace_sync.py sync\n"
        };
        fs::write(templates.join(format!("{unit}.in")), text).unwrap();
    }
    let restored = render_unit_templates(&source, &temporary.path().join("restored")).unwrap();
    assert_eq!(rendered_names(&restored), ["jope-selfplay-gc.service"]);
}

#[test]
fn manifest_python_role_cannot_expand_the_allowlist() {
    assert!(
        validate_manifest_python_target("runtime/scripts/queue.py", "pytorch_training").is_err()
    );
    validate_manifest_python_target(
        "pytorch/training/prime_policy_trainer.py",
        "pytorch_training",
    )
    .unwrap();
}

#[test]
fn content_identity_is_ordered_and_hash_sensitive() {
    let rows = vec![InventoryRow {
        source: "/irrelevant/source".to_owned(),
        target: "runtime/bin/queue".to_owned(),
        sha256: "a".repeat(64),
        bytes: 42,
        python_role: None,
    }];
    let first = inventory_identity(&rows).unwrap();
    let mut changed_source = rows.clone();
    changed_source[0].source = "/elsewhere/same-content".to_owned();
    assert_eq!(first, inventory_identity(&changed_source).unwrap());
    let mut changed_hash = rows;
    changed_hash[0].sha256 = "b".repeat(64);
    assert_ne!(first, inventory_identity(&changed_hash).unwrap());
}

#[test]
fn declarative_service_manifest_rejects_split_inventory() {
    let valid = ServiceManifest {
        schema: "JopeSelfplayServiceManifest.v3".to_owned(),
        binaries: vec!["queue".to_owned()],
        services: vec![ServiceRow {
            unit: "queue.service".to_owned(),
            binary: "queue".to_owned(),
            owner: "submission".to_owned(),
            active: true,
        }],
    };
    validate_service_manifest(&valid).unwrap();
    let mut invalid = valid;
    invalid.services[0].binary = "missing".to_owned();
    assert!(validate_service_manifest(&invalid).is_err());
}

#[test]
fn recovery_manifest_rejects_missing_reporting_or_active_expansion() {
    let mut manifest = ServiceManifest {
        schema: "JopeSelfplayServiceManifest.v3".to_owned(),
        binaries: vec![
            "jope_reporting".to_owned(),
            "jope_selfplay_expansion_driver".to_owned(),
        ],
        services: vec![ServiceRow {
            unit: "jope-selfplay-expansion.service".to_owned(),
            binary: "jope_selfplay_expansion_driver".to_owned(),
            owner: "encoder_v2_identity_gate_inactive".to_owned(),
            active: false,
        }],
    };
    assert!(validate_recovery_service_manifest(&manifest).is_err());

    manifest.services.push(ServiceRow {
        unit: "jope-reporting.service".to_owned(),
        binary: "jope_reporting".to_owned(),
        owner: "recovery_reporting".to_owned(),
        active: true,
    });
    validate_recovery_service_manifest(&manifest).unwrap();

    manifest.services[0].active = true;
    assert!(validate_recovery_service_manifest(&manifest).is_err());
}

#[test]
fn rollback_identity_mismatch_is_rejected_before_activation() {
    let temporary = tempfile::tempdir().unwrap();
    let manifest = json!({"bundle_sha256": "b".repeat(64)});
    assert!(
        verify_rollback_candidate(temporary.path(), &manifest, ROLLBACK_BUNDLE_SHA256).is_err()
    );
}

#[test]
fn superseded_staging_bundle_is_verifiable_but_never_activatable() {
    assert!(validate_activation_target(
        ActivationOperationV1::Activate,
        SUPERSEDED_STAGING_BUNDLE_SHA256
    )
    .is_err());
    assert!(validate_activation_target(
        ActivationOperationV1::Rollback,
        SUPERSEDED_STAGING_BUNDLE_SHA256
    )
    .is_err());
    validate_activation_target(ActivationOperationV1::Rollback, ROLLBACK_BUNDLE_SHA256).unwrap();
}

#[test]
fn rollback_rebase_is_exactly_the_seven_working_directory_replacements() {
    assert_eq!(ROLLBACK_REBASE_UNITS.len(), 7);
    assert_eq!(
        ROLLBACK_REBASE_UNITS.iter().collect::<BTreeSet<_>>().len(),
        7
    );
    let before = format!(
            "[Unit]\nDescription=unchanged\n[Service]\nWorkingDirectory={rollback_working_directory_from_str}\nExecStart=/runtime/bin/tool --unchanged\n", rollback_working_directory_from_str = rollback_working_directory_from_str()
        );
    let reviewed = format!(
            "[Unit]\nDescription=reviewed-but-not-copied\n[Service]\nWorkingDirectory={immutable_flywheel_root_str}\nExecStart=/different/tool\n", immutable_flywheel_root_str = immutable_flywheel_root_str()
        );
    for unit in ROLLBACK_REBASE_UNITS {
        let after = transform_rollback_unit(unit, before.as_bytes(), reviewed.as_bytes())
            .expect("strict unit transformation");
        let expected = before.replacen(
            &format!(
                "WorkingDirectory={rollback_working_directory_from_str}",
                rollback_working_directory_from_str = rollback_working_directory_from_str()
            ),
            &format!(
                "WorkingDirectory={immutable_flywheel_root_str}",
                immutable_flywheel_root_str = immutable_flywheel_root_str()
            ),
            1,
        );
        assert_eq!(after, expected.as_bytes());
        assert!(String::from_utf8(after)
            .unwrap()
            .contains("ExecStart=/runtime/bin/tool --unchanged"));
    }

    let no_source_line = b"[Service]\nExecStart=/runtime/bin/tool\n";
    assert!(transform_rollback_unit(
        ROLLBACK_REBASE_UNITS[0],
        no_source_line,
        reviewed.as_bytes()
    )
    .is_err());
}

#[test]
fn activation_retains_all_available_evaluators_and_indexes_missing_history() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path();
    let bundles = root.join("runtime-bundles");
    let active = "a".repeat(64);
    let previous = "b".repeat(64);
    for (bundle, bytes) in [
        (&active, b"active".as_slice()),
        (&previous, b"previous".as_slice()),
    ] {
        let evaluator = bundles.join(bundle).join("runtime/bin/corpus_eval");
        fs::create_dir_all(evaluator.parent().unwrap()).unwrap();
        fs::write(&evaluator, bytes).unwrap();
    }
    std::os::unix::fs::symlink(&active, bundles.join("current")).unwrap();
    let missing = "c".repeat(64);
    atomic_json(
        &root.join("ops/causal-pair-repair-state.v1.json"),
        &json!({
            "schema": "CausalPairRepairExecutorState.v1",
            "entries": {"pair": {"original_evaluator_sha256": missing}},
        }),
    )
    .unwrap();
    let index = retain_evaluators(&bundles, &active).unwrap();
    let replayed = retain_evaluators(&bundles, &active).unwrap();
    assert_eq!(replayed, index);
    assert_eq!(index["schema"], "JopeEvaluatorRetentionIndex.v1");
    assert_eq!(index["active_bundle_sha256"], active);
    assert_eq!(index["prepared_for_bundle_sha256"], active);
    assert_eq!(index["all_referenced_evaluators_retained"], false);
    assert_eq!(index["unavailable_referenced_evaluator_sha256"][0], missing);
    for (bundle, bytes) in [
        (&active, b"active".as_slice()),
        (&previous, b"previous".as_slice()),
    ] {
        let digest = sha256_bytes(bytes);
        let source = bundles.join(bundle).join("runtime/bin/corpus_eval");
        let retained = root
            .join("training/evaluator-vault")
            .join(digest)
            .join("corpus_eval");
        assert_eq!(
            sha256_file(&retained).unwrap(),
            sha256_file(&source).unwrap()
        );
        assert_eq!(
            source.metadata().unwrap().ino(),
            retained.metadata().unwrap().ino()
        );
    }
    assert!(root
        .join("training/evaluator-retention-index.v1.json")
        .is_file());
}
