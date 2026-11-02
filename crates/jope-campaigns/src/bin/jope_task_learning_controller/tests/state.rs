#[test]
fn causal_recipe_requires_32_exact_pairs_and_is_shadow_only() {
    let directory = tempdir().unwrap();
    let packed_root = directory.path().join("packed");
    fs::create_dir_all(&packed_root).unwrap();
    let pair_index = packed_root.join("causal_pairs.jsonl");
    let accepted = Sha256Digest::parse("a".repeat(64)).unwrap();
    let rows = (0..32)
        .map(|index| {
            serde_json::to_string(&json!({
                "pair_id": format!("{index:064x}"),
                "task": "binary",
                "accepted_parent_checkpoint_sha256": accepted,
            }))
            .unwrap()
        })
        .collect::<Vec<_>>();
    fs::write(&pair_index, format!("{}\n", rows[..31].join("\n"))).unwrap();
    let packed_manifest = packed_root.join("manifest.json");
    let write_manifest = || {
        atomic_json(
            &packed_manifest,
            &json!({
                "causal_pair_index": "causal_pairs.jsonl",
                "files": {
                    "causal_pairs.jsonl": {
                        "sha256": sha256_file(&pair_index).unwrap(),
                    }
                }
            }),
        )
        .unwrap();
    };
    write_manifest();
    let corpus = CorpusPointer {
        watermark: 50_000,
        manifest: directory.path().join("unused-corpus.json"),
        manifest_sha256: String::new(),
        packed_manifest: packed_manifest.clone(),
        packed_manifest_sha256: String::new(),
    };

    let (recipe, _) =
        trainer_recipe(&corpus, Task::Binary, TrainerMode::Shadow, &accepted, 0, 0).unwrap();
    assert_eq!(recipe, "a");

    fs::write(&pair_index, format!("{}\n", rows.join("\n"))).unwrap();
    write_manifest();
    let (recipe, metadata) =
        trainer_recipe(&corpus, Task::Binary, TrainerMode::Shadow, &accepted, 0, 0).unwrap();
    assert_eq!(recipe, "d");
    assert_eq!(metadata["complete_parent_pairs"], 32);
    assert_eq!(metadata["broad_replay_fraction"], 0.75);
    assert_eq!(metadata["exact_causal_pair_fraction"], 0.25);

    let (recipe, metadata) = trainer_recipe(
        &corpus,
        Task::Binary,
        TrainerMode::Production,
        &accepted,
        0,
        0,
    )
    .unwrap();
    assert_eq!(recipe, "a");
    assert_eq!(metadata["exact_causal_pair_fraction"], 0.0);

    let evidence_index = packed_root.join("recipe_e_evidence.jsonl");
    fs::write(
        &evidence_index,
        format!(
            "{}\n{}\n",
            serde_json::to_string(&json!({
                "ordinal": 0,
                "task": "binary",
                "lane": "completed_descendant_trajectories",
                "descendant_credit_receipt_sha256": "b".repeat(64),
                "descendant_horizon": 4,
            }))
            .unwrap(),
            serde_json::to_string(&json!({
                "ordinal": 1,
                "task": "binary",
                "lane": "validity_instability_hard_negatives",
            }))
            .unwrap(),
        ),
    )
    .unwrap();
    atomic_json(
        &packed_manifest,
        &json!({
            "causal_pair_index": "causal_pairs.jsonl",
            "recipe_e_evidence_index": "recipe_e_evidence.jsonl",
            "files": {
                "causal_pairs.jsonl": {
                    "sha256": sha256_file(&pair_index).unwrap(),
                },
                "recipe_e_evidence.jsonl": {
                    "sha256": sha256_file(&evidence_index).unwrap(),
                }
            }
        }),
    )
    .unwrap();
    let (recipe, metadata) =
        trainer_recipe(&corpus, Task::Binary, TrainerMode::Shadow, &accepted, 0, 0).unwrap();
    assert_eq!(recipe, "e");
    assert_eq!(metadata["completed_descendant_receipts"], 1);
    assert_eq!(metadata["hard_negative_rows"], 1);
    assert_eq!(metadata["benchmark_valid"], false);

    let (recipe, _) = trainer_recipe(
        &corpus,
        Task::Binary,
        TrainerMode::Shadow,
        &accepted,
        49_999,
        1,
    )
    .unwrap();
    assert_eq!(recipe, "d");
}

#[test]
fn changed_task_parent_resets_only_that_shadow_lineage() {
    let directory = tempdir().unwrap();
    let model = directory.path().join("shadow.model.pt");
    fs::write(&model, b"model").unwrap();
    let old = Sha256Digest::parse("a".repeat(64)).unwrap();
    let changed = Sha256Digest::parse("b".repeat(64)).unwrap();
    let stable = Sha256Digest::parse("c".repeat(64)).unwrap();
    let mut state = ControllerState::default();
    state.shadow_models.insert(
        Task::Regression,
        ModelRef {
            accepted_parent_checkpoint_sha256: old,
            checkpoint_sha256: changed.clone(),
            checkpoint_path: model.clone(),
        },
    );
    state.shadow_models.insert(
        Task::Binary,
        ModelRef {
            accepted_parent_checkpoint_sha256: stable.clone(),
            checkpoint_sha256: changed.clone(),
            checkpoint_path: model,
        },
    );
    let accepted = BTreeMap::from([
        (
            Task::Regression,
            AcceptedPointer {
                digest: changed,
                path: PathBuf::new(),
            },
        ),
        (
            Task::Binary,
            AcceptedPointer {
                digest: stable,
                path: PathBuf::new(),
            },
        ),
    ]);
    reset_changed_shadow_epochs(&mut state, &accepted);
    assert!(!state.shadow_models.contains_key(&Task::Regression));
    assert!(state.shadow_models.contains_key(&Task::Binary));
}
