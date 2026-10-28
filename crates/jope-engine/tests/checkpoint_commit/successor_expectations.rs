//! Identity-expectation regressions using real checkpoint files and a zero-update journal.
//! This fixture initializes/serializes CPU tensors; it never executes an optimizer update.
use super::*;

fn untrained_checkpoint() -> Fixture {
    tch::set_num_threads(1);
    let base = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/task-evidence/P6/checkpoint-expectations");
    fs::create_dir_all(&base).unwrap();
    let dir = tempfile::tempdir_in(base).unwrap();
    let root = dir.path().canonicalize().unwrap();
    let run = root.join("run");
    let launch = run.join("launch-0");
    fs::create_dir_all(&launch).unwrap();
    let vs = nn::VarStore::new(Device::Cpu);
    let w = vs.root().var("w", &[4], nn::Init::Const(0.5));
    let mut optimizer = AdamW::new(1e-2, 0.01);
    optimizer
        .moments
        .insert("w".into(), (w.zeros_like(), w.zeros_like()));
    let journal = Journal::create(
        &launch.join("consumption-rank-0.jsonl"),
        Header {
            schema: consumption::SCHEMA.into(),
            rank: 0,
            world: 1,
            microbatch: MICRO,
            global_batch: MICRO,
            accumulation: 1,
            seed: SEED,
            sampler_id: "fixture".into(),
            rng: consumption::RNG_ID.into(),
            sampling_offset: 0,
            initial_optimizer_step: 0,
            target_optimizer_step: 10,
            corpus_sha256: CORPUS.into(),
            replay_policy_sha256: None,
            resume_state_sha256: None,
        },
    )
    .unwrap();
    let mut fixture = Fixture {
        _dir: dir,
        root,
        run,
        launch,
        vs,
        w,
        optimizer,
        journal,
    };
    write_zero_checkpoint(&mut fixture);
    assert_eq!(fixture.optimizer.step, 0);
    fixture
}
fn write_zero_checkpoint(fixture: &mut Fixture) {
    let binding = fixture.journal.sync().unwrap();
    let fixture: &Fixture = fixture;
    let journal = json!({"path":fixture.journal.path().strip_prefix(&fixture.root).unwrap(), "binding":binding});
    let receipt = fixture.launch.join("checkpoint-0-rank-0.json");
    let parameter = fixture.parameter_sha256();
    write_json(
        &receipt,
        &json!({"rank":0,"step":0,"parameter_sha256":parameter,
        "consumption_journal":journal}),
    )
    .unwrap();
    // Publish an initialized checkpoint through the production writer. There is no applied
    // update and no advancing journal commit to fabricate or acknowledge at step zero.
    checkpoint_commit::commit(&fixture.root, &fixture.run, 0, |staging| {
        fixture.vs.save(staging.join("model.ot"))?;
        fixture.optimizer.save(&staging.join("optimizer.ot"))?;
        write_json(&staging.join("config.json"), &Config::regression(8))
    }, |staging| {
        let digest = |name: &str| file_hash(&staging.join(name));
        Ok(json!({"schema":"parallel.checkpoint.v1","step":0,"world":1,
            "global_batch":MICRO,"model_sha256":digest("model.ot")?,
            "optimizer_sha256":digest("optimizer.ot")?,"optimizer_metadata_sha256":digest("optimizer.json")?,
            "config_sha256":digest("config.json")?,"corpus_sha256":CORPUS,"vocabulary_sha256":VOCABULARY,
            "parameter_sha256":parameter,"seed":SEED,"rng":"fixture counter rng","sampler":"fixture sampler",
            "sampler_id":"fixture","scheduler":{"type":"constant","learning_rate":fixture.optimizer.lr},
            "initialization":{"kind":"fresh"},"autocast_cache_lifetime":"scoped_v2",
            "ranking_backward":"dense_pairwise_v2","runtime_objective":"huber_log1p_seconds_v2",
            "execution_partition":training_partition(1,MICRO,MICRO)?,
            "committed_exposure_by_rank":[{"rank":0,"counts":{
                "program_presentations":0,"comparison_groups":0,"actual_fresh_program_presentations":0,
                "groups_with_actual_fresh_program":0,"designated_fresh_groups":0,"designated_broad_groups":0},
                "consumption_journal":journal}],"checkpoint_format":checkpoint_commit::FORMAT,
            "required_objects":checkpoint_commit::required_object_names(),"objective":objective::identity(),
            "objective_sha256":objective::identity_sha256(),"rank_receipts":[{"rank":0,
                "path":receipt.strip_prefix(&fixture.root)?,"sha256":file_hash(&receipt)?}]}))
    }, &mut checkpoint_commit::no_stop).unwrap();
}

fn edit_state(fixture: &Fixture, edit: impl FnOnce(&mut Value)) {
    let path = fixture.checkpoint_dir(0).join("state.json");
    let mut state: Value = read_json(&path).unwrap();
    edit(&mut state);
    write_json(&path, &state).unwrap();
    let mut latest: Value = read_json(&fixture.run.join("latest.json")).unwrap();
    latest["state_sha256"] = json!(file_hash(&path).unwrap());
    write_json(&fixture.run.join("latest.json"), &latest).unwrap();
}
fn make_legacy(fixture: &Fixture) {
    edit_state(fixture, |state| {
        state.as_object_mut().unwrap().remove("checkpoint_format");
    });
}
fn make_successor(fixture: &Fixture) {
    // Same v3 binding/prefix fixture as the existing checkpoint tests, at zero updates.
    edit_state(fixture, |state| {
        let lineage = json!({"schema":"jope.policy-lineage.v1","id":"fixture-lineage"});
        let intent = json!({"schema":"jope.training-intent.v1","id":"fixture-intent"});
        let journal = state["committed_exposure_by_rank"][0]["consumption_journal"].clone();
        state["checkpoint_format"] = json!(checkpoint_commit::SUCCESSOR_FORMAT);
        state["lineage"] = json!({"schema":lineage["schema"],"value":lineage,
            "sha256":hash(&serde_json::to_vec(&lineage).unwrap())});
        state["intent"] = json!({"schema":intent["schema"],"value":intent,
            "sha256":hash(&serde_json::to_vec(&intent).unwrap())});
        state["attempt_id"] = json!("attempt-0");
        state["window_id"] = json!("window-0");
        state["applied_boundary"] = state["step"].clone();
        state["journal_prefixes"] = json!([{"rank":0,"attempt_id":"attempt-0","window_id":"window-0",
            "applied_through":state["step"],"path":journal["path"],"binding":journal["binding"]}]);
        state["closure_objects"] = json!([{"path":journal["path"],
            "sha256":file_hash(&fixture.root.join(journal["path"].as_str().unwrap())).unwrap()}]);
    });
}
fn refused_by_both_loaders(fixture: &Fixture, expected: &Expectation) {
    let directory =
        checkpoint_commit::verify_dir(&fixture.root, &fixture.checkpoint_dir(0), 0, expected);
    let latest = load_latest(&fixture.root, &fixture.run, expected);
    assert!(directory.is_err() && latest.is_err(),
        "requested successor identities were silently ignored: directory={directory:?}, latest={latest:?}");
}

#[test]
fn expected_successor_identities_refuse_v2_and_legacy_instead_of_ignoring_pins() {
    for legacy in [false, true] {
        let fixture = untrained_checkpoint();
        if legacy {
            make_legacy(&fixture);
        }
        for expected in [
            Expectation {
                lineage_sha256: Some(CORPUS),
                ..Expectation::default()
            },
            Expectation {
                intent_sha256: Some(VOCABULARY),
                ..Expectation::default()
            },
            Expectation {
                attempt_id: Some("attempt-0"),
                ..Expectation::default()
            },
            Expectation {
                window_id: Some("window-0"),
                ..Expectation::default()
            },
        ] {
            refused_by_both_loaders(&fixture, &expected);
        }
    }
}
#[test]
fn no_successor_pins_preserves_v2_and_legacy_loader_behavior() {
    for legacy in [false, true] {
        let fixture = untrained_checkpoint();
        if legacy {
            make_legacy(&fixture);
        }
        let expected = if legacy {
            Format::Legacy
        } else {
            Format::Complete
        };
        assert_eq!(
            load_latest(&fixture.root, &fixture.run, &Expectation::default())
                .unwrap()
                .format,
            expected
        );
        assert_eq!(
            checkpoint_commit::verify_dir(
                &fixture.root,
                &fixture.checkpoint_dir(0),
                0,
                &Expectation::default()
            )
            .unwrap()
            .format,
            expected
        );
    }
}
#[test]
fn matching_successor_pins_load_v3_and_each_mismatch_refuses() {
    let fixture = untrained_checkpoint();
    make_successor(&fixture);
    let state: Value = read_json(&fixture.checkpoint_dir(0).join("state.json")).unwrap();
    let lineage = state["lineage"]["sha256"].as_str().unwrap();
    let intent = state["intent"]["sha256"].as_str().unwrap();
    for expected in [
        Expectation {
            lineage_sha256: Some(lineage),
            ..Expectation::default()
        },
        Expectation {
            intent_sha256: Some(intent),
            ..Expectation::default()
        },
        Expectation {
            attempt_id: Some("attempt-0"),
            ..Expectation::default()
        },
        Expectation {
            window_id: Some("window-0"),
            ..Expectation::default()
        },
        Expectation {
            lineage_sha256: Some(lineage),
            intent_sha256: Some(intent),
            attempt_id: Some("attempt-0"),
            window_id: Some("window-0"),
            ..Expectation::default()
        },
    ] {
        assert_eq!(
            load_latest(&fixture.root, &fixture.run, &expected)
                .unwrap()
                .format,
            Format::Successor
        );
        assert_eq!(
            checkpoint_commit::verify_dir(&fixture.root, &fixture.checkpoint_dir(0), 0, &expected)
                .unwrap()
                .format,
            Format::Successor
        );
    }
    for expected in [
        Expectation {
            lineage_sha256: Some(CORPUS),
            ..Expectation::default()
        },
        Expectation {
            intent_sha256: Some(VOCABULARY),
            ..Expectation::default()
        },
        Expectation {
            attempt_id: Some("another-attempt"),
            ..Expectation::default()
        },
        Expectation {
            window_id: Some("another-window"),
            ..Expectation::default()
        },
    ] {
        refused_by_both_loaders(&fixture, &expected);
    }
}

#[test]
fn unknown_format_retains_its_existing_refusal_with_or_without_identity_pins() {
    let fixture = untrained_checkpoint();
    edit_state(&fixture, |state| {
        state["checkpoint_format"] = json!("unknown-checkpoint-format")
    });
    for expected in [
        Expectation::default(),
        Expectation {
            intent_sha256: Some(VOCABULARY),
            ..Expectation::default()
        },
    ] {
        let error = load_latest(&fixture.root, &fixture.run, &expected).unwrap_err();
        assert!(
            error.to_string().contains("unknown checkpoint format"),
            "{error}"
        );
    }
}
