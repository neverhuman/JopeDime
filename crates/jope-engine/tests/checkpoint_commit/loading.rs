use super::*;

/// Rewrite the published manifest and re-bind latest.json to it, so only the
/// manifest's content (not a digest mismatch) is under test.
fn rewrite_state(f: &Fixture, edit: impl FnOnce(&mut Value)) {
    let path = f.checkpoint_dir(2).join("state.json");
    let mut state: Value = read_json(&path).unwrap();
    edit(&mut state);
    write_json(&path, &state).unwrap();
    rebind(f);
}

fn rebind(f: &Fixture) {
    let mut latest: Value = read_json(&f.run.join("latest.json")).unwrap();
    latest["state_sha256"] = json!(file_hash(&f.checkpoint_dir(2).join("state.json")).unwrap());
    write_json(&f.run.join("latest.json"), &latest).unwrap();
}

fn committed_fixture() -> Fixture {
    let mut f = fixture();
    f.train_to(2);
    f.commit(&mut checkpoint_commit::no_stop).unwrap();
    f
}

#[test]
fn the_loader_refuses_torn_incomplete_stale_or_inconsistent_checkpoints() {
    let f = committed_fixture();
    let ok = f.load().unwrap();
    assert_eq!((ok.step, ok.format), (2, Format::Complete));
    type Mutation = Box<dyn Fn(&Fixture)>;
    let cases: Vec<(&str, Mutation)> = vec![
        (
            "optimizer.ot missing",
            Box::new(|f| fs::remove_file(f.checkpoint_dir(2).join("optimizer.ot")).unwrap()),
        ),
        (
            "optimizer.json missing",
            Box::new(|f| fs::remove_file(f.checkpoint_dir(2).join("optimizer.json")).unwrap()),
        ),
        (
            "model.ot truncated",
            Box::new(|f| {
                let path = f.checkpoint_dir(2).join("model.ot");
                let bytes = fs::read(&path).unwrap();
                fs::write(&path, &bytes[..bytes.len() / 2]).unwrap();
            }),
        ),
        (
            "state.json torn",
            Box::new(|f| {
                let path = f.checkpoint_dir(2).join("state.json");
                let bytes = fs::read(&path).unwrap();
                fs::write(&path, &bytes[..bytes.len() / 2]).unwrap();
                rebind(f);
            }),
        ),
        (
            "latest names another step",
            Box::new(|f| {
                let mut latest: Value = read_json(&f.run.join("latest.json")).unwrap();
                latest["step"] = json!(3);
                write_json(&f.run.join("latest.json"), &latest).unwrap();
            }),
        ),
        (
            "latest digest stale",
            Box::new(|f| {
                let mut latest: Value = read_json(&f.run.join("latest.json")).unwrap();
                latest["state_sha256"] = json!("c".repeat(64));
                write_json(&f.run.join("latest.json"), &latest).unwrap();
            }),
        ),
        (
            "rng identity absent",
            Box::new(|f| rewrite_state(f, |s| drop(s.as_object_mut().unwrap().remove("rng")))),
        ),
        (
            "seed absent",
            Box::new(|f| rewrite_state(f, |s| drop(s.as_object_mut().unwrap().remove("seed")))),
        ),
        (
            "optimizer step inconsistent",
            Box::new(|f| {
                let path = f.checkpoint_dir(2).join("optimizer.json");
                let mut optimizer: Value = read_json(&path).unwrap();
                optimizer["step"] = json!(3);
                write_json(&path, &optimizer).unwrap();
                let digest = file_hash(&path).unwrap();
                rewrite_state(f, |s| s["optimizer_metadata_sha256"] = json!(digest));
            }),
        ),
        (
            "exposure counters inconsistent",
            Box::new(|f| {
                rewrite_state(f, |s| {
                    let counts = &mut s["committed_exposure_by_rank"][0]["counts"];
                    counts["program_presentations"] = json!(24);
                    counts["comparison_groups"] = json!(3);
                    counts["designated_broad_groups"] = json!(3);
                })
            }),
        ),
        (
            "execution partition inconsistent",
            Box::new(|f| rewrite_state(f, |s| s["execution_partition"]["microbatch"] = json!(16))),
        ),
        (
            "rank receipt changed",
            Box::new(|f| {
                let path = f.launch.join("checkpoint-2-rank-0.json");
                let mut receipt: Value = read_json(&path).unwrap();
                receipt["parameter_sha256"] = json!("d".repeat(64));
                write_json(&path, &receipt).unwrap();
            }),
        ),
        (
            "consumption journal prefix changed",
            Box::new(|f| {
                let path = f.journal.path();
                let mut bytes = fs::read(path).unwrap();
                bytes[50] ^= 1;
                fs::write(path, bytes).unwrap();
            }),
        ),
        (
            "unknown format",
            Box::new(|f| rewrite_state(f, |s| s["checkpoint_format"] = json!("v3"))),
        ),
        (
            "objective digest inconsistent",
            Box::new(|f| {
                rewrite_state(f, |s| {
                    s["objective"]["coefficients"]["ranking_bt"] = json!(0.5)
                })
            }),
        ),
        (
            "objective identity absent",
            Box::new(|f| {
                rewrite_state(f, |s| {
                    let fields = s.as_object_mut().unwrap();
                    fields.remove("objective");
                    fields.remove("objective_sha256");
                })
            }),
        ),
        (
            "legacy where complete is required",
            Box::new(|f| {
                rewrite_state(f, |s| {
                    drop(s.as_object_mut().unwrap().remove("checkpoint_format"))
                })
            }),
        ),
        (
            "symlinked model",
            Box::new(|f| {
                let path = f.checkpoint_dir(2).join("model.ot");
                let copy = f.run.join("model-copy.ot");
                fs::copy(&path, &copy).unwrap();
                fs::remove_file(&path).unwrap();
                std::os::unix::fs::symlink(&copy, &path).unwrap();
            }),
        ),
    ];
    for (case, mutate) in cases {
        let f = committed_fixture();
        mutate(&f);
        assert!(f.load().is_err(), "{case}: accepted");
    }
    // Stale inputs and a different topology are refused by expectation.
    for expect in [
        Expectation {
            corpus_sha256: Some(VOCABULARY),
            ..Expectation::default()
        },
        Expectation {
            vocabulary_sha256: Some(CORPUS),
            ..Expectation::default()
        },
        Expectation {
            world: Some(2),
            ..Expectation::default()
        },
        Expectation {
            global_batch: Some(16),
            ..Expectation::default()
        },
    ] {
        assert!(load_latest(&f.root, &f.run, &expect).is_err());
    }
    // A legacy manifest is still loadable when completeness is not required,
    // and is reported as legacy rather than complete.
    let legacy = committed_fixture();
    rewrite_state(&legacy, |s| {
        drop(s.as_object_mut().unwrap().remove("checkpoint_format"))
    });
    let loaded = load_latest(&legacy.root, &legacy.run, &Expectation::default()).unwrap();
    assert_eq!(loaded.format, Format::Legacy);
}

#[test]
fn a_complete_checkpoint_reloads_the_committed_state() {
    let mut f = committed_fixture();
    let loaded = f.load().unwrap();
    let mut vs = nn::VarStore::new(Device::Cpu);
    let w = vs.root().var("w", &[4], nn::Init::Const(0.0));
    vs.load(loaded.dir.join("model.ot")).unwrap();
    let mut optimizer = AdamW::load(&loaded.dir.join("optimizer.ot"), Device::Cpu).unwrap();
    assert_eq!(optimizer.step, f.optimizer.step);
    assert!(w.equal(&f.w));
    for (name, (m, v)) in &f.optimizer.moments {
        let (lm, lv) = &optimizer.moments[name];
        assert!(m.equal(lm) && v.equal(lv), "{name}");
    }
    // The next update from the reloaded state equals the uninterrupted one.
    let target = Tensor::from_slice(&[0.1f32, -0.2, 0.3, 0.0]);
    (&f.w - &target).square().sum(Kind::Float).backward();
    f.optimizer.update_reference(&f.vs).unwrap();
    (&w - &target).square().sum(Kind::Float).backward();
    optimizer.update_reference(&vs).unwrap();
    assert!(w.equal(&f.w));
    assert_eq!(optimizer.step, f.optimizer.step);
    let _: &Path = &loaded.dir;
}

#[test]
fn v3_loader_requires_all_rank_prefixes_and_bound_lineage_and_intent() {
    let good = committed_fixture();
    rewrite_state(&good, |state| {
        let lineage = json!({"schema":"jope.policy-lineage.v1","id":"fixture-lineage"});
        let intent = json!({"schema":"jope.training-intent.v1","id":"fixture-intent"});
        let journal = state["committed_exposure_by_rank"][0]["consumption_journal"].clone();
        state["checkpoint_format"] = json!(checkpoint_commit::SUCCESSOR_FORMAT);
        state["lineage"] = json!({"schema":lineage["schema"],"value":lineage,"sha256":hash(&serde_json::to_vec(&lineage).unwrap())});
        state["intent"] = json!({"schema":intent["schema"],"value":intent,"sha256":hash(&serde_json::to_vec(&intent).unwrap())});
        state["attempt_id"] = json!("attempt-0");
        state["window_id"] = json!("window-0");
        state["applied_boundary"] = state["step"].clone();
        state["journal_prefixes"] = json!([{"rank":0,"attempt_id":"attempt-0","window_id":"window-0",
            "applied_through":state["step"],"path":journal["path"],"binding":journal["binding"]}]);
        state["closure_objects"] = json!([{"path":journal["path"],"sha256":file_hash(&good.root.join(journal["path"].as_str().unwrap())).unwrap()}]);
    });
    let loaded = good.load().unwrap();
    assert_eq!(loaded.format, Format::Successor);
    assert!(
        load_latest(
            &good.root,
            &good.run,
            &Expectation {
                intent_sha256: Some(&"f".repeat(64)),
                ..Expectation::default()
            }
        )
        .is_err(),
        "caller-pinned intent mismatch must refuse exact resume"
    );

    let missing_rank = committed_fixture();
    rewrite_state(&missing_rank, |state| {
        state["checkpoint_format"] = json!(checkpoint_commit::SUCCESSOR_FORMAT);
        state["lineage"] = json!({"schema":"jope.policy-lineage.v1","value":{"schema":"jope.policy-lineage.v1"},"sha256":hash(br#"{"schema":"jope.policy-lineage.v1"}"#)});
        state["intent"] = json!({"schema":"jope.training-intent.v1","value":{"schema":"jope.training-intent.v1"},"sha256":hash(br#"{"schema":"jope.training-intent.v1"}"#)});
        state["attempt_id"] = json!("attempt-0");
        state["window_id"] = json!("window-0");
        state["applied_boundary"] = state["step"].clone();
        state["journal_prefixes"] = json!([]);
        state["closure_objects"] = json!([{"path":"unused","sha256":"0".repeat(64)}]);
    });
    assert!(missing_rank.load().is_err(), "missing rank must not load");
}

#[test]
fn a_stale_writer_fence_cannot_publish_latest() {
    let mut f = fixture();
    f.train_to(2);
    let mut stale = || anyhow::bail!("stale attempt");
    assert!(f
        .commit_with_fence(&|_| {}, false, &mut stale, &mut checkpoint_commit::no_stop)
        .is_err());
    assert!(!f.checkpoint_dir(2).exists());
    assert!(!f.run.join("latest.json").exists());

    let mut f = fixture();
    f.train_to(2);
    let mut checks = 0;
    let mut superseded = || {
        checks += 1;
        if checks == 1 {
            Ok(())
        } else {
            anyhow::bail!("stale attempt")
        }
    };
    assert!(f
        .commit_with_fence(
            &|_| {},
            false,
            &mut superseded,
            &mut checkpoint_commit::no_stop
        )
        .is_err());
    assert!(
        f.checkpoint_dir(2).exists(),
        "rename completed before the fence changed"
    );
    assert!(
        !f.run.join("latest.json").exists(),
        "stale writer advanced latest"
    );
    checkpoint_recovery::reconcile(&f.root, &f.run).unwrap();
    assert!(
        !f.checkpoint_dir(2).exists(),
        "unpublished state was not quarantined"
    );
}
