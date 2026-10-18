#![cfg(feature = "neural")]
//! W6-T05: atomic checkpoint commit and complete-checkpoint loading (CPU only).
//!
//! A tiny CPU model, the CPU reference AdamW and the real consumption journal
//! produce real checkpoint objects; publication is stopped between each pair of
//! steps to show no interruption exposes a partial checkpoint. This proves the
//! local commit protocol, not GPU training, NCCL or remote publication (W8).

use jope_parallel::{
    checkpoint_recovery,
    manifest::{file_hash, hash, read_json, write_json},
    neural::{AdamW, Config},
    task_contract::training_partition,
    training::{
        checkpoint_commit::{self, load_latest, replace_json, Expectation, Format, Phase, PHASES},
        consumption::{self, Header, Journal},
        objective::{self, RowLabels},
    },
    training_exposure::Counts,
};
use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
};
use tch::{nn, Device, Kind, Tensor};

const CORPUS: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const VOCABULARY: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const SEED: u64 = 11;
const MICRO: usize = 8;

struct Fixture {
    _dir: tempfile::TempDir,
    root: PathBuf,
    run: PathBuf,
    launch: PathBuf,
    vs: nn::VarStore,
    w: Tensor,
    optimizer: AdamW,
    journal: Journal,
}

fn fixture() -> Fixture {
    tch::set_num_threads(1);
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().canonicalize().unwrap();
    let run = root.join("run");
    let launch = run.join("launch-0");
    fs::create_dir_all(&launch).unwrap();
    let vs = nn::VarStore::new(Device::Cpu);
    let w = vs.root().var("w", &[4], nn::Init::Const(0.5));
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
    Fixture {
        _dir: dir,
        root,
        run,
        launch,
        vs,
        w,
        optimizer: AdamW::new(1e-2, 0.01),
        journal,
    }
}

fn rows() -> (Vec<RowLabels>, Vec<usize>) {
    (0..MICRO)
        .map(|i| {
            (
                RowLabels {
                    record: i,
                    context: 0,
                    value: 1.0 / (1.0 + i as f32),
                    percentile: 1.0 - i as f32 / 7.0,
                    runtime: 0.1,
                },
                5,
            )
        })
        .unzip()
}

impl Fixture {
    fn train_to(&mut self, step: u64) {
        let (labels, lengths) = rows();
        for s in self.optimizer.step..step {
            let target = Tensor::from_slice(&[0.1f32, -0.2, 0.3, 0.0]);
            (&self.w - target).square().sum(Kind::Float).backward();
            let norm = self.w.grad().norm().double_value(&[]);
            self.journal
                .microbatch(
                    s,
                    0,
                    s * MICRO as u64,
                    consumption::dropout_seed(SEED, s, 0, 0),
                    &(0..MICRO).collect::<Vec<_>>(),
                    objective::supervision(&labels, &lengths).unwrap(),
                )
                .unwrap();
            self.optimizer.update_reference(&self.vs).unwrap();
            self.journal.applied(s, norm).unwrap();
        }
    }

    fn parameter_sha256(&self) -> String {
        let values: Vec<f32> = Vec::try_from(self.w.detach()).unwrap();
        hash(
            &values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
    }

    /// Commit the current optimizer step the way `training::checkpoint` does.
    fn commit(
        &mut self,
        stop: &mut dyn FnMut(Phase) -> anyhow::Result<()>,
    ) -> anyhow::Result<Value> {
        self.commit_with(&|_| {}, false, stop)
    }

    /// As [`Fixture::commit`], with one manifest edit or the optimizer omitted.
    fn commit_with(
        &mut self,
        edit: &dyn Fn(&mut Value),
        omit_optimizer: bool,
        stop: &mut dyn FnMut(Phase) -> anyhow::Result<()>,
    ) -> anyhow::Result<Value> {
        self.commit_with_fence(edit, omit_optimizer, &mut || Ok(()), stop)
    }

    fn commit_with_fence(
        &mut self,
        edit: &dyn Fn(&mut Value),
        omit_optimizer: bool,
        fence: &mut dyn FnMut() -> anyhow::Result<()>,
        stop: &mut dyn FnMut(Phase) -> anyhow::Result<()>,
    ) -> anyhow::Result<Value> {
        let step = self.optimizer.step;
        let binding = self.journal.sync()?;
        let journal_path = self.journal.path().strip_prefix(&self.root)?.to_path_buf();
        let consumption = json!({"path":journal_path,"binding":binding});
        let receipt = self.launch.join(format!("checkpoint-{step}-rank-0.json"));
        write_json(
            &receipt,
            &json!({"rank":0,"step":step,"parameter_sha256":self.parameter_sha256(),
                "consumption_journal":consumption}),
        )?;
        let receipt_entry = json!([{"rank":0,"path":receipt.strip_prefix(&self.root)?,
            "sha256":file_hash(&receipt)?}]);
        let parameter = self.parameter_sha256();
        let (vs, optimizer) = (&self.vs, &self.optimizer);
        let latest = checkpoint_commit::commit_fenced(
            &self.root,
            &self.run,
            step,
            |staging| {
                vs.save(staging.join("model.ot"))?;
                if !omit_optimizer {
                    optimizer.save(&staging.join("optimizer.ot"))?;
                }
                write_json(&staging.join("config.json"), &Config::regression(8))
            },
            |staging| {
                let digest = |name: &str| {
                    if omit_optimizer && name.starts_with("optimizer") {
                        Ok("0".repeat(64))
                    } else {
                        file_hash(&staging.join(name))
                    }
                };
                let counts = Counts {
                    program_presentations: step * MICRO as u64,
                    comparison_groups: step,
                    actual_fresh_program_presentations: 0,
                    groups_with_actual_fresh_program: 0,
                    designated_fresh_groups: 0,
                    designated_broad_groups: step,
                };
                let mut state = json!({"schema":"parallel.checkpoint.v1","step":step,"world":1,
                    "global_batch":MICRO,"model_sha256":digest("model.ot")?,
                    "optimizer_sha256":digest("optimizer.ot")?,
                    "optimizer_metadata_sha256":digest("optimizer.json")?,
                    "config_sha256":digest("config.json")?,"corpus_sha256":CORPUS,
                    "vocabulary_sha256":VOCABULARY,"parameter_sha256":parameter,"seed":SEED,
                    "rng":"fixture counter rng","sampler":"fixture sampler","sampler_id":"fixture",
                    "scheduler":{"type":"constant","learning_rate":optimizer.lr},
                    "initialization":{"kind":"fresh"},"autocast_cache_lifetime":"scoped_v2",
                    "ranking_backward":"dense_pairwise_v2","runtime_objective":"huber_log1p_seconds_v2",
                    "execution_partition":training_partition(1, MICRO, MICRO)?,
                    "committed_exposure_by_rank":[{"rank":0,"counts":counts,
                        "consumption_journal":consumption}],
                    "checkpoint_format":checkpoint_commit::FORMAT,
                    "required_objects":checkpoint_commit::required_object_names(),
                    "objective":objective::identity(),
                    "objective_sha256":objective::identity_sha256(),
                    "rank_receipts":receipt_entry});
                edit(&mut state);
                Ok(state)
            },
            fence,
            stop,
        )?;
        self.journal
            .commit(step, latest["state_sha256"].as_str().unwrap())?;
        Ok(latest)
    }

    fn checkpoint_dir(&self, step: u64) -> PathBuf {
        self.run.join(format!("checkpoints/step-{step}"))
    }

    fn load(&self) -> anyhow::Result<checkpoint_commit::Committed> {
        load_latest(
            &self.root,
            &self.run,
            &Expectation {
                corpus_sha256: Some(CORPUS),
                vocabulary_sha256: Some(VOCABULARY),
                world: Some(1),
                global_batch: Some(MICRO),
                require_complete: true,
            },
        )
    }
}

fn stop_at(target: Phase) -> impl FnMut(Phase) -> anyhow::Result<()> {
    move |phase| {
        if phase == target {
            anyhow::bail!("injected stop after {phase:?}");
        }
        Ok(())
    }
}

#[test]
fn a_stale_latest_temporary_from_a_killed_writer_cannot_block_publication() {
    // A writer killed between creating latest.json's temporary and renaming it
    // leaves `latest.partial.<pid>`. The previous publication reused that
    // fixed name with create_new, so a later writer with the same process id
    // (a restarted container, a recycled pid) failed every checkpoint.
    let mut f = fixture();
    f.train_to(2);
    let pid = std::process::id();
    fs::write(f.run.join(format!("latest.partial.{pid}")), b"{\"torn\":").unwrap();
    fs::write(f.run.join(format!("initialization.partial.{pid}")), b"{").unwrap();
    let latest = f.commit(&mut checkpoint_commit::no_stop).unwrap();
    assert_eq!(latest["step"], 2);
    assert_eq!(f.load().unwrap().step, 2);
    // The same holds for the run's initialization record.
    replace_json(&f.run.join("initialization.json"), &json!({"kind":"fresh"})).unwrap();
    assert_eq!(
        read_json::<Value>(&f.run.join("initialization.json")).unwrap(),
        json!({"kind":"fresh"})
    );
    // Stale temporaries are left in place (inert, never read), not deleted.
    assert!(f.run.join(format!("latest.partial.{pid}")).exists());
}

#[test]
fn a_stop_between_any_two_publication_steps_never_exposes_a_partial_checkpoint() {
    for phase in PHASES {
        let mut f = fixture();
        f.train_to(2);
        f.commit(&mut checkpoint_commit::no_stop).unwrap();
        f.train_to(4);
        let error = f.commit(&mut stop_at(phase)).unwrap_err();
        assert!(
            error.to_string().contains("injected stop"),
            "{phase:?}: {error}"
        );
        let published = f.load().unwrap();
        if phase == Phase::LatestPublished {
            // The publication itself completed; only the caller was stopped.
            assert_eq!(published.step, 4, "{phase:?}");
            let journal = fs::read(f.journal.path()).unwrap();
            let r = consumption::reconcile(&journal, Some(4)).unwrap();
            assert!(r.commit_event_missing && r.committed_through == 4);
            continue;
        }
        assert_eq!(
            published.step, 2,
            "{phase:?}: an interrupted publication was exposed"
        );
        if matches!(phase, Phase::Renamed | Phase::LatestStaged) {
            // The complete but unpublished directory is never overwritten.
            assert!(
                f.commit(&mut checkpoint_commit::no_stop).is_err(),
                "{phase:?}"
            );
        }
        // Stopped-rank recovery quarantines the partial work without deleting it.
        checkpoint_recovery::reconcile(&f.root, &f.run).unwrap();
        let quarantine = fs::read_dir(f.run.join("checkpoint-recovery"))
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let receipt: Value = read_json(&quarantine.join("receipt.json")).unwrap();
        assert_eq!(receipt["completed"], true, "{phase:?}");
        assert_eq!(receipt["moves"].as_array().unwrap().len(), 1, "{phase:?}");
        assert_eq!(f.load().unwrap().step, 2, "{phase:?}");
        // The re-executed publication commits the same step.
        f.commit(&mut checkpoint_commit::no_stop).unwrap();
        assert_eq!(f.load().unwrap().step, 4, "{phase:?}");
    }
}

#[test]
fn commit_refuses_to_expose_an_incomplete_or_inconsistent_checkpoint() {
    type Edit = Box<dyn Fn(&mut Value)>;
    let cases: Vec<(&str, bool, Edit)> = vec![
        ("missing optimizer", true, Box::new(|_| {})),
        (
            "rng absent",
            false,
            Box::new(|s| drop(s.as_object_mut().unwrap().remove("rng"))),
        ),
        (
            "counters inconsistent",
            false,
            Box::new(|s| {
                let counts = &mut s["committed_exposure_by_rank"][0]["counts"];
                counts["program_presentations"] = json!(16);
                counts["comparison_groups"] = json!(2);
                counts["designated_broad_groups"] = json!(2);
            }),
        ),
        (
            "rank receipt unbound",
            false,
            Box::new(|s| s["rank_receipts"][0]["sha256"] = json!("e".repeat(64))),
        ),
        (
            "parameters differ from receipt",
            false,
            Box::new(|s| s["parameter_sha256"] = json!("f".repeat(64))),
        ),
        ("manifest step", false, Box::new(|s| s["step"] = json!(2))),
    ];
    // Positive control: the unedited manifest commits.
    let mut control = fixture();
    control.train_to(3);
    control
        .commit_with(&|_| {}, false, &mut checkpoint_commit::no_stop)
        .unwrap();
    for (case, omit_optimizer, edit) in cases {
        let mut f = fixture();
        f.train_to(2);
        f.commit(&mut checkpoint_commit::no_stop).unwrap();
        f.train_to(3);
        let latest_before = fs::read(f.run.join("latest.json")).unwrap();
        let result = f.commit_with(&*edit, omit_optimizer, &mut checkpoint_commit::no_stop);
        assert!(
            result.is_err(),
            "{case}: an invalid checkpoint was committed"
        );
        assert!(!f.checkpoint_dir(3).exists(), "{case}: exposed step-3");
        assert_eq!(
            fs::read(f.run.join("latest.json")).unwrap(),
            latest_before,
            "{case}: latest.json changed"
        );
        assert_eq!(f.load().unwrap().step, 2);
    }
}

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
