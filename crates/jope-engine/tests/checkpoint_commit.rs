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
                lineage_sha256: None,
                intent_sha256: None,
                attempt_id: None,
                window_id: None,
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

mod loading;
mod publication;

#[path = "checkpoint_commit/successor_expectations.rs"]
mod successor_expectations;
