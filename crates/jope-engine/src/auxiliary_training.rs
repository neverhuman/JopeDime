//! Bounded neural learning while CPU GP searches use an immutable proposal checkpoint.
use crate::{corpus::Index, manifest::*, telemetry::now, training};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread::{self, JoinHandle},
};

pub const PROTOCOL: &str = "frozen_proposals_auxiliary_fixed_panel_v1";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Options {
    pub block_steps: usize,
    pub max_steps: usize,
    pub patience: usize,
    pub relative_improvement: f64,
}
impl Options {
    pub fn for_cycle(steps: usize) -> Self {
        Self {
            block_steps: 300.min(steps),
            max_steps: steps,
            patience: 3,
            relative_improvement: 0.001,
        }
    }
    fn validate(&self) -> Result<()> {
        ensure!(
            self.block_steps > 0
                && self.max_steps > 0
                && self.patience > 0
                && self.relative_improvement.is_finite()
                && self.relative_improvement > 0.0
                && self.relative_improvement < 1.0,
            "invalid auxiliary budget or plateau guard"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Selection {
    pub checkpoint: PathBuf,
    pub state_sha256: String,
    pub step: usize,
    pub validation_nll: f64,
    pub panel_sha256: String,
}

#[derive(Debug, Clone)]
struct Plateau {
    best: f64,
    significant: f64,
    stale: usize,
    minimum_steps: usize,
    options: Options,
}
impl Plateau {
    fn observe(&mut self, step: usize, nll: f64) -> Result<bool> {
        ensure!(
            nll.is_finite() && nll >= 0.0,
            "invalid auxiliary validation NLL"
        );
        let better = nll < self.best;
        self.best = self.best.min(nll);
        if nll < self.significant * (1.0 - self.options.relative_improvement) {
            self.significant = nll;
            self.stale = 0;
        } else if step >= self.minimum_steps {
            self.stale += 1;
        }
        Ok(better)
    }
    fn stopped(&self, step: usize) -> bool {
        step >= self.minimum_steps && self.stale >= self.options.patience
    }
}

fn checkpoint(
    root: &Path,
    path: &Path,
    corpus_sha: &str,
    panel_sha: &str,
) -> Result<(Selection, Value)> {
    let state: Value = read_json(&path.join("state.json"))?;
    let step = state["step"].as_u64().context("checkpoint step absent")? as usize;
    let validation = &state["validation"];
    let nll = validation["sequence_nll"]
        .as_f64()
        .context("checkpoint has no exact validation NLL")?;
    ensure!(
        state["corpus_sha256"] == corpus_sha
            && state["sampler_id"] == training::SAMPLER_ID
            && validation["sampler_id"] == training::SAMPLER_ID
            && validation["panel_sha256"] == panel_sha
            && validation["evaluated_step"] == step
            && validation["family_balanced"] == true
            && nll.is_finite()
            && nll >= 0.0,
        "auxiliary checkpoint/panel identity mismatch"
    );
    ensure!(
        file_hash(&path.join("model.ot"))? == state["model_sha256"]
            && file_hash(&path.join("optimizer.ot"))? == state["optimizer_sha256"]
            && file_hash(&path.join("optimizer.json"))? == state["optimizer_metadata_sha256"]
            && file_hash(&path.join("config.json"))? == state["config_sha256"],
        "auxiliary checkpoint payload changed"
    );
    Ok((
        Selection {
            checkpoint: path.strip_prefix(root)?.to_path_buf(),
            state_sha256: file_hash(&path.join("state.json"))?,
            step,
            validation_nll: nll,
            panel_sha256: panel_sha.into(),
        },
        state,
    ))
}

/// Validate the frozen selection receipt without starting a trainer.
pub fn load_selection(root: &Path, output: &Path) -> Result<Selection> {
    let complete: Value = read_json(&output.join("complete.json"))?;
    let spec: Value = read_json(&output.join("spec.json"))?;
    ensure!(
        complete["protocol"] == PROTOCOL
            && complete["spec_sha256"] == file_hash(&output.join("spec.json"))?
            && complete["final_tests_opened"] == false,
        "auxiliary completion identity changed"
    );
    let selected: Selection = serde_json::from_value(complete["selection"].clone())?;
    let (actual, _) = checkpoint(
        root,
        &resolve(root, &selected.checkpoint)?,
        spec["corpus_sha256"]
            .as_str()
            .context("auxiliary corpus hash absent")?,
        spec["panel_sha256"]
            .as_str()
            .context("auxiliary panel hash absent")?,
    )?;
    ensure!(
        serde_json::to_value(&actual)? == serde_json::to_value(&selected)?,
        "selected auxiliary checkpoint changed"
    );
    Ok(actual)
}

pub struct Auxiliary {
    stop: Arc<AtomicBool>,
    gp_done: Arc<AtomicBool>,
    worker: Option<JoinHandle<Result<Selection>>>,
}
impl Auxiliary {
    pub fn start(
        root: &Path,
        corpus: &Path,
        frozen_proposal: &Path,
        output: &Path,
        options: Options,
        work_pending: bool,
    ) -> Result<Self> {
        options.validate()?;
        let stop = Arc::new(AtomicBool::new(!work_pending));
        let signal = stop.clone();
        let gp_done = Arc::new(AtomicBool::new(!work_pending));
        let completed = gp_done.clone();
        let (root, corpus, frozen_proposal, output) = (
            root.to_owned(),
            corpus.to_owned(),
            frozen_proposal.to_owned(),
            output.to_owned(),
        );
        let worker = thread::spawn(move || {
            let result = run(
                &root,
                &corpus,
                &frozen_proposal,
                &output,
                options,
                &signal,
                &completed,
            );
            if let Err(error) = &result {
                let _ = write_json(
                    &output.join("failure.json"),
                    &json!({"timestamp":now(),"failure":format!("{error:#}"),"resumable":true}),
                );
            }
            result
        });
        Ok(Self {
            stop,
            gp_done,
            worker: Some(worker),
        })
    }
    pub fn finish(mut self) -> Result<Selection> {
        self.gp_done.store(true, Ordering::Relaxed);
        self.stop.store(true, Ordering::Relaxed);
        self.worker
            .take()
            .unwrap()
            .join()
            .map_err(|_| anyhow::anyhow!("auxiliary trainer thread panicked"))?
    }
}
impl Drop for Auxiliary {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        // Controlled training checks this signal in lease waits, rank polling and
        // retries. Its Ranks guard kills/reaps owned children before this returns.
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn run(
    root: &Path,
    corpus: &Path,
    frozen: &Path,
    output: &Path,
    options: Options,
    stop: &AtomicBool,
    gp_done: &AtomicBool,
) -> Result<Selection> {
    fs::create_dir_all(output)?;
    let corpus_sha = file_hash(&corpus.join("index.json"))?;
    let index: Index = read_json(&corpus.join("index.json"))?;
    let panel = training::validation_panel(&index, 8, SEED)?;
    let panel_sha = hash(&serde_json::to_vec(&panel)?);
    let minimum_steps = index
        .groups
        .iter()
        .filter(|g| g.partition == "meta_train")
        .map(|g| g.records.len())
        .sum::<usize>()
        .div_ceil(2048)
        .max(1);
    let (initial, _) = checkpoint(root, frozen, &corpus_sha, &panel_sha)?;
    let spec = json!({"protocol":PROTOCOL,"corpus_sha256":corpus_sha,"frozen_proposal_checkpoint":initial,
        "panel_sha256":panel_sha,"sampler_id":training::SAMPLER_ID,"executable_sha256":file_hash(&std::env::current_exe()?)?,"training_executable_sha256":file_hash(&crate::controller::training_executable(root)?)?,
        "minimum_steps_before_plateau":minimum_steps,"options":options,"world":8,"microbatch":256,
        "optimizer_continuation":"one auxiliary optimizer exact-resumed between blocks; initial model warm-start only",
        "qualification":true,"final_tests_opened":false});
    let spec_path = output.join("spec.json");
    if spec_path.exists() {
        ensure!(
            read_json::<Value>(&spec_path)? == spec,
            "auxiliary resume specification changed"
        );
    } else {
        write_json(&spec_path, &spec)?;
    }
    if output.join("complete.json").exists() {
        return load_selection(root, output);
    }
    let train = output.join("train");
    let blocks = output.join("blocks");
    fs::create_dir_all(&blocks)?;
    let mut best = initial;
    let mut plateau = Plateau {
        best: best.validation_nll,
        significant: best.validation_nll,
        stale: 0,
        minimum_steps,
        options: options.clone(),
    };
    let mut observed = 0usize;
    let mut history = fs::read_dir(&blocks)?
        .map(|e| e.map(|e| e.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    history.sort();
    for path in history {
        let receipt: Value = read_json(&path)?;
        ensure!(
            receipt["spec_sha256"] == file_hash(&spec_path)?
                && receipt["checkpoint_state_canonical_sha256"]
                    == hash(&serde_json::to_vec(&receipt["checkpoint_state"])?),
            "auxiliary block spec or copied state changed"
        );
        let selected: Selection = serde_json::from_value(receipt["selection"].clone())?;
        let state = &receipt["checkpoint_state"];
        ensure!(
            state["step"] == selected.step
                && state["corpus_sha256"] == corpus_sha
                && state["validation"]["panel_sha256"] == panel_sha
                && state["validation"]["evaluated_step"] == selected.step
                && state["validation"]["sequence_nll"] == selected.validation_nll
                && selected.step > observed,
            "auxiliary block history changed"
        );
        if plateau.observe(selected.step, selected.validation_nll)? {
            best = selected.clone();
        }
        observed = selected.step;
    }
    // The selected checkpoint is protected before the next epoch/interval save.
    write_json(&train.join("best.json"), &best)?;
    let reason;
    loop {
        // A complete epoch checkpoint may precede a service interruption, even
        // when no training-complete receipt or auxiliary block receipt exists.
        if train.join("latest.json").is_file() {
            let latest = crate::pipeline::latest(root, &train)?;
            let raw: Value = read_json(&latest.join("state.json"))?;
            let step = raw["step"].as_u64().context("auxiliary step absent")? as usize;
            if step > observed && raw["validation"]["evaluated_step"] == step {
                let (candidate, state) = checkpoint(root, &latest, &corpus_sha, &panel_sha)?;
                if plateau.observe(step, candidate.validation_nll)? {
                    best = candidate.clone();
                }
                write_json(
                    &blocks.join(format!("step-{step:010}.json")),
                    &json!({"spec_sha256":file_hash(&spec_path)?,"selection":candidate,"checkpoint_state_canonical_sha256":hash(&serde_json::to_vec(&state)?),"checkpoint_state":state}),
                )?;
                observed = step;
                write_json(&train.join("best.json"), &best)?;
            }
        }
        let current = if train.join("latest.json").exists() {
            let latest: Value = read_json(&train.join("latest.json"))?;
            latest["step"]
                .as_u64()
                .context("auxiliary latest step absent")? as usize
        } else {
            0
        };
        if stop.load(Ordering::Relaxed) {
            reason = "GP searches completed or pipeline stopped";
            break;
        }
        if crate::campaign::check_deadline().is_err() {
            reason = "fixed campaign deadline";
            break;
        }
        if plateau.stopped(current) {
            reason = "fixed-panel validation NLL plateau";
            break;
        }
        if current >= options.max_steps {
            reason = "auxiliary step budget reached";
            break;
        }
        let target = (current + options.block_steps).min(options.max_steps);
        write_json(
            &root.join("runs/auxiliary-training.json"),
            &json!({"timestamp":now(),"phase":"training while CPU GP uses frozen proposals",
            "output":output,"step":current,"target_step":target,"maximum_steps":options.max_steps,"best":best,
            "validation_stale_checks":plateau.stale,"minimum_steps":minimum_steps,"panel_sha256":panel_sha,"qualification":true}),
        )?;
        let resume = train.join("run.json").is_file();
        // best.json alone is metadata, not an existing training launch.
        if !resume && train.join("best.json").is_file() {
            fs::remove_file(train.join("best.json"))?;
        }
        let result = crate::controller::train_controlled(
            root,
            corpus,
            &train,
            8,
            256,
            target,
            1000,
            resume,
            None,
            Some(frozen),
            Some(stop),
        );
        if let Err(error) = result {
            write_json(
                &output.join("last-interruption.json"),
                &json!({"timestamp":now(),"failure":format!("{error:#}"),"stop_requested":stop.load(Ordering::Relaxed),"deadline_reached":crate::campaign::check_deadline().is_err()}),
            )?;
            if stop.load(Ordering::Relaxed) || crate::campaign::check_deadline().is_err() {
                continue;
            }
            return Err(error);
        }
    }
    let selected_path = resolve(root, &best.checkpoint)?;
    let (verified, _) = checkpoint(root, &selected_path, &corpus_sha, &panel_sha)?;
    ensure!(
        verified.state_sha256 == best.state_sha256,
        "best auxiliary checkpoint changed"
    );
    if stop.load(Ordering::Relaxed) && !gp_done.load(Ordering::Relaxed) {
        write_json(
            &output.join("interrupted.json"),
            &json!({"timestamp":now(),"reason":reason,"selection":best,"completed_optimizer_steps":observed,"resumable":true}),
        )?;
        return Ok(best);
    }
    write_json(
        &output.join("complete.json"),
        &json!({"schema":"parallel.auxiliary_training_complete.v1","protocol":PROTOCOL,
        "spec_sha256":file_hash(&spec_path)?,"timestamp":now(),"reason":reason,"selection":best,
        "completed_optimizer_steps":observed,"validation_stale_checks":plateau.stale,"qualification":true,
        "proposal_checkpoint_unchanged":true,"final_tests_opened":false}),
    )?;
    write_json(
        &root.join("runs/auxiliary-training.json"),
        &json!({"timestamp":now(),"phase":"completed","status":"complete","complete":true,"reason":reason,"output":output,"best":best,"completed_optimizer_steps":observed,"qualification":true}),
    )?;
    Ok(best)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn already_completed_gp_selects_bound_parent_without_launching_training() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let corpus = root.join("corpus");
        let frozen = root.join("runs/frozen/checkpoints/step-3000");
        let output = root.join("runs/cycle/auxiliary");
        let mut groups = Vec::new();
        for i in 0..8 {
            groups.push(crate::corpus::Group {
                context: i,
                family: format!("family-{i}"),
                partition: "meta_validation".into(),
                records: (i * 8..i * 8 + 8).collect(),
                max_length: 4,
            });
        }
        let index = Index {
            schema: "parallel.teaching_corpus.v1".into(),
            qualification: true,
            vocabulary_sha256: "vocab".into(),
            context_index_sha256: "context".into(),
            records_sha256: "records".into(),
            record_bytes: 0,
            records: 64,
            legal_sets: vec![],
            contexts: vec![],
            groups,
        };
        let panel = training::validation_panel(&index, 8, SEED).unwrap();
        let mut counts = [0usize; 8];
        for &id in panel.iter().flatten() {
            counts[id / 8] += 1;
        }
        assert_eq!(counts, [256; 8]);
        assert_eq!(panel, training::validation_panel(&index, 8, SEED).unwrap());
        let panel_sha = hash(&serde_json::to_vec(&panel).unwrap());
        write_json(&corpus.join("index.json"), &index).unwrap();
        fs::create_dir_all(&frozen).unwrap();
        let mut state = json!({"step":3000,"sampler_id":training::SAMPLER_ID,"corpus_sha256":file_hash(&corpus.join("index.json")).unwrap(),"validation":{"sequence_nll":1.0,"evaluated_step":3000,"panel_sha256":panel_sha,"sampler_id":training::SAMPLER_ID,"family_balanced":true}});
        for (name, key) in [
            ("model.ot", "model_sha256"),
            ("optimizer.ot", "optimizer_sha256"),
            ("optimizer.json", "optimizer_metadata_sha256"),
            ("config.json", "config_sha256"),
        ] {
            fs::write(frozen.join(name), name).unwrap();
            state[key] = file_hash(&frozen.join(name)).unwrap().into();
        }
        write_json(&frozen.join("state.json"), &state).unwrap();
        let selected = Auxiliary::start(
            root,
            &corpus,
            &frozen,
            &output,
            Options::for_cycle(3000),
            false,
        )
        .unwrap()
        .finish()
        .unwrap();
        assert_eq!(selected.checkpoint, frozen.strip_prefix(root).unwrap());
        assert!(!output.join("train/run.json").exists());
        assert_eq!(
            load_selection(root, &output).unwrap().state_sha256,
            selected.state_sha256
        );
        fs::write(frozen.join("model.ot"), b"changed weights").unwrap();
        assert!(load_selection(root, &output).is_err());
    }

    #[test]
    fn plateau_requires_an_epoch_and_counts_only_fixed_panel_checks() {
        let options = Options::for_cycle(3000);
        let mut p = Plateau {
            best: 1.,
            significant: 1.,
            stale: 0,
            minimum_steps: 400,
            options,
        };
        assert!(!p.observe(100, 1.1).unwrap());
        assert!(!p.observe(300, 1.2).unwrap());
        assert_eq!(p.stale, 0);
        assert!(p.observe(600, 0.9995).unwrap());
        assert_eq!(p.stale, 1);
        assert!(p.observe(900, 0.9985).unwrap());
        assert_eq!(p.stale, 0); // cumulative improvement clears the0.1% threshold.
        for step in [1200, 1500, 1800] {
            p.observe(step, 1.).unwrap();
        }
        assert!(p.stopped(1800));
        assert!(p.observe(2100, 0.98).unwrap());
        assert!(!p.stopped(2100));
        assert!(p.observe(2400, f64::NAN).is_err());
    }
}
