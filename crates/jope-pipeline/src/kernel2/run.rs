//! Opt-in single-cut shipping run, using the bounded profile coordinator.

use anyhow::{bail, ensure, Result};
use feat_math::evolve::FaultPolicy;
use std::path::Path;

use super::{
    publication,
    receipt::{self, Receipt},
    search::{self, Refinement},
};
use crate::{
    dataset, emit, prepare, receipt as legacy,
    run::{self, RunOptions, Stages},
};

#[derive(Clone, Debug)]
pub struct Options {
    pub run: RunOptions,
    profile_id: String,
    fault_policy: FaultPolicy,
    refinement: Refinement,
}

impl Options {
    pub fn new(
        run: RunOptions,
        profile: &str,
        policy: FaultPolicy,
        refinement: Refinement,
    ) -> Result<Self> {
        let options = Self {
            run,
            profile_id: profile.into(),
            fault_policy: policy,
            refinement,
        };
        options.preflight()?;
        Ok(options)
    }

    fn coordinator(&self) -> Result<search::Options> {
        let r = &self.run;
        let options = crate::search::SearchOptions {
            threads: r.threads,
            islands: r.islands,
            migration_interval: r.migration_interval,
            cancel: r.cancel.clone(),
            clock: r.clock,
            ..crate::search::SearchOptions::new(r.budget, r.rng_seed)
        };
        search::Options::new(
            options,
            &self.profile_id,
            self.fault_policy,
            self.refinement,
        )
    }

    fn preflight(&self) -> Result<()> {
        run::preflight(&self.run)?;
        ensure!(
            self.run.cuts == 1,
            "kernel2 shipping currently requires --cuts 1"
        );
        ensure!(
            self.run.incumbent.is_none(),
            "kernel2 --incumbent is not supported yet; use --seed-genomes"
        );
        self.coordinator()?;
        run::check_cancelled(&self.run)
    }
}

pub fn run(options: &Options) -> Result<Receipt> {
    options.preflight()?;
    let r = &options.run;
    for entry in ["receipt.json", "prepared", "search", "emit"] {
        ensure!(
            std::fs::symlink_metadata(r.output.join(entry)).is_err(),
            "{}: already holds a pipeline run ({entry}); choose a new --output",
            r.output.display()
        );
    }
    let (seeds, seeding) = run::resolve_seeding(r)?;
    let mut stages = Stages::new();
    let started = r.clock.start();
    let resolved = run::resolve_regression(&r.dataset)?;
    let rows = dataset::training_rows(&resolved)?;
    stages.record("dataset", "passed", None, started);
    let mut coordinator = options.coordinator()?;
    coordinator.search.seeds = seeds;
    let width = rows
        .first()
        .context("no training rows")?
        .len()
        .checked_sub(1)
        .context("no feature columns")?;
    for seed in &coordinator.search.seeds {
        feat_math::kernel::search::admit_child(seed, coordinator.evaluator().profile(), width)?;
    }
    let started = r.clock.start();
    let partition = prepare::partition(&rows, &resolved.row.dataset_hash, r.seed, r.cuts)?;
    run::check_cancelled(r)?;
    std::fs::create_dir_all(&r.output)?;
    let root = r.output.canonicalize()?;
    let preparation = partition.write(&root.join("prepared"))?;
    stages.record("prepare", "passed", None, started);
    eprintln!(
        "jope pipeline: propose stage SKIPPED - {}",
        run::PROPOSE_SKIP_REASON
    );
    stages.record(
        "propose",
        "skipped",
        Some(run::PROPOSE_SKIP_REASON.into()),
        r.clock.start(),
    );
    let started = r.clock.start();
    let cut = &preparation.cuts[0];
    let searched = search::run_cut(cut, &coordinator)?;
    let (search, search_sha256) = publication::publish(cut, &searched, seeding, &root)?;
    stages.record("search", "passed", None, started);
    let genome = &search.best.observation.genome;
    stages.record("select", "passed", None, r.clock.start());
    run::check_cancelled(r)?;
    let started = r.clock.start();
    let emitted = emit::emit_with_training(genome, &r.languages, &root.join("emit"), &rows)?;
    ensure!(
        emitted.canonical_genome == *genome
            && emitted.genome_sha256 == search.best.observation.program_sha256,
        "emitted program is not the selected one"
    );
    let ok = !emitted.artifacts.is_empty();
    let reason = (!emitted.exclusions.is_empty()).then(|| {
        emitted
            .exclusions
            .iter()
            .map(|e| format!("{}: {}", e.language, e.reason))
            .collect::<Vec<_>>()
            .join("; ")
    });
    stages.record(
        "emit",
        if ok { "passed" } else { "failed" },
        reason,
        started,
    );
    stages.record("receipt", "passed", None, r.clock.start());
    let receipt = Receipt {
        schema: receipt::SCHEMA_ID.into(),
        ok,
        tool_version: crate::TOOL_VERSION.into(),
        git_sha: r.source_commit.clone().or_else(run::git_sha),
        created_at: r.clock.now_rfc3339(),
        policy: r.policy.as_str().into(),
        dataset: dataset_record(r, &resolved),
        prepared: legacy::Prepared {
            sha256: preparation.sha256,
            seed: r.seed,
            rows: preparation.rows as u64,
            folds: prepare::FOLDS as u64,
            cuts: vec![0],
        },
        stages: stages.stages,
        search,
        search_sha256,
        emit_manifest_sha256: crate::sha256_file(&root.join("emit/manifest.json"))?,
        artifacts: emitted
            .artifacts
            .into_iter()
            .map(|a| legacy::Artifact {
                language: a.language.into(),
                path: a.path,
                sha256: a.sha256,
                bytes: a.bytes,
                build: a.build,
            })
            .collect(),
        emit_exclusions: emitted
            .exclusions
            .into_iter()
            .map(|e| legacy::Exclusion {
                language: e.language.into(),
                reason: e.reason,
            })
            .collect(),
    };
    receipt::validate_value(&serde_json::to_value(&receipt)?)?;
    run::check_cancelled(r)?;
    let path = root.join("receipt.json");
    std::fs::write(&path, serde_json::to_vec_pretty(&receipt)?)?;
    if !ok {
        bail!(
            "no code was emitted; failed v4 receipt in {}",
            root.display()
        );
    }
    Ok(receipt)
}

fn dataset_record(r: &RunOptions, resolved: &dataset::Resolved) -> legacy::Dataset {
    let row = &resolved.row;
    legacy::Dataset {
        id: row.id.clone(),
        dataset_hash: row.dataset_hash.clone(),
        path: r.dataset.clone(),
        task: row.task.as_str().into(),
        rows: row.shape.rows,
        columns: row.shape.columns,
        training_rows: row.split.train_rows,
        files: resolved
            .files
            .iter()
            .map(|file| legacy::DatasetFile {
                name: file.name.clone(),
                sha256: file.sha256.clone(),
                bytes: file.bytes,
            })
            .collect(),
    }
}

use anyhow::Context;

/// v4 is explicit at this entry point; legacy consumers do not reinterpret it.
pub(crate) fn is_v4(path: &Path) -> bool {
    std::fs::read(path)
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .is_some_and(|value| value["schema"] == receipt::SCHEMA_ID)
}
