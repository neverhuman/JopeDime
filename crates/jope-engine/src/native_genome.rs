//! Explicit per-genome evaluation and budgeted continuation. No standalone execution grant.
use crate::{
    corpus::ContextRow,
    manifest::{file_hash, hash, read_verified, resolve},
    native_graph::Artifact,
};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
#[cfg(test)]
use serde_json::json;
use serde_json::Value;
use std::{
    collections::BTreeMap,
    fs,
    io::Read,
    os::unix::fs::OpenOptionsExt,
    path::{Path, PathBuf},
};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum Arm {
    Model,
    Generic,
    Shuffled,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    Direct,
    Continuation,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Candidate {
    pub schema: String,
    pub run: String,
    pub generation: u64,
    pub arm: Arm,
    pub matched_slot: u64,
    pub context: ContextRow,
    pub genome: String,
    pub genome_sha256: String,
    pub proposal: Artifact,
    pub parent_checkpoint: Artifact,
    pub sampling_seed: u64,
    pub origin_capture: Artifact,
    pub origin_row: usize,
    pub generic_draw: Option<u64>,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenomeTask {
    pub schema: String,
    pub stage: Stage,
    pub candidate: Artifact,
    pub output: PathBuf,
    pub vocabulary: Artifact,
    pub admission_policy: Artifact,
    pub qualification: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preparation_overlay: Option<Artifact>,
    pub evaluations: usize,
    pub seconds: u64,
    pub rng_seed: u64,
    /// Original direct completion wrapper, never an unverified score or arbitrary model.
    pub direct_parent: Option<Artifact>,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenomeResult {
    pub schema: String,
    pub task: Artifact,
    pub candidate: Artifact,
    pub stage: Stage,
    pub search: Artifact,
    pub observations: Artifact,
    pub direct_parent: Option<Artifact>,
    pub artifacts: BTreeMap<PathBuf, String>,
}
/// Reviewed first-window continuation protocol (root preregistration 2026-09-14T19:06:10Z).
pub const CONTINUATION_PROTOCOL: &str = "plain_seeded_continuation_v1";
/// A continuation is ordinary training evidence only with at least this many completed children.
pub const MINIMUM_COMPLETED_CHILDREN: usize = 1;
pub const TRAINING_ADMITTED: &str = "admitted";
pub const REFUSED_NO_COMPLETED_CHILD: &str = "refused_no_completed_child";
/// A different learning target, not a rewritten original evaluator observation.
/// Refused targets keep every original row, cost and best observed value as incomplete evidence.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContinuationTarget {
    pub schema: String,
    pub protocol: String,
    pub candidate: Artifact,
    pub direct: Artifact,
    pub continuation: Artifact,
    pub parent_genome_sha256: String,
    pub winning_original_observation: Option<Artifact>,
    pub winning_row: Option<usize>,
    pub quality_rmse_cut_normalized: Option<f64>,
    pub fit_runtime_seconds: f64,
    pub completed_evaluations: usize,
    pub failed_evaluations: usize,
    /// Continuation children with a null original failure and a measured RMSE; the parent is excluded.
    pub completed_children: usize,
    pub training_admission: String,
    pub evaluation_budget: usize,
    pub search_seconds_budget: u64,
    pub target_kind: String,
}
fn valid_hash(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}
pub(crate) fn checked(root: &Path, a: &Artifact, maximum: u64) -> Result<Vec<u8>> {
    ensure!(valid_hash(&a.sha256), "invalid native genome artifact hash");
    let path = resolve(root, &a.path)?;
    let f = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)?;
    ensure!(
        f.metadata()?.is_file() && f.metadata()?.len() <= maximum,
        "bounded native genome regular artifact required"
    );
    let mut b = Vec::new();
    f.take(maximum + 1).read_to_end(&mut b)?;
    ensure!(
        b.len() as u64 <= maximum && hash(&b) == a.sha256,
        "native genome artifact changed"
    );
    Ok(b)
}
pub(crate) fn load<T: serde::de::DeserializeOwned>(root: &Path, a: &Artifact) -> Result<T> {
    Ok(serde_json::from_slice(&checked(
        root,
        a,
        16 * 1024 * 1024,
    )?)?)
}
pub(crate) const DATASET_METADATA_LIMIT: u64 = 256 * 1024 * 1024;
/// Explicit large scientific metadata reader; candidate/task generic bounds stay unchanged.
pub(crate) fn load_corpus_index(root: &Path, a: &Artifact) -> Result<Value> {
    let value: Value = serde_json::from_slice(&checked(root, a, DATASET_METADATA_LIMIT)?)?;
    ensure!(
        matches!(
            value["schema"].as_str(),
            Some("parallel.teaching_corpus.v1" | "parallel.teaching_shards.v1")
        ),
        "unsupported corpus index metadata"
    );
    Ok(value)
}
pub(crate) fn load_dataset_receipt(root: &Path, a: &Artifact) -> Result<Value> {
    let value: Value = serde_json::from_slice(&checked(root, a, DATASET_METADATA_LIMIT)?)?;
    ensure!(
        value["schema"] == "jain.datasets_service_receipt.v1",
        "unsupported dataset receipt metadata"
    );
    Ok(value)
}

impl GenomeTask {
    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == "jope.native_genome_task.v1"
                && self.evaluations > 0
                && self.evaluations <= 1000
                && self.seconds > 0
                && self.seconds <= 600,
            "native genome task budget/schema invalid"
        );
        crate::manifest::relative(&self.output)?;
        ensure!(
            match self.stage {
                Stage::Direct => self.evaluations == 1 && self.direct_parent.is_none(),
                Stage::Continuation => self.direct_parent.is_some(),
            },
            "native genome direct/continuation contract mismatch"
        );
        Ok(())
    }
    pub(crate) fn candidate(&self, root: &Path) -> Result<Candidate> {
        self.validate()?;
        let c: Candidate = load(root, &self.candidate)?;
        ensure!(
            c.schema == "jope.native_candidate.v1"
                && !c.run.is_empty()
                && c.context.partition == "meta_train"
                && hash(c.genome.as_bytes()) == c.genome_sha256
                && feat_math::GenomeIr::parse(&c.genome)?.canonical() == c.genome,
            "candidate identity, canonical program or sealed partition differs"
        );
        let prepared = resolve(root, &c.context.prepared)?;
        // Route R8: the boundary is decided on exactly the bound preparation bytes.
        let p: crate::prepare::Prepared = serde_json::from_slice(&read_verified(
            &prepared.join("prepared.json"),
            &c.context.prepared_sha256,
            "candidate preparation",
        )?)?;
        ensure!(
            p.task == "regression"
                && p.meta_partition == "meta_train"
                && p.family == c.context.family
                && p.dataset == c.context.dataset
                && (p.family_partition_frozen || self.preparation_overlay.is_some())
                && (p.clean_evaluation || (self.qualification && p.qualification_only)),
            "candidate scientific boundary changed"
        );
        if let Some(overlay) = &self.preparation_overlay {
            ensure!(
                !self.qualification,
                "scoped overlay cannot relabel qualification material"
            );
            crate::preparation_overlay::verify(root, overlay, &c, &p)?;
        }
        let v: crate::grammar::Vocab = load(root, &self.vocabulary)?;
        ensure!(
            v.encode(&c.genome)?.len() <= crate::corpus::SEQUENCE,
            "candidate cannot retain exact frozen tokens"
        );
        let proposal: crate::native_graph::VerifiedResult = load(root, &c.proposal)?;
        ensure!(
            proposal.schema == "jope.native_graph_result.v1"
                && proposal.run == c.run
                && proposal.generation == c.generation,
            "candidate original proposal scope differs"
        );
        let capture: Value = load(root, &c.origin_capture)?;
        ensure!(
            capture["schema"] == "jope.all_valid_neural.v1"
                && capture["final_tests_opened"] == false,
            "candidate source capture differs"
        );
        let row = capture["rows"]
            .as_array()
            .and_then(|rows| rows.get(c.origin_row))
            .context("candidate source occurrence absent")?;
        ensure!(
            row["program_sha256"]
                == hash(row["genome"].as_str().context("source genome")?.as_bytes()),
            "source genome digest differs"
        );
        if c.arm == Arm::Generic {
            let draw = c.generic_draw.context("generic source draw absent")?;
            ensure!(
                c.sampling_seed
                    == crate::native_loop::generation_seed(
                        &c.run,
                        c.generation,
                        &format!(
                            "generic:{}:{}:{}",
                            c.context.prepared_sha256, c.context.cut, c.matched_slot
                        )
                    )?,
                "generic RNG is not generation-derived"
            );
            ensure!(
                draw < 10000 && capture["arm"] == "model",
                "generic comparison origin differs"
            );
            let raw =
                feat_math::evolve::random_canonical_pipeline(c.sampling_seed.wrapping_add(draw), 6);
            ensure!(
                v.project_proposal(&raw)? == c.genome,
                "generic comparison RNG/program differs"
            );
        } else {
            ensure!(
                c.generic_draw.is_none()
                    && row["genome"] == c.genome
                    && capture["sampling_seed"] == c.sampling_seed
                    && capture["arm"] == serde_json::to_value(c.arm)?,
                "model/shuffled original occurrence differs"
            );
        }
        let producer: Value = load(root, &proposal.output)?;
        let original: BTreeMap<PathBuf, String> =
            serde_json::from_value(producer["artifacts"].clone())?;
        ensure!(
            original
                .iter()
                .any(|(path, sha)| c.origin_capture.path.ends_with(path)
                    && sha == &c.origin_capture.sha256),
            "candidate capture absent from original proposal output closure"
        );
        let _: Value = load(root, &c.parent_checkpoint)?;
        Ok(c)
    }
}
/// Revalidate every original strict observation; no parent score is accepted from the planner.
pub(crate) fn verified_rows(
    root: &Path,
    reference: &Artifact,
) -> Result<(GenomeTask, Candidate, GenomeResult, Vec<Value>)> {
    let result: GenomeResult = load(root, reference)?;
    let task: GenomeTask = load(root, &result.task)?;
    let candidate = task.candidate(root)?;
    let rows = verify_result(root, &task, &result)?;
    Ok((task, candidate, result, rows))
}
pub(crate) fn verify_result(
    root: &Path,
    task: &GenomeTask,
    result: &GenomeResult,
) -> Result<Vec<Value>> {
    ensure!(
        result.schema == "jope.native_genome_completion.v1",
        "native genome receipt schema"
    );
    let candidate = task.candidate(root)?;
    ensure!(
        result.stage == task.stage
            && result.candidate == task.candidate
            && result.direct_parent == task.direct_parent,
        "native genome result changes original task"
    );
    let directory = resolve(root, &task.output)?;
    ensure!(
        result.search.path == task.output.join("search.json")
            && result.observations.path == task.output.join("observations.json"),
        "native genome result outside original output"
    );
    ensure!(
        crate::smartcluster_worker::hash_directory(&directory)? == result.artifacts,
        "native genome output closure changed"
    );
    let search: Value = load(root, &result.search)?;
    let rows: Vec<Value> = load(root, &result.observations)?;
    ensure!(
        search["terminal"] == true
            && search["final_tests_opened"] == false
            && search["observations_sha256"] == result.observations.sha256
            && rows.len() <= task.evaluations
            && search["completed_evaluations"] == rows.len(),
        "incomplete native genome search"
    );
    let request: Value = crate::label_admission::artifact(root, &search["search_request"])?;
    ensure!(
        request["native_genome_task"] == serde_json::to_value(&result.task)?
            && request["evaluations"] == task.evaluations
            && request["seconds"] == task.seconds
            && request["rng_seed"] == task.rng_seed,
        "original genome request budget changed"
    );
    let expected: Value = load(root, &task.admission_policy)?;
    // Route R8: the preparation these observations are admitted against is the
    // bound one; `verify_inputs` refuses any parse that is not those bytes.
    let prepared: crate::prepare::Prepared = serde_json::from_slice(&read_verified(
        &resolve(root, &candidate.context.prepared)?.join("prepared.json"),
        &candidate.context.prepared_sha256,
        "admission prepared artifact",
    )?)?;
    let training = crate::label_admission::verify_inputs(root, &candidate.context, &prepared)?;
    let mut authority = crate::label_admission::approved_authority(root, &expected)?;
    for (i, row) in rows.iter().enumerate() {
        ensure!(
            row["evaluation_index"] == i
                && row["native_genome_task"] == serde_json::to_value(&result.task)?
                && row["native_candidate"] == serde_json::to_value(&task.candidate)?
                && row["comparison_arm"] == serde_json::to_value(candidate.arm)?
                && row["continuation_parent"] == serde_json::to_value(&task.direct_parent)?,
            "native genome observation ordering changed"
        );
        crate::label_admission::validate_receipt(&search, row, rows.len(), &expected)?;
        crate::label_admission::validate_preverified(
            root,
            row,
            &candidate.context,
            &prepared,
            crate::label_admission::AdmissionInputs {
                vocabulary_sha: &task.vocabulary.sha256,
                expected: &expected,
                fit_training: &training,
            },
            &mut authority,
        )?;
        if task.stage == Stage::Direct {
            ensure!(
                row["genome"] == candidate.genome,
                "direct task evaluated another program"
            );
        }
    }
    if task.stage == Stage::Direct {
        ensure!(
            rows.len() == 1,
            "direct task lacks its exact original evaluation outcome"
        );
    }
    Ok(rows)
}
pub(crate) fn direct_parent(
    root: &Path,
    task: &GenomeTask,
    c: &Candidate,
) -> Result<Option<(Value, Artifact)>> {
    let Some(parent) = &task.direct_parent else {
        return Ok(None);
    };
    let (old, prior, result, rows) = verified_rows(root, parent)?;
    ensure!(
        old.stage == Stage::Direct
            && result.candidate == task.candidate
            && prior.genome_sha256 == c.genome_sha256
            && old.vocabulary == task.vocabulary
            && old.admission_policy == task.admission_policy
            && old.qualification == task.qualification
            && old.preparation_overlay == task.preparation_overlay,
        "continuation changed original direct evaluation binding"
    );
    Ok(Some((
        rows.into_iter().next().context("direct row absent")?,
        result.observations,
    )))
}
/// Include every transitive immutable JSON artifact reference plus actual prepared bytes.
/// Live authority databases are not dependencies; original committed receipts are.
pub(crate) fn required_inputs(
    root: &Path,
    reference: &Artifact,
) -> Result<BTreeMap<PathBuf, String>> {
    let task: GenomeTask = load(root, reference)?;
    let c = task.candidate(root)?;
    let mut refs = vec![
        reference.clone(),
        task.candidate.clone(),
        task.vocabulary.clone(),
        task.admission_policy.clone(),
        c.proposal.clone(),
        c.parent_checkpoint.clone(),
        c.origin_capture.clone(),
    ];
    refs.extend(task.direct_parent.iter().cloned());
    refs.extend(task.preparation_overlay.iter().cloned());
    let mut out = artifact_closure(root, refs)?;
    prepared_closure(root, &c.context, &mut out)?;
    Ok(out)
}
/// Every transitive immutable JSON artifact reference reachable from `refs`.
fn artifact_closure(root: &Path, mut refs: Vec<Artifact>) -> Result<BTreeMap<PathBuf, String>> {
    let mut out = BTreeMap::new();
    fn nested(v: &Value, refs: &mut Vec<Artifact>) {
        if let Some(o) = v.as_object() {
            if let (Some(p), Some(h)) = (
                o.get("path").and_then(Value::as_str),
                o.get("sha256").and_then(Value::as_str),
            ) {
                if !Path::new(p).is_absolute() && valid_hash(h) {
                    refs.push(Artifact {
                        path: p.into(),
                        sha256: h.into(),
                    });
                }
            }
            // Genome output maps use paths relative to their declared search directory.
            if o.get("schema").and_then(Value::as_str) == Some("jope.native_genome_completion.v1") {
                if let (Some(map), Some(search)) = (
                    o.get("artifacts").and_then(Value::as_object),
                    o.get("search")
                        .and_then(|s| s.get("path"))
                        .and_then(Value::as_str),
                ) {
                    if let Some(base) = Path::new(search).parent() {
                        for (path, digest) in map {
                            if let Some(digest) = digest.as_str().filter(|h| valid_hash(h)) {
                                refs.push(Artifact {
                                    path: base.join(path),
                                    sha256: digest.into(),
                                });
                            }
                        }
                    }
                }
            }
            for x in o.values() {
                nested(x, refs);
            }
        } else if let Some(a) = v.as_array() {
            for x in a {
                nested(x, refs);
            }
        }
    }
    while let Some(a) = refs.pop() {
        ensure!(out.len() < 100000, "native genome closure too large");
        if let Some(old) = out.get(&a.path) {
            ensure!(old == &a.sha256, "conflicting native genome artifact path");
            continue;
        }
        let path = resolve(root, &a.path)?;
        // Route R8: nested references are taken from exactly the verified
        // bytes. Inputs above the parse bound are hashed and never parsed.
        match read_parse_bounded(&path, 16 * 1024 * 1024)? {
            Some(bytes) => {
                ensure!(hash(&bytes) == a.sha256, "native genome input changed");
                crate::manifest::after_verification(&path);
                if let Ok(value) = serde_json::from_slice::<Value>(&bytes) {
                    nested(&value, &mut refs);
                }
            }
            None => ensure!(file_hash(&path)? == a.sha256, "native genome input changed"),
        }
        out.insert(a.path, a.sha256);
    }
    Ok(out)
}
/// The whole file through one handle when it is within `limit` bytes; `None`
/// when it is larger (it is then hashed by the caller and never parsed).
fn read_parse_bounded(path: &Path, limit: u64) -> Result<Option<Vec<u8>>> {
    let file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    if file.metadata()?.len() > limit {
        return Ok(None);
    }
    let mut bytes = Vec::new();
    file.take(limit + 1).read_to_end(&mut bytes)?;
    Ok((bytes.len() as u64 <= limit).then_some(bytes))
}
/// The prepared directory's actual bytes and its original observation source.
fn prepared_closure(
    root: &Path,
    context: &ContextRow,
    out: &mut BTreeMap<PathBuf, String>,
) -> Result<()> {
    let dir = resolve(root, &context.prepared)?;
    for e in walkdir::WalkDir::new(&dir).follow_links(false) {
        let e = e?;
        ensure!(!e.file_type().is_symlink(), "prepared symlink");
        if e.file_type().is_file() {
            out.insert(
                e.path().strip_prefix(root)?.to_owned(),
                file_hash(e.path())?,
            );
        }
    }
    // Route R8: the closure records the preparation the candidate bound, and
    // its original observation source is read from exactly those bytes.
    let prepared = dir.join("prepared.json");
    let bytes = read_verified(&prepared, &context.prepared_sha256, "native genome input")?;
    ensure!(
        out.get(prepared.strip_prefix(root)?) == Some(&context.prepared_sha256),
        "native genome input changed"
    );
    let p: crate::prepare::Prepared = serde_json::from_slice(&bytes)?;
    if let Some(s) = p.storage {
        let path = if s.source.is_absolute() {
            s.source.strip_prefix(root)?.to_owned()
        } else {
            s.source
        };
        ensure!(
            file_hash(&resolve(root, &path)?)? == s.source_sha256,
            "original observation source changed"
        );
        out.insert(path, s.source_sha256);
    }
    Ok(())
}
pub(crate) fn execute(
    root: &Path,
    reference: &Artifact,
    expected: Stage,
    receipt: &Path,
) -> Result<()> {
    let task: GenomeTask = load(root, reference)?;
    ensure!(
        task.stage == expected,
        "fixed native genome service mismatch"
    );
    let c = task.candidate(root)?;
    let parent = direct_parent(root, &task, &c)?;
    // A crash after the original numerical search fsync but before wrapper fsync
    // must not execute the direct fit or continuation budget a second time.
    if !resolve(root, &task.output)?
        .join("search.json")
        .try_exists()?
    {
        crate::gp::search_native_genome(root, reference, &task, &c, parent.as_ref().map(|p| &p.0))?;
    }
    let result = GenomeResult {
        schema: "jope.native_genome_completion.v1".into(),
        task: reference.clone(),
        candidate: task.candidate.clone(),
        stage: task.stage,
        search: Artifact {
            path: task.output.join("search.json"),
            sha256: file_hash(&resolve(root, &task.output)?.join("search.json"))?,
        },
        observations: Artifact {
            path: task.output.join("observations.json"),
            sha256: file_hash(&resolve(root, &task.output)?.join("observations.json"))?,
        },
        direct_parent: task.direct_parent.clone(),
        artifacts: crate::smartcluster_worker::hash_directory(&resolve(root, &task.output)?)?,
    };
    verify_result(root, &task, &result)?;
    crate::smartcluster_worker::write_immutable_record(receipt, &serde_json::to_value(result)?)
}
/// Verified continuation outcomes have a separate identity and retain all original rows.
/// Under plain_seeded_continuation_v1 a zero-child, all-failed or timeout-only continuation is still
/// returned with its rows and cost, but `training_admission` refuses it as an ordinary training label.
pub fn target(root: &Path, continuation: &Artifact) -> Result<ContinuationTarget> {
    let (task, c, result, rows) = verified_rows(root, continuation)?;
    ensure!(
        task.stage == Stage::Continuation,
        "population/direct score is not a continuation target"
    );
    let direct = task
        .direct_parent
        .clone()
        .context("original direct parent absent")?;
    let (parent, parent_rows) = direct_parent(root, &task, &c)?.context("direct parent absent")?;
    let children = completed_children(&rows)?;
    let mut all = vec![(parent_rows, 0, parent)];
    all.extend(
        rows.into_iter()
            .enumerate()
            .map(|(i, r)| (result.observations.clone(), i, r)),
    );
    let (best, runtime, failures) = summarize_outcomes(&all)?;
    Ok(ContinuationTarget {
        schema: "jope.continuation_target.v2".into(),
        protocol: CONTINUATION_PROTOCOL.into(),
        candidate: task.candidate,
        direct,
        continuation: continuation.clone(),
        parent_genome_sha256: c.genome_sha256,
        winning_original_observation: best.as_ref().map(|x| x.1.clone()),
        winning_row: best.as_ref().map(|x| x.2),
        quality_rmse_cut_normalized: best.map(|x| x.0),
        fit_runtime_seconds: runtime,
        completed_evaluations: all.len() - 1,
        failed_evaluations: failures,
        completed_children: children,
        training_admission: training_admission(children).into(),
        evaluation_budget: task.evaluations,
        search_seconds_budget: task.seconds,
        target_kind: "best_observed_outcome_after_plain_seeded_continuation_v1".into(),
    })
}
/// Children only: the verified direct parent row never counts toward continuation admission.
fn completed_children(rows: &[Value]) -> Result<usize> {
    let mut completed = 0;
    for row in rows {
        if !row["failure"].is_null() {
            continue;
        }
        let quality = row["metrics"]["rmse"]
            .as_f64()
            .context("original child quality absent")?;
        ensure!(
            quality.is_finite() && quality >= 0.,
            "invalid original child quality"
        );
        completed += 1;
    }
    Ok(completed)
}
pub(crate) fn training_admission(completed_children: usize) -> &'static str {
    if completed_children >= MINIMUM_COMPLETED_CHILDREN {
        TRAINING_ADMITTED
    } else {
        REFUSED_NO_COMPLETED_CHILD
    }
}
// Called only after original receipt validation; keeps parent and descendant identities distinct.
type OutcomeSummary = (Option<(f64, Artifact, usize)>, f64, usize);
fn summarize_outcomes(all: &[(Artifact, usize, Value)]) -> Result<OutcomeSummary> {
    let mut best: Option<(f64, Artifact, usize)> = None;
    let mut runtime = 0.;
    let mut failures = 0;
    for (artifact, index, row) in all {
        let cost = row["runtime_seconds"]
            .as_f64()
            .context("original measured cost absent")?;
        ensure!(
            cost.is_finite() && cost >= 0.,
            "invalid original measured cost"
        );
        runtime += cost;
        if !row["failure"].is_null() {
            failures += 1;
            continue;
        }
        let quality = row["metrics"]["rmse"]
            .as_f64()
            .context("original quality absent")?;
        ensure!(
            quality.is_finite() && quality >= 0.,
            "invalid original quality"
        );
        if best.as_ref().is_none_or(|old| quality < old.0) {
            best = Some((quality, artifact.clone(), *index));
        }
    }
    ensure!(runtime.is_finite(), "continuation cost overflow");
    Ok((best, runtime, failures))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn omitted_overlay_preserves_original_task_serialization() {
        let t = task();
        let b = serde_json::to_value(&t).unwrap();
        assert!(b.get("preparation_overlay").is_none());
        let decoded: GenomeTask = serde_json::from_value(b.clone()).unwrap();
        assert!(decoded.preparation_overlay.is_none());
        assert_eq!(serde_json::to_value(decoded).unwrap(), b);
    }
    fn task() -> GenomeTask {
        let a = Artifact {
            path: "a.json".into(),
            sha256: "a".repeat(64),
        };
        GenomeTask {
            preparation_overlay: None,
            schema: "jope.native_genome_task.v1".into(),
            stage: Stage::Direct,
            candidate: a.clone(),
            output: "out".into(),
            vocabulary: a.clone(),
            admission_policy: a,
            qualification: false,
            evaluations: 1,
            seconds: 30,
            rng_seed: 7,
            direct_parent: None,
        }
    }
    #[test]
    fn continuation_keeps_original_parent_until_measured_descendant_improves_it() {
        let parent = Artifact {
            path: "direct/observations.json".into(),
            sha256: "a".repeat(64),
        };
        let child = Artifact {
            path: "continuation/observations.json".into(),
            sha256: "b".repeat(64),
        };
        let mut rows = vec![
            (
                parent.clone(),
                0,
                json!({"runtime_seconds":2.,"failure":null,"metrics":{"rmse":1.}}),
            ),
            (
                child.clone(),
                0,
                json!({"runtime_seconds":3.,"failure":"timeout","metrics":null}),
            ),
            (
                child.clone(),
                1,
                json!({"runtime_seconds":4.,"failure":null,"metrics":{"rmse":2.}}),
            ),
        ];
        let (best, cost, failures) = summarize_outcomes(&rows).unwrap();
        assert_eq!(best.unwrap().1, parent);
        assert_eq!(cost, 9.);
        assert_eq!(failures, 1);
        rows.push((
            child.clone(),
            2,
            json!({"runtime_seconds":5.,"failure":null,"metrics":{"rmse":0.5}}),
        ));
        let (best, cost, failures) = summarize_outcomes(&rows).unwrap();
        assert_eq!(best.unwrap(), (0.5, child, 2));
        assert_eq!(cost, 14.);
        assert_eq!(failures, 1);
    }
    #[test]
    fn all_failed_outcomes_do_not_invent_a_quality_target() {
        let a = Artifact {
            path: "original.json".into(),
            sha256: "a".repeat(64),
        };
        let rows = vec![(
            a,
            0,
            json!({"runtime_seconds":1.5,"failure":"invalid","metrics":null}),
        )];
        let (best, cost, failed) = summarize_outcomes(&rows).unwrap();
        assert!(best.is_none());
        assert_eq!(cost, 1.5);
        assert_eq!(failed, 1);
    }
    #[test]
    fn exact_direct_and_original_parent_contract_cannot_be_relabelled() {
        let mut t = task();
        assert!(t.validate().is_ok());
        t.evaluations = 2;
        assert!(t.validate().is_err());
        t.stage = Stage::Continuation;
        assert!(t.validate().is_err());
        t.direct_parent = Some(t.candidate.clone());
        assert!(t.validate().is_ok());
        t.stage = Stage::Direct;
        assert!(t.validate().is_err());
    }
    #[test]
    fn sealed_partition_and_wrong_program_rejected_before_fit() {
        let tmp = tempfile::tempdir().unwrap();
        let mut t = task();
        let c = json!({"schema":"jope.native_candidate.v1","run":"r","generation":0,"arm":"model","matched_slot":0,"context":{"dataset":"d","family":"f","cut":0,"partition":"meta_test","path":"x","sha256":"a".repeat(64),"prepared":"p","prepared_sha256":"a".repeat(64)},"genome":"(pipe X (ridge))","genome_sha256":hash(b"(pipe X (ridge))"),"proposal":t.candidate,"parent_checkpoint":t.candidate,"sampling_seed":1,"origin_capture":t.candidate,"origin_row":0,"generic_draw":null});
        let bytes = serde_json::to_vec(&c).unwrap();
        fs::write(tmp.path().join("a.json"), &bytes).unwrap();
        t.candidate.sha256 = hash(&bytes);
        assert!(t
            .candidate(tmp.path())
            .err()
            .unwrap()
            .to_string()
            .contains("sealed partition"));
    }
    fn preparation(clean: bool) -> Vec<u8> {
        serde_json::to_vec(&json!({"schema":"parallel.prepared.v1","dataset":"d","family":"f",
            "task":"regression","meta_partition":"meta_train","seed":1,"family_partition_frozen":true,
            "clean_evaluation":clean,"clean_status":"fixture","original_train_sha256":"0".repeat(64),
            "original_test_sha256":"1".repeat(64),"selected_train_indices":[0],
            "selected_test_indices":[],"target_last":true,"rows_total":1,"cuts":[]}))
        .unwrap()
    }
    /// A direct task whose candidate binds `prepared` by digest. Its
    /// vocabulary is deliberately unbound, so a candidate that passes the
    /// scientific boundary stops at the vocabulary.
    fn task_binding(root: &Path, prepared: &[u8]) -> GenomeTask {
        let mut t = task();
        fs::create_dir_all(root.join("p")).unwrap();
        fs::write(root.join("p/prepared.json"), prepared).unwrap();
        let c = json!({"schema":"jope.native_candidate.v1","run":"r","generation":0,"arm":"model","matched_slot":0,"context":{"dataset":"d","family":"f","cut":0,"partition":"meta_train","path":"x","sha256":"a".repeat(64),"prepared":"p","prepared_sha256":hash(prepared)},"genome":"(pipe X (ridge))","genome_sha256":hash(b"(pipe X (ridge))"),"proposal":t.candidate,"parent_checkpoint":t.candidate,"sampling_seed":1,"origin_capture":t.candidate,"origin_row":0,"generic_draw":null});
        let bytes = serde_json::to_vec(&c).unwrap();
        fs::write(root.join("a.json"), &bytes).unwrap();
        t.candidate.sha256 = hash(&bytes);
        t
    }
    /// Replace `target` with `bytes` the first time a receiver reports it verified.
    fn replace_on_verification(
        target: &Path,
        bytes: Vec<u8>,
    ) -> crate::manifest::verification_hook::Guard {
        let target = fs::canonicalize(target).unwrap();
        let mut done = false;
        crate::manifest::verification_hook::install(move |path| {
            if !done && fs::canonicalize(path).ok().as_deref() == Some(target.as_path()) {
                done = true;
                let staged = target.with_extension("replacement");
                fs::write(&staged, &bytes).unwrap();
                fs::rename(&staged, &target).unwrap();
            }
        })
    }
    fn refusal(t: &GenomeTask, root: &Path) -> String {
        match t.candidate(root) {
            Ok(_) => "accepted".into(),
            Err(e) => format!("{e:#}"),
        }
    }
    /// W1 route R8 (native_genome.rs:195-200 before the fix): the candidate's
    /// scientific boundary is decided on exactly the preparation bytes whose
    /// digest the candidate binds.
    #[test]
    fn the_candidate_boundary_is_decided_on_the_verified_preparation() {
        const BOUNDARY: &str = "candidate scientific boundary changed";
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let prepared = root.join("p/prepared.json");
        // A bound preparation that is not a clean evaluation is refused, also
        // when a clean preparation replaces it after its digest check.
        let t = task_binding(root, &preparation(false));
        assert!(refusal(&t, root).contains(BOUNDARY));
        {
            let _hook = replace_on_verification(&prepared, preparation(true));
            let refused = refusal(&t, root);
            assert!(
                refused.contains(BOUNDARY),
                "the boundary was decided on a preparation other than the verified one: {refused}"
            );
        }
        // Positive control: a bound clean preparation passes the boundary (and
        // stops at the unbound vocabulary), also when replaced after its check.
        let t = task_binding(root, &preparation(true));
        let stopped = refusal(&t, root);
        assert!(
            !stopped.contains(BOUNDARY) && stopped.contains("native genome artifact"),
            "{stopped}"
        );
        let _hook = replace_on_verification(&prepared, preparation(false));
        let stopped = refusal(&t, root);
        assert!(
            !stopped.contains(BOUNDARY),
            "the boundary was decided on a preparation other than the verified one: {stopped}"
        );
    }
    fn referencing(root: &Path, target: &str) -> Vec<u8> {
        let digest = hash(&fs::read(root.join(target)).unwrap());
        serde_json::to_vec(&json!({"input":{"path":target,"sha256":digest}})).unwrap()
    }
    /// W1 route R8 (required_inputs :484-490 before the fix): the input
    /// closure follows the nested references of exactly the verified bytes.
    #[test]
    fn the_input_closure_follows_only_the_verified_bytes() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().canonicalize().unwrap();
        fs::write(root.join("x.txt"), b"bound dependency").unwrap();
        fs::write(root.join("y.txt"), b"other dependency").unwrap();
        let bound = referencing(&root, "x.txt");
        let other = referencing(&root, "y.txt");
        fs::write(root.join("a.json"), &bound).unwrap();
        let a = Artifact {
            path: "a.json".into(),
            sha256: hash(&bound),
        };
        let follows = |c: &BTreeMap<PathBuf, String>| {
            (
                c.contains_key(Path::new("x.txt")),
                c.contains_key(Path::new("y.txt")),
            )
        };
        // Positive control.
        let closure = artifact_closure(&root, vec![a.clone()]).unwrap();
        assert_eq!(follows(&closure), (true, false));
        // Negative control: other bytes than the bound ones are refused.
        fs::write(root.join("a.json"), &other).unwrap();
        let refused = format!(
            "{:#}",
            artifact_closure(&root, vec![a.clone()]).unwrap_err()
        );
        assert!(refused.contains("native genome input changed"), "{refused}");
        fs::write(root.join("a.json"), &bound).unwrap();
        // The input is replaced right after its digest check.
        let _hook = replace_on_verification(&root.join("a.json"), other);
        let raced = artifact_closure(&root, vec![a]).unwrap();
        assert_eq!(
            follows(&raced),
            (true, false),
            "the closure followed references of bytes other than the verified ones: {raced:?}"
        );
    }
    /// W1 route R8 (required_inputs :493-516 before the fix): the closure
    /// records the preparation the candidate bound, and its original
    /// observation source is read from exactly those bytes.
    #[test]
    fn the_prepared_closure_is_the_preparation_the_candidate_bound() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().canonicalize().unwrap();
        fs::create_dir_all(root.join("p")).unwrap();
        fs::write(root.join("s1.csv"), b"x,y\n1,2\n").unwrap();
        fs::write(root.join("s2.csv"), b"x,y\n3,4\n").unwrap();
        let with_source = |source: &str| {
            let mut v: Value = serde_json::from_slice(&preparation(true)).unwrap();
            v["storage"] = json!({"source":source,
                "source_sha256":hash(&fs::read(root.join(source)).unwrap()),
                "feature_columns":["x"],"target_column":"y","original_training_row_map":[0]});
            serde_json::to_vec(&v).unwrap()
        };
        let bound = with_source("s1.csv");
        let other = with_source("s2.csv");
        let prepared = root.join("p/prepared.json");
        fs::write(&prepared, &bound).unwrap();
        let context: ContextRow = serde_json::from_value(json!({"dataset":"d","family":"f",
            "cut":0,"partition":"meta_train","path":"x","sha256":"a".repeat(64),"prepared":"p",
            "prepared_sha256":hash(&bound)}))
        .unwrap();
        let closure = || {
            let mut out = BTreeMap::new();
            prepared_closure(&root, &context, &mut out).map(|()| out)
        };
        // Positive control.
        let out = closure().unwrap();
        assert_eq!(out[Path::new("p/prepared.json")], hash(&bound));
        assert!(out.contains_key(Path::new("s1.csv")) && !out.contains_key(Path::new("s2.csv")));
        // The candidate's check (`GenomeTask::candidate`) passes, and the
        // preparation is replaced right after it.
        let _hook = replace_on_verification(&prepared, other);
        read_verified(&prepared, &context.prepared_sha256, "candidate preparation").unwrap();
        let raced = closure().map_err(|e| format!("{e:#}"));
        assert!(
            matches!(&raced, Err(e) if e.contains("native genome input changed")),
            "the closure recorded a preparation other than the one the candidate bound: {raced:?}"
        );
    }
    #[test]
    fn large_corpus_metadata_is_explicitly_typed_and_hash_bound() {
        let tmp = tempfile::tempdir().unwrap();
        let bytes = serde_json::to_vec(&json!({"schema":"parallel.teaching_shards.v1","shards":[],"retained_metadata":"x".repeat(17 * 1024 * 1024)})).unwrap();
        fs::write(tmp.path().join("index.json"), &bytes).unwrap();
        let mut artifact = Artifact {
            path: "index.json".into(),
            sha256: hash(&bytes),
        };
        assert!(load::<Value>(tmp.path(), &artifact).is_err());
        assert_eq!(
            load_corpus_index(tmp.path(), &artifact).unwrap()["schema"],
            "parallel.teaching_shards.v1"
        );
        assert!(load_dataset_receipt(tmp.path(), &artifact).is_err());
        artifact.sha256 = "0".repeat(64);
        assert!(load_corpus_index(tmp.path(), &artifact).is_err());
    }
    #[test]
    fn plain_seeded_continuation_admits_only_measured_children() {
        assert_eq!(CONTINUATION_PROTOCOL, "plain_seeded_continuation_v1");
        assert_eq!(MINIMUM_COMPLETED_CHILDREN, 1);
        assert_eq!(completed_children(&[]).unwrap(), 0);
        assert_eq!(training_admission(0), REFUSED_NO_COMPLETED_CHILD);
        let failed = vec![
            json!({"runtime_seconds":30.,"failure":"candidate_or_view_timeout","metrics":null}),
            json!({"runtime_seconds":1.,"failure":"invalid","metrics":null}),
        ];
        assert_eq!(completed_children(&failed).unwrap(), 0);
        assert_eq!(training_admission(0), REFUSED_NO_COMPLETED_CHILD);
        let mut measured = failed.clone();
        measured.push(json!({"runtime_seconds":2.,"failure":null,"metrics":{"rmse":0.25}}));
        assert_eq!(completed_children(&measured).unwrap(), 1);
        assert_eq!(training_admission(1), TRAINING_ADMITTED);
        let absent = json!({"runtime_seconds":2.,"failure":null,"metrics":{"rmse":null}});
        assert!(completed_children(&[absent]).is_err());
        let negative = json!({"runtime_seconds":2.,"failure":null,"metrics":{"rmse":-1.}});
        assert!(completed_children(&[negative]).is_err());
    }
}
