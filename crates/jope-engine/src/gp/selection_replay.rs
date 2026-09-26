//! Independent selector and stopping replay for one completed GP search (W5-T06).
//!
//! The searcher in `gp.rs` keeps an in-memory ranked list and publishes only its
//! first entry. This module re-derives the measured winner, the complete measured
//! ranking and the terminal stop decision from the durable ledger alone, under
//! the frozen contract below. It never calls the searcher's ranking, parent
//! selection or stop code: it reads `search-request.json`, the append-only
//! `observations.jsonl`, the published `observations.json`, `progress.json`,
//! `search.json` and each measured row's original completion evidence.
//!
//! Frozen contract `parallel.gp_selection_contract.v1` (regression, one cut):
//!
//! * **Objective.** Minimise the measured validation RMSE in original target
//!   units, `metrics.rmse * target_scale`, where `target_scale` comes from the
//!   cut's `preprocessing.json` bound by the request's `preprocessing_sha256`.
//! * **Evidence completeness.** A row is a measured candidate only when its
//!   original failure is null, its single cut measurement is complete (finite,
//!   non-negative RMSE, a prediction signature, the frozen validation partition)
//!   and its original completion/fit/model/output evidence verifies. A failed,
//!   timed-out, denied or unobserved execution is retained as a failed row and is
//!   never ranked. Predicted, proposal or reward scores carried in any field are
//!   ignored: only `metrics.rmse` is read.
//! * **Distinctness.** Candidates with identical prediction signatures are one
//!   ranked entry; the earliest ledger position represents it. An inherited
//!   continuation incumbent occupies position -1, before every ledger row.
//! * **Ties.** Equal objectives (IEEE total order) resolve to the earlier ledger
//!   position, so the result never depends on row arrival or hash order.
//! * **Stop.** The ledger never exceeds `evaluations`; no row is dispatched after
//!   an earlier row completed at or beyond the `seconds` deadline; every proposal
//!   drawn after the last row was refused. The terminal reason is exactly one of
//!   the evaluation budget, the search deadline, or one slot's full
//!   10,000-proposal cap without a novel program.
//!
//! Anything the contract cannot explain refuses the whole ledger rather than
//! producing a winner: reordered rows, identity changes, duplicate programs,
//! success rows without a valid measurement, failed rows carrying quality,
//! semantic duplicates with different measurements, late dispatch and
//! unexplained early stops.
use crate::manifest::{file_hash, hash, read_json};
use anyhow::{bail, ensure, Context, Result};
use serde::Serialize;
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

pub const CONTRACT: &str = "parallel.gp_selection_contract.v1";
/// The only random-stream protocol whose ledgers this contract replays.
const RANDOM_PROTOCOL: &str = "parallel.gp_random_streams.v2";
const OBSERVATION_SCHEMA: &str = "parallel.gp_observation.v1";
const SEARCH_CLOCK: &str = "parallel.gp_search_clock.v1";
const SCORE_IDENTITY_SCHEMA: &str = "parallel.gp_score_identity.v1";
/// Novel-program attempts a single slot may draw before the search stops.
pub const PROPOSAL_CAP_PER_SLOT: u64 = 10_000;
/// Request identity every ledger row must repeat exactly.
const ROW_IDENTITY: [&str; 9] = [
    "prepared_sha256",
    "cut",
    "evaluator_sha256",
    "training_sha256",
    "validation_sha256",
    "preprocessing_sha256",
    "seeds_sha256",
    "vocabulary_sha256",
    "rng_seed",
];
/// Evaluation identity an inherited incumbent must share with the continuation.
const INCUMBENT_IDENTITY: [&str; 7] = [
    "prepared_sha256",
    "cut",
    "evaluator_sha256",
    "training_sha256",
    "validation_sha256",
    "preprocessing_sha256",
    "vocabulary_sha256",
];

/// The request-frozen parameters of one search.
#[derive(Clone, Debug)]
pub struct Frozen {
    identity: BTreeMap<&'static str, Value>,
    pub evaluations: usize,
    pub seconds: u64,
    pub target_scale: f64,
}

impl Frozen {
    /// Bind the contract to an accepted `parallel.search_request.v1`.
    pub fn from_request(request: &Value, target_scale: f64) -> Result<Self> {
        ensure!(
            request["schema"] == "parallel.search_request.v1",
            "unsupported search request"
        );
        ensure!(
            request["random_protocol"] == RANDOM_PROTOCOL,
            "search request random protocol is outside the replay contract"
        );
        // Requests that froze their selection rules must have frozen these;
        // older requests predate the field and are replayed under this contract.
        if let Some(contract) = request.get("selection_contract") {
            ensure!(
                contract == CONTRACT && request["proposal_cap_per_slot"] == PROPOSAL_CAP_PER_SLOT,
                "search was frozen under another selection contract"
            );
        }
        ensure!(
            target_scale.is_finite() && target_scale > 0.0,
            "invalid frozen target scale"
        );
        let evaluations = request["evaluations"]
            .as_u64()
            .context("request evaluation budget absent")?;
        let seconds = request["seconds"]
            .as_u64()
            .context("request search deadline absent")?;
        ensure!(
            (1..=1000).contains(&evaluations) && (1..=600).contains(&seconds),
            "request budget outside the admitted search range"
        );
        let mut identity = BTreeMap::new();
        for key in ROW_IDENTITY {
            let value = request
                .get(key)
                .with_context(|| format!("request identity {key} absent"))?;
            identity.insert(key, value.clone());
        }
        ensure!(
            identity["rng_seed"].is_u64() && identity["cut"].is_u64(),
            "request cut/seed identity malformed"
        );
        Ok(Self {
            identity,
            evaluations: evaluations as usize,
            seconds,
            target_scale,
        })
    }
}

/// Terminal counters the searcher published after its last observation.
#[derive(Clone, Copy, Debug)]
pub struct Terminal {
    pub wall_seconds: f64,
    pub proposal_attempts: u64,
    pub rejected_proposals: u64,
}

/// A continuation's inherited, previously measured direct parent.
#[derive(Clone, Debug)]
pub struct Incumbent {
    genome: String,
    objective: f64,
    semantic_sha256: String,
}

impl Incumbent {
    /// `None` for a failed parent: a failure is never an inherited score.
    pub fn from_parent_row(frozen: &Frozen, row: &Value) -> Result<Option<Self>> {
        if !row["failure"].is_null() {
            return Ok(None);
        }
        for key in INCUMBENT_IDENTITY {
            ensure!(
                row[key] == frozen.identity[key],
                "inherited incumbent was measured under a different {key}"
            );
        }
        ensure!(
            same_scale(&row["target_scale"], frozen.target_scale),
            "inherited incumbent was measured under a different target scale"
        );
        let genome = row["genome"]
            .as_str()
            .context("inherited incumbent program absent")?;
        ensure!(
            hash(genome.as_bytes()) == row["program_sha256"],
            "inherited incumbent program changed"
        );
        let (objective, semantic_sha256) = measurement(frozen, &row["metrics"])
            .context("inherited incumbent measurement invalid")?;
        Ok(Some(Self {
            genome: genome.into(),
            objective,
            semantic_sha256,
        }))
    }
}

/// One distinct measured candidate in the replayed ranking.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Ranked {
    /// Ledger position; -1 is an inherited continuation incumbent.
    pub position: i64,
    pub genome: String,
    pub objective: f64,
    pub semantic_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Eligibility {
    Measured { objective: f64 },
    SemanticDuplicate { objective: f64, of_position: i64 },
    Failed { failure: String },
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RowDecision {
    pub position: usize,
    pub program_sha256: String,
    pub eligibility: Eligibility,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EvaluationBudget,
    SearchDeadline,
    ProposalSpaceExhausted,
}

#[derive(Clone, Debug, Serialize)]
pub struct Replay {
    pub contract: &'static str,
    pub rows: Vec<RowDecision>,
    pub ranking: Vec<Ranked>,
    pub winner: Option<Ranked>,
    pub stop: StopReason,
    pub measured: usize,
    pub failed: usize,
    pub semantic_duplicates: usize,
    pub inherited_incumbent: bool,
}

fn is_sha256(value: &Value) -> bool {
    value
        .as_str()
        .is_some_and(|s| s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')))
}

fn same_scale(value: &Value, scale: f64) -> bool {
    value
        .as_f64()
        .is_some_and(|v| v.to_bits() == scale.to_bits())
}

/// The complete single-cut measurement or a refusal. Only `rmse` is read.
fn measurement(frozen: &Frozen, metrics: &Value) -> Result<(f64, String)> {
    let metrics = metrics
        .as_object()
        .context("success claimed without a measurement")?;
    let rmse = metrics
        .get("rmse")
        .and_then(Value::as_f64)
        .context("measurement lacks a numeric RMSE")?;
    ensure!(
        rmse.is_finite() && rmse >= 0.0,
        "measurement RMSE is nonfinite or negative"
    );
    let objective = rmse * frozen.target_scale;
    ensure!(objective.is_finite(), "measured objective overflows");
    let signature = metrics
        .get("semantic_sha256")
        .filter(|value| is_sha256(value))
        .and_then(Value::as_str)
        .context("measurement lacks a prediction signature")?;
    // A score identity (W5-T03) binds the score to its fitted subject, scored
    // rows and protocol; it must name this fold and these predictions.
    if let Some(identity) = metrics.get("score_identity").filter(|v| !v.is_null()) {
        ensure!(
            metrics.get("score_identity_sha256")
                == Some(&Value::String(hash(&serde_json::to_vec(identity)?))),
            "score identity digest differs"
        );
        ensure!(
            identity["schema"] == SCORE_IDENTITY_SCHEMA
                && identity["scoring_rows_sha256"] == frozen.identity["validation_sha256"]
                && identity["training_rows_sha256"] == frozen.identity["training_sha256"]
                && identity["preprocessing_sha256"] == frozen.identity["preprocessing_sha256"]
                && identity["evaluator_protocol"] == crate::label_admission::PROTOCOL
                && identity["predictions_sha256"] == signature,
            "score identity names another fold, protocol or prediction set"
        );
    }
    // Where the evaluator recorded its scoring membership, it must be the frozen
    // validation partition and training fold; a score from other rows is stale.
    if let Some(baseline) = metrics.get("baseline_evidence").filter(|v| !v.is_null()) {
        ensure!(
            baseline["scoring_sha256"] == frozen.identity["validation_sha256"]
                && baseline["training_sha256"] == frozen.identity["training_sha256"],
            "measurement scored rows outside the frozen fold"
        );
    }
    Ok((objective, signature.to_owned()))
}

/// Replay one ledger. `evidence` authenticates a measured row's original
/// completion before it may be ranked; its refusal refuses the ledger.
pub fn replay_ledger(
    frozen: &Frozen,
    inherited: Option<&Incumbent>,
    rows: &[Value],
    terminal: &Terminal,
    evidence: &mut dyn FnMut(usize, &Value) -> Result<()>,
) -> Result<Replay> {
    ensure!(
        rows.len() <= frozen.evaluations,
        "ledger holds {} rows beyond the admitted {} evaluations",
        rows.len(),
        frozen.evaluations
    );
    let mut programs = BTreeSet::new();
    let mut signatures = BTreeMap::<String, (i64, f64)>::new();
    let mut ranking = Vec::new();
    if let Some(parent) = inherited {
        programs.insert(feat_math::GenomeIr::parse(&parent.genome)?.canonical());
        signatures.insert(parent.semantic_sha256.clone(), (-1, parent.objective));
        ranking.push(Ranked {
            position: -1,
            genome: parent.genome.clone(),
            objective: parent.objective,
            semantic_sha256: parent.semantic_sha256.clone(),
        });
    }
    let mut decisions = Vec::with_capacity(rows.len());
    let (mut attempts, mut rejected, mut clock) = (0u64, 0u64, 0.0f64);
    for (position, row) in rows.iter().enumerate() {
        let context = || format!("ledger row {position}");
        ensure!(
            row["schema"] == OBSERVATION_SCHEMA,
            "{}: unsupported observation schema",
            context()
        );
        ensure!(
            row["evaluation_index"].as_u64() == Some(position as u64),
            "{}: ledger order differs from evaluation order",
            context()
        );
        for key in ROW_IDENTITY {
            ensure!(
                row[key] == frozen.identity[key],
                "{}: {key} differs from the frozen request",
                context()
            );
        }
        ensure!(
            same_scale(&row["target_scale"], frozen.target_scale)
                && row["evaluator_protocol"] == crate::label_admission::PROTOCOL
                && row["resource_protocol"] == crate::label_admission::RESOURCE
                && row["metric_units"] == "cut_normalized_target",
            "{}: evaluator protocol, units or scale differ",
            context()
        );
        let genome = row["genome"].as_str().with_context(context)?;
        ensure!(
            hash(genome.as_bytes()) == row["program_sha256"],
            "{}: executed program identity differs",
            context()
        );
        ensure!(
            row["evaluation_identity_sha256"]
                == hash(&serde_json::to_vec(&crate::label_admission::identity(row))?),
            "{}: evaluation identity differs",
            context()
        );
        ensure!(
            programs.insert(feat_math::GenomeIr::parse(genome)?.canonical()),
            "{}: program evaluated twice",
            context()
        );
        let row_attempts = row["proposal_attempts"].as_u64().with_context(context)?;
        let row_rejected = row["rejected_proposals"].as_u64().with_context(context)?;
        ensure!(
            row_attempts > attempts
                && row_rejected >= rejected
                && row_attempts >= row_rejected.saturating_add(position as u64 + 1),
            "{}: proposal accounting is not monotonic",
            context()
        );
        (attempts, rejected) = (row_attempts, row_rejected);
        let completed = row["completed_search_seconds"]
            .as_f64()
            .with_context(|| format!("{}: legacy row without a search clock", context()))?;
        ensure!(
            row["search_clock"] == SEARCH_CLOCK && completed.is_finite() && completed >= clock,
            "{}: search clock is unknown or not monotonic",
            context()
        );
        clock = completed;
        if position + 1 < rows.len() {
            ensure!(
                completed < frozen.seconds as f64,
                "{}: later work was dispatched after the search deadline",
                context()
            );
        }
        let eligibility = if row["failure"].is_null() {
            let (objective, signature) = measurement(frozen, &row["metrics"])
                .with_context(|| format!("{}: success without a valid measurement", context()))?;
            evidence(position, row)
                .with_context(|| format!("{}: original evidence refused", context()))?;
            if let Some(&(of_position, earlier)) = signatures.get(&signature) {
                ensure!(
                    earlier.to_bits() == objective.to_bits(),
                    "{}: identical predictions carry a different measurement",
                    context()
                );
                Eligibility::SemanticDuplicate {
                    objective,
                    of_position,
                }
            } else {
                signatures.insert(signature.clone(), (position as i64, objective));
                ranking.push(Ranked {
                    position: position as i64,
                    genome: genome.into(),
                    objective,
                    semantic_sha256: signature,
                });
                Eligibility::Measured { objective }
            }
        } else {
            let failure = row["failure"]
                .as_str()
                .filter(|s| !s.trim().is_empty())
                .with_context(|| format!("{}: failure is not explicit", context()))?;
            ensure!(
                row["metrics"].is_null(),
                "{}: failed execution carries a measurement",
                context()
            );
            Eligibility::Failed {
                failure: failure.into(),
            }
        };
        decisions.push(RowDecision {
            position,
            program_sha256: row["program_sha256"].as_str().unwrap_or_default().into(),
            eligibility,
        });
    }
    ranking.sort_by(|a, b| {
        a.objective
            .total_cmp(&b.objective)
            .then(a.position.cmp(&b.position))
    });
    ensure!(
        terminal.wall_seconds.is_finite() && terminal.wall_seconds >= 0.0,
        "terminal search clock invalid"
    );
    ensure!(
        terminal.proposal_attempts >= attempts && terminal.rejected_proposals >= rejected,
        "terminal proposal counters regress"
    );
    let drawn = terminal.proposal_attempts - attempts;
    ensure!(
        drawn == terminal.rejected_proposals - rejected,
        "proposals drawn after the last observation were not all refused"
    );
    let stop = if rows.len() == frozen.evaluations {
        ensure!(drawn == 0, "proposals drawn beyond the evaluation budget");
        StopReason::EvaluationBudget
    } else if drawn == PROPOSAL_CAP_PER_SLOT {
        StopReason::ProposalSpaceExhausted
    } else {
        ensure!(
            drawn < PROPOSAL_CAP_PER_SLOT,
            "proposal draws exceed one slot's cap"
        );
        ensure!(
            terminal.wall_seconds >= frozen.seconds as f64,
            "search stopped before its budget, deadline or proposal cap"
        );
        StopReason::SearchDeadline
    };
    let count =
        |f: fn(&Eligibility) -> bool| decisions.iter().filter(|d| f(&d.eligibility)).count();
    Ok(Replay {
        contract: CONTRACT,
        measured: count(|e| !matches!(e, Eligibility::Failed { .. })),
        failed: count(|e| matches!(e, Eligibility::Failed { .. })),
        semantic_duplicates: count(|e| matches!(e, Eligibility::SemanticDuplicate { .. })),
        winner: ranking.first().cloned(),
        rows: decisions,
        ranking,
        stop,
        inherited_incumbent: inherited.is_some(),
    })
}

/// A replay of a completed search directory, checked against its publication.
#[derive(Clone, Debug, Serialize)]
pub struct SearchReplay {
    pub search_id: String,
    pub observations_sha256: String,
    pub replay: Replay,
    /// Result/intent files outside the ledger. Retained for audit, never read
    /// for selection: late, unobserved or unreconciled work cannot be ranked.
    pub unledgered_artifacts: Vec<PathBuf>,
}

/// Complete lines only; a torn tail is unreconciled evidence, not a shorter ledger.
fn read_ledger(path: &Path) -> Result<Vec<Value>> {
    let bytes = fs::read(path).context("observation journal absent")?;
    ensure!(
        bytes.is_empty() || bytes.last() == Some(&b'\n'),
        "observation journal has a torn tail"
    );
    bytes
        .split(|&b| b == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice(line).context("observation journal row"))
        .collect()
}

/// `(continuation, incumbent)`: a continuation whose direct parent failed has no
/// inherited score, yet its parent program is still excluded from the ledger.
fn inherited_incumbent(
    root: &Path,
    request: &Value,
    frozen: &Frozen,
) -> Result<(bool, Option<Incumbent>)> {
    match request.get("native_genome_stage").and_then(Value::as_str) {
        None | Some("direct") => Ok((false, None)),
        Some("continuation") => {
            let reference: crate::native_graph::Artifact =
                serde_json::from_value(request["native_genome_task"].clone())?;
            let task: crate::native_genome::GenomeTask =
                crate::native_genome::load(root, &reference)?;
            let candidate: crate::native_genome::Candidate =
                crate::native_genome::load(root, &task.candidate)?;
            let parent: crate::native_genome::GenomeResult = crate::native_genome::load(
                root,
                task.direct_parent
                    .as_ref()
                    .context("continuation without its direct parent")?,
            )?;
            ensure!(
                parent.stage == crate::native_genome::Stage::Direct
                    && parent.candidate == task.candidate,
                "inherited incumbent is not this candidate's direct evaluation"
            );
            let rows: Vec<Value> = crate::native_genome::load(root, &parent.observations)?;
            ensure!(rows.len() == 1, "direct parent lacks its single outcome");
            ensure!(
                rows[0]["genome"] == candidate.genome,
                "direct parent evaluated another program"
            );
            Ok((true, Incumbent::from_parent_row(frozen, &rows[0])?))
        }
        Some(other) => bail!("unsupported native genome stage {other}"),
    }
}

fn target_scale(root: &Path, request: &Value) -> Result<f64> {
    let prepared = root.join(
        request["prepared"]
            .as_str()
            .context("request preparation path absent")?,
    );
    ensure!(
        file_hash(&prepared.join("prepared.json"))? == request["prepared_sha256"],
        "preparation changed after the search"
    );
    let path = prepared.join(format!(
        "cut-{}/preprocessing.json",
        request["cut"].as_u64().context("request cut absent")?
    ));
    ensure!(
        file_hash(&path)? == request["preprocessing_sha256"],
        "cut preprocessing changed after the search"
    );
    read_json::<Value>(&path)?["target_scale"]
        .as_f64()
        .context("cut target scale absent")
}

fn same_number(published: &Value, replayed: Option<f64>) -> bool {
    match replayed {
        Some(value) => published
            .as_f64()
            .is_some_and(|p| p.to_bits() == value.to_bits()),
        None => published.is_null(),
    }
}

/// Replay a completed search directory and require its publication to agree.
pub fn verify_search(root: &Path, output: &Path) -> Result<SearchReplay> {
    let root = root.canonicalize().context("search artifact root")?;
    let output = output.canonicalize().context("search output")?;
    let request_path = output.join("search-request.json");
    let request: Value = read_json(&request_path)?;
    ensure!(
        request["artifact_root"].as_str() == root.to_str()
            && request["output"].as_str().map(Path::new) == Some(output.strip_prefix(&root)?),
        "search request belongs to another root or output"
    );
    let frozen = Frozen::from_request(&request, target_scale(&root, &request)?)?;
    let search: Value = read_json(&output.join("search.json"))?;
    let search_id = file_hash(&request_path)?;
    ensure!(
        search["schema"] == "parallel.search.v1"
            && search["terminal"] == true
            && search["final_tests_opened"] == false
            && search["search_id"] == search_id
            && search["random_protocol"] == RANDOM_PROTOCOL,
        "search publication is not a terminal search of this request"
    );
    ensure!(
        search["search_request"]["sha256"] == search_id,
        "search publication binds another request"
    );
    for key in [
        "prepared_sha256",
        "cut",
        "evaluator_sha256",
        "vocabulary_sha256",
        "seeds_sha256",
        "rng_seed",
    ] {
        ensure!(
            search[key] == request[key],
            "search publication {key} differs from its request"
        );
    }
    let rows = read_ledger(&output.join("observations.jsonl"))?;
    let snapshot: Vec<Value> = read_json(&output.join("observations.json"))?;
    let observations_sha256 = file_hash(&output.join("observations.json"))?;
    ensure!(
        snapshot == rows && search["observations_sha256"] == observations_sha256,
        "published observations differ from the complete ledger"
    );
    let terminal = Terminal {
        wall_seconds: search["wall_seconds"]
            .as_f64()
            .context("terminal clock absent")?,
        proposal_attempts: search["proposal_attempts"]
            .as_u64()
            .context("terminal proposal attempts absent")?,
        rejected_proposals: search["rejected_proposals"]
            .as_u64()
            .context("terminal rejections absent")?,
    };
    let (continuation, inherited) = inherited_incumbent(&root, &request, &frozen)?;
    let mut evidence = |position: usize, row: &Value| -> Result<()> {
        let program = fs::read(output.join(format!("candidate-{position}.dsl")))?;
        ensure!(
            hash(&program) == row["program_sha256"],
            "executed program bytes differ"
        );
        crate::label_admission::validate_completion(&root, row)?;
        // The score identity must name the fitted subject this row's original
        // completion attests: its fit key and model, not another subject's.
        if let Some(identity) = row["metrics"]
            .get("score_identity")
            .filter(|v| !v.is_null())
        {
            let completion = crate::label_admission::artifact(&root, &row["clean_completion"])?;
            let receipt = crate::eval_registry::verify_completion_receipt(
                &crate::label_admission::artifact_bytes(&root, &completion["fit_completion"])?,
            )?;
            ensure!(
                identity["fit_key"] == receipt.fit_key
                    && identity["model_sha256"] == serde_json::json!(receipt.model_sha256)
                    && completion["model"]["sha256"] == identity["model_sha256"],
                "score identity names another fitted subject"
            );
        }
        Ok(())
    };
    let replay = replay_ledger(&frozen, inherited.as_ref(), &rows, &terminal, &mut evidence)?;
    let winner = replay.winner.as_ref();
    ensure!(
        search["best_program"].as_str() == winner.map(|w| w.genome.as_str())
            && same_number(&search["best_validation_rmse"], winner.map(|w| w.objective)),
        "published winner differs from the independent replay"
    );
    ensure!(
        search["completed_evaluations"] == rows.len()
            && search["successful_measurements"] == replay.measured
            && search["failed_or_unavailable_measurements"] == replay.failed
            && search["unique_programs"] == rows.len()
            && search["inherited_parent_excluded"] == continuation,
        "published search counters differ from the ledger"
    );
    // Searches that publish their terminal disposition must agree with the
    // replayed stop decision; older publications predate the field.
    if let Some(published) = search.get("stop_reason") {
        ensure!(
            *published == serde_json::to_value(replay.stop)?,
            "published stop reason differs from the independent replay"
        );
    }
    let progress: Value = read_json(&output.join("progress.json"))?;
    ensure!(
        progress["evaluations"] == rows.len()
            && (progress.get("best_validation_rmse").is_none()
                || same_number(
                    &progress["best_validation_rmse"],
                    winner.map(|w| w.objective)
                )),
        "progress receipt differs from the replayed ledger"
    );
    // Every candidate file beyond the ledger (an unmeasured intent, a stray or
    // late result), and any metric file of a row the ledger records as failed.
    let mut unledgered = Vec::new();
    for entry in fs::read_dir(&output)? {
        let name = entry?.file_name().to_string_lossy().into_owned();
        let Some(rest) = name.strip_prefix("candidate-") else {
            continue;
        };
        let index = rest.split('.').next().and_then(|i| i.parse::<usize>().ok());
        let listed = match index.and_then(|i| replay.rows.get(i)) {
            None => true,
            Some(row) => {
                rest == format!("{}.json", row.position)
                    && matches!(row.eligibility, Eligibility::Failed { .. })
            }
        };
        if listed {
            unledgered.push(PathBuf::from(name));
        }
    }
    unledgered.sort();
    Ok(SearchReplay {
        search_id,
        observations_sha256,
        replay,
        unledgered_artifacts: unledgered,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const SCALE: f64 = 2.0;

    fn request() -> Value {
        json!({"schema":"parallel.search_request.v1","random_protocol":RANDOM_PROTOCOL,
            "evaluations":6,"seconds":60,"prepared_sha256":"a".repeat(64),"cut":0,
            "evaluator_sha256":"b".repeat(64),"training_sha256":"c".repeat(64),
            "validation_sha256":"d".repeat(64),"preprocessing_sha256":"e".repeat(64),
            "seeds_sha256":null,"vocabulary_sha256":"f".repeat(64),"rng_seed":7})
    }

    fn frozen() -> Frozen {
        Frozen::from_request(&request(), SCALE).unwrap()
    }

    fn program(position: usize) -> String {
        format!("(pipe X (ridge alpha={}))", position + 1)
    }

    fn signature(tag: &str) -> String {
        hash(tag.as_bytes())
    }

    /// A ledger row consistent with the frozen request; `rmse: None` is a failure.
    fn row(position: usize, rmse: Option<f64>, semantic: &str) -> Value {
        let request = request();
        let genome = program(position);
        let mut row = json!({"schema":OBSERVATION_SCHEMA,"evaluation_index":position,
            "genome":genome,"program_sha256":hash(genome.as_bytes()),"target_scale":SCALE,
            "evaluator_protocol":crate::label_admission::PROTOCOL,
            "resource_protocol":crate::label_admission::RESOURCE,
            "metric_units":"cut_normalized_target","family":"f","partition":"meta_train",
            "feature_view_sha256":"0".repeat(64),"evaluation_seed":0,
            "proposal_attempts":position as u64 + 1,"rejected_proposals":0,
            "completed_search_seconds":position as f64,"search_clock":SEARCH_CLOCK,
            "failure":null,"metrics":null});
        for key in ROW_IDENTITY {
            row[key] = request[key].clone();
        }
        match rmse {
            Some(rmse) => {
                row["metrics"] = json!({"rmse":rmse,"semantic_sha256":signature(semantic)});
            }
            None => row["failure"] = json!("candidate_or_view_timeout"),
        }
        rehash(row)
    }

    fn rehash(mut row: Value) -> Value {
        row["evaluation_identity_sha256"] = json!(hash(
            &serde_json::to_vec(&crate::label_admission::identity(&row)).unwrap()
        ));
        row
    }

    fn budget(rows: &[Value]) -> Terminal {
        let last = rows.last();
        Terminal {
            wall_seconds: 1.0,
            proposal_attempts: last.map_or(0, |r| r["proposal_attempts"].as_u64().unwrap()),
            rejected_proposals: last.map_or(0, |r| r["rejected_proposals"].as_u64().unwrap()),
        }
    }

    fn replay(rows: &[Value], terminal: &Terminal) -> Result<Replay> {
        replay_ledger(&frozen(), None, rows, terminal, &mut |_, _| Ok(()))
    }

    fn full(rows: Vec<Value>) -> Vec<Value> {
        assert!(rows.len() <= 6);
        rows
    }

    fn six() -> Vec<Value> {
        full(vec![
            row(0, Some(0.9), "p0"),
            row(1, None, "p1"),
            row(2, Some(0.4), "p2"),
            row(3, Some(0.4), "p2"),
            row(4, Some(0.4), "p4"),
            row(5, Some(0.7), "p5"),
        ])
    }

    #[test]
    fn hand_checked_ledger_selects_the_measured_winner_with_deterministic_ties() {
        let rows = six();
        let r = replay(&rows, &budget(&rows)).unwrap();
        let winner = r.winner.clone().unwrap();
        assert_eq!((winner.position, winner.objective), (2, 0.8));
        assert_eq!(winner.genome, program(2));
        // Distinct measured order: exact tie 2/4 resolves to the earlier row;
        // row 3 repeats row 2's predictions and is not a second entry.
        let order: Vec<i64> = r.ranking.iter().map(|e| e.position).collect();
        assert_eq!(order, [2, 4, 5, 0]);
        assert_eq!(
            r.rows[3].eligibility,
            Eligibility::SemanticDuplicate {
                objective: 0.8,
                of_position: 2
            }
        );
        assert_eq!((r.measured, r.failed, r.semantic_duplicates), (5, 1, 1));
        assert!(matches!(r.rows[1].eligibility, Eligibility::Failed { .. }));
        assert_eq!(r.stop, StopReason::EvaluationBudget);
    }

    #[test]
    fn predicted_and_proposal_scores_never_rank_candidates() {
        let mut rows = six();
        // Reverse every non-measured score against the measured order, and give
        // the failed row the best proposal reward and predicted RMSE of all.
        for (i, row) in rows.iter_mut().enumerate() {
            row["rank_log_weight"] = json!(i as f64 * 10.0);
            row["predicted_rmse"] = json!(1.0 / (i as f64 + 1.0));
            row["proposal_reward"] = json!(100.0 - i as f64);
            row["policy_logprob"] = json!(-(i as f64));
        }
        rows[1]["predicted_rmse"] = json!(0.0);
        rows[1]["proposal_reward"] = json!(1e9);
        let plain = replay(&six(), &budget(&six())).unwrap();
        let scored = replay(&rows, &budget(&rows)).unwrap();
        assert_eq!(scored.winner, plain.winner);
        assert_eq!(scored.ranking, plain.ranking);
        assert_eq!(scored.rows, plain.rows);
    }

    #[test]
    fn nonfinite_negative_and_incomplete_measurements_refuse_the_ledger() {
        let base = six();
        let mut cases: Vec<(Value, &str)> = vec![
            (
                json!({"rmse":"NaN","semantic_sha256":signature("p2")}),
                "numeric RMSE",
            ),
            (
                json!({"rmse":null,"semantic_sha256":signature("p2")}),
                "numeric RMSE",
            ),
            (
                json!({"rmse":-0.1,"semantic_sha256":signature("p2")}),
                "negative",
            ),
            (
                json!({"rmse":1.7e308,"semantic_sha256":signature("p2")}),
                "overflows",
            ),
            (json!({"rmse":0.1}), "prediction signature"),
            (
                json!({"rmse":0.1,"semantic_sha256":"not-a-digest"}),
                "prediction signature",
            ),
            (Value::Null, "without a measurement"),
            (
                json!({"rmse":0.1,"semantic_sha256":signature("p2"),
                "baseline_evidence":{"scoring_sha256":"9".repeat(64),
                "training_sha256":"c".repeat(64)}}),
                "outside the frozen fold",
            ),
        ];
        for (metrics, expected) in cases.drain(..) {
            let mut rows = base.clone();
            rows[2]["metrics"] = metrics;
            let error = format!("{:#}", replay(&rows, &budget(&rows)).unwrap_err());
            assert!(error.contains(expected), "{expected}: {error}");
        }
        // The frozen fold itself is accepted.
        let mut rows = base.clone();
        rows[2]["metrics"]["baseline_evidence"] =
            json!({"scoring_sha256":"d".repeat(64),"training_sha256":"c".repeat(64)});
        assert!(replay(&rows, &budget(&rows)).is_ok());
    }

    #[test]
    fn failures_and_delayed_outcomes_are_retained_but_never_ranked() {
        let mut rows = six();
        // An intent without an observed completion and a timeout both stay failed,
        // even though the timeout was the best predicted program.
        rows[0]["failure"] = json!("unknown_original_execution_without_completion");
        rows[0]["metrics"] = Value::Null;
        rows[2]["failure"] = json!("candidate_or_view_timeout");
        rows[2]["metrics"] = Value::Null;
        let r = replay(&rows, &budget(&rows)).unwrap();
        assert_eq!(r.winner.unwrap().position, 3);
        assert_eq!(r.failed, 3);
        // A failed row that still carries quality is contradictory evidence.
        rows[1]["metrics"] = json!({"rmse":0.0,"semantic_sha256":signature("late")});
        let error = format!("{:#}", replay(&rows, &budget(&rows)).unwrap_err());
        assert!(error.contains("carries a measurement"), "{error}");
        // An empty failure is not an explicit failure.
        let mut rows = six();
        rows[1]["failure"] = json!(" ");
        assert!(replay(&rows, &budget(&rows)).is_err());
        // No measured candidate: a typed no-winner outcome, not a substitute.
        let failed: Vec<Value> = (0..6).map(|i| row(i, None, "x")).collect();
        let r = replay(&failed, &budget(&failed)).unwrap();
        assert!(r.winner.is_none() && r.ranking.is_empty());
    }

    #[test]
    fn reordered_duplicated_or_foreign_rows_refuse_the_ledger() {
        let mut swapped = six();
        swapped.swap(2, 4);
        let error = format!("{:#}", replay(&swapped, &budget(&six())).unwrap_err());
        assert!(error.contains("ledger order"), "{error}");
        let mut duplicate = six();
        duplicate[4]["genome"] = duplicate[2]["genome"].clone();
        duplicate[4]["program_sha256"] = duplicate[2]["program_sha256"].clone();
        let duplicate: Vec<Value> = duplicate.into_iter().map(rehash).collect();
        let error = format!("{:#}", replay(&duplicate, &budget(&duplicate)).unwrap_err());
        assert!(error.contains("evaluated twice"), "{error}");
        let mut conflicting = six();
        conflicting[3]["metrics"]["rmse"] = json!(0.3);
        let error = format!(
            "{:#}",
            replay(&conflicting, &budget(&conflicting)).unwrap_err()
        );
        assert!(error.contains("different measurement"), "{error}");
        for key in ["evaluator_sha256", "validation_sha256", "cut", "rng_seed"] {
            let mut foreign = six();
            foreign[5][key] = if key == "cut" || key == "rng_seed" {
                json!(99)
            } else {
                json!("9".repeat(64))
            };
            let foreign: Vec<Value> = foreign.into_iter().map(rehash).collect();
            let error = format!("{:#}", replay(&foreign, &budget(&foreign)).unwrap_err());
            assert!(error.contains(key), "{key}: {error}");
        }
        let mut scaled = six();
        scaled[0]["target_scale"] = json!(SCALE * 2.0);
        let scaled: Vec<Value> = scaled.into_iter().map(rehash).collect();
        assert!(replay(&scaled, &budget(&scaled)).is_err());
        let mut relabelled = six();
        relabelled[0]["program_sha256"] = json!(hash(b"another program"));
        assert!(replay(&relabelled, &budget(&relabelled)).is_err());
        let mut identity = six();
        identity[0]["evaluation_identity_sha256"] = json!("0".repeat(64));
        assert!(replay(&identity, &budget(&identity)).is_err());
    }

    #[test]
    fn evidence_refusal_refuses_the_ledger_before_ranking() {
        let rows = six();
        let mut calls = Vec::new();
        let result = replay_ledger(&frozen(), None, &rows, &budget(&rows), &mut |i, _| {
            calls.push(i);
            ensure!(i != 4, "completion output differs");
            Ok(())
        });
        let error = format!("{:#}", result.unwrap_err());
        assert!(error.contains("row 4") && error.contains("completion output"));
        // Failed rows are never presented as measured evidence.
        assert_eq!(calls, [0, 2, 3, 4]);
    }

    #[test]
    fn stop_decisions_follow_the_frozen_budget_deadline_and_proposal_cap() {
        let rows = six();
        assert_eq!(
            replay(&rows, &budget(&rows)).unwrap().stop,
            StopReason::EvaluationBudget
        );
        let mut beyond = budget(&rows);
        beyond.proposal_attempts += 1;
        beyond.rejected_proposals += 1;
        assert!(replay(&rows, &beyond).is_err());
        let partial = &rows[..4];
        let mut deadline = budget(partial);
        deadline.wall_seconds = 60.0;
        assert_eq!(
            replay(partial, &deadline).unwrap().stop,
            StopReason::SearchDeadline
        );
        let mut early = budget(partial);
        early.wall_seconds = 59.9;
        let error = format!("{:#}", replay(partial, &early).unwrap_err());
        assert!(error.contains("stopped before"), "{error}");
        early.proposal_attempts += PROPOSAL_CAP_PER_SLOT;
        early.rejected_proposals += PROPOSAL_CAP_PER_SLOT;
        assert_eq!(
            replay(partial, &early).unwrap().stop,
            StopReason::ProposalSpaceExhausted
        );
        early.proposal_attempts += 1;
        early.rejected_proposals += 1;
        assert!(replay(partial, &early).is_err());
        let mut unrefused = budget(partial);
        unrefused.wall_seconds = 60.0;
        unrefused.proposal_attempts += 3;
        unrefused.rejected_proposals += 2;
        let error = format!("{:#}", replay(partial, &unrefused).unwrap_err());
        assert!(error.contains("not all refused"), "{error}");
        // Work dispatched after an earlier row already reached the deadline.
        let mut late = six();
        late[3]["completed_search_seconds"] = json!(60.0);
        late[4]["completed_search_seconds"] = json!(60.5);
        late[5]["completed_search_seconds"] = json!(61.0);
        let error = format!("{:#}", replay(&late, &budget(&late)).unwrap_err());
        assert!(error.contains("row 3") && error.contains("after the search deadline"));
        // The final row itself may complete at the deadline.
        let mut last = six();
        last[5]["completed_search_seconds"] = json!(60.2);
        assert!(replay(&last, &budget(&last)).is_ok());
        let mut regress = six();
        regress[4]["proposal_attempts"] = json!(4);
        assert!(replay(&regress, &budget(&regress)).is_err());
        let mut legacy = six();
        legacy[0]["completed_search_seconds"] = Value::Null;
        assert!(replay(&legacy, &budget(&legacy)).is_err());
        let seven: Vec<Value> = (0..7).map(|i| row(i, Some(1.0), &i.to_string())).collect();
        let error = format!("{:#}", replay(&seven, &budget(&seven)).unwrap_err());
        assert!(error.contains("beyond the admitted"), "{error}");
    }

    #[test]
    fn inherited_incumbent_precedes_rows_and_must_share_the_evaluation_identity() {
        let rows = six();
        let parent = row(0, Some(0.4), "parent");
        let mut parent = parent;
        parent["genome"] = json!("(pipe X (standardize) (ridge))");
        parent["program_sha256"] = json!(hash(b"(pipe X (standardize) (ridge))"));
        parent["rng_seed"] = json!(123);
        let incumbent = Incumbent::from_parent_row(&frozen(), &parent)
            .unwrap()
            .unwrap();
        let r = replay_ledger(
            &frozen(),
            Some(&incumbent),
            &rows,
            &budget(&rows),
            &mut |_, _| Ok(()),
        )
        .unwrap();
        // An exact tie with row 2 keeps the inherited incumbent.
        let winner = r.winner.unwrap();
        assert_eq!((winner.position, winner.objective), (-1, 0.8));
        assert!(r.inherited_incumbent);
        let mut failed = parent.clone();
        failed["failure"] = json!("timeout");
        assert!(Incumbent::from_parent_row(&frozen(), &failed)
            .unwrap()
            .is_none());
        for (key, value) in [
            ("evaluator_sha256", json!("9".repeat(64))),
            ("validation_sha256", json!("9".repeat(64))),
            ("target_scale", json!(SCALE + 1.0)),
        ] {
            let mut foreign = parent.clone();
            foreign[key] = value;
            assert!(
                Incumbent::from_parent_row(&frozen(), &foreign).is_err(),
                "{key}"
            );
        }
        let mut negative = parent;
        negative["metrics"]["rmse"] = json!(-1.0);
        assert!(Incumbent::from_parent_row(&frozen(), &negative).is_err());
    }

    #[test]
    fn score_identities_bind_the_fold_protocol_and_predictions() {
        let rows = six();
        let identity = json!({"schema":SCORE_IDENTITY_SCHEMA,"fit_key":"k","model_sha256":"m",
            "training_rows_sha256":"c".repeat(64),"scoring_rows_sha256":"d".repeat(64),
            "preprocessing_sha256":"e".repeat(64),
            "evaluator_protocol":crate::label_admission::PROTOCOL,
            "predictions_sha256":signature("p2")});
        let with = |identity: &Value| {
            let mut rows = rows.clone();
            rows[2]["metrics"]["score_identity_sha256"] =
                json!(hash(&serde_json::to_vec(identity).unwrap()));
            rows[2]["metrics"]["score_identity"] = identity.clone();
            rows
        };
        let valid = with(&identity);
        assert_eq!(
            replay(&valid, &budget(&valid))
                .unwrap()
                .winner
                .unwrap()
                .position,
            2
        );
        for (key, value) in [
            ("scoring_rows_sha256", json!("9".repeat(64))),
            ("training_rows_sha256", json!("9".repeat(64))),
            ("preprocessing_sha256", json!("9".repeat(64))),
            ("evaluator_protocol", json!("another_protocol")),
            ("predictions_sha256", json!(signature("p4"))),
            ("schema", json!("parallel.gp_score_identity.v0")),
        ] {
            let mut stale = identity.clone();
            stale[key] = value;
            let rows = with(&stale);
            let error = format!("{:#}", replay(&rows, &budget(&rows)).unwrap_err());
            assert!(error.contains("another fold, protocol"), "{key}: {error}");
        }
        let mut digest = with(&identity);
        digest[2]["metrics"]["score_identity"]["fit_key"] = json!("another fit");
        let error = format!("{:#}", replay(&digest, &budget(&digest)).unwrap_err());
        assert!(error.contains("digest differs"), "{error}");
    }

    #[test]
    fn frozen_request_refuses_unknown_protocols_and_budgets() {
        let mut other = request();
        other["random_protocol"] = json!("parallel.gp_random_streams.v1");
        assert!(Frozen::from_request(&other, SCALE).is_err());
        let mut frozen = request();
        frozen["selection_contract"] = json!(CONTRACT);
        frozen["proposal_cap_per_slot"] = json!(PROPOSAL_CAP_PER_SLOT);
        assert!(Frozen::from_request(&frozen, SCALE).is_ok());
        let mut contract = frozen.clone();
        contract["selection_contract"] = json!("parallel.gp_selection_contract.v0");
        assert!(Frozen::from_request(&contract, SCALE).is_err());
        let mut cap = frozen;
        cap["proposal_cap_per_slot"] = json!(1);
        assert!(Frozen::from_request(&cap, SCALE).is_err());
        let mut large = request();
        large["evaluations"] = json!(1001);
        assert!(Frozen::from_request(&large, SCALE).is_err());
        assert!(Frozen::from_request(&request(), f64::NAN).is_err());
        assert!(Frozen::from_request(&request(), 0.0).is_err());
    }
}
