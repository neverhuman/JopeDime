//! Paired-arm comparison manifest (W5-T02).
//!
//! [`compare`] replays every arm of one comparison independently
//! ([`super::selection_replay`]) and requires their frozen search requests
//! (preparation, cut and folds, evaluator, budget, seed, random-stream and
//! selection contracts, vocabulary, seed population, ...) to be identical
//! except for the declared arm difference, which must actually differ. Each
//! arm's feasibility rules must be the same rule, and a program measured in
//! more than one arm must carry the identical measurement in each: an origin
//! label or arm order cannot change what the evaluator returns.
//!
//! The manifest records, per arm, non-overlapping counters (proposal draws,
//! rejected proposals, unique programs, completed, measured and failed
//! evaluations, physical fit starts and reused completions), its generation-zero
//! inventory, stop reason and replayed winner, plus any shared program whose
//! outcome diverged between arms (for example a timeout in one arm only), so
//! every observed difference stays auditable rather than averaged away.
use super::selection_replay;
use crate::manifest::read_json;
use anyhow::{ensure, Context, Result};
use serde_json::{json, Map, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};

pub const PAIRED_SCHEMA: &str = "parallel.gp_paired_comparison.v1";
/// Request fields that locate an arm rather than define its semantics.
const LOCATION_KEYS: [&str; 1] = ["output"];

/// One completed search taking part in the comparison.
pub struct Arm<'a> {
    pub label: &'a str,
    pub output: &'a Path,
}

/// Outcome of one program in one arm: a measured (objective, predictions)
/// pair or the explicit failure.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Outcome {
    Measured(u64, String),
    Failed(String),
}

fn outcome(row: &Value, scale: f64) -> Result<Outcome> {
    Ok(if row["failure"].is_null() {
        let rmse = row["metrics"]["rmse"].as_f64().context("measured RMSE")?;
        Outcome::Measured(
            (rmse * scale).to_bits(),
            row["metrics"]["semantic_sha256"]
                .as_str()
                .context("prediction signature")?
                .to_owned(),
        )
    } else {
        Outcome::Failed(row["failure"].as_str().unwrap_or_default().to_owned())
    })
}

pub fn compare(root: &Path, arms: &[Arm], declared: &[&str]) -> Result<Value> {
    ensure!(
        arms.len() >= 2,
        "a paired comparison needs at least two arms"
    );
    let labels: BTreeSet<&str> = arms.iter().map(|arm| arm.label).collect();
    ensure!(labels.len() == arms.len(), "arm labels must be unique");
    let declared: BTreeSet<String> = declared.iter().map(|key| key.to_string()).collect();
    ensure!(
        !declared.is_empty(),
        "a comparison must declare its intended arm difference"
    );
    let mut requests = Vec::new();
    let mut inventories = Vec::new();
    let mut programs = BTreeMap::<String, Vec<(String, Outcome)>>::new();
    let mut feasibility_rules = BTreeSet::new();
    for arm in arms {
        let replayed = selection_replay::verify_search(root, arm.output)
            .with_context(|| format!("arm {}", arm.label))?;
        let output = arm.output.canonicalize()?;
        let request: Value = read_json(&output.join("search-request.json"))?;
        let search: Value = read_json(&output.join("search.json"))?;
        let rows: Vec<Value> = read_json(&output.join("observations.json"))?;
        let replay = &replayed.replay;
        let mut origins = BTreeMap::<String, usize>::new();
        let mut initial = 0usize;
        let mut replaced = 0usize;
        for (position, row) in rows.iter().enumerate() {
            let scale = row["target_scale"].as_f64().context("target scale")?;
            programs
                .entry(
                    row["program_sha256"]
                        .as_str()
                        .context("program identity")?
                        .to_owned(),
                )
                .or_default()
                .push((arm.label.to_owned(), outcome(row, scale)?));
            feasibility_rules.insert(row["feasibility"]["protocol"].as_str().map(str::to_owned));
            if row["initial_seed_slot"].is_null() {
                // A generation-zero slot whose proposal was refused (for example
                // a duplicate after adaptation) is filled by variation instead.
                replaced +=
                    usize::from(position < 80 && request.get("native_genome_stage").is_none());
            } else {
                initial += 1;
                *origins
                    .entry(row["origin"].as_str().unwrap_or("unknown").to_owned())
                    .or_default() += 1;
            }
        }
        let accounting = &search["execution_accounting"];
        inventories.push(json!({"label":arm.label,
            "output":request["output"],"search_id":replayed.search_id,
            "observations_sha256":replayed.observations_sha256,"stop":replay.stop,
            "winner":replay.winner.as_ref().map(|w| json!({"position":w.position,
                "program":w.genome,"objective_rmse_original_units":w.objective})),
            "counters":{"proposal_draws":search["proposal_attempts"],
                "rejected_proposals":search["rejected_proposals"],
                "unique_programs":search["unique_programs"],
                "completed_evaluations":rows.len(),"measured":replay.measured,
                "failed":replay.failed,"semantic_duplicates":replay.semantic_duplicates,
                "physical_fit_starts":accounting["physical_fit_starts"],
                "recorded_physical_fit_starts":accounting["recorded_physical_fit_starts"],
                "reused_completions":accounting["reused_completions"],
                "unknown_start_outcomes":accounting["unknown_start_outcomes"]},
            "generation_zero":{"initial_slots_evaluated":initial,"origins":origins,
                "initial_slots_replaced_by_variation":replaced}}));
        requests.push(request);
    }
    // Every frozen request field is identical across arms except the declared
    // difference, and every declared difference is real.
    let keys: BTreeSet<&String> = requests
        .iter()
        .filter_map(Value::as_object)
        .flat_map(Map::keys)
        .collect();
    let differing: BTreeSet<String> = keys
        .into_iter()
        .filter(|key| !LOCATION_KEYS.contains(&key.as_str()))
        .filter(|key| {
            requests
                .iter()
                .any(|request| request.get(key.as_str()) != requests[0].get(key.as_str()))
        })
        .cloned()
        .collect();
    ensure!(
        differing == declared,
        "arm request differences {differing:?} are not exactly the declared {declared:?}"
    );
    ensure!(
        feasibility_rules.len() <= 1,
        "arms recorded different feasibility rules: {feasibility_rules:?}"
    );
    // A program measured in several arms carries one measurement.
    let mut shared = 0usize;
    let mut divergent = Vec::new();
    for (program, outcomes) in &programs {
        let arms_seen: BTreeSet<&String> = outcomes.iter().map(|(arm, _)| arm).collect();
        if arms_seen.len() < 2 {
            continue;
        }
        shared += 1;
        let measured: BTreeSet<&Outcome> = outcomes
            .iter()
            .map(|(_, o)| o)
            .filter(|o| matches!(o, Outcome::Measured(..)))
            .collect();
        ensure!(
            measured.len() <= 1,
            "program {program} carries different measurements in different arms"
        );
        if outcomes
            .iter()
            .any(|(_, o)| !matches!(o, Outcome::Measured(..)))
            && !measured.is_empty()
        {
            divergent.push(json!({"program_sha256":program,"outcomes":outcomes.iter()
                .map(|(arm, o)| json!({"arm":arm,"measured":matches!(o, Outcome::Measured(..)),
                    "failure":match o { Outcome::Failed(f) => json!(f), _ => Value::Null }}))
                .collect::<Vec<_>>()}));
        }
    }
    Ok(
        json!({"schema":PAIRED_SCHEMA,"selection_contract":selection_replay::CONTRACT,
        "declared_arm_difference":declared,"observed_request_differences":differing,
        "feasibility_protocol":feasibility_rules.into_iter().next().flatten(),
        "shared_programs":shared,"shared_programs_with_divergent_outcomes":divergent,
        "arms":inventories}),
    )
}
