//! Immutable candidate lineage envelope and its verifier (W5-T01).
//!
//! Before dispatch, every GP attempt records one `parallel.gp_candidate_lineage.v1`
//! envelope in `candidate-{i}.attempt.json` (bound by the original completion) and
//! repeats it in the ledger row:
//!
//! * `original` — the exact proposal the source produced;
//! * `source` — a seed-population slot (with its critic weight, which estimates
//!   the *original* proposal only), the fixed reference, a generic bootstrap
//!   generator draw, a native candidate, or a variation event naming its operator,
//!   parent ledger positions and random-stream words;
//! * `adaptation` — at most one explicit vocabulary projection of `original`;
//! * `executable` — the recipe that was fitted.
//!
//! The fold fit is the row's original completion, the selected subject is the
//! independent [`super::selection_replay`] winner, and a final refit is linked by
//! an explicit fitted-identity transformation. The verifier below re-derives every
//! link (generator output, operator output, projection, parent eligibility and
//! the uniform parent draws) instead of trusting labels, so parent swaps, omitted
//! variation events, recipe substitutions and double adaptation are refused.
use super::{selection_replay, Predictor, SeedPopulation};
use crate::manifest::{file_hash, hash, read_json};
use anyhow::{bail, ensure, Context, Result};
use feat_math::evolve::{
    crossover_canonical_pipelines, mutate_canonical_pipeline, random_canonical_pipeline,
};
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};

pub const LINEAGE_SCHEMA: &str = "parallel.gp_candidate_lineage.v1";
/// The only reference program a generic search starts from.
pub const FIXED_REFERENCE: &str = "(pipe X (ridge))";
const GENERATOR: &str = "feat_math.random_canonical_pipeline.depth6";
/// The only stream protocol whose variation events this verifier re-derives.
pub const RANDOM_PROTOCOL: &str = "parallel.gp_random_streams.v3";
/// Variation parents are drawn from at most this many best measured entries.
const PARENT_POOL: usize = 10;

/// Independent re-derivation of the searcher's keyed random words: keyed by
/// evaluation slot and the draw within that slot (W5-T02).
fn random_word(seed: u64, slot: u64, draw: u64, stream: &str) -> u64 {
    use sha2::{Digest, Sha256};
    let mut digest = Sha256::new();
    digest.update(RANDOM_PROTOCOL.as_bytes());
    digest.update(seed.to_le_bytes());
    digest.update(slot.to_le_bytes());
    digest.update(draw.to_le_bytes());
    digest.update(stream.as_bytes());
    u64::from_le_bytes(digest.finalize()[..8].try_into().unwrap())
}

/// Frozen inputs a lineage is verified against.
pub struct LineageContext<'a> {
    pub rng_seed: u64,
    pub vocabulary: Option<&'a crate::grammar::Vocab>,
    pub vocabulary_sha256: Value,
    pub seeds: Option<&'a SeedPopulation>,
    pub seeds_sha256: Value,
    /// `(artifact reference, genome, continuation)` of a native genome task.
    pub native: Option<(Value, String, bool)>,
    /// A continuation's measured direct parent `(objective in original target
    /// units, genome)` at position -1.
    pub inherited: Option<(f64, String)>,
}

/// One candidate's verified chain.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Chain {
    pub position: usize,
    pub source: String,
    pub original_sha256: String,
    pub adapted: bool,
    pub executable_sha256: String,
    /// Direct parents (ledger positions; -1 is an inherited incumbent).
    pub parents: Vec<i64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Lineage {
    pub chains: Vec<Chain>,
    pub variations: usize,
    pub adapted: usize,
}

fn program<'a>(envelope: &'a Value, part: &str) -> Result<&'a str> {
    let text = envelope[part]["program"]
        .as_str()
        .with_context(|| format!("lineage {part} program absent"))?;
    ensure!(
        hash(text.as_bytes()) == envelope[part]["program_sha256"],
        "lineage {part} identity differs from its program"
    );
    Ok(text)
}

/// Verify every row's envelope in ledger order.
pub fn verify_rows(context: &LineageContext, rows: &[Value]) -> Result<Lineage> {
    // The searcher's parent pool: an inherited incumbent (outside semantic
    // de-duplication) and the first measured row of each prediction signature.
    let mut pool: Vec<(f64, i64, String)> = Vec::new();
    if let Some((rmse, genome)) = &context.inherited {
        pool.push((*rmse, -1, genome.clone()));
    }
    let mut signatures = BTreeSet::new();
    let mut bootstrap_seeds = Vec::<(u64, u64)>::new();
    let mut chains = Vec::with_capacity(rows.len());
    // The search-wide proposal counter before each slot began.
    let mut counter = 0u64;
    for (position, row) in rows.iter().enumerate() {
        let chain = verify_row(context, position, row, counter, &pool, &mut bootstrap_seeds)
            .with_context(|| format!("candidate {position} lineage"))?;
        counter = row["proposal_attempts"]
            .as_u64()
            .with_context(|| format!("candidate {position} proposal counter absent"))?;
        if row["failure"].is_null() {
            let signature = row["metrics"]["semantic_sha256"]
                .as_str()
                .context("measured row lacks a prediction signature")?;
            if signatures.insert(signature.to_owned()) {
                let rmse = row["metrics"]["rmse"]
                    .as_f64()
                    .filter(|r| r.is_finite() && *r >= 0.0)
                    .context("measured row lacks a valid RMSE")?;
                let scale = row["target_scale"]
                    .as_f64()
                    .context("measured row lacks its target scale")?;
                pool.push((
                    rmse * scale,
                    position as i64,
                    row["genome"].as_str().unwrap_or_default().into(),
                ));
                pool.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            }
        }
        chains.push(chain);
    }
    Ok(Lineage {
        variations: chains.iter().filter(|c| c.source == "variation").count(),
        adapted: chains.iter().filter(|c| c.adapted).count(),
        chains,
    })
}

fn verify_row(
    context: &LineageContext,
    position: usize,
    row: &Value,
    counter_before: u64,
    pool: &[(f64, i64, String)],
    bootstrap_seeds: &mut Vec<(u64, u64)>,
) -> Result<Chain> {
    let envelope = &row["lineage"];
    ensure!(
        envelope["schema"] == LINEAGE_SCHEMA,
        "lineage envelope absent or unsupported"
    );
    let original = program(envelope, "original")?;
    let executable = program(envelope, "executable")?;
    ensure!(
        row["genome"] == executable && row["program_sha256"] == hash(executable.as_bytes()),
        "executable recipe differs from the evaluated program"
    );
    // Exactly one explicit adaptation of the original, or none.
    let adaptation = &envelope["adaptation"];
    match (context.vocabulary, adaptation["kind"].as_str()) {
        (Some(vocabulary), Some("vocabulary_projection")) => {
            ensure!(
                adaptation["applications"] == 1
                    && adaptation["vocabulary_sha256"] == context.vocabulary_sha256
                    && adaptation["input_sha256"] == hash(original.as_bytes())
                    && adaptation["output_sha256"] == hash(executable.as_bytes())
                    && adaptation["changed"] == (original != executable),
                "adaptation record is not one projection of the original"
            );
            ensure!(
                vocabulary.project_proposal(original)? == executable,
                "executable is not the single projection of the original proposal"
            );
        }
        (None, Some("none")) => ensure!(
            adaptation["applications"] == 0 && original == executable,
            "unadapted executable differs from its original"
        ),
        _ => bail!("adaptation differs from the frozen request"),
    }
    let adapted = original != executable;
    ensure!(
        row["proposal_before_projection"]
            == if adapted {
                json!(original)
            } else {
                Value::Null
            },
        "ledger original proposal differs from the envelope"
    );
    let source = &envelope["source"];
    let kind = source["kind"].as_str().context("lineage source omitted")?;
    let slot = row["initial_seed_slot"].as_u64();
    let mut parents = Vec::new();
    match kind {
        "seed_population_slot" => {
            let seeds = context
                .seeds
                .context("seed slot without a seed population")?;
            let slot = slot.context("seed slot on a non-initial row")?;
            let candidate = seeds
                .candidates
                .get(slot as usize)
                .context("seed slot outside the population")?;
            ensure!(
                source["slot"] == slot
                    && source["seeds_sha256"] == context.seeds_sha256
                    && candidate.genome == original
                    && source["origin"] == candidate.origin
                    && row["origin"] == candidate.origin
                    && source["rank_log_weight"] == json!(candidate.policy_logprob)
                    && source["score_subject"] == "original_proposal"
                    // W4-T07 join: exactly the proposal record the slot names (none for an
                    // older population, which a source may not invent).
                    && source["proposal_record_sha256"] == json!(candidate.proposal_record_sha256),
                "seed slot source differs from the frozen seed population"
            );
        }
        "fixed_reference" => ensure!(
            context.seeds.is_none()
                && context.native.is_none()
                && slot == Some(0)
                && source["slot"] == 0
                && original == FIXED_REFERENCE
                && row["origin"] == "generic_bootstrap",
            "fixed reference source differs"
        ),
        "generic_bootstrap" => {
            let slot = slot.context("bootstrap slot on a non-initial row")?;
            let generator_seed = source["generator_seed"]
                .as_u64()
                .context("bootstrap generator seed absent")?;
            ensure!(
                context.seeds.is_none()
                    && context.native.is_none()
                    && slot > 0
                    && source["slot"] == slot
                    && source["generator"] == GENERATOR
                    && generator_seed.wrapping_sub(context.rng_seed) < 100_000
                    && row["origin"] == "generic_bootstrap",
                "bootstrap source differs from the frozen request"
            );
            ensure!(
                random_canonical_pipeline(generator_seed, 6) == original,
                "original is not the recorded generator draw"
            );
            // Later slots come from strictly later generator draws.
            for &(other_slot, other_seed) in bootstrap_seeds.iter() {
                ensure!(
                    (other_slot < slot)
                        == (other_seed.wrapping_sub(context.rng_seed)
                            < generator_seed.wrapping_sub(context.rng_seed))
                        && other_slot != slot,
                    "bootstrap slots and generator draws are out of order"
                );
            }
            bootstrap_seeds.push((slot, generator_seed));
        }
        "native_candidate" => {
            let (reference, genome, continuation) = context
                .native
                .as_ref()
                .context("native source outside a native task")?;
            ensure!(
                !continuation
                    && slot == Some(0)
                    && source["candidate"] == *reference
                    && original == genome,
                "native candidate source differs from the task"
            );
        }
        "variation" => {
            ensure!(slot.is_none(), "variation recorded as an initial slot");
            let attempt = source["proposal_attempt"]
                .as_u64()
                .context("variation attempt absent")?;
            let draw = source["slot_draw"]
                .as_u64()
                .context("variation slot draw absent")?;
            // The event is this slot's, and its draw is the one the ledger's
            // search-wide counters say this slot accepted.
            ensure!(
                source["slot"] == position as u64
                    && row["proposal_attempts"] == attempt
                    && Some(attempt)
                        == draw
                            .checked_add(1)
                            .and_then(|draws| counter_before.checked_add(draws)),
                "variation slot or draw differs from the ledger counters"
            );
            let slot = position as u64;
            let seed = random_word(context.rng_seed, slot, draw, "variation");
            let word = random_word(context.rng_seed, slot, draw, "operator") % 16;
            ensure!(
                source["variation_seed"] == seed && source["operator_word"] == word,
                "variation random streams differ from the frozen seed"
            );
            let keep = pool.len().min(PARENT_POOL);
            let recorded: Vec<(i64, &Value)> = source["parents"]
                .as_array()
                .context("variation parents omitted")?
                .iter()
                .map(|p| {
                    (
                        p["position"].as_i64().unwrap_or(i64::MIN),
                        &p["program_sha256"],
                    )
                })
                .collect();
            // A recorded parent must be one of the top measured entries at the time.
            let parent = |index: usize| -> Result<String> {
                let (at, sha) = recorded.get(index).context("variation parent omitted")?;
                let entry = pool[..keep]
                    .iter()
                    .find(|entry| entry.1 == *at)
                    .context("variation parent was not an eligible measured entry")?;
                ensure!(
                    **sha == hash(entry.2.as_bytes()),
                    "variation parent identity differs"
                );
                Ok(entry.2.clone())
            };
            // Without critic weights the left parent is the uniform stream draw.
            let uniform = context
                .seeds
                .is_none_or(|s| s.candidates.iter().all(|c| c.policy_logprob.is_none()));
            let left_word = random_word(context.rng_seed, slot, draw, "left_parent");
            let operator = source["operator"]
                .as_str()
                .context("variation operator omitted")?;
            let continuation = context.native.as_ref().is_some_and(|n| n.2);
            let child = match operator {
                "continuation_mutate" => {
                    let (_, genome, _) = context.native.as_ref().context("continuation parent")?;
                    ensure!(
                        continuation
                            && keep == 0
                            && recorded.len() == 1
                            && source["parents"][0]["source"] == "native_candidate"
                            && *recorded[0].1 == hash(genome.as_bytes()),
                        "continuation mutation event differs"
                    );
                    parents.push(-1);
                    mutate_canonical_pipeline(genome, seed, 6).map_err(anyhow::Error::msg)?
                }
                "fresh_random" => {
                    ensure!(
                        recorded.is_empty()
                            && !(keep == 0 && context.native.is_some())
                            && (keep == 0 || word < 2),
                        "fresh random event contradicts the operator stream"
                    );
                    random_canonical_pipeline(seed, 6)
                }
                "mutate" => {
                    ensure!(
                        keep > 0 && (2..9).contains(&word) && recorded.len() == 1,
                        "mutation event contradicts the operator stream"
                    );
                    let left = parent(0)?;
                    if uniform {
                        ensure!(
                            recorded[0].0 == pool[(left_word % keep as u64) as usize].1,
                            "mutation parent is not the uniform stream draw"
                        );
                    }
                    parents.push(recorded[0].0);
                    mutate_canonical_pipeline(&left, seed, 6).map_err(anyhow::Error::msg)?
                }
                "crossover" => {
                    ensure!(
                        keep > 0 && word >= 9 && recorded.len() == 2,
                        "crossover event contradicts the operator stream"
                    );
                    let left = parent(0)?;
                    let right = parent(1)?;
                    let right_word = random_word(context.rng_seed, slot, draw, "right_parent");
                    ensure!(
                        recorded[1].0 == pool[(right_word % keep as u64) as usize].1,
                        "crossover right parent is not the stream draw"
                    );
                    if uniform {
                        ensure!(
                            recorded[0].0 == pool[(left_word % keep as u64) as usize].1,
                            "crossover left parent is not the uniform stream draw"
                        );
                    }
                    parents.extend([recorded[0].0, recorded[1].0]);
                    crossover_canonical_pipelines(&left, &right, seed, 6)
                        .map_err(anyhow::Error::msg)?
                }
                other => bail!("unknown variation operator {other}"),
            };
            ensure!(
                child == original,
                "original is not the recorded operator's output from its parents"
            );
        }
        other => bail!("unknown lineage source {other}"),
    }
    Ok(Chain {
        position,
        source: kind.into(),
        original_sha256: hash(original.as_bytes()),
        adapted,
        executable_sha256: hash(executable.as_bytes()),
        parents,
    })
}

mod final_refit;
mod selection;

pub use final_refit::{verify_final_refit, verify_final_refit_bytes, FinalRefit};
pub use selection::{verify_search, verify_search_recorded, Selected, SelectedLineage};

#[cfg(test)]
mod tests;
