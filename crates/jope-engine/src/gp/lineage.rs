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
                    && source["score_subject"] == "original_proposal",
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

/// The selected subject's authenticated chain within one completed search.
#[derive(Clone, Debug, Serialize)]
pub struct SelectedLineage {
    pub selection: selection_replay::SearchReplay,
    pub lineage: Lineage,
    /// `None` when no candidate was measured (a typed no-result, not a
    /// substitute) or when the inherited incumbent remains selected.
    pub selected: Option<Selected>,
    pub inherited_selected: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct Selected {
    pub chain: Chain,
    /// Every ancestor ledger position reachable through variation parents.
    pub ancestry: Vec<i64>,
    pub fold_fit_completion: Value,
    pub fold_model: Value,
    /// The selected fold fit's training partition (the cut's training rows).
    pub training_sha256: String,
}

/// Verify a completed search's selection and every candidate's lineage. The
/// seed population and vocabulary are the request's frozen inputs, supplied
/// explicitly and authenticated by their request digests.
pub fn verify_search(
    root: &Path,
    output: &Path,
    seeds: Option<&Path>,
    vocabulary: Option<&Path>,
) -> Result<SelectedLineage> {
    let selection = selection_replay::verify_search(root, output)?;
    let root = root.canonicalize()?;
    let output = output.canonicalize()?;
    let request: Value = read_json(&output.join("search-request.json"))?;
    ensure!(
        request["random_protocol"] == RANDOM_PROTOCOL,
        "lineage is verified under the slot-keyed random stream protocol only"
    );
    ensure!(
        request["seeds_sha256"] == json!(seeds.map(file_hash).transpose()?),
        "seed population differs from the frozen request"
    );
    ensure!(
        request["vocabulary_sha256"] == json!(vocabulary.map(file_hash).transpose()?),
        "vocabulary differs from the frozen request"
    );
    let seeds: Option<SeedPopulation> = seeds.map(read_json).transpose()?;
    let vocabulary: Option<crate::grammar::Vocab> = vocabulary.map(read_json).transpose()?;
    let native = match request.get("native_genome_stage").and_then(Value::as_str) {
        None => None,
        Some(stage) => {
            let reference: crate::native_graph::Artifact =
                serde_json::from_value(request["native_genome_task"].clone())?;
            let task: crate::native_genome::GenomeTask =
                crate::native_genome::load(&root, &reference)?;
            let candidate: crate::native_genome::Candidate =
                crate::native_genome::load(&root, &task.candidate)?;
            Some((
                serde_json::to_value(&task.candidate)?,
                candidate.genome,
                stage == "continuation",
            ))
        }
    };
    let rows: Vec<Value> = read_json(&output.join("observations.json"))?;
    // Each envelope is the one its pre-dispatch attempt recorded, and that
    // attempt is the one the original completion binds where one exists.
    for (position, row) in rows.iter().enumerate() {
        let path = output.join(format!("candidate-{position}.attempt.json"));
        let attempt: Value = read_json(&path)?;
        ensure!(
            attempt["lineage"] == row["lineage"] && attempt["genome"] == row["genome"],
            "candidate {position}: ledger lineage differs from its pre-dispatch attempt"
        );
        if !row["clean_completion"].is_null() {
            let completion = crate::label_admission::artifact(&root, &row["clean_completion"])?;
            ensure!(
                completion["attempt"]["sha256"] == file_hash(&path)?,
                "candidate {position}: attempt changed after its completion"
            );
        }
    }
    let inherited = selection
        .replay
        .ranking
        .iter()
        .find(|entry| entry.position == -1)
        .map(|entry| (entry.objective, entry.genome.clone()));
    let context = LineageContext {
        rng_seed: request["rng_seed"].as_u64().context("request seed")?,
        vocabulary: vocabulary.as_ref(),
        vocabulary_sha256: request["vocabulary_sha256"].clone(),
        seeds: seeds.as_ref(),
        seeds_sha256: request["seeds_sha256"].clone(),
        native,
        inherited,
    };
    let lineage = verify_rows(&context, &rows)?;
    let inherited_selected = selection
        .replay
        .winner
        .as_ref()
        .is_some_and(|winner| winner.position < 0);
    let selected = match &selection.replay.winner {
        // No measured candidate is a typed no-result; an inherited incumbent's
        // chain is authenticated by its own direct search, not this ledger.
        None => None,
        Some(winner) if winner.position < 0 => None,
        Some(winner) => {
            let position = winner.position as usize;
            let chain = lineage.chains[position].clone();
            let mut ancestry = BTreeSet::new();
            let mut pending = chain.parents.clone();
            while let Some(parent) = pending.pop() {
                if ancestry.insert(parent) && parent >= 0 {
                    pending.extend(lineage.chains[parent as usize].parents.iter().copied());
                }
            }
            let completion =
                crate::label_admission::artifact(&root, &rows[position]["clean_completion"])?;
            Some(Selected {
                chain,
                ancestry: ancestry.into_iter().collect(),
                fold_fit_completion: completion["fit_completion"].clone(),
                fold_model: completion["model"].clone(),
                training_sha256: rows[position]["training_sha256"]
                    .as_str()
                    .context("selected training identity")?
                    .into(),
            })
        }
    };
    Ok(SelectedLineage {
        selection,
        lineage,
        selected,
        inherited_selected,
    })
}

/// The explicit selected-to-final-refit transformation.
#[derive(Clone, Debug, Serialize)]
pub struct FinalRefit {
    pub transformation: &'static str,
    pub predictor: PathBuf,
    pub predictor_sha256: String,
    pub fit_key: String,
    pub model_sha256: String,
}

fn native_receipt(
    root: &Path,
    reference: &Value,
) -> Result<crate::eval_registry::CompletionReceipt> {
    let bytes = crate::label_admission::artifact_bytes(root, reference)?;
    crate::eval_registry::verify_completion_receipt(&bytes)
}

/// Link an exported predictor to the selected subject. The only final-fit scope
/// the fitted-view export implements is the selected cut's training rows, so a
/// valid final refit carries the selected fold fit's exact fitted identity.
/// Anything else is a different subject requiring its own qualification.
pub fn verify_final_refit(
    root: &Path,
    selected: &SelectedLineage,
    predictor: &Path,
) -> Result<FinalRefit> {
    verify_final_refit_bytes(root, selected, predictor, &std::fs::read(predictor)?)
}

/// [`verify_final_refit`] of the exact predictor bytes a caller already read
/// and bound (W1 route R8): the verified predictor is the recorded one.
pub fn verify_final_refit_bytes(
    root: &Path,
    selected: &SelectedLineage,
    predictor: &Path,
    predictor_bytes: &[u8],
) -> Result<FinalRefit> {
    let root = root.canonicalize()?;
    let chosen = selected
        .selected
        .as_ref()
        .context("no selected subject to refit")?;
    let winner = selected
        .selection
        .replay
        .winner
        .as_ref()
        .context("no selected subject to refit")?;
    let exported: Predictor = serde_json::from_slice(predictor_bytes)?;
    ensure!(
        exported.schema == "parallel.predictor.v1"
            && exported.task == "regression"
            && hash(exported.canonical.as_bytes()) == chosen.chain.executable_sha256
            && exported.canonical == feat_math::GenomeIr::parse(&winner.genome)?.canonical(),
        "final refit is not the selected executable recipe"
    );
    ensure!(
        exported.training_sha256 == chosen.training_sha256,
        "final refit declares another training scope"
    );
    let fold = native_receipt(&root, &chosen.fold_fit_completion)?;
    let reference = exported
        .fit_completion
        .as_ref()
        .context("final refit lacks its original fit completion")?;
    let path = Path::new(reference["path"].as_str().context("final fit path")?);
    // Route R8: the completion verified is the one whose digest the predictor records.
    let bytes =
        crate::manifest::read_bound(path, &reference["sha256"], "final refit completion changed")?;
    let refit = crate::eval_registry::verify_completion_receipt(&bytes)?;
    ensure!(
        refit.fit_key == fold.fit_key
            && refit.model_sha256 == fold.model_sha256
            && refit.program_sha256 == fold.program_sha256
            && refit.training_sha256 == fold.training_sha256,
        "final refit fitted state is not the selected fold fit"
    );
    Ok(FinalRefit {
        transformation: "selected_cut_training_refit_reuses_fold_fit",
        predictor: predictor.to_path_buf(),
        predictor_sha256: hash(predictor_bytes),
        fit_key: refit.fit_key,
        model_sha256: refit.model_sha256.context("final refit model absent")?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEED: u64 = 7;

    fn context<'a>() -> LineageContext<'a> {
        LineageContext {
            rng_seed: SEED,
            vocabulary: None,
            vocabulary_sha256: Value::Null,
            seeds: None,
            seeds_sha256: Value::Null,
            native: None,
            inherited: None,
        }
    }

    fn part(program: &str) -> Value {
        json!({"program":program,"program_sha256":hash(program.as_bytes())})
    }

    /// An unadapted (no vocabulary) ledger row carrying its envelope.
    fn row(
        original: &str,
        source: Value,
        slot: Option<u64>,
        attempts: u64,
        rmse: Option<f64>,
    ) -> Value {
        let origin = if slot.is_some() {
            "generic_bootstrap"
        } else {
            "gp_evolution"
        };
        json!({"genome":original,"program_sha256":hash(original.as_bytes()),
            "proposal_before_projection":null,"initial_seed_slot":slot,"origin":origin,
            "proposal_attempts":attempts,"target_scale":1.0,
            "failure":if rmse.is_some() {Value::Null} else {json!("timeout")},
            "metrics":rmse.map(|r| json!({"rmse":r,"semantic_sha256":hash(original.as_bytes())})),
            "lineage":{"schema":LINEAGE_SCHEMA,"original":part(original),"source":source,
                "adaptation":{"kind":"none","applications":0},"executable":part(original)}})
    }

    /// Slot `slot`'s variation accepted at `draw`, after a counter of `before`.
    fn variation(
        slot: u64,
        draw: u64,
        before: u64,
        operator: &str,
        parents: &[(i64, &str)],
    ) -> Value {
        json!({"kind":"variation","operator":operator,"slot":slot,"slot_draw":draw,
            "proposal_attempt":before + draw + 1,
            "variation_seed":random_word(SEED, slot, draw, "variation"),
            "operator_word":random_word(SEED, slot, draw, "operator") % 16,
            "parents":parents.iter().map(|(at, program)| json!({"position":at,
                "program_sha256":hash(program.as_bytes())})).collect::<Vec<_>>()})
    }

    /// First draw within `slot` whose operator word satisfies `accept` and
    /// whose operator output exists.
    fn draw_where(
        slot: u64,
        accept: impl Fn(u64) -> bool,
        child: impl Fn(u64, u64) -> Option<String>,
    ) -> (u64, String) {
        (0..10_000)
            .find_map(|d| {
                let word = random_word(SEED, slot, d, "operator") % 16;
                accept(word)
                    .then(|| child(d, random_word(SEED, slot, d, "variation")))
                    .flatten()
                    .map(|c| (d, c))
            })
            .expect("a matching operator draw")
    }

    /// Reference, one bootstrap draw, then a uniform mutation, a uniform
    /// crossover and a fresh draw, all re-derivable from the stream words.
    fn ledger() -> Vec<Value> {
        let reference = FIXED_REFERENCE.to_string();
        let bootstrap = random_canonical_pipeline(SEED, 6);
        let mut rows = vec![
            row(
                &reference,
                json!({"kind":"fixed_reference","slot":0}),
                Some(0),
                1,
                Some(0.5),
            ),
            row(
                &bootstrap,
                json!({"kind":"generic_bootstrap","slot":1,"generator":GENERATOR,
                    "generator_seed":SEED}),
                Some(1),
                2,
                Some(0.4),
            ),
        ];
        // Pool order by objective: bootstrap (0.4), reference (0.5).
        let pool2 = [(1i64, bootstrap.clone()), (0, reference.clone())];
        let (a, child) = draw_where(
            2,
            |w| (2..9).contains(&w),
            |d, seed| {
                let left = &pool2[(random_word(SEED, 2, d, "left_parent") % 2) as usize].1;
                mutate_canonical_pipeline(left, seed, 6).ok()
            },
        );
        let left = pool2[(random_word(SEED, 2, a, "left_parent") % 2) as usize].clone();
        let after_mutation = 2 + a + 1;
        rows.push(row(
            &child,
            variation(2, a, 2, "mutate", &[(left.0, left.1.as_str())]),
            None,
            after_mutation,
            Some(0.3),
        ));
        let pool3 = [
            (2i64, child.clone()),
            (1, bootstrap.clone()),
            (0, reference.clone()),
        ];
        let (b, cross) = draw_where(
            3,
            |w| w >= 9,
            |d, seed| {
                let l = &pool3[(random_word(SEED, 3, d, "left_parent") % 3) as usize].1;
                let r = &pool3[(random_word(SEED, 3, d, "right_parent") % 3) as usize].1;
                crossover_canonical_pipelines(l, r, seed, 6)
                    .ok()
                    .filter(|c| c != &child && c != &bootstrap && c != &reference)
            },
        );
        let l = pool3[(random_word(SEED, 3, b, "left_parent") % 3) as usize].clone();
        let r = pool3[(random_word(SEED, 3, b, "right_parent") % 3) as usize].clone();
        let after_crossover = after_mutation + b + 1;
        rows.push(row(
            &cross,
            variation(
                3,
                b,
                after_mutation,
                "crossover",
                &[(l.0, l.1.as_str()), (r.0, r.1.as_str())],
            ),
            None,
            after_crossover,
            None,
        ));
        let (c, fresh) = draw_where(
            4,
            |w| w < 2,
            |_, seed| Some(random_canonical_pipeline(seed, 6)),
        );
        rows.push(row(
            &fresh,
            variation(4, c, after_crossover, "fresh_random", &[]),
            None,
            after_crossover + c + 1,
            Some(0.9),
        ));
        rows
    }

    fn refused(rows: &[Value], expected: &str) {
        let error = format!("{:#}", verify_rows(&context(), rows).unwrap_err());
        assert!(error.contains(expected), "expected {expected:?}: {error}");
    }

    #[test]
    fn recorded_sources_operators_and_parents_re_derive_every_original() {
        let rows = ledger();
        let lineage = verify_rows(&context(), &rows).unwrap();
        let sources: Vec<&str> = lineage.chains.iter().map(|c| c.source.as_str()).collect();
        assert_eq!(
            sources,
            [
                "fixed_reference",
                "generic_bootstrap",
                "variation",
                "variation",
                "variation"
            ]
        );
        assert_eq!(lineage.variations, 3);
        assert_eq!(lineage.adapted, 0);
        assert_eq!(lineage.chains[2].parents.len(), 1);
        assert_eq!(lineage.chains[3].parents.len(), 2);
        assert!(lineage.chains[4].parents.is_empty());
    }

    #[test]
    fn parent_swaps_and_ineligible_parents_are_refused() {
        let rows = ledger();
        let parent = rows[2]["lineage"]["source"]["parents"][0]["position"]
            .as_i64()
            .unwrap();
        let (other, program) = if parent == 0 {
            (1, rows[1]["genome"].clone())
        } else {
            (0, rows[0]["genome"].clone())
        };
        let mut swapped = rows.clone();
        swapped[2]["lineage"]["source"]["parents"][0] = json!({"position":other,
            "program_sha256":hash(program.as_str().unwrap().as_bytes())});
        refused(&swapped, "uniform stream draw");
        // A parent whose identity does not match the ledger row it names.
        let mut relabelled = rows.clone();
        relabelled[2]["lineage"]["source"]["parents"][0]["program_sha256"] =
            json!(hash(b"another parent"));
        refused(&relabelled, "parent identity differs");
        // A failed row was never an eligible parent.
        let mut failed = rows.clone();
        let at = parent as usize;
        failed[at]["failure"] = json!("timeout");
        failed[at]["metrics"] = Value::Null;
        refused(&failed, "not an eligible measured entry");
        // A later row cannot parent an earlier one.
        let mut future = rows.clone();
        future[2]["lineage"]["source"]["parents"][0] = json!({"position":4,
            "program_sha256":rows[4]["program_sha256"]});
        refused(&future, "not an eligible measured entry");
    }

    #[test]
    fn omitted_or_contradictory_variation_events_are_refused() {
        let rows = ledger();
        let mut omitted = rows.clone();
        omitted[2]["lineage"]["source"] = Value::Null;
        refused(&omitted, "source omitted");
        let mut orphan = rows.clone();
        orphan[3]["lineage"]["source"]["parents"] = json!([]);
        refused(&orphan, "contradicts the operator stream");
        let mut relabelled = rows.clone();
        relabelled[2]["lineage"]["source"] = json!({"kind":"generic_bootstrap","slot":2,
            "generator":GENERATOR,"generator_seed":SEED + 1});
        refused(&relabelled, "non-initial row");
        let mut operator = rows.clone();
        operator[2]["lineage"]["source"]["operator"] = json!("crossover");
        refused(&operator, "contradicts the operator stream");
        let mut streams = rows.clone();
        streams[4]["lineage"]["source"]["variation_seed"] = json!(1);
        refused(&streams, "random streams");
        // Streams belong to one slot and one draw the ledger counters confirm:
        // an event cannot claim another slot, another draw, or reuse another
        // slot's words (W5-T02).
        let mut slot = rows.clone();
        slot[4]["lineage"]["source"]["slot"] = json!(3);
        refused(&slot, "slot or draw differs");
        let mut draw = rows.clone();
        let recorded = draw[4]["lineage"]["source"]["slot_draw"].as_u64().unwrap();
        draw[4]["lineage"]["source"]["slot_draw"] = json!(recorded + 1);
        refused(&draw, "slot or draw differs");
        draw[4]["lineage"]["source"]["slot_draw"] = json!(u64::MAX);
        refused(&draw, "slot or draw differs");
        let mut borrowed = rows.clone();
        let from_slot_three = rows[3]["lineage"]["source"].clone();
        for key in ["variation_seed", "operator_word"] {
            borrowed[4]["lineage"]["source"][key] = from_slot_three[key].clone();
        }
        refused(&borrowed, "random streams");
        let mut missing = rows.clone();
        missing[0]["lineage"] = Value::Null;
        refused(&missing, "envelope absent");
        // A bootstrap slot must be the recorded generator draw.
        let mut draw = rows.clone();
        draw[1]["lineage"]["source"]["generator_seed"] = json!(SEED + 1);
        refused(&draw, "generator draw");
    }

    #[test]
    fn recipe_substitution_and_double_adaptation_are_refused() {
        let rows = ledger();
        let substitute = random_canonical_pipeline(99, 6);
        let mut executable = rows.clone();
        executable[1]["lineage"]["executable"] = part(&substitute);
        refused(&executable, "executable recipe differs");
        let mut consistent = rows.clone();
        consistent[1]["genome"] = json!(substitute);
        consistent[1]["program_sha256"] = json!(hash(substitute.as_bytes()));
        consistent[1]["lineage"]["executable"] = part(&substitute);
        refused(&consistent, "unadapted executable differs");
        let mut declared = rows.clone();
        declared[1]["lineage"]["adaptation"] = json!({"kind":"vocabulary_projection",
            "applications":1});
        refused(&declared, "adaptation differs from the frozen request");

        // With the frozen vocabulary: one projection of the original only.
        let vocabulary: crate::grammar::Vocab =
            read_json(&Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/grammar/featmath.json"))
                .unwrap();
        let (seed, raw, adapted) = (SEED..SEED + 1000)
            .find_map(|seed| {
                let raw = random_canonical_pipeline(seed, 6);
                let adapted = vocabulary.project_proposal(&raw).ok()?;
                (adapted != raw).then_some((seed, raw, adapted))
            })
            .unwrap();
        let mut adapted_row = row(
            &raw,
            json!({"kind":"generic_bootstrap","slot":1,"generator":GENERATOR,
                "generator_seed":seed}),
            Some(1),
            1,
            Some(0.4),
        );
        adapted_row["genome"] = json!(adapted);
        adapted_row["program_sha256"] = json!(hash(adapted.as_bytes()));
        adapted_row["proposal_before_projection"] = json!(raw);
        adapted_row["lineage"]["executable"] = part(&adapted);
        adapted_row["lineage"]["adaptation"] = json!({"kind":"vocabulary_projection",
            "applications":1,"vocabulary_sha256":"v","input_sha256":hash(raw.as_bytes()),
            "output_sha256":hash(adapted.as_bytes()),"changed":true});
        let mut context = context();
        context.vocabulary = Some(&vocabulary);
        context.vocabulary_sha256 = json!("v");
        let check = |row: &Value| verify_rows(&context, std::slice::from_ref(row));
        check(&adapted_row).unwrap();
        // The adapted program posing as the original: a second adaptation.
        let mut twice = adapted_row.clone();
        twice["lineage"]["original"] = part(&adapted);
        twice["lineage"]["adaptation"]["input_sha256"] = json!(hash(adapted.as_bytes()));
        twice["lineage"]["adaptation"]["changed"] = json!(false);
        twice["proposal_before_projection"] = Value::Null;
        let error = format!("{:#}", check(&twice).unwrap_err());
        assert!(error.contains("generator draw"), "{error}");
        let mut repeated = adapted_row.clone();
        repeated["lineage"]["adaptation"]["applications"] = json!(2);
        assert!(check(&repeated).is_err());
        let mut unrecorded = adapted_row.clone();
        unrecorded["proposal_before_projection"] = Value::Null;
        let error = format!("{:#}", check(&unrecorded).unwrap_err());
        assert!(error.contains("ledger original proposal"), "{error}");
    }
}
