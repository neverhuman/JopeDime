//! CPU proposer policy, following the RES-04 retrieval/edit/synthesis schedule.
//!
//! RES-04 measured evaluated/adaptively admitted proposals, not unconditional
//! generator weights. This module never fits a sampler to those marginals or
//! claims M3 from an unmatched generated-only histogram. Raw-bank replay and
//! discovery-loop registration have separate integration/qualification work.

use super::landmark_pools::select_landmark_preferred_pools;
use anyhow::{bail, ensure, Context, Result};
use feat_math::dsl::{DslExpr, ParamValue, Stage};
use feat_math::evolve::api::{
    capability::{self, CapabilityKind, Language, LanguageFilter, Registry},
    mutate_canonical_pipeline_with_arm, parse_canonical_pipeline, CanonicalMutationArm,
};
use feat_math::GenomeIr;
use rand_chacha::{
    rand_core::{RngCore, SeedableRng},
    ChaCha12Rng,
};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

const MAX_DSL_BYTES: usize = 16 * 1024;
const MAX_PARENTS: usize = 48;
const RAW_COORDINATES: &str = "raw_svd_lm48_first12";

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn hash_valid(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Source identity stays separate from the canonical memo identity. In
/// particular, retrieving a bank program does not rewrite its DSL/hash.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Program {
    pub dsl: String,
    pub program_sha256: String,
}

impl Program {
    pub fn from_dsl(dsl: impl Into<String>) -> Self {
        let dsl = dsl.into();
        Self {
            program_sha256: digest(dsl.as_bytes()),
            dsl,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BankRole {
    Proposer,
    ModelPrior,
}

#[derive(Clone, Debug)]
pub struct BankContract {
    pub role: BankRole,
    pub coordinate_space: String,
    /// Identity of the admitted roster AND promotion epoch, verified by the provider.
    pub snapshot_sha256: String,
}

#[derive(Clone, Debug)]
pub struct Retrieved {
    pub program: Program,
    pub family_id: String,
    pub dl_train: bool,
    pub distance_squared: f64,
}

/// Until RNK-09 admits a pinned raw snapshot, the provider must return `None`.
/// Never adapt the normalized model-prior bank to this interface.
pub trait ChampionBank {
    fn contract(&self) -> Option<BankContract>;

    /// Return at most `limit` TRAIN records from other families, ordered by
    /// (float64 squared distance, original program SHA), stably for exact ties.
    /// Distances use the raw first twelve SVD coordinates without normalization.
    fn nearest(
        &self,
        raw_lm12: &[f64; 12],
        excluded_family: &str,
        limit: usize,
    ) -> Result<Vec<Retrieved>>;
}

pub struct NoBank;

impl ChampionBank for NoBank {
    fn contract(&self) -> Option<BankContract> {
        None
    }

    fn nearest(&self, _: &[f64; 12], _: &str, _: usize) -> Result<Vec<Retrieved>> {
        bail!("no admitted raw proposer snapshot")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ProposalFamily {
    Retrieval,
    Parameter,
    ReplaceHead,
    Insert,
    Delete,
    Crossover,
    Synthesis,
}

impl ProposalFamily {
    pub fn name(self) -> &'static str {
        match self {
            Self::Retrieval => "champion_retrieval",
            Self::Parameter => "edit_parameter",
            Self::ReplaceHead => "edit_replace_head",
            Self::Insert => "edit_insert_stage",
            Self::Delete => "edit_delete_stage",
            Self::Crossover => "edit_crossover",
            Self::Synthesis => "landmark_synthesis",
        }
    }

    fn edit(step: usize) -> Self {
        [
            Self::Parameter,
            Self::ReplaceHead,
            Self::Insert,
            Self::Delete,
            Self::Crossover,
        ][step % 5]
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proposal {
    pub program: Program,
    pub memo_key: String,
    pub family: ProposalFamily,
    pub parents: Vec<String>,
}

pub struct Round<'a> {
    pub seed: u64,
    pub round: u64,
    pub exploring: bool,
    pub target_family: &'a str,
    /// Raw SVD lm48, before Stage1B standardization; never prior-bank summaries.
    pub raw_lm48: &'a [f64; 48],
    /// Already ordered by the caller's committed parent policy.
    pub parents: &'a [Program],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Batch {
    pub proposals: Vec<Proposal>,
    pub requested: usize,
    pub attempts: BTreeMap<ProposalFamily, usize>,
    /// Attempts with no eligible output, including inapplicable edit arms.
    pub invalid_or_filtered: usize,
    pub duplicates_or_memo: usize,
    pub synthesis_exhausted: bool,
    pub raw_bank_available: bool,
}

/// Explicit endian encoding, independent of worker count, globals and call order.
pub fn round_rng(seed: u64, round: u64) -> ChaCha12Rng {
    let mut hash = Sha256::new();
    hash.update(seed.to_le_bytes());
    hash.update(round.to_le_bytes());
    ChaCha12Rng::from_seed(hash.finalize().into())
}

fn index(rng: &mut ChaCha12Rng, length: usize) -> usize {
    debug_assert!(length > 0);
    let bound = length as u64;
    let threshold = bound.wrapping_neg() % bound;
    loop {
        let value = rng.next_u64();
        if value >= threshold {
            return (value % bound) as usize;
        }
    }
}

fn uniform(rng: &mut ChaCha12Rng, low: f64, high: f64) -> f64 {
    let unit = (rng.next_u64() >> 11) as f64 / ((1_u64 << 53) as f64);
    low + (high - low) * unit
}

fn mutate_parameters(stage: &mut Stage, rng: &mut ChaCha12Rng) {
    for (key, value) in &mut stage.params {
        if key == "seed" {
            continue;
        }
        let forced_float = matches!(key.as_str(), "q" | "lr" | "gamma" | "lambda_frac");
        match value.clone() {
            ParamValue::Float(v) if v.is_finite() => {
                *value = ParamValue::Float(jitter_float(key, v, rng));
            }
            ParamValue::Int(v) if forced_float => {
                *value = ParamValue::Float(jitter_float(key, v as f64, rng));
            }
            ParamValue::Int(v) => {
                let next = match key.as_str() {
                    "depth" | "degree" | "knots" => {
                        let upper = match key.as_str() {
                            "depth" => 6,
                            "degree" => 7,
                            _ => 8,
                        };
                        v.saturating_add(if index(rng, 2) == 0 { -1 } else { 1 })
                            .clamp(2, upper)
                    }
                    "rounds" => ((v as f64 * uniform(rng, 0.75, 1.30)).round_ties_even() as i64)
                        .clamp(40, 280),
                    "bins" | "topk" | "k" | "n" => {
                        ((v as f64 * uniform(rng, 0.75, 1.35)).round_ties_even() as i64)
                            .clamp(2, 128)
                    }
                    _ => v,
                };
                *value = ParamValue::Int(next);
            }
            _ => {}
        }
    }
}

fn jitter_float(key: &str, value: f64, rng: &mut ChaCha12Rng) -> f64 {
    let next = value * uniform(rng, -0.30, 0.30).exp();
    let bounded = match key {
        "q" => next.clamp(0.01, 0.25),
        "lr" => next.clamp(0.015, 0.30),
        "gamma" => next.clamp(0.05, 8.0),
        "lambda_frac" => next.clamp(0.0001, 1.0),
        _ => next,
    };
    (bounded * 100_000_000.0).round_ties_even() / 100_000_000.0
}

fn assemble(transforms: Vec<Stage>, head: Stage) -> String {
    let mut stages = transforms;
    stages.push(head);
    GenomeIr::Pipeline(DslExpr { stages }).canonical()
}

/// Canonical memo identity does not replace the model's exact source DSL hash.
pub fn memo_key(dsl: &str) -> Result<String> {
    ensure!(
        dsl.len() <= MAX_DSL_BYTES,
        "proposal DSL exceeds byte limit"
    );
    let pipeline = parse_canonical_pipeline(dsl).map_err(anyhow::Error::msg)?;
    Ok(digest(GenomeIr::Pipeline(pipeline).canonical().as_bytes()))
}

pub struct Proposer {
    filter: LanguageFilter,
    heads: BTreeSet<String>,
}

impl Proposer {
    pub fn new(languages: &[Language]) -> Result<Self> {
        let filter = capability::allowed_ops(languages)?;
        let heads = Registry::current()
            .entries
            .into_iter()
            .filter(|row| row.kind == CapabilityKind::Head)
            .map(|row| row.name.to_string())
            .collect();
        Ok(Self { filter, heads })
    }

    fn checked_pipeline(&self, program: &Program) -> Result<DslExpr> {
        ensure!(
            program.dsl.len() <= MAX_DSL_BYTES,
            "proposal DSL exceeds byte limit"
        );
        ensure!(
            program.program_sha256 == digest(program.dsl.as_bytes()),
            "program source hash mismatch"
        );
        let pipeline = parse_canonical_pipeline(&program.dsl).map_err(anyhow::Error::msg)?;
        ensure!(
            pipeline
                .stages
                .iter()
                .flat_map(|stage| stage.params.values())
                .all(|value| { !matches!(value, ParamValue::Float(v) if !v.is_finite()) }),
            "proposal parameters must be finite"
        );
        ensure!(
            pipeline
                .stages
                .last()
                .is_some_and(|stage| self.heads.contains(&stage.op)),
            "proposal requires a supported head"
        );
        Ok(pipeline)
    }

    fn pipeline(&self, program: &Program) -> Result<DslExpr> {
        let pipeline = self.checked_pipeline(program)?;
        ensure!(
            self.filter
                .accepts_seed(&GenomeIr::Pipeline(pipeline.clone())),
            "proposal is excluded by the language capability filter"
        );
        Ok(pipeline)
    }

    fn offer(
        &self,
        dsl: String,
        family: ProposalFamily,
        parents: Vec<String>,
        memo: &BTreeSet<String>,
        seen: &mut BTreeSet<String>,
        batch: &mut Batch,
    ) {
        let program = Program::from_dsl(dsl);
        let Ok(pipeline) = self.pipeline(&program) else {
            batch.invalid_or_filtered += 1;
            return;
        };
        let key = digest(GenomeIr::Pipeline(pipeline).canonical().as_bytes());
        if memo.contains(&key) || !seen.insert(key.clone()) {
            batch.duplicates_or_memo += 1;
            return;
        }
        batch.proposals.push(Proposal {
            program,
            memo_key: key,
            family,
            parents,
        });
    }

    /// Generate8 or16 proposals. Admission/evaluation of at most4 is a separate policy.
    pub fn propose(
        &self,
        round: &Round<'_>,
        bank: &dyn ChampionBank,
        memo: &BTreeSet<String>,
    ) -> Result<Batch> {
        ensure!(!round.target_family.is_empty(), "target family is required");
        ensure!(
            round.raw_lm48.iter().all(|v| v.is_finite()),
            "raw lm48 must be finite"
        );
        ensure!(round.parents.len() <= MAX_PARENTS, "parent pool exceeds48");
        let width = if round.exploring { 16 } else { 8 };
        let mut rng = round_rng(round.seed, round.round);
        let mut seen = BTreeSet::new();
        for parent in round.parents {
            self.pipeline(parent).context("invalid parent")?;
            seen.insert(memo_key(&parent.dsl)?);
        }
        let mut parents = round.parents.to_vec();
        let mut batch = Batch {
            proposals: Vec::new(),
            requested: width,
            attempts: BTreeMap::new(),
            invalid_or_filtered: 0,
            duplicates_or_memo: 0,
            synthesis_exhausted: false,
            raw_bank_available: false,
        };
        if let Some(contract) = bank.contract() {
            ensure!(
                contract.role == BankRole::Proposer
                    && contract.coordinate_space == RAW_COORDINATES
                    && hash_valid(&contract.snapshot_sha256),
                "admitted raw proposer snapshot required; model_prior is not retrieval input"
            );
            batch.raw_bank_available = true;
            let quota = 4.min(width / 3);
            let mut raw12 = [0.0; 12];
            raw12.copy_from_slice(&round.raw_lm48[..12]);
            let retrieved = bank.nearest(&raw12, round.target_family, quota)?;
            ensure!(retrieved.len() <= quota, "bank exceeded retrieval quota");
            let mut previous: Option<(f64, &str)> = None;
            for record in &retrieved {
                ensure!(
                    record.dl_train
                        && !record.family_id.is_empty()
                        && record.family_id != round.target_family,
                    "retrieval requires another DL-Train family"
                );
                ensure!(
                    record.distance_squared.is_finite() && record.distance_squared >= 0.0,
                    "invalid retrieval distance"
                );
                if let Some((distance, hash)) = previous {
                    ensure!(
                        record.distance_squared > distance
                            || (record.distance_squared == distance
                                && record.program.program_sha256.as_str() >= hash),
                        "bank nearest order differs from (distance, program hash)"
                    );
                }
                previous = Some((record.distance_squared, &record.program.program_sha256));
                *batch.attempts.entry(ProposalFamily::Retrieval).or_default() += 1;
                let checked = self
                    .checked_pipeline(&record.program)
                    .context("invalid retrieved program")?;
                let usable_parent = self.filter.accepts_seed(&GenomeIr::Pipeline(checked));
                self.offer(
                    record.program.dsl.clone(),
                    ProposalFamily::Retrieval,
                    vec![record.program.program_sha256.clone()],
                    memo,
                    &mut seen,
                    &mut batch,
                );
                if parents.is_empty() && usable_parent {
                    parents.push(record.program.clone());
                }
            }
        }
        let (stage_templates, head_templates) = select_landmark_preferred_pools(round.raw_lm48);
        let pool_stage = |template: &str| -> Result<Stage> {
            let text = format!("(pipe X {template} (ridge))");
            let mut parsed = parse_canonical_pipeline(&text).map_err(anyhow::Error::msg)?;
            ensure!(parsed.stages.len() == 2, "invalid preferred stage template");
            Ok(parsed.stages.remove(0))
        };
        let stages: Vec<Stage> = stage_templates
            .into_iter()
            .map(pool_stage)
            .collect::<Result<Vec<_>>>()?;
        let heads: Vec<Stage> = head_templates
            .into_iter()
            .map(|template| {
                let mut parsed = parse_canonical_pipeline(&format!("(pipe X {template})"))
                    .map_err(anyhow::Error::msg)?;
                ensure!(parsed.stages.len() == 1, "invalid preferred head template");
                Ok(parsed.stages.remove(0))
            })
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            !stages.is_empty() && !heads.is_empty(),
            "empty preferred pools"
        );
        if !parents.is_empty() {
            for step in 0..3 * width {
                if batch.proposals.len() == width {
                    break;
                }
                let family = ProposalFamily::edit(step);
                *batch.attempts.entry(family).or_default() += 1;
                let parent = &parents[step % parents.len()];
                let mut parsed = self.pipeline(parent)?;
                let mut head = parsed.stages.pop().expect("validated head");
                let transforms = &mut parsed.stages;
                let mut provenance = vec![parent.program_sha256.clone()];
                let generated = match family {
                    ProposalFamily::Parameter => {
                        for stage in transforms.iter_mut() {
                            mutate_parameters(stage, &mut rng);
                        }
                        mutate_parameters(&mut head, &mut rng);
                        Some(assemble(parsed.stages, head))
                    }
                    ProposalFamily::ReplaceHead => {
                        let first = index(&mut rng, heads.len());
                        let before = batch.proposals.len();
                        let first_dsl = assemble(parsed.stages.clone(), heads[first].clone());
                        let first_key = memo_key(&first_dsl)?;
                        let collision = batch.proposals.iter().any(|p| p.memo_key == first_key);
                        let limit = if collision { heads.len() } else { 1 };
                        // A repeated candidate should not consume this typed
                        // mutation slot while a legal, novel head is available.
                        // Keep the initial weighted draw and RNG position; the
                        // finite pool scan has no new randomness or quota.
                        for offset in 0..limit {
                            if offset > 0 {
                                *batch.attempts.entry(family).or_default() += 1;
                            }
                            self.offer(
                                assemble(
                                    parsed.stages.clone(),
                                    heads[(first + offset) % heads.len()].clone(),
                                ),
                                family,
                                provenance.clone(),
                                memo,
                                &mut seen,
                                &mut batch,
                            );
                            if batch.proposals.len() > before {
                                break;
                            }
                        }
                        continue;
                    }
                    ProposalFamily::Insert => {
                        let inserted = stages[index(&mut rng, stages.len())].clone();
                        let position = index(&mut rng, transforms.len() + 1);
                        transforms.insert(position, inserted);
                        (transforms.len() <= 6).then(|| assemble(parsed.stages, head))
                    }
                    ProposalFamily::Delete if transforms.len() >= 2 => Some(
                        mutate_canonical_pipeline_with_arm(
                            &parent.dsl,
                            rng.next_u64(),
                            6,
                            false,
                            CanonicalMutationArm::Delete,
                        )
                        .map_err(anyhow::Error::msg)?,
                    ),
                    ProposalFamily::Crossover if parents.len() >= 2 => {
                        let other = &parents[(step + 1) % parents.len()];
                        let mut other_parsed = self.pipeline(other)?;
                        let other_head = other_parsed.stages.pop().expect("validated head");
                        let left_cut = (transforms.len() / 2).max(1).min(transforms.len());
                        let mut cross = transforms[..left_cut].to_vec();
                        cross.extend_from_slice(
                            &other_parsed.stages[other_parsed.stages.len() / 2..],
                        );
                        cross.truncate(5);
                        provenance.push(other.program_sha256.clone());
                        Some(assemble(cross, other_head))
                    }
                    _ => None,
                };
                if let Some(dsl) = generated {
                    self.offer(dsl, family, provenance, memo, &mut seen, &mut batch);
                } else {
                    batch.invalid_or_filtered += 1;
                }
            }
        }
        // Filtering/memo can make the finite source intersection empty. Never
        // reproduce the historical infinite fallback loop in a service worker.
        for _ in 0..64 * width {
            if batch.proposals.len() == width {
                break;
            }
            *batch.attempts.entry(ProposalFamily::Synthesis).or_default() += 1;
            let count = index(&mut rng, 3) + 1;
            let chosen: Vec<Stage> = (0..count)
                .map(|_| {
                    let mut stage = stages[index(&mut rng, stages.len())].clone();
                    mutate_parameters(&mut stage, &mut rng);
                    stage
                })
                .collect();
            let mut head = heads[index(&mut rng, heads.len())].clone();
            mutate_parameters(&mut head, &mut rng);
            self.offer(
                assemble(chosen, head),
                ProposalFamily::Synthesis,
                vec![],
                memo,
                &mut seen,
                &mut batch,
            );
        }
        batch.synthesis_exhausted = batch.proposals.len() < width;
        Ok(batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    fn parents() -> Vec<Program> {
        vec![
            Program::from_dsl("(pipe X (standardize) (cheb degree=3) (ridge lambda_frac=0.1))"),
            Program::from_dsl(
                "(pipe X (robust_scale) (winsor_clip q=0.05) (histgbm depth=3 lr=0.08 rounds=160))",
            ),
        ]
    }

    fn request<'a>(parents: &'a [Program], lm: &'a [f64; 48], exploring: bool) -> Round<'a> {
        Round {
            seed: 42,
            round: 7,
            exploring,
            target_family: "target",
            raw_lm48: lm,
            parents,
        }
    }

    struct Bank {
        role: BankRole,
        records: Vec<Retrieved>,
        called: RefCell<Vec<([f64; 12], String, usize)>>,
    }

    impl Bank {
        fn new() -> Self {
            Self {
                role: BankRole::Proposer,
                records: vec![
                    Retrieved {
                        program: Program::from_dsl("(pipe X (quantile_bins bins=16) (ridge))"),
                        family_id: "other-a".into(),
                        dl_train: true,
                        distance_squared: 1.0,
                    },
                    Retrieved {
                        program: Program::from_dsl("(pipe X (winsor_clip q=0.1) (ridge))"),
                        family_id: "other-b".into(),
                        dl_train: true,
                        distance_squared: 2.0,
                    },
                ],
                called: RefCell::new(vec![]),
            }
        }
    }

    impl ChampionBank for Bank {
        fn contract(&self) -> Option<BankContract> {
            Some(BankContract {
                role: self.role,
                coordinate_space: RAW_COORDINATES.into(),
                snapshot_sha256: "1".repeat(64),
            })
        }
        fn nearest(&self, raw: &[f64; 12], family: &str, limit: usize) -> Result<Vec<Retrieved>> {
            self.called.borrow_mut().push((*raw, family.into(), limit));
            Ok(self.records.iter().take(limit).cloned().collect())
        }
    }

    #[test]
    fn repeated_head_draw_keeps_a_novel_head_slot_when_the_pool_has_an_alternative() {
        // Original TRAIN request130: round775/context2. This is a row-free
        // generator regression, not a model/fitness or release qualification.
        let proposer = Proposer::new(&[Language::Rust]).unwrap();
        let parents = [Program::from_dsl("(pipe X (poly_interact append=true) (cheb degree=4) (winsor_clip append=true q=0.15759416738534485) (sigmoid append=true) (ridge))")];
        let lm = [
            -7.822644233703613,
            -3.335017681121826,
            2.0916075706481934,
            0.17733953893184662,
            -0.23421242833137512,
            -1.8542579412460327,
            -0.39526352286338806,
            1.5499933958053589,
            0.3646467924118042,
            -0.23557567596435547,
            2.554443597793579,
            -0.7511533498764038,
            1.6888071298599243,
            0.965538501739502,
            -0.9864354729652405,
            -1.153594970703125,
            -1.8711278438568115,
            -0.2296914905309677,
            1.2335679531097412,
            -0.15032795071601868,
            1.465488314628601,
            -1.6561437845230103,
            -0.8125762939453125,
            0.8135456442832947,
            -0.35853514075279236,
            -0.9137781858444214,
            -0.4460272192955017,
            0.6699191927909851,
            0.8401845097541809,
            0.07990206778049469,
            -0.5336035490036011,
            0.4456247091293335,
            0.6210421323776245,
            -0.6575910449028015,
            -0.8605937361717224,
            0.04108719900250435,
            -0.318228542804718,
            -1.1542876958847046,
            0.020905081182718277,
            -0.4025436341762543,
            1.037981390953064,
            0.8037648797035217,
            0.1720881313085556,
            -0.14072567224502563,
            -0.21567422151565552,
            0.2446334958076477,
            -0.9094048738479614,
            0.4294348657131195,
        ];
        let bank = Bank {
            role: BankRole::Proposer,
            records: vec![
                    Retrieved {
                        program: Program::from_dsl("(pipe X (cheb append=true degree=4) (winsor_clip q=0.12691538581586245) (histgbm depth=6 lr=0.14350417339409188 rounds=210))"),
                        family_id: "other-0".into(),
                        dl_train: true,
                        distance_squared: 1.0,
                    },
                    Retrieved {
                        program: Program::from_dsl("(pipe X (blom append=true) (sigmoid append=true) (pair_prod topk=13) (ridge_gcv))"),
                        family_id: "other-1".into(),
                        dl_train: true,
                        distance_squared: 2.0,
                    },
            ],
            called: RefCell::new(vec![]),
        };
        let req = Round {
            seed: 20261031,
            round: 775,
            exploring: false,
            target_family: "original-target",
            raw_lm48: &lm,
            parents: &parents,
        };
        let batch = proposer.propose(&req, &bank, &BTreeSet::new()).unwrap();
        let heads: Vec<_> = batch
            .proposals
            .iter()
            .filter(|p| p.family == ProposalFamily::ReplaceHead)
            .collect();
        assert_eq!(batch.proposals.len(), 8);
        assert_eq!(
            heads.len(),
            2,
            "a repeated head draw consumed a scheduled head slot"
        );
        assert_ne!(heads[0].memo_key, heads[1].memo_key);
        assert_eq!(
            heads[0].program.program_sha256,
            "f5add21fc43bf261f379d1cda8de3500816025f6225458166871b03005baabdc"
        );
        assert!(batch.duplicates_or_memo > 0);
        assert_eq!(
            batch.attempts.values().sum::<usize>(),
            batch.proposals.len() + batch.invalid_or_filtered + batch.duplicates_or_memo
        );
        assert_eq!(
            batch,
            proposer.propose(&req, &bank, &BTreeSet::new()).unwrap()
        );

        // A collision does not authorize an existing memo entry or an unbounded
        // retry when every other preferred head has already been measured.
        let mut transforms = proposer.pipeline(&parents[0]).unwrap().stages;
        transforms.pop();
        let (_, templates) = select_landmark_preferred_pools(&lm);
        let mut memo = BTreeSet::new();
        for template in templates {
            let mut parsed = parse_canonical_pipeline(&format!("(pipe X {template})")).unwrap();
            let dsl = assemble(transforms.clone(), parsed.stages.pop().unwrap());
            let key = memo_key(&dsl).unwrap();
            if key != heads[0].memo_key {
                memo.insert(key);
            }
        }
        let exhausted = proposer.propose(&req, &bank, &memo).unwrap();
        assert_eq!(exhausted.proposals.len(), 8);
        assert_eq!(
            exhausted
                .proposals
                .iter()
                .filter(|p| p.family == ProposalFamily::ReplaceHead)
                .count(),
            1
        );
        assert!(exhausted
            .proposals
            .iter()
            .all(|p| !memo.contains(&p.memo_key)));
        assert_eq!(
            exhausted.attempts.values().sum::<usize>(),
            exhausted.proposals.len()
                + exhausted.invalid_or_filtered
                + exhausted.duplicates_or_memo
        );
    }

    #[test]
    fn deterministic_rounds_are_independent_and_exploration_only_changes_width() {
        let proposer = Proposer::new(&[Language::Rust]).unwrap();
        let parents = parents();
        let lm = [0.0; 48];
        let a = proposer
            .propose(&request(&parents, &lm, false), &NoBank, &BTreeSet::new())
            .unwrap();
        let b = proposer
            .propose(&request(&parents, &lm, false), &NoBank, &BTreeSet::new())
            .unwrap();
        assert_eq!(a, b);
        assert_eq!(a.proposals.len(), 8);
        let wide = proposer
            .propose(&request(&parents, &lm, true), &NoBank, &BTreeSet::new())
            .unwrap();
        assert_eq!(wide.proposals.len(), 16);
        let mut other = request(&parents, &lm, false);
        other.round += 1;
        assert_ne!(
            a.proposals,
            proposer
                .propose(&other, &NoBank, &BTreeSet::new())
                .unwrap()
                .proposals
        );
        assert!(!a.raw_bank_available);
    }

    #[test]
    fn raw_bank_contract_family_order_and_source_identity_are_enforced() {
        let proposer = Proposer::new(&[Language::Rust]).unwrap();
        let lm = std::array::from_fn(|i| i as f64 / 4.0);
        let bank = Bank::new();
        let result = proposer
            .propose(&request(&[], &lm, false), &bank, &BTreeSet::new())
            .unwrap();
        assert_eq!(result.proposals.len(), 8);
        assert_eq!(result.proposals[0].program, bank.records[0].program);
        assert_eq!(
            bank.called.borrow()[0],
            (
                <[f64; 12]>::try_from(&lm[..12]).unwrap(),
                "target".to_string(),
                2
            )
        );
        let mut bad = Bank::new();
        bad.role = BankRole::ModelPrior;
        assert!(proposer
            .propose(&request(&[], &lm, false), &bad, &BTreeSet::new())
            .is_err());
        assert!(bad.called.borrow().is_empty());
        for mutation in 0..4 {
            let mut bad = Bank::new();
            match mutation {
                0 => bad.records[0].family_id = "target".into(),
                1 => bad.records[0].dl_train = false,
                2 => bad.records[0].distance_squared = f64::NAN,
                _ => bad.records.reverse(),
            }
            assert!(proposer
                .propose(&request(&[], &lm, false), &bad, &BTreeSet::new())
                .is_err());
        }
    }

    #[test]
    fn memo_dedup_retains_parents_and_never_returns_known_identity() {
        let proposer = Proposer::new(&[Language::Rust]).unwrap();
        let parents = parents();
        let before = parents.clone();
        let lm = [0.0; 48];
        let first = proposer
            .propose(&request(&parents, &lm, true), &NoBank, &BTreeSet::new())
            .unwrap();
        let memo = first.proposals.iter().map(|p| p.memo_key.clone()).collect();
        let second = proposer
            .propose(&request(&parents, &lm, true), &NoBank, &memo)
            .unwrap();
        assert_eq!(parents, before);
        assert!(second.proposals.iter().all(|p| !memo.contains(&p.memo_key)));
        assert_eq!(
            second
                .proposals
                .iter()
                .map(|p| &p.memo_key)
                .collect::<BTreeSet<_>>()
                .len(),
            second.proposals.len()
        );
        assert!(second.duplicates_or_memo > 0);
    }

    #[test]
    fn actual_language_intersection_filters_every_generated_operation() {
        let languages = [Language::Rust, Language::Python, Language::Javascript];
        let proposer = Proposer::new(&languages).unwrap();
        let lm = [0.0; 48];
        let batch = proposer
            .propose(&request(&[], &lm, true), &NoBank, &BTreeSet::new())
            .unwrap();
        assert_eq!(batch.proposals.len(), 16);
        let filter = capability::allowed_ops(&languages).unwrap();
        for proposal in batch.proposals {
            let genome = GenomeIr::parse(&proposal.program.dsl).unwrap();
            assert!(filter.accepts_seed(&genome));
            let GenomeIr::Pipeline(pipeline) = genome else {
                panic!("pipeline required")
            };
            assert!(pipeline
                .stages
                .iter()
                .all(|s| !filter.excluded_ops().contains(&s.op.as_str())));
        }
        assert!(Proposer::new(&[Language::Typescript]).is_err());
    }

    #[test]
    fn an_unexportable_retrieval_is_filtered_without_aborting_the_round() {
        let proposer =
            Proposer::new(&[Language::Rust, Language::Python, Language::Javascript]).unwrap();
        let mut bank = Bank::new();
        bank.records[0].program = Program::from_dsl("(pipe X (gpr))");
        let lm = [0.0; 48];
        let batch = proposer
            .propose(&request(&[], &lm, false), &bank, &BTreeSet::new())
            .unwrap();
        assert_eq!(batch.proposals.len(), 8);
        assert!(batch.invalid_or_filtered > 0);
        assert!(batch
            .proposals
            .iter()
            .all(|p| proposer.pipeline(&p.program).is_ok()));
        assert!(batch
            .proposals
            .iter()
            .all(|p| p.program != bank.records[0].program));
    }

    #[test]
    fn round_key_matches_an_independent_little_endian_sha256_oracle() {
        // Python hashlib over struct.pack("<QQ", 123, 7), retained in the source receipt.
        let mut expected = ChaCha12Rng::from_seed([
            4, 152, 66, 165, 136, 157, 146, 169, 112, 244, 157, 201, 69, 32, 60, 213, 154, 218, 19,
            128, 81, 80, 181, 162, 170, 106, 147, 165, 229, 143, 224, 7,
        ]);
        let mut actual = round_rng(123, 7);
        for _ in 0..8 {
            assert_eq!(actual.next_u64(), expected.next_u64());
        }
    }

    #[test]
    fn typed_cycle_is_not_a_random_twenty_percent_sampler() {
        let counts = (0..24).fold(BTreeMap::<ProposalFamily, usize>::new(), |mut m, step| {
            *m.entry(ProposalFamily::edit(step)).or_default() += 1;
            m
        });
        assert_eq!(
            [
                counts[&ProposalFamily::Parameter],
                counts[&ProposalFamily::ReplaceHead],
                counts[&ProposalFamily::Insert],
                counts[&ProposalFamily::Delete],
                counts[&ProposalFamily::Crossover]
            ],
            [5, 5, 5, 5, 4]
        );
        let source: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../research/flywheel_selfplay/custody/proposer_frequencies.json"
        ))
        .unwrap();
        assert_eq!(
            source["population"]["kind"],
            "evaluated GP proposal attempts"
        );
        assert_eq!(source["population"]["records"], 1024);
        assert_eq!(
            source["population"]["not_individually_operator_labelled"],
            2688
        );
    }

    #[test]
    fn ten_thousand_proposals_have_real_filter_provenance_and_bounded_source_schedule() {
        let proposer = Proposer::new(&[Language::Rust]).unwrap();
        let parents = parents();
        let lm = [0.0; 48];
        let bank = Bank::new();
        let mut counts = BTreeMap::<ProposalFamily, usize>::new();
        for round in 0..1250 {
            let mut req = request(&parents, &lm, false);
            req.round = round;
            let batch = proposer.propose(&req, &bank, &BTreeSet::new()).unwrap();
            assert_eq!(batch.proposals.len(), 8);
            assert!(!batch.synthesis_exhausted);
            assert_eq!(
                batch.attempts.values().sum::<usize>(),
                batch.proposals.len() + batch.invalid_or_filtered + batch.duplicates_or_memo,
            );
            assert!(
                batch
                    .attempts
                    .iter()
                    .filter(|(f, _)| **f != ProposalFamily::Retrieval
                        && **f != ProposalFamily::Synthesis)
                    .map(|(_, n)| n)
                    .sum::<usize>()
                    <= 24
            );
            for proposal in batch.proposals {
                *counts.entry(proposal.family).or_default() += 1;
                assert_eq!(
                    proposal.program.program_sha256,
                    digest(proposal.program.dsl.as_bytes())
                );
                assert!(proposer.pipeline(&proposal.program).is_ok());
            }
        }
        assert_eq!(counts.values().sum::<usize>(), 10_000);
        assert_eq!(counts[&ProposalFamily::Retrieval], 2500);
        println!("generated-only CPU fixture histogram (not matched evaluated-population M3): {counts:?}");
    }

    #[test]
    fn ten_thousand_train_parent_outputs_report_m3_frequency_comparison() {
        const N: usize = 10_000;
        const MAX_ROUNDS: usize = 4_000;

        let source: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../research/flywheel_selfplay/custody/proposer_frequencies.json"
        ))
        .unwrap();
        assert_eq!(source["schema"], "jope.proposer-frequencies/v1");
        assert_eq!(
            source["population"]["kind"],
            "evaluated GP proposal attempts"
        );
        assert_eq!(source["population"]["records"], 1024);
        assert_eq!(source["population"]["generated_in_round_receipts"], 3712);
        assert_eq!(
            source["population"]["not_individually_operator_labelled"],
            2688
        );
        let target_sha256 = digest(
            include_str!(
                "../../../../research/flywheel_selfplay/custody/proposer_frequencies.json"
            )
            .as_bytes(),
        );
        let exploration_pattern: Vec<bool> = source["round_journals"]
            .as_array()
            .unwrap()
            .iter()
            .map(|round| round["exploration"].as_bool().unwrap())
            .collect();
        assert_eq!(exploration_pattern.len(), 16);

        // This is the committed row-free TRAIN parent index already used by the
        // real-parent determinism controls. No evaluation rows or labels are read.
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../research/flywheel_selfplay/custody/files")
            .join("home")
            .join("ubuntu")
            .join("jope-data/res-01/input/champions.json");
        let bytes = std::fs::read(path).unwrap();
        assert_eq!(
            digest(&bytes),
            "a444047ffb937dc8d69012cbdbf0891faf3e7d743b127e7cac3baed114cf141d"
        );
        let rows: Vec<serde_json::Value> = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(rows.len(), 1219);
        let parents: Vec<Program> = rows
            .iter()
            .map(|row| {
                assert_eq!(row["dl_split"], "DL-Train");
                assert!(hash_valid(row["context_sha256"].as_str().unwrap()));
                let parent = Program::from_dsl(row["dsl"].as_str().unwrap());
                assert_eq!(
                    parent.program_sha256,
                    row["program_sha256"].as_str().unwrap()
                );
                parent
            })
            .collect();

        // No raw proposer snapshot/evolving historical bank is supplied, so use
        // NoBank and report this as a generated-only diagnostic, never M3 parity.
        let proposer =
            Proposer::new(&[Language::Rust, Language::Python, Language::Javascript]).unwrap();
        let eligible_parent_indices: Vec<usize> = parents
            .iter()
            .enumerate()
            .filter_map(|(index, parent)| proposer.pipeline(parent).ok().map(|_| index))
            .collect();
        assert!(!eligible_parent_indices.is_empty());
        let lm = [0.0; 48];
        let mut counts = BTreeMap::<ProposalFamily, usize>::new();
        let mut generated = 0usize;
        let mut round_index = 0usize;
        while generated < N && round_index < MAX_ROUNDS {
            let parent_index = eligible_parent_indices[round_index % eligible_parent_indices.len()];
            let exploring = exploration_pattern[round_index % exploration_pattern.len()];
            let round = Round {
                seed: 0xd5c06 ^ round_index as u64,
                round: 767 + round_index as u64,
                exploring,
                target_family: rows[parent_index]["family_id"].as_str().unwrap(),
                raw_lm48: &lm,
                parents: std::slice::from_ref(&parents[parent_index]),
            };
            let batch = proposer.propose(&round, &NoBank, &BTreeSet::new()).unwrap();
            for proposal in batch.proposals {
                if generated == N {
                    break;
                }
                *counts.entry(proposal.family).or_default() += 1;
                generated += 1;
            }
            round_index += 1;
        }
        assert_eq!(
            generated, N,
            "bounded schedule did not emit 10,000 proposals"
        );

        let comparisons: Vec<serde_json::Value> = source["operators"]
            .as_array()
            .unwrap()
            .iter()
            .map(|operator| {
                let name = operator["operator"].as_str().unwrap();
                let family = [
                    ProposalFamily::Retrieval,
                    ProposalFamily::Parameter,
                    ProposalFamily::ReplaceHead,
                    ProposalFamily::Insert,
                    ProposalFamily::Delete,
                    ProposalFamily::Crossover,
                    ProposalFamily::Synthesis,
                ]
                .into_iter()
                .find(|family| family.name() == name)
                .unwrap_or_else(|| panic!("unknown reference family {name}"));
                let count = counts.get(&family).copied().unwrap_or_default();
                let observed = count as f64 / N as f64;
                let reference = operator["fraction"].as_f64().unwrap();
                serde_json::json!({
                    "operator": name,
                    "observed_count": count,
                    "observed_fraction": observed,
                    "reference_count": operator["count"],
                    "reference_fraction": reference,
                    "absolute_delta_percentage_points": (observed - reference).abs() * 100.0,
                    "within_two_percentage_points": (observed - reference).abs() <= 0.02,
                })
            })
            .collect();
        let within_tolerance = comparisons
            .iter()
            .all(|row| row["within_two_percentage_points"] == true);
        println!(
            "DSC06_M3_FREQUENCY_REPORT={}",
            serde_json::json!({
                "schema": "jopedime.dsc06-generated-frequency-diagnostic/v1",
                "reference_sha256": target_sha256,
                "reference_population": source["population"],
                "proposal_count": generated,
                "parent_source_sha256": digest(&bytes),
                "parent_count": parents.len(),
                "eligible_parent_count": eligible_parent_indices.len(),
                "parent_split": "DL-Train",
                "parent_rotation": "round_index modulo eligible committed TRAIN parent order",
                "bank": "NoBank; no exact evolving raw proposer snapshot supplied",
                "exploration_pattern_rounds": exploration_pattern.len(),
                "exploration_rounds_in_pattern": exploration_pattern.iter().filter(|&&x| x).count(),
                "generated_distribution_within_two_pp": within_tolerance,
                "population_comparable_to_reference": false,
                "m3_qualified": false,
                "comparisons": comparisons,
            })
        );
    }

    #[test]
    fn every_proposer_attempt_has_an_outcome() {
        let proposer = Proposer::new(&[Language::Rust]).unwrap();
        let lm = [0.0; 48];
        let cases = [
            "(pipe X (ridge lambda_frac=0.1))",
            "(pipe X (standardize) (ridge lambda_frac=0.1))",
            "(pipe X (standardize) (standardize) (standardize) (standardize) (standardize) (standardize) (ridge lambda_frac=0.1))",
        ];
        let mut unaccounted = vec![];
        let mut trace = Vec::new();
        for (case, dsl) in cases.into_iter().enumerate() {
            let parents = [Program::from_dsl(dsl)];
            for exploring in [false, true] {
                for round in 0..8 {
                    let mut req = request(&parents, &lm, exploring);
                    req.round = round;
                    let batch = proposer.propose(&req, &NoBank, &BTreeSet::new()).unwrap();
                    assert_eq!(batch.proposals.len(), if exploring { 16 } else { 8 });
                    assert!(batch.attempts[&ProposalFamily::Crossover] > 0);
                    let attempted: usize = batch.attempts.values().sum();
                    let accounted = batch.proposals.len()
                        + batch.invalid_or_filtered
                        + batch.duplicates_or_memo;
                    if attempted != accounted {
                        unaccounted.push((case, exploring, round, attempted, accounted));
                    }
                    // Deliberately exclude the repaired outcome counter so the
                    // before/after trace binds generation, order and provenance.
                    trace.push(serde_json::json!({
                        "case":case,"exploring":exploring,"round":round,
                        "attempts":batch.attempts.iter().map(|(f,n)| (f.name(),n)).collect::<Vec<_>>(),
                        "proposals":batch.proposals.iter().map(|p| serde_json::json!({
                            "dsl":p.program.dsl,"program_sha256":p.program.program_sha256,
                            "memo_key":p.memo_key,"family":p.family.name(),"parents":p.parents,
                        })).collect::<Vec<_>>(),
                    }));
                }
            }
        }
        println!(
            "DSC06_GENERATION_TRACE_SHA256={}",
            digest(&serde_json::to_vec(&trace).unwrap())
        );
        assert!(
            unaccounted.is_empty(),
            "missing attempt outcomes in {} rounds: {:?}",
            unaccounted.len(),
            &unaccounted[..unaccounted.len().min(3)]
        );
    }

    #[test]
    fn real_train_parents_preserve_round_determinism_and_language_exclusions() {
        // Real row-free parents with controlled NoBank/zero landmarks; this is
        // software acceptance, not matched evaluated-population M3 evidence.
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../research/flywheel_selfplay/custody/files")
            .join("home")
            .join("ubuntu")
            .join("jope-data/res-01/input/champions.json");
        let bytes = std::fs::read(path).unwrap();
        assert_eq!(
            digest(&bytes),
            "a444047ffb937dc8d69012cbdbf0891faf3e7d743b127e7cac3baed114cf141d"
        );
        let rows: Vec<serde_json::Value> = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(rows.len(), 1219);
        let parents: Vec<_> = rows
            .iter()
            .map(|row| {
                assert_eq!(row["dl_split"], "DL-Train");
                assert!(hash_valid(row["context_sha256"].as_str().unwrap()));
                assert!(!row["family_id"].as_str().unwrap().is_empty());
                let parent = Program::from_dsl(row["dsl"].as_str().unwrap());
                assert_eq!(
                    parent.program_sha256,
                    row["program_sha256"].as_str().unwrap()
                );
                parent
            })
            .collect();
        let lm = [0.0; 48];
        for languages in [
            &[Language::Rust][..],
            &[Language::Rust, Language::Python, Language::Javascript][..],
        ] {
            let original = Proposer::new(languages).unwrap();
            let replay = Proposer::new(languages).unwrap();
            let filter = capability::allowed_ops(languages).unwrap();
            let run = |proposer: &Proposer, index: usize, exploring| {
                proposer
                    .propose(
                        &Round {
                            seed: 0xd5c06 ^ index as u64,
                            round: index as u64 + 7,
                            exploring,
                            target_family: rows[index]["family_id"].as_str().unwrap(),
                            raw_lm48: &lm,
                            parents: std::slice::from_ref(&parents[index]),
                        },
                        &NoBank,
                        &BTreeSet::new(),
                    )
                    .map_err(|error| format!("{error:#}"))
            };
            let expected: Vec<_> = (0..rows.len())
                .flat_map(|index| [run(&original, index, false), run(&original, index, true)])
                .collect();
            let mut accepted = 0;
            let mut refused = 0;
            let mut generated = 0;
            for index in (0..rows.len()).rev() {
                for (mode, exploring) in [false, true].into_iter().enumerate() {
                    let actual = run(&replay, index, exploring);
                    assert_eq!(actual, expected[2 * index + mode], "record {index}");
                    match actual {
                        Ok(batch) => {
                            accepted += 1;
                            let width = if exploring { 16 } else { 8 };
                            assert_eq!(batch.requested, width);
                            assert!(batch.proposals.len() <= width);
                            assert_eq!(batch.synthesis_exhausted, batch.proposals.len() < width);
                            assert_eq!(
                                batch.attempts.values().sum::<usize>(),
                                batch.proposals.len()
                                    + batch.invalid_or_filtered
                                    + batch.duplicates_or_memo
                            );
                            let mut unique = BTreeSet::new();
                            for proposal in batch.proposals {
                                assert_eq!(
                                    proposal.program.program_sha256,
                                    digest(proposal.program.dsl.as_bytes())
                                );
                                assert!(unique.insert(proposal.memo_key.clone()));
                                let GenomeIr::Pipeline(parsed) =
                                    GenomeIr::parse(&proposal.program.dsl).unwrap()
                                else {
                                    panic!("proposer must return a pipeline")
                                };
                                for stage in parsed.stages {
                                    assert!(filter.allowed_ops().contains(&stage.op.as_str()));
                                    assert!(!filter.excluded_ops().contains(&stage.op.as_str()));
                                }
                                generated += 1;
                            }
                        }
                        Err(reason) => {
                            assert!(reason.starts_with("invalid parent:"), "{reason}");
                            refused += 1;
                        }
                    }
                }
            }
            assert_eq!(accepted + refused, 2 * rows.len());
            assert!(
                accepted > 0 && generated > 0,
                "no eligible real parents exercised"
            );
            println!("DSC06 real TRAIN parent software controls: languages={languages:?}, records={}, replayed_requests={}, accepted={accepted}, refused={refused}, generated={generated}; reverse-order fresh-instance equality; M3_qualified=false", rows.len(), accepted + refused);
        }
    }

    #[test]
    fn numeric_mutation_preserves_seed_and_source_clamps() {
        let mut parsed = parse_canonical_pipeline(
            "(pipe X (winsor_clip q=0.5 seed=701) (histgbm depth=9 lr=0.9 rounds=999))",
        )
        .unwrap();
        let mut rng = round_rng(11, 9);
        for stage in &mut parsed.stages {
            mutate_parameters(stage, &mut rng);
        }
        assert_eq!(parsed.stages[0].get_u64("seed", 0), 701);
        assert!((0.01..=0.25).contains(&parsed.stages[0].get_f64("q", 0.0)));
        assert!((2..=6).contains(&parsed.stages[1].get_u64("depth", 0)));
        assert!((0.015..=0.30).contains(&parsed.stages[1].get_f64("lr", 0.0)));
        assert!((40..=280).contains(&parsed.stages[1].get_u64("rounds", 0)));
    }

    #[test]
    fn odd_length_crossover_uses_the_source_floor_half_and_other_head() {
        let proposer = Proposer::new(&[Language::Rust]).unwrap();
        let parents = vec![
            Program::from_dsl("(pipe X (standardize) (cheb degree=3) (winsor_clip q=0.05) (ridge))"),
            Program::from_dsl("(pipe X (robust_scale) (cheb degree=4) (quantile_bins bins=16) (row_stats) (histgbm))"),
        ];
        let lm = [0.0; 48];
        let batch = proposer
            .propose(&request(&parents, &lm, true), &NoBank, &BTreeSet::new())
            .unwrap();
        let cross = batch
            .proposals
            .iter()
            .find(|p| p.family == ProposalFamily::Crossover)
            .unwrap();
        let parsed = parse_canonical_pipeline(&cross.program.dsl).unwrap();
        assert_eq!(
            parsed
                .stages
                .iter()
                .map(|s| s.op.as_str())
                .collect::<Vec<_>>(),
            ["standardize", "quantile_bins", "row_stats", "histgbm"]
        );
        assert_eq!(
            cross.parents,
            parents
                .iter()
                .map(|p| p.program_sha256.clone())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn empty_capability_intersection_cannot_spin_forever_in_synthesis() {
        let mut proposer = Proposer::new(&[Language::Rust]).unwrap();
        let mut registry = Registry::current();
        for entry in &mut registry.entries {
            for supported in entry.supported.values_mut() {
                *supported = entry.kind == CapabilityKind::Head && entry.name == "ridge_gcv";
            }
        }
        proposer.filter = registry.allowed_ops(&[Language::Rust]).unwrap();
        let lm = [0.0; 48];
        let batch = proposer
            .propose(&request(&[], &lm, false), &NoBank, &BTreeSet::new())
            .unwrap();
        assert!(batch.synthesis_exhausted);
        assert!(batch.proposals.is_empty());
        assert_eq!(batch.attempts[&ProposalFamily::Synthesis], 64 * 8);
        assert_eq!(batch.invalid_or_filtered, 64 * 8);
    }

    #[test]
    fn invalid_parent_hash_or_landmarks_fail_before_bank_reads() {
        let proposer = Proposer::new(&[Language::Rust]).unwrap();
        let bank = Bank::new();
        let mut parents = parents();
        let lm = [0.0; 48];
        parents[0].program_sha256 = "0".repeat(64);
        assert!(proposer
            .propose(&request(&parents, &lm, false), &bank, &BTreeSet::new())
            .is_err());
        let mut lm = lm;
        lm[47] = f64::INFINITY;
        assert!(proposer
            .propose(&request(&[], &lm, false), &bank, &BTreeSet::new())
            .is_err());
        assert!(bank.called.borrow().is_empty());
    }
}
