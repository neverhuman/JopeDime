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
                    ProposalFamily::ReplaceHead => Some(assemble(
                        parsed.stages,
                        heads[index(&mut rng, heads.len())].clone(),
                    )),
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
