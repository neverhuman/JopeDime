//! GP novelty, collapse and evolution summaries: TELEMETRY_SPEC rows G1–G12, plus the funnel
//! counts of E2.
//!
//! Everything here runs once per generation, off the training hot path, on the GP loop's own
//! thread. Inputs are program hashes, interned clause tokens and numbers. The DSL text is parsed
//! only to intern clauses, and it never enters a record.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};

pub const GEN_SCHEMA: &str = "jope.telemetry.gp-generation/v1";
/// Row G4b's null reason: the Rust evaluator returns scalar fitness, never predictions.
pub const NO_PREDICTION_SIGNATURES: &str = "no_prediction_signatures";

/// Split `(pipe X (stage …) … (head …))` into `(stages, head)`, exactly as
/// `research/flywheel_selfplay/rust_evaluator.py::parse_pipe_dsl` does, including its fallback
/// head `(ridge_gcv)` for anything that is not a pipe or has no clauses.
pub fn parse_pipe_dsl(dsl: &str) -> (Vec<&str>, &str) {
    const FALLBACK: &str = "(ridge_gcv)";
    const PREFIX: &str = "(pipe X";
    let s = dsl.trim();
    if !s.starts_with(PREFIX) {
        return (Vec::new(), FALLBACK);
    }
    // Python: s[len("(pipe X"):-1].strip(). The slice drops the last character, whatever it is.
    let end = s.char_indices().last().map_or(0, |(i, _)| i);
    let inner = if end > PREFIX.len() {
        s[PREFIX.len()..end].trim()
    } else {
        ""
    };
    let mut clauses = Vec::new();
    let mut depth: i64 = 0;
    let mut start: Option<usize> = None;
    for (idx, ch) in inner.char_indices() {
        if ch == '(' {
            if depth == 0 {
                start = Some(idx);
            }
            depth += 1;
        } else if ch == ')' {
            depth -= 1;
            if depth == 0 {
                if let Some(st) = start.take() {
                    clauses.push(&inner[st..=idx]);
                }
            }
        }
    }
    match clauses.pop() {
        Some(head) => (clauses, head),
        None => (Vec::new(), FALLBACK),
    }
}

/// The operator of one clause: its first token, e.g. `winsor_clip` for `(winsor_clip q=0.07)`.
pub fn clause_op(clause: &str) -> &str {
    let body = clause.trim_start_matches('(');
    let end = body
        .find(|c: char| c.is_whitespace() || c == '(' || c == ')')
        .unwrap_or(body.len());
    &body[..end]
}

/// `sha256(dsl)`, the flywheel's `program_sha256`.
pub fn program_sha256(dsl: &str) -> [u8; 32] {
    Sha256::digest(dsl.as_bytes()).into()
}

/// Run-scoped, bounded string interner for clause and op tokens.
///
/// The first `max_entries` distinct strings (and at most `max_bytes` of text) get dense ids,
/// and `dictionary()` is what the manifest records. Once the bound is reached, a new string gets
/// a hashed id with the top bit set and is not stored. Every summary here only compares ids for
/// equality, so hashed ids keep edit distances and op counts correct (barring a 31-bit hash
/// collision) while memory stays fixed. `overflowed()` counts how often that happened.
pub struct Interner {
    /// Each string is stored once, as the map key; `dictionary()` orders them by id on demand.
    ids: HashMap<Box<str>, u32>,
    bytes: usize,
    max_entries: usize,
    max_bytes: usize,
    overflowed: u64,
}

/// Default bounds: 16,384 strings or 1 MiB of text, whichever comes first.
pub const INTERNER_MAX_ENTRIES: usize = 16_384;
pub const INTERNER_MAX_BYTES: usize = 1 << 20;
const HASHED_ID_BIT: u32 = 1 << 31;

impl Default for Interner {
    fn default() -> Self {
        Self::with_bounds(INTERNER_MAX_ENTRIES, INTERNER_MAX_BYTES)
    }
}

impl Interner {
    pub fn with_bounds(max_entries: usize, max_bytes: usize) -> Self {
        Self {
            ids: HashMap::new(),
            bytes: 0,
            max_entries: max_entries.min(HASHED_ID_BIT as usize),
            max_bytes,
            overflowed: 0,
        }
    }

    pub fn intern(&mut self, s: &str) -> u32 {
        if let Some(&id) = self.ids.get(s) {
            return id;
        }
        if self.ids.len() >= self.max_entries || self.bytes + s.len() > self.max_bytes {
            self.overflowed += 1;
            let h = Sha256::digest(s.as_bytes());
            return HASHED_ID_BIT | (u32::from_le_bytes([h[0], h[1], h[2], h[3]]) & !HASHED_ID_BIT);
        }
        let id = self.ids.len() as u32;
        self.bytes += s.len();
        self.ids.insert(s.into(), id);
        id
    }
    pub fn len(&self) -> usize {
        self.ids.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
    /// Stored strings in id order (dense ids only; hashed overflow ids aren't stored).
    pub fn dictionary(&self) -> Vec<&str> {
        let mut v: Vec<(&str, u32)> = self.ids.iter().map(|(k, &id)| (&**k, id)).collect();
        v.sort_unstable_by_key(|x| x.1);
        v.into_iter().map(|x| x.0).collect()
    }
    pub fn overflowed(&self) -> u64 {
        self.overflowed
    }
}

/// Longest genome kept: programs with more clauses keep their first `MAX_CLAUSES - 1` stages
/// and their head, and are counted as truncated. Real flywheel programs have under 10.
pub const MAX_CLAUSES: usize = 64;

/// Clause and op ids of one program, stages followed by the head.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Genome {
    pub clauses: Vec<u32>,
    pub ops: Vec<u32>,
    pub truncated: bool,
}

impl Genome {
    pub fn from_dsl(dsl: &str, clauses: &mut Interner, ops: &mut Interner) -> Self {
        let (mut stages, head) = parse_pipe_dsl(dsl);
        let mut g = Genome::default();
        if stages.len() >= MAX_CLAUSES {
            stages.truncate(MAX_CLAUSES - 1);
            g.truncated = true;
        }
        for c in stages.into_iter().chain(std::iter::once(head)) {
            g.clauses.push(clauses.intern(c));
            g.ops.push(ops.intern(clause_op(c)));
        }
        g
    }
}

/// Labels written to each result's `operator` field by `propose_flywheel_candidates`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EditOp {
    EditParameter,
    EditReplaceHead,
    EditInsertStage,
    EditDeleteStage,
    EditCrossover,
    ChampionRetrieval,
    LandmarkSynthesis,
    IncumbentParent,
    G0Seed,
    Other,
}

impl EditOp {
    pub fn from_label(s: &str) -> Self {
        match s {
            "edit_parameter" => Self::EditParameter,
            "edit_replace_head" => Self::EditReplaceHead,
            "edit_insert_stage" => Self::EditInsertStage,
            "edit_delete_stage" => Self::EditDeleteStage,
            "edit_crossover" => Self::EditCrossover,
            "champion_retrieval" => Self::ChampionRetrieval,
            "landmark_synthesis" => Self::LandmarkSynthesis,
            "incumbent_parent" => Self::IncumbentParent,
            "g0_seed" => Self::G0Seed,
            _ => Self::Other,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvalStatus {
    Ok,
    /// `rc = -1` from the evaluator's subprocess timeout.
    Timeout,
    NonzeroRc,
    ParseFail,
    /// `status == "failed"` with no cause: the evaluator's result dict folds timeouts, nonzero
    /// exits and parse failures into one status (fitness 100.0). The cause split needs the
    /// worker-level return code (rows E5/E7).
    Failed,
}

/// Where a candidate's program came from, judged against the bank and corpus at round start.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Provenance {
    Novel,
    Replayed,
    Bank,
}

/// One evaluated candidate of a generation.
#[derive(Clone, Debug)]
pub struct Candidate {
    pub program: [u8; 32],
    pub genome: Genome,
    pub parent: Option<Genome>,
    pub edit_op: EditOp,
    pub status: EvalStatus,
    pub gate_passed: bool,
    /// Lower is better. Failures and gate failures carry the evaluator's sentinels (>= 100),
    /// which is why fitness quantiles only ever use gate-passers.
    pub fitness: f64,
    /// Whether this child beat its parent. `None` when the source doesn't say (imports of
    /// retained scores), so success rates are never computed from a guess.
    pub improved: Option<bool>,
    pub hwm_winner: bool,
    pub complexity: Option<u32>,
    pub proj16: Option<u16>,
    /// Per-probe-context log ratios over a fixed panel, NaN where a context is missing. The
    /// evaluator returns no prediction vectors, so this is only the G4 *proxy* descriptor.
    pub behavior: Vec<f64>,
    pub provenance: Provenance,
}

/// Generation-level counters the GP loop keeps itself (proposal-time dedup, row G2).
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct ProposalCounters {
    pub proposals: u64,
    pub seen_rejects: u64,
}

/// The G4 proxy's descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BehaviorKind {
    /// One candidate's error profile on its own context, e.g. `[cv_mean_log_ratio,
    /// cv_tail_log_ratio, cv_log_ratio_sd, ln test_rmse_ratio]`. The components have different
    /// scales and meanings, so pairs are compared by RMS distance; rank correlation would mostly
    /// measure the components' scale order. This is all the current flywheel provides: each
    /// candidate is evaluated on its own context only.
    ErrorProfile,
    /// Log ratios of one candidate over a fixed panel of probe contexts. Pairs are compared by
    /// `1 − Spearman ρ`. Needs evaluations the flywheel doesn't run today.
    ProbePanel,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub seed: u64,
    pub max_pairs: usize,
    pub behavior_quantum: f64,
    pub min_common_positions: usize,
    /// What the behavior vectors are, which decides the pairwise distance.
    pub behavior_kind: BehaviorKind,
    /// Behavior archive bound. With `max_behavior_len`, this caps a tracker's archive at
    /// `archive_capacity × max_behavior_len × 8` bytes (64 KiB at the defaults).
    pub archive_capacity: usize,
    /// Behavior vectors longer than this are truncated to their first `max_behavior_len` components.
    pub max_behavior_len: usize,
    /// Candidates summarized per generation; the rest are counted in `candidates_truncated`.
    pub max_candidates: usize,
    pub knn_k: usize,
    pub knn_max_queries: usize,
    /// `|STAGE_POOL_BY_REGIME ∪ HEAD_POOL_BY_REGIME|` op vocabulary size, when known.
    pub op_vocab_size: Option<usize>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            seed: 0x6a6f_7065_7465_6c00,
            max_pairs: 1024,
            behavior_quantum: 1e-3,
            behavior_kind: BehaviorKind::ErrorProfile,
            min_common_positions: 3,
            archive_capacity: 256,
            max_behavior_len: 32,
            max_candidates: 1024,
            knn_k: 5,
            knn_max_queries: 256,
            op_vocab_size: None,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Quantiles {
    pub n: u32,
    pub min: Option<f64>,
    pub p10: Option<f64>,
    pub p50: Option<f64>,
    pub p90: Option<f64>,
    pub max: Option<f64>,
    pub iqr: Option<f64>,
}

impl Quantiles {
    pub fn of(values: &mut [f64]) -> Self {
        values.sort_by(f64::total_cmp);
        let q = |p| quantile_sorted(values, p);
        Self {
            n: values.len() as u32,
            min: values.first().copied(),
            p10: q(0.10),
            p50: q(0.50),
            p90: q(0.90),
            max: values.last().copied(),
            iqr: q(0.75).zip(q(0.25)).map(|(a, b)| a - b),
        }
    }
}

/// numpy's default (`linear`) quantile on an ascending slice.
pub fn quantile_sorted(sorted: &[f64], p: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let h = (sorted.len() - 1) as f64 * p;
    let lo = h.floor() as usize;
    let hi = h.ceil() as usize;
    Some(sorted[lo] + (h - lo as f64) * (sorted[hi] - sorted[lo]))
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct StatusCounts {
    pub ok: u32,
    pub timeout: u32,
    pub nonzero_rc: u32,
    pub parse_fail: u32,
    pub failed_unknown_cause: u32,
    pub gate_failed: u32,
    pub gate_passed: u32,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct EditOpStats {
    pub n: u32,
    pub share: f64,
    /// Children with a known improvement outcome, and how many of those improved.
    pub known: u32,
    pub improved: u32,
    pub success_rate: Option<f64>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ProvenanceFracs {
    pub n: u32,
    pub novel: Option<f64>,
    pub replayed: Option<f64>,
    pub bank: Option<f64>,
}

impl ProvenanceFracs {
    fn of<'a>(it: impl Iterator<Item = &'a Candidate>) -> Self {
        let (mut n, mut nv, mut rp, mut bk) = (0u32, 0u32, 0u32, 0u32);
        for c in it {
            n += 1;
            match c.provenance {
                Provenance::Novel => nv += 1,
                Provenance::Replayed => rp += 1,
                Provenance::Bank => bk += 1,
            }
        }
        let f = |k: u32| (n > 0).then(|| k as f64 / n as f64);
        Self {
            n,
            novel: f(nv),
            replayed: f(rp),
            bank: f(bk),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BehaviorStats {
    pub kind: Option<BehaviorKind>,
    pub n_with_behavior: u32,
    pub pairs: u32,
    /// G4 proxy: mean pairwise distance over sampled pairs, RMS or `1 − Spearman ρ` by `kind`.
    pub fitness_profile_diversity: Option<f64>,
    pub fitness_profile_unique_frac: Option<f64>,
    /// G4b: prediction-vector diversity. Always null until the evaluator emits prediction
    /// signatures; see `prediction_availability`.
    pub prediction_diversity: Option<f64>,
    pub prediction_availability: Option<String>,
    /// Mean RMS distance to the k nearest archive members (row G12).
    pub archive_knn_mean: Option<f64>,
    pub archive_size: u32,
    pub availability: Option<String>,
}

/// One generation's summary record (rows G1–G12 and the E2 status split).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenSummary {
    pub schema: String,
    pub context: String,
    pub round: u32,
    pub generation: u32,
    pub pop_size: u32,
    /// Candidates beyond `Config::max_candidates`, left out of every statistic below.
    pub candidates_truncated: u32,
    /// Genomes cut to `MAX_CLAUSES`.
    pub genomes_truncated: u32,
    pub unique_genotype_frac: Option<f64>,
    pub dedup: ProposalCounters,
    pub dedup_reject_frac: Option<f64>,
    pub edit_distance_pairs: u32,
    pub edit_distance_mean: Option<f64>,
    pub edit_distance_p10: Option<f64>,
    /// Parent→child clause edit distance: buckets 0..=6, then 7 or more.
    pub parent_child_hist: [u32; 8],
    pub behavior: BehaviorStats,
    pub proj16_unique_frac: Option<f64>,
    pub proj16_mean_hamming: Option<f64>,
    pub op_counts: BTreeMap<u32, u64>,
    pub op_entropy_nats: Option<f64>,
    pub op_entropy_norm: Option<f64>,
    pub ops_distinct: u32,
    pub ops_unused: Option<u32>,
    pub edit_ops: BTreeMap<EditOp, EditOpStats>,
    pub size_clauses: Quantiles,
    pub complexity: Quantiles,
    pub status: StatusCounts,
    /// Gate-passers only. Sentinel fitness values never enter these quantiles.
    pub fitness_passers: Quantiles,
    pub best_fitness: Option<f64>,
    pub gens_since_improved: u32,
    pub provenance_population: ProvenanceFracs,
    pub provenance_passers: ProvenanceFracs,
    pub provenance_hwm: ProvenanceFracs,
}

/// Per-context state that spans generations: best fitness, stagnation and the behavior archive.
pub struct Tracker {
    cfg: Config,
    context: String,
    best: Option<f64>,
    gens_since_improved: u32,
    archive: Vec<Vec<f64>>,
    archive_seen: u64,
    rng: SplitMix64,
}

impl Tracker {
    pub fn new(context: impl Into<String>, cfg: Config) -> Self {
        let rng = SplitMix64(cfg.seed);
        Self {
            cfg,
            context: context.into(),
            best: None,
            gens_since_improved: 0,
            archive: Vec::new(),
            archive_seen: 0,
            rng,
        }
    }

    pub fn summarize(
        &mut self,
        round: u32,
        generation: u32,
        cands: &[Candidate],
        dedup: ProposalCounters,
    ) -> GenSummary {
        let candidates_truncated = cands.len().saturating_sub(self.cfg.max_candidates) as u32;
        let cands = &cands[..cands.len().min(self.cfg.max_candidates)];
        let genomes_truncated = cands.iter().filter(|c| c.genome.truncated).count() as u32;
        let n = cands.len();
        let frac = |k: usize, d: usize| (d > 0).then(|| k as f64 / d as f64);

        let unique: HashSet<&[u8; 32]> = cands.iter().map(|c| &c.program).collect();

        let mut rng = SplitMix64(self.cfg.seed ^ ((round as u64) << 32 | generation as u64));
        let pairs = sample_pairs(&mut rng, n, self.cfg.max_pairs);
        let mut dists: Vec<f64> = pairs
            .iter()
            .map(|&(i, j)| levenshtein(&cands[i].genome.clauses, &cands[j].genome.clauses) as f64)
            .collect();
        let edit_distance_mean = mean(&dists);
        dists.sort_by(f64::total_cmp);
        let edit_distance_p10 = quantile_sorted(&dists, 0.10);

        let mut parent_child_hist = [0u32; 8];
        for c in cands {
            if let Some(p) = &c.parent {
                parent_child_hist[levenshtein(&p.clauses, &c.genome.clauses).min(7)] += 1;
            }
        }

        let behavior = self.behavior_stats(&mut rng, cands);

        let p16: Vec<u16> = cands.iter().filter_map(|c| c.proj16).collect();
        let proj16_unique_frac =
            frac(p16.iter().collect::<HashSet<_>>().len(), p16.len()).filter(|_| !p16.is_empty());
        let hp = sample_pairs(&mut rng, p16.len(), self.cfg.max_pairs);
        let proj16_mean_hamming = mean(
            &hp.iter()
                .map(|&(i, j)| (p16[i] ^ p16[j]).count_ones() as f64)
                .collect::<Vec<_>>(),
        );

        let mut op_counts: BTreeMap<u32, u64> = BTreeMap::new();
        for c in cands {
            for &op in &c.genome.ops {
                *op_counts.entry(op).or_default() += 1;
            }
        }
        let total_ops: u64 = op_counts.values().sum();
        let op_entropy_nats = (total_ops > 0).then(|| {
            -op_counts
                .values()
                .map(|&k| {
                    let p = k as f64 / total_ops as f64;
                    p * p.ln()
                })
                .sum::<f64>()
        });
        let ops_distinct = op_counts.len() as u32;
        let op_entropy_norm = match (op_entropy_nats, self.cfg.op_vocab_size) {
            (Some(h), Some(v)) if v > 1 => Some(h / (v as f64).ln()),
            _ => None,
        };
        let ops_unused = self
            .cfg
            .op_vocab_size
            .map(|v| (v as u32).saturating_sub(ops_distinct));

        let mut edit_ops: BTreeMap<EditOp, EditOpStats> = BTreeMap::new();
        for c in cands {
            let e = edit_ops.entry(c.edit_op).or_default();
            e.n += 1;
            if let Some(imp) = c.improved {
                e.known += 1;
                e.improved += imp as u32;
            }
        }
        for e in edit_ops.values_mut() {
            e.share = e.n as f64 / n as f64;
            e.success_rate = (e.known > 0).then(|| e.improved as f64 / e.known as f64);
        }

        let mut status = StatusCounts::default();
        let mut passer_fitness = Vec::new();
        for c in cands {
            match c.status {
                EvalStatus::Ok => status.ok += 1,
                EvalStatus::Timeout => status.timeout += 1,
                EvalStatus::NonzeroRc => status.nonzero_rc += 1,
                EvalStatus::ParseFail => status.parse_fail += 1,
                EvalStatus::Failed => status.failed_unknown_cause += 1,
            }
            if c.status == EvalStatus::Ok {
                if c.gate_passed && c.fitness.is_finite() {
                    status.gate_passed += 1;
                    passer_fitness.push(c.fitness);
                } else {
                    status.gate_failed += 1;
                }
            }
        }
        let fitness_passers = Quantiles::of(&mut passer_fitness);
        match (fitness_passers.min, self.best) {
            (Some(m), Some(b)) if m < b => {
                self.best = Some(m);
                self.gens_since_improved = 0;
            }
            (Some(m), None) => {
                self.best = Some(m);
                self.gens_since_improved = 0;
            }
            _ => self.gens_since_improved += 1,
        }

        GenSummary {
            schema: GEN_SCHEMA.to_owned(),
            context: self.context.clone(),
            round,
            generation,
            pop_size: n as u32,
            candidates_truncated,
            genomes_truncated,
            unique_genotype_frac: frac(unique.len(), n),
            dedup,
            dedup_reject_frac: frac(dedup.seen_rejects as usize, dedup.proposals as usize),
            edit_distance_pairs: pairs.len() as u32,
            edit_distance_mean,
            edit_distance_p10,
            parent_child_hist,
            behavior,
            proj16_unique_frac,
            proj16_mean_hamming,
            op_counts,
            op_entropy_nats,
            op_entropy_norm,
            ops_distinct,
            ops_unused,
            edit_ops,
            size_clauses: Quantiles::of(
                &mut cands
                    .iter()
                    .map(|c| c.genome.clauses.len() as f64)
                    .collect::<Vec<_>>(),
            ),
            complexity: Quantiles::of(
                &mut cands
                    .iter()
                    .filter_map(|c| c.complexity.map(f64::from))
                    .collect::<Vec<_>>(),
            ),
            status,
            fitness_passers,
            best_fitness: self.best,
            gens_since_improved: self.gens_since_improved,
            provenance_population: ProvenanceFracs::of(cands.iter()),
            provenance_passers: ProvenanceFracs::of(cands.iter().filter(|c| c.gate_passed)),
            provenance_hwm: ProvenanceFracs::of(cands.iter().filter(|c| c.hwm_winner)),
        }
    }

    fn behavior_stats(&mut self, rng: &mut SplitMix64, cands: &[Candidate]) -> BehaviorStats {
        let cap = self.cfg.max_behavior_len;
        let with: Vec<&[f64]> = cands
            .iter()
            .map(|c| &c.behavior[..c.behavior.len().min(cap)])
            .filter(|b| b.iter().filter(|v| v.is_finite()).count() >= self.cfg.min_common_positions)
            .collect();
        if with.is_empty() {
            return BehaviorStats {
                archive_size: self.archive.len() as u32,
                availability: Some("no_behavior_vectors".into()),
                prediction_availability: Some(NO_PREDICTION_SIGNATURES.into()),
                ..Default::default()
            };
        }
        let pairs = sample_pairs(rng, with.len(), self.cfg.max_pairs);
        let min = self.cfg.min_common_positions;
        let div: Vec<f64> = pairs
            .iter()
            .filter_map(|&(i, j)| match self.cfg.behavior_kind {
                BehaviorKind::ProbePanel => spearman(with[i], with[j], min).map(|rho| 1.0 - rho),
                BehaviorKind::ErrorProfile => rms_distance(with[i], with[j], min),
            })
            .collect();
        let q = self.cfg.behavior_quantum;
        let keys: HashSet<Vec<i64>> = with
            .iter()
            .map(|b| {
                b.iter()
                    .map(|&v| {
                        if v.is_finite() {
                            (v / q).round() as i64
                        } else {
                            i64::MIN
                        }
                    })
                    .collect()
            })
            .collect();

        let queries: Vec<usize> = if with.len() <= self.cfg.knn_max_queries {
            (0..with.len()).collect()
        } else {
            let mut idx: Vec<usize> = (0..with.len()).collect();
            partial_shuffle(rng, &mut idx, self.cfg.knn_max_queries);
            idx.truncate(self.cfg.knn_max_queries);
            idx
        };
        let archive_knn_mean = if self.archive.is_empty() {
            None
        } else {
            let k = self.cfg.knn_k.min(self.archive.len()).max(1);
            let per: Vec<f64> = queries
                .iter()
                .filter_map(|&qi| {
                    let mut d: Vec<f64> = self
                        .archive
                        .iter()
                        .filter_map(|a| rms_distance(with[qi], a, self.cfg.min_common_positions))
                        .collect();
                    if d.is_empty() {
                        return None;
                    }
                    d.sort_by(f64::total_cmp);
                    Some(d[..k.min(d.len())].iter().sum::<f64>() / k.min(d.len()) as f64)
                })
                .collect();
            mean(&per)
        };
        for b in &with {
            self.archive_insert(b);
        }
        BehaviorStats {
            kind: Some(self.cfg.behavior_kind),
            n_with_behavior: with.len() as u32,
            pairs: div.len() as u32,
            fitness_profile_diversity: mean(&div),
            fitness_profile_unique_frac: Some(keys.len() as f64 / with.len() as f64),
            prediction_diversity: None,
            prediction_availability: Some(NO_PREDICTION_SIGNATURES.into()),
            archive_knn_mean,
            archive_size: self.archive.len() as u32,
            availability: None,
        }
    }

    /// Reservoir sampling (Algorithm R) over every behavior vector this tracker has seen.
    fn archive_insert(&mut self, b: &[f64]) {
        self.archive_seen += 1;
        if self.archive.len() < self.cfg.archive_capacity {
            self.archive.push(b.to_vec());
        } else {
            let j = self.rng.below(self.archive_seen);
            if (j as usize) < self.cfg.archive_capacity {
                self.archive[j as usize] = b.to_vec();
            }
        }
    }
}

fn mean(v: &[f64]) -> Option<f64> {
    (!v.is_empty()).then(|| v.iter().sum::<f64>() / v.len() as f64)
}

/// Clause-level Levenshtein distance.
pub fn levenshtein(a: &[u32], b: &[u32]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0; b.len() + 1];
    for (i, x) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, y) in b.iter().enumerate() {
            cur[j + 1] = (prev[j] + (x != y) as usize)
                .min(prev[j + 1] + 1)
                .min(cur[j] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// Spearman ρ over the positions finite in both vectors (average ranks for ties). `None` when
/// fewer than `min_common` positions remain or either side is constant.
pub fn spearman(a: &[f64], b: &[f64], min_common: usize) -> Option<f64> {
    let (xa, xb): (Vec<f64>, Vec<f64>) = a
        .iter()
        .zip(b)
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .map(|(x, y)| (*x, *y))
        .unzip();
    if xa.len() < min_common.max(2) {
        return None;
    }
    pearson(&ranks(&xa), &ranks(&xb))
}

fn ranks(v: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&i, &j| v[i].total_cmp(&v[j]));
    let mut r = vec![0.0; v.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && v[idx[j + 1]] == v[idx[i]] {
            j += 1;
        }
        let avg = (i + j) as f64 / 2.0 + 1.0;
        for &k in &idx[i..=j] {
            r[k] = avg;
        }
        i = j + 1;
    }
    r
}

fn pearson(a: &[f64], b: &[f64]) -> Option<f64> {
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        sab += (x - ma) * (y - mb);
        saa += (x - ma) * (x - ma);
        sbb += (y - mb) * (y - mb);
    }
    (saa > 0.0 && sbb > 0.0).then(|| sab / (saa * sbb).sqrt())
}

fn rms_distance(a: &[f64], b: &[f64], min_common: usize) -> Option<f64> {
    let (mut s, mut k) = (0.0, 0usize);
    for (x, y) in a.iter().zip(b) {
        if x.is_finite() && y.is_finite() {
            s += (x - y) * (x - y);
            k += 1;
        }
    }
    (k >= min_common.max(1)).then(|| (s / k as f64).sqrt())
}

/// Up to `max` distinct unordered index pairs: all of them when that is few enough, otherwise
/// a seeded uniform sample without replacement.
fn sample_pairs(rng: &mut SplitMix64, n: usize, max: usize) -> Vec<(usize, usize)> {
    let total = n * n.saturating_sub(1) / 2;
    if total <= max {
        let mut out = Vec::with_capacity(total);
        for i in 0..n {
            for j in i + 1..n {
                out.push((i, j));
            }
        }
        return out;
    }
    let mut seen = HashSet::with_capacity(max);
    let mut out = Vec::with_capacity(max);
    while out.len() < max {
        let i = rng.below(n as u64) as usize;
        let j = rng.below(n as u64) as usize;
        if i == j {
            continue;
        }
        let p = (i.min(j), i.max(j));
        if seen.insert(p) {
            out.push(p);
        }
    }
    out
}

fn partial_shuffle(rng: &mut SplitMix64, v: &mut [usize], k: usize) {
    for i in 0..k.min(v.len()) {
        let j = i + rng.below((v.len() - i) as u64) as usize;
        v.swap(i, j);
    }
}

/// SplitMix64: small, seedable, and identical on every platform.
#[derive(Clone)]
pub struct SplitMix64(pub u64);

impl SplitMix64 {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in `0..n` (Lemire's multiply-shift; the bias is below 2^-32 for these sizes).
    pub fn below(&mut self, n: u64) -> u64 {
        ((self.next_u64() as u128 * n as u128) >> 64) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_matches_python_reference_cases() {
        let dsl = "(pipe X (poly_interact) (bin_onehot append=true bins=14) (histgbm depth=6 lr=0.08 rounds=160))";
        let (stages, head) = parse_pipe_dsl(dsl);
        assert_eq!(
            stages,
            vec!["(poly_interact)", "(bin_onehot append=true bins=14)"]
        );
        assert_eq!(head, "(histgbm depth=6 lr=0.08 rounds=160)");
        assert_eq!(parse_pipe_dsl("(ridge)"), (vec![], "(ridge_gcv)"));
        assert_eq!(parse_pipe_dsl("(pipe X"), (vec![], "(ridge_gcv)"));
        assert_eq!(parse_pipe_dsl("(pipe X )"), (vec![], "(ridge_gcv)"));
        assert_eq!(
            parse_pipe_dsl("  (pipe X (a (b)) (c))  "),
            (vec!["(a (b))"], "(c)")
        );
        assert_eq!(clause_op("(winsor_clip q=0.07)"), "winsor_clip");
        assert_eq!(clause_op("(poly_interact)"), "poly_interact");
    }

    #[test]
    fn interner_is_bounded_and_overflow_ids_stay_distinct() {
        let mut i = Interner::with_bounds(2, 1 << 20);
        let a = i.intern("(a)");
        let b = i.intern("(b)");
        let c = i.intern("(c)");
        let d = i.intern("(d)");
        assert_eq!((a, b), (0, 1));
        assert!(c & HASHED_ID_BIT != 0 && d & HASHED_ID_BIT != 0 && c != d);
        assert_eq!(i.intern("(c)"), c, "hashed ids are stable");
        assert_eq!((i.len(), i.overflowed()), (2, 3));
        let mut j = Interner::with_bounds(100, 4);
        j.intern("(ab)");
        assert!(
            j.intern("(cd)") & HASHED_ID_BIT != 0,
            "byte bound applies too"
        );
    }

    #[test]
    fn candidate_genome_and_behavior_bounds_are_enforced() {
        let (mut ci, mut oi) = (Interner::default(), Interner::default());
        let long = format!("(pipe X {} (h))", "(s) ".repeat(100));
        let g = Genome::from_dsl(&long, &mut ci, &mut oi);
        assert!(g.truncated && g.clauses.len() == MAX_CLAUSES);
        assert_eq!(
            *g.clauses.last().unwrap(),
            ci.intern("(h)"),
            "the head is kept"
        );
        let cand = |i: usize| Candidate {
            program: program_sha256(&i.to_string()),
            genome: g.clone(),
            parent: None,
            edit_op: EditOp::Other,
            status: EvalStatus::Ok,
            gate_passed: true,
            fitness: -(i as f64),
            improved: None,
            hwm_winner: false,
            complexity: None,
            proj16: None,
            behavior: vec![0.5; 100],
            provenance: Provenance::Novel,
        };
        let cands: Vec<Candidate> = (0..12).map(cand).collect();
        let cfg = Config {
            max_candidates: 10,
            max_behavior_len: 4,
            ..Default::default()
        };
        let mut t = Tracker::new("c", cfg);
        let s = t.summarize(0, 0, &cands, ProposalCounters::default());
        assert_eq!(
            (s.pop_size, s.candidates_truncated, s.genomes_truncated),
            (10, 2, 10)
        );
        t.summarize(0, 1, &cands, ProposalCounters::default());
        assert!(t.archive.iter().all(|b| b.len() == 4));
        assert_eq!(ci.dictionary(), vec!["(s)", "(h)"]);
    }

    #[test]
    fn levenshtein_and_spearman() {
        assert_eq!(levenshtein(&[1, 2, 3], &[1, 3]), 1);
        assert_eq!(levenshtein(&[], &[4, 5]), 2);
        assert_eq!(levenshtein(&[1, 2], &[2, 1]), 2);
        let r = spearman(&[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0], 3).unwrap();
        assert!((r - 1.0).abs() < 1e-12);
        let r = spearman(&[1.0, 2.0, 3.0, 4.0], &[4.0, 3.0, 2.0, 1.0], 3).unwrap();
        assert!((r + 1.0).abs() < 1e-12);
        assert!(spearman(&[1.0, f64::NAN, 3.0], &[1.0, 2.0, 3.0], 3).is_none());
        assert!(spearman(&[1.0, 1.0, 1.0], &[1.0, 2.0, 3.0], 3).is_none());
    }

    #[test]
    fn quantiles_match_numpy_linear() {
        let mut v = vec![4.0, 1.0, 3.0, 2.0];
        let q = Quantiles::of(&mut v);
        assert_eq!(q.p50, Some(2.5));
        assert!((q.p10.unwrap() - 1.3).abs() < 1e-12);
        assert!((q.iqr.unwrap() - 1.5).abs() < 1e-12);
    }

    #[test]
    fn sampled_pairs_are_distinct_and_seeded() {
        let mut a = SplitMix64(7);
        let mut b = SplitMix64(7);
        let pa = sample_pairs(&mut a, 100, 50);
        assert_eq!(pa, sample_pairs(&mut b, 100, 50));
        assert_eq!(pa.iter().collect::<HashSet<_>>().len(), 50);
        assert!(pa.iter().all(|&(i, j)| i < j && j < 100));
        assert_eq!(sample_pairs(&mut a, 4, 50).len(), 6);
    }
}
