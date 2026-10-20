//! RP-02 executable adapters, ported from the standalone rp02_adapters crate (SC-DEC-184AT item 2).
//! Provenance: HWM_OPERATORS is verified: the 24 names equal the operator heads used by all 7120
//! programs of hwm_programs.json (sha256 48fa43eb...). The grammar caps are NOT verified: the grammar.json
//! copies on this host hash to 93b1f076..., not the cited af991704.... EPSILON_SCALE_FACTOR is prompt-only.
//! Standalone, dependency-free crate kept outside the W2 workspace so it cannot change W2's Cargo.lock.
//! It runs no model, reads no dataset, and performs no evaluation.
//!
//! Provenance of constants (read this before relying on them):
//! - `SEED_CAP` = 4: packet-v1 WORKPLAN.md line 90 ("Seed cap is four").
//! - `HWM_OPERATORS` (24 names), `GRAMMAR_MAX_BYTES`, `GRAMMAR_PIPE_STAGES`, `GRAMMAR_SYM_NODES`:
//!   supplied by the RP-02 prompt. The files `hwm_programs.json` and `grammar.json` named there are NOT
//!   present on this host, so these values are NOT verified against them.
//! - `EPSILON_SCALE_FACTOR` = 1e-6: supplied by the RP-02 prompt. Packet-v1 SUCCESS_CRITERIA.md and
//!   experiment-design.md say only "fixed from permitted fitting-side target scale"; neither gives 1e-6.
use std::collections::{BTreeMap, BTreeSet};

pub const FORBIDDEN: &[&str] = &[
    "sberbank",
    "housing",
    "beyondarena",
    "kaggle",
    "competition",
    "golden42",
    "raw/",
    "s3://",
];

/// Prompt-sourced; see module note.
pub const HWM_OPERATORS: &[&str] = &[
    "bin_onehot",
    "blom",
    "cheb",
    "elasticnet",
    "hinge",
    "histgbm",
    "ksmooth",
    "oblique_hinge",
    "pair_prod",
    "pipe",
    "poly_interact",
    "quantile_bins",
    "rand_proj",
    "rank",
    "rff_cos",
    "rff_sin",
    "ridge",
    "ridge_gcv",
    "robust_scale",
    "row_stats",
    "sigmoid",
    "standardize",
    "target_bin_encode",
    "winsor_clip",
];
/// Prompt-sourced; see module note.
pub const GRAMMAR_MAX_BYTES: usize = 32768;
pub const GRAMMAR_PIPE_STAGES: usize = 10;
pub const GRAMMAR_SYM_NODES: usize = 64;
/// Packet-v1 WORKPLAN.md line 90.
pub const SEED_CAP: usize = 4;
/// Prompt-sourced; see module note.
pub const EPSILON_SCALE_FACTOR: f64 = 1e-6;
/// Packet-v1 `A0` contract minimum.
pub const MIN_FITTING_TARGETS: usize = 8;

#[derive(Debug, PartialEq)]
pub enum AdapterError {
    Forbidden(String),
    SameFamily,
    NoEligibleCandidate,
    Malformed(String),
    ProtectedCore(String),
    UnknownOperator(String),
    BudgetExhausted(&'static str),
    OverlappingRows(&'static str),
    TooFewTargets { n: usize },
    NonFiniteTarget,
    DegenerateTarget,
    BootstrapPopulationExceedsCap { requested: usize, cap: usize },
    SeedCapExceeded { requested: usize, cap: usize },
    NonFiniteScore,
    EvaluatorFailed(String),
}

#[derive(Clone, Debug)]
pub struct ColumnMoments {
    pub token: String,
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
}

pub fn render_card(slot: &str, columns: &[ColumnMoments]) -> Result<String, AdapterError> {
    let mut out = format!("slot {slot}\n");
    for c in columns {
        out.push_str(&format!(
            "{} mean={:.4} std={:.4} min={:.4} max={:.4}\n",
            c.token, c.mean, c.std, c.min, c.max
        ));
    }
    let lower = out.to_lowercase();
    for bad in FORBIDDEN {
        if lower.contains(bad) {
            return Err(AdapterError::Forbidden(bad.to_string()));
        }
    }
    Ok(out)
}

#[derive(Clone, Debug)]
pub struct Candidate {
    pub family: String,
    pub confirmation: bool,
    pub score: f64,
}

pub fn retrieve(
    query_family: &str,
    candidates: &[Candidate],
    k: usize,
) -> Result<Vec<Candidate>, AdapterError> {
    if candidates
        .iter()
        .any(|c| c.family == query_family && !c.confirmation)
    {
        return Err(AdapterError::SameFamily);
    }
    let mut eligible: Vec<Candidate> = candidates
        .iter()
        .filter(|c| !c.confirmation && c.family != query_family)
        .cloned()
        .collect();
    if eligible.is_empty() {
        return Err(AdapterError::NoEligibleCandidate);
    }
    eligible.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.family.cmp(&b.family))
    });
    eligible.truncate(k);
    Ok(eligible)
}

#[derive(Clone, Debug, PartialEq)]
pub enum TypedEdit {
    ReplaceSubtree { path: Vec<usize>, with: String },
    WrapCombinator { name: String },
    TuneHyperparam { key: String, value: f64 },
}

/// Parse one proposer line: `ReplaceSubtree <path,comma> <expr>`, `WrapCombinator <name>`,
/// `TuneHyperparam <key> <f64>`. Protected-core and unknown operators fail closed.
pub fn parse_edit(line: &str) -> Result<TypedEdit, AdapterError> {
    let trimmed = line.trim();
    let mut parts = trimmed.splitn(3, ' ');
    let op = parts.next().unwrap_or("");
    match op {
        "ReplaceSubtree" => {
            let path_text = parts
                .next()
                .ok_or_else(|| AdapterError::Malformed("missing path".into()))?;
            let with = parts
                .next()
                .ok_or_else(|| AdapterError::Malformed("missing expression".into()))?;
            let path = path_text
                .split(',')
                .map(|p| {
                    p.parse::<usize>()
                        .map_err(|_| AdapterError::Malformed(format!("bad path {p}")))
                })
                .collect::<Result<Vec<_>, _>>()?;
            validate_expression(with)?;
            Ok(TypedEdit::ReplaceSubtree {
                path,
                with: with.to_string(),
            })
        }
        "WrapCombinator" => {
            let name = parts
                .next()
                .ok_or_else(|| AdapterError::Malformed("missing name".into()))?;
            if !HWM_OPERATORS.contains(&name) {
                return Err(AdapterError::UnknownOperator(name.to_string()));
            }
            Ok(TypedEdit::WrapCombinator {
                name: name.to_string(),
            })
        }
        "TuneHyperparam" => {
            let key = parts
                .next()
                .ok_or_else(|| AdapterError::Malformed("missing key".into()))?;
            let value_text = parts
                .next()
                .ok_or_else(|| AdapterError::Malformed("missing value".into()))?;
            let value: f64 = value_text
                .parse()
                .map_err(|_| AdapterError::Malformed(format!("bad value {value_text}")))?;
            if !value.is_finite() {
                return Err(AdapterError::Malformed("non-finite hyperparameter".into()));
            }
            Ok(TypedEdit::TuneHyperparam {
                key: key.to_string(),
                value,
            })
        }
        "ProtectedCoreEdit" => Err(AdapterError::ProtectedCore(trimmed.to_string())),
        other => Err(AdapterError::UnknownOperator(other.to_string())),
    }
}

/// Grammar checks on an S-expression: byte cap, balanced parentheses, every operator head in the
/// whitelist, at most `GRAMMAR_SYM_NODES` opening parentheses, and at most `GRAMMAR_PIPE_STAGES`
/// stages in any `(pipe ...)` (counted as its direct child forms).
pub fn validate_expression(expr: &str) -> Result<(), AdapterError> {
    if expr.len() > GRAMMAR_MAX_BYTES {
        return Err(AdapterError::Malformed(format!(
            "expression exceeds {GRAMMAR_MAX_BYTES} bytes"
        )));
    }
    let tokens = tokenize(expr)?;
    let sym_nodes = tokens.iter().filter(|t| t.as_str() == "(").count();
    if sym_nodes > GRAMMAR_SYM_NODES {
        return Err(AdapterError::Malformed(format!(
            "{sym_nodes} symbol nodes exceed {GRAMMAR_SYM_NODES}"
        )));
    }
    for (i, t) in tokens.iter().enumerate() {
        if t == "(" {
            let head = tokens
                .get(i + 1)
                .ok_or_else(|| AdapterError::Malformed("empty form".into()))?;
            if head == "(" || head == ")" {
                return Err(AdapterError::Malformed("form without operator head".into()));
            }
            if !HWM_OPERATORS.contains(&head.as_str()) {
                return Err(AdapterError::UnknownOperator(head.clone()));
            }
        }
    }
    check_pipe_stages(&tokens)
}

fn tokenize(expr: &str) -> Result<Vec<String>, AdapterError> {
    let mut tokens = Vec::new();
    let mut depth: i64 = 0;
    let spaced = expr.replace('(', " ( ").replace(')', " ) ");
    for tok in spaced.split_whitespace() {
        match tok {
            "(" => {
                depth += 1;
                tokens.push("(".to_string());
            }
            ")" => {
                depth -= 1;
                if depth < 0 {
                    return Err(AdapterError::Malformed("unbalanced parentheses".into()));
                }
                tokens.push(")".to_string());
            }
            other => tokens.push(other.to_string()),
        }
    }
    if depth != 0 {
        return Err(AdapterError::Malformed("unbalanced parentheses".into()));
    }
    Ok(tokens)
}

fn check_pipe_stages(tokens: &[String]) -> Result<(), AdapterError> {
    let mut i = 0;
    while i < tokens.len() {
        if tokens[i] == "(" && tokens.get(i + 1).map(String::as_str) == Some("pipe") {
            let mut depth = 0i64;
            let mut stages = 0usize;
            let mut j = i;
            while j < tokens.len() {
                match tokens[j].as_str() {
                    "(" => {
                        depth += 1;
                        if depth == 2 {
                            stages += 1;
                        }
                    }
                    ")" => {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                    _ => {}
                }
                j += 1;
            }
            if stages > GRAMMAR_PIPE_STAGES {
                return Err(AdapterError::Malformed(format!(
                    "pipe has {stages} stages, cap {GRAMMAR_PIPE_STAGES}"
                )));
            }
        }
        i += 1;
    }
    Ok(())
}

pub fn serialize_edit(edit: &TypedEdit) -> String {
    match edit {
        TypedEdit::ReplaceSubtree { path, with } => {
            let p: Vec<String> = path.iter().map(|x| x.to_string()).collect();
            format!("ReplaceSubtree {} {}", p.join(","), with)
        }
        TypedEdit::WrapCombinator { name } => format!("WrapCombinator {name}"),
        TypedEdit::TuneHyperparam { key, value } => format!("TuneHyperparam {key} {value}"),
    }
}

pub fn dup_key(canonical: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in canonical.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotOutcome {
    Ok,
    Malformed,
    Duplicate,
    ProtectedCore,
    Timeout,
    NonFinite,
}

#[derive(Clone, Debug, PartialEq)]
pub struct JournalEntry {
    pub label: String,
    pub outcome: SlotOutcome,
    pub fits_charged: u64,
    pub fallback_used: bool,
}

#[derive(Debug)]
pub struct Journal {
    slot_cap: u64,
    fit_cap: u64,
    slots_charged: u64,
    fits_charged: u64,
    seen: BTreeSet<u64>,
    entries: Vec<JournalEntry>,
    open_label: Option<String>,
}

impl Journal {
    pub fn new(slot_cap: u64, fit_cap: u64) -> Self {
        Self {
            slot_cap,
            fit_cap,
            slots_charged: 0,
            fits_charged: 0,
            seen: BTreeSet::new(),
            entries: Vec::new(),
            open_label: None,
        }
    }

    pub fn begin_slot(&mut self, label: &str) -> Result<(), AdapterError> {
        if self.open_label.is_some() {
            return Err(AdapterError::Malformed(
                "a previous slot was not closed".into(),
            ));
        }
        if self.slots_charged >= self.slot_cap {
            return Err(AdapterError::BudgetExhausted("candidate slots"));
        }
        self.slots_charged += 1;
        self.open_label = Some(label.to_string());
        Ok(())
    }

    pub fn charge_fits(&mut self, n: u64) -> Result<(), AdapterError> {
        if self.fits_charged + n > self.fit_cap {
            return Err(AdapterError::BudgetExhausted("physical fold fits"));
        }
        self.fits_charged += n;
        Ok(())
    }

    /// True if this canonical edit was already credited in this journal.
    pub fn is_seen(&self, canonical: &str) -> bool {
        self.seen.contains(&dup_key(canonical))
    }

    pub fn close_slot(
        &mut self,
        canonical: &str,
        outcome: SlotOutcome,
        fits: u64,
    ) -> Result<(), AdapterError> {
        let label = self
            .open_label
            .take()
            .ok_or_else(|| AdapterError::Malformed("no open slot".into()))?;
        let key = dup_key(canonical);
        let effective = if outcome == SlotOutcome::Ok && !self.seen.insert(key) {
            SlotOutcome::Duplicate
        } else {
            outcome
        };
        self.entries.push(JournalEntry {
            label,
            outcome: effective,
            fits_charged: fits,
            fallback_used: effective != SlotOutcome::Ok,
        });
        Ok(())
    }

    pub fn entries(&self) -> &[JournalEntry] {
        &self.entries
    }

    pub fn totals(&self) -> BTreeMap<&'static str, u64> {
        let mut t = BTreeMap::new();
        t.insert("slots_charged", self.slots_charged);
        t.insert("fits_charged", self.fits_charged);
        t.insert(
            "credited",
            self.entries
                .iter()
                .filter(|e| e.outcome == SlotOutcome::Ok)
                .count() as u64,
        );
        t.insert(
            "fallbacks",
            self.entries.iter().filter(|e| e.fallback_used).count() as u64,
        );
        t
    }
}

pub fn check_disjoint(a: &[u32], b: &[u32], o: &[u32]) -> Result<(), AdapterError> {
    let sa: BTreeSet<u32> = a.iter().copied().collect();
    let sb: BTreeSet<u32> = b.iter().copied().collect();
    let so: BTreeSet<u32> = o.iter().copied().collect();
    if !sa.is_disjoint(&sb) {
        return Err(AdapterError::OverlappingRows("A and B overlap"));
    }
    if !sa.is_disjoint(&so) {
        return Err(AdapterError::OverlappingRows("A and O overlap"));
    }
    if !sb.is_disjoint(&so) {
        return Err(AdapterError::OverlappingRows("B and O overlap"));
    }
    Ok(())
}

/// Epsilon from the fitting-side target scale only. `y_a` must be the A-partition targets. Refuses
/// fewer than `MIN_FITTING_TARGETS` values, non-finite values, and degenerate (near-constant) targets.
pub fn compute_epsilon_f(y_a: &[f64]) -> Result<f64, AdapterError> {
    if y_a.len() < MIN_FITTING_TARGETS {
        return Err(AdapterError::TooFewTargets { n: y_a.len() });
    }
    if y_a.iter().any(|v| !v.is_finite()) {
        return Err(AdapterError::NonFiniteTarget);
    }
    let n = y_a.len() as f64;
    let mean = y_a.iter().sum::<f64>() / n;
    let var = y_a.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1.0);
    let sample_std = var.sqrt();
    if sample_std <= 1e-12 {
        return Err(AdapterError::DegenerateTarget);
    }
    Ok(EPSILON_SCALE_FACTOR * sample_std)
}

/// `log((rmse_m + eps) / (rmse_b + eps))`, refusing non-finite or non-positive inputs.
pub fn paired_log_rmse_ratio(
    rmse_m: f64,
    rmse_b: f64,
    epsilon_f: f64,
) -> Result<f64, AdapterError> {
    let finite = [rmse_m, rmse_b, epsilon_f].iter().all(|v| v.is_finite());
    if !finite || epsilon_f <= 0.0 || rmse_m + epsilon_f <= 0.0 || rmse_b + epsilon_f <= 0.0 {
        return Err(AdapterError::NonFiniteScore);
    }
    Ok(((rmse_m + epsilon_f) / (rmse_b + epsilon_f)).ln())
}

/// Seed and bootstrap budget validation. `low_budget_cap` is the level's whole-request cap, 64 for low.
pub fn validate_seed_and_bootstrap_budget(
    route: &str,
    initial_population: usize,
    seed_genomes_count: usize,
    low_budget_cap: usize,
) -> Result<(), AdapterError> {
    let _ = route;
    if initial_population > low_budget_cap {
        return Err(AdapterError::BootstrapPopulationExceedsCap {
            requested: initial_population,
            cap: low_budget_cap,
        });
    }
    if seed_genomes_count > SEED_CAP {
        return Err(AdapterError::SeedCapExceeded {
            requested: seed_genomes_count,
            cap: SEED_CAP,
        });
    }
    Ok(())
}

/// Scores candidates on the fitting-selection side (`B`) and, once the selection is frozen, on the
/// assessment side (`O`). The episode driver never gives this trait any row of `O` during selection.
pub trait FoldEvaluator {
    /// B-side RMSE and the physical fold fits this evaluation consumed.
    fn score_selection(&self, serialized: &str) -> Result<(f64, u64), AdapterError>;
    /// O-side RMSE of the frozen selection. Called at most once per episode.
    fn score_assessment(&self, serialized: &str) -> Result<f64, AdapterError>;
}

pub struct SeedBudgetConfig {
    pub route: String,
    pub initial_population: usize,
    pub seed_genomes: usize,
    pub low_budget_cap: usize,
    pub fits_per_candidate: u64,
}

#[derive(Debug, PartialEq)]
pub struct EpisodeResult {
    pub best: Option<(String, f64)>,
    pub credited: usize,
    pub episode_failed: bool,
}

/// Runs the proposer script through the journal. Each slot is charged before parsing or evaluation.
/// Parse failures, duplicates and protected-core edits close their slot without evaluation. Valid
/// novel edits charge their fits, then the selection evaluator. Selection uses B only.
pub fn run_fake_proposer_episode(
    script: &[&str],
    journal: &mut Journal,
    cfg: &SeedBudgetConfig,
    evaluator: &dyn FoldEvaluator,
) -> Result<EpisodeResult, AdapterError> {
    validate_seed_and_bootstrap_budget(
        &cfg.route,
        cfg.initial_population,
        cfg.seed_genomes,
        cfg.low_budget_cap,
    )?;
    let mut best: Option<(String, f64)> = None;
    let mut credited = 0usize;
    for (index, raw) in script.iter().enumerate() {
        let label = format!("slot-{index}");
        journal.begin_slot(&label)?;
        let edit = match parse_edit(raw) {
            Ok(e) => e,
            Err(err) => {
                let outcome = match err {
                    AdapterError::ProtectedCore(_) => SlotOutcome::ProtectedCore,
                    _ => SlotOutcome::Malformed,
                };
                journal.close_slot(raw, outcome, 0)?;
                continue;
            }
        };
        let canonical = serialize_edit(&edit);
        if journal.is_seen(&canonical) {
            journal.close_slot(&canonical, SlotOutcome::Duplicate, 0)?;
            continue;
        }
        journal.charge_fits(cfg.fits_per_candidate)?;
        let (rmse_b, fits) = evaluator.score_selection(&canonical)?;
        if !rmse_b.is_finite() {
            journal.close_slot(&canonical, SlotOutcome::NonFinite, fits)?;
            continue;
        }
        journal.close_slot(&canonical, SlotOutcome::Ok, fits)?;
        credited += 1;
        let better = match &best {
            None => true,
            Some((_, b)) => rmse_b < *b,
        };
        if better {
            best = Some((canonical, rmse_b));
        }
    }
    Ok(EpisodeResult {
        best,
        credited,
        episode_failed: credited == 0,
    })
}

#[derive(Debug, PartialEq)]
pub struct DisjointEpisode {
    pub epsilon_f: f64,
    pub selected: Option<String>,
    pub o_score: f64,
    pub episode_failed: bool,
}

/// Full disjoint episode. The fallback predictor is preregistered; `o_fallback` is its O-side RMSE,
/// computed once at completion. It is used only when no candidate was credited.
/// One argument per partition and budget input, so each stays named at the call site.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_episode_on_disjoint_folds(
    a_rows: &[u32],
    b_rows: &[u32],
    o_rows: &[u32],
    y_a: &[f64],
    script: &[&str],
    journal: &mut Journal,
    cfg: &SeedBudgetConfig,
    evaluator: &dyn FoldEvaluator,
    o_fallback: f64,
) -> Result<DisjointEpisode, AdapterError> {
    check_disjoint(a_rows, b_rows, o_rows)?;
    let epsilon_f = compute_epsilon_f(y_a)?;
    let result = run_fake_proposer_episode(script, journal, cfg, evaluator)?;
    if result.episode_failed {
        if !o_fallback.is_finite() {
            return Err(AdapterError::NonFiniteScore);
        }
        return Ok(DisjointEpisode {
            epsilon_f,
            selected: None,
            o_score: o_fallback,
            episode_failed: true,
        });
    }
    let (selected, _) = result.best.expect("credited > 0 implies a best candidate");
    let o_score = evaluator.score_assessment(&selected)?;
    if !o_score.is_finite() {
        return Err(AdapterError::NonFiniteScore);
    }
    Ok(DisjointEpisode {
        epsilon_f,
        selected: Some(selected),
        o_score,
        episode_failed: false,
    })
}
/// A5: a deterministic ridge fold fit over the `A` (fitting) rows. A candidate's canonical edit names
/// its feature subset (column `j` is used when bit `j` of its `dup_key` is set), and the parameters are
/// fitted on `A` alone. Selection scores `B`. `O` is read only by `score_assessment`, after the
/// selection is frozen, so `o_reads` stays at zero during selection and reaches one at closure.
pub struct RidgeFoldEvaluator<'a> {
    features: &'a [Vec<f64>],
    targets: &'a [f64],
    a_rows: &'a [u32],
    b_rows: &'a [u32],
    o_rows: &'a [u32],
    ridge_lambda: f64,
    counters: crate::search::ExecutionBoundaryCounters,
    o_reads: std::cell::Cell<u64>,
}

impl<'a> RidgeFoldEvaluator<'a> {
    pub fn new(
        features: &'a [Vec<f64>],
        targets: &'a [f64],
        a_rows: &'a [u32],
        b_rows: &'a [u32],
        o_rows: &'a [u32],
        ridge_lambda: f64,
    ) -> Result<Self, AdapterError> {
        check_disjoint(a_rows, b_rows, o_rows)?;
        if !(ridge_lambda.is_finite() && ridge_lambda > 0.0) {
            return Err(AdapterError::EvaluatorFailed(
                "ridge lambda must be finite and positive".into(),
            ));
        }
        let width = features.first().map_or(0, Vec::len);
        if features.len() != targets.len()
            || width == 0
            || width > 64
            || features.iter().any(|row| row.len() != width)
        {
            return Err(AdapterError::EvaluatorFailed(
                "features must be a rectangular table of 1..=64 columns, one target per row".into(),
            ));
        }
        for &row in a_rows.iter().chain(b_rows).chain(o_rows) {
            if row as usize >= features.len() {
                return Err(AdapterError::EvaluatorFailed(format!(
                    "row {row} is outside the table"
                )));
            }
        }
        Ok(Self {
            features,
            targets,
            a_rows,
            b_rows,
            o_rows,
            ridge_lambda,
            o_reads: std::cell::Cell::new(0),
            counters: crate::search::ExecutionBoundaryCounters::new(
                crate::search::ExecutionBoundaryCaps::default(),
            ),
        })
    }

    /// How many times the `O` rows have been read. Selection never reads them.
    pub fn o_reads(&self) -> u64 {
        self.o_reads.get()
    }

    fn subset(&self, serialized: &str) -> Vec<usize> {
        let mask = dup_key(serialized);
        (0..self.features[0].len())
            .filter(|&j| (mask >> j) & 1 == 1)
            .collect()
    }

    fn design(&self, row: u32, subset: &[usize]) -> Vec<f64> {
        let r = &self.features[row as usize];
        std::iter::once(1.0)
            .chain(subset.iter().map(|&j| r[j]))
            .collect()
    }

    /// Ridge coefficients fitted on the `A` rows alone. The intercept is not penalised.
    fn fit_on_a(&self, subset: &[usize]) -> Result<Vec<f64>, AdapterError> {
        let design: Vec<Vec<f64>> = self
            .a_rows
            .iter()
            .map(|&row| self.design(row, subset))
            .collect();
        let targets: Vec<f64> = self
            .a_rows
            .iter()
            .map(|&row| self.targets[row as usize])
            .collect();
        self.counters
            .charge_physical_fit()
            .map_err(AdapterError::BudgetExhausted)?;
        fit_ridge(&design, &targets, self.ridge_lambda)
    }

    fn rmse(&self, beta: &[f64], subset: &[usize], rows: &[u32]) -> f64 {
        let sse: f64 = rows
            .iter()
            .map(|&row| {
                let pred: f64 = beta
                    .iter()
                    .zip(self.design(row, subset))
                    .map(|(b, x)| b * x)
                    .sum();
                let err = pred - self.targets[row as usize];
                err * err
            })
            .sum();
        (sse / rows.len() as f64).sqrt()
    }
}

/// Gaussian elimination with partial pivoting. A pivot below `1e-12` refuses the fit rather than
/// returning coefficients from a singular system.
fn solve_linear(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Result<Vec<f64>, AdapterError> {
    let n = b.len();
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))
            .unwrap_or(col);
        if a[pivot][col].abs() < 1e-12 {
            return Err(AdapterError::EvaluatorFailed(
                "singular normal equations on the fitting rows".into(),
            ));
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        for row in col + 1..n {
            let factor = a[row][col] / a[col][col];
            let (upper, lower) = a.split_at_mut(row);
            for (dst, src) in lower[0][col..].iter_mut().zip(&upper[col][col..]) {
                *dst -= factor * src;
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let tail: f64 = (row + 1..n).map(|k| a[row][k] * x[k]).sum();
        x[row] = (b[row] - tail) / a[row][row];
    }
    Ok(x)
}

impl FoldEvaluator for RidgeFoldEvaluator<'_> {
    fn score_selection(&self, serialized: &str) -> Result<(f64, u64), AdapterError> {
        let subset = self.subset(serialized);
        let beta = self.fit_on_a(&subset)?;
        Ok((self.rmse(&beta, &subset, self.b_rows), 1))
    }

    fn score_assessment(&self, serialized: &str) -> Result<f64, AdapterError> {
        self.o_reads.set(self.o_reads.get() + 1);
        let subset = self.subset(serialized);
        let beta = self.fit_on_a(&subset)?;
        Ok(self.rmse(&beta, &subset, self.o_rows))
    }
}

/// Ridge least squares. `design` rows carry the intercept as their first column, which is not
/// penalised. Shared by the diagnostic ridge evaluator and the GP program evaluator.
fn fit_ridge(
    design: &[Vec<f64>],
    targets: &[f64],
    ridge_lambda: f64,
) -> Result<Vec<f64>, AdapterError> {
    let dim = design.first().map_or(0, Vec::len);
    let mut gram = vec![vec![0.0; dim]; dim];
    let mut rhs = vec![0.0; dim];
    for (x, &y) in design.iter().zip(targets) {
        for (i, (r, xi)) in rhs.iter_mut().zip(x).enumerate() {
            *r += xi * y;
            for (g, xj) in gram[i].iter_mut().zip(x) {
                *g += xi * xj;
            }
        }
    }
    for (i, row) in gram.iter_mut().enumerate().skip(1) {
        row[i] += ridge_lambda;
    }
    solve_linear(gram, rhs)
}

/// Deepest GP program accepted. Matches the authoritative grammar's `sym_depth` cap.
pub const GP_MAX_DEPTH: usize = 16;
/// Most nodes in one GP program. Matches the authoritative grammar's `sym_nodes` cap.
pub const GP_MAX_NODES: usize = GRAMMAR_SYM_NODES;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpOp {
    Add,
    Sub,
    Mul,
    /// Protected division: a denominator with magnitude below `1e-12` yields `1`.
    Div,
}

impl GpOp {
    fn name(self) -> &'static str {
        match self {
            GpOp::Add => "add",
            GpOp::Sub => "sub",
            GpOp::Mul => "mul",
            GpOp::Div => "div",
        }
    }
}

/// A GP program: a tree whose leaves are feature columns `f<j>` and whose inner nodes combine two
/// subtrees with [`GpOp`]. Its output is one feature column per row.
#[derive(Clone, Debug, PartialEq)]
pub enum GpProgram {
    Feature(usize),
    Op(GpOp, Box<GpProgram>, Box<GpProgram>),
}

impl GpProgram {
    /// Parses the canonical text form, for example `add(f0,f1)` or `mul(f0,sub(f2,f1))`.
    /// Refuses anything else, and any program past [`GP_MAX_DEPTH`] or [`GP_MAX_NODES`].
    pub fn parse(text: &str) -> Result<GpProgram, AdapterError> {
        let mut pos = 0;
        let mut nodes = 0;
        let program = parse_gp_node(text, &mut pos, 1, &mut nodes)?;
        if pos != text.len() {
            return Err(AdapterError::Malformed(format!(
                "trailing text after GP program: {text:?}"
            )));
        }
        Ok(program)
    }

    /// Evaluates the program on one row of feature values.
    pub fn eval(&self, row: &[f64]) -> f64 {
        match self {
            GpProgram::Feature(j) => row[*j],
            GpProgram::Op(op, a, b) => {
                let (x, y) = (a.eval(row), b.eval(row));
                match op {
                    GpOp::Add => x + y,
                    GpOp::Sub => x - y,
                    GpOp::Mul => x * y,
                    GpOp::Div => {
                        if y.abs() < 1e-12 {
                            1.0
                        } else {
                            x / y
                        }
                    }
                }
            }
        }
    }

    /// The largest feature index the program reads.
    pub fn max_feature(&self) -> usize {
        match self {
            GpProgram::Feature(j) => *j,
            GpProgram::Op(_, a, b) => a.max_feature().max(b.max_feature()),
        }
    }
}

impl std::fmt::Display for GpProgram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpProgram::Feature(j) => write!(f, "f{j}"),
            GpProgram::Op(op, a, b) => write!(f, "{}({a},{b})", op.name()),
        }
    }
}

fn parse_gp_node(
    text: &str,
    pos: &mut usize,
    depth: usize,
    nodes: &mut usize,
) -> Result<GpProgram, AdapterError> {
    let refuse = |why: &str| AdapterError::Malformed(format!("GP program {text:?}: {why}"));
    if depth > GP_MAX_DEPTH {
        return Err(refuse("exceeds the depth cap"));
    }
    *nodes += 1;
    if *nodes > GP_MAX_NODES {
        return Err(refuse("exceeds the node cap"));
    }
    let rest = &text[*pos..];
    if let Some(after) = rest.strip_prefix('f') {
        let digits: String = after.chars().take_while(char::is_ascii_digit).collect();
        if digits.is_empty() || (digits.len() > 1 && digits.starts_with('0')) {
            return Err(refuse("a feature must be f followed by a canonical index"));
        }
        let index: usize = digits
            .parse()
            .map_err(|_| refuse("feature index out of range"))?;
        *pos += 1 + digits.len();
        return Ok(GpProgram::Feature(index));
    }
    for op in [GpOp::Add, GpOp::Sub, GpOp::Mul, GpOp::Div] {
        let head = format!("{}(", op.name());
        if rest.starts_with(&head) {
            *pos += head.len();
            let a = parse_gp_node(text, pos, depth + 1, nodes)?;
            if text.as_bytes().get(*pos) != Some(&b',') {
                return Err(refuse("expected ',' between the operands"));
            }
            *pos += 1;
            let b = parse_gp_node(text, pos, depth + 1, nodes)?;
            if text.as_bytes().get(*pos) != Some(&b')') {
                return Err(refuse("expected ')' after the operands"));
            }
            *pos += 1;
            return Ok(GpProgram::Op(op, Box::new(a), Box::new(b)));
        }
    }
    Err(refuse("unknown operator or leaf"))
}

/// A5, real program semantics. The candidate's own GP program (see [`GpProgram`]) produces the
/// feature column, and ridge is fitted on that column over the `A` rows alone (intercept
/// unpenalised, lambda `1e-3`). Selection scores the `B` rows. `O` is read only by
/// `score_assessment`, after the selection is frozen. This is the scored path for the GP claim.
/// [`RidgeFoldEvaluator`] stays a diagnostic fixture.
pub struct GpProgramFoldEvaluator<'a> {
    features: &'a [Vec<f64>],
    targets: &'a [f64],
    a_rows: &'a [u32],
    b_rows: &'a [u32],
    o_rows: &'a [u32],
    ridge_lambda: f64,
    counters: crate::search::ExecutionBoundaryCounters,
    o_reads: std::cell::Cell<u64>,
}

impl<'a> GpProgramFoldEvaluator<'a> {
    pub fn new(
        features: &'a [Vec<f64>],
        targets: &'a [f64],
        a_rows: &'a [u32],
        b_rows: &'a [u32],
        o_rows: &'a [u32],
        ridge_lambda: f64,
    ) -> Result<Self, AdapterError> {
        // The same table and partition checks as the ridge evaluator, applied to the same inputs.
        RidgeFoldEvaluator::new(features, targets, a_rows, b_rows, o_rows, ridge_lambda)?;
        Ok(Self {
            features,
            targets,
            a_rows,
            b_rows,
            o_rows,
            ridge_lambda,
            o_reads: std::cell::Cell::new(0),
            counters: crate::search::ExecutionBoundaryCounters::new(
                crate::search::ExecutionBoundaryCaps::default(),
            ),
        })
    }

    /// Number of feature columns a program may read.
    pub fn width(&self) -> usize {
        self.features[0].len()
    }

    /// How many times the `O` rows have been read. Selection never reads them.
    pub fn o_reads(&self) -> u64 {
        self.o_reads.get()
    }

    /// Fits ridge for `program` on `A` and returns its RMSE on `rows`. A program whose output is
    /// non-finite on any `A` or scored row scores `NaN`, which the episode records as non-finite.
    fn fit_and_score(&self, program: &GpProgram, rows: &[u32]) -> Result<f64, AdapterError> {
        let column = |row: u32| {
            let value = program.eval(&self.features[row as usize]);
            value.is_finite().then(|| vec![1.0, value])
        };
        let Some(train) = self
            .a_rows
            .iter()
            .map(|&row| column(row))
            .collect::<Option<Vec<_>>>()
        else {
            return Ok(f64::NAN);
        };
        let Some(test) = rows
            .iter()
            .map(|&row| column(row))
            .collect::<Option<Vec<_>>>()
        else {
            return Ok(f64::NAN);
        };
        let y_a: Vec<f64> = self
            .a_rows
            .iter()
            .map(|&row| self.targets[row as usize])
            .collect();
        self.counters
            .charge_physical_fit()
            .map_err(AdapterError::BudgetExhausted)?;
        let beta = fit_ridge(&train, &y_a, self.ridge_lambda)?;
        let sse: f64 = test
            .iter()
            .zip(rows)
            .map(|(x, &row)| {
                let pred = beta[0] * x[0] + beta[1] * x[1];
                let err = pred - self.targets[row as usize];
                err * err
            })
            .sum();
        Ok((sse / rows.len() as f64).sqrt())
    }
}

impl FoldEvaluator for GpProgramFoldEvaluator<'_> {
    fn score_selection(&self, serialized: &str) -> Result<(f64, u64), AdapterError> {
        let program = GpProgram::parse(serialized)?;
        Ok((self.fit_and_score(&program, self.b_rows)?, 1))
    }

    fn score_assessment(&self, serialized: &str) -> Result<f64, AdapterError> {
        self.o_reads.set(self.o_reads.get() + 1);
        let program = GpProgram::parse(serialized)?;
        self.fit_and_score(&program, self.o_rows)
    }
}

/// The disjoint episode for GP programs. Each program text is one slot, charged before it is
/// parsed. A parse failure, a program reading a feature the table does not have, or a duplicate
/// closes its slot without evaluation. Selection uses `B` only. `O` is scored once, after the
/// selection is frozen. The fallback applies only when no program was credited.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_program_episode(
    a_rows: &[u32],
    b_rows: &[u32],
    o_rows: &[u32],
    y_a: &[f64],
    programs: &[&str],
    journal: &mut Journal,
    cfg: &SeedBudgetConfig,
    evaluator: &GpProgramFoldEvaluator,
    o_fallback: f64,
) -> Result<DisjointEpisode, AdapterError> {
    check_disjoint(a_rows, b_rows, o_rows)?;
    let epsilon_f = compute_epsilon_f(y_a)?;
    validate_seed_and_bootstrap_budget(
        &cfg.route,
        cfg.initial_population,
        cfg.seed_genomes,
        cfg.low_budget_cap,
    )?;
    let mut best: Option<(String, f64)> = None;
    let mut credited = 0usize;
    for (index, raw) in programs.iter().enumerate() {
        journal.begin_slot(&format!("slot-{index}"))?;
        let program = match GpProgram::parse(raw) {
            Ok(program) if program.max_feature() < evaluator.width() => program,
            _ => {
                journal.close_slot(raw, SlotOutcome::Malformed, 0)?;
                continue;
            }
        };
        let canonical = program.to_string();
        if journal.is_seen(&canonical) {
            journal.close_slot(&canonical, SlotOutcome::Duplicate, 0)?;
            continue;
        }
        journal.charge_fits(cfg.fits_per_candidate)?;
        let rmse_b = evaluator.score_selection(&canonical)?.0;
        if !rmse_b.is_finite() {
            journal.close_slot(&canonical, SlotOutcome::NonFinite, 1)?;
            continue;
        }
        journal.close_slot(&canonical, SlotOutcome::Ok, 1)?;
        credited += 1;
        if best.as_ref().is_none_or(|(_, b)| rmse_b < *b) {
            best = Some((canonical, rmse_b));
        }
    }
    if credited == 0 {
        if !o_fallback.is_finite() {
            return Err(AdapterError::NonFiniteScore);
        }
        return Ok(DisjointEpisode {
            epsilon_f,
            selected: None,
            o_score: o_fallback,
            episode_failed: true,
        });
    }
    let (selected, _) = best.ok_or(AdapterError::NoEligibleCandidate)?;
    let o_score = evaluator.score_assessment(&selected)?;
    if !o_score.is_finite() {
        return Err(AdapterError::NonFiniteScore);
    }
    Ok(DisjointEpisode {
        epsilon_f,
        selected: Some(selected),
        o_score,
        episode_failed: false,
    })
}

impl<'a> RidgeFoldEvaluator<'a> {
    /// The executing-boundary counters for this evaluator's physical fits.
    pub fn counters(&self) -> &crate::search::ExecutionBoundaryCounters {
        &self.counters
    }

    /// Replaces the default ceilings. A fit past them is refused fail-closed.
    pub fn with_caps(mut self, caps: crate::search::ExecutionBoundaryCaps) -> Self {
        self.counters = crate::search::ExecutionBoundaryCounters::new(caps);
        self
    }
}

impl<'a> GpProgramFoldEvaluator<'a> {
    /// The executing-boundary counters for this evaluator's physical fits.
    pub fn counters(&self) -> &crate::search::ExecutionBoundaryCounters {
        &self.counters
    }

    /// Replaces the default ceilings. A fit past them is refused fail-closed.
    pub fn with_caps(mut self, caps: crate::search::ExecutionBoundaryCaps) -> Self {
        self.counters = crate::search::ExecutionBoundaryCounters::new(caps);
        self
    }
}

/// GATE 1. A canonical `search.rs` pipeline genome, fitted with feat-math on the `A` rows and scored
/// on `B` (selection) or `O` (assessment). It takes the production path: `GenomeIr::parse`,
/// `feat_math::fit`, then `predict`. Each physical fit is counted before it runs (GATE 3).
pub struct NativeTeacherFoldEvaluator<'a> {
    features: &'a [Vec<f64>],
    targets: &'a [f64],
    a_rows: &'a [u32],
    b_rows: &'a [u32],
    o_rows: &'a [u32],
    counters: std::sync::Arc<crate::search::ExecutionBoundaryCounters>,
    o_reads: std::cell::Cell<u64>,
}

impl<'a> NativeTeacherFoldEvaluator<'a> {
    /// Checks the partitions are disjoint and in range, and the table is a finite rectangle. The
    /// targets of `O` may be non-finite: they are read only at assessment, and the tests rely on
    /// that.
    pub fn new(
        features: &'a [Vec<f64>],
        targets: &'a [f64],
        a_rows: &'a [u32],
        b_rows: &'a [u32],
        o_rows: &'a [u32],
    ) -> Result<Self, AdapterError> {
        check_disjoint(a_rows, b_rows, o_rows)?;
        let width = features.first().map_or(0, Vec::len);
        if features.is_empty()
            || width == 0
            || features.len() != targets.len()
            || features.iter().any(|row| row.len() != width)
            || features.iter().flatten().any(|v| !v.is_finite())
        {
            return Err(AdapterError::EvaluatorFailed(
                "features must be a finite rectangular table with one target per row".into(),
            ));
        }
        if a_rows.len() < 2 || b_rows.is_empty() || o_rows.is_empty() {
            return Err(AdapterError::TooFewTargets { n: a_rows.len() });
        }
        for &row in a_rows.iter().chain(b_rows).chain(o_rows) {
            if row as usize >= features.len() {
                return Err(AdapterError::EvaluatorFailed(format!(
                    "row {row} is outside the table"
                )));
            }
        }
        for &row in a_rows.iter().chain(b_rows) {
            if !targets[row as usize].is_finite() {
                return Err(AdapterError::NonFiniteTarget);
            }
        }
        Ok(Self {
            features,
            targets,
            a_rows,
            b_rows,
            o_rows,
            counters: std::sync::Arc::new(crate::search::ExecutionBoundaryCounters::new(
                crate::search::ExecutionBoundaryCaps::default(),
            )),
            o_reads: std::cell::Cell::new(0),
        })
    }

    /// The executing-boundary counters for this evaluator's physical fits.
    pub fn counters(&self) -> &crate::search::ExecutionBoundaryCounters {
        &self.counters
    }

    /// Replaces the default ceilings. A fit past them is refused fail-closed.
    pub fn with_caps(mut self, caps: crate::search::ExecutionBoundaryCaps) -> Self {
        self.counters = std::sync::Arc::new(crate::search::ExecutionBoundaryCounters::new(caps));
        self
    }

    /// Shares the caller's counters, so the search and this assessment draw on one quota.
    pub fn with_shared_counters(
        mut self,
        counters: std::sync::Arc<crate::search::ExecutionBoundaryCounters>,
    ) -> Self {
        self.counters = counters;
        self
    }

    /// The one O read allowed. A second call is refused before any fit runs or any O row is read.
    pub fn score_assessment_once(&self, serialized: &str) -> Result<f64, AdapterError> {
        if self.o_reads.get() != 0 {
            return Err(AdapterError::EvaluatorFailed(
                "O was already assessed; a second read is refused".into(),
            ));
        }
        self.score_assessment(serialized)
    }

    /// How many times the `O` rows have been read. Selection never reads them.
    pub fn o_reads(&self) -> u64 {
        self.o_reads.get()
    }

    /// Fits the genome on `A` and returns the RMSE on `rows`. A genome that does not parse is an
    /// error. A fit that fails, panics, or predicts non-finite values scores `NaN`, which the
    /// episode records as a non-finite candidate. A charge past the fit ceiling is an error.
    fn scored(&self, serialized: &str, rows: &[u32]) -> Result<f64, AdapterError> {
        let genome = feat_math::genome::GenomeIr::parse(serialized).map_err(|error| {
            AdapterError::Malformed(format!("native pipeline {serialized:?}: {error}"))
        })?;
        let x_a: Vec<Vec<f64>> = self
            .a_rows
            .iter()
            .map(|&row| self.features[row as usize].clone())
            .collect();
        let y_a: Vec<f64> = self
            .a_rows
            .iter()
            .map(|&row| self.targets[row as usize])
            .collect();
        let x_s: Vec<Vec<f64>> = rows
            .iter()
            .map(|&row| self.features[row as usize].clone())
            .collect();
        let y_s: Vec<f64> = rows.iter().map(|&row| self.targets[row as usize]).collect();
        self.counters
            .charge_physical_fit()
            .map_err(AdapterError::BudgetExhausted)?;
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let model = feat_math::fit(&genome, &x_a, &y_a)?;
            let predictions = model.predict(&x_s)?;
            anyhow::Ok(predictions)
        }));
        let predictions = match outcome {
            Ok(Ok(predictions)) => predictions,
            _ => return Ok(f64::NAN),
        };
        if predictions.len() != y_s.len() || predictions.iter().any(|v| !v.is_finite()) {
            return Ok(f64::NAN);
        }
        Ok(feat_math::metrics::rmse(&y_s, &predictions))
    }
}

impl FoldEvaluator for NativeTeacherFoldEvaluator<'_> {
    fn score_selection(&self, serialized: &str) -> Result<(f64, u64), AdapterError> {
        Ok((self.scored(serialized, self.b_rows)?, 1))
    }

    fn score_assessment(&self, serialized: &str) -> Result<f64, AdapterError> {
        self.o_reads.set(self.o_reads.get() + 1);
        self.scored(serialized, self.o_rows)
    }
}

/// The native candidates for one episode: the bootstrap genome first, then feat-math's random,
/// mutated and crossed canonical pipelines, drawn deterministically from `seed`. Duplicates are
/// skipped. Drawing stops at `count` candidates or after `64 * count` draws.
pub fn propose_native_teacher_candidates(seed: u64, count: usize) -> Vec<String> {
    let max = crate::search::MAX_TRANSFORMS;
    let mut out: Vec<String> = Vec::new();
    if count == 0 {
        return out;
    }
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let bootstrap = crate::search::BOOTSTRAP_GENOME.to_string();
    seen.insert(bootstrap.clone());
    out.push(bootstrap);
    let mut draw: u64 = 0;
    while out.len() < count && draw < (count as u64) * 64 {
        let s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(draw);
        let candidate = match draw % 3 {
            0 => feat_math::evolve::random_canonical_pipeline(s, max),
            1 => {
                let parent = &out[(s as usize) % out.len()];
                feat_math::evolve::mutate_canonical_pipeline(parent, s, max)
                    .unwrap_or_else(|_| feat_math::evolve::random_canonical_pipeline(s, max))
            }
            _ => {
                let left = &out[(s as usize) % out.len()];
                let right = &out[((s >> 17) as usize) % out.len()];
                feat_math::evolve::crossover_canonical_pipelines(left, right, s, max)
                    .unwrap_or_else(|_| feat_math::evolve::random_canonical_pipeline(s, max))
            }
        };
        if seen.insert(candidate.clone()) {
            out.push(candidate);
        }
        draw += 1;
    }
    out
}

/// GATE 1 episode. Each native candidate is one slot, charged before it is parsed or fitted. A
/// candidate that does not parse closes its slot as malformed, and a duplicate closes as a
/// duplicate. Selection scores `B` only. `O` is scored once, after the selection is frozen. The
/// fallback applies only when nothing was credited.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_native_teacher_episode(
    a_rows: &[u32],
    b_rows: &[u32],
    o_rows: &[u32],
    y_a: &[f64],
    candidates: &[String],
    journal: &mut Journal,
    cfg: &SeedBudgetConfig,
    evaluator: &NativeTeacherFoldEvaluator,
    o_fallback: f64,
) -> Result<DisjointEpisode, AdapterError> {
    check_disjoint(a_rows, b_rows, o_rows)?;
    let epsilon_f = compute_epsilon_f(y_a)?;
    validate_seed_and_bootstrap_budget(
        &cfg.route,
        cfg.initial_population,
        cfg.seed_genomes,
        cfg.low_budget_cap,
    )?;
    let mut best: Option<(String, f64)> = None;
    let mut credited = 0usize;
    for (index, raw) in candidates.iter().enumerate() {
        journal.begin_slot(&format!("slot-{index}"))?;
        if feat_math::genome::GenomeIr::parse(raw).is_err() {
            journal.close_slot(raw, SlotOutcome::Malformed, 0)?;
            continue;
        }
        let canonical = raw.clone();
        if journal.is_seen(&canonical) {
            journal.close_slot(&canonical, SlotOutcome::Duplicate, 0)?;
            continue;
        }
        journal.charge_fits(cfg.fits_per_candidate)?;
        let rmse_b = evaluator.score_selection(&canonical)?.0;
        if !rmse_b.is_finite() {
            journal.close_slot(&canonical, SlotOutcome::NonFinite, 1)?;
            continue;
        }
        journal.close_slot(&canonical, SlotOutcome::Ok, 1)?;
        credited += 1;
        if best.as_ref().is_none_or(|(_, b)| rmse_b < *b) {
            best = Some((canonical, rmse_b));
        }
    }
    if credited == 0 {
        if !o_fallback.is_finite() {
            return Err(AdapterError::NonFiniteScore);
        }
        return Ok(DisjointEpisode {
            epsilon_f,
            selected: None,
            o_score: o_fallback,
            episode_failed: true,
        });
    }
    let (selected, _) = best.ok_or(AdapterError::NoEligibleCandidate)?;
    let o_score = evaluator.score_assessment(&selected)?;
    if !o_score.is_finite() {
        return Err(AdapterError::NonFiniteScore);
    }
    Ok(DisjointEpisode {
        epsilon_f,
        selected: Some(selected),
        o_score,
        episode_failed: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::collections::BTreeMap;

    fn col(token: &str) -> ColumnMoments {
        ColumnMoments {
            token: token.into(),
            mean: 0.5,
            std: 1.25,
            min: -2.0,
            max: 3.0,
        }
    }

    fn cand(family: &str, confirmation: bool, score: f64) -> Candidate {
        Candidate {
            family: family.into(),
            confirmation,
            score,
        }
    }

    /// Test evaluator. Selection scores are fixed per canonical edit. The assessment count is recorded so
    /// that the tests can prove the O-side is scored at most once, after selection.
    struct MockEval {
        selection: BTreeMap<String, (f64, u64)>,
        assess_value: f64,
        assess_calls: RefCell<Vec<String>>,
    }

    impl FoldEvaluator for MockEval {
        fn score_selection(&self, serialized: &str) -> Result<(f64, u64), AdapterError> {
            self.selection
                .get(serialized)
                .copied()
                .ok_or_else(|| AdapterError::EvaluatorFailed(format!("no score for {serialized}")))
        }
        fn score_assessment(&self, serialized: &str) -> Result<f64, AdapterError> {
            self.assess_calls.borrow_mut().push(serialized.to_string());
            Ok(self.assess_value)
        }
    }

    fn cfg() -> SeedBudgetConfig {
        SeedBudgetConfig {
            route: "low64".into(),
            initial_population: 64,
            seed_genomes: 4,
            low_budget_cap: 64,
            fits_per_candidate: 2,
        }
    }

    // ---- card and retrieval (V1 behaviour kept) ----

    #[test]
    fn card_renders_anonymous_tokens_and_is_deterministic() {
        let a = render_card("fam_07", &[col("col_00"), col("col_01")]).unwrap();
        let b = render_card("fam_07", &[col("col_00"), col("col_01")]).unwrap();
        assert_eq!(a, b);
        assert!(a.contains("col_00 mean=0.5000"));
    }

    #[test]
    fn card_with_a_forbidden_name_is_refused_not_edited() {
        assert_eq!(
            render_card("fam_07", &[col("sberbank_col")]).unwrap_err(),
            AdapterError::Forbidden("sberbank".into())
        );
        assert!(render_card("golden42_slot", &[col("col_00")]).is_err());
    }

    #[test]
    fn retrieval_refuses_same_family_and_drops_confirmation_families() {
        let pool = vec![
            cand("fam_01", false, 0.9),
            cand("fam_02", true, 0.99),
            cand("fam_03", false, 0.5),
        ];
        assert_eq!(
            retrieve("fam_01", &pool, 2).unwrap_err(),
            AdapterError::SameFamily
        );
        let names: Vec<String> = retrieve("fam_09", &pool, 5)
            .unwrap()
            .into_iter()
            .map(|c| c.family)
            .collect();
        assert_eq!(names, vec!["fam_01", "fam_03"]);
    }

    #[test]
    fn retrieval_is_deterministic_on_score_ties() {
        let pool = vec![
            cand("fam_05", false, 0.7),
            cand("fam_02", false, 0.7),
            cand("fam_04", false, 0.7),
        ];
        let got = retrieve("fam_99", &pool, 2).unwrap();
        assert_eq!(
            (got[0].family.as_str(), got[1].family.as_str()),
            ("fam_02", "fam_04")
        );
    }

    // ---- operator whitelist and grammar caps (item 1) ----

    #[test]
    fn whitelist_is_the_24_prompt_operators() {
        assert_eq!(HWM_OPERATORS.len(), 24);
        assert!(HWM_OPERATORS.contains(&"pipe") && HWM_OPERATORS.contains(&"ridge"));
        assert!(!HWM_OPERATORS.contains(&"pca") && !HWM_OPERATORS.contains(&"logit_ridge"));
        assert!(matches!(
            parse_edit("WrapCombinator pca").unwrap_err(),
            AdapterError::UnknownOperator(_)
        ));
        assert!(parse_edit("WrapCombinator rank").is_ok());
    }

    #[test]
    fn typed_edits_roundtrip_through_canonical_text() {
        for line in [
            "ReplaceSubtree 0,2 (pipe X (ridge))",
            "WrapCombinator rank",
            "TuneHyperparam alpha 0.25",
        ] {
            let parsed = parse_edit(line).unwrap();
            assert_eq!(
                parse_edit(&serialize_edit(&parsed)).unwrap(),
                parsed,
                "roundtrip failed for {line}"
            );
        }
    }

    #[test]
    fn expressions_with_unknown_heads_or_bad_structure_are_refused() {
        assert!(matches!(
            parse_edit("ReplaceSubtree 0 (gbm X)").unwrap_err(),
            AdapterError::UnknownOperator(_)
        ));
        assert!(matches!(
            parse_edit("ReplaceSubtree 0 (ridge").unwrap_err(),
            AdapterError::Malformed(_)
        ));
        assert!(matches!(
            parse_edit("ReplaceSubtree 0 (ridge))").unwrap_err(),
            AdapterError::Malformed(_)
        ));
        assert!(matches!(
            parse_edit("ReplaceSubtree 0,x (ridge)").unwrap_err(),
            AdapterError::Malformed(_)
        ));
    }

    #[test]
    fn grammar_caps_are_enforced() {
        let deep = format!("(pipe X {})", "(ridge X) ".repeat(GRAMMAR_PIPE_STAGES + 1));
        assert!(
            matches!(validate_expression(&deep).unwrap_err(), AdapterError::Malformed(m) if m.contains("stages"))
        );
        let many_nodes = format!("(pipe X {})", "(rank X) ".repeat(GRAMMAR_SYM_NODES));
        assert!(
            matches!(validate_expression(&many_nodes).unwrap_err(), AdapterError::Malformed(m) if m.contains("symbol"))
        );
        let long = format!("(ridge {})", "X ".repeat(GRAMMAR_MAX_BYTES));
        assert!(
            matches!(validate_expression(&long).unwrap_err(), AdapterError::Malformed(m) if m.contains("bytes"))
        );
        assert!(validate_expression("(pipe X (standardize X) (ridge X))").is_ok());
    }

    #[test]
    fn protected_core_and_unknown_operators_fail_closed() {
        assert!(matches!(
            parse_edit("ProtectedCoreEdit anything").unwrap_err(),
            AdapterError::ProtectedCore(_)
        ));
        assert!(matches!(
            parse_edit("DeleteEverything now").unwrap_err(),
            AdapterError::UnknownOperator(_)
        ));
    }

    // ---- epsilon (item 2) ----

    #[test]
    fn epsilon_uses_fitting_targets_only_and_refuses_bad_targets() {
        let y: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let eps = compute_epsilon_f(&y).unwrap();
        let mean = 4.5;
        let var: f64 = y.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / 9.0;
        assert!((eps - 1e-6 * var.sqrt()).abs() < 1e-15);
        assert_eq!(
            compute_epsilon_f(&y[..7]).unwrap_err(),
            AdapterError::TooFewTargets { n: 7 }
        );
        assert_eq!(
            compute_epsilon_f(&[1.0, f64::NAN, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap_err(),
            AdapterError::NonFiniteTarget
        );
        assert_eq!(
            compute_epsilon_f(&[2.0; 8]).unwrap_err(),
            AdapterError::DegenerateTarget
        );
    }

    #[test]
    fn paired_log_ratio_matches_definition_and_refuses_bad_inputs() {
        let v = paired_log_rmse_ratio(2.0, 1.0, 0.5).unwrap();
        assert!((v - (2.5f64 / 1.5f64).ln()).abs() < 1e-12);
        assert_eq!(
            paired_log_rmse_ratio(f64::NAN, 1.0, 0.5).unwrap_err(),
            AdapterError::NonFiniteScore
        );
        assert_eq!(
            paired_log_rmse_ratio(1.0, 1.0, 0.0).unwrap_err(),
            AdapterError::NonFiniteScore
        );
    }

    // ---- seed and bootstrap budget (item 4) ----

    #[test]
    fn population_80_under_low64_is_refused_explicitly() {
        assert_eq!(
            validate_seed_and_bootstrap_budget("low64", 80, 0, 64).unwrap_err(),
            AdapterError::BootstrapPopulationExceedsCap {
                requested: 80,
                cap: 64
            }
        );
        assert!(validate_seed_and_bootstrap_budget("medium256", 80, 0, 256).is_ok());
    }

    #[test]
    fn seed_genomes_above_the_cap_of_four_are_refused() {
        assert_eq!(
            validate_seed_and_bootstrap_budget("medium256", 80, 5, 256).unwrap_err(),
            AdapterError::SeedCapExceeded {
                requested: 5,
                cap: 4
            }
        );
        assert!(validate_seed_and_bootstrap_budget("medium256", 80, 4, 256).is_ok());
    }

    // ---- journal (V1 behaviour kept) ----

    #[test]
    fn journal_charges_before_attempt_and_non_ok_slots_consume_budget() {
        let mut j = Journal::new(3, 100);
        j.begin_slot("s1").unwrap();
        j.close_slot("canon-a", SlotOutcome::Ok, 4).unwrap();
        j.begin_slot("s2").unwrap();
        j.close_slot("canon-a", SlotOutcome::Ok, 4).unwrap();
        j.begin_slot("s3").unwrap();
        j.close_slot("canon-b", SlotOutcome::Timeout, 1).unwrap();
        assert_eq!(
            j.begin_slot("s4").unwrap_err(),
            AdapterError::BudgetExhausted("candidate slots")
        );
        let t = j.totals();
        assert_eq!(
            (t["slots_charged"], t["credited"], t["fallbacks"]),
            (3, 1, 2)
        );
        assert_eq!(j.entries()[1].outcome, SlotOutcome::Duplicate);
    }

    #[test]
    fn physical_fit_cap_and_slot_nesting_are_enforced() {
        let mut j = Journal::new(10, 5);
        j.begin_slot("s1").unwrap();
        assert!(j.begin_slot("s2").is_err());
        j.charge_fits(5).unwrap();
        assert_eq!(
            j.charge_fits(1).unwrap_err(),
            AdapterError::BudgetExhausted("physical fold fits")
        );
    }

    #[test]
    fn fitting_selection_and_assessment_rows_must_be_disjoint() {
        assert!(check_disjoint(&[0, 1], &[2, 3], &[4, 5]).is_ok());
        assert_eq!(
            check_disjoint(&[0, 1], &[1, 3], &[4]).unwrap_err(),
            AdapterError::OverlappingRows("A and B overlap")
        );
        assert_eq!(
            check_disjoint(&[0], &[2], &[0]).unwrap_err(),
            AdapterError::OverlappingRows("A and O overlap")
        );
        assert_eq!(
            check_disjoint(&[0], &[2], &[2]).unwrap_err(),
            AdapterError::OverlappingRows("B and O overlap")
        );
    }

    // ---- fake proposer driver (item 3) and disjoint episode (item 5) ----

    #[test]
    fn proposer_slots_are_charged_before_evaluation_and_bad_slots_are_closed_without_fits() {
        let script = [
            "ReplaceSubtree 0 (ridge X)",
            "ProtectedCoreEdit nope",
            "ReplaceSubtree 0 (gbm X)",
            "ReplaceSubtree 0 (ridge X)",
            "WrapCombinator rank",
        ];
        let mut sel = BTreeMap::new();
        sel.insert("ReplaceSubtree 0 (ridge X)".to_string(), (0.9, 2));
        sel.insert("WrapCombinator rank".to_string(), (0.4, 2));
        let eval = MockEval {
            selection: sel,
            assess_value: 0.3,
            assess_calls: RefCell::new(vec![]),
        };
        let mut j = Journal::new(10, 100);
        let res = run_fake_proposer_episode(&script, &mut j, &cfg(), &eval).unwrap();
        assert_eq!(res.credited, 2);
        assert_eq!(res.best, Some(("WrapCombinator rank".to_string(), 0.4)));
        let outcomes: Vec<SlotOutcome> = j.entries().iter().map(|e| e.outcome).collect();
        assert_eq!(
            outcomes,
            vec![
                SlotOutcome::Ok,
                SlotOutcome::ProtectedCore,
                SlotOutcome::Malformed,
                SlotOutcome::Duplicate,
                SlotOutcome::Ok
            ]
        );
        // Two valid novel edits at 2 fits each. The malformed, protected and duplicate slots charge none.
        assert_eq!(j.totals()["fits_charged"], 4);
        assert!(
            eval.assess_calls.borrow().is_empty(),
            "the proposer loop must not score O"
        );
    }

    #[test]
    fn episode_selects_on_b_then_scores_o_once_and_leaves_o_out_of_selection() {
        let y_a: Vec<f64> = (0..12).map(|i| (i as f64).sin()).collect();
        let mut sel = BTreeMap::new();
        sel.insert("WrapCombinator rank".to_string(), (0.4, 2));
        sel.insert("WrapCombinator ridge".to_string(), (0.6, 2));
        let eval = MockEval {
            selection: sel,
            assess_value: 0.35,
            assess_calls: RefCell::new(vec![]),
        };
        let mut j = Journal::new(10, 100);
        let out = evaluate_episode_on_disjoint_folds(
            &[0, 1, 2],
            &[3, 4, 5],
            &[6, 7, 8],
            &y_a,
            &["WrapCombinator ridge", "WrapCombinator rank"],
            &mut j,
            &cfg(),
            &eval,
            9.9,
        )
        .unwrap();
        assert_eq!(out.selected.as_deref(), Some("WrapCombinator rank"));
        assert_eq!(out.o_score, 0.35);
        assert!(!out.episode_failed);
        assert_eq!(
            eval.assess_calls.borrow().as_slice(),
            ["WrapCombinator rank"]
        );
        assert!(out.epsilon_f > 0.0);
    }

    /// A5 fixture: 40 rows, 3 features, targets with a known linear signal and a small nonlinear term.
    fn ridge_table() -> (Vec<Vec<f64>>, Vec<f64>) {
        let features: Vec<Vec<f64>> = (0..40)
            .map(|i| {
                let t = i as f64;
                vec![(t * 0.37).sin(), (t * 0.11).cos(), t / 40.0]
            })
            .collect();
        let targets = features
            .iter()
            .map(|x| 2.0 * x[0] - x[2] + 0.1 * (7.0 * x[1]).sin())
            .collect();
        (features, targets)
    }

    fn ridge_cfg() -> SeedBudgetConfig {
        SeedBudgetConfig {
            route: "low64".into(),
            initial_population: 64,
            seed_genomes: 4,
            low_budget_cap: 64,
            fits_per_candidate: 1,
        }
    }

    fn ridge_partitions() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
        ((0..20).collect(), (20..30).collect(), (30..40).collect())
    }

    /// The fit is strictly on `A`: moving the `B` and `O` targets leaves the coefficients unchanged.
    /// On an exactly linear target the solver recovers the generating coefficients.
    #[test]
    fn ridge_fits_its_parameters_on_the_a_rows_alone() {
        let (features, _) = ridge_table();
        let targets: Vec<f64> = features.iter().map(|x| 2.0 * x[0] - x[2] + 0.5).collect();
        let (a, b, o) = ridge_partitions();
        let fitted = RidgeFoldEvaluator::new(&features, &targets, &a, &b, &o, 1e-3)
            .unwrap()
            .fit_on_a(&[0, 1, 2])
            .unwrap();
        let mut moved = targets.clone();
        for &row in b.iter().chain(&o) {
            moved[row as usize] = 1.0e6;
        }
        let refit = RidgeFoldEvaluator::new(&features, &moved, &a, &b, &o, 1e-3)
            .unwrap()
            .fit_on_a(&[0, 1, 2])
            .unwrap();
        assert_eq!(
            fitted, refit,
            "B or O targets moved the fitting-side coefficients"
        );
        let expected = [0.5, 2.0, 0.0, -1.0];
        // The solver is exact: with the penalty at 1e-9 the generating coefficients come back to 1e-6.
        // The 1e-3 fit above is a penalised fit, so its coefficients sit within the ridge bias.
        let exact = RidgeFoldEvaluator::new(&features, &targets, &a, &b, &o, 1e-9)
            .unwrap()
            .fit_on_a(&[0, 1, 2])
            .unwrap();
        for (got, want) in exact.iter().zip(expected) {
            assert!(
                (got - want).abs() < 1e-6,
                "exact fit {exact:?} != {expected:?}"
            );
        }
        for (got, want) in fitted.iter().zip(expected) {
            assert!(
                (got - want).abs() < 0.2,
                "penalised fit {fitted:?} strays from {expected:?}"
            );
        }
    }

    /// Selection runs with the `O` targets poisoned to NaN. It must still credit both candidates
    /// and must never read a row of `O`. Only the closing assessment reads `O`, once.
    #[test]
    fn selection_never_reads_o_and_the_assessment_reads_it_once() {
        let (features, clean) = ridge_table();
        let (a, b, o) = ridge_partitions();
        let script = ["WrapCombinator rank", "WrapCombinator ridge"];
        let mut poisoned = clean.clone();
        for &row in &o {
            poisoned[row as usize] = f64::NAN;
        }
        let eval = RidgeFoldEvaluator::new(&features, &poisoned, &a, &b, &o, 1e-3).unwrap();
        let mut journal = Journal::new(10, 100);
        let selection =
            run_fake_proposer_episode(&script, &mut journal, &ridge_cfg(), &eval).unwrap();
        assert_eq!(selection.credited, 2);
        assert_eq!(eval.o_reads(), 0, "selection read the O rows");

        let y_a: Vec<f64> = a.iter().map(|&row| clean[row as usize]).collect();
        let eval = RidgeFoldEvaluator::new(&features, &clean, &a, &b, &o, 1e-3).unwrap();
        let mut journal = Journal::new(10, 100);
        let episode = evaluate_episode_on_disjoint_folds(
            &a,
            &b,
            &o,
            &y_a,
            &script,
            &mut journal,
            &ridge_cfg(),
            &eval,
            9.9,
        )
        .unwrap();
        assert!(episode.selected.is_some());
        assert!(episode.o_score.is_finite() && !episode.episode_failed);
        assert_eq!(
            eval.o_reads(),
            1,
            "O must be scored exactly once, after selection"
        );
    }

    /// Overlapping partitions and non-finite lambda are refused before any fit runs.
    #[test]
    fn ridge_refuses_overlapping_partitions_and_a_bad_lambda() {
        let (features, targets) = ridge_table();
        let (a, b, _) = ridge_partitions();
        let overlapping_o = vec![a[0], 35];
        assert!(
            RidgeFoldEvaluator::new(&features, &targets, &a, &b, &overlapping_o, 1e-3).is_err()
        );
        let (a, b, o) = ridge_partitions();
        assert!(RidgeFoldEvaluator::new(&features, &targets, &a, &b, &o, 0.0).is_err());
        assert!(RidgeFoldEvaluator::new(&features, &targets, &a, &b, &o, f64::NAN).is_err());
    }

    /// Targets are the true product of the first two features, so `mul(f0,f1)` is the right program.
    fn product_table() -> (Vec<Vec<f64>>, Vec<f64>) {
        let (features, _) = ridge_table();
        let targets = features.iter().map(|x| x[0] * x[1]).collect();
        (features, targets)
    }

    /// Two distinct GP programs produce two distinct feature columns and two distinct fits. The
    /// columns come from the programs themselves, not from a hash of the text.
    #[test]
    fn distinct_gp_programs_build_distinct_columns_and_distinct_fits() {
        let (features, targets) = product_table();
        let (a, b, o) = ridge_partitions();
        let add = GpProgram::parse("add(f0,f1)").unwrap();
        let mul = GpProgram::parse("mul(f0,f1)").unwrap();
        let column =
            |p: &GpProgram| -> Vec<f64> { features.iter().map(|row| p.eval(row)).collect() };
        assert_ne!(
            column(&add),
            column(&mul),
            "the two programs built the same column"
        );
        let eval = GpProgramFoldEvaluator::new(&features, &targets, &a, &b, &o, 1e-3).unwrap();
        let add_b = eval.score_selection("add(f0,f1)").unwrap().0;
        let mul_b = eval.score_selection("mul(f0,f1)").unwrap().0;
        let add_o = eval.score_assessment("add(f0,f1)").unwrap();
        let mul_o = eval.score_assessment("mul(f0,f1)").unwrap();
        assert!(
            mul_b < 0.01 && mul_o < 0.01,
            "the true product must fit: B {mul_b} O {mul_o}"
        );
        assert!(
            add_b > 10.0 * mul_b,
            "B fits did not separate: add {add_b} mul {mul_b}"
        );
        assert!(
            add_o > 10.0 * mul_o,
            "O fits did not separate: add {add_o} mul {mul_o}"
        );
    }

    /// Selection with `O` poisoned still credits both programs and reads `O` only once, at the
    /// assessment. With clean targets the episode selects the true product and scores `O` once.
    #[test]
    fn gp_selection_reads_b_only_and_o_is_scored_once_after_selection() {
        let (features, clean) = product_table();
        let (a, b, o) = ridge_partitions();
        let programs = ["add(f0,f1)", "mul(f0,f1)"];
        let y_a: Vec<f64> = a.iter().map(|&row| clean[row as usize]).collect();
        let mut poisoned = clean.clone();
        for &row in &o {
            poisoned[row as usize] = f64::NAN;
        }
        let eval = GpProgramFoldEvaluator::new(&features, &poisoned, &a, &b, &o, 1e-3).unwrap();
        let mut journal = Journal::new(10, 100);
        let refused = evaluate_program_episode(
            &a,
            &b,
            &o,
            &y_a,
            &programs,
            &mut journal,
            &ridge_cfg(),
            &eval,
            9.9,
        );
        assert!(matches!(refused, Err(AdapterError::NonFiniteScore)));
        assert_eq!(
            journal
                .entries()
                .iter()
                .filter(|e| matches!(e.outcome, SlotOutcome::Ok))
                .count(),
            2,
            "selection must credit both programs before O is read"
        );
        assert_eq!(
            eval.o_reads(),
            1,
            "O was read by selection, or more than once"
        );

        let eval = GpProgramFoldEvaluator::new(&features, &clean, &a, &b, &o, 1e-3).unwrap();
        let mut journal = Journal::new(10, 100);
        let episode = evaluate_program_episode(
            &a,
            &b,
            &o,
            &y_a,
            &programs,
            &mut journal,
            &ridge_cfg(),
            &eval,
            9.9,
        )
        .unwrap();
        assert_eq!(episode.selected.as_deref(), Some("mul(f0,f1)"));
        assert!(!episode.episode_failed);
        assert_eq!(eval.o_reads(), 1);
    }

    /// The program language is closed: other operators, malformed text, out-of-range features,
    /// and programs past the depth or node caps are refused. An out-of-range program closes its
    /// slot as malformed and is never evaluated.
    #[test]
    fn gp_parser_refuses_anything_outside_the_program_language() {
        for bad in [
            "add(f0)",
            "f01",
            "f",
            "foo(f0,f1)",
            "add(f0,f1)x",
            "add(f0 ,f1)",
            "add(f0,f1",
        ] {
            assert!(GpProgram::parse(bad).is_err(), "accepted {bad:?}");
        }
        fn chain(depth_wraps: usize) -> String {
            (0..depth_wraps).fold("f0".to_string(), |acc, _| format!("add({acc},f0)"))
        }
        fn balanced(leaves: usize) -> String {
            if leaves == 1 {
                "f0".to_string()
            } else {
                let left = leaves / 2;
                format!("add({},{})", balanced(left), balanced(leaves - left))
            }
        }
        assert!(
            GpProgram::parse(&chain(GP_MAX_DEPTH - 1)).is_ok(),
            "depth cap is too tight"
        );
        assert!(
            GpProgram::parse(&chain(GP_MAX_DEPTH)).is_err(),
            "depth cap not enforced"
        );
        // A balanced tree of n leaves has 2n-1 nodes: 32 leaves is 63 nodes, 33 leaves is 65.
        assert!(
            GpProgram::parse(&balanced(GP_MAX_NODES / 2)).is_ok(),
            "node cap too tight"
        );
        assert!(
            GpProgram::parse(&balanced(GP_MAX_NODES / 2 + 1)).is_err(),
            "node cap not enforced"
        );

        let (features, targets) = product_table();
        let (a, b, o) = ridge_partitions();
        let y_a: Vec<f64> = a.iter().map(|&row| targets[row as usize]).collect();
        let eval = GpProgramFoldEvaluator::new(&features, &targets, &a, &b, &o, 1e-3).unwrap();
        let mut journal = Journal::new(10, 100);
        let episode = evaluate_program_episode(
            &a,
            &b,
            &o,
            &y_a,
            &["add(f0,f9)", "mul(f0,f1)"],
            &mut journal,
            &ridge_cfg(),
            &eval,
            9.9,
        )
        .unwrap();
        assert_eq!(episode.selected.as_deref(), Some("mul(f0,f1)"));
        assert!(matches!(
            journal.entries()[0].outcome,
            SlotOutcome::Malformed
        ));
        assert_eq!(eval.o_reads(), 1);
    }

    /// GATE 1. The native teacher fits real feat-math canonical pipelines on `A`, selects on `B`, and
    /// never reads `O` during selection. `O` is poisoned to NaN, so a read would show.
    #[test]
    fn native_teacher_fits_canonical_pipelines_on_a_and_selects_on_b_with_o_untouched() {
        let (features, clean) = ridge_table();
        let (a, b, o) = ridge_partitions();
        let mut poisoned = clean.clone();
        for &row in &o {
            poisoned[row as usize] = f64::NAN;
        }
        let eval = NativeTeacherFoldEvaluator::new(&features, &poisoned, &a, &b, &o).unwrap();
        for genome in ["(pipe X (ridge))", "(pipe X (standardize) (ridge))"] {
            let (rmse_b, fits) = eval.score_selection(genome).unwrap();
            assert!(rmse_b.is_finite(), "{genome}: B score {rmse_b}");
            assert_eq!(fits, 1);
        }
        assert_eq!(eval.o_reads(), 0, "selection read the O rows");
        assert!(eval.score_assessment("(pipe X (ridge))").unwrap().is_nan());
        assert_eq!(eval.o_reads(), 1, "O must be read once, at assessment");
        assert_eq!(eval.counters().physical_fits(), 3);
    }

    /// GATE 1. The native episode runs proposals end to end on `A/B/O` and selects a pipeline that beats
    /// the constant fallback on scaled, nonlinear data.
    #[test]
    fn native_teacher_episode_beats_the_fallback_on_nonlinear_scaled_data() {
        let n = 60;
        let features: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let t = i as f64;
                vec![1000.0 * (t * 0.37).sin(), (t * 0.11).cos(), t / n as f64]
            })
            .collect();
        let targets: Vec<f64> = features
            .iter()
            .map(|x| 0.003 * x[0] * x[0] + x[1] - 2.0 * x[2])
            .collect();
        let a: Vec<u32> = (0..30).collect();
        let b: Vec<u32> = (30..45).collect();
        let o: Vec<u32> = (45..60).collect();
        let y_a: Vec<f64> = a.iter().map(|&r| targets[r as usize]).collect();
        let mean_a = y_a.iter().sum::<f64>() / y_a.len() as f64;
        let o_fallback = (o
            .iter()
            .map(|&r| (targets[r as usize] - mean_a).powi(2))
            .sum::<f64>()
            / o.len() as f64)
            .sqrt();
        let candidates = propose_native_teacher_candidates(7, 8);
        let eval = NativeTeacherFoldEvaluator::new(&features, &targets, &a, &b, &o).unwrap();
        let mut journal = Journal::new(20, 200);
        let episode = evaluate_native_teacher_episode(
            &a,
            &b,
            &o,
            &y_a,
            &candidates,
            &mut journal,
            &ridge_cfg(),
            &eval,
            o_fallback,
        )
        .unwrap();
        assert!(!episode.episode_failed, "no native candidate was credited");
        let selected = episode.selected.expect("a selection");
        assert!(
            episode.o_score < o_fallback,
            "{selected} did not beat the fallback: {} vs {o_fallback}",
            episode.o_score
        );
        assert_eq!(eval.o_reads(), 1);
    }

    /// GATE 3. The physical-fit counter refuses the fit past its ceiling before the fit runs, and the
    /// ridge and GP evaluators count their fits the same way.
    #[test]
    fn executing_boundary_counters_refuse_a_fit_past_its_cap_before_it_runs() {
        use crate::search::ExecutionBoundaryCaps;
        let caps = ExecutionBoundaryCaps {
            max_physical_fits: 2,
            max_model_calls: 2,
            max_reasoning_tokens_per_call: Some(8_192),
            max_generated_tokens_total: None,
        };
        let (features, targets) = ridge_table();
        let (a, b, o) = ridge_partitions();
        let native = NativeTeacherFoldEvaluator::new(&features, &targets, &a, &b, &o)
            .unwrap()
            .with_caps(caps);
        native.score_selection("(pipe X (ridge))").unwrap();
        native.score_selection("(pipe X (ridge))").unwrap();
        let refused = native.score_selection("(pipe X (ridge))");
        assert!(
            matches!(
                refused,
                Err(AdapterError::BudgetExhausted("physical fold fits"))
            ),
            "{refused:?}"
        );
        assert_eq!(
            native.counters().physical_fits(),
            2,
            "a refused fit was counted as run"
        );

        let ridge = RidgeFoldEvaluator::new(&features, &targets, &a, &b, &o, 1e-3)
            .unwrap()
            .with_caps(caps);
        ridge.score_selection("ridge").unwrap();
        ridge.score_selection("ridge").unwrap();
        assert!(matches!(
            ridge.score_selection("ridge"),
            Err(AdapterError::BudgetExhausted("physical fold fits"))
        ));

        let gp = GpProgramFoldEvaluator::new(&features, &targets, &a, &b, &o, 1e-3)
            .unwrap()
            .with_caps(caps);
        gp.score_selection("add(f0,f1)").unwrap();
        gp.score_selection("add(f0,f1)").unwrap();
        assert!(matches!(
            gp.score_selection("add(f0,f1)"),
            Err(AdapterError::BudgetExhausted("physical fold fits"))
        ));
    }

    /// GATE 3. The model-call boundary refuses a call past the call ceiling, and a call whose reasoning
    /// tokens exceed the per-call ceiling. A refused call is not counted.
    #[test]
    fn model_call_boundary_refuses_past_the_call_and_per_call_token_ceilings() {
        use crate::search::{ExecutionBoundaryCaps, ExecutionBoundaryCounters};
        let counters = ExecutionBoundaryCounters::new(ExecutionBoundaryCaps {
            max_physical_fits: 20_000,
            max_model_calls: 2,
            max_reasoning_tokens_per_call: Some(8_192),
            max_generated_tokens_total: None,
        });
        assert_eq!(counters.record_model_call(100), Ok(()));
        assert_eq!(counters.record_model_call(8_192), Ok(()));
        assert_eq!(counters.record_model_call(1), Err("model calls"));
        assert_eq!(counters.model_calls(), 2);
        assert_eq!(counters.reasoning_tokens_total(), 8_292);
        assert_eq!(counters.max_call_tokens_observed(), 8_192);

        let per_call = ExecutionBoundaryCounters::new(ExecutionBoundaryCaps::default());
        assert_eq!(
            per_call.record_model_call(8_193),
            Err("reasoning tokens per call")
        );
        assert_eq!(per_call.model_calls(), 0, "a refused call was counted");
        assert_eq!(per_call.reasoning_tokens_total(), 0);
    }

    /// The candidate proposer: the bootstrap first, every candidate a parseable canonical pipeline, no
    /// duplicates, and deterministic from the seed.
    #[test]
    fn native_candidates_are_parseable_unique_and_deterministic() {
        let first = propose_native_teacher_candidates(11, 10);
        assert_eq!(first, propose_native_teacher_candidates(11, 10));
        assert_eq!(first[0], crate::search::BOOTSTRAP_GENOME);
        let unique: BTreeSet<&String> = first.iter().collect();
        assert_eq!(unique.len(), first.len());
        for candidate in &first {
            assert!(
                feat_math::genome::GenomeIr::parse(candidate).is_ok(),
                "unparseable proposal {candidate:?}"
            );
        }
        assert!(first.len() > 1, "the proposer produced no variation");
    }

    #[test]
    fn episode_with_no_credited_candidate_uses_the_preregistered_fallback_once() {
        let y_a: Vec<f64> = (0..12).map(|i| (i as f64).sin()).collect();
        let eval = MockEval {
            selection: BTreeMap::new(),
            assess_value: 0.0,
            assess_calls: RefCell::new(vec![]),
        };
        let mut j = Journal::new(10, 100);
        let out = evaluate_episode_on_disjoint_folds(
            &[0],
            &[1],
            &[2],
            &y_a,
            &["ProtectedCoreEdit x", "garbage"],
            &mut j,
            &cfg(),
            &eval,
            0.77,
        )
        .unwrap();
        assert!(out.episode_failed);
        assert_eq!(out.selected, None);
        assert_eq!(out.o_score, 0.77);
        assert!(
            eval.assess_calls.borrow().is_empty(),
            "O must not be scored when nothing was selected"
        );
    }

    #[test]
    fn overlapping_rows_are_refused_before_any_slot_is_charged() {
        let y_a: Vec<f64> = (0..12).map(|i| i as f64).collect();
        let eval = MockEval {
            selection: BTreeMap::new(),
            assess_value: 0.0,
            assess_calls: RefCell::new(vec![]),
        };
        let mut j = Journal::new(10, 100);
        assert!(evaluate_episode_on_disjoint_folds(
            &[0, 1],
            &[1],
            &[2],
            &y_a,
            &["WrapCombinator rank"],
            &mut j,
            &cfg(),
            &eval,
            1.0
        )
        .is_err());
        assert_eq!(j.totals()["slots_charged"], 0);
    }

    #[test]
    fn a_nonfinite_selection_score_closes_its_slot_as_nonfinite() {
        let mut sel = BTreeMap::new();
        sel.insert("WrapCombinator rank".to_string(), (f64::NAN, 2));
        let eval = MockEval {
            selection: sel,
            assess_value: 0.0,
            assess_calls: RefCell::new(vec![]),
        };
        let mut j = Journal::new(10, 100);
        let res =
            run_fake_proposer_episode(&["WrapCombinator rank"], &mut j, &cfg(), &eval).unwrap();
        assert_eq!(res.credited, 0);
        assert_eq!(j.entries()[0].outcome, SlotOutcome::NonFinite);
    }
}
