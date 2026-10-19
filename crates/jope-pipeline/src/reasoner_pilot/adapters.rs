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
