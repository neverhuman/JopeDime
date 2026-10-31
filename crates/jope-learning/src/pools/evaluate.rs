//! Parent-grouped cross-fitted selector evaluation on frozen pools, and its shadow receipt.
//!
//! Parents are assigned to `folds` by `sha256(salt ‖ parent)`; fitted selectors for fold `k`
//! learn only from pools whose parent is outside fold `k`, so no evaluated context, sibling cut
//! or parent contributes to its own selector. Every selector is scored on exactly the same
//! pools, so per-context differences are paired. The primary contrast is each candidate against
//! the strongest baseline by mean realized gain; beating uniform alone never passes.

use super::bootstrap::{cluster_bootstrap_ci, PairedInterval, Verdict};
use super::metrics::{score_context, uniform_context, ContextMetrics};
use super::pool::{verify, Manifest, Pool, Reference};
use super::selectors::{
    token_bag_features, Class, ContextualSurrogateQHat, CvEstimate, CvUnavailable,
    ExternalProgramPrior, Features, ProgramPrior, RetrievalKnn, RidgeProgram, Selector,
    MAX_SURROGATE_DESIGN_DIM,
};
use crate::canon::canonical_sha256;
use anyhow::{ensure, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

/// v3 uses strict nested/v3 fitting and records unavailable CV without a numeric substitute.
/// Historical v1/v2 receipts retain their recorded procedure and probability semantics.
pub const RECEIPT_SCHEMA: &str = "jope.j9-selection-receipt/v3";
/// Constant identifying the uniform.
pub const UNIFORM: &str = "uniform_exact";

#[derive(Clone, Debug, Serialize)]
/// `Config` is a record for the config contract in this module: Parent-grouped cross-fitted selector evaluation on frozen pools, and its shadow receipt.
pub struct Config {
    /// Reference recorded for this config.
    pub reference: Reference,
    /// Budget recorded for this config.
    pub budget: usize,
    /// Folds recorded for this config.
    pub folds: usize,
    /// Fold salt recorded for this config.
    pub fold_salt: String,
    /// Prior weight recorded for this config.
    pub prior_weight: f64,
    /// Knn k recorded for this config.
    pub knn_k: usize,
    /// Ridge lambda recorded for this config.
    pub ridge_lambda: f64,
    /// Bootstrap draws recorded for this config.
    pub bootstrap_draws: usize,
    /// Bootstrap seed recorded for this config.
    pub bootstrap_seed: u64,
    /// Alpha recorded for this config.
    pub alpha: f64,
}

/// Inputs that fitted baselines may read. Absent features disable the selector that needs them;
/// the receipt lists what ran.
pub struct Inputs<'a> {
    /// Optional context features value; `None` means it is not available.
    pub context_features: Option<&'a Features>,
    /// Optional program features value; `None` means it is not available.
    pub program_features: Option<&'a Features>,
    /// Externally scored selectors (PolicyLM q-head, native GP ranking, learned candidates).
    pub external: Vec<&'a dyn Selector>,
    /// Selectors CLOVER names that cannot run on this source, with the reason; recorded, not faked.
    pub unavailable: Vec<Unavailable>,
}

/// Performs the fold of operation and returns an error if it cannot be completed.
pub fn fold_of(salt: &str, parent: &str, folds: usize) -> usize {
    let mut h = Sha256::new();
    h.update((salt.len() as u64).to_le_bytes());
    h.update(salt.as_bytes());
    h.update(parent.as_bytes());
    let d = h.finalize();
    (u64::from_le_bytes(d[..8].try_into().expect("8 bytes")) % folds as u64) as usize
}

#[derive(Clone, Debug, Serialize)]
/// `Unavailable` is a record for the unavailable contract in this module: Parent-grouped cross-fitted selector evaluation on frozen pools, and its shadow receipt.
pub struct Unavailable {
    /// Name recorded for this unavailable.
    pub name: String,
    /// Explanation associated with the recorded disposition.
    pub reason: String,
}

#[derive(Clone, Debug, Serialize)]
/// `SelectorRecord` is a record for the selector record contract in this module: Parent-grouped cross-fitted selector evaluation on frozen pools, and its shadow receipt.
pub struct SelectorRecord {
    /// Name recorded for this selector record.
    pub name: String,
    /// Class recorded for this selector record.
    pub class: Class,
    /// Rule recorded for this selector record.
    pub rule: String,
}

#[derive(Clone, Debug, Serialize)]
/// `ContextRecord` is a record for the context record contract in this module: Parent-grouped cross-fitted selector evaluation on frozen pools, and its shadow receipt.
pub struct ContextRecord {
    /// Identifier for the associated context.
    pub context_id: String,
    /// Identifier for the associated parent.
    pub parent_id: String,
    /// Fold recorded for this context record.
    pub fold: usize,
    /// SHA-256 digest identifying the associated pool.
    pub pool_sha256: String,
    /// Members recorded for this context record.
    pub members: usize,
    /// Oracle gain recorded for this context record.
    pub oracle_gain: f64,
    /// Mapping of metrics values carried by this context record.
    pub metrics: BTreeMap<String, ContextMetrics>,
}

#[derive(Clone, Debug, Serialize)]
/// `Summary` is a record for the summary contract in this module: Parent-grouped cross-fitted selector evaluation on frozen pools, and its shadow receipt.
pub struct Summary {
    /// Contexts recorded for this summary.
    pub contexts: usize,
    /// Mean realized gain recorded for this summary.
    pub mean_realized_gain: f64,
    /// Mean top1 gain recorded for this summary.
    pub mean_top1_gain: f64,
    /// Mean regret recorded for this summary.
    pub mean_regret: f64,
    /// Optional mean normalized regret value; `None` means it is not available.
    pub mean_normalized_regret: Option<f64>,
    /// Mean probability that the selection includes a member strictly above the reference.
    pub beats_reference_rate: f64,
    /// Pairwise contexts recorded for this summary.
    pub pairwise_contexts: usize,
    /// Optional mean pairwise accuracy value; `None` means it is not available.
    pub mean_pairwise_accuracy: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
/// `Comparison` is a record for the comparison contract in this module: Parent-grouped cross-fitted selector evaluation on frozen pools, and its shadow receipt.
pub struct Comparison {
    /// Candidate recorded for this comparison.
    pub candidate: String,
    /// Baseline recorded for this comparison.
    pub baseline: String,
    /// Paired `candidate - baseline` per context; positive favours the candidate.
    pub realized_gain: PairedInterval,
    /// Optional pairwise accuracy value; `None` means it is not available.
    pub pairwise_accuracy: Option<PairedInterval>,
}

#[derive(Clone, Debug, Serialize)]
/// `FoldCvRecord` is a record for the fold cross-validation record contract in this module: Parent-grouped cross-fitted selector evaluation on frozen pools, and its shadow receipt.
pub struct FoldCvRecord {
    /// Fold recorded for this fold cross-validation record.
    pub fold: usize,
    /// Estimate recorded for this fold cross-validation record.
    pub estimate: CvEstimate,
    /// Collection of fits values carried by this fold cross-validation record.
    pub fits: Vec<super::selectors::InnerFoldFit>,
}

#[derive(Clone, Debug, Serialize)]
/// `SelectorCoverage` is a record for the selector coverage contract in this module: Parent-grouped cross-fitted selector evaluation on frozen pools, and its shadow receipt.
pub struct SelectorCoverage {
    /// Unit: frozen outer contexts. Whole-selector refusal removes every partial score.
    pub attempted: usize,
    /// Scored recorded for this selector coverage.
    pub scored: usize,
    /// Excluded recorded for this selector coverage.
    pub excluded: usize,
    /// Failed recorded for this selector coverage.
    pub failed: usize,
}

#[derive(Clone, Debug, Serialize)]
/// `Receipt` is a record for the receipt contract in this module: Parent-grouped cross-fitted selector evaluation on frozen pools, and its shadow receipt.
pub struct Receipt {
    /// Schema identifier and version for the serialized value.
    pub schema: &'static str,
    /// Shadow only recorded for this receipt.
    pub shadow_only: bool,
    /// Qualified claim recorded for this receipt.
    pub qualified_claim: bool,
    /// L6 claim recorded for this receipt.
    pub l6_claim: bool,
    /// SHA-256 digest identifying the associated manifest.
    pub manifest_sha256: String,
    /// Config recorded for this receipt.
    pub config: Config,
    /// Collection of selectors values carried by this receipt.
    pub selectors: Vec<SelectorRecord>,
    /// Collection of unavailable selectors values carried by this receipt.
    pub unavailable_selectors: Vec<Unavailable>,
    /// Collection of surrogate cross-validation values carried by this receipt.
    pub surrogate_cv: Vec<FoldCvRecord>,
    /// Mapping of selector coverage values carried by this receipt.
    pub selector_coverage: BTreeMap<String, SelectorCoverage>,
    /// Program features recorded for this receipt.
    pub program_features: String,
    /// Collection of fold training pools values carried by this receipt.
    pub fold_training_pools: Vec<usize>,
    /// Mapping of summaries values carried by this receipt.
    pub summaries: BTreeMap<String, Summary>,
    /// Strongest baseline recorded for this receipt.
    pub strongest_baseline: String,
    /// Collection of comparisons values carried by this receipt.
    pub comparisons: Vec<Comparison>,
    /// `ESTABLISHED_SHADOW` only when a candidate's realized-gain cluster CI against the
    /// strongest baseline lies above zero; otherwise `NOT_ESTABLISHED`. Never a qualification.
    pub ranking_learnability: String,
    /// Collection of contexts values carried by this receipt.
    pub contexts: Vec<ContextRecord>,
}

impl Receipt {
    /// Computes the canonical SHA-256 digest of this value.
    pub fn sha256(&self) -> Result<String> {
        canonical_sha256(self)
    }
}

fn summarize(contexts: &[ContextRecord], name: &str) -> Summary {
    let rows: Vec<&ContextMetrics> = contexts.iter().map(|c| &c.metrics[name]).collect();
    let n = rows.len() as f64;
    let mean = |f: fn(&ContextMetrics) -> f64| rows.iter().map(|m| f(m)).sum::<f64>() / n;
    let pairwise: Vec<f64> = rows.iter().filter_map(|m| m.pairwise_accuracy).collect();
    Summary {
        contexts: rows.len(),
        mean_realized_gain: mean(|m| m.realized_gain),
        mean_top1_gain: mean(|m| m.top1_gain),
        mean_regret: mean(|m| m.regret),
        mean_normalized_regret: {
            let v: Vec<f64> = rows.iter().filter_map(|m| m.normalized_regret).collect();
            (!v.is_empty()).then(|| v.iter().sum::<f64>() / v.len() as f64)
        },
        beats_reference_rate: mean(|m| m.beats_reference),
        pairwise_contexts: pairwise.len(),
        mean_pairwise_accuracy: (!pairwise.is_empty())
            .then(|| pairwise.iter().sum::<f64>() / pairwise.len() as f64),
    }
}

fn paired(
    contexts: &[ContextRecord],
    a: &str,
    b: &str,
    f: impl Fn(&ContextMetrics) -> Option<f64>,
    cfg: &Config,
) -> Result<Option<PairedInterval>> {
    let mut groups: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    for c in contexts {
        if let (Some(x), Some(y)) = (f(&c.metrics[a]), f(&c.metrics[b])) {
            groups.entry(&c.parent_id).or_default().push(x - y);
        }
    }
    if groups.values().map(Vec::len).sum::<usize>() < 2 {
        return Ok(None);
    }
    let groups: Vec<Vec<f64>> = groups.into_values().collect();
    Ok(Some(cluster_bootstrap_ci(
        &groups,
        cfg.bootstrap_draws,
        cfg.bootstrap_seed,
        cfg.alpha,
    )?))
}

/// Performs the evaluate operation and returns an error if it cannot be completed.
pub fn evaluate(
    pools: &[Pool],
    manifest: &Manifest,
    inputs: &Inputs<'_>,
    cfg: &Config,
) -> Result<Receipt> {
    evaluate_with_fold_train_pools(pools, None, manifest, inputs, cfg)
}

/// Performs the evaluate with fold train pools operation and returns an error if it cannot be completed.
pub fn evaluate_with_fold_train_pools(
    pools: &[Pool],
    fold_train_pools: Option<&[Vec<Pool>]>,
    manifest: &Manifest,
    inputs: &Inputs<'_>,
    cfg: &Config,
) -> Result<Receipt> {
    evaluate_with_fold_train_pools_and_priors(pools, fold_train_pools, None, manifest, inputs, cfg)
}

/// Performs the evaluate with fold train pools and priors operation and returns an error if it cannot be completed.
pub fn evaluate_with_fold_train_pools_and_priors(
    pools: &[Pool],
    fold_train_pools: Option<&[Vec<Pool>]>,
    fold_external_priors: Option<&[ExternalProgramPrior]>,
    manifest: &Manifest,
    inputs: &Inputs<'_>,
    cfg: &Config,
) -> Result<Receipt> {
    verify(pools, manifest)?;
    ensure!(cfg.budget >= 1, "budget must be positive");
    ensure!(cfg.folds >= 2, "cross-fitting needs at least two folds");
    ensure!(!pools.is_empty(), "no frozen pools");
    if let Some(custom) = fold_train_pools {
        ensure!(
            custom.len() == cfg.folds,
            "fold_train_pools length {} must match cfg.folds {}",
            custom.len(),
            cfg.folds
        );
        for (k, fold_pools) in custom.iter().enumerate() {
            for tp in fold_pools {
                ensure!(
                    fold_of(&cfg.fold_salt, &tp.parent_id, cfg.folds) != k,
                    "fold {k} training pool {} (parent {}) belongs to held-out fold {k}",
                    tp.context_id,
                    tp.parent_id
                );
            }
        }
    }
    if let Some(priors) = fold_external_priors {
        ensure!(
            priors.len() == cfg.folds,
            "fold_external_priors length {} must match cfg.folds {}",
            priors.len(),
            cfg.folds
        );
        for (k, ext) in priors.iter().enumerate() {
            for (parent_id, _) in ext.parent_stats.keys() {
                ensure!(
                    fold_of(&cfg.fold_salt, parent_id, cfg.folds) != k,
                    "fold {k} external prior parent {parent_id} belongs to held-out fold {k}"
                );
            }
        }
    }
    let folds: Vec<usize> = pools
        .iter()
        .map(|p| fold_of(&cfg.fold_salt, &p.parent_id, cfg.folds))
        .collect();
    let refs: Vec<f64> = pools
        .iter()
        .map(|p| p.reference(cfg.reference))
        .collect::<Result<_>>()?;
    // Token-bag program features are outcome-free, so they may be derived from every pool.
    let derived = match inputs.program_features {
        Some(_) => None,
        None => token_bag_features(pools),
    };
    let program_features = inputs.program_features.or(derived.as_ref());
    let mut unavailable_selectors = inputs.unavailable.clone();
    let context_verified = inputs
        .context_features
        .is_some_and(|cf| cf.verify_context_descriptor().is_ok());
    if inputs.context_features.is_some() && !context_verified {
        unavailable_selectors.push(Unavailable { name: "retrieval_knn".into(), reason: "nested/v3 requires verified target-free context descriptors; unbound context provenance".into() });
    }
    let already_marked_unavailable = unavailable_selectors
        .iter()
        .any(|u| u.name == "contextual_surrogate_qhat");
    let can_fit_surrogate = if already_marked_unavailable {
        false
    } else {
        match (inputs.context_features, program_features) {
            (Some(cf), Some(pf)) => {
                if cf.dim == 0 || pf.dim == 0 {
                    unavailable_selectors.push(Unavailable {
                        name: "contextual_surrogate_qhat".into(),
                        reason: format!(
                            "non-empty feature dimensions required (d_ctx={}, d_prog={})",
                            cf.dim, pf.dim
                        ),
                    });
                    false
                } else {
                    match ContextualSurrogateQHat::effective_design_dim(cf.dim, pf.dim) {
                        Some(d) if d <= MAX_SURROGATE_DESIGN_DIM => true,
                        Some(d) => {
                            unavailable_selectors.push(Unavailable {
                                name: "contextual_surrogate_qhat".into(),
                                reason: format!(
                                    "effective design dimension {d} exceeds MAX_SURROGATE_DESIGN_DIM ({MAX_SURROGATE_DESIGN_DIM})"
                                ),
                            });
                            false
                        }
                        None => {
                            unavailable_selectors.push(Unavailable {
                                name: "contextual_surrogate_qhat".into(),
                                reason: format!(
                                    "effective design dimension overflow (d_ctx={}, d_prog={})",
                                    cf.dim, pf.dim
                                ),
                            });
                            false
                        }
                    }
                }
            }
            _ => false,
        }
    };
    let mut contexts: Vec<ContextRecord> = pools
        .iter()
        .zip(&folds)
        .zip(&manifest.pools)
        .enumerate()
        .map(|(i, ((p, &fold), entry))| {
            let outcomes = p.outcomes();
            let mut metrics = BTreeMap::new();
            metrics.insert(
                UNIFORM.to_owned(),
                uniform_context(&outcomes, refs[i], cfg.budget),
            );
            ContextRecord {
                context_id: p.context_id.clone(),
                parent_id: p.parent_id.clone(),
                fold,
                pool_sha256: entry.pool_sha256.clone(),
                members: p.members.len(),
                oracle_gain: super::metrics::oracle_gain(&outcomes, refs[i]),
                metrics,
            }
        })
        .collect();
    let mut records: BTreeMap<String, SelectorRecord> = BTreeMap::new();
    records.insert(
        UNIFORM.into(),
        SelectorRecord {
            name: UNIFORM.into(),
            class: Class::Baseline,
            rule: "uniform_exact/v1: exact expectation of B distinct members drawn uniformly without replacement; pairwise 0.5".into(),
        },
    );
    let mut score_with =
        |sel: &dyn Selector, idx: &[usize], contexts: &mut [ContextRecord]| -> Result<()> {
            let rec = records
                .entry(sel.name().to_owned())
                .or_insert_with(|| SelectorRecord {
                    name: sel.name().to_owned(),
                    class: sel.class(),
                    rule: sel.rule(),
                });
            ensure!(
                rec.class == sel.class(),
                "selector {} changes class across folds",
                sel.name()
            );
            for &i in idx {
                let pool = &pools[i];
                let view = pool.view();
                let scores = sel.scores(&view)?;
                ensure!(
                    scores.len() == view.programs.len(),
                    "{}: one score per member",
                    sel.name()
                );
                ensure!(
                    scores.iter().all(|s| s.is_finite()),
                    "{}: non-finite score",
                    sel.name()
                );
                let m = score_context(
                    &pool.outcomes(),
                    refs[i],
                    &scores,
                    &view.programs,
                    cfg.budget,
                );
                ensure!(
                    contexts[i]
                        .metrics
                        .insert(sel.name().to_owned(), m)
                        .is_none(),
                    "selector name {} is not unique",
                    sel.name()
                );
            }
            Ok(())
        };
    let mut fold_training_pools = Vec::with_capacity(cfg.folds);
    let mut surrogate_fold_cv = Vec::with_capacity(cfg.folds);
    let mut surrogate_failed = false;
    let mut surrogate_failed_contexts = 0;
    for k in 0..cfg.folds {
        let held: Vec<usize> = (0..pools.len()).filter(|&i| folds[i] == k).collect();
        let train: Vec<&Pool> = match fold_train_pools {
            Some(custom) => custom[k].iter().collect(),
            None => (0..pools.len())
                .filter(|&i| folds[i] != k)
                .map(|i| &pools[i])
                .collect(),
        };
        fold_training_pools.push(train.len());
        if held.is_empty() {
            continue;
        }
        ensure!(!train.is_empty(), "fold {k} has no training pools");
        let prior = ProgramPrior::fit(&train, cfg.prior_weight);
        score_with(&prior, &held, &mut contexts)?;
        if let Some(cf) = inputs.context_features.filter(|_| context_verified) {
            let knn = RetrievalKnn::fit(&train, cf, cfg.knn_k, cfg.prior_weight)?;
            score_with(&knn, &held, &mut contexts)?;
        }
        if let Some(pf) = program_features {
            let ridge = RidgeProgram::fit(&train, pf, cfg.ridge_lambda)?;
            score_with(&ridge, &held, &mut contexts)?;
        }
        if can_fit_surrogate {
            if let (Some(cf), Some(pf)) = (inputs.context_features, program_features) {
                let ext_prior = fold_external_priors.map(|ps| &ps[k]);
                match ContextualSurrogateQHat::fit_with_prior(
                    &train,
                    cf,
                    pf,
                    cfg.knn_k,
                    cfg.prior_weight,
                    cfg.ridge_lambda,
                    ext_prior,
                ) {
                    Ok(surrogate) => {
                        surrogate_fold_cv.push(FoldCvRecord {
                            fold: k,
                            estimate: surrogate.inner_cv_estimate().clone(),
                            fits: surrogate.inner_fold_fits().to_vec(),
                        });
                        if let Err(error) = score_with(&surrogate, &held, &mut contexts) {
                            surrogate_failed = true;
                            surrogate_failed_contexts += held.len();
                            unavailable_selectors.push(Unavailable {
                                name: "contextual_surrogate_qhat".into(),
                                reason: format!("outer fold {k} scoring failed: {error}"),
                            });
                        }
                    }
                    Err(error) => {
                        surrogate_failed = true;
                        surrogate_failed_contexts += held.len();
                        if let Some(cv) = error.downcast_ref::<CvUnavailable>() {
                            surrogate_fold_cv.push(FoldCvRecord {
                                fold: k,
                                estimate: cv.0.clone(),
                                fits: Vec::new(),
                            });
                        }
                        unavailable_selectors.push(Unavailable {
                            name: "contextual_surrogate_qhat".into(),
                            reason: format!("outer fold {k}: {error}"),
                        });
                    }
                }
            }
        }
    }
    let all: Vec<usize> = (0..pools.len()).collect();
    for sel in &inputs.external {
        score_with(*sel, &all, &mut contexts)?;
    }
    if surrogate_failed {
        records.remove("contextual_surrogate_qhat");
        for context in &mut contexts {
            context.metrics.remove("contextual_surrogate_qhat");
        }
    }
    let mut selector_coverage: BTreeMap<String, SelectorCoverage> = records
        .keys()
        .map(|name| {
            (
                name.clone(),
                SelectorCoverage {
                    attempted: pools.len(),
                    scored: pools.len(),
                    excluded: 0,
                    failed: 0,
                },
            )
        })
        .collect();
    for missing in &unavailable_selectors {
        selector_coverage
            .entry(missing.name.clone())
            .or_insert(SelectorCoverage {
                attempted: pools.len(),
                scored: 0,
                excluded: if surrogate_failed && missing.name == "contextual_surrogate_qhat" {
                    pools.len() - surrogate_failed_contexts
                } else {
                    pools.len()
                },
                failed: if surrogate_failed && missing.name == "contextual_surrogate_qhat" {
                    surrogate_failed_contexts
                } else {
                    0
                },
            });
    }
    let selectors: Vec<SelectorRecord> = records.into_values().collect();
    for c in &contexts {
        ensure!(
            c.metrics.len() == selectors.len(),
            "context {} is not scored by every selector",
            c.context_id
        );
    }
    let summaries: BTreeMap<String, Summary> = selectors
        .iter()
        .map(|s| (s.name.clone(), summarize(&contexts, &s.name)))
        .collect();
    let strongest_baseline = selectors
        .iter()
        .filter(|s| s.class == Class::Baseline)
        .max_by(|a, b| {
            summaries[&a.name]
                .mean_realized_gain
                .total_cmp(&summaries[&b.name].mean_realized_gain)
                .then_with(|| b.name.cmp(&a.name))
        })
        .map(|s| s.name.clone())
        .expect("uniform is always a baseline");
    let mut comparisons = Vec::new();
    let mut established = false;
    for cand in selectors.iter().filter(|s| s.class == Class::Candidate) {
        for base in selectors.iter().filter(|s| s.class == Class::Baseline) {
            let Some(gain) = paired(
                &contexts,
                &cand.name,
                &base.name,
                |m| Some(m.realized_gain),
                cfg,
            )?
            else {
                continue;
            };
            if base.name == strongest_baseline && gain.verdict == Verdict::Greater {
                established = true;
            }
            comparisons.push(Comparison {
                candidate: cand.name.clone(),
                baseline: base.name.clone(),
                realized_gain: gain,
                pairwise_accuracy: paired(
                    &contexts,
                    &cand.name,
                    &base.name,
                    |m| m.pairwise_accuracy,
                    cfg,
                )?,
            });
        }
    }
    contexts.sort_by(|a, b| a.context_id.cmp(&b.context_id));
    Ok(Receipt {
        schema: RECEIPT_SCHEMA,
        shadow_only: true,
        qualified_claim: false,
        l6_claim: false,
        manifest_sha256: manifest.sha256()?,
        config: cfg.clone(),
        selectors,
        unavailable_selectors,
        surrogate_cv: surrogate_fold_cv,
        selector_coverage,
        program_features: match (inputs.program_features, &derived) {
            (Some(f), _) => format!("supplied: {} dimensions", f.dim),
            (None, Some(f)) => format!(
                "token_bag/v1 derived from pool tokens: {} dimensions",
                f.dim
            ),
            (None, None) => "none".into(),
        },
        fold_training_pools,
        summaries,
        strongest_baseline,
        comparisons,
        ranking_learnability: if established {
            "ESTABLISHED_SHADOW"
        } else {
            "NOT_ESTABLISHED"
        }
        .into(),
        contexts,
    })
}
