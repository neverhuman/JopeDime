//! End-to-end orchestration: residual -> GP synthesis -> augmented matrix -> importance epochs ->
//! EDA selection (single, then optional MIMO) -> lift report (baseline raw vs selected augmented).
use crate::backend::{Backend, TrainSpec};
use crate::buildmatrix::{build_augmented_from_search, Augmented, FeatureInfo};
use crate::canary::{self, CanaryAugmentation};
use crate::data::{Dataset, Matrix};
use crate::foldscope::FoldScope;
use crate::gp::{run_gp, GpConfig, GpData};
use crate::hashing::dataset_hash;
use crate::leakage::{scan, LeakageReport};
use crate::metrics::{auc, task_score};
use crate::phase2::{importance_epochs_detailed, P2Config};
use crate::phase3::SingleScorer;
use crate::phase4::MimoScorer;
use crate::progress::{EnrichedFeature, Event, Sink};
use crate::residual::oof_residual;
use crate::select::{run_eda, Champion, EdaConfig, MaskScorer};
use crate::task::{model_predict, TaskKind};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::time::Instant;

#[derive(Clone)]
pub struct PipelineConfig {
    pub gp: GpConfig,
    pub p2: P2Config,
    pub eda: EdaConfig,
    pub cv_spec: TrainSpec,
    pub ho_spec: TrainSpec,
    pub resid_spec: TrainSpec,
    pub select_subsample: usize,
    pub max_gp: usize,
    pub do_mimo: bool,
    pub seed: u64,
    /// Learning task (drives residual / metric / prediction dispatch). Defaults to Binary so existing
    /// binary runs are unchanged.
    pub task: crate::task::TaskKind,
    /// Number of classes for MultiClass (else unused).
    pub n_classes: usize,
    /// Multi-target stacking (Phase 5). Inert unless enabled AND the dataset has sibling targets.
    pub stack: crate::stacking::StackConfig,
    /// Ensemble-dominance engine (Phase 5). Diverse OOF library + BBC-gated fusion. Default off.
    pub dominance: crate::dominance::DominanceConfig,
    /// Optional Starforge final-stage candidates. Inert unless compiled with a Starforge feature and
    /// enabled by CLI/config.
    pub starforge: crate::starforge_integration::StarforgeConfig,
    /// Optional v301 fair router receipt. The router is advisory for this pipeline run and maps Python-only
    /// Starlight Core-family names to internal Starlight Core/Starforge routes.
    pub router: crate::router::RouterConfig,
    /// CatBoost-guided coreset threshold for Phase-3/5 row selection. Datasets at or below the threshold
    /// keep the historical stratified random subsample.
    pub coreset_threshold: usize,
    /// Fraction of the selected coreset reserved for random guard rows.
    pub coreset_guard_frac: f64,
    /// Opt-in Phase-3.5 strategy bake-off (model × feature-set CV competition). Default off.
    pub bakeoff: crate::bakeoff::BakeoffConfig,
    /// Pre-GP Hyperion feature injection policy. Data loaders consume this before `run_pipeline`.
    pub hyperion: crate::hyperion::HyperionConfig,
    /// Phase-6 algorithm invention (feat-math pipeline-genome evolution). Default off (gens = 0).
    pub invention: crate::invention::InventionConfig,
    /// Synthetic CANARY_FEATURES negative controls. Default on; disable with `--no-canary-features`
    /// or emergency env `JAIN_CANARY_FEATURES=0|off|false`.
    pub canary: crate::canary::CanaryConfig,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        PipelineConfig {
            gp: GpConfig::default(),
            p2: P2Config::default(),
            eda: EdaConfig::default(),
            cv_spec: TrainSpec::from_basic_fields(120, 5, 0.1, 2, 0),
            ho_spec: TrainSpec::from_basic_fields(300, 5, 0.05, 16, 0),
            resid_spec: TrainSpec::from_basic_fields(200, 5, 0.05, 16, 0),
            select_subsample: 28000,
            max_gp: 4000,
            do_mimo: false,
            seed: 20260616,
            task: crate::task::TaskKind::Binary,
            n_classes: 2,
            stack: crate::stacking::StackConfig::default(),
            dominance: crate::dominance::DominanceConfig::default(),
            starforge: crate::starforge_integration::StarforgeConfig::default(),
            router: crate::router::RouterConfig::default(),
            coreset_threshold: 50_000,
            coreset_guard_frac: 0.20,
            bakeoff: crate::bakeoff::BakeoffConfig::default(),
            hyperion: crate::hyperion::HyperionConfig::default(),
            invention: crate::invention::InventionConfig::default(),
            canary: crate::canary::CanaryConfig::default(),
        }
    }
}

/// Trust receipt: stable identity hashes + the pre-search leakage scan (report-only).
pub struct TrustReport {
    pub dataset_hash: u64,
    pub fold_hash: u64,
    pub leakage: LeakageReport,
    /// Augmented features dropped by the leak guard (perfect single-feature separators).
    pub leak_features_dropped: usize,
}

pub struct PipelineResult {
    pub baseline_holdout_auc: f64,
    pub champion_holdout_auc: f64,
    pub champion_cv_auc: f64,
    pub champion_size: usize,
    pub mimo_holdout_auc: Option<f64>,
    pub n_survivors: usize,
    pub n_features: usize,
    pub lift: f64,
    pub secs: f64,
    /// Final CatBoost-vs-ensemble CV selection on the champion features (when enabled).
    pub ensemble: Option<crate::ensemble::EnsembleResult>,
    /// Fairness/identity receipt for this run.
    pub trust: TrustReport,
    /// State needed to replay preprocessing/feature synthesis and refit the selected serving model.
    pub artifact: PipelineArtifactState,
    /// v301 fair-router decision when router assets were configured.
    pub router: Option<crate::router::RouterDecision>,
    /// Phase-3.5 strategy bake-off report (all strategies' CV + test), when `--bakeoff` was set.
    pub bakeoff: Option<crate::bakeoff::BakeoffReport>,
    /// Phase-6 invention report (top-3 evolved genomes with proper CV + emitted Rust/Python), when
    /// `invention.gens > 0` and the task supports it.
    pub invention: Option<crate::invention::InventionReport>,
    /// True when the sink requested cancellation and later stages were skipped. The partial result
    /// is still well-formed (skipped stages report empty/None), but callers should treat it as aborted.
    pub cancelled: bool,
    /// CANARY_FEATURES summary. GP survivors that reference a canary column are quarantined before
    /// feature reduction (see `build_augmented_from_search`), so on a clean run `selected_canary_count`
    /// and `top_importance_contaminated_count` are 0 — this report is the receipt proving it.
    pub canary: CanaryReport,
}

#[derive(Clone)]
pub struct PipelineArtifactState {
    pub features: Vec<FeatureInfo>,
    pub selected_cols: Vec<u32>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CanaryReport {
    pub enabled: bool,
    pub count: usize,
    pub seed: u64,
    pub selected_canary_count: usize,
    pub top_importance_contaminated_count: usize,
    pub contaminated_formulas: Vec<String>,
}

/// Baseline: CatBoost (ho_spec) on the RAW base features, holdout AUC.
pub fn baseline_holdout(
    train: &Dataset,
    hold: &Dataset,
    be: &dyn Backend,
    spec: &TrainSpec,
    task: TaskKind,
    n_classes: usize,
) -> f64 {
    let p = train.p();
    let cols: Vec<u32> = (0..p as u32).collect();
    let xtr = train.x.select_cols(&cols);
    let m = be.train_single(&xtr, train.n(), p, &train.y, spec);
    let xh = hold.x.select_cols(&cols);
    let pr = model_predict(&*m, &xh, hold.n(), p, task, n_classes);
    task_score(task, &hold.y, &pr, n_classes)
}

/// Emit the top importance features, classified by origin so the cockpit can call out engineered features:
/// base, `gp` (synthesized), `hyperion` (embedding column), and GP features whose expression references a
/// Hyperion column (`refs_hyperion`). `mv[j]` is the importance of `features[j]` (same post-leak-drop frame).
fn emit_top_features(
    sink: &dyn Sink,
    features: &[FeatureInfo],
    mv: &[f32],
    names: &[String],
    t0: Instant,
) -> Vec<EnrichedFeature> {
    if features.is_empty() || mv.len() != features.len() {
        return Vec::new();
    }
    // Hyperion embedding columns are base columns named `hyperion::vNNN`, in the GP's coordinate frame.
    let hyper: HashSet<usize> = names
        .iter()
        .enumerate()
        .filter(|(_, n)| n.starts_with("hyperion::"))
        .map(|(i, _)| i)
        .collect();
    let canary_cols = canary_column_indices(names, features);
    let mut order: Vec<usize> = (0..features.len()).collect();
    order.sort_unstable_by(|&a, &b| {
        mv[b]
            .partial_cmp(&mv[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let top: Vec<EnrichedFeature> = order
        .iter()
        .take(24)
        .map(|&j| {
            let f = &features[j];
            let is_gp = f.source == "gp";
            let direct_canary = canary::feature_is_direct_canary(f);
            let refs_canary = if direct_canary {
                true
            } else if is_gp {
                let mut refs = Vec::new();
                f.expr.col_refs(&mut refs);
                refs.iter().any(|i| canary_cols.contains(i))
            } else {
                false
            };
            let source = if direct_canary {
                "canary"
            } else if is_gp {
                "gp"
            } else if f.formula.starts_with("hyperion::") {
                "hyperion"
            } else {
                "base"
            };
            let refs_hyperion = if is_gp {
                let mut refs = Vec::new();
                f.expr.col_refs(&mut refs);
                refs.iter().any(|i| hyper.contains(i))
            } else {
                false
            };
            EnrichedFeature {
                name: f.formula.clone(),
                importance: mv[j] as f64,
                source: source.to_string(),
                refs_hyperion,
                refs_canary,
            }
        })
        .collect();
    let mut e = Event::new(2, "importance", t0.elapsed().as_secs_f64());
    e.top_features = Some(top.clone());
    sink.emit(&e);
    top
}

fn canary_column_indices(names: &[String], features: &[FeatureInfo]) -> HashSet<usize> {
    let mut cols: HashSet<usize> = names
        .iter()
        .enumerate()
        .filter(|(_, n)| canary::is_canary_name(n))
        .map(|(i, _)| i)
        .collect();
    for f in features
        .iter()
        .filter(|f| canary::feature_is_direct_canary(f))
    {
        let mut refs = Vec::new();
        f.expr.col_refs(&mut refs);
        cols.extend(refs);
    }
    cols
}

fn feature_refs_canary(f: &FeatureInfo, canary_cols: &HashSet<usize>) -> bool {
    if canary::feature_is_direct_canary(f) {
        return true;
    }
    if f.source != "gp" {
        return false;
    }
    let mut refs = Vec::new();
    f.expr.col_refs(&mut refs);
    refs.iter().any(|i| canary_cols.contains(i))
}

fn build_canary_report(
    aug: &CanaryAugmentation,
    features: &[FeatureInfo],
    selected_cols: &[u32],
    base_names: &[String],
    top_features: &[EnrichedFeature],
) -> CanaryReport {
    let canary_cols = canary_column_indices(base_names, features);
    let contaminated_formulas = selected_cols
        .iter()
        .filter_map(|&idx| {
            let f = features.get(idx as usize)?;
            feature_refs_canary(f, &canary_cols).then(|| f.formula.clone())
        })
        .collect::<Vec<_>>();
    CanaryReport {
        enabled: aug.enabled,
        count: aug.specs.len(),
        seed: aug.seed,
        selected_canary_count: contaminated_formulas.len(),
        top_importance_contaminated_count: top_features
            .iter()
            .filter(|f| f.refs_canary || f.source == "canary")
            .count(),
        contaminated_formulas,
    }
}

fn emit_mvp_fast_progress(
    sink: &dyn Sink,
    t0: Instant,
    done: u64,
    total: u64,
    selected: &crate::mvp::MvpFastSelection,
) {
    let mut e = Event::new(3, "genprog", t0.elapsed().as_secs_f64());
    e.model = Some("mvp-fast".into());
    e.done = Some(done);
    e.blocks_total = Some(total.max(1));
    e.done_frac = Some(done as f64 / total.max(1) as f64);
    e.best_size = Some(selected.cols.len() as u64);
    e.accepted_features = Some(selected.report.accepted_gp as u64);
    e.selected_features = Some(selected.report.selected_gp as u64);
    e.final_features = Some(selected.report.final_features as u64);
    sink.emit(&e);
}

/// Task-aware single-feature "strength" vs the target, used by the leak guard. Classification:
/// `max(auc, 1-auc)` (Binary byte-identical to the original); Regression: `|Pearson|`; MultiClass:
/// max one-vs-rest auc.
fn feat_strength(task: TaskKind, y: &[f32], col: &[f32], n_classes: usize) -> f64 {
    match task {
        TaskKind::Binary | TaskKind::MultiLabel => {
            let a = auc(y, col);
            a.max(1.0 - a)
        }
        TaskKind::Regression => {
            let (ys, ok1) = crate::metrics::standardize(y);
            let (cs, ok2) = crate::metrics::standardize(col);
            if ok1 && ok2 {
                crate::metrics::corr_std(&ys, &cs) as f64
            } else {
                0.0
            }
        }
        TaskKind::MultiClass => (0..n_classes)
            .map(|c| {
                let yc: Vec<f32> = y
                    .iter()
                    .map(|&v| if (v as usize) == c { 1.0 } else { 0.0 })
                    .collect();
                let a = auc(&yc, col);
                a.max(1.0 - a)
            })
            .fold(0.5_f64, f64::max),
    }
}

/// Drop augmented features that single-handedly separate the target at |AUC| >= thresh: these are leaks or
/// degenerate GP features (a real clinical feature tops out far below 0.99) and they would poison the EDA's
/// CV and the holdout. No-op on clean data (nothing reaches the threshold).
fn drop_leak_columns(
    xtr: Matrix,
    xho: Matrix,
    features: Vec<FeatureInfo>,
    y: &[f32],
    thresh: f64,
    task: TaskKind,
    n_classes: usize,
) -> (Matrix, Matrix, Vec<FeatureInfo>, usize) {
    let mut keep: Vec<u32> = (0..xtr.c)
        .into_par_iter()
        .filter(|&j| feat_strength(task, y, xtr.col(j), n_classes) < thresh)
        .map(|j| j as u32)
        .collect();
    keep.sort_unstable();
    let dropped = xtr.c - keep.len();
    if dropped == 0 {
        return (xtr, xho, features, 0);
    }
    let new_tr = Matrix::new(xtr.select_cols(&keep), xtr.n, keep.len());
    let new_ho = Matrix::new(xho.select_cols(&keep), xho.n, keep.len());
    let new_feat: Vec<FeatureInfo> = keep.iter().map(|&j| features[j as usize].clone()).collect();
    (new_tr, new_ho, new_feat, dropped)
}

fn guided_coreset_rows(
    y: &[f32],
    residual: &[f32],
    target: usize,
    seed: u64,
    guard_frac: f64,
) -> Vec<usize> {
    use crate::rng::seeded;
    use rand::seq::SliceRandom;

    let n = y.len();
    let target = target.min(n).max(1);
    if target >= n {
        return (0..n).collect();
    }
    let guard = ((target as f64) * guard_frac.clamp(0.0, 0.8)).round() as usize;
    let hard_target = target.saturating_sub(guard);
    let mut selected = Vec::with_capacity(target);

    let mut by_class = [Vec::<usize>::new(), Vec::<usize>::new()];
    for i in 0..n {
        by_class[(y[i] > 0.5) as usize].push(i);
    }
    if by_class[0].is_empty() || by_class[1].is_empty() {
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_unstable_by(|&a, &b| {
            residual[b]
                .abs()
                .partial_cmp(&residual[a].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        selected.extend(order.into_iter().take(hard_target));
    } else {
        for class_rows in &mut by_class {
            class_rows.sort_unstable_by(|&a, &b| {
                residual[b]
                    .abs()
                    .partial_cmp(&residual[a].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        let pos_share = by_class[1].len() as f64 / n as f64;
        let take_pos = ((hard_target as f64) * pos_share).round() as usize;
        let take_neg = hard_target.saturating_sub(take_pos);
        selected.extend(by_class[0].iter().take(take_neg).copied());
        selected.extend(by_class[1].iter().take(take_pos).copied());
    }

    let mut taken = vec![false; n];
    for &i in &selected {
        taken[i] = true;
    }
    let mut guard_pool: Vec<usize> = (0..n).filter(|&i| !taken[i]).collect();
    guard_pool.shuffle(&mut seeded(seed ^ 0xC0DE_5017));
    selected.extend(
        guard_pool
            .into_iter()
            .take(target.saturating_sub(selected.len())),
    );
    selected.sort_unstable();
    selected.dedup();
    selected
}

/// Per-value percentile rank in (0,1) (average-ish; ties broken by index).
fn to_percentile(v: &[f32]) -> Vec<f32> {
    let n = v.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut p = vec![0f32; n.max(1)];
    for (rank, &i) in idx.iter().enumerate() {
        p[i] = (rank as f32 + 0.5) / n as f32;
    }
    p
}

/// Hyperion record-outlier magnitude: per-column percentile-normalize each hyperion sweep column, then
/// sum |pctl - 0.5| across the row. High = record sits at the extreme of many sweep dimensions.
fn hyperion_row_magnitude(x: &Matrix, cols: &[usize]) -> Vec<f32> {
    let n = x.n;
    let mut acc = vec![0f32; n];
    for &c in cols {
        let pc = to_percentile(x.col(c));
        for (a, p) in acc.iter_mut().zip(pc.iter()) {
            *a += (p - 0.5).abs();
        }
    }
    acc
}

/// Build a new Dataset with only `keep` rows (column-major row subset). Single-target only.
fn subset_dataset(train: &Dataset, keep: &[usize]) -> Dataset {
    let n = train.x.n;
    let c = train.x.c;
    let m = keep.len();
    let mut data = vec![0f32; m * c];
    for col in 0..c {
        let src = &train.x.data[col * n..col * n + n];
        for (new_row, &old_row) in keep.iter().enumerate() {
            data[col * m + new_row] = src[old_row];
        }
    }
    let y: Vec<f32> = keep.iter().map(|&i| train.y[i]).collect();
    Dataset {
        x: Matrix::new(data, m, c),
        y,
        y_multi: None,
        names: train.names.clone(),
    }
}

fn visible_base_indices(names: &[String]) -> Vec<usize> {
    names
        .iter()
        .enumerate()
        .filter(|(_, name)| !name.starts_with("hyperion::") && !canary::is_canary_name(name))
        .map(|(i, _)| i)
        .collect()
}

fn select_dataset_columns(ds: &Dataset, cols: &[usize]) -> Dataset {
    let n = ds.x.n;
    let mut data = Vec::with_capacity(n * cols.len());
    for &col in cols {
        data.extend_from_slice(ds.x.col(col));
    }
    Dataset {
        x: Matrix::new(data, n, cols.len()),
        y: ds.y.clone(),
        y_multi: ds.y_multi.clone(),
        names: cols.iter().map(|&j| ds.names[j].clone()).collect(),
    }
}

fn visible_base_features(search_names: &[String], cols: &[usize]) -> Vec<FeatureInfo> {
    cols.iter()
        .map(|&j| FeatureInfo {
            source: "base".into(),
            formula: search_names[j].clone(),
            expr: crate::dsl::Expr::Col(j as u16),
            fill: 0.0,
            scale_min: 0.0,
            scale_span: 1.0,
            canary: None,
            canary_refs: Vec::new(),
        })
        .collect()
}

/// F_AND_WP outlier trim: consensus of CV-loss (|OOF residual|) and the Hyperion percentile-magnitude,
/// dropping the worst `cfg.trim_outliers` fraction of rows. Falls back to pure CV-loss with no Hyperion.
fn trim_outlier_rows(
    train: &Dataset,
    be: &dyn Backend,
    cfg: &PipelineConfig,
    frac: f32,
) -> Dataset {
    let n = train.n();
    let r = oof_residual(
        &train.x,
        &train.y,
        be,
        &cfg.resid_spec,
        5,
        cfg.seed,
        cfg.task,
        cfg.n_classes,
    );
    let cv_score: Vec<f32> = r.iter().map(|v| v.abs()).collect();
    let pct_cv = to_percentile(&cv_score);
    let hyp_cols: Vec<usize> = train
        .names
        .iter()
        .enumerate()
        .filter(|(_, name)| name.starts_with("hyperion::"))
        .map(|(j, _)| j)
        .collect();
    let fused: Vec<f32> = if hyp_cols.len() >= 8 {
        let pct_h = to_percentile(&hyperion_row_magnitude(&train.x, &hyp_cols));
        (0..n).map(|i| pct_cv[i].min(pct_h[i])).collect() // consensus (AND)
    } else {
        pct_cv
    };
    let k_drop = (((n as f32) * frac).round() as usize).min(n / 5);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        fused[b]
            .partial_cmp(&fused[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut drop_row = vec![false; n];
    for &i in order.iter().take(k_drop) {
        drop_row[i] = true;
    }
    let keep: Vec<usize> = (0..n).filter(|&i| !drop_row[i]).collect();
    eprintln!(
        "[trim] dropped {k_drop}/{n} outlier rows (consensus, hyperion_fused={})",
        hyp_cols.len() >= 8
    );
    subset_dataset(train, &keep)
}

pub fn run_pipeline(
    train: &Dataset,
    hold: &Dataset,
    be: &dyn Backend,
    extra: &[&dyn Backend],
    cfg: &PipelineConfig,
    sink: &dyn Sink,
) -> PipelineResult {
    let t0 = Instant::now();

    // Opt-in outlier trimming (default off; env `JAIN_TRIM_OUTLIERS=<frac>`). Drop the worst `frac` of TRAIN
    // rows before any feature engineering, via the F_AND_WP consensus (CV-loss AND Hyperion percentile-mag).
    let trim_frac = std::env::var("JAIN_TRIM_OUTLIERS")
        .ok()
        .and_then(|s| s.trim().parse::<f32>().ok())
        .unwrap_or(0.0);
    let trimmed_owned: Dataset;
    let train: &Dataset =
        if trim_frac > 0.0 && trim_frac < 0.5 && train.y_multi.is_none() && train.n() > 200 {
            trimmed_owned = trim_outlier_rows(train, be, cfg, trim_frac);
            &trimmed_owned
        } else {
            train
        };

    let router_decision = if cfg.router.enabled {
        cfg.router
            .assets_path
            .as_ref()
            .and_then(|path| match crate::router::RouterArtifacts::load(path) {
                Ok(art) => {
                    let sketch = crate::router::sketch_dataset(train, Some(hold));
                    match crate::router::route_dataset(&art, &sketch, &cfg.router) {
                        Ok(decision) => Some(decision),
                        Err(err) => {
                            eprintln!("[router] disabled for this run: {err:#}");
                            None
                        }
                    }
                }
                Err(err) => {
                    eprintln!("[router] assets unavailable: {err:#}");
                    None
                }
            })
    } else {
        None
    };

    let search_source_train = train;
    let search_source_hold = hold;
    let visible_cols = visible_base_indices(&search_source_train.names);
    let visible_train = select_dataset_columns(search_source_train, &visible_cols);
    let visible_hold = select_dataset_columns(search_source_hold, &visible_cols);
    let visible_features = visible_base_features(&search_source_train.names, &visible_cols);
    let canary_aug = canary::append_canaries(
        search_source_train,
        search_source_hold,
        &cfg.canary,
        cfg.seed,
    );
    let search_train = &canary_aug.train;
    let search_hold = &canary_aug.hold;
    let train = &visible_train;
    let hold = &visible_hold;

    // Phase 1: residual + GP synthesis over base features.
    let r = oof_residual(
        &train.x,
        &train.y,
        be,
        &cfg.resid_spec,
        5,
        cfg.seed,
        cfg.task,
        cfg.n_classes,
    );
    let mut gd = GpData::new_with_holdout(
        search_train.x.clone(),
        search_train.names.clone(),
        r.clone(),
        &train.y,
        cfg.seed,
        cfg.gp.subsample,
        cfg.gp.holdout_ratio,
    );
    gd.canary_specs = canary_aug.specs.clone();
    // Anti-overfit CV fitness: score each candidate on K reshuffled-per-generation folds and penalize
    // cross-fold inconsistency (gain = mean - beta*std). Default ON for CLASSIFICATION only -- it is a
    // validated win there (broad sweep beats legacy GP; Elo gate binary >= legacy). For REGRESSION the
    // evidence conflicts (a broad random sweep favors GP-off, but the curated Elo coreset favors legacy
    // heavy GP), so regression keeps the legacy single-slice fitness by default to avoid regressing that
    // gate. Env `JAIN_GP_CVK` / `JAIN_GP_CVBETA` override either task.
    let (def_k, def_beta) = if matches!(cfg.task, crate::task::TaskKind::Regression) {
        (1usize, 0.0f32)
    } else {
        (3usize, 1.0f32)
    };
    gd.cv_k = std::env::var("JAIN_GP_CVK")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(def_k)
        .max(1);
    gd.cv_penalty = std::env::var("JAIN_GP_CVBETA")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(def_beta)
        .max(0.0);
    // Opt-in skip-GP heuristic. The broad 110-dataset regression sweep shows GP-off beats even the
    // CV-penalized fitness in every row bin (jain's GBDT+BMA already wins ~89% vs catboost there), so on
    // real-world regression corpora `JAIN_GP_SKIP=1` is recommended. It is NOT the default because it
    // regresses the curated Elo regression coreset. `JAIN_GP_SKIP=1/true` skips GP entirely.
    let skip_gp = matches!(
        std::env::var("JAIN_GP_SKIP").ok().as_deref(),
        Some("1") | Some("true")
    );
    let survivors = if skip_gp {
        Vec::new()
    } else {
        run_gp(&gd, &cfg.gp, sink, t0)
    };
    let n_survivors = survivors.len();

    // Augmented matrix = base + survivors. GP rejects canary-referencing expressions during survivor
    // admission after surfacing them live; build_augmented_from_search keeps the final quarantine guard.
    let Augmented {
        train: xtr_aug,
        hold: xho_aug,
        features,
    } = build_augmented_from_search(
        train,
        hold,
        search_train,
        search_hold,
        &survivors,
        cfg.max_gp,
        &visible_features,
        &canary_aug.specs,
    );
    // Leak guard: drop any augmented feature that perfectly separates the target (|AUC|>=0.995) — a leak
    // or degenerate GP feature, never legitimate signal. No-op on clean data.
    let (xtr_aug, xho_aug, features, n_leak_dropped) = drop_leak_columns(
        xtr_aug,
        xho_aug,
        features,
        &train.y,
        0.995,
        cfg.task,
        cfg.n_classes,
    );
    let n_features = features.len();

    // Phase 2: importance epochs (task-aware objective so regression/multiclass train correctly).
    let p2_obj = crate::backend::ObjectiveSpec::from_kind(cfg.task);
    let importance = importance_epochs_detailed(&xtr_aug, &train.y, be, &cfg.p2, p2_obj, sink, t0);
    let mv = importance.memory_vector;
    let starforge_top_cols =
        crate::starforge_integration::top_importance_cols(&mv, cfg.starforge.top_features);

    // Fixed folds for selection (drawn once -> CV tracks holdout). FoldScope reproduces the EXACT
    // existing split and yields a stable fold hash for the trust receipt.
    let scope = if train.n() > cfg.coreset_threshold {
        let active = guided_coreset_rows(
            &train.y,
            &r,
            cfg.select_subsample.min(train.n()),
            cfg.seed,
            cfg.coreset_guard_frac,
        );
        FoldScope::from_active_stratified(active, &train.y, cfg.eda.folds, cfg.seed)
    } else {
        FoldScope::root_stratified(&train.y, cfg.select_subsample, cfg.eda.folds, cfg.seed)
    };
    let fold_hash = scope.fold_hash;
    let folds = scope.folds;

    // Trust receipt: stable dataset/fold hashes + a pre-search leakage scan on the augmented features
    // (report-only; never alters selection).
    let feat_names: Vec<String> = features.iter().map(|f| f.formula.clone()).collect();
    let trust = TrustReport {
        dataset_hash: dataset_hash(&train.x, &train.y, &train.names),
        fold_hash,
        leakage: scan(&xtr_aug, &train.y, &feat_names, Some(&folds)),
        leak_features_dropped: n_leak_dropped,
    };

    // Phase 3: single-target EDA selection.
    let scorer = SingleScorer {
        x: &xtr_aug,
        y: &train.y,
        xh: &xho_aug,
        yh: &hold.y,
        folds: &folds,
        be,
        cv_spec: cfg.cv_spec,
        ho_spec: cfg.ho_spec,
        task: cfg.task,
        n_classes: cfg.n_classes,
    };
    let mut champ = match cfg.eda.selection_mode {
        crate::mvp::SelectionMode::Pbil => run_eda(&mv, None, &scorer, &cfg.eda, sink, 3, t0),
        crate::mvp::SelectionMode::MvpFast => {
            let mut e = Event::new(3, "start", t0.elapsed().as_secs_f64());
            e.model = Some("mvp-fast".into());
            sink.emit(&e);
            let selected = crate::mvp::reduce_features(
                &xtr_aug,
                &features,
                &importance.importance_percentile,
                &cfg.eda.mvp,
            );
            let total_steps = folds.len() as u64 + 2;
            emit_mvp_fast_progress(sink, t0, 1, total_steps, &selected);
            if selected.report.shortfall > 0 {
                let mut e = Event::new(3, "warning", t0.elapsed().as_secs_f64());
                // NB: the shortfall detail rides on the feature-count fields below, NOT on `model` — that
                // field is the model name, and overloading it with a message spawns a phantom 0.000
                // leaderboard row in the cockpit.
                e.accepted_features = Some(selected.report.accepted_gp as u64);
                e.selected_features = Some(selected.report.selected_gp as u64);
                e.final_features = Some(selected.report.final_features as u64);
                sink.emit(&e);
            }
            let cv_result = scorer.score_cv_with_progress(
                &selected.cols,
                || sink.is_cancelled(),
                |fold_done, _fold_total| {
                    emit_mvp_fast_progress(sink, t0, 1 + fold_done as u64, total_steps, &selected);
                },
            );
            match cv_result {
                Some((cv_mean, _)) if !sink.is_cancelled() => {
                    let holdout = scorer.score_holdout(&selected.cols);
                    emit_mvp_fast_progress(sink, t0, total_steps, total_steps, &selected);
                    let mut e = Event::new(3, "done", t0.elapsed().as_secs_f64());
                    e.model = Some("mvp-fast".into());
                    e.metric = Some(crate::metrics::metric_name(cfg.task).into());
                    e.best_val_auc = Some(cv_mean);
                    e.best_holdout_auc = Some(holdout);
                    e.best_size = Some(selected.cols.len() as u64);
                    e.accepted_features = Some(selected.report.accepted_gp as u64);
                    e.selected_features = Some(selected.report.selected_gp as u64);
                    e.final_features = Some(selected.report.final_features as u64);
                    sink.emit(&e);
                    Champion {
                        cols: selected.cols,
                        cv_mean_auc: cv_mean,
                        holdout_auc: holdout,
                        gen: -1,
                    }
                }
                Some((cv_mean, _)) => Champion {
                    cols: selected.cols,
                    cv_mean_auc: cv_mean,
                    holdout_auc: 0.5,
                    gen: -1,
                },
                None => Champion {
                    cols: selected.cols,
                    cv_mean_auc: 0.5,
                    holdout_auc: 0.5,
                    gen: -1,
                },
            }
        }
    };

    // Phase 3.5 (optional): strategy bake-off. Runs its own CV competition over (model × feature-set)
    // and SHIPS the CV winner as the champion; the standalone Phase-5 below is then skipped.
    let mut bakeoff_report = None;
    let mut bakeoff_ensemble = None;
    if cfg.bakeoff.enabled && extra.len() >= 2 {
        let base_cols: Vec<u32> = features
            .iter()
            .enumerate()
            .filter(|(_, f)| f.source == "base")
            .map(|(j, _)| j as u32)
            .collect();
        let reduced_cols = champ.cols.clone();
        let w = crate::bakeoff::run_bakeoff(
            be,
            extra,
            train,
            hold,
            &xtr_aug,
            &xho_aug,
            &base_cols,
            &reduced_cols,
            &importance.importance_percentile,
            &folds,
            cfg,
            sink,
            t0,
        );
        champ = Champion {
            cols: w.cols,
            cv_mean_auc: w.cv_auc,
            holdout_auc: w.test_auc,
            gen: -2,
        };
        bakeoff_ensemble = w.ensemble;
        bakeoff_report = Some(w.report);
    }

    // Preemptive cancellation: snapshot once, then guard-and-skip the remaining uninterruptible
    // single-`train_single` stages (Phase 5 / MIMO / baseline). Their outputs fall through as
    // None/0.0 to the single return below — no early-return plumbing.
    let cancelled = sink.is_cancelled();

    // Phase 6 (optional): INVENT — evolve feat-math pipeline genomes on the champion features; the
    // top-3 then get PROPER fold-pure CV on the pipeline's own folds and stream as phase-5 members
    // (leaderboard + CV heatmap). Their deliverable is emitted standalone Rust+Python source; they
    // are deliberately not fuser/GES candidates. Cancellation is honored mid-run via the sink.
    let invention = if cancelled {
        None
    } else {
        crate::invention::run_invention(
            &cfg.invention,
            &xtr_aug,
            &train.y,
            &xho_aug,
            &hold.y,
            &champ.cols,
            &folds,
            &cfg.cv_spec,
            &cfg.ho_spec,
            cfg.task,
            cfg.n_classes,
            cfg.seed,
            sink,
            t0,
        )
    };

    // Final stage (Phase 5): CV-test the champion features on CatBoost, XGBoost, LightGBM and their
    // ensemble; pick the highest-CV model. Skipped when the bake-off above already selected (it may
    // carry its own ensemble plan when the dominance strategy won).
    let ensemble = if cfg.bakeoff.enabled {
        bakeoff_ensemble
    } else if !cancelled && extra.len() >= 2 {
        let multi = train.y_multi.as_ref().is_some_and(|m| m.k > 1);
        if cfg.dominance.enabled {
            // Phase 5 ensemble-dominance engine: diverse OOF library + BBC-gated fusion.
            Some(crate::dominance::select_dominant(
                be,
                extra[0],
                extra[1],
                &xtr_aug,
                &train.y,
                &xho_aug,
                &hold.y,
                &folds,
                &champ.cols,
                &cfg.ho_spec,
                &cfg.ho_spec,
                cfg.task,
                cfg.n_classes,
                &cfg.dominance,
                sink,
                t0,
            ))
        } else if cfg.stack.enabled && multi {
            // Phase 5 multi-target STACKING: cross-target borrowing on the champion features, gated.
            let m = train.y_multi.as_ref().unwrap();
            Some(crate::stacking::select_model_stacked(
                be,
                extra[0],
                extra[1],
                &xtr_aug,
                &train.y,
                &xho_aug,
                &hold.y,
                &m.data,
                m.k,
                m.primary,
                &m.names,
                &folds,
                &champ.cols,
                &cfg.ho_spec,
                &cfg.ho_spec,
                cfg.task,
                cfg.n_classes,
                &cfg.stack,
                sink,
                t0,
            ))
        } else {
            Some(crate::ensemble::select_model(
                be,
                extra[0],
                extra[1],
                &xtr_aug,
                &train.y,
                &xho_aug,
                &hold.y,
                &folds,
                &champ.cols,
                &cfg.ho_spec,
                &cfg.ho_spec,
                cfg.task,
                cfg.n_classes,
                &cfg.starforge,
                &starforge_top_cols,
                invention.as_ref(),
                sink,
                t0,
            ))
        }
    } else {
        None
    };

    // Phase 4 (optional): MIMO warm-started from the single-target champion. Skipped under the dominance
    // engine (which subsumes fusion) to avoid a redundant native-multi pass.
    let mimo_holdout = if !cancelled && cfg.do_mimo && !cfg.dominance.enabled {
        if let Some(m) = &train.y_multi {
            let mimo = MimoScorer {
                x: &xtr_aug,
                y: &train.y,
                y_multi: &m.data,
                k: m.k,
                primary: m.primary,
                xh: &xho_aug,
                yh: &hold.y,
                folds: &folds,
                be,
                cv_spec: cfg.cv_spec,
                ho_spec: cfg.ho_spec,
                task: cfg.task,
                n_classes: cfg.n_classes,
            };
            let mc = run_eda(&mv, Some(&champ.cols), &mimo, &cfg.eda, sink, 4, t0);
            Some(mc.holdout_auc)
        } else {
            None
        }
    } else {
        None
    };

    // Lift vs baseline on raw features (skipped on cancel — it is another uninterruptible train).
    let baseline = if cancelled {
        0.0
    } else {
        baseline_holdout(train, hold, be, &cfg.ho_spec, cfg.task, cfg.n_classes)
    };

    // End-of-run payoff events for the cockpit: enriched feature importance (classified by origin) and the
    // vanilla-vs-engineered lift (both single-model on the same holdout, so the delta is features-only).
    let top_features = if !cancelled {
        emit_top_features(sink, &features, &mv, &search_train.names, t0)
    } else {
        Vec::new()
    };
    let canary_report = build_canary_report(
        &canary_aug,
        &features,
        &champ.cols,
        &search_train.names,
        &top_features,
    );
    if !cancelled {
        if canary_report.selected_canary_count > 0
            || canary_report.top_importance_contaminated_count > 0
        {
            let mut e = Event::new(5, "warning", t0.elapsed().as_secs_f64());
            e.model = Some(format!(
                "CANARY_FEATURES contamination: selected={} top_importance={}",
                canary_report.selected_canary_count,
                canary_report.top_importance_contaminated_count
            ));
            sink.emit(&e);
        }
        let mut e = Event::new(5, "done", t0.elapsed().as_secs_f64());
        e.baseline_auc = Some(baseline);
        e.best_holdout_auc = Some(champ.holdout_auc);
        e.metric = Some(crate::metrics::metric_name(cfg.task).into());
        sink.emit(&e);
    }

    PipelineResult {
        baseline_holdout_auc: baseline,
        champion_holdout_auc: champ.holdout_auc,
        champion_cv_auc: champ.cv_mean_auc,
        champion_size: champ.cols.len(),
        mimo_holdout_auc: mimo_holdout,
        n_survivors,
        n_features,
        lift: champ.holdout_auc - baseline,
        secs: t0.elapsed().as_secs_f64(),
        ensemble,
        trust,
        artifact: PipelineArtifactState {
            features,
            selected_cols: champ.cols.clone(),
        },
        router: router_decision,
        bakeoff: bakeoff_report,
        invention,
        cancelled,
        canary: canary_report,
    }
}
