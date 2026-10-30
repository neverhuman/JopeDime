//! `portfolio` — a set of member pipelines fitted on the training rows,
//! measured with out-of-fold (OOF) cross-validation predictions on those same
//! rows, and combined with a non-negative ridge (NNLS) blend in
//! deviation-from-mean space. Spec: docs/specs/portfolio_head.md (jope v35).
//!
//! Semantics (spec §4), every step mirrored bit-for-bit by the emitted
//! Python/Rust twins:
//! 1. sanitize; drop constant columns (train-fitted mask); build the shared
//!    256-component projection view when the kept width exceeds 300 columns;
//! 2. seeded shuffled K-fold OOF (K=5, or max(2, min(5, n/8)) below 50 rows;
//!    repeats 3/2/1 by n<120 / n<300 / else), every member refit per fold,
//!    per-fold member predictions clipped to the fold-train target range
//!    padded by 5%;
//! 3. NNLS by exact cyclic coordinate descent (<=500 sweeps, exit when the
//!    largest coordinate move drops under 1e-10) on G = P'P + lambda*I,
//!    lambda = 1e-3 + 4*var(y)*k/n, in deviation space (shrinkage pulls
//!    toward the mean predictor); sum(w) ~ 0 falls back to uniform;
//! 4. best-single fallback when the best member's OOF RMSE beats the blend;
//! 5. full refit of members with weight > 1e-4; broken-refit guard (train
//!    RMSE > 1.05*std(y) or non-finite replaces the member by the mean
//!    predictor); test predictions sanitized and clipped to the padded train
//!    target range, the blended output clipped again;
//! 6. degenerate guards: n<5 or std(y)~0 or all-constant X -> train mean;
//!    n<20 -> the single built-in ridge_gcv member.
//!
//! Determinism: one pinned seed drives the fold-shuffle Pcg32, streams keyed
//! by repeat; member fits are deterministic by construction (their seeds are
//! stage parameters). The fold RNG never reads dataset identity — only the
//! seed and n.

use crate::dsl::DslExpr;
use crate::members::{ResolvedMember, View};
use crate::ops::{apply_op, svd_power_fit, FittedOp};
use crate::rng::Pcg32;
use crate::{model, sanitize, GenomeIr, Rows};
use anyhow::{bail, Result};

/// One pinned seed for the fold shuffle (spec §4: fold RNG reads seed and n only).
pub const PORTFOLIO_SEED: u64 = 17;
/// Stream offset: repeat r shuffles with `Pcg32::new(PORTFOLIO_SEED, 1000 + r)`.
pub const SHUFFLE_STREAM_BASE: u64 = 1000;
/// Kept width above which the shared projection view is built for `proj` members.
pub const PROJ_WIDTH: usize = 300;
/// Components of the shared projection view.
pub const PROJ_K: usize = 256;
/// Power-iteration budget / stage-RNG seed of the projection view (the
/// `svd_project` op defaults).
pub const PROJ_ITERS: usize = 20;
pub const PROJ_SEED: u64 = 7;
/// Degenerate guards.
const EPS: f64 = 1e-12;
/// Members below this blend weight are dropped from the deployed model.
const WEIGHT_FLOOR: f64 = 1e-4;
/// Shrinkage of the duplicate-key group mean (m-estimate).
const GROUP_MEAN_SMOOTH: f64 = 5.0;

// ------------------------------- group mean --------------------------------

/// Duplicate-key shrunk group mean: rows are keyed by their values rounded to
/// nine decimals; predictions are the m-estimate-shrunk per-key train mean,
/// unknown keys fall back to the global mean. (target_bin_encode at
/// exact-match granularity — useful when duplicate rows carry the signal.)
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GroupMeanModel {
    /// (key bits, shrunk mean), sorted by key for lookup.
    pub(crate) table: Vec<(Vec<u64>, f64)>,
    pub(crate) g_mean: f64,
}

/// Round to nine decimals, half away from zero via floor(v*1e9 + 0.5), and
/// normalize -0.0 to +0.0 (`+ 0.0`) so the bit key is language-independent.
fn round9(v: f64) -> f64 {
    (v * 1e9 + 0.5).floor() / 1e9 + 0.0
}

fn row_key(row: &[f64]) -> Vec<u64> {
    row.iter().map(|v| round9(*v).to_bits()).collect()
}

impl GroupMeanModel {
    pub fn fit(x: &Rows, y: &[f64]) -> GroupMeanModel {
        let n = x.len().min(y.len());
        let mut g_mean = 0.0;
        for &target in y.iter().take(n) {
            g_mean += target;
        }
        g_mean /= n.max(1) as f64;
        let mut sums: std::collections::BTreeMap<Vec<u64>, (f64, f64)> =
            std::collections::BTreeMap::new();
        for i in 0..n {
            let e = sums.entry(row_key(&x[i])).or_insert((0.0, 0.0));
            e.0 += y[i];
            e.1 += 1.0;
        }
        let table: Vec<(Vec<u64>, f64)> = sums
            .into_iter()
            .map(|(k, (s, c))| {
                (
                    k,
                    (s + GROUP_MEAN_SMOOTH * g_mean) / (c + GROUP_MEAN_SMOOTH),
                )
            })
            .collect();
        GroupMeanModel { table, g_mean }
    }

    pub fn predict(&self, x: &Rows) -> Vec<f64> {
        x.iter()
            .map(|row| {
                let key = row_key(row);
                match self.table.binary_search_by(|(k, _)| k.cmp(&key)) {
                    Ok(i) => self.table[i].1,
                    Err(_) => self.g_mean,
                }
            })
            .collect()
    }
}

/// Fraction of training rows that are duplicates of an earlier row (keys as in
/// the group mean: values rounded to nine decimals).
pub fn dup_fraction(x: &Rows) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    let mut keys: std::collections::BTreeSet<Vec<u64>> = std::collections::BTreeSet::new();
    for row in x {
        keys.insert(row_key(row));
    }
    1.0 - keys.len() as f64 / x.len() as f64
}

// --------------------------------- members ---------------------------------

/// The internal member implementation the fold loop dispatches on.
#[derive(Clone, Debug)]
enum MemberImpl {
    Pipe(DslExpr),
    GroupMean,
    /// Test-only: a member that emits non-finite / astronomically large
    /// predictions, for the isolation negative-control (spec §6.4).
    #[cfg(test)]
    Poison,
}

struct ActiveMember {
    name: String,
    canonical: String,
    imp: MemberImpl,
    proj_view: bool,
}

/// Fit a member on rows and predict others; `None` signals a fit failure
/// (the caller substitutes the fold-train mean).
fn member_fit_predict(imp: &MemberImpl, xtr: &Rows, ytr: &[f64], xva: &Rows) -> Option<Vec<f64>> {
    match imp {
        MemberImpl::Pipe(dsl) => {
            let genome = GenomeIr::Pipeline(dsl.clone());
            match model::fit(&genome, xtr, ytr) {
                Ok(m) => m.predict(xva).ok(),
                Err(_) => None,
            }
        }
        MemberImpl::GroupMean => Some(GroupMeanModel::fit(xtr, ytr).predict(xva)),
        #[cfg(test)]
        MemberImpl::Poison => Some(
            (0..xva.len())
                .map(|i| if i % 2 == 0 { 1e300 } else { f64::NAN })
                .collect(),
        ),
    }
}

// ---------------------------------- model ----------------------------------

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum FittedMemberState {
    /// Fully fitted member pipeline.
    Fitted(Box<model::Model>),
    /// Fitted group-mean member.
    GroupMean(GroupMeanModel),
    /// Broken refit (or fit error): the member predicts the train mean, i.e.
    /// contributes zero in deviation space.
    Mean,
    /// Weight at/below the floor: never refit, contributes nothing.
    Skipped,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FittedMember {
    pub name: String,
    pub proj_view: bool,
    pub state: FittedMemberState,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MemberDiagnostics {
    pub index: usize,
    pub name: String,
    pub engine: String,
    pub canonical: String,
    pub oof_rmse: f64,
    pub nnls_weight: f64,
    pub fit_ms: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BlendDiagnostics {
    pub mode: String,
    pub folds: usize,
    pub repeats: usize,
    pub fallback_used: bool,
    pub stack_oof_rmse: f64,
    pub best_single_oof_rmse: f64,
    /// Internal r^2 of the blended OOF predictions against the training
    /// target — the no-leak negative control reads this.
    pub stack_oof_r2: f64,
    pub baseline_id: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PortfolioDiagnostics {
    pub member_diagnostics: Vec<MemberDiagnostics>,
    pub blend: BlendDiagnostics,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PortfolioModel {
    /// Degenerate path: predict the train mean everywhere.
    pub mean_only: bool,
    pub y_mean: f64,
    /// Fit-time row width (rows are padded/truncated to it at predict).
    pub width: usize,
    /// Kept (non-constant) column indices.
    pub keep: Vec<usize>,
    /// Shared projection view for wide data (members flagged proj).
    pub proj: Option<FittedOp>,
    pub members: Vec<FittedMember>,
    pub weights: Vec<f64>,
    pub clip_lo: f64,
    pub clip_hi: f64,
    pub diagnostics: PortfolioDiagnostics,
}

fn mean_only_model(y_mean: f64, width: usize, folds: usize, repeats: usize) -> PortfolioModel {
    PortfolioModel {
        mean_only: true,
        y_mean,
        width,
        keep: Vec::new(),
        proj: None,
        members: Vec::new(),
        weights: Vec::new(),
        clip_lo: y_mean,
        clip_hi: y_mean,
        diagnostics: PortfolioDiagnostics {
            member_diagnostics: Vec::new(),
            blend: BlendDiagnostics {
                mode: "mean".to_string(),
                folds,
                repeats,
                fallback_used: false,
                stack_oof_rmse: f64::NAN,
                best_single_oof_rmse: f64::NAN,
                stack_oof_r2: 0.0,
                baseline_id: crate::members::REGISTRY_VERSION.to_string(),
            },
        },
    }
}

/// Exact cyclic coordinate descent for min ½w'Gw − c'w s.t. w ≥ 0
/// (≤500 sweeps; exit when the largest coordinate move drops under 1e-10).
pub fn nnls_cd(g: &[f64], c: &[f64], k: usize) -> Vec<f64> {
    let mut w = vec![0.0; k];
    for _ in 0..500 {
        let mut max_d = 0.0;
        for j in 0..k {
            let gjj = g[j * k + j];
            if gjj <= EPS {
                continue;
            }
            let mut acc = 0.0;
            for l in 0..k {
                acc += g[j * k + l] * w[l];
            }
            let mut nj = w[j] + (c[j] - acc) / gjj;
            if nj < 0.0 {
                nj = 0.0;
            }
            let d = (nj - w[j]).abs();
            if d > 0.0 {
                w[j] = nj;
                if d > max_d {
                    max_d = d;
                }
            }
        }
        if max_d < 1e-10 {
            break;
        }
    }
    w
}

fn gather(x: &Rows, idx: &[usize]) -> Rows {
    idx.iter().map(|&r| x[r].clone()).collect()
}

/// Fit a portfolio per spec §4. `roster` is the pre-gating member list
/// (registry expansion for the compact head, raw member pipes for the
/// explicit form); `folds_param` is pinned to 5 by both grammars.
pub fn fit_portfolio(
    roster: &[ResolvedMember],
    x: &Rows,
    y: &[f64],
    folds_param: usize,
    repeats_auto: bool,
) -> Result<PortfolioModel> {
    let n = x.len().min(y.len());
    if n == 0 {
        return Ok(mean_only_model(0.0, 0, folds_param, 1));
    }
    // Sanitize a working copy; normalize row width to the first row's.
    let width = x[0].len();
    let mut xs: Rows = x[..n]
        .iter()
        .map(|row| {
            let mut r = row.clone();
            r.resize(width, 0.0);
            r
        })
        .collect();
    sanitize(&mut xs);
    let ys: Vec<f64> = y[..n]
        .iter()
        .map(|v| if v.is_finite() { *v } else { 0.0 })
        .collect();

    let mut y_mean = 0.0;
    for v in &ys {
        y_mean += v;
    }
    y_mean /= n as f64;
    let mut y_var = 0.0;
    for v in &ys {
        let d = v - y_mean;
        y_var += d * d;
    }
    y_var /= n as f64;
    let y_std = y_var.sqrt();

    // Degenerate guards: tiny n or constant target -> train mean.
    if n < 5 || y_std < EPS {
        return Ok(mean_only_model(y_mean, width, folds_param, 1));
    }

    // Constant-column drop (train-fitted mask; population std > EPS keeps).
    let mut keep: Vec<usize> = Vec::new();
    for j in 0..width {
        let mut m = 0.0;
        for row in &xs {
            m += row[j];
        }
        m /= n as f64;
        let mut v = 0.0;
        for row in &xs {
            let d = row[j] - m;
            v += d * d;
        }
        if (v / n as f64).sqrt() > EPS {
            keep.push(j);
        }
    }
    if keep.is_empty() {
        return Ok(mean_only_model(y_mean, width, folds_param, 1));
    }
    let x_raw: Rows = xs
        .iter()
        .map(|row| keep.iter().map(|&j| row[j]).collect())
        .collect();
    let p = keep.len();

    // Member gating on train stats only (n, p, dup_frac); n<20 falls back to
    // the single built-in ridge_gcv member (spec §4 step 6). A roster gated
    // to nothing (possible for preset=none adds) also falls back rather than
    // failing.
    let dup = dup_fraction(&x_raw);
    let small_fb = crate::members::small_n_fallback();
    let gated: Vec<&ResolvedMember> = if n < 20 {
        vec![&small_fb]
    } else {
        roster.iter().filter(|m| m.gate.passes(n, p, dup)).collect()
    };
    let gated = if gated.is_empty() {
        vec![&small_fb]
    } else {
        gated
    };
    #[allow(unused_mut)]
    let mut active: Vec<ActiveMember> = gated
        .iter()
        .map(|m| ActiveMember {
            name: m.name.clone(),
            canonical: m.canonical(),
            imp: match &m.pipe {
                Some(dsl) => MemberImpl::Pipe(dsl.clone()),
                None => MemberImpl::GroupMean,
            },
            proj_view: m.view == View::Proj,
        })
        .collect();
    #[cfg(test)]
    POISON_IMPLS.with(|cell| {
        if let Some(impls) = cell.borrow().as_ref() {
            for (i, imp) in impls.iter().enumerate() {
                if i < active.len() {
                    active[i].imp = imp.clone();
                }
            }
        }
    });
    let m = active.len();
    if m > crate::members::MAX_ACTIVE_MEMBERS {
        bail!(
            "portfolio: {m} active members exceeds the cap of {}",
            crate::members::MAX_ACTIVE_MEMBERS
        );
    }

    // Shared projection view for wide data.
    let need_proj = p > PROJ_WIDTH && active.iter().any(|a| a.proj_view);
    let proj: Option<FittedOp> = if need_proj {
        let k = PROJ_K.min(p);
        let (means, components) = svd_power_fit(&x_raw, k, PROJ_ITERS, PROJ_SEED);
        Some(FittedOp::SvdProject {
            means,
            components,
            append: false,
        })
    } else {
        None
    };
    let x_proj: Rows = match &proj {
        Some(op) => apply_op(op, &x_raw)?,
        None => Vec::new(),
    };

    // Fold plan.
    let k_folds = if n >= 50 {
        folds_param
    } else {
        (n / 8).clamp(2, 5)
    };
    let repeats = if repeats_auto {
        if n < 120 {
            3
        } else if n < 300 {
            2
        } else {
            1
        }
    } else {
        1
    };

    // OOF: seeded shuffled K-fold, members refit per fold, per-fold clip to
    // the padded fold-train target range.
    let rows_total = repeats * n;
    let mut oof = vec![vec![0.0f64; m]; rows_total];
    let mut fit_ms = vec![0.0f64; m];
    for rep in 0..repeats {
        let mut idx: Vec<usize> = (0..n).collect();
        let mut rng = Pcg32::new(PORTFOLIO_SEED, SHUFFLE_STREAM_BASE + rep as u64);
        for i in (1..n).rev() {
            let j = (rng.next_u32() as usize) % (i + 1);
            idx.swap(i, j);
        }
        for f in 0..k_folds {
            let lo = f * n / k_folds;
            let hi = (f + 1) * n / k_folds;
            if lo >= hi {
                continue;
            }
            let va: Vec<usize> = idx[lo..hi].to_vec();
            let mut in_va = vec![false; n];
            for &r in &va {
                in_va[r] = true;
            }
            let tr: Vec<usize> = (0..n).filter(|r| !in_va[*r]).collect();
            let ytr: Vec<f64> = tr.iter().map(|&r| ys[r]).collect();
            let mut tr_mean = 0.0;
            for v in &ytr {
                tr_mean += v;
            }
            tr_mean /= ytr.len() as f64;
            let mut f_lo = ytr[0];
            let mut f_hi = ytr[0];
            for v in &ytr {
                if *v < f_lo {
                    f_lo = *v;
                }
                if *v > f_hi {
                    f_hi = *v;
                }
            }
            let f_pad = 0.05 * (f_hi - f_lo).max(EPS);
            let xtr_raw = gather(&x_raw, &tr);
            let xva_raw = gather(&x_raw, &va);
            let (xtr_proj, xva_proj) = if need_proj {
                (gather(&x_proj, &tr), gather(&x_proj, &va))
            } else {
                (Vec::new(), Vec::new())
            };
            for (mi, member) in active.iter().enumerate() {
                let use_proj = need_proj && member.proj_view;
                let (xtr_v, xva_v) = if use_proj {
                    (&xtr_proj, &xva_proj)
                } else {
                    (&xtr_raw, &xva_raw)
                };
                let t0 = std::time::Instant::now();
                let preds = member_fit_predict(&member.imp, xtr_v, &ytr, xva_v);
                fit_ms[mi] += t0.elapsed().as_secs_f64() * 1000.0;
                match preds {
                    Some(pm) => {
                        for (t, &r) in va.iter().enumerate() {
                            let mut v = pm[t];
                            if !v.is_finite() {
                                v = tr_mean;
                            }
                            if v < f_lo - f_pad {
                                v = f_lo - f_pad;
                            }
                            if v > f_hi + f_pad {
                                v = f_hi + f_pad;
                            }
                            oof[rep * n + r][mi] = v;
                        }
                    }
                    None => {
                        for &r in &va {
                            oof[rep * n + r][mi] = tr_mean;
                        }
                    }
                }
            }
        }
    }

    // NNLS blend in deviation-from-mean space.
    let ybar = y_mean;
    let nr = rows_total as f64;
    let lam = 1e-3 + 4.0 * y_var * m as f64 / rows_total.max(1) as f64;
    let mut g = vec![0.0f64; m * m];
    let mut c = vec![0.0f64; m];
    for i in 0..rows_total {
        let ti = ys[i % n] - ybar;
        for a in 0..m {
            let pa = oof[i][a] - ybar;
            c[a] += pa * ti;
            for b in a..m {
                g[a * m + b] += pa * (oof[i][b] - ybar);
            }
        }
    }
    for a in 0..m {
        g[a * m + a] += lam;
        for b in 0..a {
            g[a * m + b] = g[b * m + a];
        }
    }
    let mut w = nnls_cd(&g, &c, m);
    let mut wsum = 0.0;
    for v in &w {
        wsum += v;
    }
    if wsum <= EPS {
        w = vec![1.0 / m as f64; m];
    }

    // Best-single fallback: deploy the best member alone when the blend loses
    // on OOF RMSE.
    let mut stack_sse = 0.0;
    let mut mem_sse = vec![0.0f64; m];
    for i in 0..rows_total {
        let ti = ys[i % n];
        let mut z = ybar;
        for a in 0..m {
            z += (oof[i][a] - ybar) * w[a];
        }
        let d = z - ti;
        stack_sse += d * d;
        for a in 0..m {
            let d = oof[i][a] - ti;
            mem_sse[a] += d * d;
        }
    }
    let stack_rmse = (stack_sse / nr).sqrt();
    let member_rmse: Vec<f64> = mem_sse.iter().map(|s| (s / nr).sqrt()).collect();
    let mut best = 0;
    for a in 1..m {
        if member_rmse[a] < member_rmse[best] {
            best = a;
        }
    }
    let fallback_used = member_rmse[best] < stack_rmse - 1e-9;
    if fallback_used {
        for (a, weight) in w.iter_mut().enumerate().take(m) {
            *weight = if a == best { 1.0 } else { 0.0 };
        }
    }
    // Internal stack r^2 against the stacked training target (diagnostics +
    // the no-leak negative control).
    let mut sst = 0.0;
    for i in 0..rows_total {
        let d = ys[i % n] - ybar;
        sst += d * d;
    }
    let final_stack_rmse = if fallback_used {
        member_rmse[best]
    } else {
        stack_rmse
    };
    let stack_oof_r2 = if sst > 0.0 {
        1.0 - (final_stack_rmse * final_stack_rmse * nr) / sst
    } else {
        0.0
    };

    // Full refits + broken-refit guard; padded train-range clip parameters.
    let mut t_lo = ys[0];
    let mut t_hi = ys[0];
    for v in &ys {
        if *v < t_lo {
            t_lo = *v;
        }
        if *v > t_hi {
            t_hi = *v;
        }
    }
    let pad = 0.05 * (t_hi - t_lo).max(EPS);
    let clip_lo = t_lo - pad;
    let clip_hi = t_hi + pad;
    let mut fitted: Vec<FittedMember> = Vec::with_capacity(m);
    for (mi, member) in active.iter().enumerate() {
        if w[mi] <= WEIGHT_FLOOR {
            w[mi] = 0.0;
            fitted.push(FittedMember {
                name: member.name.clone(),
                proj_view: member.proj_view,
                state: FittedMemberState::Skipped,
            });
            continue;
        }
        let use_proj = need_proj && member.proj_view;
        let xv = if use_proj { &x_proj } else { &x_raw };
        let t0 = std::time::Instant::now();
        let state = match &member.imp {
            MemberImpl::Pipe(dsl) => {
                let genome = GenomeIr::Pipeline(dsl.clone());
                match model::fit(&genome, xv, &ys) {
                    Ok(fm) => {
                        let ptr = fm.predict(xv)?;
                        let mut sse = 0.0;
                        for i in 0..n {
                            let d = ptr[i] - ys[i];
                            sse += d * d;
                        }
                        let tr_rmse = (sse / n as f64).sqrt();
                        if !tr_rmse.is_finite() || tr_rmse > 1.05 * y_std {
                            FittedMemberState::Mean
                        } else {
                            FittedMemberState::Fitted(Box::new(fm))
                        }
                    }
                    Err(_) => FittedMemberState::Mean,
                }
            }
            MemberImpl::GroupMean => {
                let gm = GroupMeanModel::fit(xv, &ys);
                let ptr = gm.predict(xv);
                let mut sse = 0.0;
                for i in 0..n {
                    let d = ptr[i] - ys[i];
                    sse += d * d;
                }
                let tr_rmse = (sse / n as f64).sqrt();
                if !tr_rmse.is_finite() || tr_rmse > 1.05 * y_std {
                    FittedMemberState::Mean
                } else {
                    FittedMemberState::GroupMean(gm)
                }
            }
            #[cfg(test)]
            MemberImpl::Poison => FittedMemberState::Mean,
        };
        fit_ms[mi] += t0.elapsed().as_secs_f64() * 1000.0;
        fitted.push(FittedMember {
            name: member.name.clone(),
            proj_view: member.proj_view,
            state,
        });
    }

    let member_diagnostics: Vec<MemberDiagnostics> = active
        .iter()
        .enumerate()
        .map(|(mi, member)| MemberDiagnostics {
            index: mi,
            name: member.name.clone(),
            engine: "featmath".to_string(),
            canonical: member.canonical.clone(),
            oof_rmse: member_rmse[mi],
            nnls_weight: w[mi],
            fit_ms: fit_ms[mi],
        })
        .collect();

    Ok(PortfolioModel {
        mean_only: false,
        y_mean,
        width,
        keep,
        proj,
        members: fitted,
        weights: w,
        clip_lo,
        clip_hi,
        diagnostics: PortfolioDiagnostics {
            member_diagnostics,
            blend: BlendDiagnostics {
                mode: "nnls".to_string(),
                folds: k_folds,
                repeats,
                fallback_used,
                stack_oof_rmse: stack_rmse,
                best_single_oof_rmse: member_rmse[best],
                stack_oof_r2,
                baseline_id: crate::members::REGISTRY_VERSION.to_string(),
            },
        },
    })
}

impl PortfolioModel {
    pub fn predict(
        &self,
        x: &Rows,
    ) -> std::result::Result<Vec<f64>, crate::kernel::width::WidthMismatch> {
        crate::kernel::width::reject_zero_width_rows(x)?;
        let nt = x.len();
        if self.mean_only {
            return Ok(vec![self.y_mean; nt]);
        }
        let mut xs: Rows = x
            .iter()
            .map(|row| {
                let mut r = row.clone();
                r.resize(self.width, 0.0);
                r
            })
            .collect();
        sanitize(&mut xs);
        let x_raw: Rows = xs
            .iter()
            .map(|row| self.keep.iter().map(|&j| row[j]).collect())
            .collect();
        let x_proj: Rows = match &self.proj {
            Some(op) => apply_op(op, &x_raw)?,
            None => Vec::new(),
        };
        let ybar = self.y_mean;
        let mut acc = vec![ybar; nt];
        for (mi, member) in self.members.iter().enumerate() {
            let wi = self.weights[mi];
            if wi <= 0.0 {
                continue;
            }
            let preds: Vec<f64> = match &member.state {
                FittedMemberState::Fitted(fm) => {
                    let xv = if member.proj_view && self.proj.is_some() {
                        &x_proj
                    } else {
                        &x_raw
                    };
                    fm.predict(xv)?
                }
                FittedMemberState::GroupMean(gm) => {
                    let xv = if member.proj_view && self.proj.is_some() {
                        &x_proj
                    } else {
                        &x_raw
                    };
                    gm.predict(xv)
                }
                // Mean predictor: zero contribution in deviation space.
                FittedMemberState::Mean | FittedMemberState::Skipped => continue,
            };
            for i in 0..nt {
                let mut v = preds[i];
                if !v.is_finite() {
                    v = ybar;
                }
                if v < self.clip_lo {
                    v = self.clip_lo;
                }
                if v > self.clip_hi {
                    v = self.clip_hi;
                }
                acc[i] += wi * (v - ybar);
            }
        }
        for z in acc.iter_mut() {
            let mut v = *z;
            if !v.is_finite() {
                v = ybar;
            }
            if v < self.clip_lo {
                v = self.clip_lo;
            }
            if v > self.clip_hi {
                v = self.clip_hi;
            }
            *z = v;
        }
        Ok(acc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::members::{resolve_compact, Gate, ResolvedMember};
    use crate::metrics::r2;

    fn toy(n: usize) -> (Rows, Vec<f64>) {
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut rng = Pcg32::new(1, 1);
        for _ in 0..n {
            let a = rng.next_f64();
            let b = rng.next_f64();
            let c = rng.next_f64();
            x.push(vec![a, b, c]);
            y.push(2.0 * a - b + 0.3 * (6.0 * c).sin());
        }
        (x, y)
    }

    fn core_roster() -> Vec<ResolvedMember> {
        resolve_compact(&crate::dsl::Stage::new("portfolio")).unwrap()
    }

    fn pipe_member(name: &str, src: &str) -> ResolvedMember {
        ResolvedMember::from_pipe(name.to_string(), crate::dsl::parse_dsl(src).unwrap())
    }

    #[test]
    fn portfolio_determinism() {
        let (x, y) = toy(90);
        let roster = vec![
            pipe_member("a", "(pipe X (ridge_gcv))"),
            pipe_member("b", "(pipe X (ksmooth k=8))"),
            pipe_member("c", "(pipe X (histgbm2 leaves=7 rounds=60))"),
        ];
        let m1 = fit_portfolio(&roster, &x, &y, 5, true).unwrap();
        let m2 = fit_portfolio(&roster, &x, &y, 5, true).unwrap();
        assert_eq!(m1.weights, m2.weights, "weights must be bit-identical");
        let p1 = m1.predict(&x).unwrap();
        let p2 = m2.predict(&x).unwrap();
        assert_eq!(p1, p2, "predictions must be bit-identical");
        assert!(p1.iter().all(|v| v.is_finite()));
        // Diagnostics carry the schema fields.
        assert_eq!(m1.diagnostics.member_diagnostics.len(), 3);
        assert_eq!(m1.diagnostics.blend.mode, "nnls");
        assert_eq!(m1.diagnostics.blend.repeats, 3);
    }

    #[test]
    fn portfolio_noise_target_no_leak_internal() {
        // Pure-noise target: the internal stack OOF r^2 must stay under 0.15
        // (spec §6.1's inner condition; the outer cv_r2 check lives in
        // model.rs::portfolio_noise_target_no_leak).
        let (x, _) = toy(150);
        let mut rng = Pcg32::new(99, 5);
        let y: Vec<f64> = (0..x.len()).map(|_| rng.next_f64()).collect();
        let roster = vec![
            pipe_member("a", "(pipe X (ridge_gcv))"),
            pipe_member("b", "(pipe X (target_bin_encode bins=8) (ridge))"),
            pipe_member("c", "(pipe X (ksmooth k=8))"),
        ];
        let m = fit_portfolio(&roster, &x, &y, 5, true).unwrap();
        assert!(
            m.diagnostics.blend.stack_oof_r2 < 0.15,
            "leak? stack_oof_r2={}",
            m.diagnostics.blend.stack_oof_r2
        );
    }

    #[test]
    fn portfolio_inner_oof_target_ops() {
        // A target-aware member (target_bin_encode) is refit per fold: on a
        // pure-noise target its measured OOF RMSE cannot beat the target's
        // own spread (a leaked fit-once encoding would look far better on
        // its own training rows).
        let (x, _) = toy(120);
        let mut rng = Pcg32::new(7, 3);
        let y: Vec<f64> = (0..x.len()).map(|_| rng.next_f64()).collect();
        let roster = vec![pipe_member(
            "tbe",
            "(pipe X (target_bin_encode bins=8) (ridge))",
        )];
        let m = fit_portfolio(&roster, &x, &y, 5, true).unwrap();
        let d = &m.diagnostics.member_diagnostics[0];
        let mut mean = 0.0;
        for v in &y {
            mean += v;
        }
        mean /= y.len() as f64;
        let mut var = 0.0;
        for v in &y {
            let e = v - mean;
            var += e * e;
        }
        let y_std = (var / y.len() as f64).sqrt();
        assert!(
            d.oof_rmse > 0.85 * y_std,
            "target-aware member looks leaky: oof_rmse={} vs y_std={}",
            d.oof_rmse,
            y_std
        );
        // Leak twin: the SAME member fitted once on all rows and scored on
        // them (what a fold-free implementation would measure) is far
        // better — proving the OOF path is what kept the estimate honest.
        let g = GenomeIr::Pipeline(
            crate::dsl::parse_dsl("(pipe X (target_bin_encode bins=8) (ridge))").unwrap(),
        );
        let fm = model::fit(&g, &x, &y).unwrap();
        let ptr = fm.predict(&x).unwrap();
        let mut sse = 0.0;
        for i in 0..y.len() {
            let d2 = ptr[i] - y[i];
            sse += d2 * d2;
        }
        let leak_rmse = (sse / y.len() as f64).sqrt();
        assert!(
            leak_rmse < d.oof_rmse,
            "leak twin should look (spuriously) better: {leak_rmse} vs {}",
            d.oof_rmse
        );
    }

    #[test]
    fn portfolio_member_isolation() {
        // A member forced to emit 1e300/NaN cannot poison the blend: the
        // per-fold clip bounds its OOF column, the NNLS weight collapses to
        // ~0, and the deployed blend stays finite inside the clip range.
        let (x, y) = toy(100);
        let model = fit_portfolio_with_poison(&x, &y);
        let poison_w = model.weights[1];
        assert!(
            poison_w <= WEIGHT_FLOOR || matches!(model.members[1].state, FittedMemberState::Mean),
            "poisoned member kept weight {poison_w} and state {:?}",
            model.members[1].state
        );
        let preds = model.predict(&x).unwrap();
        assert!(preds.iter().all(|v| v.is_finite()));
        assert!(preds
            .iter()
            .all(|v| *v >= model.clip_lo - 1e-12 && *v <= model.clip_hi + 1e-12));
        // The good member still carries the blend.
        assert!(r2(&y, &preds) > 0.5, "r2={}", r2(&y, &preds));
    }

    /// Timing deliverable (run explicitly): full core_v1 fit + predict on a
    /// ~700x18 dataset, single-threaded. Target <= 60 s.
    /// cargo test -p feat-math --release portfolio_timing -- --ignored --nocapture --test-threads=1
    #[test]
    #[ignore]
    fn portfolio_timing_700x18() {
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut rng = Pcg32::new(42, 7);
        for _ in 0..700 {
            let row: Vec<f64> = (0..18).map(|_| rng.next_f64() * 4.0 - 2.0).collect();
            let t = 1.7 * row[0] - 0.9 * row[1]
                + 0.5 * (3.0 * row[2]).sin()
                + 0.3 * row[3] * row[4]
                + 0.1 * rng.gaussian();
            x.push(row);
            y.push(t);
        }
        let t0 = std::time::Instant::now();
        let m = fit_portfolio(&core_roster(), &x, &y, 5, true).unwrap();
        let fit_s = t0.elapsed().as_secs_f64();
        let t1 = std::time::Instant::now();
        let preds = m.predict(&x).unwrap();
        let pred_s = t1.elapsed().as_secs_f64();
        println!(
            "portfolio timing 700x18: fit {fit_s:.1}s predict {pred_s:.2}s r2={:.4} members={}",
            r2(&y, &preds),
            m.members.len()
        );
        for d in &m.diagnostics.member_diagnostics {
            println!(
                "  member[{}] {:<14} oof_rmse={:.4} w={:.4} fit_ms={:.0}",
                d.index, d.name, d.oof_rmse, d.nnls_weight, d.fit_ms
            );
        }
        assert!(fit_s + pred_s <= 60.0, "timing gate: {fit_s:.1}s");
    }

    /// Test-only harness: fit a two-member portfolio where the second member
    /// is the Poison implementation (emits 1e300/NaN before clipping). The
    /// production fold/clip/blend path runs end-to-end; only the member's
    /// prediction function is swapped via the POISON_IMPLS hook.
    fn fit_portfolio_with_poison(x: &Rows, y: &[f64]) -> PortfolioModel {
        let roster = vec![
            pipe_member("good", "(pipe X (ridge_gcv))"),
            pipe_member("poison", "(pipe X (ridge))"),
        ];
        POISON_IMPLS.with(|cell| {
            *cell.borrow_mut() = Some(vec![
                MemberImpl::Pipe(roster[0].pipe.clone().unwrap()),
                MemberImpl::Poison,
            ]);
        });
        let out = fit_portfolio(&roster, x, y, 5, true).unwrap();
        POISON_IMPLS.with(|cell| *cell.borrow_mut() = None);
        out
    }

    #[test]
    fn portfolio_best_single_fallback() {
        // One member interpolates a noiseless linear target almost exactly;
        // the other five are junk. The n/k-scaled shrinkage biases the blend
        // toward the mean predictor by ~0.5% of std(y) — orders of magnitude
        // above the clean member's OOF RMSE — so the blend strictly loses on
        // OOF and deployment is one-hot on the dominant member.
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut rng = Pcg32::new(3, 9);
        for _ in 0..24 {
            let a = rng.next_f64();
            let b = rng.next_f64();
            x.push(vec![a, b]);
            y.push(3.0 * a - 1.5 * b);
        }
        let mut roster = vec![pipe_member("lin", "(pipe X (ridge lambda_frac=0.0))")];
        for (i, freq) in [997, 991, 983, 977, 971].iter().enumerate() {
            roster.push(pipe_member(
                &format!("junk{i}"),
                &format!("(pipe X (map expr=\"sin(x*{freq})\") (ridge))"),
            ));
        }
        let m = fit_portfolio(&roster, &x, &y, 5, true).unwrap();
        let w = &m.weights;
        assert!(
            m.diagnostics.blend.fallback_used,
            "expected the best-single fallback to fire; weights={w:?} \
             stack={} best={}",
            m.diagnostics.blend.stack_oof_rmse, m.diagnostics.blend.best_single_oof_rmse
        );
        assert_eq!(w[0], 1.0);
        assert!(w[1..].iter().all(|v| *v == 0.0));
        // The deployed one-hot model reproduces the member's predictions.
        let preds = m.predict(&x).unwrap();
        assert!(r2(&y, &preds) > 0.999, "r2={}", r2(&y, &preds));
    }

    #[test]
    fn nnls_correctness() {
        // Interior solution: matches the closed-form SPD solve when the
        // unconstrained optimum is non-negative.
        let g = [4.0, 1.0, 1.0, 3.0];
        // Choose c so w* = (1, 2): c = G w* = (6, 7).
        let c = [6.0, 7.0];
        let w = nnls_cd(&g, &c, 2);
        assert!(
            (w[0] - 1.0).abs() < 1e-8 && (w[1] - 2.0).abs() < 1e-8,
            "{w:?}"
        );
        // Active constraint: unconstrained optimum has a negative coordinate;
        // CD must satisfy w>=0 and the KKT conditions.
        let g = [4.0, 3.9, 3.9, 4.0];
        let c = [1.0, -1.0];
        let w = nnls_cd(&g, &c, 2);
        assert!(w.iter().all(|v| *v >= 0.0), "{w:?}");
        assert!((w[1]).abs() < 1e-12, "constrained coord must be 0: {w:?}");
        // KKT: gradient of active coordinate >= 0, of free coordinate == 0.
        let grad0 = g[0] * w[0] + g[1] * w[1] - c[0];
        let grad1 = g[2] * w[0] + g[3] * w[1] - c[1];
        assert!(grad0.abs() < 1e-8, "free coord gradient {grad0}");
        assert!(grad1 >= -1e-10, "active coord gradient {grad1}");
        // Degenerate diagonal is skipped, not divided by.
        let g = [0.0, 0.0, 0.0, 2.0];
        let c = [5.0, 2.0];
        let w = nnls_cd(&g, &c, 2);
        assert_eq!(w[0], 0.0);
        assert!((w[1] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn portfolio_small_n_fallbacks() {
        let (x_all, y_all) = toy(30);
        // n=3: train-mean model.
        let m = fit_portfolio(&core_roster(), &x_all[..3].to_vec(), &y_all[..3], 5, true).unwrap();
        assert!(m.mean_only);
        let p = m.predict(&x_all[..5].to_vec()).unwrap();
        let mean3 = (y_all[0] + y_all[1] + y_all[2]) / 3.0;
        assert!(p.iter().all(|v| (*v - mean3).abs() < 1e-12));
        // n=8 and n=19: single built-in ridge_gcv member.
        for n in [8usize, 19] {
            let m =
                fit_portfolio(&core_roster(), &x_all[..n].to_vec(), &y_all[..n], 5, true).unwrap();
            assert!(!m.mean_only);
            assert_eq!(m.members.len(), 1, "n={n}");
            assert_eq!(m.diagnostics.member_diagnostics[0].name, "ridge_gcv_raw");
            assert!(m.diagnostics.blend.folds <= 2, "n={n} tiny-n folds");
            let p = m.predict(&x_all[..n].to_vec()).unwrap();
            assert!(p.iter().all(|v| v.is_finite()));
        }
        // n=25: gated core roster (well over the single-member fallback),
        // folds = max(2, min(5, 25/8)) = 3.
        let m =
            fit_portfolio(&core_roster(), &x_all[..25].to_vec(), &y_all[..25], 5, true).unwrap();
        assert!(m.members.len() >= 8, "n=25 roster: {}", m.members.len());
        assert_eq!(m.diagnostics.blend.folds, 3);
        assert_eq!(m.diagnostics.blend.repeats, 3);
        // Constant target: mean model.
        let yc = vec![2.5; 30];
        let m = fit_portfolio(&core_roster(), &x_all, &yc, 5, true).unwrap();
        assert!(m.mean_only);
        // All-constant X: mean model.
        let xc: Rows = (0..30).map(|_| vec![1.0, 1.0]).collect();
        let m = fit_portfolio(&core_roster(), &xc, &y_all, 5, true).unwrap();
        assert!(m.mean_only);
    }

    #[test]
    fn portfolio_proj_view_on_wide_data() {
        // p=310 kept columns triggers the shared projection view for proj
        // members; the fit stays finite and deterministic.
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut rng = Pcg32::new(11, 2);
        for _ in 0..60 {
            let row: Vec<f64> = (0..310).map(|_| rng.next_f64()).collect();
            y.push(row[0] * 2.0 - row[1] + 0.1 * row[2]);
            x.push(row);
        }
        let roster = vec![pipe_member("lin_proj", "(pipe X (ridge_gcv))"), {
            let mut m = pipe_member("lin_raw", "(pipe X (ridge))");
            m.view = View::Raw;
            m
        }];
        let mut roster = roster;
        roster[0].view = View::Proj;
        let m = fit_portfolio(&roster, &x, &y, 5, true).unwrap();
        assert!(m.proj.is_some(), "projection view must be built for p>300");
        let p = m.predict(&x).unwrap();
        assert!(p.iter().all(|v| v.is_finite()));
        let m2 = fit_portfolio(&roster, &x, &y, 5, true).unwrap();
        assert_eq!(p, m2.predict(&x).unwrap());
    }

    #[test]
    fn group_mean_member_on_duplicate_data() {
        // Duplicate rows carry the signal: the group-mean member earns its
        // keep and the model routes unknown rows to the global mean.
        let mut x = Vec::new();
        let mut y = Vec::new();
        for i in 0..120 {
            let k = (i % 6) as f64;
            x.push(vec![k, k * 2.0]);
            y.push(k * 10.0 + if i % 2 == 0 { 0.5 } else { -0.5 });
        }
        assert!(dup_fraction(&x) > 0.3);
        let roster = vec![
            pipe_member("lin", "(pipe X (ridge))"),
            ResolvedMember {
                name: "group_mean".to_string(),
                pipe: None,
                gate: Gate::DupFracAbove(0.3),
                view: View::Raw,
            },
        ];
        let m = fit_portfolio(&roster, &x, &y, 5, true).unwrap();
        assert_eq!(m.members.len(), 2);
        let preds = m.predict(&x).unwrap();
        assert!(r2(&y, &preds) > 0.9, "r2={}", r2(&y, &preds));
        // Unknown key falls back near the global mean (clipped range).
        let far = m.predict(&vec![vec![99.0, 42.0]]).unwrap();
        assert!(far[0].is_finite());
    }

    #[test]
    fn portfolio_beats_or_matches_best_single_on_toy() {
        // stack.rs precedent (spec §6.6): the measured blend of a small
        // roster matches or beats each member fitted alone, tolerance 0.01
        // OOF r^2, judged by the same honest outer CV.
        let (x, y) = toy(160);
        let singles = [
            "(pipe X (ridge_gcv))",
            "(pipe X (ksmooth k=8))",
            "(pipe X (histgbm2 leaves=7 rounds=80 bags=1))",
        ];
        let mut best = f64::NEG_INFINITY;
        for src in singles {
            let rep =
                crate::model::cv_r2(&GenomeIr::parse(src).unwrap(), &x, &y, 3, "toy_portfolio")
                    .unwrap();
            if rep.oof_r2 > best {
                best = rep.oof_r2;
            }
        }
        let mut genome_src = String::from("(portfolio");
        for src in singles {
            genome_src.push_str(&format!(" (member {src})"));
        }
        genome_src.push_str(" (blend))");
        let rep = crate::model::cv_r2(
            &GenomeIr::parse(&genome_src).unwrap(),
            &x,
            &y,
            3,
            "toy_portfolio",
        )
        .unwrap();
        println!(
            "accuracy[toy cv oof_r2] portfolio = {} (best single {best})",
            rep.oof_r2
        );
        assert!(
            rep.oof_r2 >= best - 0.01,
            "portfolio oof={} best single={best}",
            rep.oof_r2
        );
    }
}

#[cfg(test)]
thread_local! {
    /// Test hook: when set, fit_portfolio swaps member implementations by
    /// position (used by the member-isolation negative control).
    static POISON_IMPLS: std::cell::RefCell<Option<Vec<MemberImpl>>> =
        const { std::cell::RefCell::new(None) };
}
