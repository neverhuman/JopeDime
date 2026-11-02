use super::*;

mod design;
mod fit;
mod fitting;

/// Contextual + program-interaction ridge surrogate (`GB-09` / `J9`) trained on dense GP trajectories.
///
/// Combines unpenalized intercept, inner-CV cross-fitted `RidgeProgram` prediction,
/// leave-one-parent-out `ProgramPrior`, full-dimensional cosine `RetrievalKnn`, hierarchical
/// Empirical-Bayes program/kNN shrinkage, program novelty gating, and standardized compact
/// bilinear context-program interactions `\tilde{z}_i(c) * \tilde{\phi}_j(p)` into a closed-form
/// ridge predictor capable of ranking unseen `N+1` candidate programs on high-dimensional (2,051-d)
/// S3 contexts without exceeding `MAX_SURROGATE_DESIGN_DIM`.
pub struct ContextualSurrogateQHat<'a> {
    lambda: f64,
    base_mult: f64,
    proj_mult: f64,
    inner_cv: CvEstimate,
    inner_fold_fits: Vec<InnerFoldFit>,
    prior_weight: f64,
    k_neighbours: usize,
    context_features: &'a Features,
    program_features: &'a Features,
    ridge: RidgeProgram<'a>,
    train_targets: Vec<(&'a Pool, Vec<f64>, Vec<f64>)>,
    tot_stats: BTreeMap<String, (f64, usize)>,
    parent_stats: BTreeMap<(String, String), (f64, usize)>,
    sos_tot_stats: BTreeMap<String, (f64, usize)>,
    sos_parent_stats: BTreeMap<(String, String), (f64, usize)>,
    external_prior: Option<ExternalProgramPrior>,
    ctx_proj: Vec<(usize, f64, f64)>,
    prog_proj: Vec<(usize, f64, f64)>,
    weights: Vec<f64>,
}

/// Configured upper or lower bound for max surrogate design dim.
pub const MAX_SURROGATE_DESIGN_DIM: usize = 512;
/// Configured upper or lower bound for max bilinear ctx dim.
pub const MAX_BILINEAR_CTX_DIM: usize = 8;
/// Configured upper or lower bound for max bilinear prog dim.
pub const MAX_BILINEAR_PROG_DIM: usize = 8;
const SURROGATE_BASE_FEATURES: usize = 13;

impl<'a> ContextualSurrogateQHat<'a> {
    /// Returns the `(base_mult, proj_mult)` ridge regularization multipliers selected by 5-fold
    /// parent-grouped inner cross-validation on `train`.
    pub fn selected_multipliers(&self) -> (f64, f64) {
        (self.base_mult, self.proj_mult)
    }

    /// Returns the mean inner cross-validation loss achieved by `selected_multipliers()` on `train`.
    pub fn inner_cv_loss(&self) -> Option<f64> {
        self.inner_cv.loss()
    }

    /// Performs the inner cross-validation estimate operation.
    pub fn inner_cv_estimate(&self) -> &CvEstimate {
        &self.inner_cv
    }

    /// Performs the inner fold fits operation.
    pub fn inner_fold_fits(&self) -> &[InnerFoldFit] {
        &self.inner_fold_fits
    }

    /// Raw unprojected bilinear design dimension (`2 + d_ctx + d_prog + d_ctx * d_prog`).
    pub fn design_dim(d_ctx: usize, d_prog: usize) -> Option<usize> {
        let cross = d_ctx.checked_mul(d_prog)?;
        2usize
            .checked_add(d_ctx)
            .and_then(|v| v.checked_add(d_prog))
            .and_then(|v| v.checked_add(cross))
    }

    /// Effective design dimension after training-fold variance projection to at most
    /// `MAX_BILINEAR_CTX_DIM x MAX_BILINEAR_PROG_DIM` bilinear coordinates.
    pub fn effective_design_dim(d_ctx: usize, d_prog: usize) -> Option<usize> {
        if d_ctx == 0 || d_prog == 0 {
            return None;
        }
        let dz = d_ctx.min(MAX_BILINEAR_CTX_DIM);
        let dp = d_prog.min(MAX_BILINEAR_PROG_DIM);
        let cross = dz.checked_mul(dp)?;
        SURROGATE_BASE_FEATURES
            .checked_add(dz)
            .and_then(|v| v.checked_add(dp))
            .and_then(|v| v.checked_add(cross))
    }

    /// Leave-one-parent-cluster-out Strength-of-Schedule (`SoS`) context-difficulty calibration.
    ///
    /// Estimates each context pool's difficulty offset `ctx_offset` from out-of-parent-cluster
    /// program rank means `mu_p` shrunk toward `0.5` with `prior_weight`, shifting all members in
    /// context `c` additively by `bt + ctx_offset` (without top/bottom clamping) so all within-pool
    /// pairwise target differences `bt_i - bt_j` are identically preserved. Single-member pools
    /// (`bt = 0.5`) evaluate to `0.5 + ctx_offset = mu_p` (`0.5` when unseen outside the parent
    /// cluster).
    pub(super) fn compute_sos_targets(
        raw_targets: &[(&'a Pool, Vec<f64>)],
        prior_weight: f64,
    ) -> Vec<Vec<f64>> {
        let mut tot: BTreeMap<&str, (f64, usize)> = BTreeMap::new();
        let mut par: BTreeMap<(&str, &str), (f64, usize)> = BTreeMap::new();
        for (pool, targets) in raw_targets {
            for (m, &t) in pool.members.iter().zip(targets) {
                let te = tot.entry(m.program_sha256.as_str()).or_insert((0.0, 0));
                te.0 += t;
                te.1 += 1;
                let pe = par
                    .entry((pool.parent_id.as_str(), m.program_sha256.as_str()))
                    .or_insert((0.0, 0));
                pe.0 += t;
                pe.1 += 1;
            }
        }
        raw_targets
            .iter()
            .map(|(pool, base_t)| {
                let nm = pool.members.len();
                if nm == 0 {
                    return Vec::new();
                }
                let diff_sum: f64 = pool
                    .members
                    .iter()
                    .map(|m| {
                        let (ts, tc) = tot
                            .get(m.program_sha256.as_str())
                            .copied()
                            .unwrap_or((0.0, 0));
                        let (ps, pc) = par
                            .get(&(pool.parent_id.as_str(), m.program_sha256.as_str()))
                            .copied()
                            .unwrap_or((0.0, 0));
                        let s_p = ts - ps;
                        let c_p = tc.saturating_sub(pc);
                        let mu_p = (s_p + 0.5 * prior_weight) / (c_p as f64 + prior_weight);
                        mu_p - 0.5
                    })
                    .sum();
                let ctx_offset = diff_sum / (nm as f64);
                base_t.iter().map(|&bt| bt + ctx_offset).collect()
            })
            .collect()
    }
}

impl Selector for ContextualSurrogateQHat<'_> {
    fn name(&self) -> &str {
        "contextual_surrogate_qhat"
    }
    fn class(&self) -> SelectorClass {
        SelectorClass::Candidate
    }
    fn rule(&self) -> String {
        format!(
            "contextual_surrogate_qhat/v3: strict parent-nested refit; context-id-bytes/v1; inner-CV LOPO SoS-debiased Empirical-Bayes + compact bilinear context x program ridge (lambda={}, base_mult={}, proj_mult={}, inner_cv={:?}, d_ctx={}->{}, d_prog={}->{}) + kNN/ridge stack (shadow-only candidate; no multiplicity adjustment)",
            self.lambda,
            self.base_mult,
            self.proj_mult,
            self.inner_cv,
            self.context_features.dim,
            self.ctx_proj.len(),
            self.program_features.dim,
            self.prog_proj.len()
        )
    }
    fn scores(&self, view: &PoolView<'_>) -> Result<Vec<f64>> {
        let rows = self.view_design_rows(view)?;
        let scores: Vec<f64> = rows
            .into_iter()
            .map(|x| x.iter().zip(&self.weights).map(|(xi, wi)| xi * wi).sum())
            .collect();
        ensure!(
            scores.len() == view.programs.len() && scores.iter().all(|s| s.is_finite()),
            "contextual surrogate produced nonfinite or malformed predictions"
        );
        Ok(scores)
    }
}
