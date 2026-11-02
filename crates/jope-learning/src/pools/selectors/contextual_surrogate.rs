use super::*;

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

    fn select_top_variance_coords(
        vectors: &[&[f64]],
        dim: usize,
        max_coords: usize,
    ) -> Vec<(usize, f64, f64)> {
        let n = vectors.len().max(1) as f64;
        let mut stats: Vec<(usize, f64, f64)> = (0..dim)
            .map(|j| {
                let mean = vectors.iter().map(|v| v[j]).sum::<f64>() / n;
                let var = vectors
                    .iter()
                    .map(|v| {
                        let d = v[j] - mean;
                        d * d
                    })
                    .sum::<f64>()
                    / n;
                (j, mean, var.sqrt())
            })
            .collect();
        stats.sort_by(|a, b| b.2.total_cmp(&a.2).then_with(|| a.0.cmp(&b.0)));
        stats.truncate(dim.min(max_coords));
        stats
            .into_iter()
            .map(|(j, mean, std)| (j, mean, if std > 1e-8 { std } else { 1.0 }))
            .collect()
    }

    fn project(v: &[f64], proj: &[(usize, f64, f64)]) -> Vec<f64> {
        proj.iter()
            .map(|&(j, mean, scale)| (v[j] - mean) / scale)
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn design_row(
        zt: &[f64],
        pt: &[f64],
        r_s: f64,
        prior_s: f64,
        sos_prior_s: f64,
        s3_prior: f64,
        knn_s: f64,
        sos_knn_s: f64,
        eb_prior: f64,
        eb_sos_prior: f64,
        eb_knn: f64,
        eb_sos_knn: f64,
        nov: f64,
    ) -> Vec<f64> {
        let mut x =
            Vec::with_capacity(SURROGATE_BASE_FEATURES + zt.len() + pt.len() + zt.len() * pt.len());
        x.push(1.0);
        x.push(r_s);
        x.push(prior_s);
        x.push(sos_prior_s);
        x.push(s3_prior);
        x.push(knn_s);
        x.push(sos_knn_s);
        x.push(eb_prior);
        x.push(eb_sos_prior);
        x.push(eb_knn);
        x.push(eb_sos_knn);
        x.push(nov);
        x.push(nov * (r_s - 0.5));
        x.extend_from_slice(zt);
        x.extend_from_slice(pt);
        for &zi in zt {
            for &pj in pt {
                x.push(zi * pj);
            }
        }
        x
    }

    fn view_design_rows_with_ridge(
        &self,
        view: &PoolView<'_>,
        ridge_scores: &[f64],
    ) -> Result<Vec<Vec<f64>>> {
        ensure!(
            ridge_scores.len() == view.programs.len(),
            "ridge prediction shape mismatch"
        );
        let z = self
            .context_features
            .rows
            .get(view.context_id)
            .with_context(|| format!("no context features for {}", view.context_id))?;
        let zt = Self::project(z, &self.ctx_proj);

        type NeighbourRow<'b> = (f64, &'b str, &'b Pool, &'b [f64], &'b [f64]);
        let mut neighbours: Vec<NeighbourRow<'_>> = self
            .train_targets
            .iter()
            .filter(|(p, _, _)| p.parent_id != view.parent_id)
            .map(|(p, t, st)| {
                (
                    cosine(z, &self.context_features.rows[&p.context_id]),
                    p.context_id.as_str(),
                    *p,
                    t.as_slice(),
                    st.as_slice(),
                )
            })
            .collect();
        neighbours.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(b.1)));
        neighbours.truncate(self.k_neighbours);

        let mut rows = Vec::with_capacity(view.programs.len());
        for (&prog_sha, &r_s) in view.programs.iter().zip(ridge_scores) {
            let phi = self
                .program_features
                .rows
                .get(prog_sha)
                .with_context(|| format!("no program features for {prog_sha}"))?;
            let pt = Self::project(phi, &self.prog_proj);
            let (tot_s, tot_c) = self.tot_stats.get(prog_sha).copied().unwrap_or((0.0, 0));
            let (par_s, par_c) = self
                .parent_stats
                .get(&(view.parent_id.to_owned(), prog_sha.to_owned()))
                .copied()
                .unwrap_or((0.0, 0));
            let s_p = tot_s - par_s;
            let c_p = tot_c.saturating_sub(par_c);

            let (sos_tot_s, _) = self
                .sos_tot_stats
                .get(prog_sha)
                .copied()
                .unwrap_or((0.0, 0));
            let (sos_par_s, _) = self
                .sos_parent_stats
                .get(&(view.parent_id.to_owned(), prog_sha.to_owned()))
                .copied()
                .unwrap_or((0.0, 0));
            let sos_s_p = sos_tot_s - sos_par_s;

            let s3_prior = if let Some(ext) = &self.external_prior {
                let (es, ec) = ext.tot_stats.get(prog_sha).copied().unwrap_or((0.0, 0));
                let (eps, epc) = ext
                    .parent_stats
                    .get(&(view.parent_id.to_owned(), prog_sha.to_owned()))
                    .copied()
                    .unwrap_or((0.0, 0));
                let sc = ec.saturating_sub(epc);
                let ss = (es - eps).clamp(0.0, sc as f64);
                (ss + 0.5 * self.prior_weight) / (sc as f64 + self.prior_weight)
            } else {
                0.5
            };
            let prior_s = (s_p + 0.5 * self.prior_weight) / (c_p as f64 + self.prior_weight);
            let sos_prior_s =
                (sos_s_p + 0.5 * self.prior_weight) / (c_p as f64 + self.prior_weight);
            let eb_prior = (s_p + self.prior_weight * r_s) / (c_p as f64 + self.prior_weight);
            let eb_sos_prior =
                (sos_s_p + self.prior_weight * r_s) / (c_p as f64 + self.prior_weight);

            let (mut num, mut sos_num, mut den) = (0.0, 0.0, 0.0);
            for (sim, _, pool, targets, sos_targets) in &neighbours {
                let w = sim.max(0.0);
                if let Ok(i) = pool
                    .members
                    .binary_search_by(|m| m.program_sha256.as_str().cmp(prog_sha))
                {
                    num += w * targets[i];
                    sos_num += w * sos_targets[i];
                    den += w;
                }
            }
            let knn_s = if den > 0.0 { num / den } else { prior_s };
            let sos_knn_s = if den > 0.0 {
                sos_num / den
            } else {
                sos_prior_s
            };
            let eb_knn = (num + self.prior_weight * eb_prior) / (den + self.prior_weight);
            let eb_sos_knn =
                (sos_num + self.prior_weight * eb_sos_prior) / (den + self.prior_weight);
            let nov = 1.0 / (1.0 + c_p as f64);
            rows.push(Self::design_row(
                &zt,
                &pt,
                r_s,
                prior_s,
                sos_prior_s,
                s3_prior,
                knn_s,
                sos_knn_s,
                eb_prior,
                eb_sos_prior,
                eb_knn,
                eb_sos_knn,
                nov,
            ));
        }
        Ok(rows)
    }

    fn view_design_rows(&self, view: &PoolView<'_>) -> Result<Vec<Vec<f64>>> {
        let ridge_scores = self.ridge.scores(view)?;
        self.view_design_rows_with_ridge(view, &ridge_scores)
    }

    /// Performs the fit operation and returns an error if it cannot be completed.
    pub fn fit(
        train: &[&'a Pool],
        context_features: &'a Features,
        program_features: &'a Features,
        k_neighbours: usize,
        prior_weight: f64,
        lambda: f64,
    ) -> Result<Self> {
        Self::fit_with_prior(
            train,
            context_features,
            program_features,
            k_neighbours,
            prior_weight,
            lambda,
            None,
        )
    }

    fn fit_state(
        train: &[&'a Pool],
        context_features: &'a Features,
        program_features: &'a Features,
        k_neighbours: usize,
        prior_weight: f64,
        lambda: f64,
        external_prior: Option<&ExternalProgramPrior>,
    ) -> Result<Self> {
        ensure!(k_neighbours >= 1, "k_neighbours must be positive");
        ensure!(
            prior_weight > 0.0 && prior_weight.is_finite(),
            "prior_weight must be positive and finite"
        );
        ensure!(
            lambda > 0.0 && lambda.is_finite(),
            "lambda must be positive and finite"
        );
        ensure!(
            context_features.dim > 0 && program_features.dim > 0,
            "non-empty context and program feature dimensions required"
        );
        let d = Self::effective_design_dim(context_features.dim, program_features.dim)
            .context("design dimension overflow")?;
        ensure!(
            d <= MAX_SURROGATE_DESIGN_DIM,
            "effective design dimension {d} exceeds MAX_SURROGATE_DESIGN_DIM ({MAX_SURROGATE_DESIGN_DIM})"
        );

        let ridge = RidgeProgram::fit(train, program_features, lambda)?;
        let mut raw_train_targets: Vec<(&'a Pool, Vec<f64>)> = Vec::with_capacity(train.len());
        let mut z_vecs: Vec<&[f64]> = Vec::with_capacity(train.len());
        let mut p_vecs: Vec<&[f64]> = Vec::new();
        for &pool in train {
            let z = context_features
                .rows
                .get(&pool.context_id)
                .with_context(|| format!("no context features for {}", pool.context_id))?;
            ensure!(
                z.len() == context_features.dim && z.iter().all(|v| v.is_finite()),
                "context feature shape/nonfinite"
            );
            z_vecs.push(z.as_slice());
            let targets = rank_targets(pool);
            for m in &pool.members {
                let phi = program_features
                    .rows
                    .get(&m.program_sha256)
                    .with_context(|| format!("no program features for {}", m.program_sha256))?;
                ensure!(
                    phi.len() == program_features.dim && phi.iter().all(|v| v.is_finite()),
                    "program feature shape/nonfinite"
                );
                p_vecs.push(phi.as_slice());
            }
            raw_train_targets.push((pool, targets));
        }
        ensure!(
            !p_vecs.is_empty(),
            "contextual surrogate q_hat needs training rows"
        );

        let sos_all = Self::compute_sos_targets(&raw_train_targets, prior_weight);
        let mut train_targets = Vec::with_capacity(train.len());
        let mut tot_stats: BTreeMap<String, (f64, usize)> = BTreeMap::new();
        let mut parent_stats: BTreeMap<(String, String), (f64, usize)> = BTreeMap::new();
        let mut sos_tot_stats: BTreeMap<String, (f64, usize)> = BTreeMap::new();
        let mut sos_parent_stats: BTreeMap<(String, String), (f64, usize)> = BTreeMap::new();

        for ((pool, targets), sos_targets) in raw_train_targets.into_iter().zip(sos_all) {
            for ((m, &t), &tsos) in pool.members.iter().zip(&targets).zip(&sos_targets) {
                let tot = tot_stats
                    .entry(m.program_sha256.clone())
                    .or_insert((0.0, 0));
                tot.0 += t;
                tot.1 += 1;
                let par = parent_stats
                    .entry((pool.parent_id.clone(), m.program_sha256.clone()))
                    .or_insert((0.0, 0));
                par.0 += t;
                par.1 += 1;

                let stot = sos_tot_stats
                    .entry(m.program_sha256.clone())
                    .or_insert((0.0, 0));
                stot.0 += tsos;
                stot.1 += 1;
                let spar = sos_parent_stats
                    .entry((pool.parent_id.clone(), m.program_sha256.clone()))
                    .or_insert((0.0, 0));
                spar.0 += tsos;
                spar.1 += 1;
            }
            train_targets.push((pool, targets, sos_targets));
        }

        let ctx_proj =
            Self::select_top_variance_coords(&z_vecs, context_features.dim, MAX_BILINEAR_CTX_DIM);
        let prog_proj =
            Self::select_top_variance_coords(&p_vecs, program_features.dim, MAX_BILINEAR_PROG_DIM);

        let model = Self {
            lambda,
            base_mult: 1.0,
            proj_mult: 1.0,
            inner_cv: CvEstimate::Unavailable {
                procedure: NESTED_PROCEDURE.into(),
                reason: CvUnavailableReason::EmptyUsableFolds,
                detail: "state only; not a fitted selector".into(),
                counts: CvCounts::default(),
            },
            inner_fold_fits: Vec::new(),
            prior_weight,
            k_neighbours,
            context_features,
            program_features,
            ridge,
            train_targets,
            tot_stats,
            parent_stats,
            sos_tot_stats,
            sos_parent_stats,
            external_prior: external_prior.map(|p| p.restrict_to(train)).transpose()?,
            ctx_proj,
            prog_proj,
            weights: vec![0.0; d],
        };

        Ok(model)
    }

    /// Refit target-derived state in each training-row complement, while sharing the enclosing
    /// training closure's target-free projection basis. No validation labels enter this matrix.
    fn training_equations(
        &self,
        train: &[&'a Pool],
        external: Option<&ExternalProgramPrior>,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let d = self.weights.len();
        let (mut xtx, mut xty) = (vec![0.0; d * d], vec![0.0; d]);
        let mut cross_fits = BTreeMap::new();
        for parent in evidence_parent_ids(train) {
            let complement: Vec<_> = train
                .iter()
                .copied()
                .filter(|p| p.parent_id != parent)
                .collect();
            ensure!(
                !complement.is_empty(),
                "base cross-fitting needs another parent"
            );
            let mut state = Self::fit_state(
                &complement,
                self.context_features,
                self.program_features,
                self.k_neighbours,
                self.prior_weight,
                self.lambda,
                external,
            )?;
            state.ctx_proj = self.ctx_proj.clone();
            state.prog_proj = self.prog_proj.clone();
            cross_fits.insert(parent, state);
        }
        for (pool, _, targets) in &self.train_targets {
            let state = &cross_fits[pool.parent_id.as_str()];
            let rows = state.view_design_rows(&pool.view())?;
            ensure!(
                rows.len() == targets.len(),
                "training prediction shape mismatch"
            );
            for (x, &t) in rows.iter().zip(targets) {
                ensure!(
                    x.len() == d && t.is_finite() && x.iter().all(|v| v.is_finite()),
                    "nonfinite training design"
                );
                for i in 0..d {
                    xty[i] += x[i] * t;
                    for j in 0..d {
                        xtx[i * d + j] += x[i] * x[j];
                    }
                }
            }
        }
        Ok((xtx, xty))
    }

    fn solve_weights(
        &self,
        mut xtx: Vec<f64>,
        xty: Vec<f64>,
        bm: f64,
        pm: f64,
    ) -> Result<Vec<f64>> {
        let d = self.weights.len();
        let lin_terms = (self.ctx_proj.len() + self.prog_proj.len()).max(1);
        let cross_terms = (self.ctx_proj.len() * self.prog_proj.len()).max(1);
        let lin_end = SURROGATE_BASE_FEATURES + self.ctx_proj.len() + self.prog_proj.len();
        xtx[0] += 1e-12;
        for i in 1..d {
            let penalty = if i < SURROGATE_BASE_FEATURES {
                self.lambda * bm
            } else if i < lin_end {
                self.lambda * pm * lin_terms as f64
            } else {
                self.lambda * pm * cross_terms as f64
            };
            xtx[i * d + i] += penalty;
        }
        ensure!(
            xtx.iter().chain(&xty).all(|v| v.is_finite()),
            "nonfinite normal equations"
        );
        let weights = cholesky_solve(xtx, xty, d)?;
        ensure!(weights.iter().all(|v| v.is_finite()), "nonfinite weights");
        Ok(weights)
    }

    fn state_sha256(&self) -> Result<String> {
        // Include every learned component, descriptor projection and filtered external prior.
        canonical_sha256(&(
            self.ridge.weights.clone(),
            &self.ctx_proj,
            &self.prog_proj,
            &self.tot_stats,
            self.parent_stats.iter().collect::<Vec<_>>(),
            &self.sos_tot_stats,
            self.sos_parent_stats.iter().collect::<Vec<_>>(),
            self.train_targets
                .iter()
                .map(|(p, t, s)| (&p.context_id, t, s))
                .collect::<Vec<_>>(),
            self.external_prior
                .as_ref()
                .map(|p| (&p.tot_stats, p.parent_stats.iter().collect::<Vec<_>>())),
        ))
    }

    /// Performs the fit with prior operation and returns an error if it cannot be completed.
    pub fn fit_with_prior(
        train: &[&'a Pool],
        context_features: &'a Features,
        program_features: &'a Features,
        k_neighbours: usize,
        prior_weight: f64,
        lambda: f64,
        external_prior: Option<&ExternalProgramPrior>,
    ) -> Result<Self> {
        let parents: BTreeSet<_> = train.iter().map(|p| p.parent_id.as_str()).collect();
        let folds: Vec<_> = train
            .iter()
            .map(|p| super::evaluate::fold_of(INNER_SALT, &p.parent_id, INNER_FOLDS))
            .collect();
        let mut counts = CvCounts {
            attempted_units: train.iter().map(|p| p.members.len()).sum(),
            parents: parents.len(),
            folds: folds.iter().copied().collect::<BTreeSet<_>>().len(),
            ..CvCounts::default()
        };
        if parents.len() < 3 {
            return Err(unavailable(
                CvUnavailableReason::InsufficientParents,
                "nested OOF needs at least three parents",
                counts,
            ));
        }
        if counts.attempted_units == 0 {
            return Err(unavailable(
                CvUnavailableReason::EmptyUsableFolds,
                "no member rows",
                counts,
            ));
        }
        if train
            .iter()
            .any(|p| p.members.len() < 2 || p.members.iter().all(|m| m.outcome().is_none()))
        {
            return Err(unavailable(
                CvUnavailableReason::EmptyUsableFolds,
                "tiny or wholly failed pool has no usable ranking target",
                counts,
            ));
        }
        if train
            .iter()
            .flat_map(|p| &p.members)
            .filter_map(|m| m.q)
            .any(|v| !v.is_finite())
        {
            return Err(unavailable(
                CvUnavailableReason::NonfiniteSystem,
                "nonfinite measured outcome",
                counts,
            ));
        }
        for features in [context_features, program_features] {
            if features.dim == 0
                || features.rows.is_empty()
                || features.rows.values().any(|v| v.len() != features.dim)
            {
                return Err(unavailable(
                    CvUnavailableReason::ShapeMismatch,
                    "feature width mismatch",
                    counts,
                ));
            }
            if features.rows.values().flatten().any(|v| !v.is_finite()) {
                return Err(unavailable(
                    CvUnavailableReason::NonfiniteSystem,
                    "nonfinite input features",
                    counts,
                ));
            }
        }
        context_features.verify_context_descriptor().map_err(|e| {
            unavailable(
                CvUnavailableReason::UnboundContextProvenance,
                e,
                counts.clone(),
            )
        })?;
        if let Some(prior) = external_prior {
            prior.restrict_to(train).map_err(|e| {
                unavailable(
                    CvUnavailableReason::ExternalPriorProvenance,
                    e,
                    counts.clone(),
                )
            })?;
        }
        if counts.folds < 2 {
            return Err(unavailable(
                CvUnavailableReason::EmptyUsableFolds,
                "all parents collide in one fold",
                counts,
            ));
        }
        // Recompute every component and both train/held design blocks inside each inner closure.
        let mut closures = Vec::new();
        let mut evidence = Vec::new();
        for fold in 0..INNER_FOLDS {
            let held: Vec<_> = train
                .iter()
                .copied()
                .zip(&folds)
                .filter_map(|(p, &f)| (f == fold).then_some(p))
                .collect();
            if held.is_empty() {
                continue;
            }
            let fitting: Vec<_> = train
                .iter()
                .copied()
                .zip(&folds)
                .filter_map(|(p, &f)| (f != fold).then_some(p))
                .collect();
            if fitting
                .iter()
                .map(|p| &p.parent_id)
                .collect::<BTreeSet<_>>()
                .len()
                < 2
            {
                return Err(unavailable(
                    CvUnavailableReason::InsufficientParents,
                    "inner training closure cannot cross-fit its base components",
                    counts,
                ));
            }
            let state = Self::fit_state(
                &fitting,
                context_features,
                program_features,
                k_neighbours,
                prior_weight,
                lambda,
                external_prior,
            )
            .map_err(|e| component_error(e, counts.clone()))?;
            let (xtx, xty) = state
                .training_equations(&fitting, external_prior)
                .map_err(|e| component_error(e, counts.clone()))?;
            let designs = held
                .iter()
                .map(|p| state.view_design_rows(&p.view()))
                .collect::<Result<Vec<_>>>()
                .map_err(|e| component_error(e, counts.clone()))?;
            evidence.push(InnerFoldFit {
                fold,
                training_parents: fitting
                    .iter()
                    .map(|p| p.parent_id.clone())
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect(),
                held_parents: held
                    .iter()
                    .map(|p| p.parent_id.clone())
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect(),
                training_input_sha256: canonical_sha256(
                    &fitting
                        .iter()
                        .map(|p| p.sha256())
                        .collect::<Result<Vec<_>>>()?,
                )?,
                training_features_sha256: canonical_sha256(
                    &fitting
                        .iter()
                        .map(|p| {
                            (
                                &context_features.rows[&p.context_id],
                                p.members
                                    .iter()
                                    .map(|m| &program_features.rows[&m.program_sha256])
                                    .collect::<Vec<_>>(),
                            )
                        })
                        .collect::<Vec<_>>(),
                )?,
                split_sha256: canonical_sha256(&(
                    NESTED_PROCEDURE,
                    fold,
                    &evidence_parent_ids(&fitting),
                    &evidence_parent_ids(&held),
                ))?,
                state_sha256: state.state_sha256()?,
                held_design_sha256: canonical_sha256(&designs)?,
                prediction_sha256: Vec::new(),
                selected_prediction_bits: Vec::new(),
            });
            closures.push((state, xtx, xty, held, designs));
        }
        let mut best: Option<(f64, f64, f64, usize)> = None;
        for bm in [1.0, 10.0, 100.0, 1000.0] {
            for pm in [1.0, 10.0, 100.0, 1000.0, 10_000.0, 100_000.0] {
                if pm < bm {
                    continue;
                }
                let mut sq_err = 0.0;
                let mut scored = 0;
                let mut candidate_bits = Vec::new();
                let mut valid = true;
                for (fold_index, (state, xtx, xty, held, designs)) in closures.iter().enumerate() {
                    let weights = match state.solve_weights(xtx.clone(), xty.clone(), bm, pm) {
                        Ok(w) => w,
                        Err(error) => {
                            counts.failure_details.push(format!(
                                "fold={},bm={bm},pm={pm}: {error}",
                                evidence[fold_index].fold
                            ));
                            valid = false;
                            continue;
                        }
                    };
                    let predictions: Vec<Vec<f64>> = designs
                        .iter()
                        .map(|rows| {
                            rows.iter()
                                .map(|x| x.iter().zip(&weights).map(|(x, w)| x * w).sum())
                                .collect()
                        })
                        .collect();
                    if predictions.iter().flatten().any(|v| !v.is_finite()) {
                        counts.failure_details.push(format!(
                            "fold={},bm={bm},pm={pm}: nonfinite held predictions",
                            evidence[fold_index].fold
                        ));
                        valid = false;
                        continue;
                    }
                    // Predictions are fixed before touching held outcomes.
                    evidence[fold_index]
                        .prediction_sha256
                        .push(canonical_sha256(&(format!("{bm},{pm}"), &predictions))?);
                    candidate_bits.push(
                        predictions
                            .iter()
                            .map(|row| {
                                row.iter()
                                    .map(|p| format!("{:016x}", p.to_bits()))
                                    .collect()
                            })
                            .collect(),
                    );
                    for (pool, preds) in held.iter().zip(predictions) {
                        let targets = rank_targets(pool);
                        if preds.len() != targets.len() {
                            return Err(unavailable(
                                CvUnavailableReason::ShapeMismatch,
                                "held prediction width",
                                counts,
                            ));
                        }
                        if preds.is_empty() {
                            continue;
                        }
                        let n = preds.len() as f64;
                        let pmean = preds.iter().sum::<f64>() / n;
                        let tmean = targets.iter().sum::<f64>() / n;
                        for (p, t) in preds.iter().zip(targets) {
                            sq_err += ((p - pmean) - (t - tmean)).powi(2);
                            scored += 1;
                        }
                    }
                }
                if !valid || scored == 0 || !sq_err.is_finite() {
                    counts.failed_candidates += 1;
                    continue;
                }
                let loss = sq_err / scored as f64;
                if best.is_none_or(|b| loss < b.2) {
                    best = Some((bm, pm, loss, scored));
                    for (record, bits) in evidence.iter_mut().zip(candidate_bits) {
                        record.selected_prediction_bits = bits;
                    }
                }
            }
        }
        let Some((bm, pm, loss, scored)) = best else {
            return Err(unavailable(
                CvUnavailableReason::NoFiniteCandidate,
                "no candidate produced a finite complete OOF loss",
                counts,
            ));
        };
        counts.scored_units = scored;
        counts.excluded_units = counts.attempted_units - scored;
        let mut model = Self::fit_state(
            train,
            context_features,
            program_features,
            k_neighbours,
            prior_weight,
            lambda,
            external_prior,
        )
        .map_err(|e| component_error(e, counts.clone()))?;
        let (xtx, xty) = model
            .training_equations(train, external_prior)
            .map_err(|e| component_error(e, counts.clone()))?;
        model.weights = model
            .solve_weights(xtx, xty, bm, pm)
            .map_err(|e| component_error(e, counts.clone()))?;
        model.base_mult = bm;
        model.proj_mult = pm;
        model.inner_cv = CvEstimate::Supported {
            procedure: NESTED_PROCEDURE.into(),
            metric: ExactCvLoss {
                f64_bits: format!("{:016x}", loss.to_bits()),
            },
            counts,
        };
        model.inner_fold_fits = evidence;
        Ok(model)
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
