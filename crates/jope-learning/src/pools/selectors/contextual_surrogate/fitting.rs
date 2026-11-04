use super::*;

impl<'a> ContextualSurrogateQHat<'a> {
    pub(crate) fn fit_state(
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
    pub(crate) fn training_equations(
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

    pub(crate) fn solve_weights(
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

    pub(crate) fn state_sha256(&self) -> Result<String> {
        // Include every learned component, descriptor projection and filtered external prior.
        canonical_sha256(&(
            self.ridge.weights().to_vec(),
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
}
