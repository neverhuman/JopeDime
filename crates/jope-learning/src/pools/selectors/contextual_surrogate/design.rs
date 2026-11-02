use super::*;

impl<'a> ContextualSurrogateQHat<'a> {
    pub(super) fn select_top_variance_coords(
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

    pub(super) fn project(v: &[f64], proj: &[(usize, f64, f64)]) -> Vec<f64> {
        proj.iter()
            .map(|&(j, mean, scale)| (v[j] - mean) / scale)
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn design_row(
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

    pub(super) fn view_design_rows_with_ridge(
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

    pub(super) fn view_design_rows(&self, view: &PoolView<'_>) -> Result<Vec<Vec<f64>>> {
        let ridge_scores = self.ridge.scores(view)?;
        self.view_design_rows_with_ridge(view, &ridge_scores)
    }
}
