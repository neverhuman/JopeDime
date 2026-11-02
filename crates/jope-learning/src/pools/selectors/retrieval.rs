use super::*;

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// Similar-task reuse: score each program by its training targets on the `k` training contexts
/// whose context features are most cosine-similar, weighting by similarity; programs absent from
/// those neighbours fall back to the program prior.
pub struct RetrievalKnn<'a> {
    k: usize,
    context_features: &'a Features,
    train: Vec<(&'a Pool, Vec<f64>)>,
    prior: ProgramPrior,
}

impl<'a> RetrievalKnn<'a> {
    /// Performs the fit operation and returns an error if it cannot be completed.
    pub fn fit(
        train: &[&'a Pool],
        context_features: &'a Features,
        k: usize,
        prior_weight: f64,
    ) -> Result<Self> {
        ensure!(k >= 1, "k must be positive");
        for pool in train {
            ensure!(
                context_features.rows.contains_key(&pool.context_id),
                "no context features for training context {}",
                pool.context_id
            );
        }
        Ok(Self {
            k,
            context_features,
            train: train.iter().map(|p| (*p, rank_targets(p))).collect(),
            prior: ProgramPrior::fit(train, prior_weight),
        })
    }
}

impl Selector for RetrievalKnn<'_> {
    fn name(&self) -> &str {
        "retrieval_knn"
    }
    fn class(&self) -> SelectorClass {
        SelectorClass::Baseline
    }
    fn rule(&self) -> String {
        format!(
            "retrieval_knn/v1: k={} cosine over context features, similarity-weighted training percentile, excluding the evaluated parent; fallback program_prior",
            self.k
        )
    }
    fn scores(&self, view: &PoolView<'_>) -> Result<Vec<f64>> {
        let query = self
            .context_features
            .rows
            .get(view.context_id)
            .with_context(|| format!("no context features for {}", view.context_id))?;
        let mut neighbours: Vec<(f64, &str, &Pool, &Vec<f64>)> = self
            .train
            .iter()
            .filter(|(p, _)| p.parent_id != view.parent_id)
            .map(|(p, t)| {
                (
                    cosine(query, &self.context_features.rows[&p.context_id]),
                    p.context_id.as_str(),
                    *p,
                    t,
                )
            })
            .collect();
        neighbours.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(b.1)));
        neighbours.truncate(self.k);
        Ok(view
            .programs
            .iter()
            .map(|program| {
                let (mut num, mut den) = (0.0, 0.0);
                for (sim, _, pool, targets) in &neighbours {
                    let w = sim.max(0.0);
                    if let Ok(i) = pool
                        .members
                        .binary_search_by(|m| m.program_sha256.as_str().cmp(program))
                    {
                        num += w * targets[i];
                        den += w;
                    }
                }
                if den > 0.0 {
                    num / den
                } else {
                    self.prior.score(program)
                }
            })
            .collect())
    }
}
