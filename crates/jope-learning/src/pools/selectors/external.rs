use super::*;

/// Schema identifier for external values.
pub const EXTERNAL_SCHEMA: &str = "jope.j9-external-scores/v1";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `ExternalScore` is a record for the external score contract in this module: Selectors over a frozen pool (CLOVER §6 Experiment A).
pub struct ExternalScore {
    /// Identifier for the associated context.
    pub context_id: String,
    /// SHA-256 digest of the canonical program source.
    pub program_sha256: String,
    /// Measured score for this outcome; absence denotes a failed draw.
    pub score: f64,
}

/// An externally produced score document: the PolicyLM q-head adapter, a native GP ranking
/// export, or any learned selector. `provenance` (checkpoint, source and input digests) is
/// carried into the receipt verbatim.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalDocument {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Name recorded for this external document.
    pub name: String,
    /// Selector class recorded for this external document.
    pub class: SelectorClass,
    /// Rule recorded for this external document.
    pub rule: String,
    /// Provenance recorded for this external document.
    pub provenance: serde_json::Value,
    /// Collection of scores values carried by this external document.
    pub scores: Vec<ExternalScore>,
}

/// One score per (context, program). Missing scores are an error, never a default.
pub struct External {
    name: String,
    class: SelectorClass,
    rule: String,
    scores: BTreeMap<(String, String), f64>,
}

impl External {
    /// Performs the from document operation and returns an error if it cannot be completed.
    pub fn from_document(doc: &ExternalDocument) -> Result<Self> {
        ensure!(
            doc.schema == EXTERNAL_SCHEMA,
            "unknown external score schema {:?}",
            doc.schema
        );
        ensure!(
            !doc.name.trim().is_empty() && !doc.rule.trim().is_empty(),
            "external selector needs a name and rule"
        );
        let mut scores = BTreeMap::new();
        for (i, row) in doc.scores.iter().enumerate() {
            ensure!(
                row.score.is_finite(),
                "{}: non-finite score at {i}",
                doc.name
            );
            ensure!(
                scores
                    .insert(
                        (row.context_id.clone(), row.program_sha256.clone()),
                        row.score
                    )
                    .is_none(),
                "{}: duplicate score at {i}",
                doc.name
            );
        }
        Ok(Self {
            name: doc.name.clone(),
            class: doc.class,
            rule: doc.rule.clone(),
            scores,
        })
    }
}

impl Selector for External {
    fn name(&self) -> &str {
        &self.name
    }
    fn class(&self) -> SelectorClass {
        self.class
    }
    fn rule(&self) -> String {
        self.rule.clone()
    }
    fn scores(&self, view: &PoolView<'_>) -> Result<Vec<f64>> {
        view.programs
            .iter()
            .map(|p| {
                self.scores
                    .get(&(view.context_id.to_owned(), (*p).to_owned()))
                    .copied()
                    .with_context(|| {
                        format!("{}: no score for {} / {p}", self.name, view.context_id)
                    })
            })
            .collect()
    }
}

/// Cross-family S3 `gp_distilled/v1` program prior statistics restricted to an outer training split.
#[derive(Clone, Debug, Default)]
pub struct ExternalProgramPrior {
    /// Mapping of tot stats values carried by this external program prior.
    pub tot_stats: BTreeMap<String, (f64, usize)>,
    /// Mapping of parent stats values carried by this external program prior.
    pub parent_stats: BTreeMap<(String, String), (f64, usize)>,
}

impl ExternalProgramPrior {
    pub(super) fn restrict_to(&self, train: &[&Pool]) -> Result<Self> {
        let mut total: BTreeMap<String, (f64, usize)> = BTreeMap::new();
        for ((_, program), &(sum, count)) in &self.parent_stats {
            ensure!(
                sum.is_finite() && sum >= 0.0 && sum <= count as f64 && count > 0,
                "invalid external prior statistics"
            );
            let e = total.entry(program.clone()).or_default();
            e.0 += sum;
            e.1 += count;
        }
        ensure!(
            total.len() == self.tot_stats.len()
                && total
                    .iter()
                    .all(|(p, (s, c))| self.tot_stats.get(p).is_some_and(
                        |(t, n)| c == n && (s - t).abs() <= 1e-10 * (*c as f64).max(1.0)
                    )),
            "external prior has unattributed aggregates"
        );
        let parents: BTreeSet<_> = train.iter().map(|p| &p.parent_id).collect();
        let mut out = Self::default();
        for ((parent, program), &(sum, count)) in &self.parent_stats {
            if parents.contains(parent) {
                out.parent_stats
                    .insert((parent.clone(), program.clone()), (sum, count));
                let e = out.tot_stats.entry(program.clone()).or_default();
                e.0 += sum;
                e.1 += count;
            }
        }
        Ok(out)
    }
}
