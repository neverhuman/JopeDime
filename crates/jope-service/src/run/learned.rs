//! Explicit owner-admitted r782 execution. This does not grant Auto/Q1 quality.
use crate::{
    dataset::jmat::{JmatCell, Matrix},
    discover::r#loop::{Plan, RoundAdmission, RoundContext},
    ranker::{
        admission::{AdmissionProgram, MeasuredAlgorithm, RevealedRule, ServingAdmissionSlate},
        bundle::VerifiedBundle,
        cuda::{release::R782, FusedForward},
        featurizer::Featurizer,
        landmarks, proj16,
        stage1b::TokenProjection,
    },
};
use anyhow::{ensure, Context, Result};
use jope_api::Placement;
use jope_runtime::backend::DeviceRequest;
use rand_chacha::ChaCha12Rng;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

/// Only a trusted embedding installs this capability, after release verification.
/// The t2015 admission is specific to the frozen numeric-regression candidate.
pub struct AdmittedRanker {
    bundle: Arc<VerifiedBundle>,
    forwards: Arc<AtomicUsize>,
}
impl AdmittedRanker {
    pub fn r782(bundle: VerifiedBundle) -> Result<Self> {
        ensure!(
            bundle.digest().to_string() == R782,
            "bundle is not admitted r782"
        );
        Ok(Self {
            bundle: Arc::new(bundle),
            forwards: Arc::new(AtomicUsize::new(0)),
        })
    }
    pub(crate) fn instantiate(
        &self,
        device: DeviceRequest,
        workers: u32,
    ) -> Result<LearnedAdmission> {
        #[derive(serde::Deserialize)]
        struct FeatureMetadata {
            feature_names: Vec<String>,
            recipe: String,
            proj16_domain: String,
        }
        let metadata: FeatureMetadata = serde_json::from_slice(
            self.bundle
                .asset("featurizer.json")
                .context("missing authenticated featurizer")?,
        )?;
        ensure!(
            metadata.recipe == "rank_prepare.features/412-v1"
                && metadata.proj16_domain == "jope-context-cross-v1",
            "unsupported feature recipe"
        );
        Ok(LearnedAdmission {
            graph: FusedForward::from_bundle(&self.bundle, device, workers)?,
            featurizer: Featurizer::from_names(metadata.feature_names)?,
            conditioning: None,
            plan: None,
            forwards: Arc::clone(&self.forwards),
        })
    }
    #[cfg(test)]
    pub(crate) fn execution_counter(&self) -> Arc<AtomicUsize> {
        Arc::clone(&self.forwards)
    }
}

pub(crate) struct LearnedAdmission {
    graph: FusedForward,
    featurizer: Featurizer,
    conditioning: Option<TokenProjection>,
    plan: Option<Plan>,
    forwards: Arc<AtomicUsize>,
}
impl LearnedAdmission {
    /// Borrow the actual FIT-bound conditioning for research provenance.
    /// The Plan must exactly match the one used to fit these tokens.
    #[allow(dead_code)] // Used by the qualified provider once it is admitted.
    pub(crate) fn research_conditioning<'a>(&'a self, plan: &Plan) -> Result<&'a TokenProjection> {
        ensure!(
            self.plan.as_ref() == Some(plan),
            "FIT conditioning Plan differs"
        );
        self.conditioning
            .as_ref()
            .context("FIT conditioning is not bound")
    }

    pub(crate) fn placement(&self) -> Placement {
        self.graph.placement().clone()
    }
    pub(crate) fn bind_fit(&mut self, bytes: &[u8], plan: &Plan) -> Result<usize> {
        ensure!(
            format!("sha256:{}", plan.bundle_digest) == R782
                && jope_runtime::sha256_hex(bytes) == plan.matrix_sha256,
            "learned FIT/Plan identity mismatch"
        );
        let matrix = Matrix::open(bytes)?;
        ensure!(
            matrix.rows() >= 4 && matrix.rows() <= 20_000,
            "admitted full-TRAIN two-fold recipe needs 4..20000 FIT rows"
        );
        let column = |name: &str| {
            matrix
                .columns()
                .iter()
                .position(|c| c.name == name)
                .context("unknown learned FIT column")
        };
        let target = column(&plan.target)?;
        let features = plan
            .features
            .iter()
            .map(|name| column(name))
            .collect::<Result<Vec<_>>>()?;
        ensure!(!features.is_empty(), "learned FIT has no features");
        let rows = crate::discover::splits::rung0_subsample(
            &(0..matrix.rows()).collect::<Vec<_>>(),
            1024,
            plan.seed,
            "rnk12/landmarks/full-fit-v1",
        );
        let mut x = Vec::with_capacity(rows.len() * features.len());
        let mut y = Vec::with_capacity(rows.len());
        for &row in &rows {
            for &col in &features {
                x.push(match matrix.cell(row, col) {
                    Some(JmatCell::Numeric(v)) if (v as f32).is_finite() => v as f32,
                    Some(JmatCell::Missing) => f32::NAN,
                    _ => anyhow::bail!("invalid learned numeric FIT feature"),
                });
            }
            let Some(JmatCell::Numeric(value)) = matrix.cell(row, target) else {
                anyhow::bail!("invalid learned FIT target");
            };
            ensure!((value as f32).is_finite(), "learned target overflows F32");
            y.push(value as f32);
        }
        let lm =
            landmarks::fit_transform_and_extract_bundle_high(&x, rows.len(), features.len(), &y)?;
        let columns: Vec<_> = lm.columns().iter().flatten().copied().collect();
        let stages = self.graph.stages(lm.vector(), &columns)?;
        self.conditioning = Some(TokenProjection {
            tokens: std::array::from_fn(|i| std::array::from_fn(|j| stages.tokens[i * 32 + j])),
            mask: stages
                .mask
                .try_into()
                .map_err(|_| anyhow::anyhow!("invalid token mask"))?,
        });
        self.plan = Some(plan.clone());
        Ok(rows.len())
    }
    fn program(&self, dsl: &str) -> Result<([f32; 412], [f32; 16])> {
        let features = self.featurizer.featurize(dsl)?;
        let projection = proj16::proj16(&features.map(f64::from)).map(|v| v as f32);
        ensure!(
            projection.iter().all(|v| v.is_finite()),
            "nonfinite projection"
        );
        Ok((features, projection))
    }
}
impl RoundAdmission for LearnedAdmission {
    fn admit_round(
        &self,
        context: &RoundContext<'_>,
        proposals: Vec<String>,
        explore: bool,
        rng: &mut ChaCha12Rng,
    ) -> Result<Vec<String>> {
        let bound = self
            .plan
            .as_ref()
            .context("FIT conditioning is not bound")?;
        let mut observed = context.plan.clone();
        // Publishing the immutable FIT bytes assigns their CAS path after the
        // dry-run Plan. The content digest and every other field stay pinned.
        observed.matrix_path = bound.matrix_path.clone();
        ensure!(&observed == bound, "learned admission Plan changed");
        ensure!(
            context.round > 0,
            "seed round does not invoke learned admission"
        );
        let pool = context
            .measured
            .iter()
            .map(|m| self.program(m.dsl))
            .collect::<Result<Vec<_>>>()?;
        let programs = proposals
            .iter()
            .map(|dsl| self.program(dsl))
            .collect::<Result<Vec<_>>>()?;
        let measured = context
            .measured
            .iter()
            .map(|m| -> Result<_> {
                // The frozen recipe has disjoint ranges: failed gate F>=100;
                // passed gate F<100 for the bounded native DSL. No RMSE proxy enters.
                ensure!(
                    m.dsl.len() < 200_000 && m.surrogate_fitness.is_finite(),
                    "invalid trained fitness"
                );
                Ok(MeasuredAlgorithm {
                    id: m.dsl,
                    surrogate_fitness: m.surrogate_fitness,
                    gate: m.surrogate_fitness < 100.0,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let pool_inputs: Vec<_> = context
            .measured
            .iter()
            .zip(&pool)
            .map(|(m, (f, p))| AdmissionProgram {
                key: m.dsl,
                features412: f,
                projection16: p,
            })
            .collect();
        let proposal_inputs: Vec<_> = proposals
            .iter()
            .zip(&programs)
            .map(|(dsl, (f, p))| AdmissionProgram {
                key: dsl,
                features412: f,
                projection16: p,
            })
            .collect();
        let slate = ServingAdmissionSlate::prepare(
            &measured,
            &pool_inputs,
            &proposal_inputs,
            self.conditioning
                .as_ref()
                .context("FIT conditioning is not bound")?,
            RevealedRule::SurrogateFitness,
            explore,
        )?;
        let output = self.graph.predict_costs(slate.inputs()?)?;
        let costs: Vec<_> = output.total_cost.iter().map(|&v| f64::from(v)).collect();
        let selected = slate.select(&costs, rng)?;
        self.forwards.fetch_add(1, Ordering::Relaxed);
        Ok(selected.into_iter().map(|i| proposals[i].clone()).collect())
    }
}
