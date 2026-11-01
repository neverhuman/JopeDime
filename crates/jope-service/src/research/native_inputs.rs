//! Native inputs for a provider, not an admission or qualification capability.

use super::{
    capture::{digest, Binding},
    hex, name, RecordedProgram,
};
use crate::{
    discover::r#loop::Plan,
    ranker::{
        bundle::VerifiedBundle,
        featurizer::{self, Featurizer},
        proj16,
        stage1b::TokenProjection,
    },
};
use anyhow::{ensure, Context, Result};
use serde::Deserialize;

/// Immutable native feature/conditioning closure for an eventual qualified provider.
///
/// Construction requires an authenticated bundle, the exact Plan, the actual
/// conditioning and its explicit receipt identity. It does not authenticate the
/// corpus, qualify the supplied conditioning, choose reveals/seeds or construct
/// a CaptureContext. Operator families remain the caller's source attestations.
pub struct NativeCaptureInputs {
    plan: Plan,
    conditioning_sha256: String,
    tokens: Vec<Vec<f32>>,
    mask: Vec<bool>,
    featurizer: Featurizer,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Metadata {
    feature_names: Vec<String>,
    recipe: String,
    proj16_domain: String,
}

impl NativeCaptureInputs {
    pub fn from_bundle(
        bundle: &VerifiedBundle,
        plan: &Plan,
        conditioning_sha256: &str,
        conditioning: &TokenProjection,
    ) -> Result<Self> {
        Self::from_parts(
            &bundle.digest().to_string(),
            bundle
                .asset("featurizer.json")
                .context("missing authenticated featurizer metadata")?,
            plan,
            conditioning_sha256,
            conditioning,
        )
    }

    fn from_parts(
        bundle_digest: &str,
        metadata: &[u8],
        plan: &Plan,
        conditioning_sha256: &str,
        conditioning: &TokenProjection,
    ) -> Result<Self> {
        ensure!(
            bundle_digest == format!("sha256:{}", plan.bundle_digest),
            "native input bundle/plan mismatch"
        );
        ensure!(
            hex(conditioning_sha256, 64),
            "conditioning receipt identity is not pinned"
        );
        ensure!(
            conditioning.tokens.iter().flatten().all(|v| v.is_finite())
                && conditioning.mask.iter().all(|v| *v == 0.0 || *v == 1.0),
            "invalid native conditioning"
        );
        let metadata: Metadata =
            serde_json::from_slice(metadata).context("invalid native featurizer metadata")?;
        ensure!(
            metadata.recipe == "rank_prepare.features/412-v1"
                && metadata.proj16_domain == "jope-context-cross-v1",
            "unsupported native feature recipe"
        );
        ensure!(
            metadata
                .feature_names
                .iter()
                .map(String::as_str)
                .eq(featurizer::FEATURE_NAMES),
            "native feature name order differs from frozen recipe"
        );
        Ok(Self {
            plan: plan.clone(),
            conditioning_sha256: conditioning_sha256.into(),
            tokens: conditioning.tokens.iter().map(|v| v.to_vec()).collect(),
            mask: conditioning.mask.iter().map(|v| *v == 1.0).collect(),
            featurizer: Featurizer::from_names(metadata.feature_names)?,
        })
    }

    /// Compute the actual frozen features and original f64 projection once.
    /// The family must come from the proposer/seed source, never DSL inference.
    /// The caller retains executable-DSL admission; the frozen feature recipe
    /// deliberately accepts a broader reference syntax.
    pub fn program(&self, dsl: &str, operator_family: &str) -> Result<RecordedProgram> {
        ensure!(name(operator_family), "invalid operator-family attestation");
        let feat412 = self.featurizer.featurize(dsl)?;
        let projection = proj16::proj16(&feat412.map(f64::from));
        ensure!(
            projection.iter().all(|v| v.is_finite()),
            "nonfinite native projection"
        );
        Ok(RecordedProgram {
            program_sha256: digest(dsl.as_bytes()),
            operator_family: operator_family.into(),
            feat412: feat412.to_vec(),
            proj16: projection.to_vec(),
        })
    }

    pub(super) fn validate_binding(&self, binding: &Binding, plan: &Plan) -> Result<()> {
        ensure!(&self.plan == plan, "native input Plan changed");
        self.validate_record_binding(binding)
    }

    pub(super) fn validate_record_binding(&self, binding: &Binding) -> Result<()> {
        ensure!(
            binding.identity.bundle_sha256 == self.plan.bundle_digest
                && binding.identity.problem_sha256 == self.plan.problem_digest
                && binding.matrix_sha256 == self.plan.matrix_sha256
                && binding.identity.run_id == self.plan.run_id.to_string()
                && binding.seed == self.plan.seed,
            "native input binding/Plan mismatch"
        );
        ensure!(
            binding.identity.conditioning_sha256 == self.conditioning_sha256
                && binding.mask == self.mask
                && binding.tokens.len() == self.tokens.len()
                && binding
                    .tokens
                    .iter()
                    .zip(&self.tokens)
                    .all(|(actual, expected)| actual.len() == expected.len()
                        && actual
                            .iter()
                            .zip(expected)
                            .all(|(a, b)| a.to_bits() == b.to_bits())),
            "native input conditioning changed"
        );
        Ok(())
    }

    pub(super) fn validate_observation(
        &self,
        binding: &Binding,
        dsls: &[String],
        programs: &[RecordedProgram],
    ) -> Result<()> {
        self.validate_record_binding(binding)?;
        ensure!(
            dsls.len() == programs.len(),
            "native program inventory mismatch"
        );
        for (dsl, program) in dsls.iter().zip(programs) {
            let native = self.program(dsl, &program.operator_family)?;
            ensure!(
                program.program_sha256 == native.program_sha256
                    && program.feat412.len() == native.feat412.len()
                    && program
                        .feat412
                        .iter()
                        .zip(&native.feat412)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                    && program.proj16.len() == native.proj16.len()
                    && program
                        .proj16
                        .iter()
                        .zip(&native.proj16)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                "provider inputs differ from native program features/projection"
            );
        }
        Ok(())
    }
}

// Boundary-only controls: no Engine record is written or accepted as evidence.
#[cfg(test)]
pub(super) fn input_controls(mut record: super::ResearchRound) {
    use super::{
        capture::{CaptureProvider, SlateObservation},
        CaptureConfig, CaptureContext, RecordEvidence, RecordedReveal, Rung0Measurement,
    };
    use crate::worker::{Request, Response};
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    let plan = Plan {
        run_id: "run_01JBD8Q6N4V2T5Y8K3M7P9R1S0".parse().unwrap(),
        dataset_id: "ds_21fbfe8b53dbcd605d79486b".parse().unwrap(),
        problem_digest: "1".repeat(64),
        bundle_digest: "2".repeat(64),
        binary_version: "boundary-control".into(),
        matrix_path: "not-opened.jmat".into(),
        matrix_sha256: "3".repeat(64),
        target: "target".into(),
        features: vec!["x".into(), "y".into()],
        objective: "rmse".into(),
        seed: 73,
        planned_workers: 1,
        effort: jope_api::Effort::Fast,
        evaluations: 2,
        wall_seconds: 10,
        evaluation_wall_ms: 1000,
        evaluation_rss_bytes: 1024 * 1024,
        groups: None,
        times: None,
        gap_rows: 0,
    };
    let conditioning = TokenProjection {
        tokens: [[0.25; 32]; 10],
        mask: [1.0; 10],
    };
    let metadata = serde_json::json!({"feature_names": featurizer::FEATURE_NAMES.to_vec(),
        "recipe":"rank_prepare.features/412-v1", "proj16_domain":"jope-context-cross-v1"});
    let bytes = serde_json::to_vec(&metadata).unwrap();
    let bundle = format!("sha256:{}", plan.bundle_digest);
    let make = |data: &[u8], pin: &str, cond: &TokenProjection| {
        NativeCaptureInputs::from_parts(&bundle, data, &plan, pin, cond)
    };
    let pin = "4".repeat(64);
    let native = make(&bytes, &pin, &conditioning).unwrap();
    let dsl = "(pipe X (ridge))";
    let actual = native.program(dsl, "seed").unwrap();
    assert_eq!(actual.feat412.len(), 412);
    assert_eq!(actual.proj16.len(), 16);
    assert!(actual.feat412.iter().any(|v| *v != 0.0));
    assert!(
        actual.proj16.iter().any(|v| f64::from(*v as f32) != *v),
        "projection retains f64 precision"
    );
    assert_eq!(actual.program_sha256, digest(dsl.as_bytes()));
    assert_eq!(actual.operator_family, "seed");
    assert!(native.program(dsl, "").is_err());
    assert!(native.program("( )", "seed").is_err());
    assert!(make(&bytes, "", &conditioning).is_err());
    assert!(
        NativeCaptureInputs::from_parts("sha256:wrong", &bytes, &plan, &pin, &conditioning)
            .is_err()
    );
    for change in 0..5 {
        let mut bad = metadata.clone();
        match change {
            0 => bad["recipe"] = "unknown".into(),
            1 => bad["proj16_domain"] = "unknown".into(),
            2 => {
                bad["feature_names"].as_array_mut().unwrap().swap(0, 1);
            }
            3 => {
                bad["feature_names"].as_array_mut().unwrap().pop();
            }
            _ => bad["unknown"] = true.into(),
        }
        assert!(make(&serde_json::to_vec(&bad).unwrap(), &pin, &conditioning).is_err());
    }
    let bad = TokenProjection {
        tokens: [[f32::NAN; 32]; 10],
        mask: [1.0; 10],
    };
    assert!(make(&bytes, &pin, &bad).is_err());
    let bad = TokenProjection {
        tokens: [[0.25; 32]; 10],
        mask: [0.5; 10],
    };
    assert!(make(&bytes, &pin, &bad).is_err());
    record.identity.evidence = RecordEvidence::Engine;
    record.identity.run_id = plan.run_id.to_string();
    record.identity.problem_sha256 = plan.problem_digest.clone();
    record.identity.bundle_sha256 = plan.bundle_digest.clone();
    record.identity.conditioning_sha256 = pin.clone();
    let binding = Binding {
        config: CaptureConfig {
            max_attempts: 2,
            max_rounds: 1,
            max_bytes: 1_000_000,
            evaluate_all_proposals: false,
            seed_protocol_sha256: "5".repeat(64),
        },
        identity: record.identity,
        seed: plan.seed,
        tokens: conditioning.tokens.iter().map(|v| v.to_vec()).collect(),
        mask: vec![true; 10],
        matrix_sha256: plan.matrix_sha256.clone(),
    };
    native.validate_binding(&binding, &plan).unwrap();
    for change in 0..5 {
        let mut changed = plan.clone();
        match change {
            0 => changed.seed += 1,
            1 => changed.matrix_sha256 = "6".repeat(64),
            2 => changed.features.reverse(),
            3 => changed.wall_seconds += 1,
            _ => changed.objective = "mae".into(),
        }
        assert!(native.validate_binding(&binding, &changed).is_err());
    }
    for change in 0..5 {
        let mut bad = binding.clone();
        match change {
            0 => bad.tokens[0][0] += 0.1,
            1 => bad.mask[0] = false,
            2 => bad.identity.conditioning_sha256 = "6".repeat(64),
            3 => bad.identity.bundle_sha256 = "6".repeat(64),
            _ => bad.matrix_sha256 = "6".repeat(64),
        }
        assert!(native.validate_record_binding(&bad).is_err());
    }
    struct Provider {
        native: NativeCaptureInputs,
        mode: Arc<AtomicUsize>,
    }
    impl CaptureProvider for Provider {
        fn native_inputs(&self) -> Option<&NativeCaptureInputs> {
            Some(&self.native)
        }
        fn validate_binding(&self, _: &Binding, _: &Plan) -> Result<()> {
            Ok(())
        }
        fn observe(
            &self,
            _: u64,
            dsls: &[String],
            _: &[RecordedReveal],
        ) -> Result<SlateObservation> {
            let mut programs = dsls
                .iter()
                .map(|dsl| self.native.program(dsl, "seed"))
                .collect::<Result<Vec<_>>>()?;
            match self.mode.load(Ordering::SeqCst) {
                1 => programs[0].feat412[0] += 1.0,
                2 => {
                    for v in &mut programs[0].proj16 {
                        *v = f64::from(*v as f32);
                    }
                }
                3 => programs[0].program_sha256 = "6".repeat(64),
                _ => {}
            }
            Ok(SlateObservation {
                programs,
                revealed_indices: vec![],
            })
        }
        fn measurement(&self, _: &Request, _: &Response) -> Result<Rung0Measurement> {
            anyhow::bail!("not executed")
        }
        fn label_seed(&self, _: u64, _: usize) -> Result<u64> {
            anyhow::bail!("not executed")
        }
    }
    let mode = Arc::new(AtomicUsize::new(0));
    let context = CaptureContext {
        binding,
        provider: Arc::new(Provider {
            native,
            mode: mode.clone(),
        }),
    };
    context.validate(&plan).unwrap();
    let dsls = vec![dsl.into()];
    let observed = context.observe(0, &dsls, &[]).unwrap();
    assert_eq!(observed.programs[0].proj16, actual.proj16);
    for change in 1..=3 {
        mode.store(change, Ordering::SeqCst);
        assert!(
            context.observe(0, &dsls, &[]).is_err(),
            "changed native observation {change}"
        );
    }
}
