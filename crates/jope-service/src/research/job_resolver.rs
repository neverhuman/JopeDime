//! Opaque, exact-Plan lookup for previously accepted research jobs.
//!
//! This module deliberately does not load a registry from a caller-selected path,
//! S3 URI, or JSON document. A trusted admission loader must verify the accepted
//! receipt and construct an [`AcceptedJobAdmission`] inside the service before an
//! entry can be registered. No production loader is available until the corpus,
//! evaluator, gate, conditioning, and seed receipts are accepted.

use super::{capture::CaptureContext, RecordEvidence};
use crate::discover::r#loop::Plan;
use anyhow::{ensure, Context, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, sync::Arc};

/// Opaque job admission produced only after trusted receipt verification.
///
/// The crate-private constructor prevents callers from turning an untrusted job
/// document into a trusted registry entry. It also preserves the complete plan
/// identity and receipt digest used at admission.
pub struct AcceptedJobAdmission {
    job_id: String,
    plan_sha256: String,
    qualification_receipt_sha256: String,
    context: Arc<CaptureContext>,
}

impl AcceptedJobAdmission {
    #[allow(dead_code)] // Used by the separately authorized admission loader once its receipts exist.
    pub(crate) fn from_verified_receipts(
        job_id: &str,
        plan: &Plan,
        qualification_receipt_sha256: &str,
        context: Arc<CaptureContext>,
    ) -> Result<Self> {
        validate_job_id(job_id)?;
        ensure!(
            is_sha256(qualification_receipt_sha256),
            "accepted qualification receipt is not SHA-256 pinned"
        );
        ensure!(
            context.binding.identity.evidence == RecordEvidence::Engine,
            "synthetic control cannot be registered as an accepted job"
        );
        context
            .validate(plan)
            .context("accepted job context is stale")?;
        Ok(Self {
            job_id: job_id.to_owned(),
            plan_sha256: plan_sha256(plan)?,
            qualification_receipt_sha256: qualification_receipt_sha256.to_owned(),
            context,
        })
    }
}

/// An in-memory registry for jobs admitted by a trusted service-side loader.
///
/// It is intentionally empty by default. Adding an entry is restricted to this
/// crate so a CLI caller cannot mint an admission. Lookup requires the full
/// current `Plan` and revalidates the opaque capture context before returning it.
#[derive(Default)]
pub struct TrustedJobResolver {
    jobs: BTreeMap<String, AcceptedJobAdmission>,
}

impl TrustedJobResolver {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an admission already verified by the service's trusted loader.
    pub fn register(&mut self, admission: AcceptedJobAdmission) -> Result<()> {
        validate_job_id(&admission.job_id)?;
        ensure!(
            is_sha256(&admission.plan_sha256) && is_sha256(&admission.qualification_receipt_sha256),
            "accepted job pins are invalid"
        );
        ensure!(
            !self.jobs.contains_key(&admission.job_id),
            "job ID is already registered"
        );
        self.jobs.insert(admission.job_id.clone(), admission);
        Ok(())
    }

    /// Resolve an opaque job ID only for the exact admitted plan.
    pub fn resolve(&self, job_id: &str, plan: &Plan) -> Result<Arc<CaptureContext>> {
        validate_job_id(job_id)?;
        let admission = self
            .jobs
            .get(job_id)
            .context("no accepted research job matches this ID")?;
        ensure!(
            admission.plan_sha256 == plan_sha256(plan)?,
            "research job Plan differs from its accepted admission"
        );
        admission
            .context
            .validate(plan)
            .context("accepted research job context is stale")?;
        Ok(Arc::clone(&admission.context))
    }

    pub fn contains(&self, job_id: &str) -> bool {
        self.jobs.contains_key(job_id)
    }
}

fn validate_job_id(value: &str) -> Result<()> {
    ensure!(
        !value.is_empty()
            && value.len() <= 64
            && value.as_bytes()[0].is_ascii_alphanumeric()
            && value
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-')),
        "invalid opaque research job ID"
    );
    Ok(())
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn plan_sha256(plan: &Plan) -> Result<String> {
    #[derive(Serialize)]
    struct PlanPin<'a>(&'a Plan);

    let bytes = serde_json::to_vec(&PlanPin(plan)).context("serialize accepted Plan")?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::research::{
        capture::{Binding, CaptureConfig, CaptureProvider, SlateObservation},
        native_inputs::NativeCaptureInputs,
        CaptureContext, RecordIdentity, RecordTask, Rung0Measurement, DOPE_SHA256,
    };
    use crate::{
        ranker::stage1b::TokenProjection,
        worker::{Request, Response},
    };
    use std::sync::Arc;

    #[test]
    fn job_ids_are_opaque_tokens_not_paths() {
        let too_long = "a".repeat(65);
        for invalid in ["", "../job", "a/b", ".", "job id", too_long.as_str()] {
            assert!(validate_job_id(invalid).is_err(), "accepted {invalid:?}");
        }
        assert!(validate_job_id("job-01_A").is_ok());
    }

    #[test]
    fn empty_resolver_refuses_unknown_jobs() {
        let resolver = TrustedJobResolver::new();
        assert!(!resolver.contains("known-later"));
        assert!(resolver.resolve("known-later", &test_plan()).is_err());
    }

    #[test]
    fn plan_identity_binds_every_serialized_plan_field() {
        let original = test_plan();
        let mut changed = original.clone();
        changed.seed += 1;
        assert_ne!(
            plan_sha256(&original).unwrap(),
            plan_sha256(&changed).unwrap()
        );
    }

    #[test]
    fn accepted_engine_job_resolves_only_for_its_exact_plan() {
        let (plan, context) = engine_context_fixture();
        let admission =
            AcceptedJobAdmission::from_verified_receipts("job-01", &plan, &"d".repeat(64), context)
                .unwrap();

        let mut resolver = TrustedJobResolver::new();
        resolver.register(admission).unwrap();
        assert!(resolver.contains("job-01"));

        let resolved = resolver.resolve("job-01", &plan).unwrap();
        assert_eq!(resolved.binding.identity.evidence, RecordEvidence::Engine);

        let mut changed = plan.clone();
        changed.seed += 1;
        assert!(resolver.resolve("job-01", &changed).is_err());
        assert!(resolver.resolve("unknown-job", &plan).is_err());
        assert!(resolver.resolve("../job-01", &plan).is_err());
    }

    #[test]
    fn accepted_job_constructor_rejects_invalid_receipts_and_synthetic_contexts() {
        let (plan, context) = engine_context_fixture();
        assert!(AcceptedJobAdmission::from_verified_receipts(
            "job-01",
            &plan,
            "not-a-sha256",
            Arc::clone(&context),
        )
        .is_err());

        let mut binding = context.binding.clone();
        binding.identity.evidence = RecordEvidence::SyntheticControl;
        let synthetic =
            Arc::new(CaptureContext::control(binding, Arc::clone(&context.provider)).unwrap());
        assert!(AcceptedJobAdmission::from_verified_receipts(
            "job-01",
            &plan,
            &"d".repeat(64),
            synthetic,
        )
        .is_err());

        assert!(AcceptedJobAdmission::from_verified_receipts(
            "../job-01",
            &plan,
            &"d".repeat(64),
            context,
        )
        .is_err());
    }

    #[test]
    fn resolver_refuses_duplicate_accepted_job_ids() {
        let (plan, first_context) = engine_context_fixture();
        let first = AcceptedJobAdmission::from_verified_receipts(
            "job-01",
            &plan,
            &"d".repeat(64),
            first_context,
        )
        .unwrap();
        let (_, second_context) = engine_context_fixture();
        let second = AcceptedJobAdmission::from_verified_receipts(
            "job-01",
            &plan,
            &"e".repeat(64),
            second_context,
        )
        .unwrap();

        let mut resolver = TrustedJobResolver::new();
        resolver.register(first).unwrap();
        assert!(resolver.register(second).is_err());
    }

    /// A resolver contract fixture only: synthetic in-memory metadata, no
    /// authenticated bundle, accepted corpus, qualification receipt, or record.
    fn engine_context_fixture() -> (Plan, Arc<CaptureContext>) {
        let mut plan = test_plan();
        plan.bundle_digest = crate::ranker::cuda::release::R782
            .trim_start_matches("sha256:")
            .to_owned();
        let conditioning = TokenProjection {
            tokens: [[0.25; 32]; 10],
            mask: [1.0; 10],
        };
        let conditioning_sha256 = "e".repeat(64);
        let native = super::super::native_inputs::resolver_test_inputs(
            &plan,
            &conditioning_sha256,
            &conditioning,
        )
        .unwrap();
        let identity = RecordIdentity {
            run_id: plan.run_id.to_string(),
            task: RecordTask::Regression,
            evidence: RecordEvidence::Engine,
            partition: "train".into(),
            source_commit: "9".repeat(40),
            corpus_manifest_sha256: "a".repeat(64),
            dataset_sha256: "b".repeat(64),
            family_sha256: "c".repeat(64),
            problem_sha256: plan.problem_digest.clone(),
            bundle_sha256: plan.bundle_digest.clone(),
            conditioning_sha256,
            evaluator_sha256: "f".repeat(64),
            dope_weights_sha256: DOPE_SHA256.into(),
        };
        let binding = Binding {
            config: CaptureConfig {
                max_attempts: plan.evaluations,
                max_rounds: 1,
                max_bytes: 1_000_000,
                evaluate_all_proposals: false,
                seed_protocol_sha256: "7".repeat(64),
            },
            identity,
            seed: plan.seed,
            tokens: conditioning.tokens.iter().map(|row| row.to_vec()).collect(),
            mask: conditioning
                .mask
                .iter()
                .map(|value| *value == 1.0)
                .collect(),
            matrix_sha256: plan.matrix_sha256.clone(),
        };
        let context =
            CaptureContext::engine(binding, Arc::new(ResolverFixtureProvider { native }), &plan)
                .unwrap();
        (plan, Arc::new(context))
    }

    struct ResolverFixtureProvider {
        native: NativeCaptureInputs,
    }

    impl CaptureProvider for ResolverFixtureProvider {
        fn native_inputs(&self) -> Option<&NativeCaptureInputs> {
            Some(&self.native)
        }

        fn validate_binding(&self, _: &super::super::capture::Binding, _: &Plan) -> Result<()> {
            Ok(())
        }

        fn observe(
            &self,
            _: u64,
            _: &[String],
            _: &[super::super::RecordedReveal],
        ) -> Result<SlateObservation> {
            Ok(SlateObservation {
                programs: vec![],
                revealed_indices: vec![],
            })
        }

        fn measurement(&self, _: &Request, _: &Response) -> Result<Rung0Measurement> {
            anyhow::bail!("resolver fixture does not execute worker measurements")
        }

        fn label_seed(&self, _: u64, _: usize) -> Result<u64> {
            anyhow::bail!("resolver fixture does not execute discovery rounds")
        }
    }

    fn test_plan() -> Plan {
        serde_json::from_value(serde_json::json!({
            "run_id": "run_01JBD8Q6N4V2T5Y8K3M7P9R1S0",
            "dataset_id": "ds_21fbfe8b53dbcd605d79486b",
            "problem_digest": "a".repeat(64),
            "bundle_digest": "b".repeat(64),
            "binary_version": "test",
            "matrix_path": "train.jmat",
            "matrix_sha256": "c".repeat(64),
            "target": "y",
            "features": ["x"],
            "objective": "rmse",
            "seed": 1,
            "planned_workers": 1,
            "effort": "fast",
            "evaluations": 1,
            "wall_seconds": 1,
            "evaluation_wall_ms": 1,
            "evaluation_rss_bytes": 1,
            "groups": null,
            "times": null,
            "gap_rows": 0
        }))
        .unwrap()
    }
}
