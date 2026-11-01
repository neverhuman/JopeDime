//! Empty-by-default resolver for research jobs admitted by trusted service code.
//!
//! User-provided paths, JSON and S3 URIs cannot populate this registry. The only
//! insertion path is crate-private and requires an Engine CaptureContext already
//! validated against its exact Plan and accepted receipt identity.

use super::{capture::CaptureContext, RecordEvidence};
use crate::discover::r#loop::Plan;
use anyhow::{ensure, Context, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, sync::Arc};

pub(crate) struct AcceptedJobAdmission {
    job_id: String,
    plan_sha256: String,
    receipt_sha256: String,
    context: Arc<CaptureContext>,
}

impl AcceptedJobAdmission {
    /// The trusted loader calls this only after authenticating the immutable
    /// job/cohort/evaluator receipt chain. A digest-shaped string alone is not
    /// treated as proof; this API remains crate-private.
    pub(crate) fn from_verified_receipts(
        job_id: &str,
        plan: &Plan,
        receipt_sha256: &str,
        context: Arc<CaptureContext>,
    ) -> Result<Self> {
        validate_job_id(job_id)?;
        ensure!(
            is_sha256(receipt_sha256),
            "accepted job receipt is not SHA-256 pinned"
        );
        ensure!(
            context.binding.identity.evidence == RecordEvidence::Engine,
            "synthetic control cannot be registered as a job"
        );
        context
            .validate(plan)
            .context("accepted job context is stale")?;
        Ok(Self {
            job_id: job_id.to_owned(),
            plan_sha256: plan_sha256(plan)?,
            receipt_sha256: receipt_sha256.to_owned(),
            context,
        })
    }
}

/// Registry contents exist only in memory and start empty. Resolution returns an
/// opaque context for the exact accepted Plan; IDs are tokens, never paths.
#[derive(Default)]
pub(crate) struct TrustedJobResolver {
    jobs: BTreeMap<String, AcceptedJobAdmission>,
}

impl TrustedJobResolver {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn register(&mut self, admission: AcceptedJobAdmission) -> Result<()> {
        validate_job_id(&admission.job_id)?;
        ensure!(
            is_sha256(&admission.plan_sha256),
            "invalid accepted Plan identity"
        );
        ensure!(
            is_sha256(&admission.receipt_sha256),
            "invalid accepted receipt identity"
        );
        ensure!(
            !self.jobs.contains_key(&admission.job_id),
            "job ID is already registered"
        );
        self.jobs.insert(admission.job_id.clone(), admission);
        Ok(())
    }

    pub(crate) fn resolve(&self, job_id: &str, plan: &Plan) -> Result<Arc<CaptureContext>> {
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

    pub(crate) fn contains(&self, job_id: &str) -> bool {
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
    Ok(format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&PlanPin(plan))?)
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use jope_api::{
        ids::{DatasetId, RunId},
        Effort,
    };

    #[test]
    fn resolver_starts_empty_and_refuses_paths_and_unknown_jobs() {
        let resolver = TrustedJobResolver::new();
        assert!(!resolver.contains("later"));
        assert!(resolver.resolve("later", &test_plan()).is_err());
        let too_long = "a".repeat(65);
        for invalid in ["", "../job", "a/b", ".", "job id", too_long.as_str()] {
            assert!(validate_job_id(invalid).is_err(), "accepted {invalid:?}");
        }
        assert!(validate_job_id("job-01_A").is_ok());
    }

    #[test]
    fn resolver_plan_pin_covers_every_serialized_field() {
        let original = test_plan();
        let mut changed = original.clone();
        changed.seed += 1;
        assert_ne!(
            plan_sha256(&original).unwrap(),
            plan_sha256(&changed).unwrap()
        );
        changed = original.clone();
        changed.matrix_sha256 = "d".repeat(64);
        assert_ne!(
            plan_sha256(&original).unwrap(),
            plan_sha256(&changed).unwrap()
        );
    }

    fn test_plan() -> Plan {
        Plan {
            run_id: "run_01JBD8Q6N4V2T5Y8K3M7P9R1S0".parse::<RunId>().unwrap(),
            dataset_id: "ds_21fbfe8b53dbcd605d79486b".parse::<DatasetId>().unwrap(),
            problem_digest: "a".repeat(64),
            bundle_digest: "b".repeat(64),
            binary_version: "test".into(),
            matrix_path: "train.jmat".into(),
            matrix_sha256: "c".repeat(64),
            target: "y".into(),
            features: vec!["x".into()],
            objective: "rmse".into(),
            seed: 1,
            planned_workers: 1,
            effort: Effort::Fast,
            evaluations: 1,
            wall_seconds: 1,
            evaluation_wall_ms: 1,
            evaluation_rss_bytes: 1024,
            groups: None,
            times: None,
            gap_rows: 0,
        }
    }
}
