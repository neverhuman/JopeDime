//! Backend preparation used by the live RunService, independent of HTTP parsing.
use super::learned::{AdmittedRanker, LearnedAdmission};
use crate::{
    discover::r#loop::{
        Executor, HeuristicAdmission, Plan, ProcessExecutor, RoundAdmission, RoundContext,
    },
    ranker::{
        admission::{resolve_unqualified, GuidanceUnavailableReason},
        cuda::release::R782,
    },
};
use anyhow::{ensure, Result};
use jope_api::{
    error::ErrorCode, Guidance, LandmarkMode, Placement, ResolvedGuidance, RunSpec, Task, Warning,
};
use jope_runtime::backend::DeviceRequest;
use rand_chacha::ChaCha12Rng;

pub(crate) enum RunAdmission {
    Heuristic,
    Learned(Box<LearnedAdmission>),
}

pub(crate) fn error_code(error: &anyhow::Error) -> ErrorCode {
    if error
        .to_string()
        .starts_with(ErrorCode::CapabilityUnavailable.as_str())
    {
        ErrorCode::CapabilityUnavailable
    } else if error.is::<crate::ranker::bundle::BundleIntegrityError>() {
        ErrorCode::ModelIntegrityFailed
    } else {
        match jope_runtime::error::kind_of(error) {
            jope_runtime::FailureKind::Backend => ErrorCode::BackendUnavailable,
            jope_runtime::FailureKind::Integrity => ErrorCode::ModelIntegrityFailed,
            _ => ErrorCode::InvalidInput,
        }
    }
}

pub(crate) struct BackendSpace<N> {
    pub native: N,
    pub learned: bool,
}
impl<N: crate::discover::r#loop::SearchSpace> crate::discover::r#loop::SearchSpace
    for BackendSpace<N>
{
    fn canonical(&self, dsl: &str) -> Result<String> {
        self.native.canonical(dsl)
    }
    fn seeds(&self) -> Result<Vec<String>> {
        if self.learned {
            crate::discover::r#loop::NativeSpace.seeds()
        } else {
            self.native.seeds()
        }
    }
    fn propose(
        &self,
        parents: &[String],
        count: usize,
        rng: &mut ChaCha12Rng,
    ) -> Result<Vec<String>> {
        if !self.learned {
            return self.native.propose(parents, count, rng);
        }
        let mut seen: std::collections::BTreeSet<_> = parents.iter().cloned().collect();
        let mut output = Vec::new();
        for _ in 0..64 {
            for dsl in self.native.propose(parents, count, rng)? {
                let dsl = self.native.canonical(&dsl)?;
                if seen.insert(dsl.clone()) {
                    output.push(dsl);
                }
                if output.len() == count {
                    return Ok(output);
                }
            }
        }
        anyhow::bail!("unable to build a complete fresh learned slate")
    }
}
pub(crate) struct BackendPlan {
    pub guidance: ResolvedGuidance,
    pub warnings: Vec<Warning>,
    pub placement: Placement,
    pub bundle_digest: String,
    pub landmark_rows: usize,
    pub admission: RunAdmission,
}
impl BackendPlan {
    pub fn resolve(
        spec: &RunSpec,
        admitted: Option<&AdmittedRanker>,
        workers: u32,
    ) -> Result<Self> {
        ensure!(workers > 0, "workers must be positive");
        let device = DeviceRequest::parse(&spec.device)
            .ok_or_else(|| jope_runtime::Failure::input("Unknown device"))?;
        if spec.guidance == Guidance::Learned {
            let bundle = admitted.ok_or_else(|| {
                anyhow::anyhow!(
                    "{}: no admitted signed bundle",
                    ErrorCode::CapabilityUnavailable
                )
            })?;
            ensure!(
                spec.task == Task::Regression && spec.landmarks == LandmarkMode::Full,
                "{}: admitted r782 requires regression/full landmarks",
                ErrorCode::CapabilityUnavailable
            );
            let graph = bundle.instantiate(device, workers)?;
            Ok(Self {
                guidance: ResolvedGuidance {
                    value: Guidance::Learned,
                    prior_only: false,
                    reason: "Learned guidance uses the configured regression model.".into(),
                },
                warnings: vec![],
                placement: graph.placement(),
                bundle_digest: R782.trim_start_matches("sha256:").into(),
                landmark_rows: 0,
                admission: RunAdmission::Learned(Box::new(graph)),
            })
        } else {
            jope_runtime::backend::select(device)?;
            let (guidance, warnings) = resolve_unqualified(
                spec.guidance,
                spec.landmarks,
                GuidanceUnavailableReason::Capability,
            )
            .map_err(|code| anyhow::anyhow!(code.to_string()))?;
            Ok(Self {
                guidance,
                warnings,
                placement: Placement {
                    dope: "none".into(),
                    ranker: "heuristic".into(),
                    evaluation: "cpu".into(),
                    workers,
                },
                bundle_digest: jope_runtime::sha256_hex(b"cpu-native-no-learned-bundle/v1"),
                landmark_rows: 0,
                admission: RunAdmission::Heuristic,
            })
        }
    }
    pub fn bind_fit(&mut self, fit: &[u8], plan: &Plan) -> Result<()> {
        if let RunAdmission::Learned(admission) = &mut self.admission {
            self.landmark_rows = admission.bind_fit(fit, plan)?;
        }
        Ok(())
    }
    pub fn learned(&self) -> bool {
        matches!(self.admission, RunAdmission::Learned(_))
    }
}
impl RoundAdmission for RunAdmission {
    fn admit_round(
        &self,
        context: &RoundContext<'_>,
        proposals: Vec<String>,
        explore: bool,
        rng: &mut ChaCha12Rng,
    ) -> Result<Vec<String>> {
        match self {
            Self::Heuristic => HeuristicAdmission.admit_round(context, proposals, explore, rng),
            Self::Learned(graph) => graph.admit_round(context, proposals, explore, rng),
        }
    }
}

/// Keep process isolation and the charged attempt identity. Only rung-0 calls
/// use the frozen recipe; promotion and final refit keep their native objective.
pub(crate) struct BackendExecutor<'a> {
    pub inner: &'a ProcessExecutor,
    pub learned: bool,
}
impl Executor for BackendExecutor<'_> {
    fn evaluate(&self, request: &crate::worker::Request) -> crate::worker::Response {
        if self.learned && request.attempt.slot < 65_536 {
            let mut request = request.clone();
            request.fold_spec["version"] = serde_json::json!(2);
            self.inner.evaluate(&request)
        } else {
            self.inner.evaluate(request)
        }
    }
    fn fitted_model(&self, request: &crate::worker::Request) -> Result<Vec<u8>> {
        self.inner.fitted_model(request)
    }
    fn revoke(&self, attempt: &crate::worker::Attempt) {
        self.inner.revoke(attempt)
    }
}
