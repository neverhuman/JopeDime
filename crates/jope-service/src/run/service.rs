//! Backend preparation used by the live RunService, independent of HTTP parsing.
use super::{
    landmarks::FullFitLandmarks,
    learned::{AdmittedRanker, LearnedAdmission},
};
use crate::{
    discover::r#loop::{
        Executor, HeuristicAdmission, Plan, ProcessExecutor, RoundAdmission, RoundContext,
    },
    ranker::admission::{resolve_unqualified, GuidanceUnavailableReason},
};
use anyhow::{ensure, Result};
use jope_api::{
    error::ErrorCode, Guidance, LandmarkMode, Placement, ResolvedGuidance, RunSpec, Task, Warning,
    WarningState,
};
use jope_runtime::backend::DeviceRequest;
use rand_chacha::ChaCha12Rng;

/// Reuse the established runtime objective without changing the surrogate
/// channel. This numeric random-split path has no ordered or auxiliary columns.
pub(crate) fn run_objective(
    spec: &jope_api::ObjectiveSpec,
) -> Result<(String, jope_api::ResolvedObjective)> {
    use jope_runtime::fitness::{Direction, Fitness};
    let text = match spec {
        jope_api::ObjectiveSpec::Name(name) => name.clone(),
        jope_api::ObjectiveSpec::Specification(value) => serde_json::to_string(value)?,
    };
    let objective = Fitness::parse(&text)?;
    ensure!(
        !objective.requires_order(),
        "objective requires an ordered split"
    );
    jope_runtime::fitness::require_no_columns(&objective, "numeric run")?;
    let trained_loss = matches!(objective, Fitness::Rmse {});
    let name = objective.name();
    let public_name = if Fitness::catalog().iter().any(|entry| entry.name == name) {
        name
    } else {
        "custom objective".into()
    };
    Ok((
        text,
        jope_api::ResolvedObjective {
            name: public_name,
            digest: objective.identity()?,
            direction: match objective.direction() {
                Direction::Minimize => jope_api::Direction::Minimize,
                Direction::Maximize => jope_api::Direction::Maximize,
            },
            fitting: if trained_loss {
                "squared_error"
            } else {
                "selection_only"
            }
            .into(),
        },
    ))
}

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
    landmark_mode: LandmarkMode,
    pub full_fit_landmarks: Option<FullFitLandmarks>,
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
                "{}: admitted learned guidance requires regression/full landmarks",
                ErrorCode::CapabilityUnavailable
            );
            let prior_only = !bundle.trains_objective(&spec.objective);
            let warnings = if prior_only {
                vec![crate::warnings::catalog("OBJECTIVE_SURROGATE_MISMATCH")
                    .expect("warning is catalogued")
                    .event(
                        WarningState::Raised, None, None,
                        "The objective is outside the model training set; only surrogate guidance is used.",
                        "Use heuristic guidance to omit the surrogate prior.",
                    )?]
            } else {
                vec![]
            };
            let graph = bundle.instantiate(device, workers)?;
            Ok(Self {
                guidance: ResolvedGuidance {
                    value: Guidance::Learned,
                    prior_only,
                    reason: "Learned guidance uses the configured regression model.".into(),
                },
                warnings,
                placement: graph.placement(),
                bundle_digest: bundle.digest().trim_start_matches("sha256:").into(),
                landmark_rows: 0,
                landmark_mode: spec.landmarks,
                full_fit_landmarks: None,
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
                landmark_mode: spec.landmarks,
                full_fit_landmarks: None,
                admission: RunAdmission::Heuristic,
            })
        }
    }
    pub fn bind_fit(&mut self, fit: &[u8], plan: &Plan) -> Result<()> {
        match &mut self.admission {
            RunAdmission::Learned(admission) => {
                self.landmark_rows = admission.bind_fit(fit, plan)?;
            }
            RunAdmission::Heuristic if self.landmark_mode == LandmarkMode::Full => {
                // Full requests still compute the real CPU recipe when Auto has
                // resolved to heuristic. This does not enable a learned ranker.
                let measured = FullFitLandmarks::extract(fit, plan)?;
                self.landmark_rows = measured.sampled_fit_rows.len();
                self.full_fit_landmarks = Some(measured);
            }
            RunAdmission::Heuristic => {}
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
        // Finalization reuses slot zero and supplies its private refit handoff.
        if self.learned
            && request.attempt.slot < 65_536
            && request.fold_spec.get("refit_output").is_none()
            && request.fold_spec.get("refit_max_bytes").is_none()
        {
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
