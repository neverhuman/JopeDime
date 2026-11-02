use super::*;

pub const EXECUTION_SCHEMA: &str = "jope.worker-execution/v2";
pub const HOST_LOCK: &str = "/run/lock/jope-exclusive-cpu.lock";
pub(super) const LOG_LIMIT: usize = 1024 * 1024;
pub(super) const POLL_MS: u64 = 10;
pub(super) const DRAIN_MS: u64 = 1000;
pub(super) const THREAD_VARS: &[&str] = &[
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "RAYON_NUM_THREADS",
];

/// Fixed policy for actual fit jobs; reservation-only consumers keep the whole
/// authorized service allocation. Controller and fits share the same CPU set.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FitMemoryAllocation {
    pub schema: String,
    pub experiment_sha256: String,
    pub total_bytes: u64,
    pub workers: u16,
    pub controller_reserve_bytes: u64,
    pub per_fit_bytes: u64,
}
impl FitMemoryAllocation {
    pub fn identity(&self) -> Result<String> {
        Ok(crate::sha256_hex(&serde_json::to_vec(self)?))
    }
}
/// One sixteenth is reserved for controller state, plus page/division remainder.
/// For the frozen 20-worker 64 GiB panel this is 4 GiB plus 20 independent 3 GiB fits.
/// Shares never expand when fewer jobs are active, so scheduling cannot change
/// a predictor's memory allowance. Experiment identity pins resources and source.
pub fn fit_memory_allocation(experiment: &Experiment) -> Result<FitMemoryAllocation> {
    experiment.validate()?;
    const PAGE: u64 = 4096;
    let pages = experiment.resources.memory_bytes / PAGE;
    let reserve = pages.div_ceil(16);
    let workers = u64::from(experiment.resources.workers);
    let fit_pages = pages.saturating_sub(reserve) / workers;
    ensure!(
        fit_pages > 0,
        "memory allocation cannot supply a page to each fit"
    );
    let per_fit_bytes = fit_pages * PAGE;
    Ok(FitMemoryAllocation {
        schema: "jope.fit-memory-allocation/v1:page4096:controller1of16".into(),
        experiment_sha256: experiment.identity()?,
        total_bytes: experiment.resources.memory_bytes,
        workers: experiment.resources.workers,
        controller_reserve_bytes: experiment.resources.memory_bytes - per_fit_bytes * workers,
        per_fit_bytes,
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Artifact {
    pub path: PathBuf,
    pub sha256: String,
}
impl Artifact {
    pub fn verify(&self) -> Result<()> {
        ensure!(
            crate::experiment::sha256(&self.sha256),
            "invalid artifact digest"
        );
        regular(&self.path)?;
        ensure!(
            hash_file(&self.path)? == self.sha256,
            "artifact changed: {}",
            self.path.display()
        );
        after_verification(&self.path);
        Ok(())
    }
    /// [`Artifact::verify`] for a receiver that consumes the artifact: read it
    /// once, require exactly those bytes to be the pin and return them, so the
    /// verified object is the consumed object (W2-T04 G-10e). The path is
    /// never reopened; the refusal wording is `verify`'s.
    pub fn read_verified(&self) -> Result<Vec<u8>> {
        ensure!(
            crate::experiment::sha256(&self.sha256),
            "invalid artifact digest"
        );
        regular(&self.path)?;
        let bytes = std::fs::read(&self.path)?;
        ensure!(
            crate::sha256_hex(&bytes) == self.sha256,
            "artifact changed: {}",
            self.path.display()
        );
        after_verification(&self.path);
        Ok(bytes)
    }
}

/// Test-only instrumentation point between a digest check and consumption of
/// the checked object (W2-T04 route G-10e). Compiles to nothing outside this
/// crate's own unit-test build; it grants and changes nothing.
#[inline]
pub(crate) fn after_verification(_path: &Path) {
    #[cfg(test)]
    #[cfg(feature = "neural-search")]
    verification_hook::fire(_path);
}
#[cfg(test)]
#[cfg(feature = "neural-search")]
pub(crate) mod verification_hook {
    use std::{cell::RefCell, path::Path};
    type Hook = Box<dyn FnMut(&Path)>;
    thread_local!(static HOOK: RefCell<Option<Hook>> = RefCell::new(None));
    /// Install `hook` for this thread until the returned guard drops.
    pub(crate) fn install(hook: impl FnMut(&Path) + 'static) -> Guard {
        HOOK.with(|h| *h.borrow_mut() = Some(Box::new(hook)));
        Guard
    }
    pub(crate) struct Guard;
    impl Drop for Guard {
        fn drop(&mut self) {
            HOOK.with(|h| *h.borrow_mut() = None);
        }
    }
    pub(super) fn fire(path: &Path) {
        let hook = HOOK.with(|h| h.borrow_mut().take());
        if let Some(mut hook) = hook {
            hook(path);
            HOOK.with(|h| {
                let mut slot = h.borrow_mut();
                if slot.is_none() {
                    *slot = Some(hook);
                }
            });
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerCommand {
    pub program: Artifact,
    pub args: Vec<String>,
    /// Includes scripts, the request JSON and every admitted data input. No
    /// scoring target or whole-panel path belongs in this closure.
    pub inputs: Vec<Artifact>,
    /// Explicit environment is identity-bound. Numerical thread variables cannot
    /// override the pool's one-thread setting; credentials are never inherited.
    pub environment: BTreeMap<String, String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArmBudget {
    pub id_sha256: String,
    pub seconds: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerJob {
    pub request: EvaluationRequest,
    pub command: WorkerCommand,
    /// New or previously journaled directory inside the private journal tree.
    pub directory: PathBuf,
    /// Relative output names, including the complete fitted artifact and predictions.
    pub outputs: Vec<PathBuf>,
    pub arm: Option<ArmBudget>,
}
impl WorkerJob {
    pub fn identity(&self) -> Result<String> {
        self.request.validate()?;
        ensure!(!self.outputs.is_empty(), "no declared fit outputs");
        let outputs: BTreeSet<_> = self.outputs.iter().collect();
        ensure!(
            outputs.len() == self.outputs.len(),
            "duplicate declared outputs"
        );
        for path in &self.outputs {
            ensure!(
                !path.as_os_str().is_empty()
                    && path.components().all(|c| matches!(c, Component::Normal(_))),
                "output must be a relative normal path"
            );
            ensure!(
                path != Path::new("stdout.log") && path != Path::new("stderr.log"),
                "output collides with logs"
            );
        }
        if let Some(arm) = &self.arm {
            ensure!(
                crate::experiment::sha256(&arm.id_sha256) && arm.seconds > 0 && arm.seconds <= 120,
                "invalid search arm allowance"
            );
        }
        for (key, value) in &self.command.environment {
            ensure!(
                !key.is_empty()
                    && !key.contains('=')
                    && !key.contains('\0')
                    && !value.contains('\0'),
                "invalid worker environment"
            );
            ensure!(
                !THREAD_VARS.contains(&key.as_str())
                    && !matches!(
                        key.as_str(),
                        "CUDA_VISIBLE_DEVICES"
                            | "TMPDIR"
                            | "HOME"
                            | "GH_TOKEN"
                            | "GITHUB_TOKEN"
                            | "AWS_ACCESS_KEY_ID"
                            | "AWS_SECRET_ACCESS_KEY"
                            | "AWS_SESSION_TOKEN"
                    ),
                "reserved worker environment: {key}"
            );
        }
        Ok(crate::sha256_hex(&serde_json::to_vec(self)?))
    }
    fn verify_inputs(&self) -> Result<()> {
        self.command.program.verify()?;
        ensure!(
            !self.command.inputs.is_empty(),
            "worker input closure is empty"
        );
        for input in &self.command.inputs {
            input.verify()?;
        }
        Ok(())
    }
}

/// Only deduplicate the preliminary inventory pass within one submitted batch.
/// This is not an admission/integrity cache: every actual gated launch and
/// successful exit still calls `WorkerJob::verify_inputs` with full hashes.
/// A file may change after this preflight; that later full check must reject it.
#[derive(Default)]
struct BatchPreflight {
    verified: BTreeSet<(PathBuf, String)>,
}
impl BatchPreflight {
    fn verify(&mut self, job: &WorkerJob) -> Result<()> {
        ensure!(
            !job.command.inputs.is_empty(),
            "worker input closure is empty"
        );
        for artifact in std::iter::once(&job.command.program).chain(&job.command.inputs) {
            let key = (artifact.path.clone(), artifact.sha256.clone());
            if !self.verified.contains(&key) {
                artifact.verify()?;
                self.verified.insert(key);
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DependencyCheck {
    pub job_sha256: String,
    pub request_sha256: String,
    pub available: bool,
    pub reason: Option<String>,
}
/// Verifies the declared executable/input closure without starting a window or a
/// subprocess. Adapter --version/import capability probes remain caller-owned.
pub fn inventory_dependencies(jobs: &[WorkerJob]) -> Result<Vec<DependencyCheck>> {
    jobs.iter()
        .map(|job| {
            let error = job.verify_inputs().err();
            Ok(DependencyCheck {
                job_sha256: job.identity()?,
                request_sha256: job.request.identity()?,
                available: error.is_none(),
                reason: error.map(|e| format!("{e:#}")),
            })
        })
        .collect()
}

/// Null means unknown, never zero. Snapshots before an interrupted controller
/// are lower bounds retained separately from final consumption.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservedConsumption {
    /// Actual per-fit memory.max readback; None for old/unstarted/unknown evidence.
    #[serde(default)]
    pub fit_memory_limit_bytes: Option<u64>,
    #[serde(default)]
    pub gate_released_unix_ms: Option<u64>,
    #[serde(default)]
    pub observed_exit_unix_ms: Option<u64>,
    pub wall_seconds: Option<f64>,
    pub cpu_seconds: Option<f64>,
    pub peak_rss_bytes: Option<u64>,
    pub cgroup_peak_memory_bytes: Option<u64>,
    pub sampled_peak_rss_bytes: Option<u64>,
    pub accounting_reason: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerOutcome {
    pub schema: String,
    pub job_sha256: String,
    pub request_sha256: String,
    /// Clean command exit and intact declared outputs; caller still validates
    /// prediction shape, numerical policy and scores independently.
    pub command_status: EvaluationStatus,
    pub reason: Option<String>,
    pub exit_code: Option<i32>,
    pub signal: Option<i32>,
    pub outputs: Vec<Artifact>,
    pub stdout: Option<Artifact>,
    pub stderr: Option<Artifact>,
    pub consumption: ObservedConsumption,
}
impl WorkerOutcome {
    /// Join the existing qualified child receipt to an I16 reservation. Missing
    /// final counters stay unknown even for killed/failed commands. Only an
    /// explicitly proven unopened gate has local child cost zero.
    pub fn resource_cost_v1(
        &self,
        reservation: &crate::resource_ledger::Reservation,
        ended_unix_us: u64,
    ) -> Result<crate::resource_ledger::AttemptCost> {
        use crate::resource_ledger::{
            AttemptCost, Disposition, Measurement, Reading, Unit, SCHEMA,
        };
        ensure!(
            self.schema == EXECUTION_SCHEMA
                && self.request_sha256 == reservation.request.ids.request_id,
            "worker/resource request mismatch"
        );
        let unstarted = self
            .consumption
            .accounting_reason
            .starts_with("not_started: gate never released")
            && self.consumption.gate_released_unix_ms.is_none()
            && self.consumption.cpu_seconds == Some(0.)
            && self.consumption.wall_seconds == Some(0.);
        let mut measurements = BTreeMap::new();
        for &unit in reservation.request.requested.0.keys() {
            let reading = if unstarted {
                Reading::exact(0)
            } else {
                let value = match unit {
                    Unit::CpuCoreMicros => self
                        .consumption
                        .cpu_seconds
                        .map(crate::resource_ledger::seconds_to_micros)
                        .transpose()?,
                    Unit::AttemptWallMicros => self
                        .consumption
                        .wall_seconds
                        .map(crate::resource_ledger::seconds_to_micros)
                        .transpose()?,
                    Unit::PeakMemoryBytes => self.consumption.cgroup_peak_memory_bytes,
                    Unit::Evaluations | Unit::Compilations | Unit::CustodyOperations => {
                        self.consumption.gate_released_unix_ms.map(|_| 1)
                    }
                    Unit::Retries => self
                        .consumption
                        .gate_released_unix_ms
                        .map(|_| u64::from(reservation.request.retry_of.is_some())),
                    _ => None,
                };
                value.map(Reading::exact).unwrap_or_else(|| {
                    Reading::unknown(format!(
                        "{}; {unit:?} unavailable",
                        self.consumption.accounting_reason
                    ))
                })
            };
            measurements.insert(
                unit,
                Measurement {
                    reading,
                    collector: "worker-execution/v2:final-cgroup".into(),
                    clock: "worker-monotonic-duration/unix-window".into(),
                    allocation: reservation.request.ids.experiment_sha256.clone(),
                    scope: vec!["experiment".into(), self.job_sha256.clone()],
                    from_unix_us: reservation.reserved_unix_us,
                    to_unix_us: ended_unix_us,
                },
            );
        }
        let disposition = if unstarted {
            Disposition::ProvenNotStarted
        } else if self.signal.is_some() {
            Disposition::Killed
        } else {
            match self.command_status {
                EvaluationStatus::Completed => Disposition::Completed,
                EvaluationStatus::Cancelled => Disposition::Cancelled,
                EvaluationStatus::TimedOut => Disposition::Killed,
                _ => Disposition::Failed,
            }
        };
        let mut evidence: BTreeSet<_> = self.outputs.iter().map(|a| a.sha256.clone()).collect();
        evidence.insert(crate::sha256_hex(&serde_json::to_vec(self)?));
        evidence.extend(
            self.stdout
                .iter()
                .chain(&self.stderr)
                .map(|a| a.sha256.clone()),
        );
        Ok(AttemptCost {
            schema: SCHEMA.into(),
            reservation_sha256: reservation.identity()?,
            ids: reservation.request.ids.clone(),
            disposition,
            ended_unix_us,
            exit_code: self.exit_code,
            signal: self.signal,
            measurements,
            retained_sha256: if disposition == Disposition::Completed {
                self.outputs.iter().map(|a| a.sha256.clone()).collect()
            } else {
                BTreeSet::new()
            },
            evidence_sha256: evidence,
            retained_bytes: Reading::unknown(
                "worker receipt binds artifact digests, not retained physical storage bytes",
            ),
            reason: if unstarted {
                self.consumption.accounting_reason.clone()
            } else {
                self.reason
                    .clone()
                    .unwrap_or_else(|| self.consumption.accounting_reason.clone())
            },
        })
    }

    pub(super) fn unstarted(
        job: &WorkerJob,
        status: EvaluationStatus,
        reason: String,
    ) -> Result<Self> {
        Ok(Self {
            schema: EXECUTION_SCHEMA.into(),
            job_sha256: job.identity()?,
            request_sha256: job.request.identity()?,
            command_status: status,
            reason: Some(reason),
            exit_code: None,
            signal: None,
            outputs: vec![],
            stdout: None,
            stderr: None,
            consumption: ObservedConsumption {
                wall_seconds: Some(0.),
                cpu_seconds: Some(0.),
                accounting_reason: "not_started: gate never released".into(),
                ..Default::default()
            },
        })
    }
    pub fn verify_outputs(&self) -> Result<()> {
        ensure!(self.schema == EXECUTION_SCHEMA, "unknown execution receipt");
        for artifact in self
            .outputs
            .iter()
            .chain(self.stdout.iter())
            .chain(self.stderr.iter())
        {
            artifact.verify()?;
        }
        ensure!(
            self.command_status != EvaluationStatus::Completed || !self.outputs.is_empty(),
            "completed command lacks outputs"
        );
        Ok(())
    }
}
