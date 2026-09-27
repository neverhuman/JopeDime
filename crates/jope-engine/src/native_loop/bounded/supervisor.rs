//! SmartCluster-backed rank supervision for the bounded run (W7-T05/T07).
//!
//! [`SmartClusterSupervisor`] implements [`Supervisor`] over the host's own SmartCluster daemon
//! (`scqd`), never SSH. Every request goes through the repository's authenticated daemon
//! exchange: the Unix stream's peer uid and executable digest are pinned ([`DaemonPin`]). A rank
//! is a trusted-service job (`jope-bounded-rank` by default, registered in the host's installed
//! daemon configuration) whose argv names the frozen plan, the rank's launch record and its
//! outbox; the daemon owns the process, its cgroup and its device visibility.
//!
//! * **launch** freezes the rank's launch record, then submits the job with the dispatch id as
//!   its idempotency key and label, no automatic retry, the original deadline and the plan's
//!   CPU/RAM ceilings. A GPU rank is refused: SmartCluster cannot pin the planned device UUID
//!   (see the W7-T03 open item), so no GPU rank is launched through it.
//! * **find** returns the job labelled with the dispatch id, so work submitted before a crash
//!   and never journaled is found, never launched twice.
//! * **probe** maps the job state and exit code to [`Liveness`]; **terminate** cancels and
//!   waits for a terminal state.
//! * **retained_result** reads the signed envelope and result bytes the rank published in its
//!   outbox; the coordinator authenticates them before anything is acknowledged.
use super::{
    controller::{Liveness, RankLaunch, Supervisor, SupervisorHandle},
    launch,
    rendezvous::publish_immutable,
};
use crate::eval_registry::native_scq::authenticated_exchange;
use anyhow::{bail, ensure, Context, Result};
use jain_smartcluster::{
    CancelMode, CancelRequest, Environment, GetRequest, IdempotencyKey, JobId, JobRecord, JobSpec,
    ListFilter, ListRequest, PriorityClass, ProjectId, Request, ResourceRequest, Response,
    RetryPolicy, ServiceName, SubmitRequest, TenantId, TerminalStatus, Timestamp, Workload,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs,
    io::Read,
    os::unix::fs::OpenOptionsExt,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

pub const SERVICE: &str = "jope-bounded-rank";
pub const SUPERVISOR_NAME: &str = "smartcluster";
pub const DISPATCH_LABEL: &str = "jope.bounded.dispatch_id";
pub const RUN_LABEL: &str = "jope.bounded.run";
pub const RANK_LABEL: &str = "jope.bounded.rank";
pub const PLAN_LABEL: &str = "jope.bounded.plan_sha256";
const MAX_RESULT_ENVELOPE: u64 = 1024 * 1024;

/// The pinned identity of the host's SmartCluster daemon.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DaemonPin {
    pub socket: PathBuf,
    pub uid: u32,
    pub executable_sha256: String,
}
/// What this host's supervisor needs to start its ranks. Every path is local to the host.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct HostSupervision {
    /// The planned host this supervisor runs on; it launches no rank planned elsewhere.
    pub host: String,
    pub daemon: DaemonPin,
    pub tenant: String,
    pub project: String,
    /// The trusted service the daemon maps to the installed `jope-native-loop`.
    pub service: String,
    /// The frozen plan artifact on this host.
    pub plan: PathBuf,
    /// Per-dispatch directories (launch record, outbox) owned by this supervisor.
    pub state: PathBuf,
    /// The coordinator's rendezvous address (`host:port`).
    pub coordinator: String,
    /// The participant's host installation description.
    pub installation: PathBuf,
    /// The signing key of each rank planned on this host.
    pub participant_keys: BTreeMap<u32, PathBuf>,
}

/// Acts only on this plan's jobs: every job it launches, finds, probes, cancels or reads is
/// of its tenant and project and carries the digest of the host's frozen plan artifact.
pub struct SmartClusterSupervisor {
    host: HostSupervision,
    tenant: TenantId,
    project: ProjectId,
    plan_sha256: String,
    terminate_timeout: Duration,
}
impl SmartClusterSupervisor {
    pub fn new(host: HostSupervision) -> Result<Self> {
        for p in [
            &host.plan,
            &host.state,
            &host.installation,
            &host.daemon.socket,
        ] {
            ensure!(p.is_absolute(), "{} must be absolute", p.display());
        }
        ensure!(
            super::is_digest(&host.daemon.executable_sha256),
            "daemon executable digest"
        );
        ServiceName::new(host.service.clone())?;
        // Its ranks prove their job against the daemon the host installation pins (W7-D1);
        // a supervisor submitting to another daemon or service would only launch refusals.
        let installation = super::participant::Installation::load(&host.installation)?;
        ensure!(
            installation.smartcluster.daemon == host.daemon
                && installation.smartcluster.service == host.service,
            "the host installation pins another SmartCluster daemon or service than this supervision"
        );
        let plan = launch::read_bounded(&host.plan, super::contract::MAX_CONTRACT_BYTES as u64)?;
        Ok(Self {
            tenant: TenantId::new(host.tenant.clone())?,
            project: ProjectId::new(host.project.clone())?,
            plan_sha256: crate::manifest::hash(&plan),
            host,
            terminate_timeout: Duration::from_secs(60),
        })
    }
    fn exchange(&self, request: Request) -> Result<Response> {
        let d = &self.host.daemon;
        authenticated_exchange(&d.socket, d.uid, &d.executable_sha256, None, request)
    }
    fn job(&self, handle: &SupervisorHandle) -> Result<Option<JobRecord>> {
        ensure!(
            handle.supervisor == SUPERVISOR_NAME,
            "handle belongs to another supervisor"
        );
        let job_id: JobId = handle
            .identity
            .parse()
            .map_err(|_| anyhow::anyhow!("handle is not a SmartCluster job id"))?;
        let job = match self.exchange(Request::Get(GetRequest { job_id }))? {
            Response::Job(job) => job,
            _ => bail!("SmartCluster get response mismatch"),
        };
        if let Some(job) = &job {
            ensure!(
                job.spec.tenant == self.tenant
                    && job.spec.project == self.project
                    && job.spec.labels.get(PLAN_LABEL) == Some(&self.plan_sha256)
                    && job.spec.labels.contains_key(DISPATCH_LABEL),
                "SmartCluster job {} is not a job of this supervisor's plan",
                handle.identity
            );
        }
        Ok(job)
    }
    fn dispatch_dir(&self, dispatch_id: &str) -> Result<PathBuf> {
        ensure!(super::is_digest(dispatch_id), "dispatch id");
        Ok(self.host.state.join(dispatch_id))
    }
    /// The participant argv for one launch. It names only frozen, host-local artifacts.
    fn arguments(&self, l: &RankLaunch<'_>, dir: &Path) -> Result<Vec<String>> {
        let key = self
            .host
            .participant_keys
            .get(&l.global_rank)
            .with_context(|| {
                format!("no participant key for rank {} on this host", l.global_rank)
            })?;
        let s = |p: &Path| p.display().to_string();
        Ok(vec![
            "participant".into(),
            "--plan".into(),
            s(&self.host.plan),
            "--plan-sha256".into(),
            l.plan_sha256.into(),
            "--launch-record".into(),
            s(&dir.join("launch.json")),
            "--outbox".into(),
            s(&dir.join("outbox")),
            "--coordinator".into(),
            self.host.coordinator.clone(),
            "--participant-key".into(),
            s(key),
            "--installation".into(),
            s(&self.host.installation),
        ])
    }
    fn liveness(job: &JobRecord) -> Liveness {
        match &job.terminal {
            None => Liveness::Running,
            Some(TerminalStatus::Completed { exit_code, .. }) => Liveness::Exited {
                code: Some(*exit_code),
                signal: None,
            },
            Some(TerminalStatus::Failed { exit_code, .. }) => Liveness::Exited {
                code: *exit_code,
                signal: None,
            },
            Some(TerminalStatus::Cancelled { .. }) => Liveness::Exited {
                code: None,
                signal: Some(libc::SIGTERM),
            },
        }
    }
}
fn read_retained(path: &Path, limit: u64) -> Result<Option<Vec<u8>>> {
    let f = match fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC)
        .open(path)
    {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e.into()),
    };
    ensure!(
        f.metadata()?.is_file(),
        "{} is not a regular file",
        path.display()
    );
    let mut b = Vec::new();
    f.take(limit + 1).read_to_end(&mut b)?;
    ensure!(
        b.len() as u64 <= limit,
        "{} exceeds {limit} bytes",
        path.display()
    );
    Ok(Some(b))
}

impl SmartClusterSupervisor {
    /// Freeze one rank's launch record under its dispatch directory and return the
    /// directory and the job this supervisor submits for it. Nothing is submitted.
    pub fn launch_spec(&self, l: &RankLaunch<'_>) -> Result<(PathBuf, JobSpec)> {
        let p = l
            .contract
            .participant(l.global_rank)
            .context("rank outside the plan")?;
        ensure!(
            p.host.host == self.host.host,
            "rank {} is planned on host `{}`, not this supervisor's host `{}`",
            l.global_rank,
            p.host.host,
            self.host.host
        );
        ensure!(
            p.device.is_none(),
            "rank {} plans a GPU; SmartCluster cannot pin the planned device UUID, so GPU ranks are not launched through it",
            l.global_rank
        );
        let plan =
            launch::read_bounded(&self.host.plan, super::contract::MAX_CONTRACT_BYTES as u64)?;
        ensure!(
            crate::manifest::hash(&plan) == l.plan_sha256 && self.plan_sha256 == l.plan_sha256,
            "this host's plan artifact differs from the reviewed plan"
        );
        let record = launch::launch_record(l.contract, l.plan_sha256, l.global_rank)?;
        ensure!(
            record.dispatch_id == l.dispatch_id,
            "dispatch id differs from the plan's"
        );
        let dir = self.dispatch_dir(l.dispatch_id)?;
        fs::create_dir_all(&self.host.state)?;
        fs::create_dir(&dir)
            .with_context(|| format!("dispatch {} was already launched", l.dispatch_id))?;
        fs::create_dir(dir.join("outbox"))?;
        publish_immutable(
            &dir,
            Path::new("launch.json"),
            &serde_json::to_vec(&record)?,
        )?;
        let spec = JobSpec {
            tenant: self.tenant.clone(),
            project: self.project.clone(),
            idempotency_key: IdempotencyKey::new(l.dispatch_id)?,
            priority: PriorityClass::Batch,
            workload: Workload::TrustedService {
                service: ServiceName::new(self.host.service.clone())?,
                arguments: self.arguments(l, &dir)?,
                environment: Environment::default(),
            },
            resources: ResourceRequest::cpu_and_ram(
                p.ceilings.cpu_millicores,
                p.ceilings.ram_bytes,
            ),
            retry: RetryPolicy {
                max_failures: 0,
                ..RetryPolicy::default()
            },
            preemption: Default::default(),
            io: Default::default(),
            deadline: Some(Timestamp::from(l.contract.original_deadline_unix_ms)),
            labels: BTreeMap::from([
                (DISPATCH_LABEL.to_string(), l.dispatch_id.to_string()),
                (RUN_LABEL.to_string(), l.contract.run.clone()),
                (RANK_LABEL.to_string(), l.global_rank.to_string()),
                (PLAN_LABEL.to_string(), l.plan_sha256.to_string()),
            ]),
        };
        Ok((dir, spec))
    }
}

impl Supervisor for SmartClusterSupervisor {
    fn launch(&mut self, l: &RankLaunch<'_>) -> Result<SupervisorHandle> {
        let (_, spec) = self.launch_spec(l)?;
        let job = match self.exchange(Request::Submit(Box::new(SubmitRequest {
            spec,
            effective_principal: None,
        })))? {
            Response::Submitted(job) => job,
            _ => bail!("SmartCluster submit response mismatch"),
        };
        ensure!(
            job.spec.labels.get(DISPATCH_LABEL).map(String::as_str) == Some(l.dispatch_id),
            "SmartCluster returned another job for this dispatch"
        );
        Ok(SupervisorHandle {
            supervisor: SUPERVISOR_NAME.into(),
            identity: job.id.to_string(),
        })
    }
    fn find(&mut self, dispatch_id: &str) -> Result<Option<SupervisorHandle>> {
        let request = ListRequest {
            filter: ListFilter {
                tenant: Some(self.tenant.clone()),
                project: Some(self.project.clone()),
                labels: BTreeMap::from([
                    (DISPATCH_LABEL.to_string(), dispatch_id.to_string()),
                    (PLAN_LABEL.to_string(), self.plan_sha256.clone()),
                ]),
                ..ListFilter::default()
            },
            cursor: None,
            limit: 10,
        };
        let page = match self.exchange(Request::List(request))? {
            Response::Jobs(page) => page,
            _ => bail!("SmartCluster list response mismatch"),
        };
        match page.jobs.as_slice() {
            [] => Ok(None),
            [job] => Ok(Some(SupervisorHandle {
                supervisor: SUPERVISOR_NAME.into(),
                identity: job.id.to_string(),
            })),
            _ => bail!("dispatch {dispatch_id} has more than one SmartCluster job"),
        }
    }
    fn probe(&mut self, handle: &SupervisorHandle) -> Result<Liveness> {
        Ok(match self.job(handle)? {
            None => Liveness::Gone,
            Some(job) => Self::liveness(&job),
        })
    }
    fn terminate(&mut self, handle: &SupervisorHandle) -> Result<Liveness> {
        let Some(job) = self.job(handle)? else {
            return Ok(Liveness::Gone);
        };
        if job.terminal.is_none() {
            if let Err(e) = self.exchange(Request::Cancel(CancelRequest {
                job_id: job.id,
                mode: CancelMode::Graceful { grace_ms: 2_000 },
            })) {
                // A job that became terminal meanwhile cannot be cancelled; any other
                // failure leaves the rank running and is an error.
                if self.probe(handle)? == Liveness::Running {
                    return Err(e);
                }
            }
        }
        let start = Instant::now();
        loop {
            let last = self.probe(handle)?;
            if last != Liveness::Running {
                return Ok(last);
            }
            ensure!(
                start.elapsed() < self.terminate_timeout,
                "SmartCluster job {} did not stop within {:?}",
                handle.identity,
                self.terminate_timeout
            );
            std::thread::sleep(Duration::from_millis(50));
        }
    }
    fn retained_result(&mut self, handle: &SupervisorHandle) -> Result<Option<(Vec<u8>, Vec<u8>)>> {
        let job = self.job(handle)?.context("SmartCluster job absent")?;
        let dispatch_id = job
            .spec
            .labels
            .get(DISPATCH_LABEL)
            .context("job carries no dispatch id")?;
        let outbox = self.dispatch_dir(dispatch_id)?.join("outbox");
        match (
            read_retained(&outbox.join("envelope.json"), MAX_RESULT_ENVELOPE)?,
            read_retained(
                &outbox.join("result.bin"),
                crate::native_loop::bounded::contract::MAX_RESULT_BYTES,
            )?,
        ) {
            (Some(envelope), Some(bytes)) => Ok(Some((envelope, bytes))),
            _ => Ok(None),
        }
    }
}
