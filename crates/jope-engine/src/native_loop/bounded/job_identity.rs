//! A bounded rank's proof that it runs inside its planned SmartCluster job (W7-D1).
//!
//! Before any work, a participant proves from what `scqd` exposes to the job that it is the
//! process the host's pinned daemon started for exactly its planned job and attempt:
//! 1. the job environment names the attempt (`SCQ_JOB_ID`, `SCQ_ATTEMPT_NUMBER`,
//!    `SCQ_ATTEMPT_FENCE`) and hands over the daemon's control socket (`SCQ_CONTROL_FD`);
//! 2. the peer of that control socket is this process's parent, running the pinned daemon
//!    executable as the pinned daemon uid (scqd's exec helper replaces itself with the job
//!    program, so the daemon is the job leader's parent);
//! 3. the same daemon process, over its authenticated socket, reports the job as live with
//!    exactly this attempt as its active attempt, of tenant `unix:<this process's euid>`,
//!    labelled for this plan, run, rank and dispatch, running the host's trusted service with
//!    exactly this process's arguments.
//!
//! Anything it cannot prove is a refusal: no job environment (not started by SmartCluster),
//! another parent, another daemon, a stale or replayed attempt, another tenant, or a job
//! submitted for another plan, run, rank or dispatch.
use super::supervisor::{DaemonPin, DISPATCH_LABEL, PLAN_LABEL, RANK_LABEL, RUN_LABEL};
use crate::eval_registry::{native_scq::authenticated_exchange, ProcessIdentity};
use anyhow::{bail, ensure, Context, Result};
use jain_smartcluster::{
    AttemptNumber, FenceToken, FencedAttempt, GetRequest, JobId, JobRecord, JobState, Request,
    Response, Workload,
};
use serde::{Deserialize, Serialize};
use std::{
    ffi::{OsStr, OsString},
    os::{fd::FromRawFd, unix::net::UnixStream},
    path::Path,
};

pub const JOB_ID_ENV: &str = "SCQ_JOB_ID";
pub const ATTEMPT_NUMBER_ENV: &str = "SCQ_ATTEMPT_NUMBER";
pub const ATTEMPT_FENCE_ENV: &str = "SCQ_ATTEMPT_FENCE";
pub const CONTROL_FD_ENV: &str = "SCQ_CONTROL_FD";

/// The attempt the job environment claims. Only a claim until [`verify_running_job`] proves it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JobClaim {
    pub attempt: FencedAttempt,
    pub control_fd: i32,
}
impl JobClaim {
    /// Read the claim from the job environment. Any missing or malformed value is a refusal:
    /// the process was not started by SmartCluster as a job.
    pub fn from_env(var: impl Fn(&str) -> Option<OsString>) -> Result<Self> {
        let get = |name: &str| -> Result<String> {
            var(name)
                .with_context(|| {
                    format!("{name} is unset: this process was not started by SmartCluster as a job; a bounded rank runs only inside its planned SmartCluster job")
                })?
                .into_string()
                .map_err(|_| anyhow::anyhow!("{name} is not UTF-8"))
        };
        let job_id: JobId = get(JOB_ID_ENV)?
            .parse()
            .map_err(|_| anyhow::anyhow!("{JOB_ID_ENV} is not a SmartCluster job id"))?;
        let number: u32 = get(ATTEMPT_NUMBER_ENV)?
            .parse()
            .context("SCQ_ATTEMPT_NUMBER is not a number")?;
        let fence: u64 = get(ATTEMPT_FENCE_ENV)?
            .parse()
            .context("SCQ_ATTEMPT_FENCE is not a number")?;
        let control_fd: i32 = get(CONTROL_FD_ENV)?
            .parse()
            .context("SCQ_CONTROL_FD is not a descriptor number")?;
        ensure!(
            control_fd >= 3,
            "SCQ_CONTROL_FD {control_fd} is not a job control descriptor"
        );
        Ok(Self {
            attempt: FencedAttempt::new(
                job_id,
                AttemptNumber::new(number).map_err(|e| anyhow::anyhow!("{e}"))?,
                FenceToken::new(fence).map_err(|e| anyhow::anyhow!("{e}"))?,
            ),
            control_fd,
        })
    }
}

/// What a rank expects its job to be: all of it from the frozen plan, its launch record and
/// this host's installation.
#[derive(Clone, Debug)]
pub struct ExpectedJob {
    pub plan_sha256: String,
    pub run: String,
    pub global_rank: u32,
    pub dispatch_id: String,
    /// `unix:<euid>`: SmartCluster runs a job as its submitting principal.
    pub tenant: String,
    pub service: String,
    /// This process's own arguments (after the program).
    pub arguments: Vec<OsString>,
}

/// The verified identity, recorded in the rank's result.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct VerifiedJob {
    pub job_id: String,
    pub attempt: u32,
    pub fence: u64,
    pub tenant: String,
    pub daemon_pid: u32,
}

/// The daemon's record of the job, checked against the claim and the expectation. Pure, so
/// every refusal is exercised without a daemon.
pub fn verify_job_record(
    claim: &FencedAttempt,
    job: &JobRecord,
    expected: &ExpectedJob,
) -> Result<()> {
    ensure!(
        job.id == claim.job_id,
        "SmartCluster answered with job {}, not this process's job {}",
        job.id,
        claim.job_id
    );
    ensure!(
        matches!(job.state, JobState::Starting | JobState::Running) && job.terminal.is_none(),
        "SmartCluster job {} is {:?}, not live",
        job.id,
        job.state
    );
    ensure!(
        job.active_attempt == Some(*claim),
        "attempt {} (fence {}) is not SmartCluster job {}'s active attempt: a stale or replayed attempt",
        claim.number,
        claim.fence,
        job.id
    );
    ensure!(
        job.spec.tenant.as_str() == expected.tenant,
        "SmartCluster job {} belongs to tenant `{}`, not this process's principal `{}`",
        job.id,
        job.spec.tenant.as_str(),
        expected.tenant
    );
    let label = |name: &str| job.spec.labels.get(name).map(String::as_str);
    ensure!(
        label(PLAN_LABEL) == Some(expected.plan_sha256.as_str())
            && label(RUN_LABEL) == Some(expected.run.as_str()),
        "SmartCluster job {} was submitted for another plan (plan {:?}, run {:?}), not plan {} run {}",
        job.id,
        label(PLAN_LABEL),
        label(RUN_LABEL),
        expected.plan_sha256,
        expected.run
    );
    ensure!(
        label(DISPATCH_LABEL) == Some(expected.dispatch_id.as_str())
            && label(RANK_LABEL) == Some(expected.global_rank.to_string().as_str()),
        "SmartCluster job {} was submitted for another dispatch (dispatch {:?}, rank {:?}), not rank {}'s dispatch {}",
        job.id,
        label(DISPATCH_LABEL),
        label(RANK_LABEL),
        expected.global_rank,
        expected.dispatch_id
    );
    match &job.spec.workload {
        Workload::TrustedService {
            service, arguments, ..
        } => {
            ensure!(
                service.as_str() == expected.service,
                "SmartCluster job {} runs service `{}`, not `{}`",
                job.id,
                service.as_str(),
                expected.service
            );
            ensure!(
                arguments.len() == expected.arguments.len()
                    && arguments
                        .iter()
                        .zip(&expected.arguments)
                        .all(|(a, b)| OsStr::new(a) == b.as_os_str()),
                "SmartCluster job {}'s arguments are not this process's arguments",
                job.id
            );
        }
        _ => bail!("SmartCluster job {} is not a trusted-service job", job.id),
    }
    Ok(())
}

fn parent_pid() -> u32 {
    // SAFETY: getppid has no preconditions.
    unsafe { libc::getppid() as u32 }
}

/// Prove this process is the job leader the pinned daemon started for `expected`. Every
/// step fails closed; see the module documentation.
pub fn verify_running_job(
    daemon: &DaemonPin,
    claim: &JobClaim,
    expected: &ExpectedJob,
) -> Result<VerifiedJob> {
    // The control socket's peer is the daemon that created it for this job. Inspect it
    // through a private duplicate; the inherited descriptor is left untouched.
    // SAFETY: fcntl(F_DUPFD_CLOEXEC) on an integer descriptor has no memory preconditions;
    // a closed or foreign descriptor fails and is refused below.
    let duplicate = unsafe { libc::fcntl(claim.control_fd, libc::F_DUPFD_CLOEXEC, 3) };
    ensure!(
        duplicate >= 0,
        "SCQ_CONTROL_FD {} is not an open descriptor: no SmartCluster control socket",
        claim.control_fd
    );
    // SAFETY: `duplicate` is a fresh descriptor owned only by this stream.
    let control = unsafe { UnixStream::from_raw_fd(duplicate) };
    let peer = crate::resident_protocol::peer_credentials(&control)
        .context("SCQ_CONTROL_FD is not a SmartCluster control socket")?;
    drop(control);
    let daemon_pid = peer.pid as u32;
    ensure!(
        daemon_pid == parent_pid(),
        "this process's parent ({}) is not the SmartCluster daemon ({daemon_pid}) that holds its control socket",
        parent_pid()
    );
    ensure!(
        peer.uid == daemon.uid,
        "the control socket's daemon runs as uid {}, not the pinned daemon uid {}",
        peer.uid,
        daemon.uid
    );
    let daemon_identity = ProcessIdentity::read(daemon_pid)?;
    ensure!(
        crate::manifest::file_hash(Path::new(&format!("/proc/{daemon_pid}/exe")))?
            == daemon.executable_sha256,
        "the control socket's daemon is not the pinned SmartCluster executable"
    );
    // The daemon on the pinned socket must be that same process.
    let job = match authenticated_exchange(
        &daemon.socket,
        daemon.uid,
        &daemon.executable_sha256,
        Some(&daemon_identity),
        Request::Get(GetRequest {
            job_id: claim.attempt.job_id,
        }),
    )
    .context("the pinned SmartCluster daemon did not answer as this process's parent")?
    {
        Response::Job(Some(job)) => job,
        Response::Job(None) => bail!(
            "SmartCluster has no job {}: this process is not running inside it",
            claim.attempt.job_id
        ),
        _ => bail!("SmartCluster get response mismatch"),
    };
    verify_job_record(&claim.attempt, &job, expected)?;
    ensure!(
        ProcessIdentity::read(daemon_pid)? == daemon_identity && parent_pid() == daemon_pid,
        "the SmartCluster daemon changed during the job identity check"
    );
    Ok(VerifiedJob {
        job_id: claim.attempt.job_id.to_string(),
        attempt: claim.attempt.number.get(),
        fence: claim.attempt.fence.get(),
        tenant: expected.tenant.clone(),
        daemon_pid,
    })
}

/// `unix:<euid>` of this process: the only tenant a job it runs in may have.
pub fn own_tenant() -> String {
    // SAFETY: geteuid has no preconditions.
    format!("unix:{}", unsafe { libc::geteuid() })
}
