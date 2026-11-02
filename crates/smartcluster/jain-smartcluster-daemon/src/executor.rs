use std::collections::BTreeMap;
use std::fs::File;
use std::io;
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::sync::OnceLock;
use std::time::Duration;

use jain_smartcluster_core::{
    ControlMessage, FencedAttempt, IoMode, JainWorkload, PriorityClass, ProtocolVersion,
    ResourceRequest, WorkerControlFrame, WorkerReport, WorkerReportFrame, Workload,
};
use nix::pty::{openpty, Winsize};
use nix::sys::signal::{kill, raise, Signal};
use nix::sys::wait::{waitpid, WaitPidFlag, WaitStatus};
use nix::unistd::{setsid, Pid};
use sha2::{Digest, Sha256};
use thiserror::Error;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::capacity::CpuSet;
use crate::cgroup::{
    AttemptKey, CgroupError, CgroupLimits, CgroupManager, IsolationMode, JobCgroup,
};
use crate::config::{ExecutionConfig, WorkerConfig};
use crate::control::{ControlChannel, ControlPair, DaemonControlFrameError, CONTROL_FD_ENV};

pub const EXEC_HELPER_MARKER: &str = "__scqd_exec_helper";

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExecutorError {
    #[error("workload program must be absolute: {0}")]
    ProgramNotAbsolute(PathBuf),
    #[error("workload working directory must be absolute: {0}")]
    WorkingDirectoryNotAbsolute(PathBuf),
    #[error("executor helper must be absolute: {0}")]
    HelperNotAbsolute(PathBuf),
    #[error("trusted service is not configured: {0}")]
    UnknownService(String),
    #[error("workload environment may not override reserved variable {0}")]
    ReservedEnvironment(String),
    #[error("executor received a daemon environment variable outside the MPS allowlist")]
    InvalidDaemonEnvironment,
    #[error("single-allocation executor has stopped")]
    Stopped,
    #[error("cannot spawn executor helper {program}: {source}")]
    Spawn { program: PathBuf, source: io::Error },
    #[error("spawned helper has no pid")]
    NoPid,
    #[error("workload process id is outside the supported range: {0}")]
    ProcessIdOutOfRange(u32),
    #[error("executor helper {0} did not stop before migration")]
    HelperStopTimeout(u32),
    #[error("executor helper reported unexpected wait status: {0:?}")]
    UnexpectedHelperStatus(WaitStatus),
    #[error("waitpid failed: {0}")]
    WaitPid(nix::Error),
    #[error("stopped-helper task failed: {0}")]
    Join(tokio::task::JoinError),
    #[error("cannot continue migrated helper: {0}")]
    Continue(nix::Error),
    #[error("missing child {0} pipe")]
    MissingPipe(&'static str),
    #[error("freezer is unavailable in safe single-allocation mode")]
    NoFreezer,
    #[error("adopted workload has no recoverable control channel")]
    NoControlChannel,
    #[error("attach resize is only valid for PTY workloads")]
    NotPty,
    #[error("invalid PTY dimensions: {rows}x{columns}")]
    InvalidPtySize { rows: u16, columns: u16 },
    #[error("executor helper requires parent, identity, and program arguments")]
    HelperArguments,
    #[error("executor helper received an invalid expected parent process id")]
    HelperParent,
    #[error("executor helper parent changed from {expected} to {actual} during launch fencing")]
    HelperParentChanged { expected: Pid, actual: Pid },
    #[error("executor helper received an invalid worker uid/gid pair")]
    HelperIdentity,
    #[error("setsid failed in executor helper: {0}")]
    SetSid(nix::Error),
    #[error("executor helper could not enable no-new-privileges: {0}")]
    NoNewPrivileges(nix::Error),
    #[error("executor helper could not clear supplementary groups: {0}")]
    ClearSupplementaryGroups(nix::Error),
    #[error("executor helper could not set its group: {0}")]
    SetGid(nix::Error),
    #[error("executor helper could not set its user: {0}")]
    SetUid(nix::Error),
    #[error("executor helper could not stop itself: {0}")]
    StopSelf(nix::Error),
    #[error("executor helper could not arm its parent-death launch fence: {0}")]
    ArmParentDeathSignal(nix::Error),
    #[error("executor helper could not clear its parent-death launch fence: {0}")]
    ClearParentDeathSignal(nix::Error),
    #[error("direct exec of {program} failed: {source}")]
    Exec { program: PathBuf, source: io::Error },
    #[error(transparent)]
    Cgroup(#[from] CgroupError),
    #[error(transparent)]
    Control(#[from] DaemonControlFrameError),
    #[error(transparent)]
    Nix(#[from] nix::Error),
    #[error("terminal operation failed: {0}")]
    Rustix(#[from] rustix::io::Errno),
    #[error(transparent)]
    Io(#[from] io::Error),
}

#[cfg(test)]
mod tests;

mod helper;
mod launch;
mod lifecycle;

pub use helper::run_exec_helper;
pub(super) use launch::terminate_failed_launch;
pub use launch::{ExecutionAllocation, Executor, LaunchPlan, WorkloadResolver};
pub(super) use lifecycle::LaunchGuard;
pub use lifecycle::{
    CancelOutcome, ProcessWaiter, RunningParts, RunningWorkload, WorkloadInput, WorkloadIo,
    WorkloadOutput, WorkloadOutputOwned,
};
