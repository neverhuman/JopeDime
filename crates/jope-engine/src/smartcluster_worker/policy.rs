//! Native SmartCluster execution. Legacy funding gates remain unchanged.
use crate::{manifest::*, resident_protocol as resident};
use anyhow::{ensure, Context, Result};
use jain_smartcluster::{
    jope::{TaskKind, TaskManifest},
    Client, ControlMessage, FencedAttempt, JobRecord, JobState, ProtocolVersion,
    StructuredProgress, Timestamp, Validate, WorkerControlFrame, WorkerReport, WorkerReportFrame,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs,
    io::{Read, Write},
    os::{
        fd::{AsRawFd, FromRawFd},
        unix::{
            fs::{MetadataExt, OpenOptionsExt},
            net::UnixStream,
        },
    },
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc,
    },
    time::Duration,
};

pub const POLICY_PATH: &str = "/etc/jope-smartcluster/worker.json";
pub const CONTROL_NAME: &str = "jope-subtask-v1";
const FRAME_LIMIT: usize = 1024 * 1024;

mod auth;
mod operation;
mod schema;
pub use auth::runtime_observation;
pub(crate) use auth::verify_task_inputs;
use auth::{
    authenticate, current_job, load_policy, load_task, read_bounded, retain_control_descriptor,
    verify_loaded_runtime, Allocation,
};
use operation::{
    admission_guards, available_storage_bytes, storage_bytes, storage_bytes_from_stats,
    validate_operation, verify_required_inputs, verify_training_closure,
};
use schema::read_closure;
pub use schema::{
    ClosureManifest, InputManifest, Operation, SubtaskCommand, VerifiedTask, WorkerPolicy,
    INPUT_SCHEMA_V1,
};

#[cfg(test)]
mod storage_measurement_tests {
    use super::{available_storage_bytes, storage_bytes, storage_bytes_from_stats};

    #[test]
    fn storage_byte_calculation_uses_available_blocks_and_saturates() {
        assert_eq!(storage_bytes(3, 4096), 12_288);
        assert_eq!(storage_bytes(u64::MAX, 2), u64::MAX);
    }

    #[test]
    fn available_storage_uses_available_blocks_and_fragment_size() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let stats = nix::sys::statvfs::statvfs(directory.path()).expect("filesystem stats");

        assert_eq!(
            storage_bytes_from_stats(stats),
            stats
                .blocks_available()
                .saturating_mul(stats.fragment_size())
        );
    }

    #[test]
    fn unavailable_storage_keeps_admission_error_context() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let missing = directory.path().join("missing");

        let error = available_storage_bytes(&missing).expect_err("missing path is unavailable");

        assert!(format!("{error:#}").contains("storage measurement unavailable"));
    }
}

#[path = "execution.rs"]
mod execution;
pub use execution::{freeze_dispatch_intent, serve};

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
