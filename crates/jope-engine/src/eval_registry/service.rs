//! Long-lived native authority with service-owned original child supervision.
//!
//! Clients and executors never open SQLite. Every executor receives permission
//! over a kernel-authenticated connection after one durable token consumption.
//! A lost response does not undo consumption or trigger another physical fit.
use super::{
    authority::{authenticated_peer, prepare_observed_completion},
    *,
};
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    os::{
        fd::{AsRawFd, FromRawFd, OwnedFd},
        unix::{
            ffi::OsStrExt,
            fs::{MetadataExt, OpenOptionsExt, PermissionsExt},
            net::{UnixListener, UnixStream},
            process::CommandExt,
        },
    },
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc::{sync_channel, SyncSender},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};

mod query_worker;
pub use query_worker::execute_query_worker;
mod funding_ipc;
mod native_scq;
mod sandbox;
pub use funding_ipc::{
    funding_admission, funding_request, FundingRequest, VerifiedFundingAdmission,
};
pub(crate) use native_scq::submit as submit_native;
pub use sandbox::WorkerSandbox;

mod client;
mod runtime;
mod transport;
mod worker;
pub use client::{query_evaluate, query_retrieve, stats, submit, submit_to, verify_original};
pub(super) use client::{validate_query_response, validate_response};
pub(super) use transport::*;
pub use worker::{await_start_from_environment, WorkerPermit};
pub(super) use worker::{result_from_original, supervise};

include!("service/contracts.rs");
include!("service/protocol.rs");
include!("service/state.rs");
include!("service/connection.rs");
include!("service/client_api.rs");

#[cfg(all(test, not(feature = "neural")))]
mod funding_tests;
#[cfg(test)]
mod tests;
