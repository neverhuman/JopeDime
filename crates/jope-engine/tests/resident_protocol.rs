//! Synthetic process/socket tests; these do not initialize CUDA or qualify NCCL.
use jope_parallel::manifest;
use jope_parallel::resident_protocol as protocol;
use manifest::{read_json, write_json};
use protocol::*;
use serde_json::{json, Value};
use std::{
    fs,
    os::unix::net::{UnixListener, UnixStream},
    process::{Command, Stdio},
    sync::{Arc, Barrier},
    time::{Duration, Instant},
};

include!("resident_protocol/setup.rs");
include!("resident_protocol/protocol_cases.rs");
include!("resident_protocol/readiness.rs");
