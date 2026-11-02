//! Real native runs through all five transports: same-run retries retain raw
//! values, and five distinct runs are compared under the accepted R2 field map.
#[path = "golden_mcp.rs"]
mod golden_mcp;
use super::*;
use anyhow::{bail, ensure, Context, Result};
use jope_client::{Http, InProcess, RunResult, Transport};
use jope_service::{
    dataset::register,
    engine::InProcessEngine,
    http::auth::{BearerToken, LocalAuth},
    store::RecordKey,
};
use std::{
    fs,
    io::{BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, Command, Stdio},
    sync::{mpsc, Arc},
    thread,
    time::{Duration, Instant},
};

const BIN: &str = env!("CARGO_BIN_EXE_jopedime");
const KEY: &str = "int01-real-shared-run";
const SOURCE_SHA: &str = "d9a0b97e91ff9d04bc87fefa4671a49c40cee21dfac05be58a7fc7ce08045931";

mod receipts;
mod scenarios;
mod setup;
mod transport;

pub(super) use scenarios::{batch_fallback, shared_run_surfaces};
