//! The prediction-API latency stage of the GPU timing study (W11), run by the CUDA controller
//! after the full-GP phase. The stage itself needs no GPU.
//!
//! For every dataset's warm export and each backend whose export completed and whose bundle still
//! hashes to its receipt, the bundle is materialized into a fresh
//! `prediction-<dataset>-<backend>-<rows>` directory for 1, 32 and 1024 rows (the first fitting
//! row repeated; no scoring labels). The emitted latency harness is compiled if the backend needs
//! it and run once for ten timed public-API calls. The receipt says either completed, with
//! exact-bit parity against the qualification plan's first expected value, or failed, with the
//! cause. An existing directory is an interrupted attempt and is never replaced, and the stage
//! stops at the cutoff. Receipts and generated files keep the bytes of the retired Python driver,
//! including its failure texts for spawn errors, timeouts, exits and unreadable samples.

use super::reporting::{
    at, dumps, exists, float_repr, get_or, index, is, py_iter, py_repr, py_str, sha256_hex,
    str_repr, Json, Object, Style, EMPTY_ARRAY, EMPTY_OBJECT,
};
use anyhow::{bail, ensure, Context, Result};
use std::{
    fs,
    io::Read,
    path::{Component, Path, PathBuf},
    process::{Command, ExitStatus, Stdio},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

#[path = "inference_prediction/compat.rs"]
mod compat;
#[path = "inference_prediction/runner.rs"]
mod runner;

use compat::*;
pub(super) use runner::{run, run_process, stage, strings};

#[cfg(test)]
#[path = "inference_prediction/tests.rs"]
mod tests;
