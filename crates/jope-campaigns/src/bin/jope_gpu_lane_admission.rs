//! Crash-safe admission for the three reserved self-play GPU lanes.

use anyhow::{bail, ensure, Context, Result};
use clap::{Parser, ValueEnum};
use jope_dime_regressor::selfplay_runtime::{
    atomic_json, default_root_str, default_scq_str, now_ms, read_json, scq_json, sha256_file,
};
use jope_dime_regressor::selfplay_shadow::{
    GpuAdmissionStage, GpuLaneAdmissionV1, GpuLaneRole, RecoveryDisposition, Sha256Digest,
    GPU_ADMISSION_RETRY_MS,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant};

include!("jope_gpu_lane_admission/contracts.rs");
include!("jope_gpu_lane_admission/runtime.rs");
include!("jope_gpu_lane_admission/arbitration.rs");
include!("jope_gpu_lane_admission/scheduler.rs");
include!("jope_gpu_lane_admission/tests.rs");
