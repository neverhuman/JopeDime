use std::collections::{BTreeMap, BTreeSet};
use std::io;
use std::os::unix::fs::MetadataExt;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc as std_mpsc;
use std::sync::Arc;
use std::time::Duration;

use nvml_wrapper::bitmasks::event::EventTypes;
use nvml_wrapper::enum_wrappers::device::TemperatureSensor;
use nvml_wrapper::enums::device::UsedGpuMemory;
use nvml_wrapper::error::NvmlError;
use nvml_wrapper::{Device, Nvml};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};
use tokio::process::Command;

use crate::config::MpsConfig;

const VRAM_HEADROOM_PERCENT: u64 = 10;
const MEBIBYTE: u64 = 1024 * 1024;

mod idle;
pub(crate) use idle::qualified_idle_usage;
pub use idle::QualifiedIdleBaseline;

mod inventory;
pub(super) use inventory::{is_visible, parse_visible_devices};
pub use inventory::{GpuDevice, GpuInventory, GpuProcess};

mod mps;
pub(super) use mps::{compute_percent, format_vram_limits, parse_ps, parse_termination_response};
pub use mps::{mps_client_matches_pids, MpsCapability, MpsClient, MpsClientLimits, MpsController};

mod events;
pub use events::{GpuEvent, GpuEventKind, GpuEventMonitor};

mod error;
pub use error::GpuError;

#[cfg(test)]
mod tests;
