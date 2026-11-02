use std::collections::{BTreeMap, BTreeSet};
use std::io;
use std::os::unix::fs::MetadataExt;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc as std_mpsc;
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
pub use idle::QualifiedIdleBaseline;
pub(crate) use idle::qualified_idle_usage;

mod inventory;
pub use inventory::{GpuDevice, GpuInventory, GpuProcess};

mod mps;
pub use mps::{MpsCapability, MpsClient, MpsClientLimits, MpsController, mps_client_matches_pids};

mod events;
pub use events::{GpuEvent, GpuEventKind, GpuEventMonitor};

mod error;
pub use error::GpuError;

#[cfg(test)]
mod tests;
