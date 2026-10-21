//! Device selection (SPECIFICATION.md §11.5) for what this build can execute.
//!
//! Native methods run on the CPU. This build compiles no CUDA backend, so `--device cuda:N` is a
//! structured backend failure, never a silent CPU run, and `auto` selects the CPU and says why.
//! The driver probe reads only `/proc` and `/dev`: it loads no library and initializes no device,
//! so it is safe on any host.
use crate::Failure;
use anyhow::Result;
use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceRequest {
    Auto,
    Cpu,
    Cuda(u32),
}

impl DeviceRequest {
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "auto" => Some(Self::Auto),
            "cpu" => Some(Self::Cpu),
            _ => text
                .strip_prefix("cuda:")
                .and_then(|n| n.parse().ok())
                .map(Self::Cuda),
        }
    }

    pub fn as_string(self) -> String {
        match self {
            Self::Auto => "auto".into(),
            Self::Cpu => "cpu".into(),
            Self::Cuda(n) => format!("cuda:{n}"),
        }
    }
}

/// Whether this build was compiled with a CUDA backend.
pub const CUDA_COMPILED: bool = false;

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct DeviceChoice {
    pub requested: String,
    pub selected: &'static str,
    pub reason: String,
}

/// Select the device for native methods.
pub fn select(request: DeviceRequest) -> Result<DeviceChoice> {
    match request {
        DeviceRequest::Cpu => Ok(DeviceChoice {
            requested: request.as_string(),
            selected: "cpu",
            reason: "requested".into(),
        }),
        DeviceRequest::Auto => Ok(DeviceChoice {
            requested: request.as_string(),
            selected: "cpu",
            reason: "native methods run on the CPU; this build has no CUDA backend".into(),
        }),
        DeviceRequest::Cuda(n) => Err(Failure::backend(format!(
            "cuda:{n} requested, but this build has no CUDA backend; use --device cpu or auto"
        ))),
    }
}

/// What the host shows about NVIDIA GPUs, without loading a library.
#[derive(Clone, Debug, Serialize)]
pub struct GpuProbe {
    pub driver_version: Option<String>,
    pub device_nodes: Vec<String>,
    pub cuda_backend_compiled: bool,
    pub usable: bool,
    pub reason: String,
}

pub fn probe() -> GpuProbe {
    let driver_version = std::fs::read_to_string("/proc/driver/nvidia/version")
        .ok()
        .and_then(|text| text.lines().next().map(str::to_owned));
    let mut device_nodes: Vec<String> = std::fs::read_dir("/dev")
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .filter_map(|e| e.file_name().into_string().ok())
                .filter(|name| {
                    name.strip_prefix("nvidia")
                        .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
                })
                .collect()
        })
        .unwrap_or_default();
    device_nodes.sort();
    let reason = if !CUDA_COMPILED {
        "this build has no CUDA backend".to_owned()
    } else if driver_version.is_none() {
        "no NVIDIA driver is loaded".to_owned()
    } else {
        "driver present".to_owned()
    };
    GpuProbe {
        driver_version,
        device_nodes,
        cuda_backend_compiled: CUDA_COMPILED,
        usable: false,
        reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_cuda_fails_and_auto_reports_cpu() {
        assert_eq!(DeviceRequest::parse("cuda:1"), Some(DeviceRequest::Cuda(1)));
        assert_eq!(DeviceRequest::parse("gpu"), None);
        assert_eq!(DeviceRequest::parse("cuda:"), None);
        let error = select(DeviceRequest::Cuda(0)).unwrap_err();
        assert_eq!(crate::error::kind_of(&error), crate::FailureKind::Backend);
        let auto = select(DeviceRequest::Auto).unwrap();
        assert_eq!(auto.selected, "cpu");
        assert!(auto.reason.contains("no CUDA backend"));
        assert!(!probe().usable);
    }
}
