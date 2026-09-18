use std::fs;
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use clap::Parser;
use jain_smartcluster::FrameCodec;
use jain_smartcluster_core::{EventRecord, TenantId};
use jain_smartcluster_daemon::actor::{DaemonHandle, DaemonServices};
use jain_smartcluster_daemon::auth::Authorizer;
use jain_smartcluster_daemon::capacity::CapacityDiscovery;
use jain_smartcluster_daemon::cgroup::{CgroupManager, IsolationMode};
use jain_smartcluster_daemon::config::DaemonConfig;
use jain_smartcluster_daemon::event_log::{EventLog, EventLogOptions};
use jain_smartcluster_daemon::executor::{
    run_exec_helper, Executor, WorkloadResolver, EXEC_HELPER_MARKER,
};
use jain_smartcluster_daemon::gpu::{
    GpuEvent, GpuEventMonitor, GpuInventory, MpsCapability, MpsController,
};
use jain_smartcluster_daemon::health::RuntimeHealth;
use jain_smartcluster_daemon::psi::PsiMonitor;
use jain_smartcluster_daemon::server::UnixServer;
use jain_smartcluster_daemon::store::Store;
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(name = "scqd", version, about = "Jain single-node workload daemon")]
struct Arguments {
    #[arg(
        long,
        env = "SCQD_CONFIG",
        default_value = "/etc/jain-smartcluster/scqd.toml"
    )]
    config: PathBuf,
    /// Validate configuration and host capabilities, print health JSON, then exit.
    #[arg(long)]
    check: bool,
}

struct PreparedRuntime {
    config: DaemonConfig,
    services: DaemonServices,
    psi: Option<PsiMonitor>,
    gpu_monitor: Option<GpuEventMonitor>,
    gpu_receiver: Option<tokio::sync::mpsc::Receiver<GpuEvent>>,
}

#[tokio::main]
async fn main() -> Result<()> {
    maybe_run_exec_helper();
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .try_init()
        .ok();

    // Install both handlers before startup so TERM never defaults to killing
    // main while resident children still need their checkpoint allowance.
    let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
    let mut interrupt = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())?;
    let arguments = Arguments::parse();
    let prepared = prepare(&arguments.config).await?;
    let status = prepared.services.health.status();
    if arguments.check {
        println!("{}", serde_json::to_string_pretty(&status)?);
        return Ok(());
    }

    let socket_path = prepared.config.socket_path.clone();
    let socket_mode = prepared.config.socket_mode;
    let authorizer = prepared.services.authorizer.clone();
    let handle = DaemonHandle::start(prepared.services).await?;

    if let Some(psi) = prepared.psi {
        let (sender, mut receiver) = tokio::sync::mpsc::channel(64);
        psi.spawn(sender);
        let pressure_handle = handle.clone();
        tokio::spawn(async move {
            while let Some(event) = receiver.recv().await {
                pressure_handle.pressure(event).await;
            }
        });
    }

    let gpu_monitor = prepared.gpu_monitor;
    if let Some(mut gpu_receiver) = prepared.gpu_receiver {
        let gpu_handle = handle.clone();
        tokio::spawn(async move {
            while let Some(event) = gpu_receiver.recv().await {
                gpu_handle.gpu_event(event).await;
            }
        });
    }

    let server = UnixServer::bind(
        &socket_path,
        socket_mode,
        authorizer,
        Arc::new(handle.clone()),
        FrameCodec::default(),
    )?;
    tracing::info!(
        socket = %socket_path.display(),
        mode = ?status.mode,
        capabilities = ?status.capabilities,
        "scqd ready"
    );

    let result = {
        let serving = server.serve();
        tokio::pin!(serving);
        let mut server_finished = false;
        let result = tokio::select! {
            result = &mut serving => {
                server_finished = true;
                result.map_err(anyhow::Error::from)
            },
            _ = terminate.recv() => Ok(()),
            _ = interrupt.recv() => Ok(()),
        };
        // Keep RPC processing available to finishing workers (live JobRecord and
        // authority checks). The actor closes Submit admission and scheduling.
        let shutdown = handle.shutdown();
        tokio::pin!(shutdown);
        if server_finished {
            shutdown.await?;
        } else {
            tokio::select! {
                drained = &mut shutdown => drained?,
                serving_result = &mut serving => {
                    shutdown.await?;
                    serving_result?;
                }
            }
        }
        result
    };
    drop(gpu_monitor);
    server.remove_socket()?;
    result
}

async fn prepare(config_path: &Path) -> Result<PreparedRuntime> {
    let config = DaemonConfig::load(config_path)?;
    if nix::unistd::geteuid().is_root()
        && (config.execution.worker_uid.is_none() || config.execution.worker_gid.is_none())
    {
        bail!(
            "privileged scqd requires execution.worker_uid and execution.worker_gid for workload isolation"
        );
    }
    ensure_private_directory(&config.event_dir)?;
    if let Some(parent) = config.state_path.parent() {
        ensure_private_directory(parent)?;
    }

    let cgroups = CgroupManager::initialize(&config.cgroup_root, config.embedded)
        .context("initialize cgroup-v2 isolation")?;
    let capacity_discovery = CapacityDiscovery {
        cgroup_mount: Path::new("/sys/fs/cgroup"),
        proc_self_cgroup: Path::new("/proc/self/cgroup"),
        numa_root: Path::new("/sys/devices/system/node"),
        scratch_path: &config.event_dir,
    };
    let capacity = if cgroups.mode() == IsolationMode::Delegated {
        capacity_discovery.discover_for_allocation_root(&config.cgroup_root)
    } else {
        capacity_discovery.discover()
    }
    .context("discover effective daemon capacity")?;
    let mut gpu = GpuInventory::discover(std::env::var("CUDA_VISIBLE_DEVICES").ok().as_deref());
    let (gpu_sender, gpu_receiver) = tokio::sync::mpsc::channel(64);
    let (gpu_monitor, gpu_receiver, event_reason) = if gpu.nvml_available {
        match GpuEventMonitor::spawn(gpu_sender) {
            Ok(monitor) => (Some(monitor), Some(gpu_receiver), None),
            Err(error) => {
                let reason = format!("NVML fault event delivery is unavailable: {error}");
                tracing::warn!(%reason, "GPU sharing remains disabled");
                (None, None, Some(reason))
            }
        }
    } else {
        (
            None,
            None,
            Some(
                gpu.unavailable_reason
                    .clone()
                    .unwrap_or_else(|| "NVML device inventory is unavailable".to_owned()),
            ),
        )
    };
    let mps_controller = if cgroups.mode() != IsolationMode::Delegated {
        gpu.mps = MpsCapability::unavailable(
            &config.mps,
            "MPS sharing requires delegated cgroup-v2 isolation",
        );
        None
    } else if let Some(reason) = event_reason {
        gpu.mps = MpsCapability::unavailable(
            &config.mps,
            format!("MPS sharing requires NVML event safety: {reason}"),
        );
        None
    } else {
        match MpsController::probe(&config.mps).await {
            Ok(controller) => {
                let worker_uid = config
                    .execution
                    .worker_uid
                    .unwrap_or_else(|| nix::unistd::geteuid().as_raw());
                let worker_gid = config
                    .execution
                    .worker_gid
                    .unwrap_or_else(|| nix::unistd::getegid().as_raw());
                match controller.validate_worker_access(worker_uid, worker_gid) {
                    Ok(()) => {
                        // CUDA may silently fall back to non-MPS execution when client
                        // registration fails. Keep sharing disabled until a post-launch
                        // registration attestation gate is implemented; retain the probed
                        // controller only for fail-closed recovery/termination evidence.
                        gpu.mps = MpsCapability::unavailable(
                            &config.mps,
                            "MPS sharing requires post-launch client registration attestation",
                        );
                        Some(controller)
                    }
                    Err(error) => {
                        gpu.mps = MpsCapability::unavailable(
                            &config.mps,
                            format!("MPS worker access validation failed: {error}"),
                        );
                        None
                    }
                }
            }
            Err(error) => {
                gpu.mps = MpsCapability::unavailable(
                    &config.mps,
                    format!("MPS controller probe failed: {error}"),
                );
                None
            }
        }
    };
    let psi = match PsiMonitor::register(&capacity.cgroup_path, config.psi) {
        Ok(monitor) => Some(monitor),
        Err(error) if config.embedded => {
            tracing::warn!(%error, "PSI unavailable in embedded safe mode");
            None
        }
        Err(error) => return Err(error).context("register PSI readiness triggers"),
    };
    let health = RuntimeHealth::new(&capacity, &cgroups, &gpu, psi.is_some());
    let store = Store::open(&config.state_path).context("open durable state store")?;
    let events = EventLog::<EventRecord>::open(&config.event_dir, EventLogOptions::default())
        .context("open durable event log")?;
    let helper = std::env::current_exe().context("resolve scqd executor helper")?;
    let executor = Executor::new(cgroups.clone(), helper, config.execution.clone())?;
    let interactive_headroom_per_mille = capacity_headroom_per_mille(
        config.scheduler.interactive_cpu_headroom_millicores,
        capacity.cpu_millicores,
        config.scheduler.interactive_ram_headroom_bytes,
        capacity.memory_max_bytes,
    );
    let tenant_weights = config
        .scheduler
        .tenant_weights
        .iter()
        .map(|(tenant, weight)| {
            TenantId::new(tenant.clone())
                .map(|tenant| (tenant, *weight))
                .map_err(|error| anyhow::anyhow!(error.to_string()))
        })
        .collect::<Result<std::collections::BTreeMap<_, _>>>()?;

    Ok(PreparedRuntime {
        services: DaemonServices {
            store,
            events,
            authorizer: Authorizer::new(config.auth.clone()),
            executor,
            resolver: WorkloadResolver::new(config.workers.clone()),
            cgroups,
            capacity,
            gpu,
            mps_controller,
            health,
            execution: config.execution.clone(),
            recovery_quiet: std::time::Duration::from_millis(config.scheduler.recovery_quiet_ms),
            interactive_headroom_per_mille,
            interactive_gpu_headroom_per_mille: config.scheduler.interactive_gpu_headroom_per_mille,
            max_running_jobs: config.scheduler.max_running_jobs,
            max_running_jobs_per_tenant: config.scheduler.max_running_jobs_per_tenant,
            max_queued_jobs: config.scheduler.max_queued_jobs,
            max_queued_jobs_per_tenant: config.scheduler.max_queued_jobs_per_tenant,
            tenant_weights,
        },
        config,
        psi,
        gpu_monitor,
        gpu_receiver,
    })
}

fn capacity_headroom_per_mille(
    cpu_reserved: u64,
    cpu_capacity: u64,
    ram_reserved: u64,
    ram_capacity: u64,
) -> u16 {
    fn ratio(reserved: u64, capacity: u64) -> u64 {
        if capacity == 0 {
            return 1_000;
        }
        reserved.saturating_mul(1_000).div_ceil(capacity).min(1_000)
    }
    ratio(cpu_reserved, cpu_capacity)
        .max(ratio(ram_reserved, ram_capacity))
        .try_into()
        .unwrap_or(1_000)
}

fn ensure_private_directory(path: &Path) -> Result<()> {
    if !path.exists() {
        fs::create_dir_all(path).with_context(|| format!("create {}", path.display()))?;
        fs::set_permissions(path, fs::Permissions::from_mode(0o750))
            .with_context(|| format!("set permissions on {}", path.display()))?;
    }
    let metadata = fs::metadata(path).with_context(|| format!("inspect {}", path.display()))?;
    if !metadata.is_dir() || metadata.mode() & 0o002 != 0 {
        bail!(
            "state directory must not be world-writable: {}",
            path.display()
        );
    }
    Ok(())
}

fn maybe_run_exec_helper() {
    let mut arguments = std::env::args();
    let _program = arguments.next();
    if arguments.next().as_deref() != Some(EXEC_HELPER_MARKER) {
        return;
    }
    let arguments: Vec<_> = arguments.collect();
    if let Err(error) = run_exec_helper(&arguments) {
        eprintln!("scqd executor helper failed: {error}");
        std::process::exit(126);
    }
    unreachable!("successful direct exec does not return")
}
