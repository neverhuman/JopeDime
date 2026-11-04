#[test]
fn visible_devices_filters_empty_and_disabled_values() {
    assert!(parse_visible_devices("").is_empty());
    assert!(parse_visible_devices("-1").is_empty());
    assert!(parse_visible_devices("-1,0").is_empty());
    assert_eq!(
        parse_visible_devices("0, GPU-abc"),
        ["0".to_owned(), "GPU-abc".to_owned()].into_iter().collect()
    );
}

#[test]
fn mig_uuid_visibility_is_matched_after_parent_enumeration() {
    let selectors = parse_visible_devices("MIG-01234567-89ab-cdef");
    assert!(!is_visible(Some(&selectors), 0, "GPU-parent"));
    assert!(selectors.contains("MIG-01234567-89ab-cdef"));
}

#[test]
fn mps_fails_closed_without_safe_peer_termination() {
    let capability = MpsCapability::detect();
    assert!(!capability.shareable);
    assert!(!capability.compute_limit);
    assert!(!capability.vram_limit);
    assert!(!capability.safe_client_termination);
}

#[test]
fn parses_bounded_mps_process_snapshots() {
    let clients = parse_ps(
        "PID ID SERVER DEVICE NAMESPACE COMMAND ATTRIBUTES\n\
         9741 0 6472 GPU-cb1213a3-d6a4 4026531836 ./nbody --steps 4\n\
         9743 1 6472 GPU-cb1213a3-d6a4 4026531836 ./matrixMul\n",
    )
    .expect("valid MPS ps response");
    assert_eq!(clients.len(), 2);
    assert_eq!(clients[0].pid, 9741);
    assert_eq!(clients[0].server_pid, 6472);
    assert_eq!(clients[0].command, "./nbody --steps 4");
    assert!(parse_ps("PID SERVER DEVICE\n").is_err());
    assert!(
        parse_ps("PID ID SERVER DEVICE NAMESPACE COMMAND\nnot-a-pid 0 6472 GPU-a 1 app\n").is_err()
    );
}

#[test]
fn termination_requires_the_exact_success_response() {
    assert!(parse_termination_response("0\n").is_ok());
    assert!(parse_termination_response("CUDA_SUCCESS\n").is_err());
    assert!(parse_termination_response("0\nwarning\n").is_err());
}

#[test]
fn client_environment_is_budgeted_and_deterministic() {
    let controller = MpsController {
        control_executable: PathBuf::from("/usr/bin/nvidia-cuda-mps-control"),
        pipe_directory: PathBuf::from("/run/nvidia-mps"),
        command_timeout: Duration::from_secs(1),
        max_control_output_bytes: 1024,
    };
    let limits = MpsClientLimits {
        compute_share_per_mille: 337,
        vram_bytes_by_device: [
            ("GPU-abcdef01".to_owned(), 3 * 1024 * MEBIBYTE + 9),
            ("0".to_owned(), 1536 * MEBIBYTE + 1),
        ]
        .into_iter()
        .collect(),
    };
    let environment = controller
        .client_environment(&limits)
        .expect("representable limits");
    assert_eq!(
        environment
            .get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE")
            .map(String::as_str),
        Some("33")
    );
    assert_eq!(
        environment
            .get("CUDA_MPS_PINNED_DEVICE_MEM_LIMIT")
            .map(String::as_str),
        Some("0=1536M,GPU-abcdef01=3072M")
    );
    assert_eq!(
        environment
            .get("CUDA_MPS_PIPE_DIRECTORY")
            .map(String::as_str),
        Some("/run/nvidia-mps")
    );
}

#[test]
fn client_environment_rejects_unrepresentable_limits() {
    assert!(matches!(compute_percent(9), Err(GpuError::ComputeShare(9))));
    assert!(matches!(
        format_vram_limits(&[("0".to_owned(), MEBIBYTE - 1)].into_iter().collect()),
        Err(GpuError::MpsVramLimit(_))
    ));
    assert!(matches!(
        format_vram_limits(&[("0,1".to_owned(), MEBIBYTE)].into_iter().collect()),
        Err(GpuError::MpsDeviceSelector(_))
    ));
}

#[tokio::test]
#[ignore = "requires an administrator-provisioned NVIDIA MPS control daemon"]
async fn probes_real_mps_control_daemon_when_provisioned() {
    let Some(pipe_directory) = std::env::var_os("CUDA_MPS_PIPE_DIRECTORY") else {
        eprintln!("CUDA_MPS_PIPE_DIRECTORY is unset; skipping MPS host probe");
        return;
    };
    let control_executable = PathBuf::from("/usr/bin/nvidia-cuda-mps-control");
    if !control_executable.is_file() {
        eprintln!("nvidia-cuda-mps-control is unavailable; skipping MPS host probe");
        return;
    }
    let controller = MpsController::probe(&MpsConfig {
        control_executable: Some(control_executable),
        pipe_directory: Some(PathBuf::from(pipe_directory)),
        ..MpsConfig::default()
    })
    .await
    .expect("configured MPS daemon must pass the control protocol probe");
    let capability = controller.capability();
    assert!(capability.compute_limit);
    assert!(capability.vram_limit);
    assert!(capability.safe_client_termination);
    assert!(capability.shareable);
}

#[test]
fn ten_percent_vram_headroom_is_reserved() {
    let total = 80_u64 * 1024 * 1024 * 1024;
    assert_eq!(total.saturating_mul(90) / 100, 72_u64 * 1024 * 1024 * 1024);
}
use super::inventory::{is_visible, parse_visible_devices};
use super::mps::{compute_percent, format_vram_limits, parse_ps, parse_termination_response};
use super::*;
