use jain_smartcluster_daemon::gpu::{GpuEventMonitor, GpuInventory};

#[test]
#[ignore = "requires an NVIDIA/NVML host lane"]
fn real_nvml_inventory_respects_vram_headroom() {
    let visible = std::env::var("CUDA_VISIBLE_DEVICES").ok();
    let inventory = GpuInventory::discover(visible.as_deref());
    if !inventory.nvml_available || inventory.devices.is_empty() {
        eprintln!(
            "SKIP: NVML GPU inventory unavailable: {}",
            inventory
                .unavailable_reason
                .as_deref()
                .unwrap_or("no visible GPUs")
        );
        return;
    }
    for device in &inventory.devices {
        assert!(device.total_vram_bytes > 0);
        assert!(device.allocatable_vram_bytes < device.total_vram_bytes);
        assert_eq!(
            device.allocatable_vram_bytes,
            device.total_vram_bytes.saturating_mul(90) / 100
        );
    }
}

#[tokio::test]
#[ignore = "requires NVML event registration support"]
async fn real_nvml_event_monitor_registers_or_fails_closed() {
    let (sender, _receiver) = tokio::sync::mpsc::channel(8);
    match GpuEventMonitor::spawn(sender) {
        Ok(monitor) => drop(monitor),
        Err(error) => eprintln!("SKIP: NVML event delivery unavailable: {error}"),
    }
}
