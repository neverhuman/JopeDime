use super::*;

pub(super) fn build_reports(
    nodes: &[NodeTelemetry],
    policy: &StaticPolicy,
    runtime: &BTreeMap<String, f64>,
    gpu_allocations: &BTreeMap<String, Vec<String>>,
) -> Result<Vec<NodeReport>> {
    nodes
        .iter()
        .map(|node| {
            let configured = policy.multipliers.get(&node.node).copied().unwrap_or(1.0);
            let runtime_multiplier = runtime.get(&node.node).copied().unwrap_or(1.0);
            if matches!(node.node.as_str(), "xbabe1" | "xbabe3") && runtime_multiplier != 1.0 {
                bail!("protected node {} multiplier changed", node.node);
            }
            Ok(NodeReport {
                node: node.node.clone(),
                physical_cores: node.physical_cores,
                effective_cores: node.effective_cores,
                configured_multiplier: configured,
                runtime_multiplier,
                slot_target: node_slot_target(node.effective_cores, runtime_multiplier)?,
                scheduler_multiplier: node.scheduler_multiplier,
                scheduler_slot_target: node.scheduler_slot_target,
                running_slots: node.running_slots,
                cpu_psi: node.pressure.cpu_psi,
                memory_psi: node.pressure.memory_psi,
                io_psi: node.pressure.io_psi,
                available_memory_fraction: node.pressure.available_memory_fraction.unwrap_or(0.0),
                protected_gpu_allocations: gpu_allocations
                    .get(&node.node)
                    .cloned()
                    .unwrap_or_default(),
            })
        })
        .collect()
}

pub(super) fn scheduler_matches_runtime(report: &NodeReport) -> bool {
    report.scheduler_multiplier == Some(report.runtime_multiplier)
        && report.scheduler_slot_target == Some(report.slot_target)
}

pub(super) fn compile_cache_status(root: &Path) -> Option<Value> {
    for path in [
        root.join("ops/status-v3.json"),
        root.join("ops/status.json"),
    ] {
        if let Ok(value) = read_json(&path) {
            if let Some(cache) = value.get("compile_cache") {
                return Some(cache.clone());
            }
        }
    }
    None
}

pub(super) fn receipt_dirs(root: &Path) -> (PathBuf, PathBuf) {
    let base = root.join("ops/capacity");
    (base.join("receipts"), base.join("idempotency"))
}

pub(super) fn replay_receipt(root: &Path, key: &str) -> Result<Option<Value>> {
    let (receipts, idempotency) = receipt_dirs(root);
    let key_sha256 = sha256_bytes(key.as_bytes());
    let index_path = idempotency.join(format!("{key_sha256}.json"));
    if !index_path.is_file() {
        return Ok(None);
    }
    let index = read_json(&index_path)?;
    let digest = index
        .get("receipt_sha256")
        .and_then(Value::as_str)
        .context("idempotency index lacks receipt SHA-256")?;
    let receipt_path = receipts.join(format!("{digest}.json"));
    if sha256_file(&receipt_path)? != digest {
        bail!("capacity receipt content drift");
    }
    Ok(Some(read_json(&receipt_path)?))
}
