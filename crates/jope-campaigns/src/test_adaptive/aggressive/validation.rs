use super::*;

pub(crate) fn validate_preflight_receipt(receipt: &PreflightReceipt) -> Result<()> {
    if receipt.schema != PREFLIGHT_SCHEMA
        || receipt.benchmark_valid
        || !receipt.test_adaptive
        || receipt.required_cores != scq_batch::AGGRESSIVE_CPUS
        || receipt.required_available_ram_bytes != MIN_AVAILABLE_RAM_BYTES
        || receipt.required_free_disk_bytes != MIN_FREE_DISK_BYTES
        || receipt.maximum_swap_used_bytes_exclusive != MAX_SWAP_USED_BYTES
        || !receipt.active_scq_jobs.is_empty()
        || receipt.nodes.len() != scq_batch::NODES.len()
    {
        bail!("invalid aggressive preflight receipt");
    }
    for (index, node) in receipt.nodes.iter().enumerate() {
        if node.node != scq_batch::NODES[index]
            || node.online_cores < scq_batch::AGGRESSIVE_CPUS
            || node.available_ram_bytes < MIN_AVAILABLE_RAM_BYTES
            || node.free_disk_bytes < MIN_FREE_DISK_BYTES
            || node.swap_used_bytes >= MAX_SWAP_USED_BYTES
            || !node.competing_processes.is_empty()
        {
            bail!("aggressive preflight resource binding is invalid");
        }
    }
    Ok(())
}

// Keep the expected manifest identity and scheduling contract inputs explicit.
#[allow(clippy::too_many_arguments)]
pub(crate) fn expected_manifest(
    registry: &Path,
    run_id: &str,
    release_binary_sha256: &str,
    scq_binary_sha256: &str,
    preflight_sha256: &str,
    staging_sha256: &str,
    backend_admission_sha256: &str,
    datasets: Vec<DatasetSpec>,
    golden46: Option<&Golden46Bindings>,
) -> AggressiveManifest {
    AggressiveManifest {
        schema: SCHEMA.to_string(),
        benchmark_valid: false,
        test_adaptive: true,
        run_id: run_id.to_string(),
        selected_backend: "rust".to_string(),
        cpp17_admitted: false,
        cpp17_rejection: "full live-registry fitted lowering/runtime parity is unavailable; the standalone P-1C Plan emitter is not BoundModel execution, so performance admission is fail-closed"
            .to_string(),
        backend_admission_sha256: backend_admission_sha256.to_string(),
        release_binary_sha256: release_binary_sha256.to_string(),
        scq_binary_sha256: scq_binary_sha256.to_string(),
        preflight_sha256: preflight_sha256.to_string(),
        staging_sha256: staging_sha256.to_string(),
        panel_manifest: golden46.map(|_| golden46_panel_manifest_str().to_string()),
        panel_manifest_sha256: golden46.map(|binding| binding.panel_sha256.clone()),
        leader_manifest: golden46.map(|_| golden46_leader_manifest_str().to_string()),
        leader_manifest_sha256: golden46.map(|binding| binding.leader_sha256.clone()),
        evolution_view_manifest_sha256: golden46
            .map(|binding| binding.evolution_view_manifest_sha256.clone()),
        arm: golden46.map(|binding| binding.arm.clone()),
        leaderboard_default: golden46.map(|binding| binding.leaderboard_default),
        registry_root: registry.to_string_lossy().into_owned(),
        population: TOTAL_POPULATION,
        islands: ISLANDS,
        population_per_island: ISLAND_POPULATION,
        workers_per_node: scq_batch::AGGRESSIVE_WORKERS,
        generation_limit: None,
        routing: if golden46.is_some() {
            "four islands of nine globally unique logical members; generation zero evaluates 36 fresh full-genome mutations of one best parent, later generations cache eight copied elites and route 28 history-unique fresh descendants by nodes[(slot + generation) % 3] with single-threaded workers"
                .to_string()
        } else {
            "up to 42 concurrent datasets; each 24-slot generation routes 8 candidates to each node by nodes[(slot + generation) % 3], while each node drains its over-decomposed queue through 240 single-threaded worker processes pinned two per each of 120 CPUs"
                .to_string()
        },
        retry: if golden46.is_some() {
            "one immutable missing-result retry on a different node".to_string()
        } else {
            "up to three immutable missing-result retries on rotating nodes".to_string()
        },
        stop_rule: "no generation cap; record generation-zero crossings but continue until a generation-1-or-later evolved descendant has finite test RMSE strictly below the best declared external baseline, then reproduce identical prediction and RMSE bits on all three nodes"
            .to_string(),
        datasets,
    }
}

pub(crate) fn manifest_matches_golden46_bindings(
    manifest: &AggressiveManifest,
    golden46: Option<&Golden46Bindings>,
) -> bool {
    match golden46 {
        Some(binding) => {
            manifest.panel_manifest.as_deref() == Some(golden46_panel_manifest_str())
                && manifest.panel_manifest_sha256.as_deref() == Some(binding.panel_sha256.as_str())
                && manifest.leader_manifest.as_deref() == Some(golden46_leader_manifest_str())
                && manifest.leader_manifest_sha256.as_deref()
                    == Some(binding.leader_sha256.as_str())
                && manifest.evolution_view_manifest_sha256.as_deref()
                    == Some(binding.evolution_view_manifest_sha256.as_str())
                && manifest.arm.as_deref() == Some(binding.arm.as_str())
                && manifest.leaderboard_default == Some(binding.leaderboard_default)
                && (binding.arm != "target-family-informed"
                    || (manifest.datasets.len() == 1 && manifest.datasets[0].dataset_hash == "ZTS"))
        }
        None => {
            manifest.panel_manifest.is_none()
                && manifest.panel_manifest_sha256.is_none()
                && manifest.leader_manifest.is_none()
                && manifest.leader_manifest_sha256.is_none()
                && manifest.evolution_view_manifest_sha256.is_none()
                && manifest.arm.is_none()
                && manifest.leaderboard_default.is_none()
        }
    }
}
