use super::*;

/// Admission ownership only: native fixtures deliberately make no CUDA calls.
#[tokio::test]
#[ignore = "requires isolated delegated cgroup lane, one qualified idle GPU and baseline file"]
async fn exclusive_gpu_lease_blocks_competitor_until_recursive_drain() -> Result<()> {
    let Some(baseline_path) = std::env::var_os("SCQ_TEST_GPU_BASELINES") else {
        eprintln!("SKIP: SCQ_TEST_GPU_BASELINES not supplied");
        return Ok(());
    };
    let mut baseline_bytes = Vec::new();
    File::open(baseline_path)?
        .take(16385)
        .read_to_end(&mut baseline_bytes)?;
    ensure!(
        baseline_bytes.len() <= 16384,
        "oversized baseline configuration"
    );
    let baselines: BTreeMap<String, jain_smartcluster_daemon::gpu::QualifiedIdleBaseline> =
        serde_json::from_slice(&baseline_bytes)?;
    ensure!(
        baselines.len() == 1,
        "GPU test requires exactly one qualified device"
    );
    let uuid = baselines.keys().next().unwrap().clone();
    baselines[&uuid]
        .validate(&uuid)
        .map_err(anyhow::Error::msg)?;
    ensure!(
        std::env::var("CUDA_VISIBLE_DEVICES")? == uuid,
        "test daemon visibility must be exactly the qualified UUID"
    );
    let inventory = jain_smartcluster_daemon::gpu::GpuInventory::discover(Some(&uuid));
    if !inventory.nvml_available {
        eprintln!("SKIP: NVML unavailable: {:?}", inventory.unavailable_reason);
        return Ok(());
    }
    ensure!(
        inventory.devices.len() == 1
            && inventory.devices[0].uuid == uuid
            && inventory.devices[0].processes.is_empty(),
        "GPU unavailable or other compute/graphics process present"
    );
    let Some(h) = Harness::start_with_gpu_baselines(1000, baselines).await? else {
        return Ok(());
    };
    durable(
        &h.dir.path().join("gpu-inventory.json"),
        &serde_json::to_vec(&inventory)?,
    )?;
    let first_dir = h.dir.path().join("gpu-first");
    let second_dir = h.dir.path().join("gpu-second");
    fs::create_dir(&first_dir)?;
    fs::create_dir(&second_dir)?;
    let make_spec = |directory: &Path, key: &str| -> Result<JobSpec> {
        let mut spec = Harness::spec(directory, "gpu-lease", 1000)?;
        spec.idempotency_key = IdempotencyKey::new(key)?;
        spec.retry.max_failures = 0;
        spec.resources.cpu = CpuRequest::fixed(1000);
        spec.resources.gpu = Some(GpuRequest {
            count: 1,
            vram_bytes: 1024 * 1024,
            compute_share_per_mille: 1000,
            mode: GpuMode::Exclusive,
        });
        spec.validate()?;
        Ok(spec)
    };
    let first = h
        .client()?
        .submit(make_spec(&first_dir, "exclusive-first")?)
        .await?;
    h.running(first.id).await?;
    let first_receipt = wait_gpu_attempt(&first_dir).await?;
    // Prove the competing job's CPU/RAM fit while the first lease runs. Without
    // this control, a queued GPU competitor could merely indicate CPU pressure.
    let control_dir = h.dir.path().join("cpu-capacity-control");
    fs::create_dir(&control_dir)?;
    let mut control_spec = make_spec(&control_dir, "cpu-capacity-control")?;
    control_spec.resources.gpu = None;
    let control = h.client()?.submit(control_spec).await?;
    h.running(control.id).await?;
    h.client()?
        .cancel(CancelRequest {
            job_id: control.id,
            mode: CancelMode::Graceful { grace_ms: 200 },
        })
        .await?;
    let old = PathBuf::from(first_receipt["cgroup"].as_str().context("first cgroup")?);
    ensure!(old.starts_with(&h.cgroup), "foreign first cgroup");
    durable(
        &second_dir.join("gpu-predecessor.json"),
        &serde_json::to_vec(&first_receipt)?,
    )?;
    let second = h
        .client()?
        .submit(make_spec(&second_dir, "exclusive-second")?)
        .await?;
    for _ in 0..40 {
        ensure!(
            h.record(first.id).await?.state == JobState::Running && populated(&old)?,
            "first allocation unexpectedly ended"
        );
        let held = h.record(second.id).await?;
        ensure!(
            held.state == JobState::Queued
                && held.attempt_count == 0
                && !second_dir.join("attempt-1.json").exists(),
            "exclusive GPU competitor launched concurrently"
        );
        sleep(Duration::from_millis(25)).await;
    }
    let cancelled = h
        .client()?
        .cancel(CancelRequest {
            job_id: first.id,
            mode: CancelMode::Graceful { grace_ms: 200 },
        })
        .await?;
    ensure!(
        cancelled.state == JobState::Cancelled,
        "first job cancellation failed"
    );
    h.running(second.id).await?;
    let second_receipt = wait_gpu_attempt(&second_dir).await?;
    ensure!(
        !old.exists() || !populated(&old)?,
        "exclusive allocations overlapped at handoff"
    );
    ensure!(
        first_receipt["cuda_visible_devices"].as_str() == Some(uuid.as_str())
            && second_receipt["cuda_visible_devices"].as_str() == Some(uuid.as_str()),
        "allocations do not identify the exact qualified GPU"
    );
    durable(
        &h.dir.path().join("exclusive-gpu-proof.json"),
        &serde_json::to_vec(
            &serde_json::json!({"schema":"jope.exclusive_gpu_allocation_test.v1","uuid":uuid,"first":first_receipt,"second":second_receipt,"blocked_samples":40,"gpu_utilization_tested":false}),
        )?,
    )?;
    Ok(())
}

async fn wait_gpu_attempt(directory: &Path) -> Result<serde_json::Value> {
    let until = Instant::now() + WAIT;
    loop {
        if let Ok(file) = File::open(directory.join("attempt-1.json")) {
            let mut bytes = Vec::new();
            file.take(1024 * 1024 + 1).read_to_end(&mut bytes)?;
            ensure!(bytes.len() <= 1024 * 1024, "oversized GPU fixture receipt");
            if let Ok(value) = serde_json::from_slice(&bytes) {
                return Ok(value);
            }
        }
        ensure!(Instant::now() < until, "GPU fixture receipt timed out");
        sleep(Duration::from_millis(20)).await;
    }
}
pub(super) fn populated(path: &Path) -> Result<bool> {
    Ok(fs::read_to_string(path.join("cgroup.events"))?
        .lines()
        .any(|l| l == "populated 1"))
}
fn remove_empty_tree(path: &Path) {
    if let Ok(entries) = fs::read_dir(path) {
        for e in entries.flatten() {
            if e.file_type().is_ok_and(|t| t.is_dir()) {
                remove_empty_tree(&e.path());
            }
        }
    }
    let _ = fs::remove_dir(path);
}
impl Drop for Harness {
    fn drop(&mut self) {
        if let Some(mut c) = self.child.take() {
            let _ = c.kill();
            let _ = c.wait();
        } // Only this newly-created subtree, never process-name/marker scans.
        let _ = fs::write(self.cgroup.join("cgroup.kill"), "1");
        for _ in 0..100 {
            if !populated(&self.cgroup).unwrap_or(false) {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        remove_empty_tree(&self.cgroup);
    }
}
