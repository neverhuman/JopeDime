use super::*;

/// Actual peer-authenticated placement, child-start gate and cancellation propagation.
#[tokio::test]
#[ignore = "requires real delegated cgroup and SCQ_TEST_FIT_FIXTURE native example binary"]
async fn fit_child_real_peer_placement_duplicate_and_target_cancellation() -> Result<()> {
    use jain_smartcluster::jope::{MANIFEST_SCHEMA, TaskKind, TaskManifest};
    use sha2::{Digest, Sha256};
    let fixture = fs::canonicalize(
        std::env::var_os("SCQ_TEST_FIT_FIXTURE").context("build and set native fit fixture")?,
    )?;
    let Some(h) = Harness::start_inner(1000, BTreeMap::new(), Some(fixture)).await? else {
        return Ok(());
    };
    let now_ms: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_millis()
        .try_into()?;
    let now = Timestamp::from_unix_millis(now_ms);
    let input = serde_json::to_vec(
        &serde_json::json!({"schema":"jope.smartcluster_input.v1","operation":{"kind":"Search","evaluations":1,"seeds":null}}),
    )?;
    let input_sha = hex::encode(Sha256::digest(&input));
    durable(&h.dir.path().join("fit.input.json"), &input)?;
    let manifest = TaskManifest {
        schema: MANIFEST_SCHEMA.into(),
        run: "native-fit-kernel".into(),
        generation: 0,
        task_id: "target-0".into(),
        kind: TaskKind::GpSearch,
        input_manifest_sha256: input_sha.clone(),
        dependencies: BTreeMap::from([("input".into(), input_sha)]),
        parent_checkpoint_sha256: None,
        corpus_sha256: None,
        idempotency_key: IdempotencyKey::new("fit-kernel-target")?,
        deadline: Timestamp::from_unix_millis(now_ms + 120000),
        retry_allowance: 0,
        resources: ResourceRequest {
            cpu: CpuRequest::fixed(100),
            ram: MemoryRequest::fixed(128 * 1024 * 1024),
            scratch_bytes: 0,
            gpu: None,
        },
        expected_receipt_id: "fit-kernel-result".into(),
        checkpoint_allowance_ms: None,
    };
    let bytes = serde_json::to_vec(&manifest)?;
    let manifest_sha = hex::encode(Sha256::digest(&bytes));
    let path = h.dir.path().join("fit.json");
    durable(&path, &bytes)?;
    let target = h
        .client()?
        .submit(jain_smartcluster::jope::job_spec(
            &path,
            &manifest_sha,
            TenantId::new(format!("unix:{}", geteuid().as_raw()))?,
            ProjectId::new("resident-kernel")?,
            now,
        )?)
        .await?;
    h.running(target.id).await?;
    h.file("target-started.json").await?;
    let target_record = h.record(target.id).await?;
    durable(
        &h.dir.path().join("target-request.json"),
        &serde_json::to_vec(
            &serde_json::json!({"attempt":target_record.active_attempt.context("target fence")?,"manifest_sha256":manifest_sha}),
        )?,
    )?;
    let mut authority = Harness::spec(h.dir.path(), "fit-authority", 1000)?;
    authority.workload = Workload::TrustedService {
        service: ServiceName::new("jope-fit-authority")?,
        arguments: vec![
            "--authority".into(),
            h.dir.path().display().to_string(),
            h.socket.display().to_string(),
        ],
        environment: Environment::default(),
    };
    authority.preemption = PreemptionPolicy::non_preemptible();
    authority.deadline = Some(Timestamp::from_unix_millis(now_ms + 120000));
    authority.retry.max_failures = 0;
    let authority = h.client()?.submit(authority).await?;
    h.running(authority.id).await?;
    let placement: TrustedFitChildPlacement =
        serde_json::from_slice(&h.file("placed.json").await?)?;
    let request: AttachTrustedFitChildRequest =
        serde_json::from_slice(&h.file("request.json").await?)?;
    ensure!(
        placement.target_attempt == target_record.active_attempt.unwrap(),
        "wrong target fence"
    );
    ensure!(
        placement.cgroup_path.starts_with(&h.cgroup),
        "foreign target cgroup"
    );
    ensure!(
        h.client()?.attach_trusted_fit_child(request).await.is_err(),
        "same UID nonauthority peer accepted"
    );
    ensure!(
        !h.dir.path().join("child-started.json").exists(),
        "child passed gate before test permits start"
    );
    durable(
        &h.dir.path().join("allow-child-start"),
        b"start after receipt\n",
    )?;
    let started: serde_json::Value = serde_json::from_slice(&h.file("child-started.json").await?)?;
    ensure!(
        started["cgroup"] == placement.cgroup_path.display().to_string(),
        "child started outside target allocation"
    );
    h.client()?
        .cancel(CancelRequest {
            job_id: target.id,
            mode: CancelMode::Graceful { grace_ms: 200 },
        })
        .await?;
    let limit = Instant::now() + WAIT;
    loop {
        if h.record(target.id).await?.state.is_terminal() {
            // Observe the kernel AFTER the terminal receipt. A pre-RPC snapshot
            // may correctly be populated immediately before completion commits.
            ensure!(
                !placement.cgroup_path.exists() || !populated(&placement.cgroup_path)?,
                "terminal target released while child subtree populated"
            );
            break;
        }
        ensure!(
            Instant::now() < limit,
            "target subtree not drained before terminal release"
        );
        sleep(Duration::from_millis(20)).await;
    }
    h.file("child-reaped.json").await?;
    h.file("authority-survived.json").await?;
    ensure!(
        h.record(authority.id).await?.state == JobState::Running,
        "target cancellation killed authority"
    );
    durable(&h.dir.path().join("release-authority"), b"complete\n")?;
    Ok(())
}
