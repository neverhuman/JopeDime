use super::*;

#[tokio::test]
async fn real_daemon_runs_idempotent_work_and_releases_a_held_queue() -> Result<()> {
    let daemon = DaemonHarness::spawn(true).await?;
    let client = daemon.client()?;

    let status = client.status().await.context("read daemon status")?;
    assert_eq!(status.mode, DaemonMode::SafeSingleAllocation);
    assert!(!status.capabilities.cgroup_delegated);

    let blocker = client
        .submit(admin_spec(
            "blocker",
            sleep_executable()?,
            vec!["30".to_owned()],
            Environment(BTreeMap::from([(
                "JAIN_SMARTCLUSTER_E2E".to_owned(),
                daemon.cleanup_marker.clone(),
            )])),
        ))
        .await
        .context("submit blocking allocation")?;
    wait_for_state(&client, blocker.id, JobState::Running).await?;

    let held_spec = admin_spec(
        "held-output",
        daemon.daemon_executable.clone(),
        vec!["--version".to_owned()],
        Environment::default(),
    );
    let held = client
        .submit(held_spec.clone())
        .await
        .context("submit held job")?;
    let duplicate = client
        .submit(held_spec)
        .await
        .context("repeat idempotent submission")?;
    assert_eq!(duplicate.id, held.id);
    assert_eq!(
        client
            .get(blocker.id)
            .await
            .context("confirm blocking allocation")?
            .context("blocking allocation must exist")?
            .state,
        JobState::Running
    );
    assert_eq!(
        client
            .get(held.id)
            .await
            .context("confirm held job is queued")?
            .context("held job must exist")?
            .state,
        JobState::Queued
    );
    assert_eq!(client.status().await?.queued_jobs, 1);

    let paused = client.pause(held.id).await.context("hold queued job")?;
    assert_eq!(paused.state, JobState::Paused);
    assert!(paused.active_attempt.is_none());
    assert_eq!(client.status().await?.queued_jobs, 0);
    let resumed = client.resume(held.id).await.context("release held job")?;
    assert_eq!(resumed.state, JobState::Queued);
    assert!(resumed.active_attempt.is_none());
    assert_eq!(client.status().await?.queued_jobs, 1);

    let fetched = client
        .get(held.id)
        .await
        .context("get held job")?
        .context("held job must exist")?;
    assert_eq!(fetched.id, held.id);
    let listed = client
        .list(ListRequest::default())
        .await
        .context("list caller jobs")?;
    assert!(listed.jobs.iter().any(|job| job.id == blocker.id));
    assert!(listed.jobs.iter().any(|job| job.id == held.id));

    let cancelled = client
        .cancel(CancelRequest {
            job_id: blocker.id,
            mode: CancelMode::Graceful { grace_ms: 500 },
        })
        .await
        .context("cancel blocking allocation")?;
    assert_eq!(cancelled.state, JobState::Cancelled);

    let completed = client
        .wait(held.id)
        .await
        .context("wait for released job")?;
    assert_eq!(completed.state, JobState::Completed);
    assert!(matches!(
        completed.terminal,
        Some(TerminalStatus::Completed { exit_code: 0, .. })
    ));
    assert_eq!(
        client
            .get(held.id)
            .await
            .context("get completed job")?
            .context("completed job must exist")?
            .state,
        JobState::Completed
    );

    assert_durable_output(&client, held.id, b"scqd").await?;
    drop(daemon);
    Ok(())
}

#[tokio::test]
async fn real_daemon_fails_closed_for_delegation_tenant_and_environment() -> Result<()> {
    let daemon = DaemonHarness::spawn(true).await?;
    let client = daemon.client()?;

    let delegated = jain_spec(
        "delegation-denied",
        TenantId::new("web:owner-1")?,
        Environment::default(),
    );
    assert_server_code(
        client
            .submit_as(delegated, Principal::web("owner-1")?)
            .await,
        ErrorCode::Forbidden,
    );

    let mut wrong_tenant = admin_spec(
        "tenant-denied",
        daemon.daemon_executable.clone(),
        vec!["--version".to_owned()],
        Environment::default(),
    );
    wrong_tenant.tenant = TenantId::new("unix:424242")?;
    assert_server_code(client.submit(wrong_tenant).await, ErrorCode::Forbidden);

    drop(daemon);
    let daemon = DaemonHarness::spawn(false).await?;
    let client = daemon.client()?;

    let mut environment = BTreeMap::new();
    environment.insert("MODEL_PROFILE".to_owned(), "untrusted".to_owned());
    assert_server_code(
        client
            .submit(jain_spec(
                "environment-denied",
                unix_tenant(),
                Environment(environment),
            ))
            .await,
        ErrorCode::Forbidden,
    );

    drop(daemon);
    Ok(())
}
