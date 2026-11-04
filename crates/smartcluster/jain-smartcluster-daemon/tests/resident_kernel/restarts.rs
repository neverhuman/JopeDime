use super::*;

#[tokio::test]
#[ignore = "requires real delegated cgroup-v2 kernel lane"]
async fn resident_two_restarts_hold_nested_descendant_before_new_fence() -> Result<()> {
    let Some(mut h) = Harness::start(15000).await? else {
        return Ok(());
    };
    let job = h.submit("nested", 15000).await?;
    h.file("descendant-ready").await?;
    let first: serde_json::Value = serde_json::from_slice(&h.file("attempt-1.json").await?)?;
    let old = PathBuf::from(first["cgroup"].as_str().context("cgroup path")?);
    ensure!(old.starts_with(&h.cgroup), "foreign cgroup");
    h.crash()?;
    h.file("eof-receipt").await?;
    // Main leader may take a scheduling tick to exit. Observe the actual kernel procs file.
    let until = Instant::now() + WAIT;
    while !fs::read_to_string(old.join("cgroup.procs"))?
        .trim()
        .is_empty()
    {
        ensure!(Instant::now() < until, "leader did not exit");
        sleep(Duration::from_millis(20)).await;
    }
    ensure!(
        populated(&old)?,
        "nested descendant must keep old cgroup populated"
    );
    h.restart().await?;
    h.no_retry(job.id, &old).await?;
    h.crash()?;
    h.restart().await?;
    h.no_retry(job.id, &old).await?;
    durable(&h.dir.path().join("release-descendant"), b"release\n")?;
    let second: serde_json::Value = serde_json::from_slice(&h.file("attempt-2.json").await?)?;
    ensure!(
        !old.exists() || !populated(&old)?,
        "retry launched before recursive drain"
    );
    ensure!(
        first["attempt"]["fence"] != second["attempt"]["fence"],
        "attempt fence reused"
    );
    ensure!(
        h.record(job.id).await?.attempt_count == 2,
        "unexpected attempt count"
    );
    Ok(())
}

#[tokio::test]
#[ignore = "requires real delegated cgroup-v2 kernel lane"]
async fn resident_ignored_eof_is_bounded_by_checkpoint_allowance() -> Result<()> {
    let Some(mut h) = Harness::start(1000).await? else {
        return Ok(());
    };
    let job = h.submit("ignore-eof", 1000).await?;
    let first: serde_json::Value = serde_json::from_slice(&h.file("attempt-1.json").await?)?;
    let old = PathBuf::from(first["cgroup"].as_str().unwrap());
    h.crash()?;
    let started = Instant::now();
    h.restart().await?;
    h.file("attempt-2.json").await?;
    ensure!(
        started.elapsed() >= Duration::from_millis(800)
            && started.elapsed() < Duration::from_millis(3500),
        "checkpoint recovery {:?} violates 1000ms allowance (200ms tolerance) plus bounded escalation/readiness overhead",
        started.elapsed()
    );
    ensure!(
        !old.exists() || !populated(&old)?,
        "old worker overlaps replacement"
    );
    ensure!(
        h.record(job.id).await?.attempt_count == 2,
        "wrong retry count"
    );
    Ok(())
}

#[tokio::test]
#[ignore = "requires real delegated cgroup-v2 kernel lane"]
async fn resident_immediate_dependency_failure_report_is_not_lost() -> Result<()> {
    let Some(h) = Harness::start(1000).await? else {
        return Ok(());
    };
    let job = h.submit("dependency", 1000).await?;
    let until = Instant::now() + WAIT;
    loop {
        let record = h.record(job.id).await?;
        if record.state.is_terminal() {
            ensure!(
                matches!(
                    record.terminal,
                    Some(TerminalStatus::Failed {
                        class: FailureClass::Dependency,
                        ..
                    })
                ),
                "dependency report lost: {:?}",
                record.terminal
            );
            ensure!(record.attempt_count == 1, "dependency failure retried");
            break;
        }
        ensure!(Instant::now() < until, "failure reporting timeout");
        sleep(Duration::from_millis(20)).await;
    }
    Ok(())
}

#[tokio::test]
#[ignore = "requires real delegated cgroup-v2 kernel lane"]
async fn resident_fsynced_result_before_crash_is_reused_exactly() -> Result<()> {
    use std::os::unix::fs::MetadataExt;
    let Some(h) = Harness::start(1000).await? else {
        return Ok(());
    };
    let job = h.submit("result-crash", 1000).await?;
    h.file("attempt-1.json").await?;
    let request = SendControlRequest {
        job_id: job.id,
        control: ControlMessage::Application {
            name: "fixture.compute".into(),
            payload: b"task-1".to_vec(),
        },
    };
    h.running(job.id).await?;
    h.client()?.send_control(request.clone()).await?;
    let bytes = h.file("task-1.result").await?;
    let before = fs::metadata(h.dir.path().join("task-1.result"))?;
    h.file("attempt-2.json").await?;
    h.running(job.id).await?;
    h.client()?.send_control(request).await?;
    h.file("result-reused").await?;
    let after = fs::metadata(h.dir.path().join("task-1.result"))?;
    ensure!(
        bytes == b"task-1:deterministic-result\n"
            && fs::read(h.dir.path().join("task-1.result"))? == bytes,
        "result bytes changed"
    );
    ensure!(
        (before.dev(), before.ino(), before.mtime_nsec())
            == (after.dev(), after.ino(), after.mtime_nsec()),
        "durable result was rewritten"
    );
    Ok(())
}

#[tokio::test]
#[ignore = "requires real delegated cgroup-v2 kernel lane"]
async fn graceful_sigterm_waits_for_checkpoint_and_recursive_drain() -> Result<()> {
    let Some(mut h) = Harness::start(15000).await? else {
        return Ok(());
    };
    let job = h.submit("nested", 15000).await?;
    h.file("descendant-ready").await?;
    let first: serde_json::Value = serde_json::from_slice(&h.file("attempt-1.json").await?)?;
    let old = PathBuf::from(first["cgroup"].as_str().context("old cgroup")?);
    ensure!(old.starts_with(&h.cgroup), "foreign worker cgroup");
    let daemon = h.child.as_ref().context("daemon missing")?.id();
    nix::sys::signal::kill(
        nix::unistd::Pid::from_raw(daemon as i32),
        nix::sys::signal::Signal::SIGTERM,
    )?;
    h.file("eof-receipt").await?;
    // Main remains alive after checkpoint fsync while a nested child remains.
    for _ in 0..20 {
        ensure!(populated(&old)?, "nested child ended too early");
        ensure!(
            h.child.as_mut().unwrap().try_wait()?.is_none(),
            "main exited before recursive drain"
        );
        ensure!(
            !h.dir.path().join("attempt-2.json").exists(),
            "replacement launched during shutdown"
        );
        sleep(Duration::from_millis(25)).await;
    }
    // Repeat signal must not terminate main or extend the persisted recovery deadline.
    nix::sys::signal::kill(
        nix::unistd::Pid::from_raw(daemon as i32),
        nix::sys::signal::Signal::SIGTERM,
    )?;
    durable(&h.dir.path().join("release-descendant"), b"release\n")?;
    let until = Instant::now() + WAIT;
    loop {
        if let Some(status) = h.child.as_mut().unwrap().try_wait()? {
            ensure!(status.success(), "graceful daemon failed: {status}");
            break;
        }
        ensure!(
            Instant::now() < until,
            "shutdown did not finish after drain"
        );
        sleep(Duration::from_millis(20)).await;
    }
    h.child.take();
    ensure!(
        !old.exists() || !populated(&old)?,
        "main exited with old subtree populated"
    );
    ensure!(
        !h.dir.path().join("attempt-2.json").exists(),
        "shutdown launched retry"
    );
    h.restart().await?;
    h.file("attempt-2.json").await?;
    ensure!(
        h.record(job.id).await?.attempt_count == 2,
        "durable retry not resumed exactly once"
    );
    Ok(())
}
