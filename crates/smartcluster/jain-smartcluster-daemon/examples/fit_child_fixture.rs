//! CPU-only native kernel fixture. Never grants numerical execution.
use anyhow::{ensure, Context, Result};
use jain_smartcluster::{Client, ClientConfig, DispatchError};
use jain_smartcluster_core::*;
use nix::sys::{
    signal::{kill, Signal},
    wait::{waitpid, WaitPidFlag, WaitStatus},
};
use nix::unistd::{getpid, Pid};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    str::FromStr,
    time::{Duration, Instant},
};
fn sha(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}
fn save(path: &Path, value: &impl serde::Serialize) -> Result<()> {
    let parent = path.parent().context("parent")?;
    let mut f = tempfile::NamedTempFile::new_in(parent)?;
    f.write_all(&serde_json::to_vec(value)?)?;
    f.as_file().sync_all()?;
    f.persist_noclobber(path)?;
    fs::File::open(parent)?.sync_all()?;
    Ok(())
}
fn group(pid: u32) -> Result<PathBuf> {
    let text = fs::read_to_string(format!("/proc/{pid}/cgroup"))?;
    let path = text
        .lines()
        .find_map(|l| l.strip_prefix("0::"))
        .context("unified group")?;
    Ok(Path::new("/sys/fs/cgroup").join(path.trim_start_matches('/')))
}
fn birth(pid: u32) -> Result<u64> {
    let text = fs::read_to_string(format!("/proc/{pid}/stat"))?;
    Ok(text
        .rsplit_once(") ")
        .context("stat")?
        .1
        .split_whitespace()
        .nth(19)
        .context("birth")?
        .parse()?)
}
async fn until(path: &Path) -> Result<()> {
    let limit = Instant::now() + Duration::from_secs(60);
    while !path.exists() {
        ensure!(
            Instant::now() < limit,
            "fixture gate timeout {}",
            path.display()
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    Ok(())
}
// Native pressure may temporarily pause either allocation without changing its fence.
// Wait for the original allocations, never manufacture a replacement child/attempt.
async fn attach_ready(
    client: &Client,
    request: AttachTrustedFitChildRequest,
) -> Result<std::result::Result<TrustedFitChildPlacement, DispatchError>> {
    let until = Instant::now() + Duration::from_secs(20);
    loop {
        ensure!(
            Instant::now() < until,
            "original placement pressure wait expired"
        );
        let mut paused = false;
        for attempt in [request.authority_attempt, request.target_attempt] {
            let job = client
                .get(attempt.job_id)
                .await?
                .context("original fit allocation absent")?;
            ensure!(
                job.active_attempt == Some(attempt)
                    && matches!(
                        job.state,
                        JobState::Starting | JobState::Running | JobState::Paused
                    ),
                "original fit attempt superseded"
            );
            paused |= job.state != JobState::Running;
        }
        if paused {
            tokio::time::sleep(Duration::from_millis(20)).await;
            continue;
        }
        let result = client.attach_trusted_fit_child(request.clone()).await;
        if matches!(&result, Err(DispatchError::Server(e)) if e.code == ErrorCode::SupersededAttempt)
        {
            // Re-read exact original fences/state on the next iteration; an expired or
            // replaced allocation cannot become an accepted retry.
            tokio::time::sleep(Duration::from_millis(20)).await;
            continue;
        }
        return Ok(result);
    }
}
fn forbidden(result: &std::result::Result<TrustedFitChildPlacement, DispatchError>) -> bool {
    matches!(result, Err(DispatchError::Server(e)) if e.code == ErrorCode::Forbidden)
}
#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.first().is_some_and(|x| x == "--child") {
        // This branch uses the current-thread runtime without spawning any threads.
        let dir = PathBuf::from(&args[1]);
        kill(getpid(), Signal::SIGSTOP)?;
        save(
            &dir.join("child-started.json"),
            &serde_json::json!({"pid":std::process::id(),"cgroup":group(std::process::id())?}),
        )?;
        until(&dir.join("release-child")).await?;
        return Ok(());
    }
    if args.first().is_some_and(|x| x == "--task-manifest") {
        let dir = Path::new(&args[1]).parent().context("task parent")?;
        save(
            &dir.join("target-started.json"),
            &serde_json::json!({"pid":std::process::id(),"cgroup":group(std::process::id())?}),
        )?;
        until(&dir.join("release-target")).await?;
        return Ok(());
    }
    ensure!(
        args.len() == 3 && args[0] == "--authority",
        "fixture authority arguments"
    );
    let dir = PathBuf::from(&args[1]);
    let client = Client::with_config(ClientConfig {
        socket_path: PathBuf::from(&args[2]),
        ..Default::default()
    })?;
    let authority_attempt = FencedAttempt::new(
        JobId::from_str(&std::env::var("SCQ_JOB_ID")?)?,
        AttemptNumber::new(std::env::var("SCQ_ATTEMPT_NUMBER")?.parse()?)?,
        FenceToken::new(std::env::var("SCQ_ATTEMPT_FENCE")?.parse()?)?,
    );
    until(&dir.join("target-request.json")).await?;
    let instruction: serde_json::Value =
        serde_json::from_slice(&fs::read(dir.join("target-request.json"))?)?;
    let target_attempt = serde_json::from_value(instruction["attempt"].clone())?;
    let own = client
        .inspect_attempt(authority_attempt)
        .await?
        .context("authority inspection absent")?;
    let original = own
        .active_process
        .context("original authority process absent")?;
    ensure!(
        own.record.attempt == authority_attempt
            && original.pid == std::process::id()
            && original.birth_ticks == birth(std::process::id())?
            && original.boot_id == fs::read_to_string("/proc/sys/kernel/random/boot_id")?.trim(),
        "authority inspection identity mismatch"
    );
    let target_inspection = client
        .inspect_attempt(target_attempt)
        .await?
        .context("target inspection absent")?;
    ensure!(
        target_inspection.record.attempt == target_attempt
            && target_inspection.active_process.is_some(),
        "live target inspection mismatch"
    );
    save(
        &dir.join("initial-attempt-inspection.json"),
        &serde_json::json!({"authority":original,"target":target_inspection}),
    )?;

    let mut child = Command::new(std::env::current_exe()?)
        .args(["--child", dir.to_str().context("path")?])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()?;
    let pid = Pid::from_raw(i32::try_from(child.id())?);
    let limit = Instant::now() + Duration::from_secs(10);
    loop {
        match waitpid(pid, Some(WaitPidFlag::WUNTRACED | WaitPidFlag::WNOHANG))? {
            WaitStatus::Stopped(_, Signal::SIGSTOP) => break,
            WaitStatus::StillAlive => {
                ensure!(Instant::now() < limit, "child stop timeout");
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            other => anyhow::bail!("unexpected child status {other:?}"),
        }
    }
    // Child remains owned and unreaped for the complete RPC/retry window.
    let request = AttachTrustedFitChildRequest {
        authority_attempt,
        target_attempt,
        child_pid: child.id(),
        child_birth_ticks: birth(child.id())?,
        child_boot_id: fs::read_to_string("/proc/sys/kernel/random/boot_id")?
            .trim()
            .into(),
        task_manifest_sha256: instruction["manifest_sha256"]
            .as_str()
            .context("manifest hash")?
            .into(),
        descriptor_sha256: sha(b"native-fit-kernel-descriptor"),
        candidate_index: 0,
        idempotency_key: IdempotencyKey::new("fit-kernel-candidate-0")?,
        child_executable_sha256: sha(&fs::read(std::env::current_exe()?)?),
    };
    save(&dir.join("request.json"), &request)?;
    let mut invalid = request.clone();
    invalid.child_birth_ticks += 1;
    ensure!(
        forbidden(&attach_ready(&client, invalid).await?),
        "changed birth accepted"
    );
    let mut invalid = request.clone();
    invalid.candidate_index = 1;
    ensure!(
        forbidden(&attach_ready(&client, invalid).await?),
        "out-of-budget candidate accepted"
    );
    let mut invalid = request.clone();
    invalid.task_manifest_sha256 = sha(b"wrong-manifest");
    ensure!(
        forbidden(&attach_ready(&client, invalid).await?),
        "changed manifest accepted"
    );
    ensure!(
        group(child.id())? == group(std::process::id())?,
        "negative RPC moved child"
    );
    let placed = attach_ready(&client, request.clone()).await??;
    let mut value = serde_json::to_value(&placed)?;
    value
        .as_object_mut()
        .context("receipt object")?
        .remove("receipt_sha256");
    ensure!(
        placed.receipt_sha256 == sha(&serde_json::to_vec(&value)?),
        "placement receipt hash"
    );
    ensure!(
        group(child.id())? == placed.cgroup_path
            && group(std::process::id())? != placed.cgroup_path,
        "kernel placement mismatch"
    );
    ensure!(
        !dir.join("child-started.json").exists(),
        "child ran before verified placement"
    );
    let duplicate = attach_ready(&client, request.clone()).await??;
    ensure!(
        serde_json::to_vec(&duplicate)? == serde_json::to_vec(&placed)?,
        "duplicate receipt changed"
    );
    let mut conflict = request.clone();
    conflict.descriptor_sha256 = sha(b"conflicting-descriptor");
    ensure!(
        matches!(attach_ready(&client, conflict).await?, Err(DispatchError::Server(e)) if e.code == ErrorCode::Conflict),
        "conflicting candidate accepted"
    );
    // A cached receipt cannot silently move a child that somebody relocated.
    let authority_group = group(std::process::id())?;
    fs::write(authority_group.join("cgroup.procs"), child.id().to_string())?;
    ensure!(
        forbidden(&attach_ready(&client, request.clone()).await?),
        "cached receipt accepted externally relocated child"
    );
    ensure!(
        group(child.id())? == authority_group,
        "cached retry moved child again"
    );
    save(
        &dir.join("cached-relocation-refused.json"),
        &serde_json::json!({"child_pid":child.id(),"still_stopped":true,"authority_cgroup":authority_group,"original_receipt_sha256":placed.receipt_sha256}),
    )?;
    // Restore only this fixture's original stopped child for the cancellation lane.
    fs::write(
        placed.cgroup_path.join("cgroup.procs"),
        child.id().to_string(),
    )?;
    ensure!(
        group(child.id())? == placed.cgroup_path,
        "fixture restoration failed"
    );
    save(&dir.join("placed.json"), &placed)?;
    // Test driver performs a same-UID nonauthority request while child is stopped.
    until(&dir.join("allow-child-start")).await?;
    kill(pid, Signal::SIGCONT)?;
    let limit = Instant::now() + Duration::from_secs(30);
    loop {
        if let Some(status) = child.try_wait()? {
            save(
                &dir.join("child-reaped.json"),
                &serde_json::json!({"status":status.to_string(),"authority_pid":std::process::id(),"authority_cgroup":group(std::process::id())?}),
            )?;
            break;
        }
        ensure!(
            Instant::now() < limit,
            "target cancellation did not reap child"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    ensure!(
        client.attach_trusted_fit_child(request).await.is_err(),
        "stale target/child request accepted"
    );
    let retained = client
        .inspect_attempt(target_attempt)
        .await?
        .context("retained target attempt disappeared")?;
    ensure!(
        retained.record.attempt == target_attempt
            && retained.active_process.is_none()
            && retained.record.pid.is_some(),
        "terminal target inspection lost original history or reports live process"
    );
    save(&dir.join("terminal-attempt-inspection.json"), &retained)?;
    save(
        &dir.join("authority-survived.json"),
        &serde_json::json!({"pid":std::process::id(),"stale_rejected":true}),
    )?;
    until(&dir.join("release-authority")).await?;
    Ok(())
}
