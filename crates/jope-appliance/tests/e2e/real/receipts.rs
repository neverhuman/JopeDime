use super::*;
use super::{setup::*, transport::*};

pub(crate) fn terminal(
    home: &Path,
    csv: &Path,
    id: &jope_api::ids::RunId,
    projection: &Projection,
    key: &str,
) -> Result<String> {
    // Read expectations from the committed journal, independently of the renderer.
    // Public transcript scores use three decimal places (§6.2).
    let champion = projection
        .champion
        .as_ref()
        .context("run has no champion")?;
    let cv = projection.scores.cv.context("run has no CV score")?;
    let holdout = projection
        .scores
        .holdout
        .context("run has no HOLDOUT score")?;
    ensure!(
        champion.cv_value.is_finite() && cv.is_finite() && holdout.is_finite(),
        "committed scores are not finite"
    );
    ensure!(
        projection.scores.test.is_none(),
        "fixture has no TEST dataset"
    );
    let expected = format!(
        "champion {} ({} {:.3}) · CV {cv:.3} · HOLDOUT {holdout:.3} · TEST n/a",
        champion.algorithm_id, champion.metric, champion.cv_value
    );
    use portable_pty::{native_pty_system, CommandBuilder, PtySize};
    let pair = native_pty_system().openpty(PtySize {
        rows: 35,
        cols: 120,
        pixel_width: 0,
        pixel_height: 0,
    })?;
    let mut command = CommandBuilder::new(home.join("jopedime"));
    command.env("JOPEDIME_HOME", home);
    command.env("JAIN_ACCEPT_TERMS", "1");
    command.env("JAIN_ALLOW_LOCAL_FILES", "1");
    command.env("TERM", "dumb");
    command.env("NO_COLOR", "1");
    let mut child = pair.slave.spawn_command(command)?;
    drop(pair.slave);
    let mut reader = pair.master.try_clone_reader()?;
    let mut writer = pair.master.take_writer()?;
    let (send, receive) = mpsc::channel();
    thread::spawn(move || {
        let mut bytes = [0_u8; 4096];
        while let Ok(count) = reader.read(&mut bytes) {
            if count == 0 || send.send(bytes[..count].to_vec()).is_err() {
                break;
            }
        }
    });
    let attach = if key == KEY {
        format!("\\attach {id}\n")
    } else {
        String::new()
    };
    let commands=format!("\\landmarks none\n\\lang rust,python\n\\target target --task regression\n\\effort fast --evals 1 --time 120s\n\\open {}\n\\run --seed 73 --idempotency-key {key}\n{attach}",csv.display());
    writer.write_all(commands.as_bytes())?;
    writer.flush()?;
    let deadline = Instant::now() + Duration::from_secs(40);
    let marker = if key == KEY {
        format!("run {id} started")
    } else {
        " started".into()
    };
    let mut bytes = vec![];
    let result = (|| -> Result<String> {
        loop {
            bytes.extend(receive.recv_timeout(deadline.saturating_duration_since(Instant::now()))?);
            ensure!(bytes.len() < 1024 * 1024, "PTY transcript exceeds limit");
            let text = String::from_utf8_lossy(&bytes);
            // A completed idempotent retry subscribes after its final watermark.
            // Explicit attach replays that same run's actual durable score card.
            if text.contains(&marker)
                && text.rsplit_once("Run finished").is_some_and(|(_, card)| {
                    card.split_whitespace()
                        .collect::<Vec<_>>()
                        .join(" ")
                        .contains(&expected)
                })
            {
                return Ok(text.into_owned());
            }
            if text.contains("error:") {
                bail!("terminal run failed: {text}");
            }
        }
    })();
    let _ = writer.write_all(b"\\quit\n");
    let _ = writer.flush();
    if child.try_wait()?.is_none() {
        let _ = child.kill();
    }
    let _ = child.wait();
    // Preserve the actual PTY bytes even when replay or score matching fails.
    if let Some(root) = std::env::var_os("JOPE_INT01_RECEIPTS") {
        fs::create_dir_all(&root)?;
        fs::write(PathBuf::from(root).join("terminal.txt"), &bytes)?;
    }
    result.with_context(|| format!("terminal did not render committed score card: {expected}"))
}

// A live SSE subscription remains open after completion. Batch must stop at
// the committed terminal event, and a cursor already there must emit a receipt.
pub(crate) fn ndjson_receipt(
    home: &Path,
    spec: &RunSpec,
    completed: &RunSnapshot,
    after: Option<DecimalU64>,
) -> Result<()> {
    let mut command = batch_command(home, spec);
    command.args(["--events", "ndjson"]);
    if let Some(after) = after {
        command.args(["--after-event", &after.0.to_string()]);
    }
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let deadline = Instant::now() + Duration::from_secs(8);
    let mut expired = false;
    while child.try_wait()?.is_none() {
        if Instant::now() >= deadline {
            expired = true;
            child.kill()?;
            break;
        }
        thread::sleep(Duration::from_millis(20));
    }
    let output = child.wait_with_output()?;
    if let Some(root) = std::env::var_os("JOPE_INT01_RECEIPTS") {
        let name = if after.is_some() {
            "ndjson-after-final"
        } else {
            "ndjson-completed"
        };
        fs::write(
            PathBuf::from(&root).join(format!("{name}.stdout")),
            &output.stdout,
        )?;
        fs::write(
            PathBuf::from(&root).join(format!("{name}.stderr")),
            &output.stderr,
        )?;
    }
    ensure!(
        !expired,
        "batch NDJSON stayed subscribed after durable completion"
    );
    ensure!(
        output.status.success(),
        "NDJSON batch failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let lines = std::str::from_utf8(&output.stdout)?
        .lines()
        .map(serde_json::from_str::<Value>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let receipt = lines.last().context("NDJSON omitted final receipt")?;
    ensure!(
        receipt["kind"] == "receipt",
        "NDJSON did not terminate in a receipt"
    );
    let run: RunSnapshot = serde_json::from_value(receipt["run"].clone())?;
    ensure!(
        &run == completed,
        "NDJSON receipt differs from committed snapshot"
    );
    if after.is_some() {
        ensure!(
            lines.len() == 1,
            "cursor at final watermark replayed events"
        );
    } else {
        ensure!(
            lines.len() > 1 && lines[lines.len() - 2]["kind"] == "run.finished",
            "NDJSON omitted final durable event"
        );
    }
    Ok(())
}

pub(super) fn wait_complete<T: Transport>(client: &T, run: &jope_api::ids::RunId) -> Result<()> {
    let deadline = Instant::now() + Duration::from_secs(150);
    loop {
        let snapshot = client.run_status(run)?;
        if snapshot.state == jope_api::RunState::Completed {
            return Ok(());
        }
        ensure!(
            snapshot.state != jope_api::RunState::Failed,
            "independent run failed"
        );
        ensure!(Instant::now() < deadline, "independent run timed out");
        thread::sleep(Duration::from_millis(25));
    }
}

pub(super) fn start_independent<T: Transport>(
    client: &T,
    spec: &RunSpec,
    key: &str,
    surface: Surface,
) -> Result<SurfaceCapture> {
    let RunResult::Started(snapshot) = client.run_start(spec, Some(key))? else {
        bail!("independent run returned a dry plan")
    };
    wait_complete(client, &snapshot.run_id)?;
    capture(client, surface, &snapshot.run_id)
}
