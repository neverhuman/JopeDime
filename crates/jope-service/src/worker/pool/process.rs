use super::*;

pub(super) struct Process {
    pub(super) child: Child,
    // Closing this handle kills all Windows job members; it is never inherited by the child.
    pub(super) containment: Option<Containment>,
    pub(super) requests: SyncSender<Vec<u8>>,
    pub(super) responses: Receiver<io::Result<Response>>,
    pub(super) pipe_thread: Option<JoinHandle<()>>,
    pub(super) matrix: PathBuf,
    pub(super) cap: u64,
    pub(super) evaluations: u32,
    pub(super) baseline: u64,
    pub(super) last_rss: u64,
}

impl Process {
    pub(super) fn retired_after_reply(&mut self, started: Instant, deadline: Duration) -> bool {
        let waiting = Instant::now();
        loop {
            match self.child.try_wait() {
                Ok(Some(status)) => {
                    return status.code() == Some(super::super::process::RETIRE_EXIT_CODE)
                }
                Err(_) => return false,
                Ok(None) => {}
            }
            if waiting.elapsed() >= SAMPLE_INTERVAL || started.elapsed() >= deadline {
                return false;
            }
            thread::sleep(Duration::from_millis(1));
        }
    }

    pub(super) fn retire(mut self) -> io::Result<()> {
        super::super::process::kill(&mut self.child);
        if self.child.try_wait()?.is_none() {
            return Err(io::Error::other("retired worker was not reaped"));
        }
        // Drop joins the pipe helper before this operation returns.
        Ok(())
    }

    pub(super) fn spawn(config: &PoolConfig, request: &Request) -> io::Result<Self> {
        Self::start(
            config,
            PoolLaunch {
                matrix: request.matrix_path.clone(),
                cap: request.limits.rss_bytes,
            },
        )
    }

    pub(super) fn start(config: &PoolConfig, launch: PoolLaunch) -> io::Result<Self> {
        let config = config.clone();
        let (requests, input_rx) = mpsc::sync_channel::<Vec<u8>>(1);
        let (output_tx, responses) = mpsc::sync_channel(1);
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        let cap = launch.cap;
        let pipe_thread = thread::Builder::new()
            .name("worker-pipe".into())
            .spawn(move || {
                // Linux ties PDEATHSIG to the creating THREAD. This same thread
                // must own startup and remain alive for the entire process lifetime.
                let startup = (|| {
                    let mut command = Command::new(&config.executable);
                    #[cfg(test)]
                    if config.fixture {
                        command
                            .args([
                                "--exact",
                                config.fixture_test.unwrap_or("worker::tests::child_helper"),
                                "--ignored",
                                "--nocapture",
                                "--quiet",
                            ])
                            .env("JOPE_WORKER_FIXTURE", "worker");
                    } else {
                        command.args(["__worker", "--protocol", "1"]);
                    }
                    #[cfg(not(test))]
                    command.args(["__worker", "--protocol", "1"]);
                    #[cfg(test)]
                    if config.wrong_fence {
                        command.env("JOPE_WORKER_WRONG_FENCE", "1");
                    }
                    command
                        .stdin(Stdio::piped())
                        .stdout(Stdio::piped())
                        .stderr(Stdio::null());
                    super::super::process::prepare(&mut command);
                    let mut child = command.spawn()?;
                    // No request precedes Job Object assignment; the executable
                    // only boots and waits for input at this point.
                    let containment = match Containment::assign(&child, cap) {
                        Ok(value) => value,
                        Err(error) => {
                            super::super::process::kill(&mut child);
                            return Err(error);
                        }
                    };
                    let input = child.stdin.take().unwrap();
                    let output = child.stdout.take().unwrap();
                    Ok((child, containment, input, output))
                })();
                let (child, containment, mut input, mut output) = match startup {
                    Ok(value) => value,
                    Err(error) => {
                        let _ = ready_tx.send(Err(error));
                        return;
                    }
                };
                if let Err(error) = ready_tx.send(Ok((child, containment))) {
                    if let Ok((mut child, _containment)) = error.0 {
                        super::super::process::kill(&mut child);
                    }
                    return;
                }
                #[cfg(test)]
                if config.fixture {
                    use std::io::{BufRead, BufReader};
                    let mut banner = BufReader::new(output);
                    let mut line = String::new();
                    loop {
                        line.clear();
                        if banner.read_line(&mut line).unwrap_or(0) == 0 {
                            return;
                        }
                        if line.trim() == "WORKER_FIXTURE_READY" {
                            break;
                        }
                    }
                    // No request has been sent, so the buffered reader cannot hold frame bytes.
                    output = banner.into_inner();
                }
                while let Ok(bytes) = input_rx.recv() {
                    let result = input
                        .write_all(&bytes)
                        .and_then(|()| input.flush())
                        .and_then(|()| decode_frame(&mut output).map_err(io::Error::other))
                        .and_then(|v| serde_json::from_value(v).map_err(io::Error::other));
                    let failed = result.is_err();
                    if output_tx.send(result).is_err() || failed {
                        break;
                    }
                }
            })?;
        let startup = ready_rx
            .recv() // default fallback
            .unwrap_or_else(|_| {
                Err(io::Error::other(
                    "worker creator thread exited during startup",
                ))
            });
        let (child, containment) = match startup {
            Ok(value) => value,
            Err(error) => {
                let _ = pipe_thread.join();
                return Err(error);
            }
        };
        Ok(Self {
            child,
            containment: Some(containment),
            requests,
            responses,
            pipe_thread: Some(pipe_thread),
            matrix: launch.matrix,
            cap,
            evaluations: 0,
            baseline: 0,
            last_rss: 0,
        })
    }
}

impl Drop for Process {
    fn drop(&mut self) {
        super::super::process::kill(&mut self.child);
        self.containment.take();
        // A completed pipe thread can be waiting for its next request; disconnect that queue.
        let (closed_tx, _) = mpsc::sync_channel(1);
        let old = std::mem::replace(&mut self.requests, closed_tx);
        drop(old);
        if let Some(handle) = self.pipe_thread.take() {
            let _ = handle.join();
        }
    }
}
