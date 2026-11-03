//! Rust port of the host-local JSONL bridge driver in `python/src/jope/remote_eval_driver.py`.
//!
//! Each shard owns a persistent bridge process. Rows stay ordered within a shard, while
//! completed shards are written as they finish. The Python entrypoint remains in place
//! while `jope.engines` continues to invoke it.

use serde_json::{json, Value};
use std::io::{self, BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

struct Bridge {
    child: Child,
    stdin: ChildStdin,
    lines: Receiver<io::Result<Option<String>>>,
    reader: Option<JoinHandle<()>>,
}

impl Bridge {
    fn spawn(bin_path: &str, subcommand: &str) -> io::Result<Self> {
        let mut child = Command::new(bin_path)
            .arg(subcommand)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()?;
        let stdin = child.stdin.take().expect("piped bridge stdin");
        let stdout = child.stdout.take().expect("piped bridge stdout");
        let (sender, lines) = mpsc::channel();
        let reader = thread::spawn(move || {
            let mut stdout = BufReader::new(stdout);
            loop {
                let mut line = String::new();
                match stdout.read_line(&mut line) {
                    Ok(0) => {
                        let _ = sender.send(Ok(None));
                        break;
                    }
                    Ok(_) => {
                        if sender.send(Ok(Some(line))).is_err() {
                            break;
                        }
                    }
                    Err(error) => {
                        let _ = sender.send(Err(error));
                        break;
                    }
                }
            }
        });
        Ok(Self {
            child,
            stdin,
            lines,
            reader: Some(reader),
        })
    }

    fn write_row(&mut self, row: &Value) -> io::Result<()> {
        let mut bytes = serde_json::to_vec(row).map_err(invalid_data)?;
        bytes.push(b'\n');
        self.stdin.write_all(&bytes)?;
        self.stdin.flush()
    }

    fn read_line(&self, timeout: Option<Duration>) -> io::Result<Option<String>> {
        match timeout {
            None => receive_line(&self.lines),
            Some(timeout) => match self.lines.recv_timeout(timeout) {
                Ok(line) => line,
                Err(RecvTimeoutError::Timeout) => Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "bridge read timed out",
                )),
                Err(RecvTimeoutError::Disconnected) => Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "bridge output reader stopped",
                )),
            },
        }
    }
}

impl Drop for Bridge {
    fn drop(&mut self) {
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = self.child.kill();
        }
        let _ = self.child.wait();
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
    }
}

fn receive_line(lines: &Receiver<io::Result<Option<String>>>) -> io::Result<Option<String>> {
    lines
        .recv()
        .map_err(|_| io::Error::new(io::ErrorKind::UnexpectedEof, "bridge output reader stopped"))?
}

fn invalid_data(error: serde_json::Error) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, error)
}

fn row_index(row: &Value) -> Value {
    row.get("_idx").cloned().unwrap_or(Value::Null)
}

fn row_timeout(row: &Value, default_timeout: Option<f64>) -> Option<f64> {
    row.get("_timeout_s")
        .map(Value::as_f64)
        .unwrap_or(default_timeout)
}

fn timeout_duration(seconds: f64) -> Duration {
    if seconds.is_finite() {
        Duration::from_secs_f64(seconds.max(0.0))
    } else {
        Duration::from_secs(365 * 24 * 60 * 60)
    }
}

fn error_result(index: Value, field: &str, message: String) -> String {
    let mut result = json!({"_idx": index, "valid": false, "errors": [message]});
    result[field] = Value::Bool(true);
    result.to_string()
}

fn distribute(rows: Vec<Value>, workers: usize) -> Vec<Vec<Value>> {
    let count = workers.max(1).min(rows.len());
    let mut shards = vec![Vec::new(); count];
    for (index, row) in rows.into_iter().enumerate() {
        shards[index % count].push(row);
    }
    shards
}

fn shard_worker(
    bin_path: &str,
    subcommand: &str,
    rows: Vec<Value>,
    default_timeout: Option<f64>,
) -> io::Result<Vec<String>> {
    let mut output = Vec::with_capacity(rows.len());
    let mut bridge = Bridge::spawn(bin_path, subcommand)?;

    for row in rows {
        let index = row_index(&row);
        let timeout = row_timeout(&row, default_timeout);
        let started = Instant::now();
        if bridge.child.try_wait()?.is_some() {
            bridge = Bridge::spawn(bin_path, subcommand)?;
        }
        if bridge.write_row(&row).is_err() {
            bridge = Bridge::spawn(bin_path, subcommand)?;
            bridge.write_row(&row)?;
        }

        let line = match timeout {
            Some(seconds) => bridge.read_line(Some(
                timeout_duration(seconds).saturating_sub(started.elapsed()),
            )),
            None => bridge.read_line(None),
        };
        let line = match line {
            Err(error) if error.kind() == io::ErrorKind::TimedOut => {
                let message = format!(
                    "eval timeout after {seconds:.1}s",
                    seconds = timeout.unwrap_or_default()
                );
                bridge = Bridge::spawn(bin_path, subcommand)?;
                output.push(error_result(index, "timeout", message));
                continue;
            }
            Err(error) => return Err(error),
            Ok(None) => {
                output.push(error_result(index.clone(), "bridge_crash", "eof".into()));
                bridge = Bridge::spawn(bin_path, subcommand)?;
                continue;
            }
            Ok(Some(line)) => line,
        };

        match serde_json::from_str::<Value>(&line) {
            Ok(mut result) => {
                let object = result.as_object_mut().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "bridge JSON result is not an object",
                    )
                })?;
                object.insert("_idx".into(), index);
                output.push(result.to_string());
            }
            Err(_) if line.starts_with('{') => output.push(line),
            Err(_) => output.push(json!({"_idx": index, "valid": false}).to_string()),
        }
    }
    Ok(output)
}

/// Read JSONL rows, run persistent local bridge processes, and write one result per row.
pub fn run<R: BufRead, W: Write>(
    bin_path: &str,
    subcommand: &str,
    workers: usize,
    default_timeout: Option<f64>,
    input: R,
    mut output: W,
) -> io::Result<()> {
    let mut rows = Vec::new();
    for line in input.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Ok(row) = serde_json::from_str::<Value>(line) {
            rows.push(row);
        }
    }
    if rows.is_empty() {
        return Ok(());
    }

    let shards = distribute(rows, workers);
    let task_count = shards.len();
    let (sender, receiver) = mpsc::channel();
    thread::scope(|scope| -> io::Result<()> {
        for shard in shards {
            let sender = sender.clone();
            let bin_path = bin_path.to_owned();
            let subcommand = subcommand.to_owned();
            scope.spawn(move || {
                let result = shard_worker(&bin_path, &subcommand, shard, default_timeout);
                let _ = sender.send(result);
            });
        }
        drop(sender);
        for _ in 0..task_count {
            let lines = receiver
                .recv()
                .map_err(|_| io::Error::other("bridge worker stopped without a result"))??;
            for line in lines {
                output.write_all(line.as_bytes())?;
                output.write_all(b"\n")?;
            }
        }
        output.flush()
    })
}

/// CLI adapter matching the Python driver's flags and defaults.
pub fn main_entry() -> io::Result<()> {
    let mut bin_path = None;
    let mut subcommand = "eval-regression".to_owned();
    let mut workers = 64_i64;
    let mut default_timeout = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let value = |args: &mut std::iter::Skip<std::env::Args>, name: &str| {
            args.next().ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("missing value for {name}"),
                )
            })
        };
        match arg.as_str() {
            "--bin" => bin_path = Some(value(&mut args, "--bin")?),
            "--subcommand" => subcommand = value(&mut args, "--subcommand")?,
            "--workers" => {
                workers = value(&mut args, "--workers")?.parse().map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidInput, "invalid integer for --workers")
                })?;
            }
            "--default-timeout" => {
                default_timeout = Some(value(&mut args, "--default-timeout")?.parse().map_err(
                    |_| {
                        io::Error::new(
                            io::ErrorKind::InvalidInput,
                            "invalid float for --default-timeout",
                        )
                    },
                )?);
            }
            "--help" | "-h" => {
                println!("usage: remote-eval-driver --bin PATH [--subcommand NAME] [--workers N] [--default-timeout SECONDS]");
                return Ok(());
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("unrecognized argument: {arg}"),
                ));
            }
        }
    }
    let bin_path = bin_path.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "the following arguments are required: --bin",
        )
    })?;
    run(
        &bin_path,
        &subcommand,
        workers.max(1) as usize,
        default_timeout,
        io::stdin().lock(),
        io::stdout().lock(),
    )
}

#[cfg(test)]
mod tests {
    use super::{distribute, row_index, row_timeout};
    use serde_json::json;

    #[test]
    fn round_robin_shards_keep_each_shards_input_order() {
        let shards = distribute((0..5).map(|n| json!({"_idx": n})).collect(), 2);
        assert_eq!(shards.len(), 2);
        assert_eq!(
            shards[0],
            vec![json!({"_idx": 0}), json!({"_idx": 2}), json!({"_idx": 4})]
        );
        assert_eq!(shards[1], vec![json!({"_idx": 1}), json!({"_idx": 3})]);
    }

    #[test]
    fn worker_count_is_clamped_by_shard_distribution() {
        assert_eq!(distribute(vec![json!({}), json!({})], 0).len(), 1);
        assert_eq!(distribute(vec![json!({}), json!({})], 64).len(), 2);
    }

    #[test]
    fn row_metadata_matches_python_get_defaults() {
        let row = json!({"_idx": 7, "_timeout_s": 0.25});
        assert_eq!(row_index(&row), json!(7));
        assert_eq!(row_timeout(&row, Some(3.0)), Some(0.25));
        assert_eq!(row_timeout(&json!({"_idx": 8}), Some(3.0)), Some(3.0));
        assert_eq!(row_timeout(&json!({"_timeout_s": null}), Some(3.0)), None);
    }
}
