//! Tier-2 baselines: external engines run as subprocesses on the same cuts and scored by this
//! crate's evaluator from the predictions they return. The controller writes a cut's training
//! rows (features plus target) and its held-out features (no target) to a temporary directory
//! readable by its own user only, runs the adapter, reads one prediction per held-out row and
//! scores it exactly as it scores a program. The adapter is handed no held-out target, and
//! nothing it prints is trusted as a score. That is a guarantee about what an adapter is given
//! and about who scores it, not a sandbox: the child runs with the benchmark's user, working
//! directory and environment, and it may tune itself inside its own training rows (as the
//! resampled search does), so "controller-scored" does not mean "untuned".
//!
//! Contract [`crate::adapter::CONTRACT`]:
//! `<command...> <train.csv> <predict.csv> <out.csv> <task>`. `train.csv` is headerless numeric
//! with the target last; `predict.csv` is headerless numeric features only,
//! one row per held-out row in order; `out.csv` is one number per line in that order (a value for
//! regression, the probability of the positive class for binary, an integer class code for
//! multiclass); exit code 0 on success. `<command...> --version` must print one line naming the
//! engine and its version; it is recorded on the method. Reference adapters for XGBoost,
//! LightGBM, CatBoost and a scikit-learn random forest live under `crates/jope-bench/adapters/`
//! (named `<engine>_adapter.py` so they never shadow the library they import). The
//! dependency-free constant adapter the contract tests run is
//! [`crate::adapter::mean_adapter`], reached as the hidden `jope-bench mean-adapter` subcommand.

use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use feat_math::Rows;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::cuts::Cut;
use crate::eval::{pool, score, take, Deep, Outcome};
use crate::task::Task;

/// The subprocess contract every adapter implements.
pub const CONTRACT: &str = "jope.bench-adapter/v1";

/// An external engine and how to run it.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AdapterSpec {
    /// Ladder name (`adapter.<name>`).
    pub name: String,
    /// The command and its leading arguments; the four contract arguments are appended.
    pub command: Vec<String>,
    /// The first line `<command...> --version` printed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    /// Wall-clock cap per invocation, in seconds.
    pub timeout_seconds: f64,
}

/// `NAME=<command and arguments>`, split on whitespace (no quoting: keep adapter paths simple).
pub fn parse_spec(text: &str, timeout_seconds: f64) -> Result<AdapterSpec> {
    let Some((name, command)) = text.split_once('=') else {
        bail!("--adapter wants NAME=<command...>, got {text:?}");
    };
    let name = name.trim();
    if name.is_empty()
        || !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        bail!("adapter name {name:?}: use letters, digits, '-' and '_'");
    }
    let command: Vec<String> = command.split_whitespace().map(str::to_owned).collect();
    if command.is_empty() {
        bail!("adapter {name}: empty command");
    }
    if !timeout_seconds.is_finite()
        || timeout_seconds <= 0.0
        || Duration::try_from_secs_f64(timeout_seconds).is_err()
    {
        bail!(
            "adapter {name}: the timeout must be a positive number of seconds that fits a Duration"
        );
    }
    Ok(AdapterSpec {
        name: name.to_owned(),
        command,
        version: None,
        timeout_seconds,
    })
}

impl AdapterSpec {
    /// Ladder id.
    pub fn id(&self) -> String {
        format!("adapter.{}", self.name)
    }

    /// Run `<command...> --version` and record its first line. Fails when the adapter does not
    /// answer, so a broken adapter is refused before any dataset runs.
    pub fn probe(&mut self) -> Result<()> {
        let run = run_with_timeout(
            &self.command,
            &["--version".to_owned()],
            self.timeout_seconds,
        )
        .with_context(|| format!("adapter {}: --version", self.name))?;
        if !run.success {
            bail!(
                "adapter {}: `--version` exited with {}: {}",
                self.name,
                run.status,
                tail(&run.stderr)
            );
        }
        let version = run.stdout.lines().next().unwrap_or("").trim().to_owned();
        if version.is_empty() {
            bail!("adapter {}: `--version` printed nothing", self.name);
        }
        self.version = Some(version);
        Ok(())
    }
}

struct Run {
    success: bool,
    status: String,
    stdout: String,
    stderr: String,
}

/// A temporary directory readable and writable by this user only.
fn private_tempdir(prefix: &str) -> Result<tempfile::TempDir> {
    let mut builder = tempfile::Builder::new();
    builder.prefix(prefix);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        builder.permissions(std::fs::Permissions::from_mode(0o700));
    }
    builder.tempdir().context("adapter scratch directory")
}

/// Run a command with its output captured, killing it after `timeout_seconds`.
fn run_with_timeout(command: &[String], args: &[String], timeout_seconds: f64) -> Result<Run> {
    let dir = private_tempdir("jope-bench-adapter-io-")?;
    let stdout_path = dir.path().join("stdout");
    let stderr_path = dir.path().join("stderr");
    let (program, leading) = command.split_first().context("empty adapter command")?;
    let mut child = Command::new(program)
        .args(leading)
        .args(args)
        .stdin(Stdio::null())
        .stdout(std::fs::File::create(&stdout_path)?)
        .stderr(std::fs::File::create(&stderr_path)?)
        .spawn()
        .with_context(|| format!("spawn {program}"))?;
    let deadline = Instant::now()
        + Duration::try_from_secs_f64(timeout_seconds)
            .context("the timeout does not fit a Duration")?;
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            bail!("timed out after {timeout_seconds} s");
        }
        std::thread::sleep(Duration::from_millis(25));
    };
    Ok(Run {
        success: status.success(),
        status: status.to_string(),
        stdout: std::fs::read_to_string(&stdout_path).unwrap_or_default(),
        stderr: std::fs::read_to_string(&stderr_path).unwrap_or_default(),
    })
}

fn tail(text: &str) -> String {
    let trimmed = text.trim();
    let start = trimmed.len().saturating_sub(400);
    let mut cut = start;
    while cut < trimmed.len() && !trimmed.is_char_boundary(cut) {
        cut += 1;
    }
    trimmed[cut..].replace('\n', " | ")
}

fn write_csv(path: &Path, rows: &[Vec<f64>], targets: Option<&[f64]>) -> Result<()> {
    let mut text = String::new();
    for (i, row) in rows.iter().enumerate() {
        let mut cells: Vec<String> = row.iter().map(|v| format!("{v:?}")).collect();
        if let Some(targets) = targets {
            cells.push(format!("{:?}", targets[i]));
        }
        text.push_str(&cells.join(","));
        text.push('\n');
    }
    std::fs::write(path, text).with_context(|| format!("write {}", path.display()))
}

/// Run the adapter on `cut.train`, read its predictions for `cut.hold`, score for `task`.
pub fn evaluate_adapter(spec: &AdapterSpec, x: &Rows, y: &[f64], cut: &Cut, task: Task) -> Outcome {
    let hold_rows = cut.hold.len();
    let rejected =
        |reason: String, seconds: f64| Outcome::rejected(reason, seconds, 0.0, hold_rows);
    let dir = match private_tempdir("jope-bench-adapter-") {
        Ok(dir) => dir,
        Err(error) => return rejected(format!("scratch directory: {error:#}"), 0.0),
    };
    let (x_train, y_train) = take(x, y, &cut.train);
    let (x_hold, y_hold) = take(x, y, &cut.hold);
    let train = dir.path().join("train.csv");
    let predict = dir.path().join("predict.csv");
    let out = dir.path().join("out.csv");
    if let Err(error) = write_csv(&train, &x_train, Some(&y_train))
        .and_then(|()| write_csv(&predict, &x_hold, None))
    {
        return rejected(format!("write inputs: {error:#}"), 0.0);
    }
    let args = [
        train.to_string_lossy().into_owned(),
        predict.to_string_lossy().into_owned(),
        out.to_string_lossy().into_owned(),
        task.as_str().to_owned(),
    ];
    let started = Instant::now();
    let run = match run_with_timeout(&spec.command, &args, spec.timeout_seconds) {
        Ok(run) => run,
        Err(error) => {
            return rejected(
                format!("adapter: {error:#}"),
                started.elapsed().as_secs_f64(),
            )
        }
    };
    let seconds = started.elapsed().as_secs_f64();
    if !run.success {
        return rejected(
            format!("adapter exited with {}: {}", run.status, tail(&run.stderr)),
            seconds,
        );
    }
    let text = match std::fs::read_to_string(&out) {
        Ok(text) => text,
        Err(error) => return rejected(format!("adapter wrote no out.csv: {error}"), seconds),
    };
    let mut predictions = Vec::with_capacity(hold_rows);
    for (line_no, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match line.parse::<f64>() {
            Ok(v) if v.is_finite() => predictions.push(v),
            Ok(_) => {
                return rejected(
                    format!("non-finite prediction on line {}", line_no + 1),
                    seconds,
                )
            }
            Err(_) => {
                return rejected(
                    format!("out.csv line {} is not a number: {line:?}", line_no + 1),
                    seconds,
                )
            }
        }
    }
    if predictions.len() != hold_rows {
        return rejected(
            format!(
                "adapter returned {} predictions for {hold_rows} rows",
                predictions.len()
            ),
            seconds,
        );
    }
    match score(task, &y_hold, &predictions) {
        Ok((score, metrics, note)) => Outcome {
            status: "ok".to_owned(),
            reason: None,
            score,
            metrics,
            note,
            fit_seconds: seconds,
            predict_seconds: 0.0,
            hold_rows,
            predictions,
        },
        Err(reason) => rejected(reason, seconds),
    }
}

/// The adapter on every cut, pooled out of fold; cuts run in parallel (each is one process).
pub fn evaluate_deep_adapter(
    spec: &AdapterSpec,
    x: &Rows,
    y: &[f64],
    cuts: &[Cut],
    task: Task,
) -> Deep {
    let outcomes: Vec<Outcome> = cuts
        .par_iter()
        .map(|cut| evaluate_adapter(spec, x, y, cut, task))
        .collect();
    let labels: Vec<String> = cuts.iter().map(|_| String::new()).collect();
    pool(cuts, &labels, outcomes, y, task)
}

const MEAN_USAGE: &str =
    "jope-bench adapter (contract jope.bench-adapter/v1): the constant baseline.

Regression predicts the training mean, binary the training positive rate, multiclass the majority
training class (the earliest on a tie). It exercises the adapter contract in tests and is the floor
every other adapter must clear.

    jope-bench mean-adapter --version
    jope-bench mean-adapter <train.csv> <predict.csv> <out.csv> <task>
";

/// The constant adapter process: prints `mean 1` for `--version`, else fits and predicts per the
/// contract. Returns the exit code: 0 on success, 1 on unusable input, 2 on a usage error.
pub fn mean_adapter(args: &[String]) -> i32 {
    if let [flag] = args {
        if flag == "--version" {
            println!("mean 1");
            return 0;
        }
    }
    let [train, predict, out, task] = args else {
        eprint!("{MEAN_USAGE}");
        return 2;
    };
    match constant_predictions(Path::new(train), Path::new(predict), task) {
        Ok(Some((value, rows))) => {
            let text: String = (0..rows).map(|_| format!("{value:?}\n")).collect();
            match std::fs::write(out, text) {
                Ok(()) => 0,
                Err(error) => {
                    eprintln!("write {out}: {error}");
                    1
                }
            }
        }
        Ok(None) => {
            eprintln!("unknown task '{task}'");
            2
        }
        Err(error) => {
            eprintln!("{error:#}");
            1
        }
    }
}

/// The constant and the number of rows to predict, or `None` for an unknown task.
fn constant_predictions(train: &Path, predict: &Path, task: &str) -> Result<Option<(f64, usize)>> {
    let text =
        std::fs::read_to_string(train).with_context(|| format!("read {}", train.display()))?;
    let targets = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let (_, last) = line
                .rsplit_once(',')
                .with_context(|| format!("training row without a target: {line:?}"))?;
            last.trim()
                .parse::<f64>()
                .with_context(|| format!("training target {last:?} is not a number"))
        })
        .collect::<Result<Vec<f64>>>()?;
    let rows = std::fs::read_to_string(predict)
        .with_context(|| format!("read {}", predict.display()))?
        .lines()
        .filter(|line| !line.trim().is_empty())
        .count();
    if targets.is_empty() {
        bail!("no training rows");
    }
    let n = targets.len() as f64;
    let value = match task {
        "regression" => targets.iter().sum::<f64>() / n,
        "binary" => targets.iter().filter(|t| **t == 1.0).count() as f64 / n,
        "multiclass" => {
            // Class counts in first-seen order; the earliest class wins a tie.
            let mut counts: Vec<(i64, usize)> = Vec::new();
            for target in &targets {
                if !target.is_finite() {
                    bail!("class code {target} is not finite");
                }
                let class = target.trunc() as i64;
                match counts.iter_mut().find(|(c, _)| *c == class) {
                    Some((_, count)) => *count += 1,
                    None => counts.push((class, 1)),
                }
            }
            let best = counts.iter().map(|(_, count)| *count).max().unwrap_or(0);
            let (class, _) = counts
                .iter()
                .find(|(_, count)| *count == best)
                .context("no class")?;
            *class as f64
        }
        _ => return Ok(None),
    };
    Ok(Some((value, rows)))
}
