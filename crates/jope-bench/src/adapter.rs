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
//! Contract [`CONTRACT`]: `<command...> <train.csv> <predict.csv> <out.csv> <task>`. `train.csv`
//! is headerless numeric with the target last; `predict.csv` is headerless numeric features only,
//! one row per held-out row in order; `out.csv` is one number per line in that order (a value for
//! regression, the probability of the positive class for binary, an integer class code for
//! multiclass); exit code 0 on success. `<command...> --version` must print one line naming the
//! engine and its version; it is recorded on the method. Reference adapters for XGBoost,
//! LightGBM, CatBoost, a scikit-learn random forest and a dependency-free mean live under
//! `crates/jope-bench/adapters/` (named `<engine>_adapter.py` so they never shadow the library
//! they import).

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuts::{build, FoldPlan};

    fn mean_adapter(timeout: f64) -> AdapterSpec {
        let script = concat!(env!("CARGO_MANIFEST_DIR"), "/adapters/mean_adapter.py");
        parse_spec(&format!("mean=python3 {script}"), timeout).unwrap()
    }

    fn toy() -> (Rows, Vec<f64>) {
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut rng = feat_math::Pcg32::new(9, 2);
        for _ in 0..80 {
            let a = rng.next_f64();
            let b = rng.next_f64();
            x.push(vec![a, b]);
            y.push(a + 0.5 * b);
        }
        (x, y)
    }

    #[test]
    fn specs_parse_and_the_mean_adapter_answers_the_contract() {
        assert!(parse_spec("x", 1.0).is_err());
        assert!(parse_spec("bad name=cmd", 1.0).is_err());
        assert!(parse_spec("ok=", 1.0).is_err());
        assert!(parse_spec("ok=cmd", 0.0).is_err());
        let spec = parse_spec("rf=python3 adapters/x.py --trees 10", 5.0).unwrap();
        assert_eq!(spec.id(), "adapter.rf");
        assert_eq!(spec.command, ["python3", "adapters/x.py", "--trees", "10"]);

        let mut spec = mean_adapter(30.0);
        spec.probe().unwrap();
        assert_eq!(spec.version.as_deref(), Some("mean 1"));

        let (x, y) = toy();
        let cuts = build(&FoldPlan::new("m", 1, 4), "m", "s", x.len()).unwrap();
        let outcome = evaluate_adapter(&spec, &x, &y, &cuts[0], Task::Regression);
        assert!(outcome.is_ok(), "{outcome:?}");
        let (_, y_train) = take(&x, &y, &cuts[0].train);
        let mean = y_train.iter().sum::<f64>() / y_train.len() as f64;
        assert!(outcome.predictions.iter().all(|p| (p - mean).abs() < 1e-12));
        assert!(
            outcome.score.unwrap() <= 0.0,
            "a constant predicts no variance"
        );
        assert!(outcome.fit_seconds > 0.0 && outcome.predict_seconds == 0.0);

        let deep = evaluate_deep_adapter(&spec, &x, &y, &cuts, Task::Regression);
        assert_eq!(deep.status, "ok", "{deep:?}");
        assert_eq!(deep.folds.len(), 4);
        assert!(deep.folds.iter().all(|f| f.dsl.is_empty()));

        // Binary: the positive rate; multiclass: the majority class.
        let yb: Vec<f64> = y
            .iter()
            .map(|v| if *v > 0.75 { 1.0 } else { 0.0 })
            .collect();
        let binary = evaluate_adapter(&spec, &x, &yb, &cuts[0], Task::Binary);
        assert!(binary.is_ok(), "{binary:?}");
        assert!(binary.predictions.iter().all(|p| (0.0..=1.0).contains(p)));
        let ym: Vec<f64> = y
            .iter()
            .map(|v| (v * 2.0).floor().clamp(0.0, 2.0))
            .collect();
        let multi = evaluate_adapter(&spec, &x, &ym, &cuts[0], Task::Multiclass);
        assert!(multi.is_ok(), "{multi:?}");
        assert!(multi.predictions.iter().all(|p| p.fract() == 0.0));
    }

    #[test]
    fn adapter_failures_are_rejections_with_the_cause() {
        let (x, y) = toy();
        let cuts = build(&FoldPlan::new("f", 1, 3), "f", "s", x.len()).unwrap();
        // Specs have no quoting, so a command with spaces inside one argument is built directly.
        assert!(
            parse_spec("huge=python3 x.py", 1e30).is_err(),
            "timeout must fit a Duration"
        );
        let failing = AdapterSpec {
            name: "fail".to_owned(),
            command: vec![
                "python3".to_owned(),
                "-c".to_owned(),
                "import sys; sys.stderr.write('boom'); sys.exit(3)".to_owned(),
            ],
            version: None,
            timeout_seconds: 30.0,
        };
        let outcome = evaluate_adapter(&failing, &x, &y, &cuts[0], Task::Regression);
        assert_eq!(outcome.status, "rejected");
        let reason = outcome.reason.as_deref().unwrap_or("");
        assert!(
            reason.contains("exit") && reason.contains("boom"),
            "{reason}"
        );

        let silent = AdapterSpec {
            command: vec!["python3".to_owned(), "-c".to_owned(), "pass".to_owned()],
            ..failing.clone()
        };
        let outcome = evaluate_adapter(&silent, &x, &y, &cuts[0], Task::Regression);
        assert!(outcome
            .reason
            .as_deref()
            .unwrap_or("")
            .contains("no out.csv"));

        let slow = AdapterSpec {
            command: vec![
                "python3".to_owned(),
                "-c".to_owned(),
                "import time; time.sleep(5)".to_owned(),
            ],
            timeout_seconds: 0.3,
            ..failing.clone()
        };
        let started = Instant::now();
        let outcome = evaluate_adapter(&slow, &x, &y, &cuts[0], Task::Regression);
        assert!(started.elapsed() < Duration::from_secs(4));
        assert!(outcome
            .reason
            .as_deref()
            .unwrap_or("")
            .contains("timed out"));

        let mut unprobeable = silent;
        assert!(unprobeable.probe().is_err(), "prints nothing for --version");
    }
}
