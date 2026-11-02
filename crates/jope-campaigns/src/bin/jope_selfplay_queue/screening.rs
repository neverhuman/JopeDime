use super::*;

pub(super) fn validate_with_evaluator(evaluator: &Path, manifest: &Path) -> Result<()> {
    let output = Command::new(evaluator)
        .args(["validate-manifest", "--manifest"])
        .arg(manifest)
        .output()
        .with_context(|| format!("run Rust evaluator {}", evaluator.display()))?;
    if !output.status.success() {
        bail!(
            "Rust evaluator rejected {}: {}",
            manifest.display(),
            String::from_utf8_lossy(if output.stderr.is_empty() {
                &output.stdout
            } else {
                &output.stderr
            })
            .trim()
        );
    }
    Ok(())
}

pub(super) fn bounded_regression_screening(evaluator: &Path, manifest: &Path) -> Result<()> {
    let manifest_sha256 = sha256_file(manifest)?;
    let output_dir = std::env::temp_dir().join(format!(
        "jope-task-preflight-{}-{}-{}",
        &manifest_sha256[..16],
        std::process::id(),
        now_ms()
    ));
    fs::create_dir(&output_dir)
        .with_context(|| format!("create bounded screening output {}", output_dir.display()))?;
    let result = (|| -> Result<()> {
        let output = Command::new("/usr/bin/timeout")
            .args(["--signal=KILL", "--kill-after=5s"])
            .arg(format!("{REGRESSION_SCREENING_BUDGET_SECONDS}s"))
            .arg(evaluator)
            .arg("--manifest")
            .arg(manifest)
            .arg("--output-dir")
            .arg(&output_dir)
            .args(["--threads", "1", "--timeout-seconds", "5"])
            .output()
            .with_context(|| format!("run bounded regression screening {}", evaluator.display()))?;
        if !output.status.success() {
            let message = String::from_utf8_lossy(if output.stderr.is_empty() {
                &output.stdout
            } else {
                &output.stderr
            });
            bail!(
                "bounded regression screening rejected {}: {}",
                manifest.display(),
                message.trim()
            );
        }
        validate_bounded_screening_receipts(&output_dir)
    })();
    let cleanup = fs::remove_dir_all(&output_dir)
        .with_context(|| format!("remove bounded screening output {}", output_dir.display()));
    match (result, cleanup) {
        (Err(error), _) => Err(error),
        (Ok(()), Err(error)) => Err(error),
        (Ok(()), Ok(())) => Ok(()),
    }
}

#[derive(Deserialize)]
pub(super) struct BoundedScreeningReceipt {
    valid: bool,
    failure: Option<String>,
}

pub(super) fn validate_bounded_screening_receipts(output_dir: &Path) -> Result<()> {
    let mut receipt_paths = fs::read_dir(output_dir)
        .with_context(|| format!("read bounded screening output {}", output_dir.display()))?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    receipt_paths.retain(|path| {
        path.file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with(".receipts.jsonl"))
    });
    ensure!(
        receipt_paths.len() == 1,
        "bounded regression screening emitted {} receipt files",
        receipt_paths.len()
    );
    let bytes = fs::read(&receipt_paths[0])
        .with_context(|| format!("read screening receipts {}", receipt_paths[0].display()))?;
    let mut receipt_count = 0_usize;
    for (index, line) in bytes.split(|byte| *byte == b'\n').enumerate() {
        if line.is_empty() {
            continue;
        }
        receipt_count += 1;
        let receipt: BoundedScreeningReceipt = serde_json::from_slice(line)
            .with_context(|| format!("decode bounded screening receipt line {}", index + 1))?;
        ensure!(
            receipt.valid,
            "bounded regression screening candidate failed: {}",
            receipt.failure.as_deref().unwrap_or("unknown")
        );
        ensure!(
            receipt.failure.is_none(),
            "bounded regression screening valid receipt carries a failure"
        );
    }
    ensure!(
        receipt_count > 0,
        "bounded regression screening emitted no candidate receipts"
    );
    Ok(())
}

pub(super) fn evaluator_failure_is_permanent(error: &anyhow::Error) -> bool {
    format!("{error:#}").contains("Rust evaluator rejected")
}

/// The runner and evaluator binaries every spec of one cycle pins, with the digests taken
/// once at the start of preparation.
pub(super) struct PinnedTools<'a> {
    pub(super) runner: (&'a Path, &'a String),
    pub(super) evaluator: (&'a Path, &'a String),
}

impl PinnedTools<'_> {
    /// Refuse the cycle if either binary changed while its specs were written.
    pub(super) fn verify_unchanged(&self) -> Result<()> {
        for (source, digest) in [self.runner, self.evaluator] {
            if !source.is_file() || sha256_file(source)? != *digest {
                bail!("spec input content pin drift: {}", source.display());
            }
        }
        Ok(())
    }
}

pub(super) fn render_spec(
    path: &Path,
    job_id: &str,
    manifest: &ValidManifest,
    tools: &PinnedTools<'_>,
) -> Result<()> {
    let train = path_field(&manifest.value, "train_csv")?;
    let test = path_field(&manifest.value, "test_csv")?;
    let (runner, runner_sha256) = tools.runner;
    let (evaluator, evaluator_sha256) = tools.evaluator;
    let inputs = [
        ("runner", runner, "runner.sh", runner_sha256.clone()),
        (
            "manifest",
            manifest.path.as_path(),
            "manifest.json",
            manifest.sha256.clone(),
        ),
        (
            "train",
            train.as_path(),
            "train.csv",
            string_field(&manifest.value, "train_sha256")?.to_owned(),
        ),
        (
            "test",
            test.as_path(),
            "test.csv",
            string_field(&manifest.value, "test_sha256")?.to_owned(),
        ),
        (
            "evaluator",
            evaluator,
            "corpus_eval",
            evaluator_sha256.clone(),
        ),
    ];
    // The manifest and its train/test files keep their per-manifest content check. The
    // runner/evaluator digests are per cycle and rechecked by `PinnedTools::verify_unchanged`.
    for (name, source, _, digest) in &inputs {
        if matches!(*name, "runner" | "evaluator") {
            continue;
        }
        if !source.is_file() || sha256_file(source)? != *digest {
            bail!("spec input content pin drift: {}", source.display());
        }
    }
    let quoted = |value: &str| serde_json::to_string(value).expect("string JSON");
    let mut lines = vec![
        "[job]".to_owned(),
        format!("id = {}", quoted(job_id)),
        format!("project = \"{PROJECT}\""),
        "tier = \"normal\"".to_owned(),
        "command = \"/usr/bin/env\"".to_owned(),
        "args = [\"{input:runner}\", \"{input:manifest}\", \"{input:train}\", \
         \"{input:test}\", \"{input:evaluator}\", \"{output:receipt}\"]"
            .to_owned(),
        String::new(),
    ];
    lines.extend(EVAL.render_resource_block().lines().map(str::to_owned));
    lines.extend([
        String::new(),
        "[env]".to_owned(),
        format!("RAYON_NUM_THREADS = \"{}\"", EVAL.omp_threads),
        format!("OMP_NUM_THREADS = \"{}\"", EVAL.omp_threads),
        format!("OPENBLAS_NUM_THREADS = \"{}\"", EVAL.omp_threads),
        format!("MKL_NUM_THREADS = \"{}\"", EVAL.omp_threads),
        format!("NUMEXPR_NUM_THREADS = \"{}\"", EVAL.omp_threads),
        String::new(),
        "[preempt]".to_owned(),
        format!("preemptible = {}", EVAL.preemptible),
        "checkpoint = \"none\"".to_owned(),
        "resume = \"rerun\"".to_owned(),
        String::new(),
    ]);
    lines.extend(EVAL.render_limits_block().lines().map(str::to_owned));
    lines.push(String::new());
    let mut text = lines.join("\n");
    text.push('\n');
    for (name, source, target, digest) in inputs {
        text.push_str("[[input]]\n");
        text.push_str(&format!("name = {}\n", quoted(name)));
        text.push_str(&format!(
            "source = {}\n",
            quoted(source.to_string_lossy().as_ref())
        ));
        text.push_str(&format!("target = {}\n", quoted(target)));
        text.push_str(&format!("sha256 = {}\n\n", quoted(&digest)));
    }
    text.push_str("[[output]]\nname = \"receipt\"\nsource = \"receipt\"\ndirectory = true\n");
    atomic_write(path, text.as_bytes())
}
