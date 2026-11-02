use super::*;

pub(super) fn write_immutable(path: &Path, body: &[u8]) -> Result<()> {
    if path.is_file() {
        ensure!(
            fs::read(path)? == body,
            "immutable pair repair artifact drift at {}",
            path.display()
        );
        return Ok(());
    }
    fs::create_dir_all(path.parent().context("immutable path has no parent")?)?;
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .open(path)?;
    file.write_all(body)?;
    file.sync_data()?;
    Ok(())
}

pub(super) fn repair_manifest(
    root: &Path,
    request: &PairRepairRequestV1,
    target: &TargetCandidate,
    evaluator: &EvaluatorBinding,
) -> Result<PathBuf> {
    let directory = root.join("ops/causal-pair-repairs/manifests");
    let path = directory.join(format!(
        "{}-{}.json",
        request.pair_id,
        lane_name(request.missing_lane)
    ));
    let mut manifest = target.manifest.clone();
    manifest["evaluation_lane"] = json!("causal_pair_repair");
    manifest["candidates"] = json!([target.candidate]);
    manifest["causal_pair_repair"] = json!({
        "schema": REPAIR_SCHEMA,
        "pair_id": request.pair_id,
        "missing_lane": request.missing_lane,
        "original_manifest_path": target.manifest_path,
        "original_manifest_sha256": target.manifest_sha256,
        "original_evaluator_sha256": evaluator.sha256,
    });
    write_immutable(&path, &serde_json::to_vec(&manifest)?)?;
    Ok(path)
}

pub(super) fn toml_quote(value: &str) -> Result<String> {
    ensure!(
        !value.contains(['"', '\\', '\n', '\r']),
        "unsafe generated TOML value"
    );
    Ok(format!("\"{value}\""))
}

pub(super) fn append_input(
    lines: &mut Vec<String>,
    name: &str,
    source: &Path,
    target: &str,
    expected_sha256: Option<&str>,
) -> Result<()> {
    let actual = sha256_file(source)?;
    if let Some(expected) = expected_sha256 {
        ensure!(actual == expected, "{name} content pin drift");
    }
    lines.extend([
        "[[input]]".to_owned(),
        format!("name = {}", toml_quote(name)?),
        format!("source = {}", toml_quote(&source.to_string_lossy())?),
        format!("target = {}", toml_quote(target)?),
        format!("sha256 = {}", toml_quote(&actual)?),
        String::new(),
    ]);
    Ok(())
}

pub(super) fn render_spec(
    args: &Args,
    runner: &Path,
    request: &PairRepairRequestV1,
    target: &TargetCandidate,
    evaluator: &EvaluatorBinding,
    manifest: &Path,
    attempt: usize,
) -> Result<(String, PathBuf)> {
    let job_id = format!(
        "jte-pair-repair-{}-{}-a{attempt}",
        &request.pair_id.as_str()[..20],
        match request.missing_lane {
            CausalLane::NativeDlDirect => "n",
            CausalLane::MatchedGpComparator => "g",
        }
    );
    let path = args
        .root
        .join("ops/causal-pair-repairs/specs")
        .join(format!("{job_id}.toml"));
    let train = PathBuf::from(
        target
            .manifest
            .get("train_csv")
            .and_then(Value::as_str)
            .context("repair manifest lacks train CSV")?,
    );
    let test = PathBuf::from(
        target
            .manifest
            .get("test_csv")
            .and_then(Value::as_str)
            .context("repair manifest lacks test CSV")?,
    );
    let train_sha = target
        .manifest
        .get("train_sha256")
        .and_then(Value::as_str)
        .context("repair manifest lacks train hash")?;
    let test_sha = target
        .manifest
        .get("test_sha256")
        .and_then(Value::as_str)
        .context("repair manifest lacks test hash")?;
    let mut lines = vec![
        "[job]".to_owned(),
        format!("id = {}", toml_quote(&job_id)?),
        "project = \"jope-selfplay-eval\"".to_owned(),
        "tier = \"normal\"".to_owned(),
        "command = \"/usr/bin/env\"".to_owned(),
        "args = [\"{input:runner}\", \"{input:manifest}\", \"{input:train}\", \"{input:test}\", \"{input:evaluator}\", \"{output:receipt}\"]".to_owned(),
        String::new(),
    ];
    lines.extend(EVAL.render_resource_block().lines().map(str::to_owned));
    lines.push(String::new());
    lines.extend(EVAL.render_placement_block().lines().map(str::to_owned));
    lines.push(String::new());
    lines.extend([
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
    append_input(&mut lines, "runner", runner, "runner.sh", None)?;
    append_input(&mut lines, "manifest", manifest, "manifest.json", None)?;
    append_input(&mut lines, "train", &train, "train.csv", Some(train_sha))?;
    append_input(&mut lines, "test", &test, "test.csv", Some(test_sha))?;
    append_input(
        &mut lines,
        "evaluator",
        &evaluator.path,
        "corpus_eval",
        Some(&evaluator.sha256),
    )?;
    lines.extend([
        "[[output]]".to_owned(),
        "name = \"receipt\"".to_owned(),
        "source = \"receipt\"".to_owned(),
        "directory = true".to_owned(),
        String::new(),
    ]);
    write_immutable(&path, lines.join("\n").as_bytes())?;
    Ok((job_id, path))
}
