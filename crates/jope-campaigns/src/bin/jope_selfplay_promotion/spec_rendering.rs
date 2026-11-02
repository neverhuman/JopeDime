use super::*;

// ---------------------------------------------------------------------------
// Spec rendering (ports of test_exact_ops / test_exact_train renderers)
// ---------------------------------------------------------------------------

pub(super) fn toml_string(value: &str) -> Result<String> {
    py_dumps(&Value::String(value.to_owned()), PyStyle::Compact)
}

pub(super) fn input_block(name: &str, source: &Path, target: &str) -> Result<Vec<String>> {
    Ok(vec![
        "[[input]]".to_owned(),
        format!("name = {}", toml_string(name)?),
        format!("source = {}", toml_string(&source.to_string_lossy())?),
        format!("target = {}", toml_string(target)?),
        format!("sha256 = {}", toml_string(&sha256_file(source)?)?),
        String::new(),
    ])
}

pub(super) struct ProposalSpec {
    job_id: String,
    spec: PathBuf,
    checkpoint_sha256: String,
    split: Option<String>,
    cohort_sha256: Option<String>,
    seed: i64,
    proposals_per_dataset: i64,
}

impl ProposalSpec {
    pub(super) fn to_value(&self) -> Value {
        json!({
            "job_id": self.job_id,
            "spec": self.spec.to_string_lossy(),
            "checkpoint_sha256": self.checkpoint_sha256,
            "split": self.split,
            "cohort_sha256": self.cohort_sha256,
            "seed": self.seed,
            "proposals_per_dataset": self.proposals_per_dataset,
        })
    }
}

/// Port of `test_exact_ops.render_checkpoint_proposal_spec` (including this
/// session's widened `xbabe2|xbabe3` hostname dependency).
#[allow(clippy::too_many_arguments)]
pub(super) fn render_checkpoint_proposal_spec(
    path: &Path,
    checkpoint: &Path,
    split: Option<&str>,
    cohort: Option<&Path>,
    seed: i64,
    proposals_per_dataset: i64,
    root: &Path,
    attempt: u32,
) -> Result<ProposalSpec> {
    let pointer_path = root.join("catalog").join("current.json");
    let pointer = read_json(&pointer_path)?;
    let catalog = PathBuf::from(get_str(&pointer, "catalog")?);
    let data = PathBuf::from(get_str(&pointer, "data")?);
    let script = Path::new(scripts_str()).join("test_exact_dl_generate.py");
    let common = Path::new(scripts_str()).join("test_exact_common.py");
    let checkpoint_hash = sha256_file(checkpoint)?;
    let cohort_hash = match cohort {
        Some(path) => Some(sha256_file(path)?),
        None => None,
    };
    let split_name = split.unwrap_or("all");
    let cohort_or_split = cohort_hash
        .as_deref()
        .map(|hash| hash[..12].to_owned())
        .unwrap_or_else(|| split_name.to_owned());
    let mut job_id = format!(
        "jope-selfplay-propose-{}-{}-{}-cpu4-p2",
        &checkpoint_hash[..12],
        cohort_or_split,
        seed
    );
    if attempt > 1 {
        // A terminal-failed record blocks resubmission under the same id and the
        // scheduler offers no record removal, so retries get a fresh attempt id.
        job_id.push_str(&format!("-r{attempt}"));
    }
    let output = "dl-proposals.jsonl";
    // Reconstruct the exact Python-rendered args string.
    let mut args_line = format!(
        "args = [\"{{input:script}}\", \"--catalog-pointer\", \"{{input:pointer}}\", \
         \"--catalog\", \"{{input:catalog}}\", \"--data\", \"{{input:data}}\", \
         \"--checkpoint\", \"{{input:checkpoint}}\", \
         \"--output\", \"{{output:proposals}}/{output}\", \"--device\", \"cuda\", \
         \"--selection-policy\", \"{{input:selection_policy}}\", \
         \"--seed\", \"{seed}\", \"--proposals-per-dataset\", \
         \"{proposals_per_dataset}\", \"--batch-contexts\", \"64\", \
         \"--allow-deficient-floor\", \"8\""
    );
    if let Some(split) = split {
        args_line.push_str(&format!(", \"--split\", \"{split}\""));
    }
    if cohort.is_some() {
        args_line.push_str(", \"--cohort\", \"{input:cohort}\"");
    }
    args_line.push(']');
    let mut lines = vec![
        "[job]".to_owned(),
        format!("id = {}", toml_string(&job_id)?),
        "project = \"jope-selfplay-train\"".to_owned(),
        "tier = \"normal\"".to_owned(),
        "command = \"/usr/bin/python3\"".to_owned(),
        args_line,
        String::new(),
    ];
    lines.extend(PROPOSE.render_resource_block().lines().map(str::to_owned));
    lines.push(String::new());
    lines.extend(PROPOSE.render_placement_block().lines().map(str::to_owned));
    lines.push(String::new());
    lines.extend([
        "[env]".to_owned(),
        format!(
            "PYTHONPATH = \"{prime_src_str}:{scripts_str}\"",
            scripts_str = scripts_str(),
            prime_src_str = prime_src_str()
        ),
        format!("OMP_NUM_THREADS = \"{}\"", PROPOSE.omp_threads),
        String::new(),
        "[preempt]".to_owned(),
        format!("preemptible = {}", PROPOSE.preemptible),
        "checkpoint = \"none\"".to_owned(),
        "resume = \"rerun\"".to_owned(),
        String::new(),
    ]);
    lines.extend(PROPOSE.render_limits_block().lines().map(str::to_owned));
    lines.push(String::new());
    let catalog_name = catalog
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let data_name = data
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let checkpoint_name = checkpoint
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut inputs: Vec<(String, PathBuf, String)> = vec![
        (
            "script".into(),
            script,
            "scripts/test_exact_dl_generate.py".into(),
        ),
        (
            "common".into(),
            common,
            "scripts/test_exact_common.py".into(),
        ),
        (
            "selection_policy".into(),
            PathBuf::from(proposal_selection_policy_str()),
            "scripts/proposal-selection-policy.v1.json".into(),
        ),
        (
            "pointer".into(),
            pointer_path,
            "catalog/current.json".into(),
        ),
        ("catalog".into(), catalog, format!("catalog/{catalog_name}")),
        ("data".into(), data, format!("catalog/{data_name}")),
        (
            "checkpoint".into(),
            checkpoint.to_path_buf(),
            format!("checkpoint/{checkpoint_name}"),
        ),
    ];
    if let Some(cohort) = cohort {
        let cohort_name = cohort
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        inputs.push((
            "cohort".into(),
            cohort.to_path_buf(),
            format!("cohort/{cohort_name}"),
        ));
    }
    for (name, source, target) in &inputs {
        lines.extend(input_block(name, source, target)?);
    }
    lines.extend([
        "[[output]]".to_owned(),
        "name = \"proposals\"".to_owned(),
        "source = \"proposals\"".to_owned(),
        "directory = true".to_owned(),
        String::new(),
    ]);
    fs::create_dir_all(
        path.parent()
            .ok_or_else(|| anyhow!("spec path has no parent"))?,
    )?;
    fs::write(path, lines.join("\n"))?;
    Ok(ProposalSpec {
        job_id,
        spec: path.to_path_buf(),
        checkpoint_sha256: checkpoint_hash,
        split: split.map(str::to_owned),
        cohort_sha256: cohort_hash,
        seed,
        proposals_per_dataset,
    })
}

/// Port of `test_exact_train.render_spec` (trainer tournament spec).
pub(super) fn render_trainer_spec(
    corpus_manifest: &Path,
    recipe: &str,
    seed: i64,
    packed_manifest: &Path,
    root: &Path,
) -> Result<Value> {
    let manifest = read_json(corpus_manifest)?;
    let watermark = get_i64(&manifest, "source_label_watermark")?;
    if !matches!(recipe, "a" | "b" | "c") {
        bail!("checkpoint recipe must be a, b, or c");
    }
    let job_id = format!("jope-selfplay-train-{watermark:09}-{recipe}");
    let spec = root.join("specs").join(format!("{job_id}.toml"));
    let script = Path::new(scripts_str()).join("test_exact_train.py");
    let common = Path::new(scripts_str()).join("test_exact_common.py");
    let train_source = Path::new(prime_src_str()).join("jope/policy/train_v3.py");
    let output_name = format!("watermark-{watermark:09}");
    let (warm, warm_hash) = accepted_checkpoint()?;
    let mut inputs: Vec<(String, PathBuf, String)> = vec![
        (
            "script".into(),
            script.clone(),
            format!("scripts/{}", script.file_name().unwrap().to_string_lossy()),
        ),
        (
            "common".into(),
            common.clone(),
            format!("scripts/{}", common.file_name().unwrap().to_string_lossy()),
        ),
        (
            "packed_source".into(),
            Path::new(scripts_str()).join("test_exact_packed_corpus.py"),
            "scripts/test_exact_packed_corpus.py".into(),
        ),
        (
            "manifest".into(),
            corpus_manifest.to_path_buf(),
            "corpus/corpus-manifest.json".into(),
        ),
        (
            "train".into(),
            PathBuf::from(get_str(&manifest, "train")?),
            "corpus/corpus.train.jsonl".into(),
        ),
        (
            "validation".into(),
            PathBuf::from(get_str(&manifest, "validation")?),
            "corpus/corpus.val.jsonl".into(),
        ),
        (
            "checkpoint".into(),
            warm.clone(),
            format!("checkpoint/{}", warm.file_name().unwrap().to_string_lossy()),
        ),
        (
            "trainer_source".into(),
            train_source,
            "provenance/train_v3.py".into(),
        ),
    ];
    let packed = read_json(packed_manifest)?;
    let pack_root = packed_manifest
        .parent()
        .ok_or_else(|| anyhow!("packed manifest has no parent"))?;
    inputs.push((
        "packed_manifest".into(),
        packed_manifest.to_path_buf(),
        "packed/manifest.json".into(),
    ));
    inputs.push((
        "packed_catalog".into(),
        PathBuf::from(get_str(&packed, "catalog_data")?),
        "packed/catalog-data.f32le".into(),
    ));
    let mut packed_names: Vec<String> = match packed.get("files") {
        Some(Value::Object(map)) => map.keys().cloned().collect(),
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect(),
        _ => bail!("packed manifest files listing is absent"),
    };
    packed_names.sort();
    for (index, name) in packed_names.iter().enumerate() {
        inputs.push((
            format!("packed_{index}"),
            pack_root.join(name),
            format!("packed/{name}"),
        ));
    }
    let packed_arg = ", \"--packed-manifest\", \"{input:packed_manifest}\", \
         \"--packed-source-train\", \"{input:train}\", \
         \"--packed-source-validation\", \"{input:validation}\", \
         \"--packed-catalog-data\", \"{input:packed_catalog}\"";
    let args_line = format!(
        "args = [\"{{input:script}}\", \"train\", \"--corpus-manifest\", \"{{input:manifest}}\", \
         \"--output\", \"{{output:training}}/{output_name}\", \"--device\", \"cuda\", \
         \"--warm\", \"{{input:checkpoint}}\", \"--recipe\", \"{recipe}\", \"--seed\", \"{seed}\"{packed_arg}]"
    );
    let mut lines = vec![
        "[job]".to_owned(),
        format!("id = \"{job_id}\""),
        "project = \"jope-selfplay-train\"".to_owned(),
        "tier = \"normal\"".to_owned(),
        "command = \"/usr/bin/python3\"".to_owned(),
        args_line,
        String::new(),
    ];
    lines.extend(TRAINER.render_resource_block().lines().map(str::to_owned));
    lines.push(String::new());
    lines.extend(TRAINER.render_placement_block().lines().map(str::to_owned));
    lines.push(String::new());
    lines.extend([
        "[env]".to_owned(),
        format!(
            "PYTHONPATH = \"{prime_src_str}:{scripts_str}\"",
            scripts_str = scripts_str(),
            prime_src_str = prime_src_str()
        ),
        format!("OMP_NUM_THREADS = \"{}\"", TRAINER.omp_threads),
        String::new(),
        "[preempt]".to_owned(),
        format!("preemptible = {}", TRAINER.preemptible),
        "checkpoint = \"none\"".to_owned(),
        "resume = \"rerun\"".to_owned(),
        String::new(),
    ]);
    lines.extend(TRAINER.render_limits_block().lines().map(str::to_owned));
    lines.push(String::new());
    for (name, source, target) in &inputs {
        lines.extend([
            "[[input]]".to_owned(),
            format!("name = \"{name}\""),
            format!("source = {}", toml_string(&source.to_string_lossy())?),
            format!("target = {}", toml_string(target)?),
            format!("sha256 = \"{}\"", sha256_file(source)?),
            String::new(),
        ]);
    }
    lines.extend([
        "[[output]]".to_owned(),
        "name = \"training\"".to_owned(),
        "source = \"training\"".to_owned(),
        "directory = true".to_owned(),
        String::new(),
    ]);
    fs::create_dir_all(spec.parent().unwrap())?;
    fs::write(&spec, lines.join("\n"))?;
    Ok(json!({
        "job_id": job_id,
        "spec": spec.to_string_lossy(),
        "watermark": watermark,
        "recipe": recipe,
        "seed": seed,
        "accepted_ancestor_sha256": warm_hash,
    }))
}

/// Port of `test_exact_train.accepted_checkpoint` (learning → accepted →
/// warm-start bootstrap, always against the production root).
pub(super) fn accepted_checkpoint() -> Result<(PathBuf, String)> {
    for pointer in [
        Path::new(root_str()).join("training/learning.json"),
        Path::new(root_str()).join("training/accepted.json"),
    ] {
        if !pointer.is_file() {
            continue;
        }
        let value = read_json(&pointer)?;
        let path = PathBuf::from(get_str(&value, "checkpoint")?);
        let digest = get_str(&value, "checkpoint_sha256")?.to_owned();
        if path.is_file() && sha256_file(&path)? == digest {
            return Ok((path, digest));
        }
        bail!("accepted checkpoint pointer failed content verification");
    }
    Ok((PathBuf::from(warm_str()), WARM_SHA.to_owned()))
}
