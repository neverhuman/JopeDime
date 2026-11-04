use super::*;

// --------------------------------------------------------------------------
// plan
// --------------------------------------------------------------------------

pub(crate) fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, bytes)?;
    fs::rename(&tmp, path)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn plan(
    root: &Path,
    out: &Path,
    top_k: usize,
    limit_per_task: usize,
    evaluator: &Path,
    runner: &Path,
    exclude_contaminated: bool,
) -> Result<Value> {
    if out.starts_with(root) {
        bail!(
            "refusing to write plan inside the live root {} (safety contract item 4)",
            root.display()
        );
    }
    let cohort = load_sealed(root)?;
    let dirty = contaminated(root, &cohort)?;
    let portfolio = build_portfolio(root, top_k)?;

    let manifests = out.join("manifests");
    let specs = out.join("specs");
    fs::create_dir_all(&manifests)?;
    fs::create_dir_all(&specs)?;

    let evaluator_sha = sha256_hex(&fs::read(evaluator)?);
    let runner_sha = sha256_hex(&fs::read(runner)?);

    let mut planned = 0_usize;
    let mut skipped_contaminated = 0_usize;
    let mut per_task_planned: BTreeMap<String, usize> = BTreeMap::new();
    let mut spec_list = Vec::new();

    for task in TASKS {
        let mut taken = 0_usize;
        // Deterministic order: dataset_id ASC.
        let mut entries: Vec<&SealedEntry> =
            cohort.entries.iter().filter(|e| e.task == task).collect();
        entries.sort_by(|a, b| a.dataset_id.cmp(&b.dataset_id));
        for entry in entries {
            let key = format!("{}:{}", entry.task, entry.dataset_hash);
            if exclude_contaminated && dirty.contains_key(&key) {
                skipped_contaminated += 1;
                continue;
            }
            if limit_per_task > 0 && taken >= limit_per_task {
                break;
            }
            let candidates: Vec<Value> = portfolio[task]
                .iter()
                .map(|g| {
                    json!({
                        "semantic_sha256": g.semantic_sha256,
                        "canonical_genome": g.canonical_genome,
                    })
                })
                .collect();
            // label_authority is deliberately OMITTED. corpus_eval then defaults
            // to benchmark_valid=false / generation="adaptive"
            // (corpus_eval.rs:339-352), which is the most inert receipt shape
            // available: even if one of these receipts were somehow surfaced to
            // an ingester, it carries no self-play authority.
            let shard = json!({
                "schema": SHARD_SCHEMA,
                "task": entry.task,
                "dataset_hash": entry.dataset_hash,
                "train_csv": entry.source.join("train.csv").to_string_lossy(),
                "test_csv": entry.source.join("test.csv").to_string_lossy(),
                "train_sha256": entry.train_sha256,
                "test_sha256": entry.test_sha256,
                "candidates": candidates,
            });
            let manifest_path = manifests.join(format!("{}.json", entry.dataset_id));
            let bytes = serde_json::to_vec_pretty(&shard)?;
            atomic_write(&manifest_path, &bytes)?;
            let manifest_sha = sha256_hex(&bytes);

            let job_id = format!(
                "{JOB_PREFIX}-{}-{}",
                if task == "binary" { "b" } else { "r" },
                entry.dataset_id.replace("synth_", "s")
            );
            let resource_block = EVAL.render_resource_block();
            let spec = format!(
                r#"[job]
id = "{job_id}"
project = "{PROJECT}"
tier = "normal"
command = "/usr/bin/env"
args = ["{{input:runner}}", "{{input:manifest}}", "{{input:train}}", "{{input:test}}", "{{input:evaluator}}", "{{output:receipt}}"]

{resource_block}

[env]
RAYON_NUM_THREADS = "1"
OMP_NUM_THREADS = "1"
OPENBLAS_NUM_THREADS = "1"
MKL_NUM_THREADS = "1"
NUMEXPR_NUM_THREADS = "1"

[preempt]
preemptible = true
checkpoint = "none"
resume = "rerun"

[limits]
max_runtime_hours = 6
max_restarts = 1

[[input]]
name = "runner"
source = "{runner}"
target = "runner.sh"
sha256 = "{runner_sha}"

[[input]]
name = "manifest"
source = "{manifest}"
target = "manifest.json"
sha256 = "{manifest_sha}"

[[input]]
name = "train"
source = "{train}"
target = "train.csv"
sha256 = "{train_sha}"

[[input]]
name = "test"
source = "{test}"
target = "test.csv"
sha256 = "{test_sha}"

[[input]]
name = "evaluator"
source = "{evaluator}"
target = "corpus_eval"
sha256 = "{evaluator_sha}"

[[output]]
name = "receipt"
source = "receipt"
directory = true
"#,
                runner = runner.display(),
                manifest = manifest_path.display(),
                train = entry.source.join("train.csv").display(),
                train_sha = entry.train_sha256,
                test = entry.source.join("test.csv").display(),
                test_sha = entry.test_sha256,
                evaluator = evaluator.display(),
            );
            let spec_path = specs.join(format!("{job_id}.toml"));
            atomic_write(&spec_path, spec.as_bytes())?;
            spec_list.push(json!({
                "job_id": job_id,
                "task": entry.task,
                "dataset_id": entry.dataset_id,
                "dataset_hash": entry.dataset_hash,
                "source_family": entry.source_family,
                "spec": spec_path.to_string_lossy(),
                "manifest": manifest_path.to_string_lossy(),
                "mirror": entry.mirror.to_string_lossy(),
            }));
            planned += 1;
            taken += 1;
            *per_task_planned.entry(task.to_owned()).or_insert(0) += 1;
        }
    }

    let portfolio_doc = json!({
        "schema": PORTFOLIO_SCHEMA,
        "cohort_sha256": cohort.cohort_sha256,
        "top_k": top_k,
        "portfolio": portfolio,
        "contaminated": dirty.iter().map(|(k, (n, best))| {
            json!({"key": k, "observations": n, "best_score": best})
        }).collect::<Vec<_>>(),
        "jobs": spec_list,
        "evaluator_sha256": evaluator_sha,
        "runner_sha256": runner_sha,
    });
    atomic_write(
        &out.join("portfolio.json"),
        &serde_json::to_vec_pretty(&portfolio_doc)?,
    )?;

    Ok(json!({
        "cohort_sha256": cohort.cohort_sha256,
        "planned_jobs": planned,
        "per_task": per_task_planned,
        "skipped_contaminated": skipped_contaminated,
        "contaminated_total": dirty.len(),
        "candidates_per_manifest": {
            "regression": portfolio["regression"].len(),
            "binary": portfolio["binary"].len(),
        },
        "out": out.to_string_lossy(),
    }))
}
