use super::*;

// --------------------------------------------------------------------------
// report
// --------------------------------------------------------------------------

#[derive(Deserialize)]
struct ReceiptRow {
    task: String,
    #[serde(rename = "dataset_hash")]
    _dataset_hash: String,
    semantic_sha256: String,
    metric: String,
    score: Option<f64>,
    valid: bool,
    #[serde(default)]
    failure: Option<String>,
    runtime_seconds: f64,
}

/// Locate a job's receipt directory.
///
/// MEASURED: the cluster has NO shared filesystem (`ssh xbabe3 ls
/// $R/training/packed` -> "No such file or directory"), so a job that ran on a
/// remote node leaves `runs/<job>/attempts/N/outputs/receipt` on THAT node and
/// only `.portable-manifest.json` locally. Reading the local run dir therefore
/// silently reports "pending" for every remote job -- which is most of them.
/// `scq fetch` is the only correct retrieval path. The local dir is still tried
/// first because jobs that ran on this host are already complete there.
/// A receipt dir holding only dotfiles (`.portable-manifest.json`, `.stderr.log`)
/// is a stub: `scq fetch` against a job that has not yet written its outputs
/// still creates the directory. Caching such a stub is the whole bug this guards
/// against -- a stub cached while the job was still running is otherwise returned
/// forever, so the job stays "pending" in every later report even after it
/// completes and its receipts become fetchable.
fn substantive(dir: &Path) -> bool {
    fs::read_dir(dir)
        .map(|mut d| {
            d.any(|e| {
                e.map(|e| !e.file_name().to_string_lossy().starts_with('.'))
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false)
}

fn receipt_dir(runs: &Path, scq: &Path, cache: &Path, job_id: &str) -> Option<PathBuf> {
    let attempts = runs.join(job_id).join("attempts");
    let mut best: Option<(u64, PathBuf)> = None;
    if let Ok(entries) = fs::read_dir(&attempts) {
        for entry in entries.flatten() {
            let Ok(n) = entry.file_name().to_string_lossy().parse::<u64>() else {
                continue; // skip non-numeric attempt dirs rather than abort
            };
            let dir = entry.path().join("outputs").join("receipt");
            if substantive(&dir) && best.as_ref().is_none_or(|(m, _)| n > *m) {
                best = Some((n, dir));
            }
        }
    }
    if let Some((_, path)) = best {
        return Some(path);
    }
    let target = cache.join(job_id);
    let fetched = target.join("receipt");
    if substantive(&fetched) {
        return Some(fetched);
    }
    // Either nothing cached, or a stub from a fetch issued while the job was
    // still running. Drop it and re-fetch; a still-running job simply produces
    // another stub, which we discard again instead of poisoning the cache.
    let _ = fs::remove_dir_all(&target);
    fs::create_dir_all(&target).ok()?;
    let status = std::process::Command::new(scq)
        .arg("fetch")
        .arg(job_id)
        .arg("--output")
        .arg(&target)
        .output()
        .ok()?;
    if !status.status.success() {
        let _ = fs::remove_dir_all(&target);
        return None;
    }
    if !substantive(&fetched) {
        let _ = fs::remove_dir_all(&target);
        return None;
    }
    Some(fetched)
}

pub(crate) fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

pub(crate) fn report(plan_dir: &Path, runs: &Path, scq: &Path, out: &Path) -> Result<Value> {
    let cache = plan_dir.join("fetched");
    fs::create_dir_all(&cache)?;
    let doc: Value = serde_json::from_str(&fs::read_to_string(plan_dir.join("portfolio.json"))?)?;
    let jobs = doc["jobs"].as_array().context("portfolio.jobs")?;

    // semantic -> (arm, checkpoint, coverage)
    let mut meta: HashMap<String, (String, String, u64)> = HashMap::new();
    for task in TASKS {
        for g in doc["portfolio"][task].as_array().into_iter().flatten() {
            meta.insert(
                g["semantic_sha256"].as_str().unwrap_or_default().to_owned(),
                (
                    g["arm"].as_str().unwrap_or_default().to_owned(),
                    g["model_checkpoint_sha256"]
                        .as_str()
                        .unwrap_or_default()
                        .to_owned(),
                    g["champion_datasets"].as_u64().unwrap_or(0),
                ),
            );
        }
    }

    // per dataset: arm/checkpoint -> best score
    let mut done = 0_usize;
    let mut pending = 0_usize;
    let mut per_dataset: Vec<Value> = Vec::new();
    let mut lifts: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut wins: BTreeMap<String, (u64, u64)> = BTreeMap::new();
    let mut invalid_rows = 0_u64;
    let mut total_rows = 0_u64;
    let mut runtime_total = 0.0_f64;

    for job in jobs {
        let job_id = job["job_id"].as_str().unwrap_or_default();
        let Some(dir) = receipt_dir(runs, scq, &cache, job_id) else {
            pending += 1;
            continue;
        };
        let hash = job["dataset_hash"].as_str().unwrap_or_default();
        let path = dir.join(format!("{hash}.receipts.jsonl"));
        let Ok(text) = fs::read_to_string(&path) else {
            pending += 1;
            continue;
        };
        let mut best_by: BTreeMap<String, f64> = BTreeMap::new();
        let mut metric = String::new();
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let row: ReceiptRow = match serde_json::from_str(line) {
                Ok(r) => r,
                Err(_) => continue,
            };
            total_rows += 1;
            runtime_total += row.runtime_seconds;
            metric = row.metric.clone();
            if !row.valid || row.score.is_none() {
                invalid_rows += 1;
                continue;
            }
            let score = row.score.unwrap();
            let (arm, checkpoint, _) = meta
                .get(&row.semantic_sha256)
                .cloned()
                .unwrap_or_else(|| ("unknown".into(), "unknown".into(), 0));
            let bucket = if arm == "hof" {
                format!("hof:{checkpoint}")
            } else {
                arm.clone()
            };
            for key in ["ALL_HOF".to_owned(), bucket] {
                if arm == "hof" || key != "ALL_HOF" {
                    best_by
                        .entry(key)
                        .and_modify(|v| {
                            if score > *v {
                                *v = score
                            }
                        })
                        .or_insert(score);
                }
            }
            let _ = &row.task;
            let _ = &row.failure;
        }
        done += 1;
        let base = ["base_linear", "base_linear_std"]
            .iter()
            .filter_map(|k| best_by.get(*k).copied())
            .fold(f64::NEG_INFINITY, f64::max);
        if base.is_finite() {
            for (key, value) in &best_by {
                if key.starts_with("base_") {
                    continue;
                }
                lifts.entry(key.clone()).or_default().push(value - base);
                let entry = wins.entry(key.clone()).or_insert((0, 0));
                entry.1 += 1;
                if *value > base {
                    entry.0 += 1;
                }
            }
        }
        per_dataset.push(json!({
            "job_id": job_id,
            "task": job["task"],
            "dataset_id": job["dataset_id"],
            "dataset_hash": hash,
            "source_family": job["source_family"],
            "metric": metric,
            "best_by_bucket": best_by,
            "baseline_best": if base.is_finite() { json!(base) } else { Value::Null },
        }));
    }

    let mut summary = BTreeMap::new();
    for (key, values) in &lifts {
        let mut v = values.clone();
        let n = v.len() as f64;
        let mean = v.iter().sum::<f64>() / n;
        let med = median(&mut v);
        let (w, t) = wins[key];
        summary.insert(
            key.clone(),
            json!({
                "n_datasets": values.len(),
                "mean_lift_over_baseline": mean,
                "median_lift_over_baseline": med,
                "win_rate_vs_baseline": w as f64 / t.max(1) as f64,
                "wins": w, "compared": t,
            }),
        );
    }

    let result = json!({
        "schema": REPORT_SCHEMA,
        "cohort_sha256": doc["cohort_sha256"],
        "jobs_planned": jobs.len(),
        "jobs_with_receipts": done,
        "jobs_pending": pending,
        "receipt_rows": total_rows,
        "invalid_rows": invalid_rows,
        "evaluator_seconds_total": runtime_total,
        "summary": summary,
        "per_dataset": per_dataset,
    });
    atomic_write(out, &serde_json::to_vec_pretty(&result)?)?;
    let mut brief = result.clone();
    brief.as_object_mut().unwrap().remove("per_dataset");
    Ok(brief)
}
