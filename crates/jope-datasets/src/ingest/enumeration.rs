use super::*;

fn is_dir(p: &Path) -> bool {
    p.is_dir()
}

fn sorted_names(dir: &Path) -> Result<Vec<String>> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .with_context(|| format!("read_dir {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().into_owned()))
        .collect();
    names.sort();
    Ok(names)
}

/// Quarantined copies below `dir` (`<dir>/_dupes_quarantine/<id>`) are listed, never ingested.
fn quarantine(dir: &Path, prefix: &str, en: &mut Enumeration) -> Result<()> {
    let q = dir.join(QUARANTINE);
    if !is_dir(&q) {
        return Ok(());
    }
    for name in sorted_names(&q)? {
        if crate::dataset_dir_hash(&name).is_some() && is_dir(&q.join(&name)) {
            en.duplicates.push(Duplicate {
                path: format!("{prefix}{QUARANTINE}/{name}"),
                rule: "quarantine".into(),
                canonical: None,
                dataset_hash: crate::dataset_dir_hash(&name).map(str::to_string),
            });
        }
    }
    Ok(())
}

fn candidate_or_reject(
    dir: PathBuf,
    relpath: String,
    name: String,
    task: Option<Task>,
    priority: u8,
    en: &mut Enumeration,
) {
    let missing: Vec<&str> = FILES
        .iter()
        .copied()
        .filter(|f| !dir.join(f).is_file())
        .collect();
    if missing.is_empty() {
        en.candidates.push(Candidate {
            relpath,
            dir,
            name,
            task,
            expected: None,
            priority,
        });
    } else {
        en.rejects.push(RejectRecord {
            path: relpath,
            reason: "incomplete".into(),
            detail: format!("missing {}", missing.join(",")),
        });
    }
}

/// `remote_super/quant`: class directories `regression`, `binary`, `multiclass`, `multi_label`,
/// `timeseries`, `synthetic/binary`. Root-level `<hash>` / `synth_<hash>` directories that also
/// exist under `regression/` and `binary/<id>` directories that also exist under
/// `synthetic/binary/` are duplicates by path rule and are not hashed.
pub fn enumerate_quant(root: &Path) -> Result<Enumeration> {
    let mut en = Enumeration::default();
    let classes: [(&str, Task, u8); 6] = [
        ("regression", Task::Regression, 1),
        ("binary", Task::Binary, 2),
        ("multiclass", Task::Multiclass, 3),
        ("multi_label", Task::MultiLabel, 4),
        ("timeseries", Task::Timeseries, 5),
        ("synthetic/binary", Task::Binary, 6),
    ];
    for (class, task, priority) in classes {
        let dir = root.join(class);
        if !is_dir(&dir) {
            continue;
        }
        quarantine(&dir, &format!("{class}/"), &mut en)?;
        for name in sorted_names(&dir)? {
            let Some(hash) = crate::dataset_dir_hash(&name) else {
                continue;
            };
            let sub = dir.join(&name);
            if !is_dir(&sub) {
                continue;
            }
            if class == "binary" && is_dir(&root.join("synthetic/binary").join(&name)) {
                en.duplicates.push(Duplicate {
                    path: format!("{class}/{name}"),
                    rule: "binary-synthetic-duplicate".into(),
                    canonical: Some(format!("synthetic/binary/{name}")),
                    dataset_hash: Some(hash.to_string()),
                });
                continue;
            }
            candidate_or_reject(
                sub,
                format!("{class}/{name}"),
                name.clone(),
                Some(task),
                priority,
                &mut en,
            );
        }
    }
    quarantine(root, "", &mut en)?;
    for name in sorted_names(root)? {
        let Some(hash) = crate::dataset_dir_hash(&name) else {
            continue;
        };
        let sub = root.join(&name);
        if !is_dir(&sub) {
            continue;
        }
        if is_dir(&root.join("regression").join(&name)) {
            en.duplicates.push(Duplicate {
                path: name.clone(),
                rule: "root-level-duplicate".into(),
                canonical: Some(format!("regression/{name}")),
                dataset_hash: Some(hash.to_string()),
            });
            continue;
        }
        candidate_or_reject(sub, name.clone(), name.clone(), None, 9, &mut en);
    }
    Ok(en)
}

/// The node-archive mirror: `index/dataset.json` maps `home/ubuntu/quant/<class>/<id>/<file>` to
/// its sha256; the bytes live under `dataset/<path>`. A computed sha256 that differs from the
/// index is a reject (`mirror_sha_mismatch`).
pub fn enumerate_mirror(root: &Path) -> Result<Enumeration> {
    let index_path = root.join("index").join("dataset.json");
    let text = std::fs::read_to_string(&index_path)
        .with_context(|| format!("read {}", index_path.display()))?;
    let index: BTreeMap<String, String> =
        serde_json::from_str(&text).context("parse index/dataset.json")?;
    let mut groups: BTreeMap<String, BTreeMap<String, String>> = BTreeMap::new();
    for (path, sha) in index {
        if let Some((dir, file)) = path.rsplit_once('/') {
            groups
                .entry(dir.to_string())
                .or_default()
                .insert(file.to_string(), sha);
        }
    }
    let mut en = Enumeration::default();
    for (dir, files) in groups {
        let name = dir.rsplit('/').next().unwrap_or_default().to_string();
        if crate::dataset_dir_hash(&name).is_none() {
            continue;
        }
        let rel = dir
            .split_once("quant/")
            .map_or(dir.clone(), |(_, r)| r.to_string());
        if rel.split('/').any(|c| c == QUARANTINE) {
            en.duplicates.push(Duplicate {
                path: rel,
                rule: "quarantine".into(),
                canonical: None,
                dataset_hash: crate::dataset_dir_hash(&name).map(str::to_string),
            });
            continue;
        }
        let class = rel.rsplit_once('/').map_or("", |(c, _)| c);
        let task = match class {
            "regression" => Some(Task::Regression),
            "binary" | "synthetic/binary" => Some(Task::Binary),
            "multiclass" => Some(Task::Multiclass),
            "multi_label" => Some(Task::MultiLabel),
            "timeseries" => Some(Task::Timeseries),
            _ => None,
        };
        let missing: Vec<&str> = FILES
            .iter()
            .copied()
            .filter(|f| !files.contains_key(*f))
            .collect();
        if !missing.is_empty() {
            en.rejects.push(RejectRecord {
                path: rel,
                reason: "incomplete".into(),
                detail: format!("index lacks {}", missing.join(",")),
            });
            continue;
        }
        let expected = [
            files["train.csv"].clone(),
            files["test.csv"].clone(),
            files["meta.json"].clone(),
        ];
        en.candidates.push(Candidate {
            relpath: rel,
            dir: root.join("dataset").join(&dir),
            name,
            task,
            expected: Some(expected),
            priority: 0,
        });
    }
    Ok(en)
}

pub fn enumerate(source: Source, root: &Path) -> Result<Enumeration> {
    match source {
        Source::Quant => enumerate_quant(root),
        Source::Mirror => enumerate_mirror(root),
    }
}

/// Hash every candidate's three files on `jobs` threads; results are indexed like `candidates`.
pub fn hash_all(candidates: &[Candidate], jobs: usize) -> Vec<std::io::Result<Hashed>> {
    let next = AtomicUsize::new(0);
    let results: Vec<Mutex<Option<std::io::Result<Hashed>>>> =
        (0..candidates.len()).map(|_| Mutex::new(None)).collect();
    let started = Instant::now();
    std::thread::scope(|s| {
        for _ in 0..jobs.max(1) {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= candidates.len() {
                    break;
                }
                let c = &candidates[i];
                let r = (|| -> std::io::Result<Hashed> {
                    let d = |f: &str| digest_file(&c.dir.join(f));
                    Ok(Hashed {
                        train: d("train.csv")?,
                        test: d("test.csv")?,
                        meta: d("meta.json")?,
                    })
                })();
                *results[i]
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(r);
                if (i + 1).is_multiple_of(5000) {
                    eprintln!(
                        "hashed {}/{} ({:.0}s)",
                        i + 1,
                        candidates.len(),
                        started.elapsed().as_secs_f64()
                    );
                }
            });
        }
    });
    results
        .into_iter()
        .map(|m| {
            m.into_inner()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .unwrap_or_else(|| Err(std::io::Error::other("not hashed")))
        })
        .collect()
}
