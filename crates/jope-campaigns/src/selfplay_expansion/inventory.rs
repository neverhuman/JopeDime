use super::model::{
    corpus_dir_name, load_v1_quarantine, v1_identity_denied, AuditInventory, ExpansionPaths,
    InventoryEntry, MirrorConflict, TASKS,
};
use super::prelude::*;

pub(crate) fn valid_dataset_id(value: &str) -> bool {
    let hash = value.strip_prefix("synth_").unwrap_or(value);
    hash.len() == 16
        && hash
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

pub(crate) fn read_meta(path: &Path) -> Result<Value> {
    read_json(path).with_context(|| format!("read dataset metadata {}", path.display()))
}

pub(crate) fn mirror_dataset(
    paths: &ExpansionPaths,
    task: &str,
    dataset_id: &str,
) -> Option<PathBuf> {
    [
        paths
            .mirror_root
            .join(corpus_dir_name(task))
            .join(dataset_id),
        paths.mirror_root.join(dataset_id),
    ]
    .into_iter()
    .find(|candidate| candidate.is_dir())
}

pub(crate) fn write_conflict_receipt(
    paths: &ExpansionPaths,
    task: &str,
    dataset_id: &str,
    source: &Path,
    mirror: &Path,
    disagreements: &BTreeMap<String, Value>,
) -> Result<(String, PathBuf)> {
    let stable = json!({
        "schema": "JopeExpandedMirrorConflict.v1",
        "task": task,
        "dataset_id": dataset_id,
        "source": source,
        "mirror": mirror,
        "disagreements": disagreements,
    });
    let digest = canonical_json_sha256(&stable)?;
    let mut payload = stable;
    payload["conflict_sha256"] = Value::String(digest.clone());
    payload["first_observed_at_ms"] = json!(now_ms());
    let destination = paths.receipt_root.join(format!("{digest}.json"));
    if destination.is_file() {
        let existing = read_json(&destination)?;
        ensure!(
            existing["conflict_sha256"].as_str() == Some(&digest)
                && existing["task"] == payload["task"]
                && existing["dataset_id"] == payload["dataset_id"]
                && existing["source"] == payload["source"]
                && existing["mirror"] == payload["mirror"]
                && existing["disagreements"] == payload["disagreements"],
            "mirror conflict receipt identity drift: {}",
            destination.display()
        );
    } else {
        atomic_json(&destination, &payload)?;
    }
    Ok((digest, destination))
}

pub(crate) enum MirrorEvidence {
    Absent,
    Matched(PathBuf),
    Conflict(MirrorConflict),
}

pub(crate) fn compare_mirror_evidence(
    paths: &ExpansionPaths,
    task: &str,
    dataset_id: &str,
    source: &Path,
) -> Result<MirrorEvidence> {
    let Some(mirror) = mirror_dataset(paths, task, dataset_id) else {
        return Ok(MirrorEvidence::Absent);
    };
    let mut disagreements = BTreeMap::new();
    for name in ["train.csv", "test.csv"] {
        let authoritative = source.join(name);
        let mirrored = mirror.join(name);
        let source_hash = sha256_file(&authoritative)?;
        let mirror_hash = mirrored
            .is_file()
            .then(|| sha256_file(&mirrored))
            .transpose()?;
        if mirror_hash.as_deref() != Some(source_hash.as_str()) {
            disagreements.insert(
                name.to_owned(),
                json!({"source": source_hash, "mirror": mirror_hash}),
            );
        }
    }
    if disagreements.is_empty() {
        return Ok(MirrorEvidence::Matched(mirror));
    }
    let (conflict_sha256, receipt) =
        write_conflict_receipt(paths, task, dataset_id, source, &mirror, &disagreements)?;
    Ok(MirrorEvidence::Conflict(MirrorConflict {
        task: task.to_owned(),
        dataset_id: dataset_id.to_owned(),
        source: source.to_owned(),
        mirror,
        disagreements,
        conflict_sha256,
        receipt,
    }))
}

/// Compare an authoritative candidate against the optional local mirror.
///
/// Any missing or disagreeing mirrored train/test file produces a durable
/// receipt and returns an error. An absent mirror is allowed and recorded.
pub fn compare_mirror(
    paths: &ExpansionPaths,
    task: &str,
    dataset_id: &str,
    source: &Path,
) -> Result<Option<PathBuf>> {
    match compare_mirror_evidence(paths, task, dataset_id, source)? {
        MirrorEvidence::Absent => Ok(None),
        MirrorEvidence::Matched(mirror) => Ok(Some(mirror)),
        MirrorEvidence::Conflict(conflict) => {
            bail!(
                "{task}:{dataset_id}: mirror content conflict; receipt={}",
                conflict.receipt.display()
            )
        }
    }
}

pub(crate) fn candidate_directories(
    paths: &ExpansionPaths,
    task_filter: Option<&str>,
    limit: Option<usize>,
) -> Result<Vec<(String, PathBuf)>> {
    if let Some(task) = task_filter {
        ensure!(TASKS.contains(&task), "unsupported task: {task}");
    }
    let quarantine = load_v1_quarantine(paths)?;
    let mut candidates = Vec::new();
    for task in TASKS {
        if task_filter.is_some_and(|filter| filter != task) {
            continue;
        }
        let task_root = paths.source_root.join(corpus_dir_name(task));
        // The legacy corpus roots are mandatory; a new task's root may not
        // exist on every host yet and simply contributes no candidates.
        if !task_root.is_dir() && !matches!(task, "regression" | "binary") {
            continue;
        }
        let mut directories = fs::read_dir(&task_root)
            .with_context(|| format!("read authoritative root {}", task_root.display()))?
            .filter_map(std::result::Result::ok)
            .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
            .collect::<Vec<_>>();
        directories.sort_by_key(|entry| entry.file_name());
        for directory in directories {
            let dataset_id = directory.file_name().to_string_lossy().to_string();
            if !valid_dataset_id(&dataset_id) {
                continue;
            }
            if v1_identity_denied(quarantine.as_ref(), &dataset_id) {
                continue;
            }
            let source = directory.path();
            if !["train.csv", "test.csv", "meta.json"]
                .iter()
                .all(|name| source.join(name).is_file())
            {
                continue;
            }
            candidates.push((task.to_owned(), source));
            if limit.is_some_and(|maximum| candidates.len() >= maximum) {
                return Ok(candidates);
            }
        }
    }
    Ok(candidates)
}

/// Audit every selected authoritative dataset, collecting all mirror conflicts
/// rather than stopping at the first disagreement.
pub fn audit_discover(
    paths: &ExpansionPaths,
    task_filter: Option<&str>,
    limit: Option<usize>,
) -> Result<AuditInventory> {
    let candidates = candidate_directories(paths, task_filter, limit)?;
    let quarantine = load_v1_quarantine(paths)?;
    let mut entries = Vec::with_capacity(candidates.len());
    let mut conflicts = Vec::new();
    for (task, source) in &candidates {
        let dataset_id = source
            .file_name()
            .and_then(|value| value.to_str())
            .context("candidate dataset path has no UTF-8 identity")?
            .to_owned();
        if v1_identity_denied(quarantine.as_ref(), &dataset_id) {
            continue;
        }
        let mirror = match compare_mirror_evidence(paths, task, &dataset_id, source)? {
            MirrorEvidence::Absent => None,
            MirrorEvidence::Matched(mirror) => Some(mirror),
            MirrorEvidence::Conflict(conflict) => {
                conflicts.push(conflict);
                continue;
            }
        };
        let meta_path = source.join("meta.json");
        let meta = read_meta(&meta_path)?;
        entries.push(InventoryEntry {
            task: task.clone(),
            dataset_id: dataset_id.clone(),
            dataset_hash: dataset_id.clone(),
            source: source.clone(),
            source_family: source_family(&meta, &dataset_id)?,
            train_sha256: sha256_file(&source.join("train.csv"))?,
            test_sha256: sha256_file(&source.join("test.csv"))?,
            meta_sha256: sha256_file(&meta_path)?,
            mirror,
        });
    }
    conflicts.sort_by(|left, right| {
        (&left.task, &left.dataset_id).cmp(&(&right.task, &right.dataset_id))
    });
    Ok(AuditInventory {
        inspected: candidates.len(),
        entries,
        conflicts,
    })
}

pub fn write_conflict_summary(paths: &ExpansionPaths, audit: &AuditInventory) -> Result<PathBuf> {
    ensure!(
        !audit.conflicts.is_empty(),
        "cannot write an empty mirror conflict summary"
    );
    let conflicts = audit
        .conflicts
        .iter()
        .map(|conflict| {
            json!({
                "task": conflict.task,
                "dataset_id": conflict.dataset_id,
                "source": conflict.source,
                "mirror": conflict.mirror,
                "disagreements": conflict.disagreements,
                "conflict_sha256": conflict.conflict_sha256,
                "receipt": conflict.receipt,
            })
        })
        .collect::<Vec<_>>();
    let mut payload = json!({
        "schema": "JopeExpandedMirrorConflictSummary.v1",
        "authoritative_root": paths.source_root,
        "mirror_root": paths.mirror_root,
        "inspected": audit.inspected,
        "matched_or_unmirrored": audit.entries.len(),
        "conflict_count": conflicts.len(),
        "conflicts": conflicts,
    });
    let digest = canonical_json_sha256(&payload)?;
    payload["summary_sha256"] = Value::String(digest.clone());
    let summary = paths
        .receipt_root
        .join("summaries")
        .join(format!("{digest}.json"));
    if summary.is_file() {
        ensure!(
            read_json(&summary)? == payload,
            "mirror conflict summary content drift"
        );
    } else {
        atomic_json(&summary, &payload)?;
    }
    atomic_json(
        &paths.receipt_root.join("latest-summary.json"),
        &json!({
            "schema": "JopeExpandedMirrorConflictSummaryPointer.v1",
            "summary": summary,
            "summary_sha256": digest,
            "conflict_count": audit.conflicts.len(),
            "inspected": audit.inspected,
        }),
    )?;
    Ok(summary)
}

pub fn discover(
    paths: &ExpansionPaths,
    task_filter: Option<&str>,
    limit: Option<usize>,
) -> Result<Vec<InventoryEntry>> {
    let audit = audit_discover(paths, task_filter, limit)?;
    if let Some(conflict) = audit.conflicts.first() {
        bail!(
            "{}:{}: mirror content conflict; receipt={}",
            conflict.task,
            conflict.dataset_id,
            conflict.receipt.display()
        );
    }
    Ok(audit.entries)
}

pub fn cache_path(paths: &ExpansionPaths, task: &str, dataset_id: &str) -> PathBuf {
    paths
        .cache_root
        .join(format!("{task}-expanded"))
        .join(dataset_id)
}
