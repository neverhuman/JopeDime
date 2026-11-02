use super::*;

pub(super) fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

pub(super) fn get_str<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .with_context(|| format!("missing string field {key}"))
}

pub(super) fn rglob(directory: &Path, name: &str, found: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(directory) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            rglob(&path, name, found);
        } else if path.file_name().map(|n| n == name).unwrap_or(false) {
            found.push(path);
        }
    }
}

// ---------------------------------------------------------------------------
// pointers
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub(super) struct LiveCatalog {
    pub(super) pointer_path: PathBuf,
    pub(super) catalog: PathBuf,
    pub(super) catalog_sha256: String,
    pub(super) data: PathBuf,
    pub(super) data_sha256: String,
}

pub(super) fn read_live_catalog(root: &Path) -> Result<LiveCatalog> {
    let pointer_path = root.join("catalog/current.json");
    let pointer = read_json(&pointer_path)?;
    let catalog_sha256 = get_str(&pointer, "catalog_sha256")?.to_owned();
    let data_sha256 = get_str(&pointer, "data_sha256")?.to_owned();
    if !is_sha256(&catalog_sha256) || !is_sha256(&data_sha256) {
        bail!("live catalog pointer carries a malformed hash");
    }
    Ok(LiveCatalog {
        catalog: PathBuf::from(get_str(&pointer, "catalog")?),
        data: PathBuf::from(get_str(&pointer, "data")?),
        catalog_sha256,
        data_sha256,
        pointer_path,
    })
}

#[derive(Debug, Clone)]
pub(super) struct ProposalBinding {
    pub(super) pointer_path: PathBuf,
    pub(super) pointer: Value,
    pub(super) checkpoint_sha256: String,
    /// The catalog the *current* proposal file was generated against. Read
    /// from the pointer when present (this driver writes it) and otherwise
    /// recovered from the summary artifact the generator emitted.
    pub(super) bound_catalog_sha256: Option<String>,
}

pub(super) fn read_proposal_binding(root: &Path) -> Result<ProposalBinding> {
    let pointer_path = root.join("proposals/current.json");
    let pointer = read_json(&pointer_path)?;
    let checkpoint_sha256 = get_str(&pointer, "checkpoint_sha256")?.to_owned();
    if !is_sha256(&checkpoint_sha256) {
        bail!("proposal pointer checkpoint hash is malformed");
    }
    let mut bound = pointer
        .get("catalog_sha256")
        .and_then(Value::as_str)
        .map(str::to_owned);
    if bound.is_none() {
        if let Some(summary) = pointer.get("summary").and_then(Value::as_str) {
            let summary_path = resolve(&pointer_path, Path::new(summary));
            if summary_path.is_file() {
                bound = read_json(&summary_path).ok().and_then(|value| {
                    value
                        .get("catalog_sha256")
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                });
            }
        }
    }
    Ok(ProposalBinding {
        pointer_path,
        pointer,
        checkpoint_sha256,
        bound_catalog_sha256: bound,
    })
}

/// One generation lane: which branch (None = both), which checkpoint, and which
/// catalog the currently-published set for that lane was built against.
#[derive(Debug, Clone)]
pub(super) struct Lane {
    pub(super) task: Option<String>,
    pub(super) checkpoint_sha256: String,
    pub(super) checkpoint: PathBuf,
    pub(super) bound_catalog_sha256: Option<String>,
}

impl Lane {
    pub(super) fn label(&self) -> String {
        self.task.clone().unwrap_or_else(|| "combined".to_owned())
    }
    /// Family prefix that blocks a resubmission: any job for this lane's
    /// (task, checkpoint), whatever catalog it was aimed at.
    pub(super) fn family(&self) -> String {
        match &self.task {
            Some(task) => format!(
                "jope-selfplay-propose-{}-{}-c",
                task_tag(task),
                &self.checkpoint_sha256[..12]
            ),
            None => format!("jope-selfplay-propose-{}-c", &self.checkpoint_sha256[..12]),
        }
    }
    pub(super) fn prefix(&self, catalog_sha256: &str) -> String {
        format!("{}{}", self.family(), &catalog_sha256[..12])
    }
}

/// Resolve a checkpoint artifact for a per-task branch. The promotion gate
/// publishes into `$R/training/immutable/checkpoints/`; the historical warm
/// lineage lives in `checkpoints_str()`. Either is acceptable, both are hash-verified,
/// because a propose job that stages the wrong bytes produces a whole
/// generation of silently mislabelled proposals.
pub(super) fn resolve_checkpoint(root: &Path, sha256: &str, declared: Option<&str>) -> Result<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(path) = declared {
        candidates.push(PathBuf::from(path));
    }
    candidates.push(
        root.join("training/immutable/checkpoints")
            .join(format!("{sha256}.model.pt")),
    );
    candidates.push(PathBuf::from(checkpoints_str()).join(format!("{sha256}.model.pt")));
    for candidate in candidates {
        if candidate.is_file() && sha256_file(&candidate)? == sha256 {
            return Ok(candidate);
        }
    }
    bail!("no content-verified artifact for checkpoint {sha256}");
}

/// Per-task accepted checkpoints, or `None` when the branches agree (or the
/// pointer does not exist yet, which is the state before the first per-task
/// promotion lands). `None` means "run the historical combined lane".
pub(super) fn read_task_lanes(root: &Path, binding: &ProposalBinding) -> Result<Option<Vec<Lane>>> {
    let pointer_path = root.join(TASK_ACCEPTED_POINTER);
    if !pointer_path.is_file() {
        return Ok(None);
    }
    let document = read_json(&pointer_path)?;
    let mut selected: Vec<(String, String, Option<String>)> = Vec::new();
    for task in TASK_BRANCHES {
        let entry = document.pointer(&format!("/tasks/{task}"));
        let sha = entry
            .and_then(|value| value.get("checkpoint_sha256"))
            .and_then(Value::as_str)
            .unwrap_or(&binding.checkpoint_sha256)
            .to_owned();
        if !is_sha256(&sha) {
            bail!("accepted-by-task entry for {task} carries a malformed checkpoint hash");
        }
        let declared = entry
            .and_then(|value| value.get("checkpoint"))
            .and_then(Value::as_str)
            .map(str::to_owned);
        selected.push((task.to_owned(), sha, declared));
    }
    // Every branch on the same checkpoint is not a fork: keep the single
    // whole-catalog lane so `proposals/current.json` stays authoritative and no
    // extra GPU generation is spent proving the branches equal.
    if selected.iter().all(|(_, sha, _)| *sha == selected[0].1) {
        return Ok(None);
    }
    let published = read_task_proposal_pointer(root)?;
    let mut lanes = Vec::new();
    for (task, sha, declared) in selected {
        let bound = published
            .pointer(&format!("/tasks/{task}"))
            .filter(|entry| {
                entry.get("checkpoint_sha256").and_then(Value::as_str) == Some(sha.as_str())
            })
            .and_then(|entry| entry.get("catalog_sha256"))
            .and_then(Value::as_str)
            .map(str::to_owned);
        let checkpoint = resolve_checkpoint(root, &sha, declared.as_deref())
            .with_context(|| format!("resolve {task} branch checkpoint"))?;
        lanes.push(Lane {
            task: Some(task),
            checkpoint_sha256: sha,
            checkpoint,
            bound_catalog_sha256: bound,
        });
    }
    Ok(Some(lanes))
}

pub(super) fn read_task_proposal_pointer(root: &Path) -> Result<Value> {
    let path = root.join("proposals/current-by-task.json");
    match fs::symlink_metadata(&path) {
        Ok(_) => read_json(&path),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            Ok(json!({"schema": TASK_POINTER_SCHEMA, "tasks": {}}))
        }
        Err(error) => Err(error).with_context(|| format!("inspect {}", path.display())),
    }
}

pub(super) fn resolve(pointer: &Path, target: &Path) -> PathBuf {
    if target.is_absolute() {
        target.to_path_buf()
    } else {
        pointer
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(target)
    }
}
