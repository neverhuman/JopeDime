use super::*;

/// Tensor inventory of a torch `.pt` archive without loading a single tensor and
/// without touching CUDA: `<prefix>/data/<n>` holds exactly one storage per saved
/// tensor, and `<prefix>/data.pkl` carries the state-dict keys as plain ASCII
/// pickle strings. Cost is the zip central directory plus a ~100 KiB pickle.
pub(super) fn checkpoint_shape(path: &Path) -> Result<CheckpointShape> {
    let file = File::open(path).with_context(|| format!("open checkpoint {}", path.display()))?;
    let mut archive = zip::ZipArchive::new(std::io::BufReader::new(file))
        .with_context(|| format!("checkpoint is not a torch archive: {}", path.display()))?;
    let names = archive
        .file_names()
        .map(|name| name.map(|name| name.into_owned()))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let mut tensors = 0usize;
    let mut pickle = None;
    for name in &names {
        if let Some((_, index)) = name.rsplit_once("/data/") {
            if !index.is_empty() && index.bytes().all(|byte| byte.is_ascii_digit()) {
                tensors += 1;
            }
        }
        if name.ends_with("data.pkl") {
            pickle = Some(name.clone());
        }
    }
    let pickle =
        pickle.with_context(|| format!("checkpoint lacks data.pkl: {}", path.display()))?;
    let mut bytes = Vec::new();
    archive
        .by_name(&pickle)?
        .read_to_end(&mut bytes)
        .with_context(|| format!("read {pickle} from {}", path.display()))?;
    // Every group name any schema cares about, resolved once against the pickle.
    let mut present = Vec::new();
    for schema in CHECKPOINT_SCHEMAS {
        for group in schema.groups.iter().chain(schema.marker.iter()) {
            if !present.iter().any(|seen: &String| seen == group)
                && contains_subslice(&bytes, group.as_bytes())
            {
                present.push((*group).to_owned());
            }
        }
    }
    let schema = resolve_checkpoint_schema(&present);
    let missing_groups = schema
        .groups
        .iter()
        .filter(|group| !present.iter().any(|seen| seen == *group))
        .map(|group| (*group).to_owned())
        .collect::<Vec<_>>();
    Ok(CheckpointShape {
        tensors,
        missing_groups,
        schema: schema.name,
        required_tensors: schema.tensors,
    })
}

pub(crate) fn contains_subslice(haystack: &[u8], needle: &[u8]) -> bool {
    !needle.is_empty()
        && haystack.len() >= needle.len()
        && haystack
            .windows(needle.len())
            .any(|window| window == needle)
}

pub(super) fn probe_checkpoint(
    path: PathBuf,
    sha256: String,
    origin: String,
) -> Result<Checkpoint> {
    ensure!(
        path.is_file(),
        "required checkpoint is absent: {}",
        path.display()
    );
    let shape = checkpoint_shape(&path)?;
    Ok(Checkpoint {
        path,
        sha256,
        origin,
        tensors: shape.tensors,
        missing_groups: shape.missing_groups,
        schema: shape.schema,
        required_tensors: shape.required_tensors,
        ..Checkpoint::default()
    })
}

pub(crate) fn mtime_ms(path: &Path) -> u128 {
    path.metadata()
        .and_then(|meta| meta.modified())
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|delta| delta.as_millis())
        .unwrap_or(0)
}

/// Every completed trainer run on disk, newest completion first.
///
/// A run is a candidate when its `train-result.json` names a checkpoint that is
/// still present. Unreadable or vanished candidates are skipped rather than
/// failing the scan: one corrupt historical run must not wedge admission.
pub(super) fn scan_lineage(runs_root: &Path) -> Vec<LineageRun> {
    let Ok(entries) = fs::read_dir(runs_root) else {
        return Vec::new();
    };
    let mut runs = Vec::new();
    for entry in entries.flatten() {
        let Ok(job_id) = entry.file_name().into_string() else {
            continue;
        };
        if !job_id.starts_with(LINEAGE_RUN_PREFIX) {
            continue;
        }
        let Ok(attempts) = fs::read_dir(entry.path().join("attempts")) else {
            continue;
        };
        for attempt in attempts.flatten() {
            let Ok(watermarks) = fs::read_dir(attempt.path().join("outputs/training")) else {
                continue;
            };
            for watermark in watermarks.flatten() {
                let result = watermark.path().join("train-result.json");
                if !result.is_file() {
                    continue;
                }
                if let Some(run) = lineage_run(&job_id, &result) {
                    runs.push(run);
                }
            }
        }
    }
    runs.sort_by(|left, right| {
        right
            .checkpoint
            .completed_at_ms
            .cmp(&left.checkpoint.completed_at_ms)
            .then_with(|| right.job_id.cmp(&left.job_id))
    });
    runs
}

pub(super) fn lineage_run(job_id: &str, result: &Path) -> Option<LineageRun> {
    let value = read_json(result).ok()?;
    let sha256 = required_sha256(&value, "checkpoint_sha256").ok()?;
    // `checkpoint_path` is the digest-named artefact; `checkpoint` is the mutable
    // best.model.pt alias, so only the former may be pinned.
    let named = value
        .get("checkpoint_path")
        .and_then(Value::as_str)
        .map(PathBuf::from)
        .filter(|path| path.is_file());
    let path = named.or_else(|| {
        let sibling = result.parent()?.join(format!("{sha256}.model.pt"));
        sibling.is_file().then_some(sibling)
    })?;
    let shape = checkpoint_shape(&path).ok()?;
    Some(LineageRun {
        job_id: job_id.to_owned(),
        checkpoint: Checkpoint {
            path,
            sha256,
            origin: format!("lineage:{job_id}"),
            tensors: shape.tensors,
            missing_groups: shape.missing_groups,
            schema: shape.schema,
            required_tensors: shape.required_tensors,
            selected_epoch: value
                .get("selected_checkpoint_epoch")
                .and_then(Value::as_u64),
            recipe: value
                .get("recipe")
                .and_then(Value::as_str)
                .map(str::to_owned),
            completed_at_ms: mtime_ms(result),
        },
    })
}

/// Ordered warm-start candidates: accepted pointer, then completed lineage runs
/// newest first, then the original anchor.
///
/// Before this ordering existed the watchdog fell straight through to the anchor
/// whenever `training/accepted.json` was absent, which is how successor
/// `c-r000002` was auto-submitted warm-starting the 130-tensor fbcd5459 and
/// silently re-randomising the binary branch.
pub(super) fn checkpoint_candidates(root: &Path, lineage: &[LineageRun]) -> Vec<Checkpoint> {
    let mut candidates = Vec::new();
    for relative in ["training/learning.json", "training/accepted.json"] {
        let pointer = root.join(relative);
        if !pointer.is_file() {
            continue;
        }
        let Ok(value) = read_json(&pointer) else {
            continue;
        };
        let (Some(path), Ok(digest)) = (
            value.get("checkpoint").and_then(Value::as_str),
            required_sha256(&value, "checkpoint_sha256"),
        ) else {
            continue;
        };
        if let Ok(checkpoint) = probe_checkpoint(
            PathBuf::from(path),
            digest,
            format!("accepted_pointer:{relative}"),
        ) {
            candidates.push(checkpoint);
        }
    }
    candidates.extend(lineage.iter().map(|run| run.checkpoint.clone()));
    if let Ok(value) = read_json(&root.join("proposals/current.json")) {
        if value.get("legacy_adopted").and_then(Value::as_bool) == Some(true) {
            if let Ok(digest) = required_sha256(&value, "checkpoint_sha256") {
                let path =
                    PathBuf::from(legacy_checkpoint_root_str()).join(format!("{digest}.model.pt"));
                if let Ok(checkpoint) = probe_checkpoint(path, digest, "original_anchor".to_owned())
                {
                    candidates.push(checkpoint);
                }
            }
        }
    }
    let mut seen = Vec::new();
    candidates.retain(|candidate| {
        if seen.contains(&candidate.sha256) {
            return false;
        }
        seen.push(candidate.sha256.clone());
        true
    });
    candidates
}

/// Pick the first candidate that carries the whole model.
///
/// Deficient candidates are never pinned, no matter how preferred they are: a
/// short checkpoint would train a lobotomised model with no error anywhere.
pub(super) fn select_checkpoint(candidates: &[Checkpoint]) -> Result<Checkpoint> {
    ensure!(
        !candidates.is_empty(),
        "no warm-start checkpoint is present"
    );
    candidates
        .iter()
        .find(|candidate| candidate.is_full())
        .cloned()
        .with_context(|| {
            format!(
                "every warm-start candidate is tensor-deficient: {}",
                candidates
                    .iter()
                    .filter_map(Checkpoint::deficiency)
                    .collect::<Vec<_>>()
                    .join("; ")
            )
        })
}

/// Cheap resolution used while an active trainer already blocks admission: no
/// 125 MiB rehash, and the lineage scan may be up to `LINEAGE_RESCAN_MS` stale.
pub(super) fn load_checkpoint_reference(
    root: &Path,
    runs_root: &Path,
    cache: &mut LineageCache,
) -> Result<Checkpoint> {
    let lineage = cache.get(runs_root, false).to_vec();
    let candidates = checkpoint_candidates(root, &lineage);
    let selected = select_checkpoint(&candidates);
    publish_warm_start_guard(root, &candidates, selected.as_ref().ok());
    selected
}

/// Full resolution used immediately before rendering or retrying a submission:
/// forces a fresh lineage scan and re-verifies the pinned bytes.
pub(super) fn load_checkpoint(
    root: &Path,
    runs_root: &Path,
    cache: &mut LineageCache,
) -> Result<Checkpoint> {
    let lineage = cache.get(runs_root, true).to_vec();
    let candidates = checkpoint_candidates(root, &lineage);
    let selected = select_checkpoint(&candidates);
    publish_warm_start_guard(root, &candidates, selected.as_ref().ok());
    let checkpoint = selected?;
    verify_file(&checkpoint.path, &checkpoint.sha256)?;
    // Belt and braces: the guard already filtered on shape, but a spec must never
    // leave this process pinning a deficient checkpoint.
    ensure!(
        checkpoint.is_full(),
        "refusing to pin a tensor-deficient warm-start checkpoint: {}",
        checkpoint.deficiency().unwrap_or_default()
    );
    Ok(checkpoint)
}

/// Publish the warm-start guard verdict for `jope_selfplay_alarms` to lift into
/// `ops/health-alarms.v1.json`. Written on every resolution so the alarm clears
/// itself once a full checkpoint reappears. Never fails the cycle.
pub(super) fn publish_warm_start_guard(
    root: &Path,
    candidates: &[Checkpoint],
    selected: Option<&Checkpoint>,
) {
    let deficient = selected.is_none();
    let report = json!({
        "schema": "JopeTrainerWarmStartGuard.v1",
        "generated_at_ms": now_ms() as u64,
        "alarm": "trainer_warm_start_deficient",
        "deficient": deficient,
        "severity": if deficient { "critical" } else { "ok" },
        "observed": selected.map_or_else(
            || format!("{} candidate(s), none full", candidates.len()),
            |checkpoint| format!(
                "{} tensors from {} ({})",
                checkpoint.tensors,
                checkpoint.origin,
                &checkpoint.sha256[..12.min(checkpoint.sha256.len())],
            ),
        ),
        "threshold": CHECKPOINT_SCHEMAS
            .iter()
            .map(|schema| format!(
                "{} (>={} tensors including {})",
                schema.name,
                schema.tensors,
                schema.groups.join(", "),
            ))
            .collect::<Vec<_>>()
            .join(" | "),
        "selected": selected.map(Checkpoint::describe),
        "candidates": candidates.iter().map(Checkpoint::describe).collect::<Vec<_>>(),
    });
    let _ = atomic_json(&root.join("ops/trainer-warm-start-guard.v1.json"), &report);
}

pub(crate) fn load_snapshot_watermark(pointer_path: &Path, labels: u64) -> Result<u64> {
    let pointer = read_json(pointer_path)?;
    let watermark = pointer
        .get("watermark")
        .and_then(Value::as_u64)
        .context("training corpus pointer lacks watermark")?;
    ensure!(
        watermark <= labels,
        "training corpus watermark {watermark} exceeds label pointer {labels}"
    );
    Ok(watermark)
}

pub(super) fn load_training_snapshot(
    pointer_path: &Path,
    labels: u64,
) -> Result<TrainingSnapshot> {
    let pointer = read_json(pointer_path)?;
    let watermark = load_snapshot_watermark(pointer_path, labels)?;
    let manifest = PathBuf::from(
        pointer
            .get("manifest")
            .and_then(Value::as_str)
            .context("training corpus pointer lacks manifest")?,
    );
    verify_file(&manifest, &required_sha256(&pointer, "manifest_sha256")?)?;
    let value = read_json(&manifest)?;
    ensure!(
        value.get("schema").and_then(Value::as_str) == Some("JopeTestOnlyTrainingCorpus.v1"),
        "training corpus schema mismatch"
    );
    ensure!(
        value.get("source_family_disjoint").and_then(Value::as_bool) == Some(true),
        "training corpus is not family-disjoint"
    );
    ensure!(
        value.get("source_label_watermark").and_then(Value::as_u64) == Some(watermark),
        "training manifest watermark mismatch"
    );
    ensure!(
        value
            .pointer("/strict_filter/test_adaptive")
            .and_then(Value::as_bool)
            == Some(true)
            && value
                .pointer("/strict_filter/benchmark_valid")
                .and_then(Value::as_bool)
                == Some(false),
        "training corpus authority mismatch"
    );
    let packed_manifest = PathBuf::from(
        pointer
            .get("packed_manifest")
            .and_then(Value::as_str)
            .context("training pointer lacks packed manifest")?,
    );
    verify_file(
        &packed_manifest,
        &required_sha256(&pointer, "packed_manifest_sha256")?,
    )?;
    let packed = read_json(&packed_manifest)?;
    ensure!(
        packed.get("source_manifest_sha256").and_then(Value::as_str)
            == pointer.get("manifest_sha256").and_then(Value::as_str),
        "packed corpus source manifest binding mismatch"
    );
    for key in ["train_sha256", "validation_sha256"] {
        ensure!(
            packed.get(key).and_then(Value::as_str) == value.get(key).and_then(Value::as_str),
            "packed corpus {key} binding mismatch"
        );
    }
    let packed_catalog = verified_path(&packed, "catalog_data", "catalog_data_sha256")?;
    let files = packed
        .get("files")
        .and_then(Value::as_object)
        .context("packed manifest lacks files")?;
    let mut packed_files = Vec::with_capacity(files.len());
    for (name, metadata) in files {
        ensure_safe_filename(name)?;
        let digest = required_sha256(metadata, "sha256")?;
        let path = packed_manifest
            .parent()
            .context("packed manifest has no parent")?
            .join(name);
        verify_file(&path, &digest)?;
        packed_files.push((name.clone(), path));
    }
    packed_files.sort_by(|left, right| left.0.cmp(&right.0));
    ensure!(!packed_files.is_empty(), "packed training corpus is empty");
    Ok(TrainingSnapshot {
        watermark,
        manifest,
        packed_manifest,
        packed_catalog,
        packed_files,
    })
}
