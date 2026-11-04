use super::*;

pub(crate) fn prune_snapshots(args: &Args) -> Value {
    let root = &args.root;
    if args.keep_snapshots == 0 {
        return json!({"pruned": 0, "reason": "retention disabled"});
    }
    let training = root.join("training");
    let pointer_dir = fs::read(root.join(POINTER))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
        .and_then(|value| {
            value
                .get("manifest")
                .and_then(Value::as_str)
                .map(PathBuf::from)
        })
        .and_then(|path| path.parent().map(Path::to_path_buf));

    // Every watermark the live tournament state mentions, as a string, matched
    // against the zero-padded watermark in the directory name.
    let mut pinned: Vec<String> = Vec::new();
    if let Ok(bytes) = fs::read(root.join("ops/flywheel-state.json")) {
        if let Ok(state) = serde_json::from_slice::<Value>(&bytes) {
            for key in ["/tournament/watermark", "/last_frozen_watermark"] {
                if let Some(value) = state.pointer(key).and_then(Value::as_u64) {
                    pinned.push(format!("snapshot-{value:09}"));
                }
            }
        }
    }
    // A recoverable trainer intent has not necessarily been staged by SCQ yet:
    // its immutable spec may still name the source snapshot. Protect both the
    // crash-safe controller state and its last status document so retention
    // cannot turn a restart into an unrecoverable missing-input failure.
    for relative in [
        "training/task-learning-controller.v2.json",
        "ops/task-learning-controller.v2.json",
    ] {
        if let Ok(bytes) = fs::read(root.join(relative)) {
            if let Ok(state) = serde_json::from_slice::<Value>(&bytes) {
                for key in [
                    "/intent/trainer/corpus_watermark",
                    "/active_intent/trainer/corpus_watermark",
                ] {
                    if let Some(value) = state.pointer(key).and_then(Value::as_u64) {
                        pinned.push(format!("snapshot-{value:09}"));
                    }
                }
            }
        }
    }

    // EVERY complete snapshot, newest first — protections are applied to the
    // tail, not to the ranking. Excluding protected directories from the
    // ranking instead would mean `keep = 3` retained 3 snapshots PLUS the
    // pointer's PLUS every tournament-pinned one, which at ~35 GB each is how a
    // retention pass quietly fails to retain anything.
    let mut snapshots: Vec<(PathBuf, SystemTime, u64)> = Vec::new();
    let Ok(entries) = fs::read_dir(&training) else {
        return json!({"pruned": 0, "reason": "training directory is unreadable"});
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !name.starts_with("snapshot-") || !path.is_dir() {
            continue;
        }
        if !path.join("corpus-manifest.json").is_file() {
            continue;
        }
        let Ok(metadata) = entry.metadata() else {
            continue;
        };
        let Ok(modified) = metadata.modified() else {
            continue;
        };
        let bytes = directory_bytes(&path);
        snapshots.push((path, modified, bytes));
    }
    snapshots.sort_by_key(|row| std::cmp::Reverse(row.1));
    let keep_after = Duration::from_secs(args.keep_snapshots_min_age_hours * 3600);
    let mut removed = Vec::new();
    let mut freed: u64 = 0;
    for (path, modified, bytes) in snapshots.into_iter().skip(args.keep_snapshots) {
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_owned();
        if Some(&path) == pointer_dir.as_ref() {
            continue;
        }
        if pinned
            .iter()
            .any(|prefix| name.starts_with(prefix.as_str()))
        {
            continue;
        }
        let age = SystemTime::now()
            .duration_since(modified)
            .unwrap_or_default();
        if age < keep_after {
            continue;
        }
        if args.dry_run {
            freed = freed.saturating_add(bytes);
            removed.push(json!({
                "path": path.to_string_lossy(),
                "bytes": bytes,
                "age_hours": age.as_secs() / 3600,
                "dry_run": true,
            }));
            continue;
        }
        match fs::remove_dir_all(&path) {
            Ok(()) => {
                freed = freed.saturating_add(bytes);
                removed.push(json!({
                    "path": path.to_string_lossy(),
                    "bytes": bytes,
                    "age_hours": age.as_secs() / 3600,
                }));
            }
            Err(error) => removed.push(json!({
                "path": path.to_string_lossy(),
                "error": error.to_string(),
            })),
        }
    }
    json!({
        "pruned": removed.len(),
        "freed_bytes": freed,
        "keep": args.keep_snapshots,
        "min_age_hours": args.keep_snapshots_min_age_hours,
        "removed": removed,
    })
}

/// Shallow: corpus snapshots are flat directories of a few large files.
fn directory_bytes(path: &Path) -> u64 {
    fs::read_dir(path)
        .map(|entries| {
            entries
                .flatten()
                .filter_map(|entry| entry.metadata().ok())
                .filter(|metadata| metadata.is_file())
                .map(|metadata| metadata.len())
                .sum()
        })
        .unwrap_or(0)
}

pub(crate) fn prune_packed(args: &Args) -> Value {
    let root = &args.root;
    if args.keep_snapshots == 0 {
        return json!({"pruned": 0, "reason": "retention disabled"});
    }
    let packed_root = root.join("training/packed");
    let mut pinned: Vec<String> = Vec::new();
    if let Ok(bytes) = fs::read(root.join(POINTER)) {
        if let Ok(value) = serde_json::from_slice::<Value>(&bytes) {
            if let Some(path) = value.get("packed_manifest").and_then(Value::as_str) {
                if let Some(sha) = path
                    .split("packed/")
                    .nth(1)
                    .and_then(|rest| rest.split('/').next())
                {
                    pinned.push(sha.to_owned());
                }
            }
        }
    }
    for relative in [
        "ops/flywheel-state.json",
        "training/task-learning-controller.v2.json",
        "ops/task-learning-controller.v2.json",
    ] {
        if let Ok(text) = fs::read_to_string(root.join(relative)) {
            for chunk in text.split("packed/").skip(1) {
                let sha: String = chunk
                    .chars()
                    .take_while(|c| c.is_ascii_hexdigit())
                    .collect();
                if sha.len() >= 8 {
                    pinned.push(sha);
                }
            }
        }
    }

    let Ok(entries) = fs::read_dir(&packed_root) else {
        return json!({"pruned": 0, "reason": "packed directory is unreadable"});
    };
    let keep_after = Duration::from_secs(args.keep_snapshots_min_age_hours * 3600);
    let mut dirs: Vec<(PathBuf, SystemTime, u64, bool)> = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !path.is_dir() {
            continue;
        }
        let orphan_tmp = name.starts_with('.') && name.contains(".tmp-");
        if !orphan_tmp && !name.chars().all(|c| c.is_ascii_hexdigit()) {
            continue;
        }
        let Ok(metadata) = entry.metadata() else {
            continue;
        };
        let Ok(modified) = metadata.modified() else {
            continue;
        };
        let bytes = directory_bytes(&path);
        dirs.push((path, modified, bytes, orphan_tmp));
    }
    dirs.sort_by_key(|row| std::cmp::Reverse(row.1));
    let mut removed = Vec::new();
    let mut freed: u64 = 0;
    let mut rank = 0usize;
    for (path, modified, bytes, orphan_tmp) in dirs {
        if !orphan_tmp {
            rank += 1;
            if rank <= args.keep_snapshots {
                continue;
            }
        }
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_owned();
        if !orphan_tmp && pinned.iter().any(|sha| name.starts_with(sha.as_str())) {
            continue;
        }
        let age = SystemTime::now()
            .duration_since(modified)
            .unwrap_or_default();
        if age < keep_after {
            continue;
        }
        if args.dry_run {
            freed = freed.saturating_add(bytes);
            removed.push(json!({
                "path": path.to_string_lossy(),
                "bytes": bytes,
                "age_hours": age.as_secs() / 3600,
                "orphan_tmp": orphan_tmp,
                "dry_run": true,
            }));
            continue;
        }
        match fs::remove_dir_all(&path) {
            Ok(()) => {
                freed = freed.saturating_add(bytes);
                removed.push(json!({
                    "path": path.to_string_lossy(),
                    "bytes": bytes,
                    "age_hours": age.as_secs() / 3600,
                    "orphan_tmp": orphan_tmp,
                }));
            }
            Err(error) => removed.push(json!({
                "path": path.to_string_lossy(),
                "error": error.to_string(),
            })),
        }
    }
    json!({
        "pruned": removed.len(),
        "freed_bytes": freed,
        "keep": args.keep_snapshots,
        "min_age_hours": args.keep_snapshots_min_age_hours,
        "removed": removed,
    })
}
