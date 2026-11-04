use super::*;
pub(super) fn json(path: &Path) -> Value {
    fs::read(path)
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default()
}
pub(super) fn safe_path(root: &Path, path: &Path) -> Option<PathBuf> {
    let path = root.join(path);
    if !path.starts_with(root.join("runs")) {
        return None;
    }
    for ancestor in path.ancestors().take_while(|p| *p != root) {
        if ancestor.is_symlink() {
            return None;
        }
    }
    let canonical = path.canonicalize().ok()?;
    canonical
        .starts_with(root.join("runs"))
        .then_some(canonical)
}
pub(super) fn discover(root: &Path) -> (BTreeSet<PathBuf>, Vec<String>) {
    let mut cycles = BTreeSet::new();
    let mut runs = BTreeSet::new();
    let mut errors = Vec::new();
    let mut campaigns = BTreeSet::new();
    let campaign = json(&root.join("runs/campaign-progress.json"));
    let current_parent = campaign["current_cycle"]
        .as_str()
        .map(|p| root.join(p))
        .and_then(|p| p.parent().map(Path::to_owned));
    let entries = match fs::read_dir(root.join("runs")) {
        Ok(entries) => entries,
        Err(error) => {
            return (
                runs,
                vec![format!("{}: {error}", root.join("runs").display())],
            )
        }
    };
    for entry in entries.flatten() {
        let Some(path) = safe_path(root, &entry.path()) else {
            continue;
        };
        if !path.is_dir() || path.file_name().is_some_and(|n| n == "parity") {
            continue;
        }
        if path.join("metrics.jsonl").is_file() {
            runs.insert(path.clone());
        }
        if path.join("cycle.json").is_file() {
            cycles.insert(path.clone());
        }
        let manifest = path.join("completed-cycles.json");
        if manifest.is_file() || current_parent.as_ref() == Some(&path) {
            campaigns.insert(path.clone());
        }
        if manifest.is_file() {
            let value = json(&manifest);
            if let Some(rows) = value["cycles"].as_array() {
                for row in rows {
                    if let Some(p) = row["cycle"]
                        .as_str()
                        .and_then(|s| safe_path(root, Path::new(s)))
                    {
                        cycles.insert(p);
                    } else {
                        errors.push(format!(
                            "{}: missing or indirect cycle {}",
                            manifest.display(),
                            row["cycle"]
                        ));
                    }
                }
            } else {
                errors.push(format!("{}: invalid cycle inventory", manifest.display()));
            }
        }
    }
    if let Some(path) = current_parent.and_then(|p| safe_path(root, &p)) {
        if path != root.join("runs") {
            campaigns.insert(path);
        }
    }
    // Expansions have their own nested cycle namespace. Walk only campaign
    // containers and stop at each cycle; never traverse GP candidate trees.
    // Keep old expansion branches after the active campaign moves elsewhere.
    let mut visited = BTreeSet::new();
    while let Some(path) = campaigns.pop_first() {
        if !visited.insert(path.clone()) {
            continue;
        }
        let children = match fs::read_dir(&path) {
            Ok(children) => children,
            Err(error) => {
                errors.push(format!("{}: {error}", path.display()));
                continue;
            }
        };
        for child in children.flatten() {
            let Some(child) = safe_path(root, &child.path()) else {
                continue;
            };
            if !child.is_dir() {
                continue;
            }
            if child.join("cycle.json").is_file() {
                cycles.insert(child);
            } else if !child
                .file_name()
                .is_some_and(|n| n.to_string_lossy().starts_with("cycle-"))
                && !child.join("metrics.jsonl").is_file()
            {
                campaigns.insert(child);
            }
        }
    }
    if let Some(path) = json(&root.join("runs/campaign-progress.json"))["current_cycle"].as_str() {
        if let Some(path) = safe_path(root, Path::new(path)) {
            cycles.insert(path);
        } else {
            errors.push(format!("Missing or indirect current cycle: {path}"));
        }
    }
    // Continuous blocks have their own namespace and retain every intermediate run.
    if let Some(path) = json(&root.join("runs/continuous-progress.json"))["output"]
        .as_str()
        .and_then(|p| safe_path(root, Path::new(p)))
    {
        if let Ok(entries) = fs::read_dir(path) {
            for entry in entries.flatten() {
                if entry.file_name().to_string_lossy().starts_with("cycle-") {
                    if let Some(path) = safe_path(root, &entry.path().join("train-2")) {
                        runs.insert(path);
                    }
                }
            }
        }
    }
    // Include active standalone training before its first checkpoint exists.
    if let Some(path) = json(&root.join("runs/neural-service.json"))["run"].as_str() {
        if !Path::new(path)
            .components()
            .any(|p| p.as_os_str() == "parity")
        {
            if let Some(path) = safe_path(root, Path::new(path)) {
                runs.insert(path);
            }
        }
    }
    for cycle in cycles {
        for block in [
            "train-1",
            "train-2",
            "auxiliary/train",
            "auxiliary-pack/train",
        ] {
            if let Some(path) = safe_path(root, &cycle.join(block)) {
                runs.insert(path);
            } else if matches!(block, "train-1" | "train-2")
                && json(&cycle.join("complete.json"))["dl_gp_dl_complete"] == true
            {
                errors.push(format!(
                    "Missing or indirect saved training block: {}",
                    cycle.join(block).display()
                ));
            }
        }
    }
    (runs, errors)
}
