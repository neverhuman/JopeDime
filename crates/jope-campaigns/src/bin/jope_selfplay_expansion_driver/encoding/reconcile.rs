use super::super::super::*;
use super::*;

/// Pull outputs of completed encode chunks into the per-task fetch dirs.
/// Encoding and fetching are deliberately OUTSIDE the guardrail/valve — the
/// valve paces catalog admission only, never encode throughput.
pub(crate) fn reconcile_encode_fetches(log: &mut Vec<Value>) -> Result<()> {
    let Ok(out) = sh(scq_str(), &["--json", "ls"]) else {
        return Ok(());
    };
    let Ok(value) = serde_json::from_str::<Value>(&out) else {
        return Ok(());
    };
    let binding = Vec::new();
    for job in value
        .pointer("/Jobs/jobs")
        .and_then(Value::as_array)
        .unwrap_or(&binding)
    {
        let id = job.get("id").and_then(Value::as_str).unwrap_or("");
        let job_state = job
            .get("state")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        if !(id.starts_with("jope-selfplay-breadth-encode-")
            || id.starts_with("jope-binexp-")
            || id.starts_with("jope-regexp-"))
        {
            continue;
        }
        if !matches!(job_state, "completed" | "failed" | "cancelled") {
            continue;
        }
        let task =
            if id.starts_with("jope-binexp-") || id.starts_with("jope-selfplay-breadth-encode-") {
                "binary"
            } else {
                "regression"
            };
        let root = fetch_root(task);
        let marker = root.join(format!(".fetched-{id}"));
        if marker.exists() {
            continue;
        }
        std::fs::create_dir_all(root.join("targets"))?;
        std::fs::create_dir_all(root.join("foundation"))?;
        let tmp = root.join(format!(".fetch-tmp-{id}"));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp)?;
        if sh(
            scq_str(),
            &["fetch", id, "--output", &tmp.to_string_lossy()],
        )
        .is_err()
        {
            let _ = std::fs::remove_dir_all(&tmp);
            continue;
        }
        let mut moved = 0_usize;
        let mut stack = vec![tmp.clone()];
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir)?.flatten() {
                let path = entry.path();
                let name = entry.file_name().to_string_lossy().into_owned();
                if path.is_dir() && name == "targets" {
                    for ds in std::fs::read_dir(&path)?.flatten() {
                        let to = root.join("targets").join(ds.file_name());
                        if !to.exists() && std::fs::rename(ds.path(), &to).is_ok() {
                            moved += 1;
                        }
                    }
                } else if path.is_dir() && name == "foundation" {
                    for f in std::fs::read_dir(&path)?.flatten() {
                        let to = root.join("foundation").join(f.file_name());
                        if !to.exists() {
                            let _ = std::fs::rename(f.path(), &to);
                        }
                    }
                } else if path.is_dir() {
                    stack.push(path);
                }
            }
        }
        let _ = std::fs::remove_dir_all(&tmp);
        let adopted = emit_chunk_adoption_receipts(id, job_state, task, &root)?;
        std::fs::write(&marker, b"ok")?;
        log.push(json!({
            "step": "encode_fetched",
            "job": id,
            "terminal_state": job_state,
            "datasets_moved": moved,
            "datasets_adopted": adopted,
        }));
    }
    Ok(())
}
