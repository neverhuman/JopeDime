use super::*;

/// The scan document of `root`: outermost Cargo candidates older than `hours` (all when `None`).
pub fn scan(
    root: &Path,
    hours: Option<u64>,
    interrupt: Interrupt<'_>,
    measure: &dyn Fn(&Path, bool) -> Outcome<u64>,
) -> Outcome<Value> {
    let root = absolute(root)?;
    let mut candidates: std::collections::BTreeSet<PathBuf> = Default::default();
    walk(&root, interrupt, &mut |directory, dirs, files| {
        dirs.retain(|d| d != ".git");
        if files.iter().any(|f| f == ".rustc_info.json")
            && lstat(&directory.join(".rustc_info.json"))?
                .file_type()
                .is_file()
        {
            candidates.insert(directory.to_path_buf());
        }
        let name = directory.file_name().and_then(|n| n.to_str());
        if matches!(name, Some("debug" | "release")) && dirs.iter().any(|d| d == ".fingerprint") {
            let parent = directory
                .parent()
                .expect("a debug/release directory has a parent");
            if (parent == root || (parent.starts_with(&root) && parent != root))
                && lstat(&directory.join(".fingerprint"))?.file_type().is_dir()
            {
                candidates.insert(parent.to_path_buf());
            }
        }
        Ok(())
    })?;
    let cutoff = hours.map(|h| {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64()
            - h as f64 * 3600.0
    });
    let mut rows = vec![];
    // `sum(r["allocated_bytes"] for r in rows)`: Python's int has no width, so the total is held
    // exactly (in `u128`, which fewer than 2^64 measurements cannot leave) and written exactly.
    let mut total: u128 = 0;
    for candidate in &candidates {
        if candidate
            .ancestors()
            .skip(1)
            .any(|a| candidates.contains(a))
        {
            continue;
        }
        let mut newest = mtime(&lstat(candidate)?);
        // Measure all contents, including nested .git evidence, after identifying candidates.
        // Visit fresh trees too so traversal failures stay visible.
        walk(candidate, interrupt, &mut |directory, dirs, files| {
            for name in dirs.iter().chain(files) {
                newest = newest.max(mtime(&lstat(&directory.join(name))?));
            }
            Ok(())
        })?;
        if cutoff.is_some_and(|c| newest > c) {
            continue;
        }
        let bytes = measure(candidate, false)?;
        total = total
            .checked_add(u128::from(bytes))
            .ok_or_else(|| Failure::new("the allocated-bytes total leaves 128 bits"))?;
        // `du` prints the path it measured, and its output is decoded strictly, so a measured
        // candidate is UTF-8; any other measurement is held to the same rule.
        let path = candidate.to_str().ok_or_else(|| {
            Failure::new(format!(
                "candidate {} is not UTF-8; its surrogate-escaped name cannot be represented",
                crate::py::os_repr(candidate.as_os_str())
            ))
        })?;
        let row: Dict = [
            ("path", Value::from(path)),
            ("allocated_bytes", exact_int(u128::from(bytes))),
            ("newest_mtime", Value::Float(newest)),
        ]
        .into_iter()
        .collect();
        rows.push(Value::Dict(row));
    }
    // A candidate path is UTF-8 here (`du`, which prints it, was decoded strictly). A root that is
    // not UTF-8 would be printed surrogate-escaped (`\udcff`), which this JSON cannot hold: refused.
    let root_text = root.to_str().ok_or_else(|| {
        Failure::new(format!(
            "scan root {} is not UTF-8; its surrogate-escaped name cannot be represented",
            crate::py::os_repr(root.as_os_str())
        ))
    })?;
    let document: Dict = [
        ("completed_at", Value::from(utc().iso)),
        ("root", Value::from(root_text)),
        (
            "stale_hours",
            hours.map_or(Value::Null, |h| exact_int(u128::from(h))),
        ),
        ("candidates", Value::List(rows)),
        ("candidate_allocated_bytes_sum", exact_int(total)),
        ("sum_may_double_count_hardlinks", Value::from(true)),
        // This object is returned only after every traversal and measurement succeeds.
        ("report_only", Value::from(true)),
        ("complete", Value::from(true)),
    ]
    .into_iter()
    .collect();
    Ok(Value::Dict(document))
}

/// A count as the JSON integer Python wrote: `int` has no width, so one beyond `i64` is written as
/// its decimal literal ([`Value::BigInt`]), never wrapped or rounded.
pub(super) fn exact_int(n: u128) -> Value {
    i64::try_from(n).map_or_else(|_| Value::BigInt(n.to_string()), Value::Int)
}

/// `--stale HOURS`: nonnegative whole hours of at most eight digits (`value.isascii() and
/// value.isdecimal()`, then `int(value)`, which the strict number grammar reads).
pub fn whole_hours(value: &str) -> Result<u64, String> {
    if value.is_empty() || !value.bytes().all(|b| b.is_ascii_digit()) {
        return Err("--stale requires nonnegative whole hours".into());
    }
    // Bound arithmetic, not the reporting/deletion policy (there is no deletion policy).
    if value.len() > 8 {
        return Err("--stale age is too large".into());
    }
    py::int(value)
        .ok()
        .and_then(|hours| u64::try_from(hours).ok())
        .ok_or_else(|| "--stale requires nonnegative whole hours".into())
}
