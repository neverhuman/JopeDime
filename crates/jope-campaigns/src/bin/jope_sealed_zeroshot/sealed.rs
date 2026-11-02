use super::*;

// --------------------------------------------------------------------------
// sealed cohort (READ ONLY)
// --------------------------------------------------------------------------

#[derive(Clone, Deserialize)]
pub(super) struct SealedEntry {
    pub(super) task: String,
    pub(super) dataset_id: String,
    pub(super) dataset_hash: String,
    pub(super) source: PathBuf,
    pub(super) mirror: PathBuf,
    pub(super) source_family: String,
    pub(super) train_sha256: String,
    pub(super) test_sha256: String,
}

#[derive(Deserialize)]
pub(super) struct SealedCohort {
    pub(super) schema: String,
    pub(super) classification: String,
    pub(super) immutable: bool,
    pub(super) cohort_sha256: String,
    pub(super) entries: Vec<SealedEntry>,
}

pub(super) fn load_sealed(root: &Path) -> Result<SealedCohort> {
    let path = root.join("catalog/private/sealed-expanded-v1.json");
    let text = fs::read_to_string(&path)
        .with_context(|| format!("read sealed cohort {} (READ ONLY)", path.display()))?;
    let cohort: SealedCohort = serde_json::from_str(&text).context("parse sealed cohort")?;
    if cohort.schema != "JopeSealedExpandedCohort.v1" {
        bail!("unexpected sealed cohort schema {}", cohort.schema);
    }
    if cohort.classification != "sealed_zero_shot" || !cohort.immutable {
        bail!("sealed cohort is not an immutable zero-shot cohort");
    }
    if cohort.entries.len() != 512 {
        bail!("sealed cohort size drift: {} entries", cohort.entries.len());
    }
    Ok(cohort)
}

// --------------------------------------------------------------------------
// contamination: which sealed datasets already carry observations
// --------------------------------------------------------------------------

/// Index-led probe: `(task,dataset_hash)` is the covering prefix of
/// `observations_best_v1`, so this is 512 index seeks, never a scan.
pub(super) fn contaminated(
    root: &Path,
    cohort: &SealedCohort,
) -> Result<BTreeMap<String, (u64, f64)>> {
    let db = root.join("corpus/strict-test-exact.sqlite");
    let conn = rusqlite::Connection::open_with_flags(
        &db,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .with_context(|| format!("open corpus read-only {}", db.display()))?;
    conn.busy_timeout(std::time::Duration::from_secs(180))?;
    let mut statement = conn.prepare(
        "SELECT count(*), coalesce(max(score), -1e308) FROM observations
         WHERE task=?1 AND dataset_hash=?2",
    )?;
    let mut out = BTreeMap::new();
    for entry in &cohort.entries {
        let (n, best): (i64, f64) = statement
            .query_row(rusqlite::params![entry.task, entry.dataset_hash], |row| {
                Ok((row.get(0)?, row.get(1)?))
            })?;
        if n > 0 {
            out.insert(
                format!("{}:{}", entry.task, entry.dataset_hash),
                (n as u64, best),
            );
        }
    }
    Ok(out)
}
