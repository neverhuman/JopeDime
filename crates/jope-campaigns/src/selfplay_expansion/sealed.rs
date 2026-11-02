use super::model::{sealed_target, ExpansionPaths, InventoryEntry, SealedCohort, TASKS};
use super::prelude::*;

pub(super) fn deterministic_sealed_key(seed: u64, row: &InventoryEntry) -> [u8; 32] {
    Sha256::digest(
        format!(
            "{seed}|sealed|{}|{}|{}",
            row.task, row.source_family, row.dataset_hash
        )
        .as_bytes(),
    )
    .into()
}

pub(super) fn cohort_hash(cohort: &SealedCohort) -> Result<String> {
    let value = json!({
        "schema": cohort.schema,
        "immutable": cohort.immutable,
        "classification": cohort.classification,
        "training_ingestion_eligible": cohort.training_ingestion_eligible,
        "counts": cohort.counts,
        "entries": cohort.entries,
        "created_at_ms": cohort.created_at_ms,
    });
    canonical_json_sha256(&value)
}

pub(super) fn validate_sealed_cohort(cohort: &SealedCohort) -> Result<()> {
    ensure!(
        cohort.schema == "JopeSealedExpandedCohort.v1"
            && cohort.immutable
            && cohort.classification == "sealed_zero_shot"
            && !cohort.training_ingestion_eligible,
        "sealed cohort envelope drift"
    );
    ensure!(
        cohort.cohort_sha256 == cohort_hash(cohort)?,
        "sealed cohort content drift"
    );
    let mut identities = HashSet::new();
    let mut families = HashSet::new();
    for entry in &cohort.entries {
        entry.dataset_identity().validate()?;
        ensure!(
            identities.insert((entry.task.as_str(), entry.dataset_hash.as_str())),
            "sealed cohort repeats a dataset identity"
        );
        ensure!(
            families.insert(entry.source_family.as_str()),
            "sealed cohort repeats a source family"
        );
    }
    let mut expected_total = 0_usize;
    for task in TASKS {
        let declared = cohort.counts.get(task).copied().unwrap_or(0);
        let observed = cohort.entries.iter().filter(|row| row.task == task).count();
        let target = sealed_target(task);
        // Legacy tasks must carry their full frozen carve-out. The new tasks
        // are absent from cohorts frozen before their carve-out exists (D7):
        // zero is legal for them, any other shortfall is drift.
        let legacy = matches!(task, "regression" | "binary");
        let complete = declared == target || (!legacy && declared == 0);
        ensure!(
            observed == declared && complete,
            "{task}: sealed cohort count drift"
        );
        expected_total += declared;
    }
    ensure!(
        cohort.entries.len() == expected_total,
        "sealed cohort total count drift"
    );
    Ok(())
}

/// Freeze exactly `sealed_target(task)` family-disjoint datasets per task in
/// a private file (256 per legacy task; D7 carve-outs for the new tasks).
///
/// The returned value deliberately contains only counts and the cohort digest;
/// identities and families never appear in normal status output.
pub fn freeze_sealed(
    paths: &ExpansionPaths,
    entries: &[InventoryEntry],
    seed: u64,
) -> Result<Value> {
    if paths.sealed_path.is_file() {
        let cohort: SealedCohort = serde_json::from_value(read_json(&paths.sealed_path)?)?;
        validate_sealed_cohort(&cohort)?;
        return Ok(json!({"cohort_sha256": cohort.cohort_sha256, "counts": cohort.counts}));
    }
    let mut selected = Vec::with_capacity(TASKS.iter().map(|task| sealed_target(task)).sum());
    let mut used_families = HashSet::new();
    let mut counts = BTreeMap::new();
    for task in TASKS {
        let target = sealed_target(task);
        let mut candidates = entries
            .iter()
            .filter(|entry| entry.task == task)
            .cloned()
            .collect::<Vec<_>>();
        candidates.sort_by_key(|row| deterministic_sealed_key(seed, row));
        for row in candidates {
            if used_families.insert(row.source_family.clone()) {
                selected.push(row);
                if selected.iter().filter(|row| row.task == task).count() == target {
                    break;
                }
            }
        }
        ensure!(
            selected.iter().filter(|row| row.task == task).count() == target,
            "{task}: insufficient family-disjoint candidates for sealed cohort"
        );
        counts.insert(task.to_owned(), target);
    }
    let mut cohort = SealedCohort {
        schema: "JopeSealedExpandedCohort.v1".to_owned(),
        immutable: true,
        classification: "sealed_zero_shot".to_owned(),
        training_ingestion_eligible: false,
        counts,
        entries: selected,
        created_at_ms: now_ms(),
        cohort_sha256: String::new(),
    };
    cohort.cohort_sha256 = cohort_hash(&cohort)?;
    validate_sealed_cohort(&cohort)?;
    atomic_json(&paths.sealed_path, &serde_json::to_value(&cohort)?)?;
    fs::set_permissions(&paths.sealed_path, fs::Permissions::from_mode(0o600))?;
    if let Some(parent) = paths.sealed_path.parent() {
        fs::set_permissions(parent, fs::Permissions::from_mode(0o700))?;
    }
    Ok(json!({"cohort_sha256": cohort.cohort_sha256, "counts": cohort.counts}))
}

pub(super) type DatasetKeys = HashSet<(String, String)>;
pub(super) type FamilyKeys = HashSet<String>;

pub(super) fn load_sealed_exclusions(paths: &ExpansionPaths) -> Result<(DatasetKeys, FamilyKeys)> {
    ensure!(
        paths.sealed_path.is_file(),
        "sealed cohort must be frozen before adaptive activation"
    );
    let cohort: SealedCohort = serde_json::from_value(read_json(&paths.sealed_path)?)?;
    validate_sealed_cohort(&cohort)?;
    Ok((
        cohort
            .entries
            .iter()
            .map(|row| (row.task.clone(), row.dataset_hash.clone()))
            .collect(),
        cohort
            .entries
            .iter()
            .map(|row| row.source_family.clone())
            .collect(),
    ))
}

/// Remove private sealed identities and their entire source families before an
/// inventory can feed cache materialization or adaptive admission.
pub fn adaptive_inventory(
    paths: &ExpansionPaths,
    entries: &[InventoryEntry],
) -> Result<Vec<InventoryEntry>> {
    let (sealed_identities, sealed_families) = load_sealed_exclusions(paths)?;
    Ok(entries
        .iter()
        .filter(|row| {
            !sealed_identities.contains(&(row.task.clone(), row.dataset_hash.clone()))
                && !sealed_families.contains(&row.source_family)
        })
        .cloned()
        .collect())
}
