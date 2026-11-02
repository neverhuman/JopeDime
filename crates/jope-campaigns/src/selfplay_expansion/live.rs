use super::prelude::*;
use super::{
    activation::atomic_bytes,
    inventory::cache_path,
    model::{ExpansionPaths, FileLock, CONTEXT_STRIDE, TASKS},
};

pub(super) const LIVE_POINTER_SCHEMA: &str = "JopeTestOnlyContextPointer.v1";
pub(super) const LIVE_CATALOG_SCHEMA: &str = "JopeTestOnlyContextCatalog.v1";
pub const SIDECAR_NAMES: [&str; 4] = [
    "hyperion3train128dim.vs",
    "hyperion3train.vs",
    "train3.vs",
    "train_target.vs",
];

pub(super) fn read_live_pointer(live_root: &Path) -> Result<(Value, Value, Vec<u8>)> {
    let pointer_path = live_root.join("current.json");
    let pointer = read_json(&pointer_path)?;
    ensure!(
        pointer["schema"].as_str() == Some(LIVE_POINTER_SCHEMA),
        "live catalog pointer schema drift"
    );
    let catalog_path = Path::new(
        pointer["catalog"]
            .as_str()
            .context("live pointer lacks catalog")?,
    );
    ensure!(
        sha256_file(catalog_path)?
            == pointer["catalog_sha256"]
                .as_str()
                .context("live pointer lacks catalog hash")?,
        "live catalog file hash drift"
    );
    let catalog = read_json(catalog_path)?;
    ensure!(
        catalog["schema"].as_str() == Some(LIVE_CATALOG_SCHEMA),
        "live catalog schema drift"
    );
    let data_path = Path::new(
        pointer["data"]
            .as_str()
            .context("live pointer lacks data")?,
    );
    let blob = fs::read(data_path)?;
    let data_sha256 = pointer["data_sha256"]
        .as_str()
        .context("live pointer lacks data hash")?;
    ensure!(
        sha256_bytes(&blob) == data_sha256 && catalog["data_sha256"].as_str() == Some(data_sha256),
        "live catalog blob hash drift"
    );
    let entries = catalog["entries"]
        .as_array()
        .context("live catalog entries are absent")?;
    ensure!(
        blob.len() == entries.len() * CONTEXT_STRIDE
            && catalog["row_count"].as_u64() == Some(entries.len() as u64),
        "live catalog row/stride drift"
    );
    for entry in entries {
        let index = entry["catalog_index"]
            .as_u64()
            .context("live entry lacks catalog index")? as usize;
        let offset = index * CONTEXT_STRIDE;
        ensure!(
            sha256_bytes(&blob[offset..offset + CONTEXT_STRIDE])
                == entry["context_sha256"]
                    .as_str()
                    .context("live entry lacks context hash")?,
            "live catalog context slice drift at index {index}"
        );
    }
    Ok((pointer, catalog, blob))
}

pub(super) struct TrancheLoad {
    tranche_id: String,
    catalog_path: PathBuf,
    catalog_sha256: String,
    entries: Vec<CatalogEntryIdentity>,
    blob: Vec<u8>,
}

pub(super) fn load_activated_tranches(
    catalog_root: &Path,
    exclude: &[String],
) -> Result<Vec<TrancheLoad>> {
    let ledger = catalog_root.join("activations.jsonl");
    let mut loads: Vec<TrancheLoad> = Vec::new();
    let mut seen_ids = HashSet::new();
    if ledger.is_file() {
        for line in BufReader::new(File::open(&ledger)?).lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let receipt: Value = serde_json::from_str(&line)?;
            let tranche_id = receipt["tranche_id"]
                .as_str()
                .context("activation receipt lacks tranche id")?
                .to_owned();
            if !seen_ids.insert(tranche_id.clone()) {
                continue;
            }
            if exclude.contains(&tranche_id) {
                continue;
            }
            let catalog_path = PathBuf::from(
                receipt["catalog"]
                    .as_str()
                    .context("activation receipt lacks catalog")?,
            );
            let catalog_sha256 = receipt["catalog_sha256"]
                .as_str()
                .context("activation receipt lacks catalog hash")?
                .to_owned();
            ensure!(
                sha256_file(&catalog_path)? == catalog_sha256,
                "{tranche_id}: tranche catalog file hash drift"
            );
            let payload = read_json(&catalog_path)?;
            ensure!(
                payload["schema"].as_str() == Some("JopeExpandedCatalog.v2"),
                "{tranche_id}: tranche catalog schema drift"
            );
            let entries: Vec<CatalogEntryIdentity> =
                serde_json::from_value(payload["entries"].clone())?;
            let data_path = PathBuf::from(
                receipt["data"]
                    .as_str()
                    .context("activation receipt lacks data")?,
            );
            let blob = fs::read(&data_path)?;
            ensure!(
                sha256_bytes(&blob)
                    == receipt["data_sha256"]
                        .as_str()
                        .context("activation receipt lacks data hash")?
                    && blob.len() == entries.len() * CONTEXT_STRIDE,
                "{tranche_id}: tranche blob drift"
            );
            for entry in &entries {
                let offset = entry.catalog_index * CONTEXT_STRIDE;
                ensure!(
                    sha256_bytes(&blob[offset..offset + CONTEXT_STRIDE]) == entry.context_sha256,
                    "{tranche_id}: tranche context slice drift at {}",
                    entry.catalog_index
                );
            }
            loads.push(TrancheLoad {
                tranche_id,
                catalog_path,
                catalog_sha256,
                entries,
                blob,
            });
        }
    }
    for excluded in exclude {
        ensure!(
            seen_ids.contains(excluded),
            "excluded tranche {excluded} is not in the activation ledger"
        );
    }
    Ok(loads)
}

pub(super) fn append_history(live_root: &Path, previous_pointer: &Value) -> Result<()> {
    let mut record = json!({
        "schema": "JopeLiveCatalogPointerHistory.v1",
        "replaced_at_ms": now_ms(),
        "pointer": previous_pointer,
    });
    record["record_sha256"] = Value::String(canonical_json_sha256(&record)?);
    let mut ledger = OpenOptions::new()
        .create(true)
        .append(true)
        .mode(0o644)
        .open(live_root.join("pointer-history.jsonl"))?;
    serde_json::to_writer(&mut ledger, &record)?;
    ledger.write_all(b"\n")?;
    ledger.sync_all()?;
    Ok(())
}

/// Fold every activated v2 tranche into the live v1-schema catalog.
///
/// v1 entries pass through unmodified; v2 entries are appended in ledger order
/// with `current_catalog: false`, cache-local CSV paths, family-inherited
/// splits, and freshly verified context slices. The pointer flip is atomic and
/// the replaced pointer is retained in `pointer-history.jsonl`.
pub fn merge_live(
    paths: &ExpansionPaths,
    live_root: &Path,
    exclude: &[String],
    dry_run: bool,
) -> Result<Value> {
    let _lock = (!dry_run)
        .then(|| FileLock::exclusive(&live_root.join(".merge.lock")))
        .transpose()?;
    let (pointer, catalog, v1_blob) = read_live_pointer(live_root)?;
    let v1_entries = catalog["entries"]
        .as_array()
        .context("live catalog entries are absent")?
        .clone();
    let mut identities = HashSet::new();
    let mut live_contexts: BTreeMap<(String, String), String> = BTreeMap::new();
    let mut family_splits: BTreeMap<String, String> = BTreeMap::new();
    for entry in &v1_entries {
        let task = entry["task"].as_str().context("v1 entry lacks task")?;
        let dataset = entry["dataset_hash"]
            .as_str()
            .context("v1 entry lacks dataset hash")?;
        ensure!(
            identities.insert((task.to_owned(), dataset.to_owned())),
            "live catalog repeats {task}:{dataset}"
        );
        live_contexts.insert(
            (task.to_owned(), dataset.to_owned()),
            entry["context_sha256"]
                .as_str()
                .context("v1 entry lacks context hash")?
                .to_owned(),
        );
        let family = entry["source_family"]
            .as_str()
            .context("v1 entry lacks source family")?;
        let split = entry["split"].as_str().context("v1 entry lacks split")?;
        if let Some(existing) = family_splits.get(family) {
            ensure!(
                existing == split,
                "live catalog family {family} straddles train/validation"
            );
        } else {
            family_splits.insert(family.to_owned(), split.to_owned());
        }
    }
    let tranches = load_activated_tranches(&paths.catalog_root, exclude)?;
    let mut merged_entries = v1_entries.clone();
    let mut blob = v1_blob;
    let mut inherited_splits = 0_u64;
    let mut appended = Vec::new();
    let mut already_merged = Vec::new();
    for tranche in &tranches {
        // A tranche whose every identity is already live with an identical
        // context was merged by an earlier run; skip it so merge-live stays
        // idempotent. A partial or context-differing overlap fails closed.
        let merged_count = tranche
            .entries
            .iter()
            .filter(|entry| {
                live_contexts
                    .get(&(
                        entry.dataset.task.clone(),
                        entry.dataset.dataset_hash.clone(),
                    ))
                    .is_some_and(|context| *context == entry.context_sha256)
            })
            .count();
        if merged_count == tranche.entries.len() {
            already_merged.push(Value::String(tranche.tranche_id.clone()));
            continue;
        }
        ensure!(
            merged_count == 0,
            "{}: tranche is partially present in the live catalog",
            tranche.tranche_id
        );
        for entry in &tranche.entries {
            let identity = (
                entry.dataset.task.clone(),
                entry.dataset.dataset_hash.clone(),
            );
            ensure!(
                !live_contexts.contains_key(&identity),
                "{}: {}:{} exists in the live catalog with a different context",
                tranche.tranche_id,
                entry.dataset.task,
                entry.dataset.dataset_hash
            );
            ensure!(
                identities.insert(identity.clone()),
                "{}: duplicate dataset identity {}:{}",
                tranche.tranche_id,
                entry.dataset.task,
                entry.dataset.dataset_hash
            );
            let split = if let Some(v1_split) = family_splits.get(&entry.source_family) {
                if *v1_split != entry.split {
                    inherited_splits += 1;
                }
                v1_split.clone()
            } else {
                family_splits.insert(entry.source_family.clone(), entry.split.clone());
                entry.split.clone()
            };
            ensure!(
                family_splits.get(&entry.source_family) == Some(&split),
                "{}: family {} straddles train/validation",
                tranche.tranche_id,
                entry.source_family
            );
            let cache = cache_path(paths, &entry.dataset.task, &entry.dataset.dataset_hash);
            let train_csv = cache.join("train.csv");
            let test_csv = cache.join("test.csv");
            ensure!(
                sha256_file(&train_csv)? == entry.dataset.train_sha256
                    && sha256_file(&test_csv)? == entry.dataset.test_sha256,
                "{}: expanded cache CSV drift for {}",
                tranche.tranche_id,
                entry.dataset.dataset_hash
            );
            let metric = crate::task::TaskKind::parse(&entry.dataset.task)
                .with_context(|| {
                    format!("unknown task in expanded tranche: {}", entry.dataset.task)
                })?
                .primary_metric();
            let catalog_index = merged_entries.len();
            merged_entries.push(json!({
                "catalog_index": catalog_index,
                "context_sha256": entry.context_sha256,
                "current_catalog": false,
                "dataset_hash": entry.dataset.dataset_hash,
                "metric": metric,
                "source_catalog_index": Value::Null,
                "source_family": entry.source_family,
                "split": split,
                "task": entry.dataset.task,
                "test_csv": test_csv,
                "test_sha256": entry.dataset.test_sha256,
                "train_csv": train_csv,
                "train_sha256": entry.dataset.train_sha256,
            }));
        }
        blob.extend_from_slice(&tranche.blob);
        appended.push(json!({
            "tranche_id": tranche.tranche_id,
            "catalog": tranche.catalog_path,
            "catalog_sha256": tranche.catalog_sha256,
            "row_count": tranche.entries.len(),
        }));
    }
    if appended.is_empty() {
        return Ok(json!({
            "schema": "JopeLiveCatalogMergeResult.v1",
            "nothing_to_merge": true,
            "dry_run": dry_run,
            "row_count": v1_entries.len(),
            "already_merged_tranches": already_merged,
        }));
    }
    ensure!(
        blob.len() == merged_entries.len() * CONTEXT_STRIDE,
        "merged blob stride drift"
    );
    for entry in &merged_entries {
        let index = entry["catalog_index"].as_u64().context("index absent")? as usize;
        let offset = index * CONTEXT_STRIDE;
        ensure!(
            entry["context_sha256"].as_str()
                == Some(sha256_bytes(&blob[offset..offset + CONTEXT_STRIDE]).as_str()),
            "merged context slice drift at index {index}"
        );
    }
    let mut train_families = HashSet::new();
    let mut validation_families = HashSet::new();
    for entry in &merged_entries {
        let family = entry["source_family"].as_str().context("family absent")?;
        match entry["split"].as_str().context("split absent")? {
            "validation" => validation_families.insert(family.to_owned()),
            _ => train_families.insert(family.to_owned()),
        };
    }
    ensure!(
        train_families.is_disjoint(&validation_families),
        "merged catalog families straddle train/validation"
    );
    let data_sha256 = sha256_bytes(&blob);
    let task_counts = TASKS
        .into_iter()
        .filter_map(|task| {
            let count = merged_entries
                .iter()
                .filter(|entry| entry["task"].as_str() == Some(task))
                .count();
            (count != 0).then(|| (task.to_owned(), count))
        })
        .collect::<BTreeMap<_, _>>();
    let regression_count = merged_entries
        .iter()
        .filter(|entry| entry["task"].as_str() == Some("regression"))
        .count();
    let binary_count = merged_entries
        .iter()
        .filter(|entry| entry["task"].as_str() == Some("binary"))
        .count();
    let mut merged_catalog = catalog.clone();
    merged_catalog["entries"] = Value::Array(merged_entries);
    merged_catalog["row_count"] = json!(
        v1_entries.len()
            + appended
                .iter()
                .map(|row| row["row_count"].as_u64().unwrap_or(0) as usize)
                .sum::<usize>()
    );
    merged_catalog["regression_count"] = json!(regression_count);
    merged_catalog["binary_count"] = json!(binary_count);
    merged_catalog["task_counts"] = json!(task_counts);
    merged_catalog["data_sha256"] = Value::String(data_sha256.clone());
    merged_catalog["source_family_disjoint"] = Value::Bool(true);
    merged_catalog["expanded_tranches"] = Value::Array(appended.clone());
    merged_catalog["merged_from_catalog_sha256"] = pointer["catalog_sha256"].clone();
    let catalog_content_sha256 = canonical_json_sha256(&merged_catalog)?;
    let summary = json!({
        "schema": "JopeLiveCatalogMergeResult.v1",
        "nothing_to_merge": false,
        "dry_run": dry_run,
        "row_count": merged_catalog["row_count"],
        "regression_count": regression_count,
        "binary_count": binary_count,
        "task_counts": task_counts,
        "inherited_splits": inherited_splits,
        "tranches": appended,
        "already_merged_tranches": already_merged,
        "data_sha256": data_sha256,
        "catalog_content_sha256": catalog_content_sha256,
    });
    if dry_run {
        return Ok(summary);
    }
    let data_path = live_root.join(format!("{data_sha256}.f32.bin"));
    if data_path.is_file() {
        ensure!(
            sha256_file(&data_path)? == data_sha256,
            "merged blob content drift on disk"
        );
    } else {
        atomic_bytes(&data_path, &blob)?;
    }
    let catalog_path = live_root.join(format!("{catalog_content_sha256}.catalog.json"));
    if catalog_path.is_file() {
        ensure!(
            read_json(&catalog_path)? == merged_catalog,
            "refusing to rewrite a differing merged catalog"
        );
    } else {
        atomic_json(&catalog_path, &merged_catalog)?;
    }
    append_history(live_root, &pointer)?;
    atomic_json(
        &live_root.join("current.json"),
        &json!({
            "schema": LIVE_POINTER_SCHEMA,
            "catalog": catalog_path,
            "catalog_sha256": sha256_file(&catalog_path)?,
            "data": data_path,
            "data_sha256": data_sha256,
        }),
    )?;
    Ok(summary)
}
