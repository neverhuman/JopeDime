use super::*;
use crate::catalog::{Ingest, IngestSource};
use crate::hash::digest_file;
use crate::meta::{build_row, task_from_meta, Hashed};
use crate::paths::now_rfc3339;

/// Seed the cache from a local tree by hashing. Two layouts are understood per subdirectory:
/// the registry layout `<id>/{train.csv,test.csv,meta.json}` (rows are built from `meta.json`)
/// and the repository layout `<slug>/{train.csv,test.csv,metadata.json}` (the row is committed
/// and every file must match it). A `manifest.json` collection next to or above the tree is
/// installed under `collections/`.
pub fn link_from(
    cache: &Cache,
    from: &Path,
    only: &[String],
    default_task: Option<Task>,
) -> Result<LinkReport> {
    let mut report = LinkReport::default();
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(from)
        .with_context(|| format!("read_dir {}", from.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    let ingested_at = now_rfc3339();
    for dir in dirs {
        let name = dir
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let result = (|| -> Result<Option<CatalogRow>> {
            if dir.join("metadata.json").is_file() {
                let row = read_manifest(&dir.join("metadata.json"))?;
                if !only.is_empty()
                    && !only
                        .iter()
                        .any(|o| o == &name || o == &row.id || o == &row.dataset_hash)
                {
                    return Ok(None);
                }
                for (file, f) in row.files.entries() {
                    let path = dir.join(file);
                    let d =
                        digest_file(&path).with_context(|| format!("hash {}", path.display()))?;
                    if d.sha256 != f.sha256 {
                        return Err(anyhow!(
                            "{}: sha256 {} != metadata {}",
                            path.display(),
                            d.sha256,
                            f.sha256
                        ));
                    }
                    store_blob(cache, &path, &f.sha256)?;
                }
                return Ok(Some(row));
            }
            if crate::dataset_dir_hash(&name).is_none()
                || !crate::ingest::FILES.iter().all(|f| dir.join(f).is_file())
            {
                return Ok(None);
            }
            if !only.is_empty()
                && !only
                    .iter()
                    .any(|o| o == &name || Some(o.as_str()) == crate::dataset_dir_hash(&name))
            {
                return Ok(None);
            }
            let hashed = Hashed {
                train: digest_file(&dir.join("train.csv"))?,
                test: digest_file(&dir.join("test.csv"))?,
                meta: digest_file(&dir.join("meta.json"))?,
            };
            let meta: serde_json::Value =
                serde_json::from_slice(&std::fs::read(dir.join("meta.json"))?)?;
            let task = task_from_meta(&meta)
                .or(default_task)
                .ok_or_else(|| anyhow!("{name}: meta.json has no task_type; pass --task"))?;
            let ingest = Ingest {
                source: IngestSource::Link,
                relpath: name.clone(),
                ingested_at: ingested_at.clone(),
                tool_version: crate::TOOL_VERSION.into(),
            };
            let row = build_row(&name, task, &meta, &hashed, ingest)
                .map_err(|r| anyhow!("{name}: {} ({})", r.reason, r.detail))?;
            for (file, f) in row.files.entries() {
                store_blob(cache, &dir.join(file), &f.sha256)?;
            }
            Ok(Some(row))
        })();
        match result {
            Ok(Some(row)) => match materialize(cache, &row) {
                Ok(_) => report.datasets += 1,
                Err(e) => report.errors.push(format!("{name}: {e:#}")),
            },
            Ok(None) => report.skipped += 1,
            Err(e) => report.errors.push(format!("{name}: {e:#}")),
        }
    }
    for candidate in [
        from.join("manifest.json"),
        from.join("..").join("manifest.json"),
    ] {
        let Ok(text) = std::fs::read_to_string(&candidate) else {
            continue;
        };
        let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) else {
            continue;
        };
        if value.get("schema").and_then(|s| s.as_str())
            != Some(crate::catalog::COLLECTION_SCHEMA_ID)
        {
            continue;
        }
        match Collection::validate_value(&value) {
            Ok(coll) => {
                let dir = cache.collections_dir().join(&coll.name);
                std::fs::create_dir_all(&dir)?;
                std::fs::write(dir.join(format!("v{}.json", coll.version)), &text)?;
                std::fs::write(dir.join("LATEST"), format!("v{}\n", coll.version))?;
                report.collections.push(coll.name.clone());
            }
            Err(e) => report
                .errors
                .push(format!("{}: {}", candidate.display(), e.join("; "))),
        }
        break;
    }
    Ok(report)
}
