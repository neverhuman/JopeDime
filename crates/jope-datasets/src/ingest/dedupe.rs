use super::*;

pub struct Deduped {
    pub rows: Vec<CatalogRow>,
    pub duplicates: Vec<Duplicate>,
    pub rejects: Vec<RejectRecord>,
    pub blobs: HashMap<String, u64>,
}

/// Build rows in priority order, folding same-`dataset_hash` copies into `aliases`.
pub fn dedupe(
    source: Source,
    mut en: Enumeration,
    hashed: Vec<std::io::Result<Hashed>>,
) -> Deduped {
    let mut order: Vec<usize> = (0..en.candidates.len()).collect();
    order.sort_by(|&a, &b| {
        let (ca, cb) = (&en.candidates[a], &en.candidates[b]);
        (ca.priority, &ca.relpath).cmp(&(cb.priority, &cb.relpath))
    });
    let mut hashed: Vec<Option<std::io::Result<Hashed>>> = hashed.into_iter().map(Some).collect();
    let mut rows: Vec<CatalogRow> = Vec::new();
    let mut by_hash: HashMap<String, usize> = HashMap::new();
    let mut blobs: HashMap<String, u64> = HashMap::new();
    let ingested_at = now_rfc3339();
    for i in order {
        let c = &en.candidates[i];
        let h = match hashed[i]
            .take()
            .unwrap_or_else(|| Err(std::io::Error::other("not hashed")))
        {
            Ok(h) => h,
            Err(e) => {
                en.rejects.push(RejectRecord {
                    path: c.relpath.clone(),
                    reason: "io_error".into(),
                    detail: e.to_string(),
                });
                continue;
            }
        };
        if let Some(exp) = &c.expected {
            let got = [&h.train.sha256, &h.test.sha256, &h.meta.sha256];
            if let Some((f, e, g)) = FILES
                .iter()
                .zip(exp)
                .zip(got)
                .find(|((_, e), g)| e != g)
                .map(|((f, e), g)| (f, e, g))
            {
                en.rejects.push(RejectRecord {
                    path: c.relpath.clone(),
                    reason: "mirror_sha_mismatch".into(),
                    detail: format!("{f}: index {e} computed {g}"),
                });
                continue;
            }
        }
        let meta: serde_json::Value = match std::fs::read(c.dir.join("meta.json"))
            .map_err(|e| e.to_string())
            .and_then(|b| serde_json::from_slice(&b).map_err(|e| e.to_string()))
        {
            Ok(v) => v,
            Err(e) => {
                en.rejects.push(RejectRecord {
                    path: c.relpath.clone(),
                    reason: "meta_unparsable".into(),
                    detail: e,
                });
                continue;
            }
        };
        let Some(task) = c.task.or_else(|| task_from_meta(&meta)) else {
            en.rejects.push(RejectRecord {
                path: c.relpath.clone(),
                reason: "task_unknown".into(),
                detail: "no class directory and no meta.task_type".into(),
            });
            continue;
        };
        let ingest = Ingest {
            source: source.ingest_source(),
            relpath: c.relpath.clone(),
            ingested_at: ingested_at.clone(),
            tool_version: crate::TOOL_VERSION.into(),
        };
        let row = match build_row(&c.name, task, &meta, &h, ingest) {
            Ok(r) => r,
            Err(r) => {
                en.rejects.push(RejectRecord {
                    path: c.relpath.clone(),
                    reason: r.reason.into(),
                    detail: r.detail,
                });
                continue;
            }
        };
        if let Some(&idx) = by_hash.get(&row.dataset_hash) {
            let canon = &rows[idx];
            let same = canon.files.train.sha256 == h.train.sha256
                && canon.files.test.sha256 == h.test.sha256;
            let rule = if same {
                "dataset-hash-duplicate"
            } else {
                "dataset-hash-conflict"
            };
            en.duplicates.push(Duplicate {
                path: c.relpath.clone(),
                rule: rule.into(),
                canonical: canon.ingest.as_ref().map(|i| i.relpath.clone()),
                dataset_hash: Some(row.dataset_hash.clone()),
            });
            if same {
                rows[idx]
                    .aliases
                    .extend(aliases_for(&c.relpath, &h, source.ingest_source()));
            }
            continue;
        }
        for (_, f) in row.files.entries() {
            blobs.insert(f.sha256.clone(), f.bytes);
        }
        by_hash.insert(row.dataset_hash.clone(), rows.len());
        rows.push(row);
    }
    Deduped {
        rows,
        duplicates: en.duplicates,
        rejects: en.rejects,
        blobs,
    }
}
