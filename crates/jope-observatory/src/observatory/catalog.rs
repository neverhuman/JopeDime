use super::*;

/// Observe configured storage evidence, falling back to the dated catalog when needed.
pub fn storage(c: &Config) -> Value {
    let catalog = observe(&c.repo_root.join("catalogs/s3.json"));
    if let Some(path) = &c.storage_snapshot {
        let live = observe(path);
        if live.value["schema"] == "jope.storage-observation/v1"
            && live.value["buckets"].as_array().is_some_and(|rows| {
                rows.iter().all(|b| {
                    b["bucket"].is_string() && b["region"].is_string() && b["status"].is_string()
                })
            })
            && (live.value["errors"].is_null()
                || live.value["errors"]
                    .as_array()
                    .is_some_and(|rows| rows.iter().all(Value::is_string)))
        {
            return json!({"evidence":live,"catalog_fallback":false});
        }
    }
    // Historical catalogs remain explicitly dated; object metadata is not a dataset census.
    let buckets: Vec<_> = catalog.value["buckets"].as_array().into_iter().flatten().map(|b| json!({"bucket":b["bucket"],"region":b["region"],"role":b["role"],"replica_of":b["replica_of"],"bytes":b["bytes"],"objects":b["objects"],"observed_at":b["inventory"]["captured_utc"],"status":"historical","scope":"current object versions in dated metadata inventory; replicas counted physically"})).collect();
    let bytes: u64 = buckets.iter().filter_map(|b| b["bytes"].as_u64()).sum();
    let objects: u64 = buckets.iter().filter_map(|b| b["objects"].as_u64()).sum();
    json!({"evidence":{"status":if buckets.is_empty(){"unavailable"}else{"historical"},"source":catalog.source,"sha256":catalog.sha256,"observed_at":catalog.value["inventoried"],"value":{"schema":"jope.storage-observation/v1","buckets":buckets,"observed_bytes":if buckets.iter().any(|b|b["bytes"].is_u64()) {json!(bytes)}else{Value::Null},"observed_objects":if buckets.iter().any(|b|b["objects"].is_u64()){json!(objects)}else{Value::Null},"complete":false,"scope":"last catalog snapshot; not a current fleet total","errors":[]}},"catalog_fallback":true})
}

/// Summarize up to eight hash-pinned GP catalog inputs without inferring HWM admission.
pub fn gp_catalog(inputs: &[GpInput]) -> Value {
    let mut programs: BTreeMap<String, Value> = BTreeMap::new();
    let mut families: BTreeMap<String, usize> = BTreeMap::new();
    let mut sources = vec![];
    let mut all_valid = !inputs.is_empty();
    for input in inputs.iter().take(8) {
        let obs = observe(&input.path);
        let hash_matches = obs.sha256.as_deref() == Some(&input.sha256);
        let rows = obs.value["programs"].as_array();
        all_valid &= hash_matches && rows.is_some_and(|r| r.len() <= 30_000);
        sources.push(json!({"label":input.label,"scope":input.scope,"source":obs.source,"sha256":obs.sha256,"expected_sha256":input.sha256,"hash_matches":hash_matches,"declared_rows":rows.map(Vec::len),"scan_complete":hash_matches&&rows.is_some_and(|r|r.len()<=30_000),"rows_scanned":rows.map(|r|r.len().min(30_000)),"observed_at":obs.observed_at}));
        if !hash_matches {
            continue;
        }
        for row in rows.into_iter().flatten().take(30_000) {
            let (Some(id), Some(dsl)) = (row["program_sha256"].as_str(), row["dsl"].as_str())
            else {
                continue;
            };
            if id.len() != 64 || !id.bytes().all(|b| b.is_ascii_hexdigit()) || dsl.len() > 32_768 {
                continue;
            }
            let operators: BTreeSet<String> = dsl
                .split('(')
                .skip(1)
                .filter_map(|s| s.split_whitespace().next())
                .filter(|s| !s.is_empty() && *s != "pipe")
                .map(|s| s.trim_end_matches(')').to_owned())
                .collect();
            let family = operators
                .iter()
                .find(|s| {
                    s.starts_with("ridge")
                        || s.starts_with("knn")
                        || s.starts_with("gpr")
                        || s.starts_with("logistic")
                })
                .cloned()
                .unwrap_or_else(|| "other".into());
            if let Some(existing) = programs.get_mut(id) {
                existing["context_occurrences"] =
                    json!(existing["context_occurrences"].as_u64().unwrap_or(1) + 1);
                continue;
            }
            *families.entry(family.clone()).or_default() += 1;
            programs.insert(id.into(),json!({"id":id,"dsl":dsl,"operators":operators,"family":family,"context_occurrences":1,"source":input.label,"scope":input.scope,"score":null,"score_note":"No evaluator-bound score in this library record"}));
        }
    }
    json!({"status":if all_valid{"observed"}else if programs.is_empty(){"unavailable"}else{"partial"},"unique_programs":if programs.is_empty()&&!all_valid{Value::Null}else{json!(programs.len())},"scope":"unique declared program identities in the configured, hash-pinned library snapshots; not a census of every fleet library; no HWM admission inferred","global_complete":false,"families":families,"sources":sources,"programs":programs.into_values().collect::<Vec<_>>()})
}

/// Read compact, recorded assertions for training waves under `repo`.
pub fn waves(repo: &Path) -> Value {
    let artifacts = observe(&repo.join("catalogs/training-artifacts.json"));
    let root = repo.join("docs/research/waves");
    let mut names = directory_names(&root, 256);
    names.sort();
    let mut result = vec![];
    for name in names {
        let dir = root.join(&name);
        let card = observe(&dir.join("card.json"));
        if card.value.is_null() {
            continue;
        }
        let manifest = observe(&dir.join("wave.manifest.json"));
        let metrics = observe(&dir.join("metrics.json"));
        let assertions: Vec<_> = metrics.value["assertions"]
            .as_array()
            .into_iter()
            .flatten()
            .take(1000)
            .map(|v| {
                selected(
                    v,
                    &[
                        "id",
                        "value",
                        "unit",
                        "numerator",
                        "denominator",
                        "status",
                        "evidence_sha256",
                        "source_field",
                    ],
                )
            })
            .collect();
        let archive_roots: Vec<_> = manifest.value["evidence_roots"]
            .as_array()
            .into_iter()
            .flatten()
            .take(64)
            .map(|r| {
                let mut value = selected(
                    r,
                    &[
                        "bytes",
                        "codec",
                        "member_sha256",
                        "physical_version",
                        "rows",
                        "sha256",
                        "version",
                    ],
                );
                let matching = artifacts.value["artifacts"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .find(|a| a["sha256"].is_string() && a["sha256"] == r["sha256"]);
                value["s3_uri"] = matching
                    .and_then(|a| a.get("s3_uri"))
                    .cloned()
                    .unwrap_or(Value::Null);
                value
            })
            .collect();
        result.push(json!({"id":name,"card":selected(&card.value,&["wave_id","version","stage","capture_status","archive_status","disposition","interpretation"]),"assertions":assertions,"manifest_sha256":manifest.sha256,"metrics_sha256":metrics.sha256,"source_commit":manifest.value["source_commit"],"training_identity_sha256":manifest.value["training_identity_sha256"],"archive_manifest_sha256":manifest.value["archive_manifest_sha256"],"archive_roots":archive_roots,"publication":"files observed in configured repository snapshot; Git publication requires commit receipt","archive":"producer claim; dashboard does not independently restore bulk payloads"}));
    }
    json!({"waves":result,"scope":"compact recorded assertions, not raw result rows or verified scientific wins"})
}
