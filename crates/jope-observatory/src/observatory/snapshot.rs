use super::*;

/// Assemble the configured read-only observatory snapshot.
pub fn snapshot(c: &Config) -> Value {
    let s3 = storage(c);
    let catalog = observe(&c.repo_root.join("catalogs/s3.json"));
    let mut datasets: Vec<_>=catalog.value["buckets"].as_array().into_iter().flatten().filter(|b|b["catalog_v1"].is_object()).map(|b|json!({"name":b["bucket"],"count":b["catalog_v1"]["datasets"],"license_status":b["catalog_v1"]["license_status"],"origins":b["catalog_v1"]["origins"],"sha256":b["catalog_v1"]["sha256"],"observed_at":catalog.value["inventoried"],"status":"historical","scope":"dataset IDs in versioned catalog; source data is not read by dashboard"})).collect();
    if let Some(path) = &c.dataset_snapshot {
        let current = observe(path);
        if current.value["schema"] == "jope.dataset-observation/v1" {
            if let Some(rows) = current.value["datasets"].as_array() {
                datasets = rows
                    .iter()
                    .take(32)
                    .map(|r| {
                        selected(
                            r,
                            &[
                                "name",
                                "count",
                                "license_status",
                                "origins",
                                "sha256",
                                "version_id",
                                "catalog_modified_at",
                                "observed_at",
                                "status",
                                "scope",
                            ],
                        )
                    })
                    .collect();
            }
        }
    }
    let fleet: Vec<_> = c
        .fleet_snapshots
        .iter()
        .take(4)
        .map(|p| {
            let mut o = observe(p);
            let valid = o.value["schema"] == "jope.fleet-node-observation/v1";
            o.value = if valid {
                selected(
                    &o.value,
                    &[
                        "schema",
                        "host",
                        "observed_at",
                        "status",
                        "cpu_count",
                        "load",
                        "memory_total_bytes",
                        "memory_available_bytes",
                        "disk_available_bytes",
                        "gpus",
                        "workload_scope",
                    ],
                )
            } else {
                Value::Null
            };
            if !valid {
                o.status = "unqualified".into();
            }
            json!(o)
        })
        .collect();
    let mut runs: Vec<_> = c
        .campaign_roots
        .iter()
        .take(MAX_RUNS)
        .map(|r| campaign(r))
        .collect();
    let mut identity = c.code_identity.as_ref().map(|p| observe(p));
    if let Some(o) = &mut identity {
        if o.value["schema"] == "jope.code-identity-observation/v1" {
            for r in &mut runs {
                if let Some(s) = o.value["training_runs"].get(r["id"].as_str().unwrap_or_default())
                {
                    r["source_identity"] = selected(
                        s,
                        &["source_commit", "verified_files", "receipt_sha256", "scope"],
                    );
                }
            }
            o.value = selected(
                &o.value,
                &[
                    "schema",
                    "observed_at",
                    "dashboard_commit",
                    "dashboard_dirty",
                    "training_runs",
                    "scope",
                ],
            );
        } else {
            o.status = "unqualified".into();
            o.value = Value::Null;
        }
    }
    json!({"schema":SCHEMA,"generated_at":now(),"service":{"version":env!("CARGO_PKG_VERSION"),"mode":"read-only","refresh_seconds":5,"production_authority":false},"code_identity":identity,"storage":s3,"datasets":datasets,"runs":runs,"gp":gp_catalog(&c.gp_catalogs),"research":waves(&c.repo_root),"fleet":fleet,"limits":{"max_runs":MAX_RUNS,"max_json_bytes":MAX_JSON,"bulk_payloads_read":false,"credentials_read":false,"note":"Read-only observations never admit work, promote a model, or establish verified self-play gains."}})
}
