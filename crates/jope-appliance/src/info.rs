use super::*;

pub(super) fn version_value() -> Value {
    let package = Package::open_self();
    let package = match &package {
        Ok(Some(p)) => json!(bootstrap::summary(p)),
        Ok(None) => Value::Null,
        Err(error) => json!({"error": error.to_string()}),
    };
    json!({
        "product": "jopedime",
        "version": VERSION,
        "source_sha40": bootstrap::source_sha40(),
        "runtime_abi": RUNTIME_ABI,
        "package_format": jope_assets::MANIFEST_SCHEMA,
        "target": env!("JOPE_BUILD_TARGET"),
        "features": {
            "engine": cfg!(feature = "engine"),
            "mcp": cfg!(feature = "mcp"),
            "client": cfg!(feature = "client"),
            "embedded": bootstrap::EMBEDDED,
            "dev_keys": jope_assets::verify::TRUSTS_DEVELOPMENT_KEY,
            "cuda": backend::CUDA_COMPILED,
            "dope_cuda": cfg!(feature = "dope-cuda"),
        },
        "trusts_development_key": jope_assets::verify::TRUSTS_DEVELOPMENT_KEY,
        "package": package,
    })
}

fn unexecutable(promised: &[String]) -> anyhow::Result<Vec<(String, String)>> {
    #[cfg(feature = "engine")]
    {
        let mut blocked = jope_service::proposal::uncompiled(promised)?;
        if promised
            .iter()
            .any(|id| id == jope_service::proposal::METHOD)
        {
            let package = Package::open_self().map_err(bootstrap::integrity)?;
            let record = jope_service::proposal::record(package.as_ref())
                .map_err(|error| Failure::integrity(format!("{error:#}")))?;
            if !record.executable {
                blocked.push((record.id, record.reason.unwrap_or_default()));
            }
        }
        Ok(blocked)
    }
    #[cfg(not(feature = "engine"))]
    methods::unexecutable(promised)
}

pub(super) fn doctor() -> anyhow::Result<(Value, bool)> {
    let started = Instant::now();
    let (package, promised, package_ok) = match Package::open_self() {
        Ok(Some(p)) => {
            let summary = bootstrap::summary(&p);
            let methods = summary.methods.clone();
            (
                json!({"status": "authenticated", "summary": summary}),
                methods,
                true,
            )
        }
        Ok(None) if bootstrap::EMBEDDED => (
            json!({"status": "missing", "reason": "release build without its package"}),
            Vec::new(),
            false,
        ),
        Ok(None) => (json!({"status": "unpackaged"}), Vec::new(), true),
        Err(error) => (
            json!({"status": "refused", "reason": error.to_string()}),
            Vec::new(),
            false,
        ),
    };
    let unexecutable: Vec<Value> = unexecutable(&promised)?
        .into_iter()
        .map(|(id, reason)| json!({"id": id, "reason": reason}))
        .collect();
    let ok = package_ok && unexecutable.is_empty();
    let value = json!({
        "ok": ok,
        "version": VERSION,
        "package": package,
        "promised_methods": promised,
        "unexecutable_promised_methods": unexecutable,
        "devices": backend::probe(),
        "native_device": "cpu",
        "dope_cuda_compiled": cfg!(feature = "dope-cuda"),
        "limits": Limits::default(),
        "cache": "none: packaged assets are authenticated and decoded in memory when a method \
                  needs them; this build keeps no derived cache",
        "worker_processes": "none: fitting runs in this process with a cooperative deadline",
        "seconds": started.elapsed().as_secs_f64(),
    });
    Ok((value, ok))
}

pub(super) fn methods_value(task_filter: Option<&str>) -> anyhow::Result<Value> {
    if let Some(name) = task_filter {
        if !matches!(name, "regression" | "binary" | "multiclass") {
            return Err(Failure::input(format!("unknown task {name:?}")));
        }
    }
    let promised: Vec<String> = match Package::open_self() {
        Ok(Some(p)) => p.manifest().methods.clone(),
        Ok(None) => Vec::new(),
        Err(error) => return Err(bootstrap::integrity(error)),
    };
    let records = methods::records()?;
    #[cfg(feature = "engine")]
    let records = {
        let mut records = records;
        let package = Package::open_self().map_err(bootstrap::integrity)?;
        records.push(
            jope_service::proposal::record(package.as_ref())
                .map_err(|error| Failure::integrity(format!("{error:#}")))?,
        );
        records
    };
    let records: Vec<Value> = records
        .into_iter()
        .filter(|r| task_filter.is_none_or(|t| r.task == t))
        .map(|r| {
            let mut value = json!(r);
            value["promised_by_package"] = json!(promised.contains(&r.id));
            value
        })
        .collect();
    Ok(json!({
        "auto": {
            "id": methods::AUTO,
            "policy": jope_runtime::automl::SELECTION_POLICY,
            "fold_policy": jope_runtime::automl::FOLD_POLICY,
            "default_folds": jope_runtime::automl::DEFAULT_FOLDS,
            "metrics": {"regression": "rmse", "binary": "auroc", "multiclass": "log_loss"},
        },
        "methods": records,
    }))
}

pub(super) fn licenses(json_out: bool) -> anyhow::Result<()> {
    let package = bootstrap::open()?;
    let mut notices = Vec::new();
    let mut assets = Vec::new();
    if let Some(package) = &package {
        for entry in &package.manifest().entries {
            assets.push(json!({
                "id": entry.id,
                "origin": entry.origin,
                "spdx": entry.license.spdx,
                "redistribution": entry.license.redistribution,
            }));
            if entry.role == "notice" {
                let bytes = package
                    .read_entry(&entry.id)
                    .map_err(bootstrap::integrity)?;
                notices.push((
                    entry.id.clone(),
                    String::from_utf8_lossy(bytes.bytes()).into_owned(),
                ));
            }
        }
    }
    if json_out {
        print_json(&json!({
            "product_license": "Apache-2.0",
            "packaged": package.is_some(),
            "assets": assets,
            "notices": notices.iter().map(|(id, text)| json!({"id": id, "text": text})).collect::<Vec<_>>(),
        }));
    } else {
        out(&format!(
            "jopedime {VERSION} is licensed under Apache-2.0.\n"
        ));
        if package.is_none() {
            out("This is an unpackaged development build: it embeds no notices.\n");
        }
        for (id, text) in notices {
            out(&format!("\n==> {id} <==\n{text}\n"));
        }
    }
    Ok(())
}

pub(super) fn verify_self(json_out: bool) -> anyhow::Result<()> {
    let package = Package::open_self()
        .map_err(bootstrap::integrity)?
        .ok_or_else(|| Failure::integrity("this executable has no package to verify"))?;
    let report = package.verify_full().map_err(bootstrap::integrity)?;
    let value = json!({
        "ok": true,
        "summary": bootstrap::summary(&package),
        "report": report,
        "note": "verify-self detects corruption; it cannot vouch for an executable that may \
                 itself be modified. Verify a download with `jope-dist verify` or SHA256SUMS.",
    });
    if json_out {
        print_json(&value);
    } else {
        eprintln!(
            "verified {} bytes: executable, {} assets, manifest, signature ({})",
            report.file_bytes,
            report.entries,
            package.signer().id
        );
    }
    Ok(())
}
