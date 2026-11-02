use super::*;

pub(super) fn publication_status(event: &str, receipt: &Value) -> Result<Option<Value>> {
    ensure!(
        receipt["schema"] == "jain.datasets_service_receipt.v1",
        "unsupported winner publication receipt"
    );
    let captures = receipt["captures"]
        .as_array()
        .context("capture rows absent")?;
    ensure!(
        captures.len() <= 4096,
        "capture metadata count exceeds bound"
    );
    let matching = captures
        .iter()
        .filter(|c| c["evaluation"] == format!("champion-event:{event}"))
        .collect::<Vec<_>>();
    ensure!(
        matching.len() <= 1,
        "winner appears twice in one capture receipt"
    );
    let Some(capture) = matching.first() else {
        return Ok(None);
    };
    ensure!(
        capture["acknowledged"] == true,
        "winner capture has no durable acknowledgement"
    );
    let publications = receipt["publications"]
        .as_array()
        .context("publication rows absent")?;
    ensure!(
        publications.len() <= 4096,
        "publication count exceeds bound"
    );
    let matching = publications
        .iter()
        .filter(|p| {
            p["producer"] == capture["producer"]
                && p["sequence"] == capture["sequence"]
                && p["envelope_sha256"] == capture["envelope_sha256"]
        })
        .collect::<Vec<_>>();
    ensure!(
        matching.len() <= 1,
        "conflicting winner publication bindings"
    );
    let Some(p) = matching.first() else {
        return Ok(Some(
            json!({"local_commit":null,"status":"original capture acknowledged; local commit pending","remote_verified_recorded":null,"capture_sequence":capture["sequence"]}),
        ));
    };
    let chain_ref = Artifact {
        path: p["chain_manifest_path"]
            .as_str()
            .context("chain path absent")?
            .into(),
        sha256: p["chain_manifest_sha256"]
            .as_str()
            .context("chain hash absent")?
            .into(),
    };
    let chain = checked(None, &chain_ref, 16 << 20)?;
    ensure!(
        chain["schema"] == "jain.publication.chain.v1"
            && chain["producer"] == capture["producer"]
            && chain["sequence"] == capture["sequence"]
            && chain["envelope_sha256"] == capture["envelope_sha256"]
            && chain["final_event_commit"] == p["final_event_commit"]
            && chain["final_event_ref"] == p["final_event_ref"],
        "original winner chain identity differs"
    );
    let commit = chain["final_event_commit"]
        .as_str()
        .context("local commit absent")?;
    ensure!(
        [40, 64].contains(&commit.len()) && commit.bytes().all(|b| b.is_ascii_hexdigit()),
        "invalid recorded commit identity"
    );
    let published = chain_ref.path.with_file_name("published-chain.json");
    let mut verified = false;
    if published.exists() {
        let v: Value = serde_json::from_slice(&regular(&published, 16 << 20)?)?;
        ensure!(
            v["schema"] == "jain.publication.chain_receipt.v1"
                && v["verified_download"] == true
                && v["chain_manifest_sha256"] == chain_ref.sha256
                && v["producer"] == capture["producer"]
                && v["sequence"] == capture["sequence"]
                && v["envelope_sha256"] == capture["envelope_sha256"]
                && v["final_event_commit"] == commit
                && v["final_event_ref"] == chain["final_event_ref"],
            "recorded remote chain verification differs"
        );
        let parts = v["parts"]
            .as_array()
            .context("remote part receipts absent")?;
        let batches = chain["batches"]
            .as_array()
            .context("original dependency batches absent")?;
        ensure!(
            !parts.is_empty() && parts.len() == batches.len() && parts.len() <= 4096,
            "remote dependency restoration count differs"
        );
        for (part, batch) in parts.iter().zip(batches) {
            ensure!(
                part["schema"] == "jain.publication.receipt.v1"
                    && part["verified_download"] == true
                    && part["commit"] == batch["commit"]
                    && part["batch_sha256"] == batch["batch_sha256"],
                "remote dependency restoration identity differs"
            );
            original_time(
                Some(
                    part["verified_at_unix_ms"]
                        .as_u64()
                        .context("original remote verification timestamp absent")?,
                ),
                now()?,
            )?;
        }
        verified = true;
    }
    Ok(Some(
        json!({"local_commit":commit,"status":if verified{"recorded original remote restoration; no fresh forge probe"}else{"recorded local commit; remote verification pending"},
        "remote_verified_recorded":verified,"capture_sequence":capture["sequence"]}),
    ))
}
pub(super) fn winners(id: &str, root: &Path, outbox: &Path, receipts: &[Artifact]) -> Result<Rows> {
    ensure!(outbox.starts_with(root), "winner outbox outside root");
    let all = bounded_files(&outbox.join("by-sequence"), 100000)?;
    ensure!(
        receipts.len() <= 64,
        "winner service receipt bound exceeded"
    );
    let receipts = receipts
        .iter()
        .map(|a| checked(None, a, MAX_METADATA).map(|v| (a.clone(), v)))
        .collect::<Result<Vec<_>>>()?;
    let mut rows = Rows::default();
    rows.notes.push(text(format!("{id}: ordered winner outbox {} identities; displaying latest16; recorded publication never implies fresh forge readback",all.len())));
    for path in all.into_iter().rev().take(16) {
        let name = path
            .file_stem()
            .and_then(|v| v.to_str())
            .context("winner sequence name")?;
        ensure!(
            name.len() == 20 && name.bytes().all(|b| b.is_ascii_digit()),
            "invalid winner sequence identity"
        );
        let digest: String = serde_json::from_slice(&regular(&path, 1024)?)?;
        let a = Artifact {
            path: outbox.join(format!("{name}-{digest}.json")),
            sha256: digest.clone(),
        };
        let raw = checked(None, &a, 1 << 20)?;
        let c: crate::champions::Champion = serde_json::from_value(raw)?;
        c.validate()?;
        ensure!(
            matches!(
                c.identity.partition.as_str(),
                "meta_train" | "meta_validation"
            ),
            "sealed winner remains opaque"
        );
        let back: u64 = serde_json::from_slice(&regular(
            &outbox.join("by-event").join(format!("{digest}.json")),
            1024,
        )?)?;
        ensure!(
            back == name.parse::<u64>()?,
            "winner ordering reverse identity differs"
        );
        let original = c
            .timestamp
            .checked_mul(1000)
            .context("winner timestamp overflow")?;
        original_time(Some(original), now()?)?;
        let mut row = json!({"identity":digest,"local_commit":null,"publication_status":"original ordered winner; matching publication receipt unavailable","remote_verified":null,
            "age_seconds":now()?.saturating_sub(original)/1000,"sequence":back,"original_at_unix_ms":original,"scientific_verified_at_unix_ms":now()?,"source":a,"verification_kind":"champion identity/hash/order checked; authority provenance not independently replayed"});
        for (source, receipt) in &receipts {
            if let Some(status) = publication_status(&digest, receipt)? {
                row["local_commit"] = status["local_commit"].clone();
                row["publication_status"] = status["status"].clone();
                row["remote_verified_recorded"] = status["remote_verified_recorded"].clone();
                row["publication_source"] = serde_json::to_value(source)?;
                row["capture_sequence"] = status["capture_sequence"].clone();
            }
        }
        rows.winners.push(row);
    }
    Ok(rows)
}
pub(super) fn catalog() -> Result<Catalog> {
    let bytes = crate::eval_registry::native_scq::policy::protected(Path::new(CATALOG))?;
    ensure!(bytes.len() <= 1 << 20, "science catalog exceeds1MiB");
    let catalog: Catalog = serde_json::from_slice(&bytes)?;
    ensure!(
        catalog.schema == "jope.native_scientific_catalog.v1" && catalog.entries.len() <= 64,
        "science catalog schema/count invalid"
    );
    let mut ids = BTreeSet::new();
    for source in &catalog.entries {
        ensure!(
            !source.id().is_empty() && text(source.id()) == source.id() && ids.insert(source.id()),
            "science source identity duplicate/invalid"
        );
    }
    Ok(catalog)
}
