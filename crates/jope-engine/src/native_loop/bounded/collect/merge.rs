use super::*;

/// Freshness window applied to every per-host document; the same bounds as a plan's telemetry
/// policy (`max_age_ms` 1..=600000, `clock_skew_ms` <= 10000).
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MergePolicy {
    pub max_age_ms: u64,
    pub clock_skew_ms: u64,
}
/// One per-host document and the SHA-256 its collector reported for it.
#[derive(Clone, Debug)]
pub struct PinnedDocument {
    pub label: String,
    pub bytes: Vec<u8>,
    pub sha256: String,
}

/// Combine pinned per-host census documents into one multi-host census.
pub fn merge(
    documents: &[PinnedDocument],
    policy: MergePolicy,
    now_unix_ms: u64,
) -> Result<(HostCensus, MergeReport)> {
    ensure!(
        (1..=600_000).contains(&policy.max_age_ms) && policy.clock_skew_ms <= 10_000,
        "merge freshness policy must be max_age_ms 1..=600000 and clock_skew_ms <= 10000"
    );
    ensure!(
        !documents.is_empty() && documents.len() <= 64,
        "merge takes 1..=64 per-host documents"
    );
    // Pin every document before interpreting any of it.
    let mut digests = BTreeSet::new();
    for d in documents {
        ensure!(
            is_digest(&d.sha256),
            "document {} pin must be 64 lowercase hex",
            d.label
        );
        ensure!(
            d.bytes.len() <= MAX_CENSUS_BYTES,
            "document {} exceeds {MAX_CENSUS_BYTES} bytes",
            d.label
        );
        let actual = hash(&d.bytes);
        ensure!(
            actual == d.sha256,
            "document {} was altered: its SHA-256 {actual} differs from the pinned {}",
            d.label,
            d.sha256
        );
        ensure!(
            digests.insert(actual),
            "document {} is given twice",
            d.label
        );
    }
    let mut schemas = BTreeSet::new();
    for d in documents {
        let v: serde_json::Value = serde_json::from_slice(&d.bytes)
            .with_context(|| format!("document {} is not JSON", d.label))?;
        schemas.insert(
            v.get("schema")
                .and_then(|s| s.as_str())
                .unwrap_or("<none>")
                .to_string(),
        );
    }
    ensure!(
        schemas.len() == 1 && schemas.contains(CENSUS_SCHEMA),
        "mixed or unsupported census schema versions {schemas:?}; every document must be `{CENSUS_SCHEMA}`"
    );
    let mut hosts: Vec<(HostObservation, MergedDocument)> = Vec::new();
    let mut captured = 0u64;
    let mut seen: BTreeMap<&str, BTreeSet<String>> = BTreeMap::new();
    for d in documents {
        let census =
            HostCensus::from_bytes(&d.bytes).with_context(|| format!("document {}", d.label))?;
        ensure!(
            census.hosts.len() == 1,
            "document {} must describe exactly one host",
            d.label
        );
        let h = census.hosts.into_iter().next().context("host")?;
        for t in [census.captured_unix_ms, h.observed_unix_ms] {
            ensure!(
                t <= now_unix_ms.saturating_add(policy.clock_skew_ms),
                "document {} is dated {} ms in the future",
                d.label,
                t - now_unix_ms
            );
            ensure!(
                now_unix_ms.saturating_sub(t) <= policy.max_age_ms,
                "document {} is stale: {} ms old, beyond {} ms",
                d.label,
                now_unix_ms - t,
                policy.max_age_ms
            );
        }
        let i = &h.identity;
        for (what, value) in [
            ("host", i.host.clone()),
            ("machine id", i.machine_id.clone()),
            ("boot id", i.boot_id.clone()),
            ("address", i.address.clone()),
            ("certificate", i.certificate_sha256.clone()),
        ] {
            ensure!(
                seen.entry(what).or_default().insert(value.clone()),
                "document {} duplicates the {what} `{value}` of another document",
                d.label
            );
        }
        captured = captured.max(census.captured_unix_ms);
        hosts.push((
            h.clone(),
            MergedDocument {
                host: h.identity.host.clone(),
                sha256: d.sha256.clone(),
                observed_unix_ms: h.observed_unix_ms,
            },
        ));
    }
    hosts.sort_by(|a, b| a.0.identity.host.cmp(&b.0.identity.host));
    let merged = HostCensus {
        schema: CENSUS_SCHEMA.into(),
        captured_unix_ms: captured,
        hosts: hosts.iter().map(|(h, _)| h.clone()).collect(),
    };
    merged.validate()?;
    let bytes = serde_json::to_vec(&merged)?;
    ensure!(bytes.len() <= MAX_CENSUS_BYTES, "merged census bound");
    let report = MergeReport {
        schema: MERGE_SCHEMA.into(),
        census_sha256: hash(&bytes),
        merged_unix_ms: now_unix_ms,
        max_age_ms: policy.max_age_ms,
        clock_skew_ms: policy.clock_skew_ms,
        documents: hosts.into_iter().map(|(_, d)| d).collect(),
        allocation_held: false,
    };
    Ok((merged, report))
}

/// Publish census bytes at `out` without replacement (an existing different file is refused
/// and kept). The parent directory must already exist.
pub fn write_census(out: &Path, census: &HostCensus) -> Result<Vec<u8>> {
    ensure!(out.is_absolute(), "--out must be an absolute path");
    let parent = out.parent().context("--out parent")?;
    let name = out.file_name().context("--out file name")?;
    ensure!(
        fs::symlink_metadata(parent)
            .map(|m| m.is_dir())
            .unwrap_or(false),
        "--out parent {} must be an existing directory",
        parent.display()
    );
    ensure!(
        fs::symlink_metadata(out).is_err(),
        "{} already exists; a census is never replaced",
        out.display()
    );
    let bytes = serde_json::to_vec(census)?;
    super::super::rendezvous::publish_immutable(parent, Path::new(name), &bytes)?;
    Ok(bytes)
}
