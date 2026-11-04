//! Independently signed, scoped compatibility views. Original preparation bytes never change.
use crate::{
    corpus::ContextRow,
    manifest::{hash, resolve},
    native_genome::{self, Candidate},
    native_graph::Artifact,
    prepare::Prepared,
};
use anyhow::{ensure, Context, Result};
use ed25519_dalek::{Signature, VerifyingKey};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    io::Read,
    os::unix::fs::{MetadataExt, OpenOptionsExt},
    path::{Path, PathBuf},
};
const DOMAIN: &[u8] = b"jope.scoped_preparation_overlay.signature.v1\0";
const SCOPE: &str = "xbabe2-first-window-60000-66000";
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignedOverlay {
    pub schema: String,
    pub authorization_public_key: String,
    pub body_sha256: String,
    pub body: Value,
    pub signature: String,
}
pub struct VerifiedOverlay {
    body: Value,
    pub not_before_unix_ms: u64,
    pub expires_unix_ms: u64,
}
fn canonical(value: &Value) -> Value {
    match value {
        Value::Object(m) => Value::Object(
            m.iter()
                .map(|(k, v)| (k.clone(), canonical(v)))
                .collect::<BTreeMap<_, _>>()
                .into_iter()
                .collect(),
        ),
        Value::Array(a) => Value::Array(a.iter().map(canonical).collect()),
        v => v.clone(),
    }
}
fn signature(s: &SignedOverlay, pinned: &str) -> Result<()> {
    ensure!(
        s.schema == "jope.scoped_preparation_overlay.v1" && s.authorization_public_key == pinned,
        "overlay signer is not the independently pinned review key"
    );
    let b = serde_json::to_vec(&canonical(&s.body))?;
    ensure!(hash(&b) == s.body_sha256, "overlay canonical body differs");
    let key = VerifyingKey::from_bytes(
        &hex::decode(pinned)?
            .try_into()
            .map_err(|_| anyhow::anyhow!("overlay key width"))?,
    )?;
    ensure!(!key.is_weak(), "weak overlay key");
    let mut message = DOMAIN.to_vec();
    message.extend(b);
    key.verify_strict(
        &message,
        &Signature::from_slice(&hex::decode(&s.signature)?)?,
    )?;
    Ok(())
}
fn checked_hash(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    let mut f = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC)
        .open(path)?;
    let a = f.metadata()?;
    ensure!(
        a.is_file() && a.len() <= 256 * 1024 * 1024,
        "bounded regular overlay dependency required"
    );
    let mut sha = Sha256::new();
    let mut buf = vec![0u8; 1024 * 1024];
    let mut bytes = 0u64;
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        bytes += n as u64;
        ensure!(bytes <= a.len(), "overlay dependency grew");
        sha.update(&buf[..n]);
    }
    let z = f.metadata()?;
    let named = std::fs::symlink_metadata(path)?;
    ensure!(
        bytes == a.len()
            && z.len() == a.len()
            && z.ctime() == a.ctime()
            && z.ctime_nsec() == a.ctime_nsec()
            && z.mtime() == a.mtime()
            && z.mtime_nsec() == a.mtime_nsec()
            && named.dev() == a.dev()
            && named.ino() == a.ino()
            && !named.file_type().is_symlink(),
        "overlay dependency changed during verification"
    );
    Ok(hex::encode(sha.finalize()))
}
fn artifact(v: &Value) -> Result<Artifact> {
    let path: PathBuf = v["path"]
        .as_str()
        .context("overlay artifact path absent")?
        .into();
    crate::manifest::relative(&path)?;
    let sha256 = v["sha256"]
        .as_str()
        .context("overlay artifact hash absent")?
        .to_owned();
    ensure!(
        sha256.len() == 64
            && sha256
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
        "overlay artifact hash invalid"
    );
    Ok(Artifact { path, sha256 })
}
fn collect(v: &Value, out: &mut BTreeMap<PathBuf, String>) -> Result<()> {
    match v {
        Value::Object(m) => {
            if m.contains_key("path") || m.contains_key("sha256") {
                // Only explicit artifact pairs are dependencies; bare provenance hashes retain their signed identity.
                if m.contains_key("path") && m.contains_key("sha256") {
                    let a = artifact(v)?;
                    if let Some(old) = out.insert(a.path, a.sha256.clone()) {
                        ensure!(old == a.sha256, "conflicting overlay artifact identity");
                    }
                }
            }
            for x in m.values() {
                collect(x, out)?;
            }
        }
        Value::Array(a) => {
            for x in a {
                collect(x, out)?;
            }
        }
        _ => {}
    }
    ensure!(out.len() <= 4096, "overlay dependency bound");
    Ok(())
}
/// Hash-bound portable input identities; callers include these in the full native closure.
pub(crate) fn dependencies(root: &Path, reference: &Artifact) -> Result<BTreeMap<PathBuf, String>> {
    let signed: SignedOverlay = native_genome::load(root, reference)?;
    let mut result = BTreeMap::from([(reference.path.clone(), reference.sha256.clone())]);
    collect(&signed.body, &mut result)?;
    for (p, h) in &result {
        ensure!(
            checked_hash(&resolve(root, p)?)? == *h,
            "overlay dependency changed: {}",
            p.display()
        );
    }
    Ok(result)
}
impl VerifiedOverlay {
    pub fn require_live(&self, now_ms: u64, deadline_ms: u64) -> Result<()> {
        ensure!(
            self.not_before_unix_ms <= now_ms
                && now_ms < self.expires_unix_ms
                && now_ms < deadline_ms
                && deadline_ms <= self.expires_unix_ms,
            "overlay original authorization window expired or task exceeds it"
        );
        Ok(())
    }
    pub fn views(&self) -> BTreeSet<(String, usize)> {
        self.body["descriptors"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|d| {
                [0, 1]
                    .into_iter()
                    .map(move |cut| (d["dataset"].as_str().unwrap().to_owned(), cut))
            })
            .collect()
    }
    pub fn verify_context(&self, root: &Path, c: &ContextRow, p: &Prepared) -> Result<()> {
        ensure!(
            c.partition == "meta_train"
                && p.meta_partition == "meta_train"
                && p.task == "regression"
                && p.clean_evaluation
                && !p.qualification_only
                && p.family == c.family
                && p.dataset == c.dataset,
            "overlay cannot admit held-out, unclean or qualification-only preparation"
        );
        let d = self.body["descriptors"]
            .as_array()
            .unwrap()
            .iter()
            .find(|d| d["dataset"] == c.dataset)
            .context("unlisted overlay dataset")?;
        let b = &d["body"];
        ensure!(
            b["preserved_family"] == c.family
                && b["preserved_partition"] == "meta_train"
                && !self.body["withdrawn_families"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|f| f == &Value::String(c.family.clone())),
            "overlay family mismatch or withdrawn family"
        );
        let original = artifact(&b["original_prepared"])?;
        ensure!(
            original.path == c.prepared.join("prepared.json")
                && original.sha256 == c.prepared_sha256
                && checked_hash(&resolve(root, &original.path)?)? == original.sha256
                && b["original_train_sha256"] == p.original_train_sha256
                && b["original_test_sha256"] == p.original_test_sha256,
            "overlay original preparation identity differs"
        );
        let storage = p
            .storage
            .as_ref()
            .context("overlay original storage absent")?;
        let source = artifact(&b["original_storage_source"])?;
        ensure!(
            source.path == storage.source && source.sha256 == storage.source_sha256,
            "overlay storage identity differs"
        );
        let cut = p
            .cuts
            .iter()
            .find(|x| x.index == c.cut)
            .context("original cut absent")?;
        let selected = b["cuts"]
            .as_array()
            .context("overlay cuts absent")?
            .iter()
            .find(|x| x["index"] == c.cut)
            .context("unlisted overlay cut")?;
        ensure!(
            c.cut <= 1
                && selected["preprocessing_sha256"] == cut.preprocessing_sha256
                && selected["training_sha256"] == cut.training_sha256
                && selected["validation_sha256"] == cut.validation_sha256,
            "overlay original cut hashes differ"
        );
        let view = b["context_views"]
            .as_array()
            .context("overlay views absent")?
            .iter()
            .find(|x| x["cut"] == c.cut)
            .context("unlisted overlay tensor")?;
        let tensor = artifact(&view["tensor"])?;
        ensure!(
            tensor.path == c.path
                && tensor.sha256 == c.sha256
                && view["original_prepared_sha256"] == c.prepared_sha256
                && checked_hash(&resolve(root, &tensor.path)?)? == tensor.sha256,
            "overlay tensor identity differs"
        );
        Ok(())
    }
}
fn shape(body: Value, run: &str) -> Result<VerifiedOverlay> {
    ensure!(
        body["schema"] == "jope.scoped_preparation_overlay_body.v1"
            && body["scope"] == SCOPE
            && body["run"] == run
            && !run.is_empty(),
        "overlay schema/run/scope differs"
    );
    ensure!(
        body["optimizer_window"]
            == serde_json::json!({"blocks":2,"global_batch":2048,"initial_step":60000,"target_step":66000,"updates_per_block":3000})
            && body["cuts"] == serde_json::json!([0, 1]),
        "overlay cannot change original first optimizer window"
    );
    let start = body["validity"]["not_before_unix_ms"]
        .as_u64()
        .context("overlay not-before absent")?;
    let end = body["validity"]["not_after_unix_ms"]
        .as_u64()
        .context("overlay expiry absent")?;
    ensure!(
        start < end && end - start <= 7 * 24 * 60 * 60 * 1000,
        "overlay window invalid or unbounded"
    );
    let descriptors = body["descriptors"]
        .as_array()
        .context("overlay descriptors absent")?;
    ensure!(
        descriptors.len() == 48,
        "first overlay requires exact48 reviewed datasets"
    );
    let mut datasets = BTreeSet::new();
    let mut prior = "";
    for d in descriptors {
        let dataset = d["dataset"].as_str().context("overlay dataset absent")?;
        ensure!(
            prior < dataset && datasets.insert(dataset),
            "overlay datasets unordered/duplicate"
        );
        prior = dataset;
        let b = &d["body"];
        ensure!(
            hash(&serde_json::to_vec(&canonical(b))?)
                == d["descriptor_sha256"]
                    .as_str()
                    .context("descriptor hash absent")?
                && b["dataset"] == dataset
                && b["schema"] == "jope.prepared_compatibility_view.v1"
                && b["scope"] == SCOPE
                && b["preserved_partition"] == "meta_train"
                && b["original_global_inventory_frozen"] == false
                && b["original_metadata_unchanged"] == true
                && b["final_test_evaluation_qualified"] == false,
            "overlay descriptor provenance differs"
        );
        for (key, field) in [("cuts", "index"), ("context_views", "cut")] {
            let list = b[key].as_array().context("descriptor views absent")?;
            ensure!(
                list.len() == 2
                    && list
                        .iter()
                        .filter_map(|v| v[field].as_u64())
                        .collect::<BTreeSet<_>>()
                        == BTreeSet::from([0, 1]),
                "descriptor must bind exact cuts0/1"
            );
        }
    }
    ensure!(
        body["partition_provenance"]["full_inventory_complete"] == false
            && body["family_similarity_exception"]["never_clean_unseen_family_heldout_for_lineage"]
                == true,
        "overlay must preserve incomplete inventory and related-family disclosure"
    );
    let withdrawn = body["withdrawn_families"]
        .as_array()
        .context("withdrawn families absent")?;
    ensure!(
        withdrawn
            .iter()
            .filter_map(Value::as_str)
            .collect::<BTreeSet<_>>()
            == BTreeSet::from(["rdatasets-cardata-leinhardt", "rdatasets-cardata-prestige"]),
        "withdrawn families changed"
    );
    for v in [
        &body["context_index"],
        &body["decision"],
        &body["verification"]["claude_context_verifier"],
        &body["verification"]["verify_prepared"],
    ] {
        artifact(v)?;
    }
    Ok(VerifiedOverlay {
        body,
        not_before_unix_ms: start,
        expires_unix_ms: end,
    })
}
/// Historical verification deliberately does not use current wall-clock time.
pub fn load(root: &Path, reference: &Artifact, run: &str) -> Result<VerifiedOverlay> {
    let signed: SignedOverlay = native_genome::load(root, reference)?;
    crate::native_review::require_signer(&signed.authorization_public_key)?;
    load_pinned(root, reference, run, &signed.authorization_public_key)
}
fn load_pinned(
    root: &Path,
    reference: &Artifact,
    run: &str,
    pinned: &str,
) -> Result<VerifiedOverlay> {
    let signed: SignedOverlay = native_genome::load(root, reference)?;
    signature(&signed, pinned)?;
    let verified = shape(signed.body, run)?;
    dependencies(root, reference)?;
    Ok(verified)
}
/// Actual ordered numerical training identities for the reviewed96views; no fits or final-test reads.
pub fn training_identities(
    root: &Path,
    reference: &Artifact,
    run: &str,
) -> Result<BTreeSet<String>> {
    let overlay = load(root, reference, run)?;
    let mut hashes = BTreeSet::new();
    for descriptor in overlay.body["descriptors"].as_array().unwrap() {
        let b = &descriptor["body"];
        let a = artifact(&b["original_prepared"])?;
        let p: Prepared = native_genome::load(root, &a)?;
        for view in b["context_views"].as_array().unwrap() {
            let tensor = artifact(&view["tensor"])?;
            let c = ContextRow {
                dataset: p.dataset.clone(),
                family: p.family.clone(),
                cut: view["cut"].as_u64().context("overlay cut")? as usize,
                partition: p.meta_partition.clone(),
                path: tensor.path,
                sha256: tensor.sha256,
                prepared: a.path.parent().context("prepared parent")?.to_owned(),
                prepared_sha256: a.sha256.clone(),
            };
            overlay.verify_context(root, &c, &p)?;
            hashes.insert(crate::label_admission::verify_inputs(root, &c, &p)?);
        }
    }
    ensure!(
        hashes.len() == 96,
        "reviewed96views do not yield96distinct ordered training identities"
    );
    Ok(hashes)
}
pub fn views(root: &Path, reference: &Artifact, run: &str) -> Result<BTreeSet<(String, usize)>> {
    Ok(load(root, reference, run)?.views())
}
pub fn verify(
    root: &Path,
    reference: &Artifact,
    candidate: &Candidate,
    prepared: &Prepared,
) -> Result<VerifiedOverlay> {
    let verified = load(root, reference, &candidate.run)?;
    verified.verify_context(root, &candidate.context, prepared)?;
    Ok(verified)
}
#[cfg(test)]
mod tests;
