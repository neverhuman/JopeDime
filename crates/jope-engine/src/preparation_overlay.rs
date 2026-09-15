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
mod tests {
    use super::*;
    use crate::manifest::file_hash;
    use ed25519_dalek::{Signer, SigningKey};
    #[test]
    fn signature_uses_fixed_key_canonical_body_and_domain() {
        let k = SigningKey::from_bytes(&[31; 32]);
        let b = serde_json::json!({"z":2,"a":{"y":1,"b":0}});
        let raw = serde_json::to_vec(&canonical(&b)).unwrap();
        let mut message = DOMAIN.to_vec();
        message.extend(&raw);
        let mut s = SignedOverlay {
            schema: "jope.scoped_preparation_overlay.v1".into(),
            authorization_public_key: hex::encode(k.verifying_key().to_bytes()),
            body_sha256: hash(&raw),
            body: b,
            signature: hex::encode(k.sign(&message).to_bytes()),
        };
        signature(&s, &s.authorization_public_key).unwrap();
        assert!(signature(
            &s,
            &hex::encode(SigningKey::from_bytes(&[32; 32]).verifying_key().to_bytes())
        )
        .is_err());
        s.signature = hex::encode(k.sign(&raw).to_bytes());
        assert!(signature(&s, &s.authorization_public_key).is_err());
    }
    #[test]
    fn historical_receipt_window_is_separate_from_live_admission() {
        let v = VerifiedOverlay {
            body: Value::Null,
            not_before_unix_ms: 100,
            expires_unix_ms: 200,
        };
        v.require_live(100, 200).unwrap();
        assert!(v.require_live(99, 150).is_err());
        assert!(v.require_live(200, 200).is_err());
        assert!(v.require_live(150, 201).is_err());
    }
    #[test]
    fn dependency_paths_cannot_escape_or_conflict() {
        let mut out = BTreeMap::new();
        let h = "a".repeat(64);
        collect(
            &serde_json::json!({"a":{"path":"data/a","sha256":h}}),
            &mut out,
        )
        .unwrap();
        assert!(collect(
            &serde_json::json!({"path":"data/a","sha256":"b".repeat(64)}),
            &mut out
        )
        .is_err());
        assert!(collect(
            &serde_json::json!({"path":"../secret","sha256":h}),
            &mut out
        )
        .is_err());
        assert!(collect(
            &serde_json::json!({"path":"/etc/secret","sha256":h}),
            &mut out
        )
        .is_err());
    }
    fn fixture() -> (tempfile::TempDir, VerifiedOverlay, ContextRow, Prepared) {
        let t = tempfile::tempdir().unwrap();
        let root = t.path();
        std::fs::create_dir_all(root.join("prepared")).unwrap();
        std::fs::write(root.join("tensor.f32"), b"original tensor").unwrap();
        let h = "a".repeat(64);
        let p:Prepared=serde_json::from_value(serde_json::json!({"schema":"parallel.prepared.v1","dataset":"dataset","family":"family","task":"regression","meta_partition":"meta_train","seed":1,"family_partition_frozen":false,"clean_evaluation":true,"clean_status":"original","original_train_sha256":h,"original_test_sha256":h,"selected_train_indices":[0],"selected_test_indices":[1],"target_last":true,"rows_total":2,"storage":{"source":"raw.csv","source_sha256":h,"feature_columns":["x"],"target_column":"y","original_training_row_map":[0]},"cuts":[{"index":0,"train_indices":[0],"validation_indices":[1],"training_sha256":h,"validation_sha256":h,"preprocessing_sha256":h}]})).unwrap();
        std::fs::write(
            root.join("prepared/prepared.json"),
            serde_json::to_vec(&p).unwrap(),
        )
        .unwrap();
        let c = ContextRow {
            dataset: "dataset".into(),
            family: "family".into(),
            cut: 0,
            partition: "meta_train".into(),
            path: "tensor.f32".into(),
            sha256: file_hash(&root.join("tensor.f32")).unwrap(),
            prepared: "prepared".into(),
            prepared_sha256: file_hash(&root.join("prepared/prepared.json")).unwrap(),
        };
        let body = serde_json::json!({"withdrawn_families":["withdrawn"],"descriptors":[{"dataset":"dataset","body":{"preserved_family":"family","preserved_partition":"meta_train","original_prepared":{"path":"prepared/prepared.json","sha256":c.prepared_sha256},"original_storage_source":{"path":"raw.csv","sha256":h},"original_train_sha256":h,"original_test_sha256":h,"cuts":[{"index":0,"preprocessing_sha256":h,"training_sha256":h,"validation_sha256":h}],"context_views":[{"cut":0,"original_prepared_sha256":c.prepared_sha256,"tensor":{"path":"tensor.f32","sha256":c.sha256}}]}}]});
        (
            t,
            VerifiedOverlay {
                body,
                not_before_unix_ms: 100,
                expires_unix_ms: 200,
            },
            c,
            p,
        )
    }
    #[test]
    fn exact_original_view_passes_without_mutating_flags_and_refuses_substitution() {
        let (t, v, c, p) = fixture();
        v.verify_context(t.path(), &c, &p).unwrap();
        assert!(!p.family_partition_frozen);
        let mut changed = c.clone();
        changed.cut = 2;
        assert!(v.verify_context(t.path(), &changed, &p).is_err());
        let mut changed = c.clone();
        changed.prepared_sha256 = "b".repeat(64);
        assert!(v.verify_context(t.path(), &changed, &p).is_err());
        let mut changed = c.clone();
        changed.sha256 = "b".repeat(64);
        assert!(v.verify_context(t.path(), &changed, &p).is_err());
        let mut changed = c.clone();
        changed.partition = "meta_test".into();
        assert!(v.verify_context(t.path(), &changed, &p).is_err());
        let mut changed = c.clone();
        changed.family = "foreign".into();
        assert!(v.verify_context(t.path(), &changed, &p).is_err());
        let mut changed = p.clone();
        changed.clean_evaluation = false;
        assert!(v.verify_context(t.path(), &c, &changed).is_err());
        let mut changed = p.clone();
        changed.qualification_only = true;
        assert!(v.verify_context(t.path(), &c, &changed).is_err());
        let mut changed = v;
        changed.body["withdrawn_families"] = serde_json::json!(["family"]);
        assert!(changed.verify_context(t.path(), &c, &p).is_err());
    }

    #[test]
    fn nonregular_overlay_dependencies_refuse_without_blocking() {
        let t = tempfile::tempdir().unwrap();
        let fifo = t.path().join("fifo");
        let name = std::ffi::CString::new(fifo.as_os_str().as_encoded_bytes()).unwrap();
        assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
        assert!(checked_hash(&fifo).is_err());
        std::os::unix::fs::symlink(&fifo, t.path().join("link")).unwrap();
        assert!(checked_hash(&t.path().join("link")).is_err());
    }

    fn scoped_body() -> Value {
        let (_, v, _, _) = fixture();
        let mut descriptor = v.body["descriptors"][0]["body"].clone();
        descriptor["schema"] = serde_json::json!("jope.prepared_compatibility_view.v1");
        descriptor["scope"] = serde_json::json!(SCOPE);
        descriptor["original_global_inventory_frozen"] = serde_json::json!(false);
        descriptor["original_metadata_unchanged"] = serde_json::json!(true);
        descriptor["final_test_evaluation_qualified"] = serde_json::json!(false);
        let mut cut = descriptor["cuts"][0].clone();
        cut["index"] = serde_json::json!(1);
        descriptor["cuts"].as_array_mut().unwrap().push(cut);
        let mut view = descriptor["context_views"][0].clone();
        view["cut"] = serde_json::json!(1);
        descriptor["context_views"]
            .as_array_mut()
            .unwrap()
            .push(view);
        let rows:Vec<_>=(0..48).map(|i|{let mut d=descriptor.clone();let dataset=format!("{i:016x}");d["dataset"]=serde_json::json!(dataset);serde_json::json!({"dataset":dataset,"descriptor_sha256":hash(&serde_json::to_vec(&canonical(&d)).unwrap()),"body":d})}).collect();
        let a = serde_json::json!({"path":"evidence.json","sha256":"a".repeat(64)});
        let b = serde_json::json!({"schema":"jope.scoped_preparation_overlay_body.v1","scope":SCOPE,"run":"run","optimizer_window":{"blocks":2,"global_batch":2048,"initial_step":60000,"target_step":66000,"updates_per_block":3000},"cuts":[0,1],"validity":{"not_before_unix_ms":100,"not_after_unix_ms":200},"descriptors":rows,"partition_provenance":{"full_inventory_complete":false},"family_similarity_exception":{"never_clean_unseen_family_heldout_for_lineage":true},"withdrawn_families":["rdatasets-cardata-leinhardt","rdatasets-cardata-prestige"],"context_index":a,"decision":a,"verification":{"claude_context_verifier":a,"verify_prepared":a}});
        b
    }
    #[test]
    fn scope_requires_all48_sorted_descriptors_and_original_window() {
        let b = scoped_body();
        let valid = shape(b.clone(), "run").unwrap();
        assert_eq!(valid.views().len(), 96);
        assert!(shape(b.clone(), "other").is_err());
        let mut changed = b.clone();
        changed["descriptors"].as_array_mut().unwrap().pop();
        assert!(shape(changed, "run").is_err());
        let mut changed = b.clone();
        changed["descriptors"].as_array_mut().unwrap().swap(0, 1);
        assert!(shape(changed, "run").is_err());
        let mut changed = b.clone();
        changed["optimizer_window"]["initial_step"] = serde_json::json!(66000);
        assert!(shape(changed, "run").is_err());
        let mut changed = b;
        changed["partition_provenance"]["full_inventory_complete"] = serde_json::json!(true);
        assert!(shape(changed, "run").is_err());
    }
    #[test]
    fn signed_dependency_closure_refuses_tamper_and_preserves_historical_window() {
        let (t, _, mut context, mut prepared) = fixture();
        let root = t.path();
        prepared.dataset = "0000000000000000".into();
        context.dataset = prepared.dataset.clone();
        std::fs::write(root.join("raw.csv"), b"original fixture raw.csv").unwrap();
        let mut prepared_value = serde_json::to_value(&prepared).unwrap();
        prepared_value["storage"]["source_sha256"] =
            serde_json::json!(file_hash(&root.join("raw.csv")).unwrap());
        prepared = serde_json::from_value(prepared_value).unwrap();
        std::fs::write(
            root.join("prepared/prepared.json"),
            serde_json::to_vec(&prepared).unwrap(),
        )
        .unwrap();
        context.prepared_sha256 = file_hash(&root.join("prepared/prepared.json")).unwrap();
        let mut body = scoped_body();
        // Replace every declared dependency with actual fixture bytes, retaining all 48 descriptors.
        fn bind(v: &mut Value, root: &Path) {
            if let Some(path) = v.get("path").and_then(Value::as_str).map(str::to_owned) {
                if v.get("sha256").is_some() {
                    let p = root.join(&path);
                    std::fs::create_dir_all(p.parent().unwrap()).unwrap();
                    if !p.exists() {
                        std::fs::write(&p, format!("original fixture {path}")).unwrap();
                    }
                    v["sha256"] = serde_json::json!(file_hash(&p).unwrap());
                }
            }
            match v {
                Value::Object(m) => {
                    for child in m.values_mut() {
                        bind(child, root);
                    }
                }
                Value::Array(a) => {
                    for child in a {
                        bind(child, root);
                    }
                }
                _ => (),
            }
        }
        bind(&mut body, root);
        for row in body["descriptors"].as_array_mut().unwrap() {
            for view in row["body"]["context_views"].as_array_mut().unwrap() {
                view["original_prepared_sha256"] = serde_json::json!(context.prepared_sha256);
            }
            row["descriptor_sha256"] =
                serde_json::json!(hash(&serde_json::to_vec(&canonical(&row["body"])).unwrap()));
        }
        let key = SigningKey::from_bytes(&[87; 32]);
        let pin = hex::encode(key.verifying_key().to_bytes());
        let publish = |body: Value| {
            let bytes = serde_json::to_vec(&canonical(&body)).unwrap();
            let mut message = DOMAIN.to_vec();
            message.extend(&bytes);
            let signed = SignedOverlay {
                schema: "jope.scoped_preparation_overlay.v1".into(),
                authorization_public_key: pin.clone(),
                body_sha256: hash(&bytes),
                body,
                signature: hex::encode(key.sign(&message).to_bytes()),
            };
            let bytes = serde_json::to_vec(&signed).unwrap();
            std::fs::write(root.join("overlay.json"), &bytes).unwrap();
            Artifact {
                path: "overlay.json".into(),
                sha256: hash(&bytes),
            }
        };
        let a = publish(body.clone());
        let verified = load_pinned(root, &a, "run", &pin).unwrap();
        assert_eq!(verified.views().len(), 96);
        verified.verify_context(root, &context, &prepared).unwrap();
        let mut unlisted = context.clone();
        unlisted.dataset = "ffffffffffffffff".into();
        assert!(verified.verify_context(root, &unlisted, &prepared).is_err());
        let mut wrong_cut = context.clone();
        wrong_cut.cut = 2;
        assert!(verified
            .verify_context(root, &wrong_cut, &prepared)
            .is_err());
        verified.require_live(100, 200).unwrap();
        assert!(verified.require_live(200, 200).is_err());
        // Historical replay remains valid after expiry; live admission remains refused.
        load_pinned(root, &a, "run", &pin).unwrap();
        assert!(load_pinned(root, &a, "other-run", &pin).is_err());
        assert!(load_pinned(root, &a, "run", &"a".repeat(64)).is_err());
        std::fs::write(root.join("raw.csv"), b"substituted source").unwrap();
        assert!(load_pinned(root, &a, "run", &pin).is_err());
        std::fs::write(root.join("raw.csv"), b"original fixture raw.csv").unwrap();
        let mut wrong = body.clone();
        wrong["descriptors"][0]["descriptor_sha256"] = serde_json::json!("a".repeat(64));
        assert!(load_pinned(root, &publish(wrong), "run", &pin).is_err());
        let mut wrong = body.clone();
        wrong["scope"] = serde_json::json!("unreviewed");
        assert!(load_pinned(root, &publish(wrong), "run", &pin).is_err());
        let a = publish(body);
        let mut bytes = std::fs::read(root.join("overlay.json")).unwrap();
        bytes[0] = b'[';
        std::fs::write(root.join("overlay.json"), bytes).unwrap();
        assert!(load_pinned(root, &a, "run", &pin).is_err());
    }
}
