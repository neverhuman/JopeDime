//! Isolated controller fixture for product integration tests; never production authority.
//! Components are deterministic synthetic vectors, not measured foundation/target coverage.
use anyhow::{ensure, Result};
use serde_json::{json, Value};
use starforge::dope_encoder::FitScope;
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub struct ComponentFixture {
    pub requests: PathBuf,
    pub controller: PathBuf,
    pub authority: PathBuf,
    pub subject_sha256: String,
    pub exposure_receipt_sha256: String,
    pub foundation1024: Vec<f32>,
    pub target512: Vec<f32>,
}
impl ComponentFixture {
    /// Deliberately change this fixture's owned authority; retained capabilities must refuse it.
    pub fn mutate_authority(&self) -> Result<()> {
        let mut value: Value = serde_json::from_slice(&std::fs::read(&self.authority)?)?;
        value["revoked_digests"] = json!([self.subject_sha256]);
        std::fs::write(&self.authority, canonical(value)?)?;
        Ok(())
    }
}

/// All controller state lives below a caller-created empty test directory. No global environment
/// or configured operator store is changed. The ordinary admission verifier authenticates it.
pub fn install_for_scope(root: &Path, scope: FitScope) -> Result<ComponentFixture> {
    if root.exists() {
        ensure!(
            std::fs::read_dir(root)?.next().is_none(),
            "fixture root must be empty"
        );
    }
    std::fs::create_dir_all(root.join("blobs/sha256"))?;
    std::fs::create_dir_all(root.join("replay"))?;
    let scope_digest = scope.digest()?;
    let episode = format!("p4-fixture-{}", &scope_digest[..16]);
    let observed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_millis()
        .try_into()?;
    let fit = retain(
        root,
        json!({"schema":"jope.p4-fit-input/v1", "fit_scope":scope}),
    )?;
    let fit_proof = proof(root, &fit, &episode, observed, vec![])?;
    let salt = u8::from_str_radix(&scope_digest[..2], 16)? as f32 / 255.;
    let foundation1024 = (0..1024)
        .map(|i| ((i % 37) as f32 * 0.03 + salt).sin())
        .collect::<Vec<_>>();
    let target512 = (0..512)
        .map(|i| ((i % 23) as f32 * 0.02 - salt).cos())
        .collect::<Vec<_>>();
    let foundation = retain(
        root,
        json!({"schema":"jope.p4-component-producer/v1",
        "recipe":"synthetic-fit-scoped-foundation/v1", "fit_scope_digest":scope_digest,
        "values_le_f32":hex_values(&foundation1024)}),
    )?;
    let target = retain(
        root,
        json!({"schema":"jope.p4-component-producer/v1",
        "recipe":"synthetic-fit-scoped-target/v1", "fit_scope_digest":scope_digest,
        "values_le_f32":hex_values(&target512)}),
    )?;
    let foundation_proof = proof(
        root,
        &foundation,
        &episode,
        observed,
        vec![fit_proof.clone()],
    )?;
    let target_proof = proof(root, &target, &episode, observed, vec![fit_proof.clone()])?;
    let subject_sha256 = retain(
        root,
        json!({"schema":"jope.dope-fusion-components/v1",
        "fit_scope":scope, "episode":episode, "permitted_use":"episode_fit",
        "recipe":starforge::dope_encoder::FUSION_COMPONENT_RECIPE,
        "producer_inputs":{"foundation":foundation,"target":target},
        "foundation1024_le_f32":hex_values(&foundation1024),
        "target512_le_f32":hex_values(&target512)}),
    )?;
    let exposure_receipt_sha256 = proof(
        root,
        &subject_sha256,
        &episode,
        observed,
        vec![foundation_proof.clone(), target_proof.clone()],
    )?;
    let boundary = retain(
        root,
        json!({"schema":"jope.p4-future-evaluation/v1",
        "purpose":"fixture-only future boundary; no outer or sealed data"}),
    )?;
    let policy = retain(
        root,
        json!({"schema":"jope.replay-admission-policy/v2",
        "issuer":"fixture-controller","epoch":1,"effective_from_ms":0,
        "effective_until_ms":4_102_444_800_000u64,"evaluation_boundary_sha256":boundary,
        "purposes":["episode_fit"],"receipts":[fit_proof,foundation_proof,target_proof,
            exposure_receipt_sha256],"supersedes":null}),
    )?;
    let adoption = canonical(json!({"schema":"jope.replay-policy-adoption/v2",
        "policy_sha256":policy,"issuer":"fixture-controller","epoch":1}))?;
    write_new(&root.join("replay/adopted-policy.json"), &adoption)?;
    let authority = root.join("replay/component-authority.json");
    write_new(
        &authority,
        &canonical(json!({"schema":"jope.component-authority/v1",
        "adoption_sha256":jope_runtime::sha256_hex(&adoption),
        "revocation_files":{},"revoked_digests":[]}))?,
    )?;
    let requests = root.join("components.json");
    write_new(
        &requests,
        &canonical(json!({"schema":"jope.fusion-component-requests/v1",
        "entries":[{"fit_scope_digest":scope_digest,"subject_sha256":subject_sha256,
            "exposure_receipt_sha256":exposure_receipt_sha256,"episode":episode}]}))?,
    )?;
    Ok(ComponentFixture {
        requests,
        controller: root.to_owned(),
        authority,
        subject_sha256,
        exposure_receipt_sha256,
        foundation1024,
        target512,
    })
}
fn proof(
    root: &Path,
    subject: &str,
    episode: &str,
    observed: u64,
    parents: Vec<String>,
) -> Result<String> {
    retain(
        root,
        json!({"schema":"jope.replay-exposure/v2","subject_sha256":subject,
        "family":"p4-product-fixture","meta_partition":"train_search",
        "original_role":"train_fit","within_task_view":"training","fidelity":"fixture",
        "producer":"fixture-controller","policy_epoch":1,"observed_at_ms":observed,
        "episode":episode,"purposes":["episode_fit"],"prior_exposure":["train_fit"],
        "parents":parents,"observation":{"status":"proposed"}}),
    )
}
fn retain(root: &Path, value: Value) -> Result<String> {
    let bytes = canonical(value)?;
    let digest = jope_runtime::sha256_hex(&bytes);
    let path = root.join("blobs/sha256").join(&digest);
    if path.exists() {
        ensure!(std::fs::read(path)? == bytes, "fixture CAS collision");
    } else {
        write_new(&path, &bytes)?;
    }
    Ok(digest)
}
fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    file.write_all(bytes)?;
    Ok(())
}
fn hex_values(values: &[f32]) -> String {
    values
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .map(|b| format!("{b:02x}"))
        .collect()
}
/// These fixture values contain only exact safe integers, strings, arrays and objects. Recursive
/// key sorting avoids depending on serde_json's optional preserve_order feature. The real
/// CanonicalPayload verifier checks these exact bytes when ComponentInputs loads the closure.
fn canonical(value: Value) -> Result<Vec<u8>> {
    fn sorted(value: Value) -> Value {
        match value {
            Value::Object(map) => Value::Object(
                map.into_iter()
                    .map(|(k, v)| (k, sorted(v)))
                    .collect::<BTreeMap<_, _>>()
                    .into_iter()
                    .collect(),
            ),
            Value::Array(values) => Value::Array(values.into_iter().map(sorted).collect()),
            other => other,
        }
    }
    Ok(serde_json::to_vec(&sorted(value))?)
}
