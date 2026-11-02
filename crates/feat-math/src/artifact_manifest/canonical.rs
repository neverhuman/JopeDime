use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactDomain {
    Native,
    Inference,
    Storage,
}

pub type Domain = ArtifactDomain;
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct OwnId {
    formula: String,
    schema_id: String,
    payload_byte_len: u64,
    sha256: String,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ArtifactEnvelope {
    envelope_schema: String,
    domain: ArtifactDomain,
    schema_id: String,
    own_id: OwnId,
    payload: ArtifactManifestV3,
}

// This closed schema has only ASCII field names, safe positive integers, strings,
// booleans and arrays. Sorted serde_json Value keys therefore have JCS UTF-16 order.
fn canonical(value: &impl Serialize) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(&serde_json::to_value(value)?)?)
}
fn domain_hash(payload: &[u8]) -> String {
    let mut bytes = SCHEMA.as_bytes().to_vec();
    bytes.push(0);
    bytes.extend_from_slice(&(payload.len() as u64).to_be_bytes());
    bytes.extend_from_slice(payload);
    sha256(&bytes)
}
fn check_digest(d: &str) -> Result<()> {
    ensure!(
        d.len() == 64
            && d.bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)),
        "invalid SHA-256"
    );
    Ok(())
}
fn safe_relative(path: &str) -> Result<()> {
    ensure!(
        !path.is_empty()
            && !path.contains('\\')
            && path
                .split('/')
                .all(|s| !s.is_empty() && s != "." && s != "..")
            && Path::new(path)
                .components()
                .all(|c| matches!(c, Component::Normal(_))),
        "unsafe artifact path"
    );
    Ok(())
}
