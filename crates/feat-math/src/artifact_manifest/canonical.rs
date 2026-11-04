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
pub(crate) struct OwnId {
    pub(crate) formula: String,
    pub(crate) schema_id: String,
    pub(crate) payload_byte_len: u64,
    pub(crate) sha256: String,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ArtifactEnvelope {
    pub(crate) envelope_schema: String,
    pub(crate) domain: ArtifactDomain,
    pub(crate) schema_id: String,
    pub(crate) own_id: OwnId,
    pub(crate) payload: ArtifactManifestV3,
}

// This closed schema has only ASCII field names, safe positive integers, strings,
// booleans and arrays. Sorted serde_json Value keys therefore have JCS UTF-16 order.
pub(crate) fn canonical(value: &impl Serialize) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(&serde_json::to_value(value)?)?)
}
pub(crate) fn domain_hash(payload: &[u8]) -> String {
    let mut bytes = SCHEMA.as_bytes().to_vec();
    bytes.push(0);
    bytes.extend_from_slice(&(payload.len() as u64).to_be_bytes());
    bytes.extend_from_slice(payload);
    sha256(&bytes)
}
pub(crate) fn check_digest(d: &str) -> Result<()> {
    ensure!(
        d.len() == 64
            && d.bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)),
        "invalid SHA-256"
    );
    Ok(())
}
pub(crate) fn safe_relative(path: &str) -> Result<()> {
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
