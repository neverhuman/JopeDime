use super::*;

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub(super) fn hash_valid(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Source identity stays separate from the canonical memo identity. In
/// particular, retrieving a bank program does not rewrite its DSL/hash.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Program {
    pub dsl: String,
    pub program_sha256: String,
}

impl Program {
    pub fn from_dsl(dsl: impl Into<String>) -> Self {
        let dsl = dsl.into();
        Self {
            program_sha256: digest(dsl.as_bytes()),
            dsl,
        }
    }
}

/// Canonical memo identity does not replace the model's exact source DSL hash.
pub fn memo_key(dsl: &str) -> Result<String> {
    ensure!(
        dsl.len() <= MAX_DSL_BYTES,
        "proposal DSL exceeds byte limit"
    );
    let pipeline = parse_canonical_pipeline(dsl).map_err(anyhow::Error::msg)?;
    Ok(digest(GenomeIr::Pipeline(pipeline).canonical().as_bytes()))
}
