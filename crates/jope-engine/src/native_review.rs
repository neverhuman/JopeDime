//! Independently established migration review trust, separate from signed policies.
use anyhow::{ensure, Result};
use serde::Deserialize;
use std::path::Path;
const TRUST_PATH: &str = "/etc/jope-dime/native-review-trust.json";
const REVIEW_PUBLIC_KEY: &str = "4e1c31a28fab4610a65d74f8499367f8ca335a2a17fcd8ac7586d93d00b42ce7";
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Trust {
    schema: String,
    authorization_public_key: String,
}
pub(crate) fn require_signer(public_key: &str) -> Result<()> {
    let bytes = crate::eval_registry::native_scq::policy::protected(Path::new(TRUST_PATH))?;
    let trust: Trust = serde_json::from_slice(&bytes)?;
    ensure!(
        trust.schema == "jope.native_review_trust.v1"
            && trust.authorization_public_key == REVIEW_PUBLIC_KEY
            && public_key == trust.authorization_public_key,
        "signed policy does not use the independently established migration reviewer key"
    );
    Ok(())
}
