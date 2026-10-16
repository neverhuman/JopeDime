//! Check announced identity-fixture successors without changing historical pins.
use serde_json::Value;
use std::collections::BTreeSet;

pub fn verify_fixture(
    path: &str,
    historical_hash: &str,
    current_hash: &str,
    transitions: &[Value],
) -> Result<(), String> {
    let identity_allowed = [
        "crates/feat-math/tests/fixtures/compat_freeze/evaluator_identity.v1.json",
        "crates/feat-math/tests/fixtures/compat_freeze/lineage_sources.v1.json",
    ];
    let emitter_allowed = [
        "crates/feat-math/tests/fixtures/compat_freeze/emitted_bytes.v1.json",
        "crates/feat-math/tests/fixtures/compat_freeze/evaluator_identity.v1.json",
        "crates/feat-math/tests/fixtures/compat_freeze/frozen_pins.v1.json",
        "crates/feat-math/tests/fixtures/compat_freeze/generator_streams.v1.json",
    ];
    let mut expected = historical_hash;
    for transition in transitions {
        let allowed: &[&str] = match transition["schema"].as_str() {
            Some("feat-math.compat-freeze.identity-rebaseline.v1") => &identity_allowed,
            Some("feat-math.compat-freeze.emitter-rebaseline.v1") => &emitter_allowed,
            _ => return Err("identity transition schema".into()),
        };
        let entries = transition["fixtures"]
            .as_array()
            .ok_or("identity fixture list")?;
        let mut seen = BTreeSet::new();
        for entry in entries {
            let fixture = entry["path"].as_str().ok_or("identity fixture path")?;
            if !allowed.contains(&fixture) || !seen.insert(fixture) {
                return Err(
                    "identity transition cannot change behavior or repeat a fixture".into(),
                );
            }
            let before = entry["before"].as_str().ok_or("identity predecessor")?;
            let after = entry["after"].as_str().ok_or("identity successor")?;
            if [before, after]
                .iter()
                .any(|hash| hash.len() != 64 || !hash.bytes().all(|b| b.is_ascii_hexdigit()))
            {
                return Err("identity transition hash".into());
            }
            if fixture == path {
                if before != expected {
                    return Err("identity transition predecessor does not match".into());
                }
                expected = after;
            }
        }
    }
    if expected != current_hash {
        return Err(format!("unledgered fixture change: {path}"));
    }
    Ok(())
}
