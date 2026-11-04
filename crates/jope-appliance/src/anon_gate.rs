//! The anonymous tier of the public jAIn CLI (feature `anon-gate`, selected only by the
//! public release build).
//!
//! Signed out, jAIn fits, predicts and registers only the bundled demo datasets: the SHA-256 of
//! the data's raw bytes must equal a demo's. Signed in (a stored veox Cloud Workspace
//! connection), any data is allowed. The same decision is enforced at every way data enters:
//!
//! - the engine's dataset registration (`run`, `dataset add`, MCP, the terminal, remote
//!   `predict`, and any local HTTP client), through [`jope_service::admission`], installed by
//!   [`install`] in every `jain` process, including a background `jain serve` engine;
//! - in-process CSV reads (`fit`, `fit-predict`, local `predict`) via [`admit`];
//! - `jain api` and `run --m1-plan`, whose data never passes through registration, which
//!   require a signed-in account ([`require_signed_in`]).
//!
//! Builds without the feature (development, CI, qualification) admit everything. In gated
//! builds `JAIN_ALLOW_LOCAL_FILES` has no effect. This is client-side policy: the source is
//! public, so it steers the public binary; anything that costs money is enforced server-side.

/// Shown wherever signed-out use of other data is refused.
#[cfg_attr(not(feature = "anon-gate"), allow(dead_code))]
pub const SIGNED_OUT: &str = "Signed out, jAIn runs only its bundled demos (start `jain` and type /demo). To use your own data, log in to a free veox Cloud Workspace with `jain auth login`; your data stays on this machine.";

/// Whether this build enforces the anonymous tier.
pub const ENFORCED: bool = cfg!(feature = "anon-gate");

#[cfg(feature = "anon-gate")]
mod enforce {
    use std::sync::OnceLock;

    fn demo_hashes() -> &'static [String] {
        static HASHES: OnceLock<Vec<String>> = OnceLock::new();
        HASHES.get_or_init(|| {
            crate::terminal::demos::DEMOS
                .iter()
                .map(|demo| jope_runtime::sha256_hex(demo.bytes))
                .collect()
        })
    }

    /// Re-read on every call: a long-lived background engine must follow `/login` and logout.
    pub fn signed_in() -> bool {
        crate::cloud::connection().is_ok()
    }

    pub fn admit(sha256_hex: &str) -> Result<(), String> {
        if signed_in() || demo_hashes().iter().any(|hash| hash == sha256_hex) {
            Ok(())
        } else {
            Err(super::SIGNED_OUT.to_owned())
        }
    }
}

/// Admit data by the lowercase hex SHA-256 of its raw bytes.
pub fn admit(sha256_hex: &str) -> Result<(), String> {
    #[cfg(feature = "anon-gate")]
    {
        enforce::admit(sha256_hex)
    }
    #[cfg(not(feature = "anon-gate"))]
    {
        let _ = sha256_hex;
        Ok(())
    }
}

/// For commands whose data cannot be checked by hash before use.
pub fn require_signed_in(command: &str) -> anyhow::Result<()> {
    #[cfg(feature = "anon-gate")]
    if !enforce::signed_in() {
        return Err(jope_runtime::Failure::input(format!(
            "`{command}` needs a signed-in account. {}",
            SIGNED_OUT
        )));
    }
    let _ = command;
    Ok(())
}

/// Install the engine-side policy for this process (before any engine starts).
pub fn install() {
    #[cfg(feature = "anon-gate")]
    jope_service::admission::set_dataset_policy(enforce::admit);
}

#[cfg(all(test, feature = "anon-gate"))]
mod tests {
    use super::*;

    #[test]
    fn signed_out_admits_exactly_the_bundled_demo_bytes() {
        let home = tempfile::tempdir().unwrap();
        std::env::set_var("JOPEDIME_HOME", home.path());
        for demo in crate::terminal::demos::DEMOS {
            assert!(
                admit(&jope_runtime::sha256_hex(demo.bytes)).is_ok(),
                "{}",
                demo.name
            );
        }
        let mut edited = crate::terminal::demos::DEMOS[0].bytes.to_vec();
        edited.push(b'\n');
        assert_eq!(
            admit(&jope_runtime::sha256_hex(&edited)).unwrap_err(),
            SIGNED_OUT
        );
        assert!(admit(&"0".repeat(64)).is_err());
        let error = require_signed_in("jain api").unwrap_err().to_string();
        assert!(error.contains("jain auth login"), "{error}");
        // The development escape hatch does not reopen a gated build.
        std::env::set_var("JAIN_ALLOW_LOCAL_FILES", "1");
        assert!(admit(&"0".repeat(64)).is_err());
        std::env::remove_var("JAIN_ALLOW_LOCAL_FILES");
    }
}
