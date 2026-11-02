//! JOPE-Dime dataset catalog: content-addressed ingest, fetch, verify and link.
//!
//! Every dataset is a directory `<id>/{train.csv,test.csv,meta.json}` whose name is the
//! `dataset_hash` (the first 16 hex digits of the canonical sha256 recorded in `meta.json`).
//! Files are stored once under `blobs/sha256/<64hex>`; a catalog row per dataset points at them.

pub mod blobs;
pub mod cache;
pub mod catalog;
pub mod cli;
pub mod credentials;
pub mod family_contracts;
pub mod hash;
pub mod ingest;
pub mod meta;
pub mod paths;
pub mod regression_release;
pub mod repo;
pub mod schema;
pub mod staging_disposition;
pub mod store;
pub mod verified_object;

/// Version stamped into `ingest.tool_version`.
pub const TOOL_VERSION: &str = concat!("jope-datasets/", env!("CARGO_PKG_VERSION"));

/// Default datasets bucket (`JOPE_DATASETS_URL` overrides).
pub const DEFAULT_DATASETS_URL: &str = "s3://veox-jopedime-datasets-use2";

/// `^(synth_)?[0-9a-f]{16}$` without a regex: returns the 16-hex hash when the name matches.
pub fn dataset_dir_hash(name: &str) -> Option<&str> {
    let hex = name.strip_prefix("synth_").unwrap_or(name);
    (hex.len() == 16 && hex.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))).then_some(hex)
}

#[cfg(test)]
mod tests {
    use super::dataset_dir_hash;

    #[test]
    fn dir_hash_pattern() {
        assert_eq!(
            dataset_dir_hash("1cb97b1d7314d43d"),
            Some("1cb97b1d7314d43d")
        );
        assert_eq!(
            dataset_dir_hash("synth_3f0416513333772c"),
            Some("3f0416513333772c")
        );
        assert_eq!(dataset_dir_hash("synth_3f0416513333772c.incoming"), None);
        assert_eq!(dataset_dir_hash("1CB97B1D7314D43D"), None);
        assert_eq!(dataset_dir_hash("1cb97b1d7314d43"), None);
        assert_eq!(dataset_dir_hash("regression"), None);
    }
}
