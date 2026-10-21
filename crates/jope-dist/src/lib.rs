//! `jope-dist`: the build-time side of the `jopedime` appliance (SPECIFICATION.md §4.1, §10, §14,
//! §16). It resolves a profile's model closure into a reviewed lock, packs the executable and the
//! locked assets deterministically, produces the exact message the release key signs, seals the
//! signed result, verifies packages independently of the executable they contain, audits the
//! shipped dependency closure, and writes size, notice, SBOM and provenance reports.
//!
//! It is never shipped inside the appliance and holds no signing credentials: release packages
//! are signed by the existing root-held cosign step on the release host.

pub mod audit;
pub mod fixture;
pub mod notices;
pub mod pack;
pub mod profile;
pub mod report;
pub mod signature;

/// The appliance's product name in every manifest.
pub const PRODUCT: &str = "jopedime";

/// The repository root: two levels above this crate.
pub fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap_or_else(|_| std::path::PathBuf::from("."))
}
