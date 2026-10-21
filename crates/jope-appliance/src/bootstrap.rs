//! Open and authenticate the running executable's package (SPECIFICATION.md §10.4, §10.6).
//!
//! Startup authenticates the footer, manifest, signature and executable region. The manifest
//! binds every asset's hashes, and each asset is authenticated when it is read
//! (AMENDMENTS.md A5). `verify-self` covers every byte.
use jope_assets::{AssetError, Package};
use jope_runtime::Failure;
use serde::Serialize;

/// Whether this build refuses to run without a package.
pub const EMBEDDED: bool = cfg!(feature = "embedded");

/// The package of this executable: `None` for an unpackaged development build.
pub fn open() -> anyhow::Result<Option<Package>> {
    let package = Package::open_self().map_err(integrity)?;
    if package.is_none() && EMBEDDED {
        return Err(Failure::integrity(
            "this release executable has no package (was its payload removed?)",
        ));
    }
    Ok(package)
}

pub fn integrity(error: AssetError) -> anyhow::Error {
    Failure::integrity(error.to_string())
}

#[derive(Clone, Debug, Serialize)]
pub struct PackageSummary {
    pub product: String,
    pub product_version: String,
    pub target: String,
    pub profile: String,
    pub scope: String,
    pub source_sha40: String,
    pub manifest_sha256: String,
    pub signer: String,
    pub signer_role: &'static str,
    pub entries: usize,
    pub decoded_bytes: u64,
    pub stored_bytes: u64,
    pub methods: Vec<String>,
}

pub fn summary(package: &Package) -> PackageSummary {
    let manifest = package.manifest();
    PackageSummary {
        product: manifest.product.clone(),
        product_version: manifest.product_version.clone(),
        target: manifest.target.clone(),
        profile: manifest.profile.clone(),
        scope: manifest.scope.as_str().to_owned(),
        source_sha40: manifest.source.sha40.clone(),
        manifest_sha256: hex(&package.footer().manifest_sha256),
        signer: package.signer().id.to_string(),
        signer_role: package.signer().role,
        entries: manifest.entries.len(),
        decoded_bytes: manifest.entries.iter().map(|e| e.bytes).sum(),
        stored_bytes: manifest.entries.iter().map(|e| e.stored_bytes).sum(),
        methods: manifest.methods.clone(),
    }
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The source commit this executable was built from, as the builder declared it.
pub fn source_sha40() -> String {
    option_env!("JOPE_BUILD_SOURCE_SHA")
        .filter(|sha| sha.len() == 40 && sha.bytes().all(|b| b.is_ascii_hexdigit()))
        .unwrap_or("unknown")
        .to_owned()
}
