//! Adapter for the owner-admitted r782 release's decoded assets and envelope.
//! Tensor expectations are frozen release metadata, never inferred from weights.
use crate::ranker::bundle::{
    load_bundle, read_hashed_asset, AssetIdentity, BundleDigest, BundleInput, BundleLimits,
    ReleaseAsset, ReleaseCatalog, SignedAssetContext, TensorSpec, VerifiedBundle,
    FROZEN_DOPE_COMPOSITE,
};
use anyhow::{ensure, Result};
use safetensors::Dtype;
use serde::Deserialize;
use std::{collections::BTreeMap, fs::File, io::Read, path::Path};

pub const R782: &str = "sha256:94fd9587675c7806bd0a0582b6e97540cd2fdc7ebbc51958411b905185cad85c";
const CATALOG_SHA: &str = "d5a4b21e6167dbd637e940acd3a11d6357c147bb021ea2a8929f06c71951e34f";

#[derive(Deserialize)]
struct ReleaseMetadata {
    schema: String,
    bundle_digest: String,
    dope_logical_composite: String,
    dope_weights_sha256: String,
    assets: BTreeMap<String, MetadataAsset>,
}
#[derive(Deserialize)]
struct MetadataAsset {
    signed_asset_id: String,
    tensors: Option<BTreeMap<String, MetadataTensor>>,
}
#[derive(Deserialize)]
struct MetadataTensor {
    shape: Vec<usize>,
    dtype: Dtype,
}
fn digest(hex: &str) -> Result<[u8; 32]> {
    ensure!(hex.len() == 64, "invalid release digest");
    let mut out = [0; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16)?;
    }
    Ok(out)
}
fn bounded(path: &Path, maximum: u64) -> Result<Vec<u8>> {
    let file = File::open(path)?;
    ensure!(
        file.metadata()?.is_file(),
        "release envelope must be regular"
    );
    let mut bytes = Vec::new();
    file.take(maximum + 1).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= maximum,
        "release envelope exceeds bound"
    );
    Ok(bytes)
}

/// Trusted host opt-in. The directory is an extraction made through jope-assets;
/// the production loader independently rechecks the existing release signature,
/// every selected decoded hash and the fixed complete tensor inventories.
pub fn load_r782_release(root: &Path) -> Result<VerifiedBundle> {
    let limits = BundleLimits::default();
    let bytes = read_hashed_asset(
        File::open(root.join("release-catalog.json"))?,
        AssetIdentity {
            bytes: 240_292,
            sha256: digest(CATALOG_SHA)?,
        },
        limits,
    )?;
    let metadata: ReleaseMetadata = serde_json::from_slice(&bytes)?;
    ensure!(
        metadata.schema == "jope.r782-release-catalog/v1"
            && metadata.bundle_digest == R782
            && metadata.dope_logical_composite == FROZEN_DOPE_COMPOSITE,
        "release is not the admitted frozen r782"
    );
    let catalog = ReleaseCatalog {
        bundle_digest: BundleDigest::parse(&metadata.bundle_digest)?,
        dope_logical_composite: digest(&metadata.dope_logical_composite)?,
        dope_weights_sha256: digest(&metadata.dope_weights_sha256)?,
        assets: metadata
            .assets
            .into_iter()
            .map(|(path, a)| {
                (
                    path,
                    ReleaseAsset {
                        signed_asset_id: a.signed_asset_id,
                        tensors: a.tensors.map(|ts| {
                            ts.into_iter()
                                .map(|(name, t)| {
                                    (
                                        name,
                                        TensorSpec {
                                            shape: t.shape,
                                            dtype: t.dtype,
                                        },
                                    )
                                })
                                .collect()
                        }),
                    },
                )
            })
            .collect(),
    };
    let manifest = bounded(&root.join("appliance-manifest.json"), limits.manifest_bytes)?;
    let footer: [u8; 128] = bounded(&root.join("appliance-footer.bin"), 128)?
        .try_into()
        .map_err(|_| anyhow::anyhow!("invalid footer length"))?;
    let signature: [u8; 64] = bounded(&root.join("appliance-signature.bin"), 64)?
        .try_into()
        .map_err(|_| anyhow::anyhow!("invalid signature length"))?;
    Ok(load_bundle(
        BundleInput::DevelopmentDirectory(root),
        SignedAssetContext {
            manifest: &manifest,
            footer: &footer,
            signature: &signature,
        },
        &catalog,
        limits,
    )?)
}
