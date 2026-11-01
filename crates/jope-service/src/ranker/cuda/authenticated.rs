//! Explicit real-release controls, never selected by ordinary unit tests.
//! The metadata pin is frozen at release preparation, independently of loaded
//! model bytes. The package adapter first verifies and decodes the signed file.
use crate::ranker::bundle::{
    load_bundle, BundleDigest, BundleInput, BundleLimits, EmbeddedAsset, ReleaseAsset,
    ReleaseCatalog, SignedAssetContext, TensorSpec, VerifiedBundle, FROZEN_DOPE_COMPOSITE,
};
use jope_api::error::ErrorCode;
use safetensors::Dtype;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, env, fs, path::PathBuf};

const CATALOG_SHA: &str = "d5a4b21e6167dbd637e940acd3a11d6357c147bb021ea2a8929f06c71951e34f";
const BUNDLE: &str = "sha256:94fd9587675c7806bd0a0582b6e97540cd2fdc7ebbc51958411b905185cad85c";

#[derive(Deserialize)]
struct Catalog {
    schema: String,
    bundle_digest: String,
    dope_logical_composite: String,
    dope_weights_sha256: String,
    assets: BTreeMap<String, Asset>,
}

#[derive(Deserialize)]
struct Asset {
    signed_asset_id: String,
    tensors: Option<BTreeMap<String, Tensor>>,
}

#[derive(Deserialize)]
struct Tensor {
    shape: Vec<usize>,
    dtype: Dtype,
}

fn digest(hex: &str) -> [u8; 32] {
    assert_eq!(hex.len(), 64);
    std::array::from_fn(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap())
}

struct Inputs {
    manifest: Vec<u8>,
    footer: [u8; 128],
    signature: [u8; 64],
    catalog: ReleaseCatalog,
    assets: BTreeMap<String, Vec<u8>>,
}

impl Inputs {
    fn open() -> Self {
        let root = PathBuf::from(
            env::var_os("JOPE_AUTHENTICATED_BUNDLE_DIR")
                .expect("release-verified package extraction required"),
        );
        let bytes = fs::read(root.join("release-catalog.json")).unwrap();
        assert_eq!(format!("{:x}", Sha256::digest(&bytes)), CATALOG_SHA);
        let c: Catalog = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(c.schema, "jope.r782-release-catalog/v1");
        assert_eq!(c.bundle_digest, BUNDLE);
        assert_eq!(c.dope_logical_composite, FROZEN_DOPE_COMPOSITE);
        let assets = c
            .assets
            .keys()
            .map(|p| (p.clone(), fs::read(root.join(p)).unwrap()))
            .collect();
        let catalog = ReleaseCatalog {
            bundle_digest: BundleDigest::parse(&c.bundle_digest).unwrap(),
            dope_logical_composite: digest(&c.dope_logical_composite),
            dope_weights_sha256: digest(&c.dope_weights_sha256),
            assets: c
                .assets
                .into_iter()
                .map(|(path, a)| {
                    (
                        path,
                        ReleaseAsset {
                            signed_asset_id: a.signed_asset_id,
                            tensors: a.tensors.map(|t| {
                                t.into_iter()
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
        Self {
            manifest: fs::read(root.join("appliance-manifest.json")).unwrap(),
            footer: fs::read(root.join("appliance-footer.bin"))
                .unwrap()
                .try_into()
                .unwrap(),
            signature: fs::read(root.join("appliance-signature.bin"))
                .unwrap()
                .try_into()
                .unwrap(),
            catalog,
            assets,
        }
    }

    fn load(&self) -> Result<VerifiedBundle, crate::ranker::bundle::BundleIntegrityError> {
        let assets: Vec<_> = self
            .assets
            .iter()
            .map(|(path, bytes)| EmbeddedAsset { path, bytes })
            .collect();
        load_bundle(
            BundleInput::Embedded(&assets),
            SignedAssetContext {
                manifest: &self.manifest,
                footer: &self.footer,
                signature: &self.signature,
            },
            &self.catalog,
            BundleLimits::default(),
        )
    }
}

pub(super) fn load() -> VerifiedBundle {
    let bundle = Inputs::open()
        .load()
        .expect("real release signature and all asset schemas must verify");
    assert_eq!(bundle.digest().to_string(), BUNDLE);
    assert!(
        bundle.declared_qualifications().is_empty(),
        "authentication never fabricates Q1 qualification"
    );
    bundle
}

#[test]
#[ignore = "requires the exact S1 release-verified external r782 package extraction"]
fn release_signature_and_model_tampering_are_refused() {
    let mut input = Inputs::open();
    input.load().unwrap();
    input.signature[0] ^= 1;
    assert_eq!(
        input.load().unwrap_err().code(),
        ErrorCode::ModelIntegrityFailed
    );
    input.signature[0] ^= 1;
    *input
        .assets
        .get_mut("jope.safetensors")
        .unwrap()
        .last_mut()
        .unwrap() ^= 1;
    assert_eq!(
        input.load().unwrap_err().code(),
        ErrorCode::ModelIntegrityFailed
    );
}
