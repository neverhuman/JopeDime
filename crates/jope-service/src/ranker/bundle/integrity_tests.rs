use super::*;
use p256::ecdsa::{signature::Signer, SigningKey};
use safetensors::tensor::{serialize, TensorView};
use serde::Serialize;

// The existing jope-assets fixture key, never part of load_bundle's trust set.
fn fixture_key() -> SigningKey {
    SigningKey::from_slice(&Sha256::digest(
        b"jope-assets development key v1 (fixture packages only, never a release)",
    ))
    .unwrap()
}

fn hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn load_limits() -> BundleLimits {
    BundleLimits {
        manifest_bytes: 16_384,
        tensor_header_bytes: 2048,
        file_bytes: 16_384,
        total_bytes: 65_536,
    }
}

fn weights(values: &[f32], buffer: f64) -> Vec<u8> {
    let values: Vec<_> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
    let buffer = buffer.to_le_bytes();
    serialize(
        [
            (
                "projection.weight",
                TensorView::new(Dtype::F32, vec![values.len() / 4], &values).unwrap(),
            ),
            (
                "running_buffer",
                TensorView::new(Dtype::F64, vec![], &buffer).unwrap(),
            ),
        ],
        None,
    )
    .unwrap()
}

fn tensor_specs() -> BTreeMap<String, TensorSpec> {
    BTreeMap::from([
        (
            "projection.weight".into(),
            TensorSpec {
                dtype: Dtype::F32,
                shape: vec![2],
            },
        ),
        (
            "running_buffer".into(),
            TensorSpec {
                dtype: Dtype::F64,
                shape: vec![],
            },
        ),
    ])
}

struct Fixture {
    files: BTreeMap<String, Vec<u8>>,
    release: ReleaseCatalog,
    signed_manifest: Vec<u8>,
    footer: [u8; 128],
    signature: [u8; 64],
}

// Field order matches jope-assets::Manifest/Entry, including unused package
// fields. This is the existing appliance fixture format, not a new envelope.
#[derive(Serialize)]
struct PackageManifest {
    schema: &'static str,
    format_version: u32,
    product: &'static str,
    product_version: &'static str,
    target: &'static str,
    profile: &'static str,
    scope: &'static str,
    code: Code,
    source: Source,
    methods: Vec<String>,
    entries: Vec<Entry>,
}
#[derive(Serialize)]
struct Code {
    len: u64,
    sha256: String,
}
#[derive(Serialize)]
struct Source {
    sha40: String,
    cargo_lock_sha256: String,
    profile_sha256: String,
    models_lock_sha256: String,
}
#[derive(Serialize)]
struct Entry {
    id: String,
    role: &'static str,
    component: &'static str,
    origin: String,
    sha256: String,
    bytes: u64,
    codec: &'static str,
    stored_sha256: String,
    stored_bytes: u64,
    frames: Vec<Frame>,
    license: License,
    backends: Vec<String>,
}
#[derive(Serialize)]
struct Frame {
    offset: u64,
    len: u64,
    raw_len: u64,
}
#[derive(Serialize)]
struct License {
    spdx: &'static str,
    redistribution: &'static str,
}

impl Fixture {
    fn new() -> Self {
        let weight_bytes = weights(&[1.25, -2.0], 0.5);
        let files = BTreeMap::from([
            (
                "manifest.json".into(),
                serde_json::to_vec(&super::tests::manifest()).unwrap(),
            ),
            ("dope.safetensors".into(), weight_bytes.clone()),
            ("jope.safetensors".into(), weight_bytes.clone()),
            ("golden/smoke.safetensors".into(), weight_bytes),
            ("featurizer.json".into(), b"{}".to_vec()),
            // An empty zstd frame; the loader authenticates, never decompresses it.
            (
                "bank.jsonl.zst".into(),
                vec![0x28, 0xb5, 0x2f, 0xfd, 0x20, 0, 1, 0, 0],
            ),
        ]);
        let release = ReleaseCatalog {
            bundle_digest: BundleDigest::parse(&format!("sha256:{}", "ab".repeat(32))).unwrap(),
            // Synthetic integrity fixture only; no real frozen-model qualification.
            dope_logical_composite: parse_sha256(FROZEN_DOPE_COMPOSITE).unwrap(),
            dope_weights_sha256: [0; 32],
            assets: files
                .keys()
                .map(|path: &String| {
                    (
                        path.clone(),
                        ReleaseAsset {
                            signed_asset_id: format!("model/fused/{path}"),
                            tensors: path.ends_with(".safetensors").then(tensor_specs),
                        },
                    )
                })
                .collect(),
        };
        let mut fixture = Self {
            files,
            release,
            signed_manifest: vec![],
            footer: [0; 128],
            signature: [0; 64],
        };
        fixture.seal();
        fixture
    }

    fn seal(&mut self) {
        let mut fused: serde_json::Value =
            serde_json::from_slice(&self.files["manifest.json"]).unwrap();
        fused["dope_weights_sha256"] = hex(&self.files["dope.safetensors"]).into();
        fused["jope_weights_sha256"] = hex(&self.files["jope.safetensors"]).into();
        self.files
            .insert("manifest.json".into(), serde_json::to_vec(&fused).unwrap());
        self.release.dope_weights_sha256 = Sha256::digest(&self.files["dope.safetensors"]).into();
        let mut offset = 0u64;
        let entries = self
            .files
            .iter()
            .map(|(path, bytes)| {
                let len = bytes.len() as u64;
                let entry = Entry {
                    id: format!("model/fused/{path}"),
                    role: "model",
                    component: "fused",
                    origin: format!("fixture/{path}"),
                    sha256: hex(bytes),
                    bytes: len,
                    codec: "raw",
                    stored_sha256: hex(bytes),
                    stored_bytes: len,
                    frames: vec![Frame {
                        offset,
                        len,
                        raw_len: len,
                    }],
                    license: License {
                        spdx: "CC0-1.0",
                        redistribution: "permitted",
                    },
                    backends: vec!["cpu".into()],
                };
                offset = (offset + len).next_multiple_of(4096);
                entry
            })
            .collect();
        self.signed_manifest = serde_json::to_vec(&PackageManifest {
            schema: "jope.appliance-manifest/v1",
            format_version: 1,
            product: "jopedime",
            product_version: "fixture",
            target: "x86_64-unknown-linux-gnu",
            profile: "fixture",
            scope: "development",
            code: Code {
                len: 1,
                sha256: hex(b"x"),
            },
            source: Source {
                sha40: "1".repeat(40),
                cargo_lock_sha256: "2".repeat(64),
                profile_sha256: "3".repeat(64),
                models_lock_sha256: "4".repeat(64),
            },
            methods: vec![],
            entries,
        })
        .unwrap();
        let at = 4096 + offset;
        let end = at + self.signed_manifest.len() as u64;
        self.footer = [0; 128];
        self.footer[..8].copy_from_slice(b"JOPEEXE1");
        self.footer[8..10].copy_from_slice(&1u16.to_le_bytes());
        self.footer[12..16].copy_from_slice(&128u32.to_le_bytes());
        self.footer[16..24].copy_from_slice(&end.to_le_bytes());
        self.footer[24..32].copy_from_slice(&1u64.to_le_bytes());
        self.footer[32..40].copy_from_slice(&at.to_le_bytes());
        self.footer[40..48].copy_from_slice(&(self.signed_manifest.len() as u64).to_le_bytes());
        self.footer[48..56].copy_from_slice(&end.to_le_bytes());
        self.footer[56..60].copy_from_slice(&64u32.to_le_bytes());
        self.footer[60..64].copy_from_slice(&1u32.to_le_bytes());
        self.footer[64..96].copy_from_slice(&Sha256::digest(
            fixture_key()
                .verifying_key()
                .to_public_key_der()
                .unwrap()
                .as_bytes(),
        ));
        self.footer[96..128].copy_from_slice(&Sha256::digest(&self.signed_manifest));
        self.sign_footer();
    }

    fn sign_footer(&mut self) {
        let mut message = SIGNING_DOMAIN.to_vec();
        message.extend_from_slice(&self.footer);
        let signature: Signature = fixture_key().sign(&message);
        self.signature = signature.normalize_s().to_bytes().into();
    }

    fn context(&self) -> SignedAssetContext<'_> {
        SignedAssetContext {
            manifest: &self.signed_manifest,
            footer: &self.footer,
            signature: &self.signature,
        }
    }

    fn embedded(&self) -> Vec<EmbeddedAsset<'_>> {
        self.files
            .iter()
            .map(|(path, bytes)| EmbeddedAsset { path, bytes })
            .collect()
    }

    fn load(&self, limits: BundleLimits) -> Result<VerifiedBundle> {
        load_with_keys(
            BundleInput::Embedded(&self.embedded()),
            self.context(),
            &self.release,
            limits,
            &[*fixture_key().verifying_key()],
        )
    }

    fn directory(&self) -> tempfile::TempDir {
        let target = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target");
        let directory = tempfile::Builder::new()
            .prefix("rnk01-")
            .tempdir_in(target)
            .unwrap();
        for (path, bytes) in &self.files {
            let path = directory.path().join(path);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, bytes).unwrap();
        }
        directory
    }

    fn load_directory(&self, path: &Path) -> Result<VerifiedBundle> {
        load_with_keys(
            BundleInput::DevelopmentDirectory(path),
            self.context(),
            &self.release,
            load_limits(),
            &[*fixture_key().verifying_key()],
        )
    }
}

pub(crate) fn test_verified_bundle() -> VerifiedBundle {
    Fixture::new().load(load_limits()).unwrap()
}

fn refuses(result: Result<VerifiedBundle>, detail: &str) {
    let error = result.unwrap_err();
    assert_eq!(error.code(), ErrorCode::ModelIntegrityFailed);
    assert_eq!(error.exit_code(), 3);
    assert!(error.detail.contains(detail), "{error}");
}

mod asset_safety;
mod signature_header;
mod tensor_contract;
