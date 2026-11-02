use super::*;

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

pub(super) fn test_verified_bundle() -> VerifiedBundle {
    Fixture::new().load(load_limits()).unwrap()
}

fn refuses(result: Result<VerifiedBundle>, detail: &str) {
    let error = result.unwrap_err();
    assert_eq!(error.code(), ErrorCode::ModelIntegrityFailed);
    assert_eq!(error.exit_code(), 3);
    assert!(error.detail.contains(detail), "{error}");
}

#[test]
fn signed_embedded_and_development_directory_load_exact_bytes() {
    let fixture = Fixture::new();
    let embedded = fixture.load(load_limits()).unwrap();
    assert_eq!(embedded.digest(), fixture.release.bundle_digest);
    assert_eq!(
        embedded.declared_qualifications()[0].task,
        QualifiedTask::Regression
    );
    assert_eq!(
        embedded.asset("dope.safetensors").unwrap(),
        fixture.files["dope.safetensors"]
    );
    let directory = fixture.directory();
    let loaded = fixture.load_directory(directory.path()).unwrap();
    assert_eq!(loaded.digest(), embedded.digest());
    for (path, bytes) in &fixture.files {
        assert_eq!(loaded.asset(path).unwrap(), bytes);
    }
    // The production API cannot opt into the fixture signer.
    refuses(
        load_bundle(
            BundleInput::Embedded(&fixture.embedded()),
            fixture.context(),
            &fixture.release,
            load_limits(),
        ),
        "untrusted package signer",
    );
}

#[test]
fn m4_wrong_hash_is_refused_with_integrity_exit3() {
    let mut fixture = Fixture::new();
    fixture.files.get_mut("jope.safetensors").unwrap()[8] ^= 1;
    refuses(fixture.load(load_limits()), "SHA-256 mismatch");
}

#[test]
fn m4_signed_corrupt_header_is_refused_with_integrity_exit3() {
    let mut fixture = Fixture::new();
    fixture.files.get_mut("jope.safetensors").unwrap()[..8]
        .copy_from_slice(&u64::MAX.to_le_bytes());
    fixture.seal();
    refuses(fixture.load(load_limits()), "header exceeds");
}

#[test]
fn signed_duplicate_tensor_header_keys_are_refused() {
    let mut header = br#"{"projection.weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"running_buffer":{"dtype":"F64","shape":[],"data_offsets":[8,16]},"projection.weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#.to_vec();
    header.resize(header.len().next_multiple_of(8), b' ');
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend(1.0f32.to_le_bytes());
    bytes.extend(2.0f32.to_le_bytes());
    bytes.extend(0.5f64.to_le_bytes());
    // The library's map projection alone collapses the duplicate name.
    assert!(SafeTensors::deserialize(&bytes).is_ok());
    let mut fixture = Fixture::new();
    fixture.files.insert("jope.safetensors".into(), bytes);
    fixture.seal();
    refuses(fixture.load(load_limits()), "duplicate");
}

#[test]
fn m4_signed_nan_weight_and_infinite_buffer_are_refused_with_integrity_exit3() {
    for (values, buffer) in [
        (vec![f32::NAN, 1.0], 0.5),
        (vec![1.0, 2.0], f64::INFINITY),
        (vec![f32::NEG_INFINITY, 2.0], 0.5),
    ] {
        let mut fixture = Fixture::new();
        fixture
            .files
            .insert("jope.safetensors".into(), weights(&values, buffer));
        fixture.seal();
        refuses(fixture.load(load_limits()), "nonfinite");
    }
}

#[test]
fn m4_signed_wrong_shape_and_oversize_are_refused_with_integrity_exit3() {
    let mut fixture = Fixture::new();
    fixture
        .files
        .insert("jope.safetensors".into(), weights(&[1.0], 0.5));
    fixture.seal();
    refuses(fixture.load(load_limits()), "shape or dtype mismatch");
    let fixture = Fixture::new();
    refuses(
        fixture.load(BundleLimits {
            file_bytes: 16,
            ..load_limits()
        }),
        "file byte cap",
    );
    refuses(
        fixture.load(BundleLimits {
            total_bytes: 16,
            ..load_limits()
        }),
        "total byte cap",
    );
}

#[test]
fn exact_tensor_names_dtypes_and_buffers_are_mandatory() {
    for change in [0, 1, 2, 3] {
        let mut fixture = Fixture::new();
        let specs = fixture
            .release
            .assets
            .get_mut("jope.safetensors")
            .unwrap()
            .tensors
            .as_mut()
            .unwrap();
        match change {
            0 => {
                specs.remove("running_buffer");
            }
            1 => {
                let spec = specs.remove("running_buffer").unwrap();
                specs.insert("other_buffer".into(), spec);
            }
            2 => {
                specs.get_mut("projection.weight").unwrap().dtype = Dtype::F16;
            }
            _ => {
                fixture
                    .release
                    .assets
                    .get_mut("jope.safetensors")
                    .unwrap()
                    .tensors = None;
            }
        }
        let error = fixture.load(load_limits()).unwrap_err();
        assert_eq!(error.code(), ErrorCode::ModelIntegrityFailed);
        assert_eq!(error.exit_code(), 3);
    }
}

#[test]
fn signature_hash_key_footer_and_high_s_are_mandatory() {
    let mut fixture = Fixture::new();
    fixture.signature[0] ^= 1;
    refuses(fixture.load(load_limits()), "signature");
    fixture.seal();
    fixture.signed_manifest[0] ^= 1;
    refuses(fixture.load(load_limits()), "catalog SHA-256 mismatch");
    fixture.seal();
    fixture.footer[64] ^= 1;
    refuses(fixture.load(load_limits()), "untrusted package signer");
    fixture.seal();
    fixture.footer[60] = 2;
    refuses(fixture.load(load_limits()), "footer");
    fixture.seal();
    let low = Signature::from_slice(&fixture.signature).unwrap();
    let (r, s) = low.split_scalars();
    fixture.signature = Signature::from_scalars(r, -*s).unwrap().to_bytes().into();
    refuses(
        fixture.load(load_limits()),
        "noncanonical package signature",
    );
}

#[test]
fn unknown_architecture_and_changed_release_pins_are_refused() {
    let mut fixture = Fixture::new();
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&fixture.files["manifest.json"]).unwrap();
    manifest["architecture"] = "unknown-architecture".into();
    fixture.files.insert(
        "manifest.json".into(),
        serde_json::to_vec(&manifest).unwrap(),
    );
    fixture.seal();
    refuses(fixture.load(load_limits()), "architecture");
    let mut fixture = Fixture::new();
    fixture.release.bundle_digest = BundleDigest([7; 32]);
    refuses(fixture.load(load_limits()), "release identities");
    fixture.release.dope_logical_composite[0] ^= 1;
    refuses(fixture.load(load_limits()), "frozen DOPE");
    let mut fixture = Fixture::new();
    fixture.release.dope_weights_sha256[0] ^= 1;
    refuses(fixture.load(load_limits()), "custody pin");
}

#[test]
fn float_masks_refuse_nan_inf_in_every_supported_width() {
    for (dtype, finite, bad) in [
        (
            Dtype::F64,
            1.0f64.to_le_bytes().to_vec(),
            f64::NAN.to_le_bytes().to_vec(),
        ),
        (
            Dtype::F32,
            1.0f32.to_le_bytes().to_vec(),
            f32::INFINITY.to_le_bytes().to_vec(),
        ),
        (
            Dtype::F16,
            0x3c00u16.to_le_bytes().to_vec(),
            0x7e00u16.to_le_bytes().to_vec(),
        ),
        (
            Dtype::BF16,
            0x3f80u16.to_le_bytes().to_vec(),
            0xff80u16.to_le_bytes().to_vec(),
        ),
    ] {
        check_finite(&finite, dtype).unwrap();
        assert_eq!(check_finite(&bad, dtype).unwrap_err().exit_code(), 3);
    }
    assert!(check_finite(&[0xff; 8], Dtype::I64).is_ok());
    assert!(check_finite(&[0], Dtype::F8_E5M2).is_err());
}

#[test]
fn missing_extra_duplicate_or_escaping_assets_are_refused() {
    let fixture = Fixture::new();
    let mut embedded = fixture.embedded();
    embedded.pop();
    refuses(
        load_with_keys(
            BundleInput::Embedded(&embedded),
            fixture.context(),
            &fixture.release,
            load_limits(),
            &[*fixture_key().verifying_key()],
        ),
        "inventory",
    );
    for path in [
        fixture.files.first_key_value().unwrap().0.as_str(),
        "unlisted.json",
    ] {
        let mut embedded = fixture.embedded();
        embedded.push(EmbeddedAsset { path, bytes: b"{}" });
        refuses(
            load_with_keys(
                BundleInput::Embedded(&embedded),
                fixture.context(),
                &fixture.release,
                load_limits(),
                &[*fixture_key().verifying_key()],
            ),
            "inventory",
        );
    }
    let mut fixture = Fixture::new();
    let spec = fixture.release.assets.remove("featurizer.json").unwrap();
    fixture.release.assets.insert("../outside".into(), spec);
    refuses(fixture.load(load_limits()), "required asset");
    for path in [
        "../outside",
        "a//b",
        "a/./b",
        "a/../b",
        "/absolute",
        "a\\b",
        "a:b",
    ] {
        assert!(!safe_path(path));
    }
}

#[cfg(unix)]
#[test]
fn directory_refuses_symlinks_and_fifo_without_a_writer_promptly() {
    use std::os::unix::fs::symlink;
    let fixture = Fixture::new();
    let directory = fixture.directory();
    let path = directory.path().join("featurizer.json");
    std::fs::remove_file(&path).unwrap();
    symlink("manifest.json", &path).unwrap();
    refuses(fixture.load_directory(directory.path()), "without links");
    std::fs::remove_file(&path).unwrap();
    nix::unistd::mkfifo(
        &path,
        nix::sys::stat::Mode::S_IRUSR | nix::sys::stat::Mode::S_IWUSR,
    )
    .unwrap();
    let start = std::time::Instant::now();
    refuses(fixture.load_directory(directory.path()), "nonregular");
    assert!(start.elapsed() < std::time::Duration::from_secs(1));
    let linked = directory.path().join("root-link");
    symlink(directory.path().canonicalize().unwrap(), &linked).unwrap();
    assert!(DirectoryReader::open(&linked).is_err());
    let golden = directory.path().join("golden");
    std::fs::rename(&golden, directory.path().join("saved-golden")).unwrap();
    symlink("saved-golden", &golden).unwrap();
    // Restore the earlier file so traversal reaches the intermediate link.
    std::fs::remove_file(&path).unwrap();
    std::fs::write(&path, &fixture.files["featurizer.json"]).unwrap();
    refuses(fixture.load_directory(directory.path()), "without links");
}
