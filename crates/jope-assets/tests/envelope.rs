//! Real packages: round trip, determinism, deduplication, and a refusal for every region of the
//! file, truncation, extension, a wrong signer and a resealed manifest (SPECIFICATION.md §17.1).

use jope_assets::verify::{development_key, normalize_der, sign_with_development_key};
use jope_assets::writer::{
    assemble, assemble_development, stage, write_new, EntryInput, StageOptions,
};
use jope_assets::{AssetError, Codec, License, Package, Scope, SourceIdentity};
use std::path::{Path, PathBuf};

fn license() -> License {
    License {
        spdx: "LicenseRef-Fixture".into(),
        redistribution: "permitted".into(),
    }
}

fn safetensors_f32(values: usize, scale: f32) -> Vec<u8> {
    let header = format!(
        "{{\"w\":{{\"dtype\":\"F32\",\"shape\":[{values}],\"data_offsets\":[0,{}]}}}}",
        values * 4
    );
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header.as_bytes());
    for i in 0..values {
        out.extend_from_slice(&((i as f32) * scale - 1.0).to_le_bytes());
    }
    out
}

fn input(component: &str, path: &str, bytes: Vec<u8>) -> EntryInput {
    EntryInput {
        role: "model".into(),
        component: component.into(),
        path: path.into(),
        origin: format!("fixture-{component}/{path}"),
        license: license(),
        backends: vec!["cpu".into()],
        codec: Codec::ShuffleZstd,
        bytes,
    }
}

fn inputs() -> Vec<EntryInput> {
    vec![
        input(
            "policy",
            "config.json",
            br#"{"version":"fixture"}"#.to_vec(),
        ),
        input(
            "encoder",
            "weights.safetensors",
            safetensors_f32(50_000, 0.001),
        ),
        // The same bytes under a second logical role are stored once.
        input(
            "encoder-alias",
            "weights.safetensors",
            safetensors_f32(50_000, 0.001),
        ),
        input(
            "policy",
            "policy.safetensors",
            safetensors_f32(30_000, 0.01),
        ),
    ]
}

fn options() -> StageOptions {
    StageOptions {
        product: "jopedime".into(),
        product_version: "0.3.0".into(),
        target: "x86_64-unknown-linux-gnu".into(),
        profile: "fixture".into(),
        scope: Scope::Development,
        source: SourceIdentity {
            sha40: "0".repeat(40),
            ..SourceIdentity::default()
        },
        methods: vec!["rust.ridge".into(), "rust.gpr.rbf_linear".into()],
        key_id: development_key().id,
        frame_bytes: 64 << 10,
        level: 3,
    }
}

fn code() -> Vec<u8> {
    let mut code = b"\x7fELF".to_vec();
    code.extend((0..10_000u32).map(|i| (i % 253) as u8));
    code
}

fn build(dir: &Path) -> PathBuf {
    let code = code();
    let staged = stage(&code, inputs(), &options()).unwrap();
    let path = dir.join("jopedime");
    write_new(&assemble_development(&code, &staged).unwrap(), &path).unwrap();
    path
}

fn trusted() -> Vec<jope_assets::TrustedKey> {
    vec![development_key()]
}

#[cfg(target_os = "macos")]
#[test]
fn open_self_resolves_the_running_executable_on_macos() {
    // Cargo's test executable is not an asset package; reaching it proves the
    // portable executable lookup succeeded and correctly reports no footer.
    assert!(Package::open_self().unwrap().is_none());
}

#[test]
fn round_trip_returns_every_asset_exactly_and_covers_every_byte() {
    let dir = tempfile::tempdir().unwrap();
    let path = build(dir.path());
    let package = Package::open_with_keys(&path, &trusted()).unwrap().unwrap();
    for input in inputs() {
        let verified = package.read_entry(&input.id()).unwrap();
        assert_eq!(verified.bytes(), input.bytes.as_slice());
        assert_eq!(verified.id(), input.id());
    }
    let manifest = package.manifest();
    assert_eq!(manifest.methods, ["rust.gpr.rbf_linear", "rust.ridge"]);
    let encoder = manifest.entry("model/encoder/weights.safetensors").unwrap();
    let alias = manifest
        .entry("model/encoder-alias/weights.safetensors")
        .unwrap();
    assert_eq!(
        encoder.frames, alias.frames,
        "identical content is stored once"
    );
    assert_eq!(encoder.codec, Codec::ShuffleZstd);
    assert_eq!(
        manifest.entry("model/policy/config.json").unwrap().codec,
        Codec::Raw
    );
    let report = package.verify_full().unwrap();
    assert_eq!(report.entries, 4);
    assert_eq!(report.file_bytes, std::fs::metadata(&path).unwrap().len());
    assert_eq!(
        report.file_sha256,
        jope_assets::sha256_hex(&std::fs::read(&path).unwrap())
    );
    assert!(report.stored_bytes < report.decoded_bytes);
    assert_eq!(package.signer().role, "development");
}

#[test]
fn packing_is_deterministic_and_independent_of_input_order() {
    let a = tempfile::tempdir().unwrap();
    let b = tempfile::tempdir().unwrap();
    let first = std::fs::read(build(a.path())).unwrap();
    let code = code();
    let mut reversed = inputs();
    reversed.reverse();
    let staged = stage(&code, reversed, &options()).unwrap();
    let second = assemble_development(&code, &staged).unwrap();
    assert_eq!(first, second);
    let path = b.path().join("again");
    write_new(&second, &path).unwrap();
    assert!(write_new(&second, &path).is_err(), "never overwrites");
}

#[test]
fn an_unpackaged_file_has_no_package() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("plain");
    std::fs::write(&path, vec![1u8; 5000]).unwrap();
    assert!(Package::open_with_keys(&path, &trusted())
        .unwrap()
        .is_none());
    let tiny = dir.path().join("tiny");
    std::fs::write(&tiny, b"x").unwrap();
    assert!(Package::open_with_keys(&tiny, &trusted())
        .unwrap()
        .is_none());
}

/// Flip one byte in each region and expect the matching refusal.
#[test]
fn every_tampered_region_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let path = build(dir.path());
    let good = std::fs::read(&path).unwrap();
    let package = Package::open_with_keys(&path, &trusted()).unwrap().unwrap();
    let footer = *package.footer();
    let entry = package
        .entry("model/policy/policy.safetensors")
        .unwrap()
        .clone();
    drop(package);
    let payload = footer.payload_offset();
    let flip = |at: u64| {
        let mut bytes = good.clone();
        bytes[at as usize] ^= 0x40;
        let p = dir.path().join(format!("tampered-{at}"));
        std::fs::write(&p, &bytes).unwrap();
        p
    };
    let open = |p: &Path| Package::open_with_keys(p, &trusted());

    // Executable: authenticated on every open, so it cannot be swapped under a signed manifest.
    assert!(matches!(open(&flip(10)), Err(AssetError::CodeHash)));
    // Padding after the executable.
    let package = open(&flip(footer.base_executable_len + 1))
        .unwrap()
        .unwrap();
    assert!(matches!(package.verify_full(), Err(AssetError::Padding)));
    // A gap between frames.
    let frames = {
        let p = open(&path).unwrap().unwrap();
        p.manifest().unique_frames()
    };
    let gap = frames
        .windows(2)
        .find(|w| w[0].offset + w[0].len < w[1].offset)
        .map(|w| payload + w[0].offset + w[0].len)
        .expect("aligned frames leave a gap");
    let package = open(&flip(gap)).unwrap().unwrap();
    assert!(matches!(package.verify_full(), Err(AssetError::Padding)));
    // A frame: refused on its stored hash, before the decompressor sees it.
    let frame = entry.frames[0];
    let package = open(&flip(payload + frame.offset + frame.len / 2))
        .unwrap()
        .unwrap();
    assert!(matches!(
        package.read_entry(&entry.id),
        Err(AssetError::StoredHash { .. })
    ));
    assert!(
        package.read_entry("model/policy/config.json").is_ok(),
        "other assets still read"
    );
    // The manifest, the signature, and the footer's manifest hash and key id.
    assert!(matches!(
        open(&flip(footer.manifest_offset + 5)),
        Err(AssetError::ManifestHash)
    ));
    assert!(matches!(
        open(&flip(footer.signed_prefix_len() + 10)),
        Err(AssetError::Signature)
    ));
    let len = good.len() as u64;
    assert!(matches!(
        open(&flip(len - 1)),
        Err(AssetError::ManifestHash)
    ));
    // A footer field the layout equations still accept is caught by the signature.
    assert!(matches!(
        open(&flip(len - 128 + 24)),
        Err(AssetError::Signature)
    ));
    assert!(matches!(
        open(&flip(len - 128 + 64)),
        Err(AssetError::UntrustedKey(_))
    ));
    // Footer layout fields.
    assert!(matches!(
        open(&flip(len - 128 + 33)),
        Err(AssetError::Layout(_))
    ));
    assert!(matches!(
        open(&flip(len - 128 + 10)),
        Err(AssetError::Footer(_))
    ));
    // Truncation and an appended copy of the footer.
    let truncated = dir.path().join("truncated");
    std::fs::write(&truncated, &good[..good.len() - 100]).unwrap();
    assert!(open(&truncated).map(|p| p.is_none()).unwrap_or(true));
    let extended = dir.path().join("extended");
    let mut bytes = good.clone();
    let tail = good[good.len() - 128..].to_vec();
    bytes.extend_from_slice(&tail);
    std::fs::write(&extended, &bytes).unwrap();
    assert!(matches!(open(&extended), Err(AssetError::Layout(_))));
    // An untrusted signer.
    assert!(matches!(
        Package::open_with_keys(&path, &[]),
        Err(AssetError::UntrustedKey(_))
    ));
}

/// A resealed manifest (new hash in the footer) is refused unless re-signed by a trusted key,
/// and a signature over the bytes of another package does not transfer.
#[test]
fn a_signature_binds_one_footer() {
    let dir = tempfile::tempdir().unwrap();
    let code = code();
    let staged = stage(&code, inputs(), &options()).unwrap();
    let mut other = options();
    other.profile = "native-only".into();
    let other = stage(&code, inputs(), &other).unwrap();
    let foreign = normalize_der(&sign_with_development_key(&other.signing_message())).unwrap();
    assert!(assemble(&code, &staged, &foreign, &development_key()).is_err());
    // A signature over the raw footer (no domain prefix) does not verify either.
    let bare = normalize_der(&sign_with_development_key(&staged.footer.encode())).unwrap();
    assert!(assemble(&code, &staged, &bare, &development_key()).is_err());
    // The wrong executable is refused before signing matters.
    let good = normalize_der(&sign_with_development_key(&staged.signing_message())).unwrap();
    assert!(assemble(b"other code", &staged, &good, &development_key()).is_err());
    let bytes = assemble(&code, &staged, &good, &development_key()).unwrap();
    let path = dir.path().join("ok");
    write_new(&bytes, &path).unwrap();
    assert!(Package::open_with_keys(&path, &trusted())
        .unwrap()
        .is_some());
}

#[test]
fn unknown_ids_are_missing_and_the_trait_reads_the_same_bytes() {
    use jope_assets::VerifiedAssetSource;
    let dir = tempfile::tempdir().unwrap();
    let path = build(dir.path());
    let package = Package::open_with_keys(&path, &trusted()).unwrap().unwrap();
    assert!(matches!(
        package.read_entry("model/policy/../policy.safetensors"),
        Err(AssetError::EntryMissing(_))
    ));
    let source: &dyn VerifiedAssetSource = &package;
    let id = "model/policy/policy.safetensors";
    assert_eq!(
        source.read_verified(id).unwrap().bytes(),
        package.read_entry(id).unwrap().bytes()
    );
    assert_eq!(source.manifest().profile, "fixture");
}

/// The development key signs only development-scope packages, and a release key never does.
#[test]
fn signer_role_must_match_the_scope() {
    let dir = tempfile::tempdir().unwrap();
    let code = code();
    let mut public = options();
    public.scope = Scope::Public;
    let staged = stage(&code, inputs(), &public).unwrap();
    let path = dir.path().join("public-by-development-key");
    write_new(&assemble_development(&code, &staged).unwrap(), &path).unwrap();
    assert!(matches!(
        Package::open_with_keys(&path, &trusted()),
        Err(AssetError::Manifest(reason)) if reason.contains("may not sign")
    ));
    // The same key presented as a release key may not sign a development package.
    let fixture = build(dir.path());
    let as_release =
        jope_assets::TrustedKey::from_pem(&development_key().to_pem(), "release").unwrap();
    assert!(matches!(
        Package::open_with_keys(&fixture, std::slice::from_ref(&as_release)),
        Err(AssetError::Manifest(_))
    ));
    let release_signed = dir.path().join("public-by-release-key");
    write_new(
        &assemble_development(&code, &staged).unwrap(),
        &release_signed,
    )
    .unwrap();
    assert!(Package::open_with_keys(&release_signed, &[as_release])
        .unwrap()
        .is_some());
}
