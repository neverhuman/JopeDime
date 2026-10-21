//! Bounds and malformed-but-authenticated input: a signed manifest that lies about sizes, a
//! frame that would decompress past its declared size, dictionaries, trailing bytes, and
//! non-canonical or traversing manifests are refused before allocation or before decoding.

use jope_assets::decode::{check_zstd_header, decode_entry};
use jope_assets::manifest::zstd_bound;
use jope_assets::verify::development_key;
use jope_assets::writer::{stage, EntryInput, StageOptions};
use jope_assets::{limits, AssetError, Codec, Entry, Frame, License, Manifest, Scope};

fn sha(bytes: &[u8]) -> String {
    jope_assets::sha256_hex(bytes)
}

/// An entry over `frames` (stored bytes, raw_len) whose stored hash matches: what a trusted but
/// faulty writer or compromised signer could produce.
fn signed_entry(codec: Codec, frames: &[(Vec<u8>, u64)], decoded: &[u8]) -> (Entry, Vec<u8>) {
    let mut blob = Vec::new();
    let mut list = Vec::new();
    let mut all = Vec::new();
    for (stored, raw_len) in frames {
        let offset = blob.len().next_multiple_of(4096) as u64;
        blob.resize(offset as usize, 0);
        blob.extend_from_slice(stored);
        all.extend_from_slice(stored);
        list.push(Frame {
            offset,
            len: stored.len() as u64,
            raw_len: *raw_len,
        });
    }
    (
        Entry {
            id: "model/x/w.bin".into(),
            role: "model".into(),
            component: "x".into(),
            origin: "x/w.bin".into(),
            sha256: sha(decoded),
            bytes: frames.iter().map(|(_, r)| r).sum(),
            codec,
            stored_sha256: sha(&all),
            stored_bytes: all.len() as u64,
            frames: list,
            license: License {
                spdx: "MIT".into(),
                redistribution: "permitted".into(),
            },
            backends: vec!["cpu".into()],
        },
        blob,
    )
}

fn decode(entry: &Entry, blob: &[u8]) -> Result<Vec<u8>, AssetError> {
    let read = |f: &Frame| Ok(blob[f.offset as usize..(f.offset + f.len) as usize].to_vec());
    decode_entry(entry, &read)
}

#[test]
fn a_frame_that_expands_past_its_declared_size_is_refused_before_decoding() {
    let big = vec![0u8; 8 << 20];
    let bomb = zstd::bulk::compress(&big, 19).unwrap();
    assert!(
        bomb.len() < 4096,
        "8 MiB of zeros compresses to a few hundred bytes"
    );
    // The manifest claims 100 bytes; the frame declares 8 MiB.
    let (entry, blob) = signed_entry(Codec::Zstd, &[(bomb, 100)], &[0u8; 100]);
    let error = decode(&entry, &blob).unwrap_err();
    assert!(
        matches!(&error, AssetError::Decode { reason, .. } if reason.contains("declares")),
        "{error}"
    );
}

#[test]
fn dictionaries_trailing_frames_and_garbage_are_refused() {
    let data = vec![3u8; 50_000];
    let frame = zstd::bulk::compress(&data, 3).unwrap();
    let mut two = frame.clone();
    two.extend_from_slice(&frame);
    let (entry, blob) = signed_entry(Codec::Zstd, &[(two, 50_000)], &data);
    assert!(matches!(
        decode(&entry, &blob),
        Err(AssetError::Decode { .. })
    ));
    let mut dict = frame.clone();
    dict[4] |= 0x01;
    assert!(check_zstd_header(&dict, 50_000).is_err());
    let (entry, blob) = signed_entry(Codec::Zstd, &[(b"not zstd".to_vec(), 8)], b"not zstd");
    assert!(matches!(
        decode(&entry, &blob),
        Err(AssetError::Decode { .. })
    ));
    let (entry, blob) = signed_entry(Codec::Raw, &[(b"short".to_vec(), 6)], b"short!");
    assert!(matches!(
        decode(&entry, &blob),
        Err(AssetError::Decode { .. })
    ));
}

#[test]
fn decoded_bytes_must_match_the_manifest_hash() {
    let data = vec![9u8; 10_000];
    let frame = zstd::bulk::compress(&data, 3).unwrap();
    let (mut entry, blob) = signed_entry(Codec::Zstd, &[(frame, 10_000)], &data);
    assert_eq!(decode(&entry, &blob).unwrap(), data);
    entry.sha256 = sha(b"something else");
    assert!(matches!(
        decode(&entry, &blob),
        Err(AssetError::EntryHash { .. })
    ));
}

fn staged_manifest() -> (Manifest, u64) {
    let options = StageOptions {
        product: "jopedime".into(),
        product_version: "0.3.0".into(),
        target: "x86_64-unknown-linux-gnu".into(),
        profile: "fixture".into(),
        scope: Scope::Development,
        source: Default::default(),
        methods: vec!["rust.ridge".into()],
        key_id: development_key().id,
        frame_bytes: 4096,
        level: 3,
    };
    let input = |path: &str, bytes: Vec<u8>| EntryInput {
        role: "model".into(),
        component: "x".into(),
        path: path.into(),
        origin: format!("x/{path}"),
        license: License {
            spdx: "MIT".into(),
            redistribution: "permitted".into(),
        },
        backends: vec!["cpu".into()],
        codec: Codec::Zstd,
        bytes,
    };
    let text: Vec<u8> = (0..20_000u32)
        .flat_map(|i| (i % 97).to_string().into_bytes())
        .collect();
    let staged = stage(
        b"\x7fELF code",
        vec![
            input("a.txt", text.clone()),
            input("b.txt", text[..9000].to_vec()),
        ],
        &options,
    )
    .unwrap();
    let payload = staged.frames.len() as u64;
    (staged.manifest, payload)
}

#[test]
fn manifest_bounds_are_enforced_before_allocation() {
    let (good, payload) = staged_manifest();
    assert!(Manifest::parse(&good.to_bytes(), payload).is_ok());
    let refuse = |m: &Manifest| Manifest::parse(&m.to_bytes(), payload).unwrap_err();

    let mut huge_frame = good.clone();
    huge_frame.entries[0].frames[0].raw_len = limits::MAX_FRAME_RAW_BYTES + 1;
    assert!(matches!(refuse(&huge_frame), AssetError::Limit(_)));

    let mut huge_entry = good.clone();
    huge_entry.entries[0].bytes = limits::MAX_ENTRY_BYTES + 1;
    assert!(matches!(refuse(&huge_entry), AssetError::Limit(_)));

    let mut swollen = good.clone();
    let raw = swollen.entries[0].frames[0].raw_len;
    swollen.entries[0].frames[0].len = zstd_bound(raw) + 1;
    assert!(matches!(refuse(&swollen), AssetError::Limit(_)));

    let mut many = good.clone();
    let entry = many.entries[0].clone();
    many.entries = (0..=limits::MAX_ENTRIES)
        .map(|i| Entry {
            id: format!("model/x/{i:05}"),
            frames: Vec::new(),
            bytes: 0,
            stored_bytes: 0,
            ..entry.clone()
        })
        .collect();
    assert!(matches!(refuse(&many), AssetError::Limit(_)));

    let mut outside = good.clone();
    outside.entries[1].frames[0].offset = payload.next_multiple_of(4096);
    assert!(matches!(refuse(&outside), AssetError::Manifest(_)));

    let mut overlapping = good.clone();
    overlapping.entries[1].frames[0].offset = overlapping.entries[0].frames[0].offset;
    assert!(matches!(refuse(&overlapping), AssetError::Manifest(_)));

    let mut unaligned = good.clone();
    unaligned.entries[0].frames[0].offset += 1;
    assert!(matches!(refuse(&unaligned), AssetError::Manifest(_)));

    let mut traversal = good.clone();
    traversal.entries[0].id = "model/../x/a.txt".into();
    assert!(matches!(refuse(&traversal), AssetError::Manifest(_)));

    let mut unsorted = good.clone();
    unsorted.entries.swap(0, 1);
    assert!(matches!(refuse(&unsorted), AssetError::Manifest(_)));

    let mut sums = good.clone();
    sums.entries[0].bytes += 1;
    assert!(matches!(refuse(&sums), AssetError::Manifest(_)));

    let mut version = good.clone();
    version.format_version = 2;
    assert!(matches!(
        refuse(&version),
        AssetError::UnsupportedVersion(2)
    ));

    // Same values, other encoding: whitespace, or an unknown field.
    let mut pretty = serde_json::to_vec_pretty(&good).unwrap();
    assert!(matches!(
        Manifest::parse(&pretty, payload),
        Err(AssetError::Manifest(_))
    ));
    pretty = good.to_bytes();
    pretty.pop();
    pretty.extend_from_slice(br#","extra":1}"#);
    assert!(matches!(
        Manifest::parse(&pretty, payload),
        Err(AssetError::Manifest(_))
    ));
}
