use super::*;

use super::*;

fn sample() -> Footer {
    Footer {
        base_executable_len: 10_000,
        manifest_offset: 20_480,
        manifest_len: 300,
        key_id: KeyId([9; 32]),
        manifest_sha256: [7; 32],
    }
}

#[test]
fn footer_round_trips_with_the_specified_offsets() {
    let footer = sample();
    let bytes = footer.encode();
    assert_eq!(&bytes[0..8], b"JOPEEXE1");
    assert_eq!(
        u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
        20_780
    );
    assert_eq!(
        u64::from_le_bytes(bytes[48..56].try_into().unwrap()),
        20_780
    );
    assert_eq!(footer.payload_offset(), 12_288);
    assert_eq!(footer.file_len(), 20_780 + 64 + 128);
    assert_eq!(
        Footer::decode(&bytes, footer.file_len()).unwrap(),
        Some(footer)
    );
}

#[test]
fn missing_magic_means_unpackaged() {
    assert_eq!(Footer::decode(&[0u8; 128], 128).unwrap(), None);
}

#[test]
fn every_footer_rule_is_enforced() {
    let good = sample().encode();
    let len = sample().file_len();
    let mutate = |at: usize, value: u8| {
        let mut bytes = good;
        bytes[at] = value;
        Footer::decode(&bytes, len)
    };
    assert!(matches!(
        mutate(8, 2),
        Err(AssetError::UnsupportedVersion(2))
    ));
    assert!(matches!(mutate(10, 1), Err(AssetError::Footer(_))));
    assert!(matches!(mutate(12, 64), Err(AssetError::Footer(_))));
    assert!(matches!(mutate(16, 0), Err(AssetError::Layout(_))));
    assert!(matches!(mutate(48, 0), Err(AssetError::Layout(_))));
    assert!(matches!(mutate(56, 72), Err(AssetError::Footer(_))));
    assert!(matches!(mutate(60, 2), Err(AssetError::Footer(_))));
    assert!(matches!(
        Footer::decode(&good, len + 1),
        Err(AssetError::Layout(_))
    ));
    let mut early = sample();
    early.manifest_offset = 4096;
    assert!(matches!(
        Footer::decode(&early.encode(), early.file_len()),
        Err(AssetError::Layout(_))
    ));
    let mut huge = sample();
    huge.manifest_len = limits::MAX_MANIFEST_BYTES + 1;
    assert!(matches!(
        Footer::decode(&huge.encode(), huge.file_len()),
        Err(AssetError::Limit(_))
    ));
    let mut overflow = good;
    overflow[32..40].copy_from_slice(&(u64::MAX - 10).to_le_bytes());
    assert!(matches!(
        Footer::decode(&overflow, len),
        Err(AssetError::Layout(_))
    ));
}
