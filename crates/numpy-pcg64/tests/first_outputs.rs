//! First SeedSequence/PCG64 outputs for the two seeds behind the frozen
//! recipes. These are regression pins for this port. External numpy 1.26.4
//! anchoring is `vendored_subset_matches_its_sources_when_available` when
//! `NUMPY_PCG64_JOPE_DIME_GIT` can read the git-store source.

use numpy_pcg64::{default_rng, Pcg64, SeedSequence};

#[test]
fn foundation_seed_20260707() {
    let seq = SeedSequence::new(20_260_707);
    assert_eq!(seq.pool(), [0xc396c7d7, 0x215b80a5, 0x9424ac70, 0xe743c5ee]);
    assert_eq!(
        seq.generate_state_u64(4),
        [
            0xcfe6074af264859e,
            0x00dc3f1f4b0b929e,
            0x874189bdfc4318c9,
            0x8197e854de03d396
        ]
    );
    let mut bits = Pcg64::from_seed_sequence(&seq);
    assert_eq!(
        [bits.next_u64(), bits.next_u64(), bits.next_u64()],
        [0x0d3d51ef55230caf, 0x5dc11455ecf1dc89, 0xd9908ea4df634232]
    );
    assert_eq!(
        default_rng(20_260_707).permutation(10),
        [8, 7, 1, 3, 6, 0, 4, 2, 5, 9]
    );
}

#[test]
fn target_seed_17() {
    let seq = SeedSequence::new(17);
    assert_eq!(seq.pool(), [0xef3fa155, 0xb380a4f8, 0x0f061797, 0x9e34f4c0]);
    assert_eq!(
        seq.generate_state_u64(4),
        [
            0x13beeb3b1cb825f0,
            0xb7edfa403a87e835,
            0x1da995accabd6a05,
            0x1da933e483c0dfc2
        ]
    );
    let mut bits = Pcg64::from_seed_sequence(&seq);
    assert_eq!(
        [bits.next_u64(), bits.next_u64(), bits.next_u64()],
        [0xd856d255bdafcc73, 0x293588521b7ab04b, 0x8ec858be753d914a]
    );
    assert_eq!(
        default_rng(17).permutation(10),
        [4, 0, 1, 7, 8, 6, 2, 9, 5, 3]
    );
}
