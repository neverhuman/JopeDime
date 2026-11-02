use super::*;

#[test]
fn packs_round_trip_and_verify() {
    let (packs, bodies) = write_packs(300, 1_200_000);
    assert!(
        packs.len() >= 2,
        "300 x 10 KB at a 1.2 MB target seals several packs"
    );
    let mut seen = 0usize;
    for (seq, p) in packs.iter().enumerate() {
        let v = verify(&p.bytes, &p.index).unwrap();
        assert_eq!(v.records, p.index.entries.len() as u64);
        assert!(p.index.fixture);
        assert_eq!(p.index.sequence as usize, seq);
        assert_eq!(p.index.pack_sha256, sha256_hex(&p.bytes));
        assert!(p
            .index
            .pack_key()
            .ends_with(&format!("{seq:05}-{}.zst", p.index.pack_sha256)));
        assert!(
            p.index.pack_bytes < p.index.decoded_bytes / 4,
            "repetitive bodies compress"
        );
        assert!(p.index.tags.contains(&("kind".into(), "pack".into())));
        for (i, e) in p.index.entries.iter().enumerate() {
            let body = read_entry(&p.bytes, &p.index, i).unwrap();
            assert_eq!(body, bodies[seen], "entry {} round-trips", e.key);
            assert_eq!(e.version_id.as_deref(), Some(format!("v{seen}").as_str()));
            seen += 1;
        }
        // Decoding every frame in order yields the records back to back (`zstd -d` view).
        let mut all = Vec::new();
        for c in &p.index.chunks {
            let frame = &p.bytes[c.offset as usize..(c.offset + c.length) as usize];
            all.extend(jope_records::zstd_frame::decompress(frame, c.decoded as usize).unwrap());
        }
        let expected: Vec<u8> = p
            .index
            .entries
            .iter()
            .flat_map(|e| {
                bodies[e.key[e.key.len() - 10..e.key.len() - 5]
                    .parse::<usize>()
                    .unwrap()]
                .clone()
            })
            .collect();
        assert_eq!(all, expected);
    }
    assert_eq!(seen, 300);
    // Optional external cross-check: dump the first pack and its index for `zstd -d`.
    if let Some(dir) = std::env::var_os("JOPE_RETENTION_PACK_SAMPLE") {
        let dir = std::path::PathBuf::from(dir);
        std::fs::write(dir.join("pack-sample.zst"), &packs[0].bytes).unwrap();
        std::fs::write(
            dir.join("pack-sample.index.json"),
            serde_json::to_vec_pretty(&packs[0].index).unwrap(),
        )
        .unwrap();
    }
}

#[test]
fn pack_tampering_and_misuse_are_refused() {
    let (packs, _) = write_packs(40, 10_000_000);
    let p = &packs[0];
    let mut bad = p.bytes.clone();
    let mid = bad.len() / 2;
    bad[mid] ^= 0xff;
    assert!(verify(&bad, &p.index).is_err(), "flipped pack byte");
    let mut idx = p.index.clone();
    idx.entries[3].sha256 = "0".repeat(64);
    // The pack sha still matches, so only the entry check can catch it.
    assert!(verify(&p.bytes, &idx)
        .unwrap_err()
        .to_string()
        .contains("sha256 mismatch"));
    let mut idx = p.index.clone();
    idx.entries[1].key = idx.entries[0].key.clone();
    assert!(verify(&p.bytes, &idx)
        .unwrap_err()
        .to_string()
        .contains("duplicate key"));
    let mut idx = p.index.clone();
    idx.decoded_bytes += 1;
    assert!(verify(&p.bytes, &idx).is_err());
    // Production writers refuse small targets; keys are unique per run.
    assert!(PackWriter::new("f", "r", tags(), MIN_PACK_BYTES - 1).is_err());
    assert!(PackWriter::new("f", "r", tags(), MIN_PACK_BYTES).is_ok());
    assert!(PackWriter::fixture("bad/family", "r", tags(), 10).is_err());
    let mut w = PackWriter::fixture("f", "r", tags(), 1 << 20).unwrap();
    w.add("k", None, b"a").unwrap();
    assert!(w.add("k", None, b"b").is_err());
}
