use jope_parallel::corpus::{Record, Records, RECORD_BYTES, SEQUENCE};
#[test]
fn packed_labels_preserve_values_and_reject_truncation() {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("records.bin");
    let mut r = Record {
        tokens: [0; SEQUENCE],
        legal: [0; SEQUENCE],
        length: 17,
        context: 23,
        value: 0.42,
        percentile: 0.91,
        runtime: 1.25,
    };
    r.tokens[0] = 1;
    r.tokens[16] = 2;
    r.legal[5] = 312;
    assert_eq!(r.bytes().len(), RECORD_BYTES);
    std::fs::write(&p, r.bytes()).unwrap();
    let m = Records::open(&p, 1).unwrap();
    let s = m.get(0);
    assert_eq!(s.tokens, r.tokens);
    assert_eq!(s.legal, r.legal);
    assert_eq!(s.context, 23);
    assert_eq!(s.value.to_bits(), r.value.to_bits());
    assert_eq!(s.length, 17);
    assert!(Records::open(&p, 2).is_err());
}
#[cfg(feature = "neural")]
#[test]
fn sampler_balances_families_and_keeps_comparisons_on_rank() {
    use jope_parallel::{
        corpus::{Group, Index},
        training::sample,
    };
    use std::collections::BTreeMap;
    let mut groups = Vec::new();
    let mut families = BTreeMap::<String, Vec<usize>>::new();
    let mut count = 0;
    for (family, cuts) in [("a", 1), ("b", 7), ("c", 2)] {
        for cut in 0..cuts {
            let records = (count..count + 16).collect();
            count += 16;
            families
                .entry(family.into())
                .or_default()
                .push(groups.len());
            groups.push(Group {
                family: family.into(),
                partition: "meta_train".into(),
                context: cut,
                records,
                max_length: 16,
            });
        }
    }
    let index = Index {
        schema: "test".into(),
        qualification: true,
        vocabulary_sha256: String::new(),
        context_index_sha256: String::new(),
        records_sha256: String::new(),
        record_bytes: RECORD_BYTES,
        records: count,
        legal_sets: vec![],
        contexts: vec![],
        groups,
    };
    let mut counts = BTreeMap::<String, usize>::new();
    for serial in (0..2400).step_by(8) {
        let ids = sample(&index, &families, serial, 8, 12);
        let g = index
            .groups
            .iter()
            .find(|g| g.records.contains(&ids[0]))
            .unwrap();
        assert!(ids.iter().all(|id| g.records.contains(id)));
        assert_eq!(ids[0], g.records[0], "every comparison must include its best observed teacher");
        assert_eq!(ids.iter().collect::<std::collections::BTreeSet<_>>().len(),8,"comparators must be distinct");
        *counts.entry(g.family.clone()).or_default() += 1;
        assert_eq!(ids, sample(&index, &families, serial, 8, 12));
    }
    assert_eq!(
        counts.values().copied().collect::<Vec<_>>(),
        vec![100, 100, 100]
    );
    let global = sample(&index, &families, 2048, 2048, 12);
    let ranks = (0..8)
        .flat_map(|r| sample(&index, &families, 2048 + r * 256, 256, 12))
        .collect::<Vec<_>>();
    assert_eq!(global, ranks);
    // An equal example budget preserves the ordered teacher/comparator stream
    // when a qualification block changes the number of optimizer updates.
    let expected = sample(&index, &families, 0, 32768, 12);
    for microbatch in [256, 512, 1024, 2048] {
        let global_batch = 8 * microbatch;
        let mut actual = Vec::new();
        for step in 0..32768 / global_batch {
            for rank in 0..8 {
                actual.extend(sample(&index, &families,
                    step * global_batch + rank * microbatch, microbatch, 12));
            }
        }
        assert_eq!(actual, expected, "batch {microbatch} changed the sampled teaching examples");
    }
}
