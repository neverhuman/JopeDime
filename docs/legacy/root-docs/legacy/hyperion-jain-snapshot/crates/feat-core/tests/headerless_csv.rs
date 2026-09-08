//! Headerless CSV ingestion. `read_frame_with(.., has_header=false)` must produce a `Frame`
//! identical to `read_frame` on the same rows prefixed with an `f0,f1,target` header, and
//! `sniff_has_header` must classify an all-numeric first row as headerless.
use feat_core::ingest::{read_frame, read_frame_with, sniff_has_header};
use std::path::PathBuf;

fn tmp(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("jain-headerless-{name}-{}", std::process::id()))
}

#[test]
fn headerless_frame_matches_equivalent_header_frame() {
    // 3 columns: f0, f1, target. Mixed scales + a negative + a float exercise tokenization.
    let rows = "0.1,2,0\n0.4,5,1\n-3,1.5,0\n";
    let with_h = tmp("hdr.csv");
    let no_h = tmp("nohdr.csv");
    std::fs::write(&with_h, format!("f0,f1,target\n{rows}")).unwrap();
    std::fs::write(&no_h, rows).unwrap();

    let a = read_frame(&with_h, &["target"], "target").unwrap();
    let b = read_frame_with(&no_h, &["target"], "target", false).unwrap();

    assert_eq!(a.n, b.n, "row counts must match");
    assert_eq!(b.n, 3, "headerless must keep row 0 as data");
    assert_eq!(a.primary, b.primary);
    assert_eq!(a.feats.len(), b.feats.len());
    assert_eq!(b.feats.len(), 2);
    for (x, y) in a.feats.iter().zip(&b.feats) {
        assert_eq!(x.name, y.name, "feature names must match");
        assert_eq!(x.kind, y.kind, "feature kinds must match");
        assert_eq!(x.tokens, y.tokens, "feature tokens must match");
    }
    assert_eq!(a.targets[0].name, b.targets[0].name);
    assert_eq!(a.targets[0].tokens, b.targets[0].tokens);
    // synthesized names: f0..f{c-2} then "target"
    assert_eq!(b.feats[0].name, "f0");
    assert_eq!(b.feats[1].name, "f1");
    assert_eq!(b.targets[0].name, "target");

    let _ = std::fs::remove_file(&with_h);
    let _ = std::fs::remove_file(&no_h);
}

#[test]
fn sniff_distinguishes_numeric_and_mixed_first_rows() {
    let numeric = tmp("sniff_num.csv");
    let mixed = tmp("sniff_mixed.csv");
    std::fs::write(&numeric, "0.6,0.8,1\n0.1,0.2,0\n").unwrap();
    std::fs::write(&mixed, "f0,f1,target\n0.1,0.2,0\n").unwrap();

    assert!(
        !sniff_has_header(&numeric).unwrap(),
        "all-numeric first row => headerless"
    );
    assert!(
        sniff_has_header(&mixed).unwrap(),
        "a non-numeric first-row token => header present"
    );

    let _ = std::fs::remove_file(&numeric);
    let _ = std::fs::remove_file(&mixed);
}
