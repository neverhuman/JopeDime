mod common;

use common::*;
use jope_datasets::catalog::{CatalogRow, LicenseStatus, Origin, Task};
use jope_datasets::ingest::{self, Duplicate, Options, RejectRecord, Source};
use std::collections::BTreeMap;

#[test]
fn dry_run_reports_rows_aliases_duplicates_and_rejects() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("quant");
    build_tree(&root);
    let out = tmp.path().join("out");
    let opts = Options {
        source: Source::Quant,
        root: root.clone(),
        out: out.clone(),
        dry_run: true,
        resume: false,
        jobs: 4,
        limit: None,
        store_url: None,
    };
    let s = ingest::run(&opts).unwrap();
    assert_eq!(s.candidates, 5, "{s:?}");
    assert_eq!(s.rows, 3);
    assert_eq!(
        s.rows_by_task,
        BTreeMap::from([("binary".to_string(), 1), ("regression".to_string(), 2)])
    );
    assert_eq!(s.distinct_blobs, 9);
    assert_eq!(
        s.duplicates_by_rule,
        BTreeMap::from([
            ("binary-synthetic-duplicate".to_string(), 1),
            ("dataset-hash-duplicate".to_string(), 1),
            ("quarantine".to_string(), 1),
            ("root-level-duplicate".to_string(), 1),
        ])
    );
    assert_eq!(
        s.rejects_by_reason,
        BTreeMap::from([
            ("incomplete".to_string(), 1),
            ("meta_sha256_mismatch".to_string(), 1)
        ])
    );

    let catalog = std::fs::read_to_string(out.join("catalog.jsonl")).unwrap();
    let rows: Vec<CatalogRow> = catalog
        .lines()
        .map(|l| {
            CatalogRow::validate_value(&serde_json::from_str(l).unwrap())
                .unwrap_or_else(|e| panic!("{e:?}"))
        })
        .collect();
    assert_eq!(rows.len(), 3);
    let a = rows.iter().find(|r| r.dataset_hash == A).unwrap();
    assert_eq!(a.task, Task::Regression);
    assert_eq!(a.origin, Origin::Real);
    assert_eq!(a.provenance.license.status, LicenseStatus::Recorded);
    assert_eq!(
        a.ingest.as_ref().unwrap().relpath,
        format!("regression/{A}")
    );
    let alias_paths: Vec<&str> = a.aliases.iter().map(|x| x.path.as_str()).collect();
    assert_eq!(
        alias_paths,
        vec![
            format!("timeseries/{A}/train.csv"),
            format!("timeseries/{A}/test.csv"),
            format!("timeseries/{A}/meta.json")
        ]
    );
    assert_eq!(a.aliases[0].sha256, a.files.train.sha256);
    let b = rows.iter().find(|r| r.dataset_hash == B).unwrap();
    assert_eq!(b.id, format!("synth_{B}"));
    assert_eq!(b.origin, Origin::Synthetic);
    assert_eq!(b.generator.as_ref().unwrap().version, "2.0.0");
    let d = rows.iter().find(|r| r.dataset_hash == D).unwrap();
    assert_eq!(d.task, Task::Binary);
    assert_eq!(
        d.ingest.as_ref().unwrap().relpath,
        format!("synthetic/binary/{D}")
    );

    let dups: Vec<Duplicate> =
        serde_json::from_slice(&std::fs::read(out.join("duplicates.json")).unwrap()).unwrap();
    let root_dup = dups
        .iter()
        .find(|d| d.rule == "root-level-duplicate")
        .unwrap();
    assert_eq!(
        (root_dup.path.as_str(), root_dup.canonical.as_deref()),
        (A, Some(format!("regression/{A}").as_str()))
    );
    let bin_dup = dups
        .iter()
        .find(|d| d.rule == "binary-synthetic-duplicate")
        .unwrap();
    assert_eq!(
        bin_dup.canonical.as_deref(),
        Some(format!("synthetic/binary/{D}").as_str())
    );
    let q = dups.iter().find(|d| d.rule == "quarantine").unwrap();
    assert_eq!(q.path, format!("regression/_dupes_quarantine/{E}"));

    let rejects: Vec<RejectRecord> =
        serde_json::from_slice(&std::fs::read(out.join("rejects.json")).unwrap()).unwrap();
    let bad = rejects
        .iter()
        .find(|r| r.reason == "meta_sha256_mismatch")
        .unwrap();
    assert_eq!(bad.path, format!("regression/{C}"));
    assert_eq!(
        std::fs::metadata(out.join("ledger.jsonl")).unwrap().len(),
        0,
        "dry-run uploads nothing"
    );
    let summary: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out.join("summary.json")).unwrap()).unwrap();
    assert_eq!(summary["dry_run"], true);
}

#[test]
fn mirror_source_checks_index_hashes() {
    let tmp = tempfile::tempdir().unwrap();
    let mirror = tmp.path().join("mirror");
    let quant = mirror
        .join("dataset")
        .join("home")
        .join("ubuntu")
        .join("quant");
    build_tree(&quant);
    let mut index = serde_json::Map::new();
    for (class, name) in [
        ("regression", A.to_string()),
        ("regression", format!("synth_{B}")),
        ("synthetic/binary", D.to_string()),
    ] {
        for f in ["train.csv", "test.csv", "meta.json"] {
            let p = quant.join(class).join(&name).join(f);
            let sha = jope_datasets::hash::digest_file(&p).unwrap().sha256;
            index.insert(
                format!("home/ubuntu/quant/{class}/{name}/{f}"),
                serde_json::Value::String(sha),
            );
        }
    }
    // A corrupted index entry must reject the dataset.
    index.insert(
        format!("home/ubuntu/quant/synthetic/binary/{D}/test.csv"),
        serde_json::Value::String("0".repeat(64)),
    );
    std::fs::create_dir_all(mirror.join("index")).unwrap();
    std::fs::write(
        mirror.join("index").join("dataset.json"),
        serde_json::to_vec(&index).unwrap(),
    )
    .unwrap();
    let out = tmp.path().join("out");
    let opts = Options {
        source: Source::Mirror,
        root: mirror,
        out,
        dry_run: true,
        resume: false,
        jobs: 2,
        limit: None,
        store_url: None,
    };
    let s = ingest::run(&opts).unwrap();
    assert_eq!((s.candidates, s.rows, s.rejects), (3, 2, 1), "{s:?}");
    assert_eq!(s.rejects_by_reason.get("mirror_sha_mismatch"), Some(&1));
}
