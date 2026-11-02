//! Genebank (a 20-row SQLite built here), a lineage-A snapshot (manifest + streamed rows) and
//! SOTArena results (two track versions) through the binary.

mod common;

use common::jl;
use jope_learning::canon::sha256_hex;
use jope_learning::sources::{genebank, snapshot, sotarena};
use jope_learning::state::State;
use serde_json::json;
use std::fs;

fn exact_current(
    state: &State,
    task: &str,
    dataset: &str,
    cut: &str,
) -> jope_learning::record::HwmRecord {
    let mut records = state.pointers().unwrap().into_iter().filter(|(key, _)| {
        key.task == task
            && key.dataset_hash == dataset
            && key.cut_id == cut
            && key.identity_sha256.is_some()
    });
    let record = records.next().unwrap().1;
    assert!(
        records.next().is_none(),
        "fixture must name one full comparison identity"
    );
    record
}

fn build_genebank(path: &std::path::Path) {
    let conn = rusqlite::Connection::open(path).unwrap();
    conn.execute_batch(
        "CREATE TABLE best_scores(task TEXT, dataset_hash TEXT, score_tier TEXT, metric TEXT, \
         score REAL, semantic_sha256 TEXT, canonical_genome TEXT, achieved_at_ms INTEGER, \
         receipt_ref TEXT, PRIMARY KEY(task,dataset_hash,score_tier));",
    )
    .unwrap();
    let mut n = 0;
    for i in 0..10u32 {
        for (tier, metric, task) in [
            ("cv_robust", "oof_r2", "regression"),
            (
                "test_exact",
                if i % 2 == 0 { "r2" } else { "auc" },
                if i % 2 == 0 { "regression" } else { "binary" },
            ),
        ] {
            n += 1;
            let ds = format!("{:016x}", 0x0259_e0b6_af04_3f58u64 + u64::from(i));
            let sem = if n == 3 {
                String::new()
            } else {
                sha256_hex(format!("sem{n}").as_bytes())
            };
            conn.execute(
                "INSERT INTO best_scores VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9)",
                rusqlite::params![
                    task,
                    ds,
                    tier,
                    metric,
                    0.5 + f64::from(n) / 100.0,
                    sem,
                    format!("(pipe X (ridge alpha={n}))"),
                    1_784_970_730_000i64 + i64::from(n),
                    ""
                ],
            )
            .unwrap();
        }
    }
    assert_eq!(n, 20);
}

#[test]
fn genebank_rows_become_unbound_records() {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path().join("genebank.db");
    build_genebank(&db);
    let state_dir = tmp.path().join("state");
    let rep = jl(
        &state_dir,
        &[
            "hwm",
            "ingest",
            "--source",
            "genebank",
            db.to_str().unwrap(),
        ],
    )
    .ok()
    .json();
    assert_eq!(
        (rep["candidates"].as_u64(), rep["accepted"].as_u64()),
        (Some(20), Some(20))
    );
    assert_eq!(rep["scan"]["rows"], 20);
    assert_eq!(rep["pointers_written"], 20);
    let state = State::at(&state_dir);
    let pointers = state.pointers().unwrap();
    assert_eq!(pointers.len(), 20);
    for (key, r) in &pointers {
        assert_eq!(r.evaluator.sha256, "unbound");
        assert_eq!(r.evaluator.version, genebank::EVALUATOR_VERSION);
        assert!(!r.replayable && !r.benchmark_valid && !r.sealed);
        assert_eq!(r.improvement_number, 0);
        assert_eq!(r.identity.protocol, "genebank-gen1");
        assert!(r.ts_unix >= 1_784_970_730);
        assert!(
            key.cut_id == genebank::cut_id("cv_robust").unwrap()
                || key.cut_id == genebank::cut_id("test_exact").unwrap()
        );
        // The blob is the canonical row; its hash is the record hash.
        let blob = state
            .read_blob(&r.provenance.record_sha256)
            .unwrap()
            .unwrap();
        let row: serde_json::Value = serde_json::from_slice(&blob).unwrap();
        assert_eq!(row["canonical_genome"], r.genome.dsl);
    }
    assert_eq!(
        pointers
            .iter()
            .filter(|(_, r)| r.genome.semantic_sha256.is_none())
            .count(),
        1
    );
    // A dataset shows its cv_robust and test_exact marks; best first by direction.
    let shown = jl(&state_dir, &["hwm", "show", "0259e0b6af043f58"])
        .ok()
        .json();
    assert_eq!(shown.as_array().unwrap().len(), 2);
    assert!(
        shown[0]["record"]["score"]["value"].as_f64()
            >= shown[1]["record"]["score"]["value"].as_f64()
    );
    // Idempotent: every row is a tie the second time.
    let rep = jl(
        &state_dir,
        &[
            "hwm",
            "ingest",
            "--source",
            "genebank",
            db.to_str().unwrap(),
            "--show-rejects",
        ],
    )
    .ok()
    .json();
    assert_eq!(rep["accepted"], 0);
    assert_eq!(rep["rejected"], 20);
    assert!(rep["rejects"]
        .as_array()
        .unwrap()
        .iter()
        .all(|r| r["reason"] == "tie"));
    // An improved score for one (task, dataset, tier) moves exactly that pointer and chains it.
    {
        let conn = rusqlite::Connection::open(&db).unwrap();
        conn.execute(
            "UPDATE best_scores SET score = 0.99, achieved_at_ms = achieved_at_ms + 5000 WHERE dataset_hash = '0259e0b6af043f58' AND score_tier = 'test_exact'",
            [],
        )
        .unwrap();
    }
    let rep = jl(
        &state_dir,
        &[
            "hwm",
            "ingest",
            "--source",
            "genebank",
            db.to_str().unwrap(),
        ],
    )
    .ok()
    .json();
    assert_eq!(
        (
            rep["accepted"].as_u64(),
            rep["rejected"].as_u64(),
            rep["pointers_written"].as_u64()
        ),
        (Some(1), Some(19), Some(1))
    );
    let cur = exact_current(
        &state,
        "regression",
        "0259e0b6af043f58",
        &genebank::cut_id("test_exact").unwrap(),
    );
    assert_eq!((cur.score.value, cur.improvement_number), (0.99, 1));
    assert!(cur.prev_event_sha256.is_some());
    let v = jl(&state_dir, &["hwm", "verify"]).ok().json();
    assert_eq!(v["breaks"].as_array().unwrap().len(), 0);
    assert_eq!(v["events"], 60);
    // Unbound records are excluded from a freeze unless asked for.
    let out = tmp.path().join("freeze");
    let f = jl(
        &state_dir,
        &["hwm", "freeze", "--out", out.to_str().unwrap()],
    )
    .ok()
    .json();
    assert_eq!(
        (f["included"].as_u64(), f["excluded_unbound"].as_u64()),
        (Some(0), Some(20))
    );
    let f = jl(
        &state_dir,
        &[
            "hwm",
            "freeze",
            "--out",
            out.to_str().unwrap(),
            "--include-unbound",
        ],
    )
    .ok()
    .json();
    assert_eq!(f["included"], 20);
}


mod snapshot;
mod sotarena;
