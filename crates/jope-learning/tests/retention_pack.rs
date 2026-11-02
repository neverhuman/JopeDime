//! S3-RET job (b): write-time tags, `jope.s3-pack/v1` packs and the `gp_distilled/v1` table.
//! Fixtures only; nothing here talks to S3.

use jope_learning::canon::sha256_hex;
use jope_learning::retention::distilled::{self, DistilledRow, Outcome, SplitRole};
use jope_learning::retention::pack::{read_entry, verify, PackWriter, SealedPack, MIN_PACK_BYTES};
use jope_learning::retention::parquet::{self, Cell, ColumnSpec, ColumnType, Table};
use jope_learning::retention::tags::{ObjectKind, Supersession, Tier, WriteTags};
use std::collections::BTreeSet;

fn tags() -> WriteTags {
    WriteTags::new(
        Tier::T2,
        "run-20261013-a",
        "gp-teacher-v6",
        false,
        None,
        ObjectKind::Pack,
    )
    .unwrap()
}

fn record(i: usize) -> Vec<u8> {
    // ~10 KB JSON-ish bodies with repetition, like GP attempt receipts.
    let line = format!("{{\"attempt\":{i},\"program\":\"(pipe X (standardize) (ridge alpha={}))\",\"score\":0.{i:04}}}\n", i % 17);
    line.repeat(10_000 / line.len() + 1).into_bytes()
}

fn write_packs(n: usize, target: u64) -> (Vec<SealedPack>, Vec<Vec<u8>>) {
    let mut w = PackWriter::fixture("gp-teacher-v6", "run-20261013-a", tags(), target).unwrap();
    let mut packs = Vec::new();
    let mut bodies = Vec::new();
    for i in 0..n {
        let body = record(i);
        let key = format!(
            "experiments/gp-teacher-v6/attempts/ctx{:02}/{i:05}.json",
            i % 7
        );
        if let Some(p) = w.add(&key, Some(&format!("v{i}")), &body).unwrap() {
            packs.push(p);
        }
        bodies.push(body);
    }
    if let Some(p) = w.finish() {
        packs.push(p);
    }
    (packs, bodies)
}

fn known() -> BTreeSet<String> {
    [
        "cv_geomean_rmse_ratio",
        "best_rust_rmse",
        "validation_rmse",
        "l6",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

fn row(prog: char, ctx: char, score: Option<f64>, outcome: Outcome) -> DistilledRow {
    DistilledRow {
        program_sha: prog.to_string().repeat(64),
        program: Some("(pipe X (standardize) (ridge alpha=1))".into()),
        dataset: "000c7ae796cb2672".into(),
        context_sha256: Some(ctx.to_string().repeat(64)),
        fold: Some(0),
        split_role: SplitRole::Train,
        outcome,
        score,
        score_name: score.map(|_| "cv_geomean_rmse_ratio".into()),
        rank_ctx: score.map(|_| 1),
        champion_id: None,
        superseded_by: None,
        seed_packet: None,
        policylm_ckpt: None,
        campaign: Some("regression-research-2k-per-fold-hwm-v1-20261001".into()),
        run_id: "run-20261013-a".into(),
        family: "gp-teacher-v6".into(),
        qualified: false,
        l6_delta: None,
        source_bucket: "veox-jopedime-learning-use2".into(),
        source_key: format!("experiments/gp-teacher-v6/attempts/{prog}{ctx}.json"),
        source_version_id: Some("v1".into()),
        source_sha256: "e".repeat(64),
        pack_key: Some("packs/v1/gp-teacher-v6/run-20261013-a/00000-x.zst".into()),
        pack_entry: Some(0),
    }
}

#[path = "retention_pack/distilled.rs"]
mod distilled;
#[path = "retention_pack/packs.rs"]
mod packs;
#[path = "retention_pack/parquet.rs"]
mod parquet;
#[path = "retention_pack/selfplay_bridge.rs"]
mod selfplay_bridge;
#[path = "retention_pack/statistics.rs"]
mod statistics;
#[path = "retention_pack/tags.rs"]
mod tags;
