//! Plan WP12 / Verification 12: the Elo recomputed from the committed results equals the
//! published SOTArena Elo for `jope-dime` (2259.8, rank 1, 11 strict wins on
//! `tabular-regression-v2`).
//!
//! Inputs are the committed copy of the benchmark clone under
//! `assets/regression/sotarena-tabular-regression-v2/` (`scores.csv.zst` is the 3.4 MB
//! `results/tabular-regression-v2/scores.csv` at clone commit e5db30a0, SHA-256
//! `5ec86898…`; see `assets/regression/SOURCES.md`). When `SOTARENA_ROOT` points at a benchmark
//! clone that carries the track, the same ladder is recomputed from it as well; otherwise that
//! extra check says so on stderr — the fixture check always runs.

use std::path::{Path, PathBuf};

use jope_arena::io::{read_bytes, read_text, sha256_hex};
use jope_arena::leaderboard::{self, Snapshot};

const TRACK: &str = "tabular-regression-v2";
const SCORES_SHA256: &str = "5ec868987c2f4286ae44728fde49215ec6e55e7d27e62100d3823744e2957db6";
const LEADERBOARD_SHA256: &str = "2d28ffbd8b5830c6adc39874dfcf98498419ef8c1df6ac5eb135c660013fb24f";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/jope-arena sits two levels below the repository root")
        .to_path_buf()
}

fn fixture_root() -> PathBuf {
    repo_root().join("assets/regression/sotarena-tabular-regression-v2")
}

fn snapshot_path() -> PathBuf {
    repo_root().join("assets/regression/elo-tabular-regression-v2.json")
}

fn assert_published_ladder(root: &Path, label: &str) {
    let (board, location) = leaderboard::compute(root, TRACK, None)
        .unwrap_or_else(|error| panic!("{label}: recompute failed: {error:#}"));
    let scores = read_bytes(&location.dir.join("scores.csv")).unwrap();
    assert_eq!(
        sha256_hex(&scores),
        SCORES_SHA256,
        "{label}: scores.csv is not the published input"
    );
    assert_eq!(board.rankable_rows, 8763, "{label}");
    assert_eq!(board.datasets_scored, 45, "{label}");
    assert_eq!(board.rows.len(), 305, "{label}");

    let top = &board.rows[0];
    assert_eq!(top.method_id, "jope-dime", "{label}");
    assert_eq!(top.elo_rank, 1, "{label}");
    assert_eq!(top.full_coverage_rank, "1", "{label}");
    assert_eq!(top.elo, "2259.8", "{label}: published Elo");
    assert_eq!(top.dataset_wins, 11, "{label}: strict wins");
    assert_eq!(top.datasets_scored, 45, "{label}");
    assert_eq!(top.average_rank, "4.622", "{label}");
    assert_eq!(top.protocols, "aspfm-generated", "{label}");
    assert!(
        (board.ratings["jope-dime"] - 2259.8).abs() < 0.05,
        "{label}: {}",
        board.ratings["jope-dime"]
    );

    let csv = leaderboard::to_csv(&board.rows).unwrap();
    assert_eq!(
        sha256_hex(csv.as_bytes()),
        LEADERBOARD_SHA256,
        "{label}: leaderboard.csv bytes"
    );
    let committed = read_text(&location.dir.join("leaderboard.csv")).unwrap();
    assert_eq!(
        csv, committed,
        "{label}: byte-identical to the committed leaderboard.csv"
    );

    let snapshot: Snapshot =
        serde_json::from_slice(&std::fs::read(snapshot_path()).unwrap()).unwrap();
    assert_eq!(snapshot.schema, "jope.arena-elo-snapshot/v1");
    assert_eq!(snapshot.scores_sha256, SCORES_SHA256);
    assert_eq!(snapshot.leaderboard_sha256, LEADERBOARD_SHA256);
    assert_eq!(
        snapshot.rows, board.rows,
        "{label}: rows equal the committed snapshot"
    );
    assert_eq!(
        Snapshot::new(&board, SCORES_SHA256).unwrap(),
        snapshot,
        "{label}: snapshot regenerates identically"
    );
}

#[test]
fn committed_fixture_reproduces_the_published_ladder() {
    assert_published_ladder(&fixture_root(), "assets/regression fixture");
}

#[test]
fn live_sotarena_root_reproduces_the_published_ladder_when_present() {
    let Some(root) = std::env::var_os("SOTARENA_ROOT").filter(|v| !v.is_empty()) else {
        eprintln!(
            "SKIPPED (visibly): SOTARENA_ROOT is unset; only the committed fixture was checked"
        );
        return;
    };
    let root = PathBuf::from(root);
    let track = root.join("tracks").join(TRACK).join("track.json");
    assert!(
        track.is_file(),
        "SOTARENA_ROOT={} does not carry {TRACK} ({} missing)",
        root.display(),
        track.display()
    );
    assert_published_ladder(&root, "SOTARENA_ROOT");
}

#[test]
fn fixture_files_are_the_published_bytes() {
    let results = fixture_root().join("results").join(TRACK);
    let ledger = read_bytes(&results.join("resolution-ledger.jsonl")).unwrap();
    assert_eq!(
        sha256_hex(&ledger),
        "d28833254eff6787b45c2280f46fc4db0d3dd14a944677c06565a903288d5cd3"
    );
    let acceptance = std::fs::read(results.join("aspfm-acceptance-manifest.json")).unwrap();
    assert_eq!(
        sha256_hex(&acceptance),
        "963b6c55988959837a351d86a2d135dcccecda6df1fef9cca85d28e068d4a190"
    );
    let registry =
        std::fs::read(fixture_root().join("registry/datasets.tabular-regression-v2.json")).unwrap();
    assert_eq!(
        sha256_hex(&registry),
        "5ec3d62df5900c3e91d64fa940775e4761ae8f68a346afc379723c249a3b0804"
    );
    let methods = std::fs::read(fixture_root().join("registry/methods.json")).unwrap();
    assert_eq!(
        sha256_hex(&methods),
        "69c88d86d9ab5c0923a4a272e4e07f68ae84f68a5e5b7addf28811594cbf6930"
    );
    let track =
        std::fs::read(fixture_root().join("tracks").join(TRACK).join("track.json")).unwrap();
    assert_eq!(
        sha256_hex(&track),
        "c6540865dbbd29e3b9163a3a6fa9aa941939683b5c0485b73826ce598ff1eb22"
    );
    // The committed compressed copy stays under the 500 KiB fixture cap.
    let zst = std::fs::metadata(results.join("scores.csv.zst"))
        .unwrap()
        .len();
    assert!(zst <= 500 * 1024, "scores.csv.zst is {zst} bytes");
}
