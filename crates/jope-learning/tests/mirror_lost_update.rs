//! Two writers, one shared mirror: the second must not silently orphan the first's batch.
//!
//! `hwm ingest --store <url>` and `scores append --store <url>` push a ledger batch under a
//! create-only PUT and then move `<ledger>/HEAD.json`. The local `$JOPE_STATE` flock serialises
//! writers on one host and nothing else; two hosts with separate state trees each append a
//! distinctly-ULID-named batch, so the create-only PUT never collides and the *pointer* is the only
//! thing that can catch the divergence. It only catches it if the expected-previous value is
//! checked against the writer's own chain — a bare read of the key that is about to be written
//! makes the precondition self-satisfying and the write last-writer-wins, and the earlier host's
//! batch stops being reachable from `HEAD` (`verify`'s chain walk starts there).
//!
//! `a_host_that_does_not_hold_the_mirrors_tip_cannot_move_it` fails on the pre-fix code: the
//! second ingest exits 0 and the mirror `HEAD.json` names host B's batch. It was named
//! `head_write_is_conditional_on_the_writers_own_pre_append_head` when that pre-append `HEAD.json`
//! was the witness; `sync::plan` now supplies a witness it has proved is an ancestor of the local
//! chain (see `mirror_reconcile.rs` for why the old witness wedged the mirror), so the scenario is
//! unchanged and the rule it enforces is stated as it now stands.

mod common;

use common::jl;
use jope_learning::canon::sha256_hex;
use std::fs;
use std::path::{Path, PathBuf};

/// Copy `champions[range]` into a directory of its own, so the two hosts ingest disjoint cuts and
/// their per-cut `current.json` pointers never collide — leaving `HEAD.json` as the only guard.
fn champion_subset(dest: &Path, skip: usize, take: usize) -> PathBuf {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/champions");
    fs::create_dir_all(dest).unwrap();
    let mut names: Vec<_> = fs::read_dir(&src)
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();
    names.sort();
    let chosen: Vec<_> = names.into_iter().skip(skip).take(take).collect();
    assert_eq!(chosen.len(), take, "fixture corpus too small");
    for name in chosen {
        fs::copy(src.join(&name), dest.join(&name)).unwrap();
    }
    dest.to_path_buf()
}

fn ingest(state: &Path, url: &str, dir: &Path, bundle: &str) -> common::Run {
    jl(
        state,
        &[
            "--store",
            url,
            "hwm",
            "ingest",
            "--source",
            "champions",
            dir.to_str().unwrap(),
            "--bundle-id",
            bundle,
        ],
    )
}

#[test]
fn a_host_that_does_not_hold_the_mirrors_tip_cannot_move_it() {
    let tmp = tempfile::tempdir().unwrap();
    let mirror = tmp.path().join("mirror");
    fs::create_dir_all(&mirror).unwrap();
    let url = format!("file://{}", mirror.display());
    let head_obj = mirror.join("hwm/ledger/HEAD.json");

    // Host A: separate $JOPE_STATE, its own half of the corpus.
    let state_a = tmp.path().join("state-a");
    let dir_a = champion_subset(&tmp.path().join("champs-a"), 0, 5);
    ingest(&state_a, &url, &dir_a, "host-a").ok();
    let head_after_a = fs::read(&head_obj).unwrap();
    let head_a: serde_json::Value = serde_json::from_slice(&head_after_a).unwrap();
    let batch_a = head_a["path"].as_str().unwrap().to_string();
    assert!(mirror.join("hwm/ledger").join(&batch_a).exists());

    // Host B: a different $JOPE_STATE tree that has never seen host A's batch. Its ledger is
    // empty, so its pre-append HEAD is absent while the mirror already holds host A's.
    let state_b = tmp.path().join("state-b");
    let dir_b = champion_subset(&tmp.path().join("champs-b"), 5, 5);
    let r = ingest(&state_b, &url, &dir_b, "host-b");
    assert_ne!(
        r.status,
        0,
        "host B's HEAD.json write must not succeed against a mirror it has not seen; \
         stdout: {}",
        String::from_utf8_lossy(&r.stdout)
    );
    assert!(
        r.stderr.contains("holds a chain this host cannot continue"),
        "expected the conditional-pointer failure, got: {}",
        r.stderr
    );
    // Refused before host B sealed anything locally, so a retry finds no wider gap.
    assert!(jope_learning::ledger::Ledger::open(&state_b, "hwm")
        .batches()
        .unwrap()
        .is_empty());

    // The first host's batch is still the tip, and still present: nothing was orphaned.
    assert_eq!(
        fs::read(&head_obj).unwrap(),
        head_after_a,
        "mirror HEAD.json was clobbered by the second writer"
    );
    assert!(mirror.join("hwm/ledger").join(&batch_a).exists());
    let tip: serde_json::Value = serde_json::from_slice(&fs::read(&head_obj).unwrap()).unwrap();
    assert_eq!(tip["path"].as_str().unwrap(), batch_a);
    assert_eq!(
        sha256_hex(&fs::read(mirror.join("hwm/ledger").join(&batch_a)).unwrap()),
        tip["sha256"].as_str().unwrap()
    );
}

#[test]
fn the_same_host_still_advances_the_mirror_head_across_batches() {
    // The guard must reject a diverged mirror, not every second push.
    let tmp = tempfile::tempdir().unwrap();
    let mirror = tmp.path().join("mirror");
    fs::create_dir_all(&mirror).unwrap();
    let url = format!("file://{}", mirror.display());
    let state = tmp.path().join("state");
    let dir_a = champion_subset(&tmp.path().join("champs-a"), 0, 5);
    let dir_b = champion_subset(&tmp.path().join("champs-b"), 5, 5);

    ingest(&state, &url, &dir_a, "first").ok();
    let first = fs::read(mirror.join("hwm/ledger/HEAD.json")).unwrap();
    ingest(&state, &url, &dir_b, "second").ok();
    let second = fs::read(mirror.join("hwm/ledger/HEAD.json")).unwrap();
    assert_ne!(first, second, "the tip must move");

    let ledger = jope_learning::ledger::Ledger::open(&state, "hwm");
    assert_eq!(second, fs::read(ledger.root().join("HEAD.json")).unwrap());
    let head: serde_json::Value = serde_json::from_slice(&second).unwrap();
    assert_eq!(head["batches"], 2);
    let prev: serde_json::Value = serde_json::from_slice(&first).unwrap();
    assert!(mirror
        .join("hwm/ledger")
        .join(prev["path"].as_str().unwrap())
        .exists());
}

const HEADER: &str = "track_id,dataset_id,method_id,protocol,metric,value,status,benchmark_valid,source_kind,source_revision,source_sha256,source_uri,notes\n";

fn score_line(track: &str, ds: &str, method: &str, value: f64) -> String {
    format!(
        "{track},{ds},{method},public-development,rmse,{value},ok,True,reproducible-run-report,2026-06-30T00:40:20+00:00,{},results/external-references/golden42/{ds}/{method}.json,Exact Golden42 report; method identity is the report filename.\n",
        sha256_hex(format!("{track}{ds}{method}").as_bytes())
    )
}

fn sotarena_results(dir: &Path, track: &str, method: &str) -> PathBuf {
    let body = HEADER.to_string() + &score_line(track, "019de5d76a718163", method, 0.0733);
    fs::create_dir_all(dir.join(track)).unwrap();
    fs::write(dir.join(track).join("scores.csv"), body).unwrap();
    dir.to_path_buf()
}

#[test]
fn the_score_ledger_tip_is_guarded_the_same_way() {
    let tmp = tempfile::tempdir().unwrap();
    let mirror = tmp.path().join("mirror");
    fs::create_dir_all(&mirror).unwrap();
    let url = format!("file://{}", mirror.display());
    let head_obj = mirror.join("scores/ledger/HEAD.json");

    let append = |state: &Path, results: &Path| {
        jl(
            state,
            &[
                "--store",
                &url,
                "scores",
                "append",
                "--source",
                "sotarena",
                results.to_str().unwrap(),
                "--git-sha",
                "e5db30a07c093661c8f6940d3e22192511f8388b",
            ],
        )
    };

    let res_a = sotarena_results(
        &tmp.path().join("res-a"),
        "tabular-regression-v1",
        "ada_boost",
    );
    append(&tmp.path().join("state-a"), &res_a).ok();
    let head_after_a = fs::read(&head_obj).unwrap();

    let res_b = sotarena_results(&tmp.path().join("res-b"), "tabular-regression-v2", "ridge");
    let r = append(&tmp.path().join("state-b"), &res_b);
    assert_ne!(
        r.status,
        0,
        "stdout: {}",
        String::from_utf8_lossy(&r.stdout)
    );
    assert!(
        r.stderr.contains("holds a chain this host cannot continue"),
        "expected the conditional-pointer failure, got: {}",
        r.stderr
    );
    assert_eq!(fs::read(&head_obj).unwrap(), head_after_a);
    assert!(
        jope_learning::ledger::Ledger::open(&tmp.path().join("state-b"), "scores")
            .batches()
            .unwrap()
            .is_empty()
    );
}
