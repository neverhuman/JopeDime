//! `scores append` through the built binary: what it publishes, what it reports, and what it is
//! allowed to write where.
//!
//! Two independent defects, both of which the command reported as success:
//!
//! * a run whose rows were all duplicates sealed no batch, and the mirror push was gated on that
//!   batch — so `--store` did nothing at all, and said so in a way (`"mirror": null`) that is
//!   indistinguishable from a run with no `--store`. The one command an operator would re-run to
//!   repair a mirror left behind by an earlier local-only append was exactly the command that
//!   skipped the mirror.
//! * the archive path was built by joining the CSV's own `track_id` column onto the state root.
//!   `Path::join` is not containment: an absolute component replaces the whole path and a `..`
//!   component walks out of it, so a published `scores.csv` and its checksum landed wherever the
//!   imported file asked for. Those writes also happened *before* the mirror divergence preflight,
//!   so a run that exits 2 saying "Nothing was written" had already overwritten last week's
//!   archive.
//!
//! Validating the *spelling* of a path is not validating where writing to it goes, so the descent
//! also refuses a symlink anywhere on the archive path — and refuses it before creating anything,
//! which `an_archive_path_through_a_symlink_is_refused_before_anything_is_created` pins.

mod common;

use common::jl;
use jope_learning::canon::sha256_hex;
use std::fs;
use std::path::Path;

const HEADER: &str = "track_id,dataset_id,method_id,protocol,metric,value,status,benchmark_valid,source_kind,source_revision,source_sha256,source_uri,notes\n";

fn score_line(track: &str, ds: &str, method: &str, value: f64) -> String {
    format!(
        "{track},{ds},{method},public-development,rmse,{value},ok,True,reproducible-run-report,2026-06-30T00:40:20+00:00,{},results/x/{ds}/{method}.json,probe\n",
        sha256_hex(format!("{track}{ds}{method}").as_bytes())
    )
}

/// A one-row `<results>/<version>/scores.csv`.
fn results_dir(at: &Path, version: &str, track: &str, value: f64) -> std::path::PathBuf {
    let dir = at.join(version);
    fs::create_dir_all(&dir).unwrap();
    fs::write(
        dir.join("scores.csv"),
        HEADER.to_string() + &score_line(track, "019de5d76a718163", "jope-dime", value),
    )
    .unwrap();
    at.to_path_buf()
}

fn files(root: &Path) -> Vec<String> {
    let mut v: Vec<String> = walkdir::WalkDir::new(root)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
        .map(|e| {
            e.path()
                .strip_prefix(root)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/")
        })
        .collect();
    v.sort();
    v
}

fn append(state: &Path, store: Option<&str>, results: &Path) -> common::Run {
    let mut args: Vec<&str> = Vec::new();
    if let Some(url) = store {
        args.extend(["--store", url]);
    }
    args.extend([
        "scores",
        "append",
        "--source",
        "sotarena",
        results.to_str().unwrap(),
        "--git-sha",
        "e5db30a07c093661c8f6940d3e22192511f8388b",
    ]);
    jl(state, &args)
}

// --------------------------------------------------------------------------------------
// The mirror push must be driven by `--store`, not by whether this run sealed a batch.
// --------------------------------------------------------------------------------------

/// Fails on the pre-fix code at the `mirror` assertion: step 2 exits 0 reporting `"rows": 0`,
/// `"mirror": null` and leaves the mirror directory empty, while the local ledger holds the batch
/// that the mirror is missing.
#[test]
fn a_duplicate_only_append_with_store_still_publishes_the_backlog() {
    let tmp = tempfile::tempdir().unwrap();
    let mirror = tmp.path().join("mirror");
    fs::create_dir_all(&mirror).unwrap();
    let url = format!("file://{}", mirror.display());
    let state = tmp.path().join("state");
    let results = results_dir(
        &tmp.path().join("results"),
        "tabular-regression-v1",
        "tabular-regression-v1",
        0.0611,
    );

    // 1. a local-only append: the local ledger has a batch the mirror has never seen.
    append(&state, None, &results).ok();
    assert_eq!(files(&mirror), Vec::<String>::new());

    // 2. the same import, now with --store. Every row is a duplicate, so this run seals nothing —
    //    but the mirror is a batch behind and `--store` was asked for, so the backlog goes up.
    let r = append(&state, Some(&url), &results).ok();
    let rep = &r.json()[0];
    assert_eq!(rep["rows"], 0);
    assert_eq!(rep["duplicates_skipped"], 1);
    assert!(rep["batch"].is_null(), "this run sealed a batch: {rep}");
    assert!(
        !rep["mirror"].is_null(),
        "a run given --store reported no mirror at all: {rep}"
    );
    assert_eq!(
        (
            rep["mirror"]["batches"].as_u64(),
            rep["mirror"]["created"].as_u64()
        ),
        (Some(1), Some(1)),
        "the backlog was not published: {rep}"
    );
    let published = files(&mirror);
    assert!(
        published.contains(&"scores/ledger/HEAD.json".to_string())
            && published.iter().any(|p| p.ends_with(".jsonl")),
        "the mirror is still empty: {published:?}"
    );
    assert_eq!(
        fs::read(mirror.join("scores/ledger/HEAD.json")).unwrap(),
        fs::read(state.join("scores/ledger/HEAD.json")).unwrap(),
        "the mirror's tip is not this host's tip"
    );

    // 3. and a run that genuinely has nothing to do says so in its own counts, which cannot be
    //    read as "no mirror" and cannot be mistaken for a publish.
    let rep = append(&state, Some(&url), &results).ok().json()[0].clone();
    assert_eq!(rep["rows"], 0);
    assert!(!rep["mirror"].is_null(), "{rep}");
    assert_eq!(rep["mirror"]["batches"], 0, "{rep}");
    assert_eq!(rep["mirror"]["created"], 0, "{rep}");
    assert_eq!(rep["mirror"]["pointers"], 0, "{rep}");
}

/// Without `--store` the report still says `"mirror": null`, so the two cases stay distinguishable
/// in the opposite direction too.
#[test]
fn an_append_without_store_reports_no_mirror() {
    let tmp = tempfile::tempdir().unwrap();
    let state = tmp.path().join("state");
    let results = results_dir(
        &tmp.path().join("results"),
        "v1",
        "tabular-regression-v1",
        0.5,
    );
    let rep = append(&state, None, &results).ok().json()[0].clone();
    assert_eq!(rep["rows"], 1);
    assert!(rep["mirror"].is_null(), "{rep}");
}

// --------------------------------------------------------------------------------------
// The archive path is data from the imported file, and must stay inside `--state`.
// --------------------------------------------------------------------------------------

/// Fails on the pre-fix code at the `outside` assertion: the run exits 0 and writes
/// `<outside>/v1/scores.csv` and `<outside>/v1/scores.csv.sha256` outside the state root.
#[test]
fn a_track_id_cannot_write_outside_the_state_root() {
    let tmp = tempfile::tempdir().unwrap();
    let outside = tmp.path().join("outside-state");
    fs::create_dir_all(&outside).unwrap();
    let state = tmp.path().join("state");

    for (label, track) in [
        ("absolute", outside.to_string_lossy().into_owned()),
        ("parent-relative", "../../outside-state".to_string()),
        ("bare-parent", "..".to_string()),
        ("nested", "a/b".to_string()),
        ("dot", ".".to_string()),
    ] {
        let results = results_dir(
            &tmp.path().join(format!("results-{label}")),
            "v1",
            &track,
            0.5,
        );
        let r = append(&state, None, &results);
        assert_ne!(
            r.status,
            0,
            "{label}: an unsafe track_id was accepted; stdout: {}",
            String::from_utf8_lossy(&r.stdout)
        );
        assert!(
            r.stderr.contains("unsafe track_id"),
            "{label}: expected the containment refusal, got: {}",
            r.stderr
        );
        assert_eq!(
            files(&outside),
            Vec::<String>::new(),
            "{label}: the import wrote outside the state root"
        );
        assert_eq!(
            files(&state),
            Vec::<String>::new(),
            "{label}: the refused import wrote inside the state root too"
        );
    }

    // The controls: an ordinary track_id still archives where it always did, and an *empty*
    // `track_id` column is not an escape — it yields no `source.track` at all, so the archive
    // falls back to the version directory exactly as before.
    let results = results_dir(
        &tmp.path().join("results-ok"),
        "v1",
        "tabular-regression-v1",
        0.5,
    );
    append(&state, None, &results).ok();
    assert!(state
        .join("scores/sotarena/tabular-regression-v1/v1/scores.csv")
        .exists());
    let results = results_dir(&tmp.path().join("results-blank"), "v2", "", 0.4);
    append(&state, None, &results).ok();
    assert!(state.join("scores/sotarena/v2/v2/scores.csv").exists());
}

/// A run the mirror refuses must not have overwritten the archived source first.
///
/// Fails on the pre-fix code at the last assertion: `run_scores` wrote `scores.csv` and its
/// checksum before calling `scores::append`, where the divergence preflight lives, so the command
/// exits 2 saying "Nothing was written" over an archive it has already replaced.
#[test]
fn a_refused_push_does_not_overwrite_the_archived_source() {
    let tmp = tempfile::tempdir().unwrap();
    let mirror = tmp.path().join("mirror");
    fs::create_dir_all(&mirror).unwrap();
    let url = format!("file://{}", mirror.display());
    let track = "tabular-regression-v1";

    // Another host owns the mirror's chain.
    let other = results_dir(&tmp.path().join("results-other"), track, track, 0.0599);
    append(&tmp.path().join("state-other"), Some(&url), &other).ok();

    // This host imports locally; its chain cannot continue the mirror's.
    let state = tmp.path().join("state");
    let results = results_dir(&tmp.path().join("results"), track, track, 0.0611);
    append(&state, None, &results).ok();
    let archived = state.join(format!("scores/sotarena/{track}/{track}/scores.csv"));
    let before = fs::read_to_string(&archived).unwrap();

    // The source changes, and this host pushes to the diverged mirror.
    fs::write(
        results.join(track).join("scores.csv"),
        HEADER.to_string() + &score_line(track, "019de5d76a718163", "jope-dime", 0.0421),
    )
    .unwrap();
    let r = append(&state, Some(&url), &results);
    assert_ne!(r.status, 0, "the diverged mirror was not refused");
    assert!(
        r.stderr.contains("holds a chain this host cannot continue")
            && r.stderr.contains("Nothing was written"),
        "expected the divergence refusal, got: {}",
        r.stderr
    );
    assert_eq!(
        fs::read_to_string(&archived).unwrap(),
        before,
        "the archived source was overwritten by a run that says nothing was written"
    );
    assert_eq!(
        fs::read_to_string(archived.with_extension("csv.sha256")).unwrap(),
        format!("{}  scores.csv\n", sha256_hex(before.as_bytes())),
        "the archived checksum was overwritten by a run that says nothing was written"
    );
}

/// The refusal is not selective about *which* run it protects: a second, independent version in
/// the same import must not be half-archived because a later one is unsafe.
#[test]
fn one_unsafe_version_refuses_the_whole_import() {
    let tmp = tempfile::tempdir().unwrap();
    let state = tmp.path().join("state");
    let results = tmp.path().join("results");
    results_dir(&results, "v1", "tabular-regression-v1", 0.5);
    results_dir(&results, "v2", "../escape", 0.4);
    let r = append(&state, None, &results);
    assert_ne!(r.status, 0);
    assert_eq!(
        files(&state),
        Vec::<String>::new(),
        "the good version was archived before the bad one was refused"
    );
}

// --------------------------------------------------------------------------------------
// A name check cannot see a symlink. The descent must refuse one *before* it creates anything.
// --------------------------------------------------------------------------------------

/// Fails on the pre-fix code at the second assertion: `archive_dir` returned the containment
/// refusal — "Nothing was written" — having already run `create_dir_all` straight through the
/// symlink and created `<outside-state>/version` on the far side of it.
#[cfg(unix)]
#[test]
fn an_archive_path_through_a_symlink_is_refused_before_anything_is_created() {
    use jope_learning::state::State;

    let tmp = tempfile::tempdir().unwrap();
    let outside = tmp.path().join("outside-state");
    fs::create_dir_all(&outside).unwrap();
    let state = State::at(&tmp.path().join("state"));
    let parent = state.root().join("scores/sotarena");
    fs::create_dir_all(&parent).unwrap();
    std::os::unix::fs::symlink(&outside, parent.join("track")).unwrap();

    let err = state
        .archive_dir("sotarena", "track", "version")
        .expect_err("an archive path through a symlink was accepted");
    // The defect first: the pre-fix code *did* return this error — after `create_dir_all` had
    // already run straight through the symlink.
    assert!(
        !outside.join("version").exists(),
        "a refusal saying nothing was written created {} on the far side of the symlink",
        outside.join("version").display()
    );
    assert_eq!(
        files(&outside),
        Vec::<String>::new(),
        "the refused call wrote outside the state root"
    );
    let msg = format!("{err:#}");
    assert!(
        msg.contains("symlink") && msg.contains("Nothing was written"),
        "expected the symlink refusal, got: {msg}"
    );

    // A dangling symlink is refused the same way, and still creates nothing.
    fs::remove_file(parent.join("track")).unwrap();
    std::os::unix::fs::symlink(outside.join("nowhere"), parent.join("track")).unwrap();
    let err = state
        .archive_dir("sotarena", "track", "version")
        .expect_err("a dangling symlink on the archive path was accepted");
    assert!(format!("{err:#}").contains("symlink"), "{err:#}");
    assert!(!outside.join("nowhere").exists());

    // A plain file where a directory must go is refused too, rather than half-created.
    fs::remove_file(parent.join("track")).unwrap();
    fs::write(parent.join("track"), b"not a directory\n").unwrap();
    let err = state
        .archive_dir("sotarena", "track", "version")
        .expect_err("a file on the archive path was accepted");
    assert!(format!("{err:#}").contains("not a directory"), "{err:#}");

    // The control: with the obstruction gone the same call succeeds and lands inside the root.
    fs::remove_file(parent.join("track")).unwrap();
    let dir = state.archive_dir("sotarena", "track", "version").unwrap();
    assert_eq!(dir, state.root().join("scores/sotarena/track/version"));
    assert!(dir.is_dir());
    assert!(fs::canonicalize(&dir)
        .unwrap()
        .starts_with(fs::canonicalize(state.root()).unwrap()));
}
