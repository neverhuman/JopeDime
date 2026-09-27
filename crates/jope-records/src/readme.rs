//! `readme-benchmarks --check|--write`: render the README's measured-results blocks from
//! committed evidence.
//!
//! Rust port of `deploy/ci/readme-benchmarks.py` (archived at `8a55c4fef`; see
//! `docs/archive/python-archive-manifest.json`) producing the same bytes. Three independent marker
//! pairs, so regenerating one can never clobber another:
//!
//! ```text
//! <!-- arena:begin --> ... <!-- arena:end -->       the Golden45 ladders
//! <!-- samecut:begin --> ... <!-- samecut:end -->   the same-cut ladder this tree reproduces
//! <!-- bench:begin --> ... <!-- bench:end -->       emit-path latency
//! ```
//!
//! Nothing here is hand-typed. Every number is read from a file that is already under test:
//! `crates/jope-arena/tests/elo_regression.rs` asserts that both ladders recompute from
//! `scores.csv`, so this renderer does not re-derive them — it formats what those tests already
//! guarantee, records where every number came from, and refuses to render when the committed
//! evidence disagrees with itself. **benchmark-valid** is the number this repository claims;
//! **published** is what the benchmark ranked, rendered as the best known program per dataset
//! and labelled as not a held-out result.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use regex::Regex;

use crate::py;
use crate::pycsv::{self, Newlines};
use crate::pyjson::{Dict, Value};
use crate::{Error, Result};

const REGRESSION: &str = "assets/regression";
const ARENA: &str = "assets/regression/sotarena-tabular-regression-v2";
const TRACK: &str =
    "assets/regression/sotarena-tabular-regression-v2/tracks/tabular-regression-v2/track.json";
const LEADERBOARD: &str =
    "assets/regression/sotarena-tabular-regression-v2/results/tabular-regression-v2/leaderboard.csv";
const SNAPSHOT: &str = "assets/regression/elo-tabular-regression-v2.json";
const VALID_LEADERBOARD: &str =
    "assets/regression/leaderboard-tabular-regression-v2.benchmark-valid.csv";
const VALID_SNAPSHOT: &str = "assets/regression/elo-tabular-regression-v2.benchmark-valid.json";
const CORRECTIONS: &str = "assets/regression/protocol-corrections-tabular-regression-v2.json";
const BENCH: &str = "assets/regression/bench-baseline.json";
const README: &str = "README.md";
const SAMECUT: &str = "assets/same-cut/v2";
const SAMECUT_ADAPTERS: [&str; 4] = ["catboost", "lightgbm", "rf", "xgboost"];
const SAMECUT_TRACK_ID: &str = "jope-same-cut-v1";
const SAMECUT_DOC: &str = "docs/same-cut-bench.md";
const REPORT_SCHEMA: &str = "jope.bench-report/v2";
const SEARCHED: &str = "afm.lite-race.v1.resampled";
const LLM_REFERENCE_HOURS: i64 = 1;
const LLM_REFERENCE_TOKENS: i64 = 10_000_000;
const SNAPSHOT_SCHEMA: &str = "jope.arena-elo-snapshot/v2";
const CORRECTIONS_SCHEMA: &str = "jope.arena-protocol-corrections/v1";
const OURS: [&str; 2] = ["jope-dime", "jope-dime-regressor"];
const ARENA_BEGIN: &str = "<!-- arena:begin -->";
const ARENA_END: &str = "<!-- arena:end -->";
const SAMECUT_BEGIN: &str = "<!-- samecut:begin -->";
const SAMECUT_END: &str = "<!-- samecut:end -->";
const BENCH_BEGIN: &str = "<!-- bench:begin -->";
const BENCH_END: &str = "<!-- bench:end -->";
const ANCHOR: &str = "## Install and check";
const TOP_N: usize = 5;

fn samecut_path(name: &str) -> String {
    format!("{SAMECUT}/{name}")
}

struct Paths {
    root: PathBuf,
}

impl Paths {
    fn at(&self, rel: &str) -> PathBuf {
        self.root.join(rel)
    }
    fn sha256(&self, rel: &str) -> Result<String> {
        Ok(py::sha256_hex(&py::read_bytes(&self.at(rel))?))
    }
    fn json(&self, rel: &str) -> Result<Value> {
        py::load_json(&self.at(rel))
    }
}

/// `refuse(why)`: the committed evidence disagrees with itself.
fn refuse<T>(why: impl Into<String>) -> Result<T> {
    Err(Error::Invalid(format!(
        "{}\nThe committed benchmark evidence disagrees with itself; refusing to publish.",
        why.into()
    )))
}

fn shape(what: &str) -> Error {
    Error::Shape(format!("{what} missing"))
}

/// `value[key]` (a missing key is Python's `KeyError`).
fn at<'a>(value: &'a Value, key: &str) -> Result<&'a Value> {
    value.get(key).ok_or_else(|| shape(&pyrepr_key(key)))
}

fn pyrepr_key(key: &str) -> String {
    crate::pyjson::str_repr(key)
}

fn get<'a>(value: &'a Value, key: &str) -> &'a Value {
    static NULL: Value = Value::Null;
    value.get(key).unwrap_or(&NULL)
}

/// `str(value)` as an f-string prints it.
fn s(value: &Value) -> String {
    value.py_str()
}

fn cell<'a>(row: &'a Dict, key: &str) -> Result<&'a str> {
    row.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| shape(&pyrepr_key(key)))
}

fn int_cell(row: &Dict, key: &str) -> Result<i64> {
    py::int(cell(row, key)?).ok_or_else(|| Error::Shape(format!("{key} is not an integer")))
}

fn float_cell(row: &Dict, key: &str) -> Result<f64> {
    py::float(cell(row, key)?).ok_or_else(|| Error::Shape(format!("{key} is not a float")))
}

fn float_of(value: &Value) -> Result<f64> {
    match value {
        Value::Int(i) => Ok(*i as f64),
        Value::Float(f) => Ok(*f),
        Value::Bool(b) => Ok(f64::from(u8::from(*b))),
        other => Err(Error::Shape(format!("{} is not a number", other.repr()))),
    }
}

/// `format(x, ".Nf")`.
fn fixed(x: f64, precision: usize) -> String {
    format!("{x:.precision$}")
}

/// `format(n, ",")` for an integer.
fn thousands(n: i64) -> String {
    let digits = n.unsigned_abs().to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    if n < 0 {
        format!("-{out}")
    } else {
        out
    }
}

/// `format(x, ".4g")` (Python's general format, precision 4).
fn general4(x: f64) -> String {
    const P: i32 = 4;
    if x == 0.0 {
        return if x.is_sign_negative() { "-0" } else { "0" }.to_owned();
    }
    if !x.is_finite() {
        return py_nonfinite(x);
    }
    // Round to P significant digits in scientific form, then choose the presentation.
    let scientific = format!("{:.*e}", (P - 1) as usize, x);
    let (mantissa, exponent) = scientific.split_once('e').expect("scientific form");
    let exponent: i32 = exponent.parse().expect("exponent");
    if (-4..P).contains(&exponent) {
        let decimals = (P - 1 - exponent).max(0) as usize;
        strip_zeros(&format!("{x:.decimals$}"))
    } else {
        let mantissa = strip_zeros(mantissa);
        let sign = if exponent < 0 { '-' } else { '+' };
        format!("{mantissa}e{sign}{:02}", exponent.abs())
    }
}

fn py_nonfinite(x: f64) -> String {
    if x.is_nan() {
        "nan".to_owned()
    } else if x > 0.0 {
        "inf".to_owned()
    } else {
        "-inf".to_owned()
    }
}

fn strip_zeros(text: &str) -> String {
    if text.contains('.') {
        text.trim_end_matches('0').trim_end_matches('.').to_owned()
    } else {
        text.to_owned()
    }
}

/// A snapshot and the CSV it hashes, cross-checked. Fails closed on any disagreement.
fn load_ladder(p: &Paths, snapshot: &str, csv: &str, ladder: &str) -> Result<(Value, Vec<Dict>)> {
    let snap = p.json(snapshot)?;
    let text = py::read_text(&p.at(csv))?;
    let rows = pycsv::dict_reader(&py::splitlines(&text).join("\n"), Newlines::LineFeed)?.rows;
    if get(&snap, "schema") != &Value::from(SNAPSHOT_SCHEMA) {
        return refuse(format!(
            "{snapshot}: schema {} is not {SNAPSHOT_SCHEMA}",
            get(&snap, "schema").repr()
        ));
    }
    if get(&snap, "ladder") != &Value::from(ladder) {
        return refuse(format!(
            "{snapshot}: ladder {} is not {}",
            get(&snap, "ladder").repr(),
            crate::pyjson::str_repr(ladder)
        ));
    }
    // The snapshot records the hash of the ladder it was derived from. If the committed CSV has
    // moved since, the two files disagree and neither is safe to publish.
    let actual = p.sha256(csv)?;
    if get(&snap, "leaderboard_sha256") != &Value::from(actual.as_str()) {
        return refuse(format!(
            "{csv} sha256 {actual}\ndoes not match {snapshot} leaderboard_sha256 {}.",
            s(get(&snap, "leaderboard_sha256"))
        ));
    }
    if at(&snap, "methods")? != &Value::from(rows.len()) {
        return refuse(format!(
            "{snapshot}: {} methods but the CSV has {} rows",
            s(at(&snap, "methods")?),
            rows.len()
        ));
    }
    Ok((snap, rows))
}

struct Evidence {
    track: Value,
    published: Value,
    published_rows: Vec<Dict>,
    valid: Value,
    valid_rows: Vec<Dict>,
    corrections: Value,
}

fn load_evidence(p: &Paths) -> Result<Evidence> {
    let track = p.json(TRACK)?;
    let (published, published_rows) = load_ladder(p, SNAPSHOT, LEADERBOARD, "published")?;
    let (valid, valid_rows) = load_ladder(p, VALID_SNAPSHOT, VALID_LEADERBOARD, "benchmark-valid")?;
    let corrections = p.json(CORRECTIONS)?;
    if !at(&published, "corrections")?.is_null()
        || at(&published, "excluded_rows")? != &Value::Int(0)
    {
        return refuse(format!(
            "{SNAPSHOT}: the published ladder must apply no corrections"
        ));
    }
    if get(&corrections, "schema") != &Value::from(CORRECTIONS_SCHEMA) {
        return refuse(format!("{CORRECTIONS}: schema is not {CORRECTIONS_SCHEMA}"));
    }
    let (a, b, c) = (
        at(&corrections, "scores_sha256")?,
        at(&published, "scores_sha256")?,
        at(&valid, "scores_sha256")?,
    );
    if !(a == b && b == c) {
        return refuse("the two ladders and the corrections do not bind the same scores.csv");
    }
    let meta = at(&valid, "corrections")?;
    if meta.is_null() || at(meta, "sha256")? != &Value::from(p.sha256(CORRECTIONS)?) {
        return refuse(format!(
            "{VALID_SNAPSHOT} was not computed with the committed {CORRECTIONS}"
        ));
    }
    if at(&valid, "excluded_rows")? != at(meta, "rows_corrected")? {
        // Equal while every published row is True. If the benchmark ever publishes a False row,
        // the ladder excludes it too and this sentence must be rewritten, not silently
        // under-counted.
        return refuse(format!(
            "the benchmark-valid ladder excluded {} rows but the corrections demoted {}; the README prose assumes they are the same set",
            s(at(&valid, "excluded_rows")?),
            s(at(meta, "rows_corrected")?)
        ));
    }
    for row in &valid_rows {
        if cell(row, "benchmark_valid")? != "True" {
            return refuse("a benchmark-invalid method survived into the benchmark-valid ladder");
        }
    }
    // Every correction must show up as fewer scored datasets for its method, so re-adding the
    // rows, or editing the corrections without regenerating the ladder, fails here.
    let mut applied: BTreeMap<String, i64> = BTreeMap::new();
    for a in at(meta, "applied")?.as_list().into_iter().flatten() {
        applied.insert(s(at(a, "id")?), at(a, "rows")?.as_int().unwrap_or(0));
    }
    let listed = at(&corrections, "corrections")?
        .as_list()
        .cloned()
        .unwrap_or_default();
    let ids: std::collections::BTreeSet<String> = listed
        .iter()
        .map(|c| at(c, "id").map(s))
        .collect::<Result<_>>()?;
    if applied
        .keys()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>()
        != ids
    {
        return refuse("the corrections applied to the ladder are not the committed corrections");
    }
    // Insertion-ordered like the Python dict.
    let mut by_method: Vec<(String, i64)> = vec![];
    for c in &listed {
        let method = s(at(c, "method_id")?);
        let n = applied[&s(at(c, "id")?)];
        match by_method.iter_mut().find(|(m, _)| *m == method) {
            Some((_, total)) => *total += n,
            None => by_method.push((method, n)),
        }
    }
    let scored = |rows: &[Dict]| -> Result<BTreeMap<String, i64>> {
        rows.iter()
            .map(|r| {
                Ok((
                    cell(r, "method_id")?.to_owned(),
                    int_cell(r, "datasets_scored")?,
                ))
            })
            .collect()
    };
    let (scored_published, scored_valid) = (scored(&published_rows)?, scored(&valid_rows)?);
    for (method, n) in &by_method {
        let before = scored_published.get(method).copied().unwrap_or(0);
        let after = scored_valid.get(method).copied().unwrap_or(0);
        if before - after != *n {
            return refuse(format!(
                "{method}: {n} rows corrected but scored datasets went {before} -> {after}"
            ));
        }
    }
    Ok(Evidence {
        track,
        published,
        published_rows,
        valid,
        valid_rows,
        corrections,
    })
}

fn table_row(r: &Dict) -> Result<String> {
    let ours = OURS.contains(&cell(r, "method_id")?);
    let bold = |text: &str| {
        if ours {
            format!("**{text}**")
        } else {
            text.to_owned()
        }
    };
    Ok(format!(
        "| {} | {} | {} | {} | {} | {}/{} | `{}` |",
        bold(cell(r, "elo_rank")?),
        bold(cell(r, "method")?),
        bold(cell(r, "elo")?),
        fixed(float_cell(r, "average_rank")?, 2),
        cell(r, "dataset_wins")?,
        cell(r, "datasets_scored")?,
        cell(r, "datasets_total")?,
        cell(r, "protocols")?
    ))
}

/// The top N plus every one of our entries, in rank order, with an ellipsis row per gap.
fn ladder_table(rows: &[Dict]) -> Result<Vec<String>> {
    let mut shown: std::collections::BTreeSet<usize> = (0..TOP_N).collect();
    for (i, r) in rows.iter().enumerate() {
        if OURS.contains(&cell(r, "method_id")?) {
            shown.insert(i);
        }
    }
    let mut out = vec![
        "| rank | method | Elo | avg rank | dataset wins | coverage | protocol |".to_owned(),
        "| ---: | --- | ---: | ---: | ---: | --- | --- |".to_owned(),
    ];
    let mut previous: i64 = -1;
    for i in shown {
        if i as i64 > previous + 1 {
            out.push("| … | | | | | | |".to_owned());
        }
        out.push(table_row(rows.get(i).ok_or_else(|| shape("ladder row"))?)?);
        previous = i as i64;
    }
    Ok(out)
}

fn render_arena(p: &Paths) -> Result<String> {
    let e = load_evidence(p)?;
    let (track, corrections, published, valid) = (&e.track, &e.corrections, &e.published, &e.valid);
    let empty = Value::Dict(Dict::new());
    let split = track.get("split").unwrap_or(&empty);
    let policy = track.get("leaderboard_policy").unwrap_or(&empty);
    let excluded_list: Vec<String> = split
        .get("excluded_datasets")
        .and_then(Value::as_list)
        .map(|l| l.iter().map(s).collect())
        .unwrap_or_default();
    let excluded = if excluded_list.is_empty() {
        "none".to_owned()
    } else {
        excluded_list.join(", ")
    };
    let excluded_rows = s(at(valid, "excluded_rows")?);
    let find = |rows: &[Dict], id: &str| -> Result<Option<Dict>> {
        let mut found = None;
        for r in rows {
            if cell(r, "method_id")? == id {
                found = Some(r.clone());
            }
        }
        Ok(found)
    };
    let mut applied: BTreeMap<String, Value> = BTreeMap::new();
    for a in at(at(valid, "corrections")?, "applied")?
        .as_list()
        .into_iter()
        .flatten()
    {
        applied.insert(s(at(a, "id")?), at(a, "rows")?.clone());
    }
    let by_id = |id: &str| -> Result<Option<Value>> {
        let mut found = None;
        for c in at(corrections, "corrections")?
            .as_list()
            .into_iter()
            .flatten()
        {
            if at(c, "id")? == &Value::from(id) {
                found = Some(c.clone());
            }
        }
        Ok(found)
    };
    let (Some(golden46), Some(golden42)) = (
        by_id("golden46-new3-20260723-v4-test-adaptive")?,
        by_id("golden42-evolution-envelope-test-selected")?,
    ) else {
        return refuse(
            "the README prose is written for the two committed corrections; update both together",
        );
    };
    let first_ref = |c: &Value| -> Result<String> {
        let evidence = at(c, "evidence")?
            .as_list()
            .and_then(|l| l.first())
            .ok_or_else(|| shape("evidence[0]"))?;
        Ok(s(evidence).split(' ').next().unwrap_or_default().to_owned())
    };
    let public = if get(split, "test_labels_public").truthy() {
        "public"
    } else {
        "**not public**"
    };
    let mut out = vec![
        ARENA_BEGIN.to_owned(),
        format!(
            "### {} — {}, {} datasets",
            s(at(track, "public_name")?),
            s(at(track, "task")?),
            s(at(published, "datasets_scored")?)
        ),
        String::new(),
        format!(
            "`{}` · primary metric **{}** ({}) · Elo by `{}` · test labels are {public}; {} datasets are restricted; excluded datasets: {excluded}.",
            s(at(track, "id")?),
            s(at(track, "primary_metric")?).to_uppercase(),
            s(at(track, "metric_direction")?),
            policy.get("elo_method").map_or_else(|| "unknown".to_owned(), s),
            split.get("restricted_dataset_count").map_or_else(|| "0".to_owned(), s),
        ),
        String::new(),
        "#### Benchmark-valid — the ladder this repository claims".to_owned(),
        String::new(),
        format!(
            "**{} methods** · {} rankable scores · {excluded_rows} published rows excluded as test-selected (see the next block). Bold rows are this repository's entries. *Benchmark-valid* here means that no artefact in this repository contradicts the row's published flag; it is not a claim that every comparator's protocol was verified.",
            s(at(valid, "methods")?),
            thousands(at(valid, "rankable_rows")?.as_int().ok_or_else(|| shape("rankable_rows"))?),
        ),
        String::new(),
    ];
    out.extend(ladder_table(&e.valid_rows)?);
    let together = policy
        .get("protocols_ranked_together")
        .map_or_else(|| "False".to_owned(), s)
        .to_lowercase();
    out.extend([
        String::new(),
        format!(
            "**Read the protocol column before the Elo column.** `leaderboard_policy.protocols_ranked_together` is `{together}`, so entries produced under different protocols share one ladder. `jope-dime-regressor` is one fixed-seed reproducible run of this repository's engine under `public-development`, the same protocol as every other comparator. The same-cut benchmark is what turns this into a like-for-like comparison."
        ),
        String::new(),
        "#### Best known program per dataset — as published; test-selected, not a held-out result"
            .to_owned(),
        String::new(),
    ]);
    let Some(pr) = find(&e.published_rows, "jope-dime")? else {
        return refuse("jope-dime is missing from the published ladder");
    };
    if find(&e.valid_rows, "jope-dime")?.is_some() {
        // The prose below is written for the case where every jope-dime row is corrected.
        return refuse(
            "jope-dime still has benchmark-valid rows; the best-known block's prose assumes it has none",
        );
    }
    let applied_of = |c: &Value| -> Result<String> {
        Ok(s(applied
            .get(&s(at(c, "id")?))
            .ok_or_else(|| shape("applied correction"))?))
    };
    let track_id = s(at(track, "id")?);
    out.extend([
        format!(
            "The benchmark published `jope-dime` at rank {}, Elo {}, {} dataset wins, {}/{} coverage, every row `benchmark_valid: True`. This repository's own code says otherwise about every one of those rows, so the benchmark-valid ladder above carries none of them:",
            cell(&pr, "elo_rank")?, cell(&pr, "elo")?, cell(&pr, "dataset_wins")?,
            cell(&pr, "datasets_scored")?, cell(&pr, "datasets_total")?
        ),
        String::new(),
        format!(
            "- **{} Golden42 rows** are each the per-dataset minimum RMSE over the ledger's test-scored candidate evaluations (63,108 across the 42 datasets), not the score of one program frozen before the test targets were opened. The red-team lane in this tree names that quantity the *best-test ASPFM envelope*, role `oracle_reference_only`, and keeps it out of its fair ladder (`{}`, `crates/jope-campaigns/src/commands/train_only_redteam.rs:402-407`).",
            applied_of(&golden42)?, first_ref(&golden42)?
        ),
        format!(
            "- **{} Golden46 rows** come from run `{}`, whose artefacts this repository's campaign code writes with `benchmark_valid: false, test_adaptive: true` (`{}`); the acceptance manifest relabelled the replayed winners `true`.",
            applied_of(&golden46)?, s(at(&golden46, "source_revision")?), first_ref(&golden46)?
        ),
        String::new(),
        "Both are the best program found per dataset when candidates were scored on the test targets: reported here as best-known results, never as a held-out claim. The reproducible fixed-seed run of the same engine, `jope-dime-regressor`, is what stands on the ladder above. The same-cut benchmark replaces every row on this page with numbers this tree can reproduce.".to_owned(),
        String::new(),
        "| ladder | `jope-dime` | Elo | dataset wins | coverage |".to_owned(),
        "| --- | ---: | ---: | ---: | --- |".to_owned(),
        format!(
            "| as published | rank {} | {} | {} | {}/{} |",
            cell(&pr, "elo_rank")?, cell(&pr, "elo")?, cell(&pr, "dataset_wins")?,
            cell(&pr, "datasets_scored")?, cell(&pr, "datasets_total")?
        ),
        format!(
            "| benchmark-valid | not ranked | — | — | 0/{} (every row test-selected) |",
            cell(&pr, "datasets_total")?
        ),
        String::new(),
        format!(
            "Sources: published `{LEADERBOARD}` (sha256 `{}…`), recomputed byte-for-byte from `scores.csv`; benchmark-valid `{VALID_LEADERBOARD}` (sha256 `{}…`), the same `scores.csv` with `{CORRECTIONS}` (sha256 `{}…`) applied. Both are asserted by `crates/jope-arena/tests/elo_regression.rs`. Reproduce offline:",
            &p.sha256(LEADERBOARD)?[..16], &p.sha256(VALID_LEADERBOARD)?[..16], &p.sha256(CORRECTIONS)?[..16]
        ),
        String::new(),
        "```sh".to_owned(),
        "cargo run -p jope-arena --bin jope-arena -- elo \\".to_owned(),
        format!("  --track {track_id} --root {ARENA} --check"),
        "cargo run -p jope-arena --bin jope-arena -- elo \\".to_owned(),
        format!("  --track {track_id} --root {ARENA} \\"),
        format!("  --ladder benchmark-valid --corrections {CORRECTIONS} \\"),
        format!("  --check --committed {VALID_LEADERBOARD}"),
        "```".to_owned(),
        ARENA_END.to_owned(),
    ]);
    let _ = REGRESSION;
    Ok(out.join("\n"))
}

struct SameCut {
    deep: Value,
    deep_rows: Vec<Dict>,
    reports: Vec<Value>,
    ours: Dict,
    engines: BTreeMap<String, Dict>,
    versions: BTreeMap<String, String>,
    specs: Vec<Value>,
}

fn kind_of(m: &Value) -> Result<String> {
    Ok(s(at(at(m, "method")?, "kind")?))
}

fn engine_specs(r: &Value) -> Result<Vec<Value>> {
    let mut specs = vec![];
    for m in at(r, "methods")?.as_list().into_iter().flatten() {
        if kind_of(m)? == "adapter" {
            specs.push(at(at(m, "method")?, "adapter")?.clone());
        }
    }
    let mut keyed: Vec<(String, Value)> = specs
        .into_iter()
        .map(|spec| Ok((s(at(&spec, "name")?), spec)))
        .collect::<Result<_>>()?;
    // `sorted(..., key=name)` is stable.
    keyed.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(keyed.into_iter().map(|(_, spec)| spec).collect())
}

/// The committed same-cut evidence, cross-checked. Fails closed on any disagreement.
fn load_samecut(p: &Paths) -> Result<SameCut> {
    let deep_csv = samecut_path(&format!("leaderboard-{SAMECUT_TRACK_ID}-deep.csv"));
    let deep_snapshot = samecut_path(&format!("elo-{SAMECUT_TRACK_ID}-deep.json"));
    let screen_csv = samecut_path(&format!("leaderboard-{SAMECUT_TRACK_ID}-screen.csv"));
    let screen_snapshot = samecut_path(&format!("elo-{SAMECUT_TRACK_ID}-screen.json"));
    let (deep, deep_rows) = load_ladder(p, &deep_snapshot, &deep_csv, "published")?;
    let (screen, _screen_rows) = load_ladder(p, &screen_snapshot, &screen_csv, "published")?;
    let track = p.json(&samecut_path(&format!(
        "track/tracks/{SAMECUT_TRACK_ID}-deep/track.json"
    )))?;
    let rel = SAMECUT;
    let scores = samecut_path(&format!("track/results/{SAMECUT_TRACK_ID}-deep/scores.csv"));
    if at(&deep, "scores_sha256")? != &Value::from(p.sha256(&scores)?) {
        return refuse(format!(
            "{rel}: the deep snapshot was not computed from the committed scores.csv"
        ));
    }
    if at(&track, "id")? != &Value::from(format!("{SAMECUT_TRACK_ID}-deep"))
        || at(&track, "primary_metric")? != &Value::from("rmse")
    {
        return refuse(format!(
            "{rel}: the deep track is not {SAMECUT_TRACK_ID}-deep ranked by rmse"
        ));
    }
    if !at(&deep, "corrections")?.is_null() || at(&deep, "excluded_rows")? != &Value::Int(0) {
        return refuse(format!("{rel}: the same-cut ladder applies no corrections"));
    }
    let mut report_paths: Vec<PathBuf> = std::fs::read_dir(p.at(&samecut_path("reports")))
        .map_err(|e| Error::io(&p.at(&samecut_path("reports")), e))?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.ends_with(".bench-report.json") && !n.starts_with('.'))
        })
        .collect();
    report_paths.sort();
    let reports: Vec<Value> = report_paths
        .iter()
        .map(|path| py::load_json(path))
        .collect::<Result<_>>()?;
    if reports.is_empty() {
        return refuse(format!("{rel}: no reports"));
    }
    let n = Value::from(reports.len());
    if !(&n == at(&deep, "datasets_scored")?
        && at(&deep, "datasets_scored")? == at(&screen, "datasets_scored")?)
    {
        return refuse(format!(
            "{rel}: {} reports but the ladders scored other dataset counts",
            reports.len()
        ));
    }
    let first = &reports[0];
    for r in &reports {
        let id = s(get(r, "dataset_id"));
        if at(r, "schema")? != &Value::from(REPORT_SCHEMA) {
            return refuse(format!(
                "{rel}: report schema {} is not {REPORT_SCHEMA}",
                at(r, "schema")?.repr()
            ));
        }
        if !(at(r, "task")? == &Value::from("regression")
            && at(r, "ladder_metric")? == &Value::from("rmse")
            && at(r, "metric_direction")? == &Value::from("minimize"))
        {
            return refuse(format!(
                "{rel}: {} is not a regression report ranked by rmse",
                s(at(r, "dataset_id")?)
            ));
        }
        for key in ["protocol", "search", "git_sha", "host", "evaluator"] {
            if get(r, key) != get(first, key) || get(r, key).is_null() {
                return refuse(format!(
                    "{rel}: {id} differs from the first report in {key}"
                ));
            }
        }
        // The plan key names the dataset; seed and fold count are what one run shares.
        let plan = at(r, "plan")?;
        let first_plan = at(first, "plan")?;
        if at(plan, "seed")? != at(first_plan, "seed")? || at(plan, "k")? != at(first_plan, "k")? {
            return refuse(format!(
                "{rel}: {id} differs from the first report in fold seed or k"
            ));
        }
        let methods = at(r, "methods")?.as_list().cloned().unwrap_or_default();
        let mut searched = vec![];
        for m in &methods {
            if kind_of(m)? == "search" {
                searched.push(m.clone());
            }
        }
        if searched.len() != 1 || at(at(&searched[0], "method")?, "id")? != &Value::from(SEARCHED) {
            return refuse(format!(
                "{rel}: {id} does not carry exactly one {SEARCHED} row"
            ));
        }
        if get(at(&searched[0], "method")?, "resample") != at(r, "search")? {
            return refuse(format!(
                "{rel}: {id}: the searched row is not the resampled procedure"
            ));
        }
        let mut adapters = vec![];
        for m in &methods {
            if kind_of(m)? == "adapter" {
                adapters.push(s(at(at(at(m, "method")?, "adapter")?, "name")?));
            }
        }
        adapters.sort();
        if adapters != SAMECUT_ADAPTERS {
            return refuse(format!(
                "{rel}: {id} carries adapters {}, not {}",
                py::str_list_repr(&adapters),
                py::str_list_repr(&SAMECUT_ADAPTERS)
            ));
        }
        for m in &methods {
            if kind_of(m)? == "adapter"
                && !get(at(at(m, "method")?, "adapter")?, "version").truthy()
            {
                return refuse(format!(
                    "{rel}: {id}: {} records no engine version",
                    s(at(at(m, "method")?, "id")?)
                ));
            }
        }
    }
    let row_for = |id: &str| -> Result<Option<Dict>> {
        for row in &deep_rows {
            if cell(row, "method_id")? == id {
                return Ok(Some(row.clone()));
            }
        }
        Ok(None)
    };
    let Some(ours) = row_for(SEARCHED)? else {
        return refuse(format!("{rel}: {SEARCHED} is missing from the deep ladder"));
    };
    let mut engines = BTreeMap::new();
    for name in SAMECUT_ADAPTERS {
        let Some(row) = row_for(&format!("adapter.{name}"))? else {
            return refuse(format!(
                "{rel}: adapter.{name} is missing from the deep ladder"
            ));
        };
        engines.insert(name.to_owned(), row);
    }
    // The engine specs come from the first report; every other report must have recorded the
    // same spec per engine, or the directory is not one run.
    let specs = engine_specs(first)?;
    for r in &reports {
        if engine_specs(r)? != specs {
            return refuse(format!(
                "{rel}: {} recorded different engine specs or versions",
                s(get(r, "dataset_id"))
            ));
        }
    }
    let versions = specs
        .iter()
        .map(|spec| Ok((s(at(spec, "name")?), s(at(spec, "version")?))))
        .collect::<Result<_>>()?;
    Ok(SameCut {
        deep,
        deep_rows,
        reports,
        ours,
        engines,
        versions,
        specs,
    })
}

/// The `--adapter` argument that recorded this engine, rendered so a shell reads it back
/// exactly; a token that would need quoting refuses rather than print a wrong command.
fn adapter_flag(spec: &Value) -> Result<String> {
    let safe = Regex::new(r"^[A-Za-z0-9_./=:+-]+$").expect("static pattern");
    let name = s(at(spec, "name")?);
    let command: Vec<String> = at(spec, "command")?
        .as_list()
        .map(|l| l.iter().map(s).collect())
        .unwrap_or_default();
    for token in std::iter::once(&name).chain(command.iter()) {
        if !safe.is_match(token) {
            return refuse(format!(
                "adapter {}: token {} is not shell-safe to render",
                crate::pyjson::str_repr(&name),
                crate::pyjson::str_repr(token)
            ));
        }
    }
    Ok(format!("--adapter \"{name}={}\"", command.join(" ")))
}

fn samecut_row(r: &Dict, versions: &BTreeMap<String, String>) -> Result<String> {
    let method_id = cell(r, "method_id")?;
    let b = if method_id == SEARCHED { "**" } else { "" };
    let mut name = cell(r, "method")?.to_owned();
    if let Some(engine) = method_id.strip_prefix("adapter.") {
        let version = versions
            .get(engine)
            .map_or("version unknown", String::as_str);
        name = format!("{name} ({version})");
    }
    Ok(format!(
        "| {b}{}{b} | {b}{name}{b} | {b}{}{b} | {} | {} | {}/{} |",
        cell(r, "elo_rank")?,
        cell(r, "elo")?,
        fixed(float_cell(r, "average_rank")?, 2),
        cell(r, "dataset_wins")?,
        cell(r, "datasets_scored")?,
        cell(r, "datasets_total")?
    ))
}

fn render_samecut(p: &Paths) -> Result<String> {
    let e = load_samecut(p)?;
    let (deep, reports, ours) = (&e.deep, &e.reports, &e.ours);
    let first = &reports[0];
    let (plan, protocol, search, host) = (
        at(first, "plan")?,
        at(first, "protocol")?,
        at(first, "search")?,
        at(first, "host")?,
    );
    let n = reports.len();
    let mut per_dataset = vec![];
    let mut search_seconds: Vec<f64> = vec![];
    let (mut finalist, mut best) = (0usize, 0usize);
    let mut ordered: Vec<&Value> = reports.iter().collect();
    ordered.sort_by(|a, b| s(get(a, "dataset_id")).cmp(&s(get(b, "dataset_id"))));
    for r in ordered {
        let methods = at(r, "methods")?.as_list().cloned().unwrap_or_default();
        let s_index = methods
            .iter()
            .position(|m| kind_of(m).is_ok_and(|k| k == "search"))
            .ok_or_else(|| shape("search method"))?;
        let sm = &methods[s_index];
        // "Best" is the ladder's own rule: the lowest pooled RMSE among the finalists that
        // scored, ties by method id.
        let mut top: Option<(usize, f64, String)> = None;
        for (i, m) in methods.iter().enumerate() {
            let deep_stage = get(m, "deep");
            if !(deep_stage.truthy() && at(deep_stage, "status")? == &Value::from("ok")) {
                continue;
            }
            let rmse = float_of(at(at(deep_stage, "oof")?, "rmse")?)?;
            let id = s(at(at(m, "method")?, "id")?);
            let better = match &top {
                None => true,
                Some((_, best_rmse, best_id)) => {
                    rmse < *best_rmse || (rmse == *best_rmse && id < *best_id)
                }
            };
            if better {
                top = Some((i, rmse, id));
            }
        }
        let Some((top_index, rmse, top_id)) = top else {
            return refuse(format!(
                "{SAMECUT}: {} has no scored finalist",
                s(at(r, "dataset_id")?)
            ));
        };
        let produce = float_of(at(at(sm, "method")?, "produce_seconds")?)?;
        search_seconds.push(produce);
        let rank = get(sm, "deep_rank");
        if !rank.is_null() {
            finalist += 1;
        }
        let is_search = top_index == s_index;
        if is_search {
            best += 1;
        }
        let marker = if is_search { "**" } else { "" };
        let rank_text = if rank.is_null() {
            format!("screen {}", s(get(sm, "screen_rank")))
        } else {
            s(rank)
        };
        per_dataset.push(format!(
            "| `{}` | {} | {} | {marker}{top_id}{marker} | {} | {rank_text} | {} s |",
            s(at(r, "dataset_id")?),
            s(at(r, "rows")?),
            s(at(r, "features")?),
            general4(rmse),
            fixed(produce, 1)
        ));
    }
    let mean_s = search_seconds.iter().fold(0.0, |a, b| a + b) / n as f64;
    let mut sorted_s = search_seconds.clone();
    sorted_s.sort_by(f64::total_cmp);
    let median_s = if n % 2 == 1 {
        sorted_s[n / 2]
    } else {
        (sorted_s[n / 2 - 1] + sorted_s[n / 2]) / 2.0
    };
    let max_s = search_seconds
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let git_sha = s(at(first, "git_sha")?);
    let mut out = vec![
        SAMECUT_BEGIN.to_owned(),
        "### Same-cut — every method on identical held-out rows, reproducible from this tree".to_owned(),
        String::new(),
        format!(
            "`{}` · {n} regression datasets from `datasets/test` · primary metric **RMSE** (minimize) · fold plan k={}, seed {}, screen fold {}, top {} deepened · search budget {} programs × {} rounds, re-run on every cut · recorded at `{}` with {} workers on {} cores.",
            s(at(deep, "track_id")?), s(at(plan, "k")?), s(at(plan, "seed")?),
            s(at(protocol, "screen_fold")?), s(at(protocol, "top")?), s(at(search, "pop")?),
            s(at(search, "gens")?), git_sha.chars().take(9).collect::<String>(),
            s(at(host, "workers")?), s(at(host, "cores")?)
        ),
        String::new(),
        format!(
            "Every row here is a number this tree produced and can reproduce: the ten single-head baselines, four external engines (XGBoost, LightGBM, CatBoost and a scikit-learn random forest, each run as a subprocess with its library defaults, a fixed seed and one thread, handed a cut's training rows and its held-out features only, and scored by this benchmark from the predictions it returns) and the program the search wrote for each dataset, all fitted on the same training rows and scored on the same held-out rows by one evaluator (`{SAMECUT_DOC}`). The search sees only a cut's training rows, and its deep row is the *procedure* re-run on every cut, not one program chosen with knowledge of the other cuts' held-out rows. Bold is the searched program."
        ),
        String::new(),
        "**The datasets are a favourable selection.** They are the `test-10` collection, chosen from the published ladder where the historical engine ranked first (`docs/datasets.md`), and they are used here because they are the only committed regression data. Identical cuts make the comparison between methods fair on these ten; they say nothing about a broader panel. The frozen 30-dataset PMLB panel of the experiment contract (`docs/asm.md`) excludes them on purpose; a same-cut run over it is the next evidence step.".to_owned(),
        String::new(),
        "| rank | method | Elo | avg rank | dataset wins | coverage |".to_owned(),
        "| ---: | --- | ---: | ---: | ---: | --- |".to_owned(),
    ];
    for r in &e.deep_rows {
        out.push(samecut_row(r, &e.versions)?);
    }
    let ours_rank = int_cell(ours, "elo_rank")?;
    let mut standing = vec![];
    for name in SAMECUT_ADAPTERS {
        let engine = &e.engines[name];
        let side = if ours_rank < int_cell(engine, "elo_rank")? {
            "above"
        } else {
            "below"
        };
        standing.push(format!(
            "{side} {} (rank {})",
            cell(engine, "method")?,
            cell(engine, "elo_rank")?
        ));
    }
    let top = s(at(protocol, "top")?);
    out.extend([
        String::new(),
        format!(
            "**Where the search stands today:** rank {} of {} on the deep ladder, Elo {}, lowest pooled RMSE on {best} of {n} datasets (the ladder's own rule), a finalist on {finalist}; on the other {} it did not make the top-{top} screen. Against the four external engines it ranks {}. Coverage counts datasets where a method reached the deep stage, so the ladder ranks finalists only. The screen ladder (every method, one cut) is committed beside it.",
            cell(ours, "elo_rank")?, s(at(deep, "methods")?), cell(ours, "elo")?,
            n - finalist, standing.join(", ")
        ),
        String::new(),
        format!(
            "**Seconds per program.** Writing the program took the search {} s per dataset on average (median {} s, longest {} s) on the screen cut's training rows, wall time under whatever else the host was running. The owner's stated reference point for a large language model writing a program for the same task, not measured here, is about {LLM_REFERENCE_HOURS} hour and {} tokens; that comparison is one of cost, not of quality, and the quality standing is the ladder above.",
            fixed(mean_s, 1), fixed(median_s, 1), fixed(max_s, 1), thousands(LLM_REFERENCE_TOKENS)
        ),
        String::new(),
        "| dataset | rows | features | lowest pooled RMSE | RMSE | search rank (penalised score) | search |".to_owned(),
        "| --- | ---: | ---: | --- | ---: | ---: | ---: |".to_owned(),
    ]);
    out.extend(per_dataset);
    let deep_csv = samecut_path(&format!("leaderboard-{SAMECUT_TRACK_ID}-deep.csv"));
    let deep_scores = samecut_path(&format!("track/results/{SAMECUT_TRACK_ID}-deep/scores.csv"));
    out.extend([
        String::new(),
        format!(
            "Sources: `{deep_csv}` (sha256 `{}…`) recomputed byte-for-byte from `{deep_scores}`, itself re-derived from the {n} reports under `{}` by `crates/jope-bench/tests/committed_evidence.rs`, which also re-evaluates one dataset's baselines on the same cuts. Provenance and the exact commands: `{SAMECUT}/SOURCES.md`. Reproduce:",
            &p.sha256(&deep_csv)?[..16], samecut_path("reports")
        ),
        String::new(),
        "```sh".to_owned(),
        "cargo run --release -p jope-bench -- run --datasets datasets/test --out target/same-cut/reports \\".to_owned(),
    ]);
    for spec in &e.specs {
        out.push(format!("  {} \\", adapter_flag(spec)?));
    }
    out.extend([
        format!(
            "  --top {top} --search --search-pop {} --search-gens {} --resample-search --workers {}",
            s(at(search, "pop")?), s(at(search, "gens")?), s(at(host, "workers")?)
        ),
        "cargo run --release -p jope-bench -- ladder --reports target/same-cut/reports \\".to_owned(),
        format!("  --track-root target/same-cut/track --track {SAMECUT_TRACK_ID}"),
        format!(
            "cargo run -p jope-arena -- elo --track {} --root target/same-cut/track \\",
            s(at(deep, "track_id")?)
        ),
        format!("  --check --committed {deep_csv}"),
        "```".to_owned(),
        SAMECUT_END.to_owned(),
    ]);
    Ok(out.join("\n"))
}

fn render_bench(p: &Paths) -> Result<String> {
    let b = p.json(BENCH)?;
    let num = |key: &str| -> Result<f64> { float_of(at(&b, key)?) };
    let git_sha = s(at(&b, "git_sha")?);
    Ok([
        BENCH_BEGIN.to_owned(),
        "### Emit-path latency".to_owned(),
        String::new(),
        format!(
            "`{}` — {} runs of the release `emit` example on `{}`.",
            s(at(&b, "bench")?),
            s(at(&b, "runs")?),
            s(at(&b, "genome")?)
        ),
        String::new(),
        "| min | median | mean | p95 |".to_owned(),
        "| ---: | ---: | ---: | ---: |".to_owned(),
        format!(
            "| {} ms | {} ms | {} ms | {} ms |",
            fixed(num("min_ms")?, 2),
            fixed(num("median_ms")?, 2),
            fixed(num("mean_ms")?, 2),
            fixed(num("p95_ms")?, 2)
        ),
        String::new(),
        format!(
            "**This is not an engine-speed number.** Its own `what` field says it measures \"{}\". The min-to-p95 spread is dominated by process spawn and scheduling, not by emission. Treat it as a regression tripwire (alert at {}% drift), not as a performance claim.",
            s(at(&b, "what")?),
            fixed(num("alert_pct")?, 0)
        ),
        String::new(),
        format!(
            "Recorded {} on `{}` ({}, {} cores, kernel {}) at `{}` with {}. Baselines are host-pinned; `bash deploy/ci/regression-suite.sh bench` compares against this file.",
            s(at(&b, "recorded")?),
            s(at(&b, "host")?),
            s(at(&b, "cpu")?),
            s(at(&b, "nproc")?),
            s(at(&b, "kernel")?),
            git_sha.chars().take(9).collect::<String>(),
            s(at(&b, "rustc")?)
        ),
        BENCH_END.to_owned(),
    ]
    .join("\n"))
}

fn splice(text: &str, begin: &str, end: &str, block: &str) -> Result<String> {
    match (text.find(begin), text.find(end)) {
        (Some(i), Some(j)) => Ok(format!("{}{block}{}", &text[..i], &text[j + end.len()..])),
        _ => Err(Error::Invalid(format!(
            "markers {begin}/{end} not found; cannot splice"
        ))),
    }
}

/// The README with every block re-rendered (inserting the section on first use).
pub fn build(root: &Path, text: &str) -> Result<String> {
    let p = Paths {
        root: root.to_path_buf(),
    };
    let mut text = text.to_owned();
    if !text.contains(ARENA_BEGIN) {
        let Some(anchor) = text.find(ANCHOR) else {
            return Err(Error::Invalid(format!(
                "anchor {} not found in README",
                crate::pyjson::str_repr(ANCHOR)
            )));
        };
        let section = format!(
            "## Measured results\n\nGenerated by `jope-records readme-benchmarks` from committed evidence. Do not edit between the markers by hand.\n\n{ARENA_BEGIN}\n{ARENA_END}\n\n{SAMECUT_BEGIN}\n{SAMECUT_END}\n\n{BENCH_BEGIN}\n{BENCH_END}\n\n"
        );
        text = format!("{}{section}{}", &text[..anchor], &text[anchor..]);
    }
    if !text.contains(SAMECUT_BEGIN) {
        // The same-cut block was added after the other two: slot it between them.
        let Some(i) = text.find(BENCH_BEGIN) else {
            return Err(Error::Invalid(format!(
                "markers {BENCH_BEGIN} not found; cannot place the same-cut block"
            )));
        };
        text = format!(
            "{}{SAMECUT_BEGIN}\n{SAMECUT_END}\n\n{}",
            &text[..i],
            &text[i..]
        );
    }
    let text = splice(&text, ARENA_BEGIN, ARENA_END, &render_arena(&p)?)?;
    let text = splice(&text, SAMECUT_BEGIN, SAMECUT_END, &render_samecut(&p)?)?;
    splice(&text, BENCH_BEGIN, BENCH_END, &render_bench(&p)?)
}

/// Every input the render reads; a missing one refuses before anything is read.
pub fn required_inputs() -> Vec<String> {
    vec![
        LEADERBOARD.to_owned(),
        TRACK.to_owned(),
        SNAPSHOT.to_owned(),
        VALID_LEADERBOARD.to_owned(),
        VALID_SNAPSHOT.to_owned(),
        CORRECTIONS.to_owned(),
        BENCH.to_owned(),
        README.to_owned(),
        samecut_path(&format!("leaderboard-{SAMECUT_TRACK_ID}-deep.csv")),
        samecut_path(&format!("elo-{SAMECUT_TRACK_ID}-deep.json")),
        samecut_path(&format!("leaderboard-{SAMECUT_TRACK_ID}-screen.csv")),
        samecut_path(&format!("elo-{SAMECUT_TRACK_ID}-screen.json")),
        samecut_path(&format!("track/tracks/{SAMECUT_TRACK_ID}-deep/track.json")),
        samecut_path(&format!("track/results/{SAMECUT_TRACK_ID}-deep/scores.csv")),
        samecut_path("reports"),
        SAMECUT_DOC.to_owned(),
    ]
}

/// `--check` (write = false) or `--write`: the message, and whether the command succeeded.
pub fn run(root: &Path, write: bool) -> Result<(String, bool)> {
    for input in required_inputs() {
        let path = root.join(&input);
        if !path.exists() {
            return Err(Error::Invalid(format!(
                "missing required input: {}",
                path.display()
            )));
        }
    }
    let readme = root.join(README);
    let current = py::read_text(&readme)?;
    let updated = build(root, &current)?;
    if write {
        if updated != current {
            std::fs::write(&readme, &updated).map_err(|e| Error::io(&readme, e))?;
            return Ok(("README.md updated\n".to_owned(), true));
        }
        return Ok(("README.md already current\n".to_owned(), true));
    }
    if updated != current {
        return Ok((
            "README.md is out of date with the committed benchmark evidence.\nRun: cargo run -q --locked -p jope-records -- readme-benchmarks --write\n".to_owned(),
            false,
        ));
    }
    Ok(("README.md benchmark blocks are current\n".to_owned(), true))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Python's `format` for the specs the blocks use.
    #[test]
    fn number_formats_match_python() {
        assert_eq!(fixed(0.125, 2), "0.12");
        assert_eq!(fixed(0.375, 2), "0.38");
        assert_eq!(fixed(2.5, 0), "2");
        assert_eq!(fixed(1.05, 1), "1.1");
        assert_eq!(thousands(10_000_000), "10,000,000");
        assert_eq!(thousands(999), "999");
        assert_eq!(thousands(8763), "8,763");
        assert_eq!(general4(0.123456), "0.1235");
        assert_eq!(general4(1234.5), "1234");
        assert_eq!(general4(12345.0), "1.234e+04");
        assert_eq!(general4(0.0001234), "0.0001234");
        assert_eq!(general4(0.00001234), "1.234e-05");
        assert_eq!(general4(2.0), "2");
        assert_eq!(general4(0.5), "0.5");
        assert_eq!(general4(9999.5), "1e+04");
    }
}
