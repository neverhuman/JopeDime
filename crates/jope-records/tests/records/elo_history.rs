//! Port of `test_elo_history.py` (18 tests), plus the no-decoder path, decoding and
//! sensitivity checks against the Python replay's recorded results.

use std::collections::HashMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use jope_records::elo_history::{
    self, authorize, continuity, derived_view, exposure, ladder, lineage_graph, parse_scores,
    Ladder, LadderOptions, Summary, Zstd, CONSUMER, ITERATIONS, SCORE_COLUMNS,
};
use jope_records::py::{self, Key};
use jope_records::pycsv::{self, Newlines};
use jope_records::pyjson::{self, Dict, Value};

use crate::{assert_refused, at, digest, dump, json, load, nth, pop, repo, scratch, set, strs};

const FAST: [&str; 4] = [
    "H-same-cut-v1-deep",
    "H-same-cut-v1-screen",
    "H-same-cut-v2-deep",
    "H-same-cut-v2-screen",
];
const GOLDEN45: [&str; 2] = [
    "H-sotarena-golden45-benchmark-valid",
    "H-sotarena-golden45-published",
];

fn record() -> Value {
    load(&repo().join(elo_history::RECORD))
}

fn committed() -> Summary {
    let repo = repo();
    elo_history::verify(&repo.join(elo_history::RECORD), &repo, Zstd::Builtin)
        .expect("the committed record verifies")
}

/// Every repository file a record pins, wherever it appears.
fn pinned(node: &Value, out: &mut Vec<String>) {
    match node {
        Value::Dict(dict) => {
            if let (Some(Value::Str(path)), true) = (dict.get("path"), dict.contains_key("sha256"))
            {
                out.push(path.clone());
            }
            for value in dict.values() {
                pinned(value, out);
            }
        }
        Value::List(list) => list.iter().for_each(|v| pinned(v, out)),
        _ => {}
    }
}

/// The record without the zstd-compressed histories and the view that needs them.
fn fast(mut document: Value) -> Value {
    at(&mut document, "histories")
        .as_list_mut()
        .unwrap()
        .retain(|h| {
            h.get("disposition") != Some(&Value::from(elo_history::REPRODUCIBLE))
                || FAST.iter().any(|id| h.get("id") == Some(&Value::from(*id)))
        });
    set(&mut document, "derived_views", Value::List(vec![]));
    document
}

fn history<'a>(document: &'a mut Value, id: &str) -> &'a mut Value {
    at(document, "histories")
        .as_list_mut()
        .unwrap()
        .iter_mut()
        .find(|h| h.get("id") == Some(&Value::from(id)))
        .unwrap_or_else(|| panic!("no history {id}"))
}

fn lineage<'a>(document: &'a mut Value, id: &str) -> &'a mut Value {
    at(document, "lineages")
        .as_list_mut()
        .unwrap()
        .iter_mut()
        .find(|h| h.get("id") == Some(&Value::from(id)))
        .unwrap_or_else(|| panic!("no lineage {id}"))
}

/// A throwaway repository holding a record and exactly the files it pins.
struct Copy {
    _directory: tempfile::TempDir,
    root: PathBuf,
    document: Value,
    path: PathBuf,
}

impl Copy {
    fn new(document: Value) -> Self {
        let directory = scratch();
        let root = directory.path().to_path_buf();
        let mut paths = Vec::new();
        pinned(&document, &mut paths);
        paths.sort();
        paths.dedup();
        let source = repo();
        for path in paths {
            let target = root.join(&path);
            std::fs::create_dir_all(target.parent().unwrap()).unwrap();
            std::fs::copy(source.join(&path), &target).unwrap_or_else(|e| panic!("{path}: {e}"));
        }
        let path = root.join("histories.json");
        Copy {
            _directory: directory,
            root,
            document,
            path,
        }
    }

    fn item(&mut self, history_id: &str, keys: &[&str]) -> &mut Value {
        let mut node = history(&mut self.document, history_id);
        for key in keys {
            node = at(node, key);
        }
        node
    }

    fn verify(&self) -> jope_records::Result<Summary> {
        dump(&self.path, &self.document);
        elo_history::verify(&self.path, &self.root, Zstd::Builtin)
    }
}

/// Change a pinned file and re-pin it, as a publisher substituting evidence would.
fn rewrite(root: &Path, item: &mut Value, change: impl FnOnce(Vec<u8>) -> Vec<u8>) {
    let path = root.join(item.get("path").and_then(Value::as_str).unwrap());
    let changed = change(std::fs::read(&path).unwrap());
    std::fs::write(&path, changed).unwrap();
    set(item, "sha256", digest(&path).into());
}

fn scores_csv(rows: &[(&str, &str, &str)]) -> String {
    let mut lines = vec![SCORE_COLUMNS.join(",")];
    for (dataset, method, value) in rows {
        lines.push(format!(
            "t,{dataset},{method},same-cut,rmse,{value},ok,True,k,r,,,"
        ));
    }
    lines.join("\n") + "\n"
}

fn board_with(rows: &[(&str, &str, &str)], minimize: bool) -> Ladder {
    let parsed = parse_scores(&scores_csv(rows), "t").expect("fixture scores");
    let datasets: std::collections::HashSet<&str> = rows.iter().map(|r| r.0).collect();
    let (names, pending) = (HashMap::new(), HashMap::new());
    ladder(
        &parsed,
        &LadderOptions {
            ladder_kind: "published",
            minimize,
            names: &names,
            pending: &pending,
            total_datasets: datasets.len(),
        },
    )
    .expect("fixture ladder")
}

fn board(rows: &[(&str, &str, &str)]) -> Ladder {
    board_with(rows, true)
}

fn groups(items: &[&[&str]]) -> Vec<Vec<String>> {
    items
        .iter()
        .map(|g| g.iter().map(|m| (*m).to_owned()).collect())
        .collect()
}

fn sorted_ids(ids: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut ids: Vec<String> = ids.into_iter().collect();
    ids.sort();
    ids
}

mod committed_record {
    use super::*;

    #[test]
    fn every_reproducible_history_is_replayed_or_explicitly_not_here() {
        let summary = committed();
        let reproducible = sorted_ids(
            record()
                .get("histories")
                .and_then(Value::as_list)
                .unwrap()
                .iter()
                .filter(|h| h.get("disposition") == Some(&Value::from(elo_history::REPRODUCIBLE)))
                .map(|h| h.get("id").unwrap().py_str()),
        );
        let replayed = summary.replayed.clone();
        let missing: Vec<String> = summary.not_replayed_here.keys().cloned().collect();
        assert_eq!(
            sorted_ids(replayed.iter().chain(&missing).cloned()),
            reproducible
        );
        assert!(replayed.iter().all(|id| !missing.contains(id)));
        assert!(FAST.iter().all(|id| replayed.iter().any(|r| r == id)));
        // The decoder is built in, so every reproducible history replays on every host.
        assert!(missing.is_empty());
    }

    #[test]
    fn only_replayed_histories_enter_exact_totals() {
        let summary = committed();
        let every = sorted_ids(
            record()
                .get("histories")
                .and_then(Value::as_list)
                .unwrap()
                .iter()
                .map(|h| h.get("id").unwrap().py_str()),
        );
        let excluded: Vec<String> = every
            .into_iter()
            .filter(|id| !summary.replayed.contains(id))
            .collect();
        assert_eq!(summary.excluded_from_exact_totals, excluded);
        let json = summary.to_json();
        let totals = json.get("exact_totals").unwrap();
        assert_eq!(
            totals.get("histories"),
            Some(&Value::from(summary.replays.len()))
        );
        assert_eq!(
            totals.get("rankable_rows"),
            Some(&Value::from(
                summary
                    .replays
                    .values()
                    .map(|r| r.rankable_rows)
                    .sum::<usize>()
            ))
        );
        for id in summary
            .irrecoverable
            .iter()
            .chain(&summary.scorer_parity_only)
        {
            assert!(!summary.replays.contains_key(id), "{id}");
        }
    }

    #[test]
    fn committed_order_of_exchangeable_methods_is_reported_not_trusted() {
        let summary = committed();
        let replays = &summary.replays;
        assert!(replays["H-same-cut-v1-deep"].byte_identical);
        let screen = &replays["H-same-cut-v1-screen"];
        assert!(!screen.byte_identical);
        let pair = groups(&[&["baseline.elasticnet", "baseline.ridge"]]);
        assert_eq!(screen.order_differs_from_replay, pair);
        assert!(screen.exchangeable_groups.contains(&pair[0]));
        let published = &replays["H-sotarena-golden45-published"];
        assert_eq!(
            published.order_differs_from_replay,
            groups(&[&["ngboost", "xgboost_hist_trees900_depth4_lr0p02"]])
        );
        assert!(!published.converged); // the retained ratings are the 250th iterate
        assert_eq!(published.iterations, ITERATIONS);
        assert!(replays["H-sotarena-golden45-benchmark-valid"].byte_identical);
    }

    #[test]
    fn every_rating_is_historical_ranking_only() {
        let json = committed().to_json();
        assert_eq!(
            json.get("consumer"),
            Some(&Value::from("historical-ranking-only"))
        );
        assert_eq!(
            json.get("refused_consumers").map(Value::repr).as_deref(),
            Some("['teaching', 'hwm', 'transfer', 'promotion']")
        );
        let replays = json.get("replays").and_then(Value::as_dict).unwrap();
        assert!(replays
            .values()
            .all(|r| r.get("consumer") == Some(&Value::from("historical-ranking-only"))));
    }

    #[test]
    fn lineage_edges_and_derived_view() {
        let summary = committed();
        let count = |lineage: &str, field: &str| {
            summary.lineages[lineage]
                .get(field)
                .and_then(Value::as_int)
                .unwrap()
        };
        let a100 = "L-lineage-b-a100-recovery";
        assert_eq!(
            (
                count(a100, "checkpoints"),
                count(a100, "retained_parent_edges"),
                count(a100, "unretained_parent_edges")
            ),
            (78, 74, 4)
        );
        let audit = "L-native-audit-108-blocks";
        assert_eq!(
            (
                count(audit, "checkpoints"),
                count(audit, "retained_parent_edges")
            ),
            (108, 73)
        );
        assert_eq!(
            summary
                .derived_view_rows
                .iter()
                .map(|(k, v)| (k.as_str(), *v))
                .collect::<Vec<_>>(),
            [("V-plot-historical-elo", 661)]
        );
    }

    #[test]
    fn without_a_zstd_decoder_golden45_is_not_replayed_and_require_all_fails() {
        // Addition: the Python replay's behaviour on a host without `zstd`, recorded in the
        // slice-2 baseline (`elo_history_nozstd.out`): the Golden45 ladders are listed, never
        // counted, and `--require-all` refuses.
        let repo = repo();
        let summary =
            elo_history::verify(&repo.join(elo_history::RECORD), &repo, Zstd::Unavailable)
                .expect("verifies without zstd");
        assert_eq!(summary.replayed, FAST);
        let reason =
            "assets/regression/sotarena-tabular-regression-v2/results/tabular-regression-v2/\
                      scores.csv.zst: no zstd decoder on this host";
        assert_eq!(
            summary
                .not_replayed_here
                .iter()
                .map(|(k, v)| (k.as_str(), v.as_str()))
                .collect::<Vec<_>>(),
            GOLDEN45.map(|id| (id, reason))
        );
        let totals = summary.to_json();
        assert_eq!(
            totals
                .get("exact_totals")
                .map(|t| pyjson::dumps(t, true, None))
                .as_deref(),
            Some(r#"{"histories": 4, "pairwise_matches": 1800, "rankable_rows": 360}"#)
        );
        assert!(GOLDEN45
            .iter()
            .all(|id| summary.excluded_from_exact_totals.iter().any(|e| e == id)));
        assert_eq!(summary.derived_view_rows["V-plot-historical-elo"], 661);
        assert_refused(
            summary.require_all(),
            r"^not replayed here: \['H-sotarena-golden45-benchmark-valid', 'H-sotarena-golden45-published'\]$",
        );
        committed()
            .require_all()
            .expect("every history replays with the built-in decoder");
    }
}

mod independent_fit {
    use super::*;

    #[test]
    fn positive_controls() {
        let strong = board(&[
            ("a", "strong", "1.0"),
            ("a", "weak", "2.0"),
            ("b", "strong", "2.0"),
            ("b", "weak", "3.0"),
        ]);
        assert!(strong.ratings["strong"] > strong.ratings["weak"]);
        let mean = strong.ratings.values().sum::<f64>() / 2.0;
        assert_eq!(py::round(mean - 1500.0, 9), 0.0);
        let tied = board(&[
            ("a", "left", "1.0"),
            ("a", "right", &pyjson::float_repr(1.0 + 1e-13)),
        ]);
        assert_eq!(
            tied.ratings
                .iter()
                .map(|(k, v)| (k.as_str(), *v))
                .collect::<Vec<_>>(),
            [("left", 1500.0), ("right", 1500.0)]
        );
        let alone = board(&[("a", "solo", "1.0"), ("b", "x", "1.0"), ("b", "y", "2.0")]);
        assert!(!alone.ratings.contains_key("solo"));
        let flipped = board_with(&[("a", "strong", "1.0"), ("a", "weak", "2.0")], false);
        assert!(flipped.ratings["weak"] > flipped.ratings["strong"]);
    }

    #[test]
    fn equal_counts_and_totals_are_exchangeable_despite_different_records() {
        let result = board(&[
            ("d1", "a", "1"),
            ("d1", "c", "2"),
            ("d1", "b", "3"),
            ("d1", "x", "4"),
            ("d2", "b", "1"),
            ("d2", "x", "2"),
            ("d2", "a", "3"),
            ("d2", "c", "4"),
        ]);
        assert_eq!(result.exchangeable, groups(&[&["a", "b"], &["c", "x"]]));
        assert_ne!(result.pairs.get("a", "c"), result.pairs.get("b", "c"));
        assert_eq!(py::round(result.ratings["a"] - result.ratings["b"], 9), 0.0);
    }

    #[test]
    fn disconnected_groups_are_reported() {
        let result = board(&[
            ("d1", "a", "1"),
            ("d1", "b", "2"),
            ("d2", "c", "1"),
            ("d2", "d", "2"),
        ]);
        assert_eq!(result.components, [2, 2]);
    }

    #[test]
    fn identical_score_vectors_are_reported_as_possible_aliases() {
        let result = board(&[
            ("d1", "a", "1"),
            ("d1", "a2", "1"),
            ("d1", "b", "2"),
            ("d2", "a", "3"),
            ("d2", "a2", "3"),
            ("d2", "b", "1"),
        ]);
        assert_eq!(result.identical_scores, groups(&[&["a", "a2"]]));
    }

    #[test]
    fn duplicate_delivery_cannot_add_or_replace_a_match() {
        let rows = [("d1", "a", "1.0"), ("d1", "b", "2.0")];
        for duplicate in [("d1", "a", "1.0"), ("d1", "a", "3.0")] {
            let mut all = rows.to_vec();
            all.push(duplicate);
            assert_refused(parse_scores(&scores_csv(&all), "t"), "duplicate delivery");
        }
        let missing = scores_csv(&rows) + &"t,d1,c,same-cut,rmse,,missing,True,k,r,,,\n".repeat(2);
        assert_eq!(parse_scores(&missing, "t").unwrap().len(), 4); // unrankable rows add no match
                                                                   // Addition: the message names the cell as Python's tuple repr did.
        let mut all = rows.to_vec();
        all.push(("d1", "a", "3.0"));
        assert_refused(
            parse_scores(&scores_csv(&all), "t"),
            r"^scores\.csv line 4: duplicate delivery of \('d1', 'a'\) cannot add or replace a match$",
        );
    }

    #[test]
    fn replay_is_deterministic() {
        let document = record();
        let spec = document
            .get("histories")
            .and_then(Value::as_list)
            .unwrap()
            .iter()
            .find(|h| h.get("id") == Some(&Value::from("H-same-cut-v2-screen")))
            .and_then(|h| h.get("replay"))
            .unwrap()
            .clone();
        let first = elo_history::replay(&repo(), &spec, Zstd::Builtin).unwrap();
        let second = elo_history::replay(&repo(), &spec, Zstd::Builtin).unwrap();
        assert_eq!(first.ladder.csv, second.ladder.csv);
        let bits = |l: &Ladder| {
            l.ratings
                .iter()
                .map(|(m, r)| (m.clone(), r.to_bits()))
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(&first.ladder), bits(&second.ladder));
    }
}

mod substitution {
    use super::*;

    const V1: &str = "H-same-cut-v1-deep";

    fn flip_winner(data: Vec<u8>) -> Vec<u8> {
        let text = String::from_utf8(data).unwrap();
        let mut rows = pycsv::reader(&text, Newlines::Universal).unwrap();
        let best = (1..rows.len())
            .filter(|&i| rows[i][6] == "ok")
            .fold(None::<usize>, |best, i| match best {
                Some(b) if py::float(&rows[b][5]).unwrap() <= py::float(&rows[i][5]).unwrap() => {
                    Some(b)
                }
                _ => Some(i),
            })
            .unwrap();
        rows[best][5] = pyjson::float_repr(py::float(&rows[best][5]).unwrap() * 10.0);
        let mut out = String::new();
        for row in &rows {
            pycsv::write_row(&mut out, row);
        }
        out.into_bytes()
    }

    fn swap_rows(data: Vec<u8>) -> Vec<u8> {
        let text = String::from_utf8(data).unwrap();
        let mut lines: Vec<String> = text.split_inclusive('\n').map(str::to_owned).collect();
        let first: Vec<String> = lines[1].splitn(3, ',').map(str::to_owned).collect();
        let second: Vec<String> = lines[2].splitn(3, ',').map(str::to_owned).collect();
        lines[1] = [first[0].clone(), first[1].clone(), second[2].clone()].join(",");
        lines[2] = [second[0].clone(), second[1].clone(), first[2].clone()].join(",");
        lines.concat().into_bytes()
    }

    fn snapshot_elo(data: Vec<u8>) -> Vec<u8> {
        let mut snapshot = pyjson::parse_bytes(&data).unwrap();
        set(nth(at(&mut snapshot, "rows"), 0), "elo", "1700.0".into());
        pyjson::dumps(&snapshot, false, None).into_bytes()
    }

    type Mutate = fn(&mut Copy);

    #[test]
    fn positive_control_and_substitutions() {
        let mutations: [(&str, Mutate, &str); 7] = [
            (
                "unchanged bytes, wrong digest",
                |c| {
                    set(
                        c.item(V1, &["replay", "inputs", "scores"]),
                        "sha256",
                        "0".repeat(64).into(),
                    )
                },
                "digest mismatch",
            ),
            (
                "re-pinned substituted scores",
                |c| {
                    let root = c.root.clone();
                    rewrite(
                        &root,
                        c.item(V1, &["replay", "inputs", "scores"]),
                        flip_winner,
                    );
                },
                "differs from replay|in the committed ladder but",
            ),
            (
                "re-pinned duplicate delivery",
                |c| {
                    let root = c.root.clone();
                    rewrite(
                        &root,
                        c.item(V1, &["replay", "inputs", "scores"]),
                        |mut d| {
                            let first = d.split_inclusive(|b| *b == b'\n').nth(1).unwrap().to_vec();
                            d.extend(first);
                            d
                        },
                    );
                },
                "duplicate delivery",
            ),
            (
                "re-pinned reordered leaderboard",
                |c| {
                    let root = c.root.clone();
                    rewrite(
                        &root,
                        c.item(V1, &["replay", "outputs", "leaderboard"]),
                        swap_rows,
                    );
                },
                "in the committed ladder but",
            ),
            (
                "re-pinned snapshot rating",
                |c| {
                    let root = c.root.clone();
                    rewrite(
                        &root,
                        c.item(V1, &["replay", "outputs", "snapshot"]),
                        snapshot_elo,
                    );
                },
                "differs from replay",
            ),
            (
                "input other than the track declares",
                |c| {
                    let stranger = c
                        .item(
                            "H-same-cut-v2-deep",
                            &["replay", "inputs", "scores", "path"],
                        )
                        .clone();
                    set(
                        c.item(V1, &["replay", "inputs", "scores"]),
                        "path",
                        stranger,
                    );
                },
                "not the track's declared input",
            ),
            (
                "path traversal",
                |c| {
                    let retained = c.item("H-legacy-round1-internal-elo", &["retained"]);
                    set(nth(retained, 0), "path", "../elo_trajectory.csv".into());
                },
                "path traversal",
            ),
        ];
        Copy::new(fast(record()))
            .verify()
            .expect("positive control");
        for (label, mutate, reason) in mutations {
            let mut copy = Copy::new(fast(record()));
            mutate(&mut copy);
            let result = copy.verify().map(|_| ());
            assert!(result.is_err(), "{label}: accepted");
            assert_refused(result, reason);
        }
    }
}

mod disposition {
    use super::*;

    #[test]
    fn claims_are_refused_outside_reproducible_histories_and_historical_consumers() {
        let document = record();
        let granted = authorize(&document, "H-same-cut-v2-deep", "ranking", CONSUMER).unwrap();
        assert_eq!(
            granted.get("consumer"),
            Some(&Value::from("historical-ranking-only"))
        );
        let refusals = [
            (
                ("H-same-cut-v2-deep", "ranking", "hwm"),
                "hwm consumers are refused",
            ),
            (
                ("H-same-cut-v2-deep", "rating", "teaching"),
                "teaching consumers are refused",
            ),
            (
                ("H-same-cut-v2-deep", "rating", "transfer"),
                "transfer consumers are refused",
            ),
            (
                ("H-same-cut-v2-deep", "ranking", "promotion"),
                "promotion consumers are refused",
            ),
            (
                ("H-same-cut-v2-deep", "continuity", CONSUMER),
                "continuity across ladders",
            ),
            (
                ("H-legacy-round1-internal-elo", "ranking", CONSUMER),
                "irrecoverable history supports no ranking",
            ),
            (
                ("H-sotarena-golden42-v1", "rating", CONSUMER),
                "irrecoverable history supports no rating",
            ),
            (
                ("H-campaign-redteam-bradley-terry", "ranking", CONSUMER),
                "scorer-parity-only history supports",
            ),
            (("H-unknown", "ranking", CONSUMER), "unknown history"),
        ];
        for ((history, claim, consumer), reason) in refusals {
            assert_refused(authorize(&document, history, claim, consumer), reason);
        }
        // Addition: an unknown consumer or claim is refused by name.
        assert_refused(
            authorize(&document, "H-same-cut-v2-deep", "ranking", "press"),
            "^H-same-cut-v2-deep: unknown consumer 'press'$",
        );
        assert_refused(
            authorize(&document, "H-same-cut-v2-deep", "trend", CONSUMER),
            "^H-same-cut-v2-deep: unknown claim 'trend'$",
        );
    }

    type Mutate = fn(&mut Value);

    #[test]
    fn record_cannot_upgrade_a_disposition() {
        const LEGACY: &str = "H-legacy-round1-internal-elo";
        const PARITY: &str = "H-campaign-redteam-bradley-terry";
        let mutations: [(&str, Mutate, &str); 11] = [
            (
                "irrecoverable relabelled reproducible",
                |d| {
                    set(
                        history(d, LEGACY),
                        "disposition",
                        elo_history::REPRODUCIBLE.into(),
                    )
                },
                "needs a replay",
            ),
            (
                "irrecoverable with a replay",
                |d| {
                    let replay = history(d, "H-same-cut-v1-deep")
                        .get("replay")
                        .unwrap()
                        .clone();
                    set(history(d, LEGACY), "replay", replay);
                },
                "carries no replay",
            ),
            (
                "irrecoverable with ratings",
                |d| set(history(d, LEGACY), "ratings", json(r#"{"gen30": 1119.2}"#)),
                "cannot carry ratings",
            ),
            (
                "irrecoverable without what is missing",
                |d| set(history(d, LEGACY), "missing", Value::List(vec![])),
                "must name what is missing",
            ),
            (
                "parity without its test",
                |d| drop(pop(history(d, PARITY), "parity")),
                "parity test",
            ),
            (
                "parity with a ranking",
                |d| set(history(d, PARITY), "ranking", strs(&["a"])),
                "cannot carry",
            ),
            (
                "reproducible without receipts",
                |d| drop(pop(history(d, "H-same-cut-v1-deep"), "receipts")),
                "receipts",
            ),
            (
                "unknown disposition",
                |d| set(history(d, LEGACY), "disposition", "recovered".into()),
                "disposition",
            ),
            (
                "duplicate history",
                |d| {
                    let copy = history(d, LEGACY).clone();
                    at(d, "histories").as_list_mut().unwrap().push(copy);
                },
                "duplicate history",
            ),
            (
                "no limitations",
                |d| set(history(d, LEGACY), "limitations", Value::List(vec![])),
                "incomplete history",
            ),
            (
                "altered retained observation",
                |d| {
                    set(
                        nth(at(history(d, LEGACY), "retained"), 0),
                        "sha256",
                        "1".repeat(64).into(),
                    )
                },
                "digest mismatch",
            ),
        ];
        for (label, mutate, reason) in mutations {
            let mut document = fast(record());
            mutate(&mut document);
            let result = Copy::new(document).verify().map(|_| ());
            assert!(result.is_err(), "{label}: accepted");
            assert_refused(result, reason);
        }
    }
}

mod lineage {
    use super::*;

    const A100: &str = "L-lineage-b-a100-recovery";
    const AUDIT: &str = "L-native-audit-108-blocks";

    #[test]
    fn continuity_only_through_retained_edges() {
        let document = record();
        let repo = repo();
        let path = |granted: Value| {
            granted
                .get("path")
                .and_then(Value::as_list)
                .unwrap()
                .clone()
        };
        let granted =
            path(continuity(&repo, &document, A100, "6000", "60000", Zstd::Builtin).unwrap());
        assert_eq!(granted[0], Value::from("60000"));
        assert_eq!(granted.last(), Some(&Value::from("6000")));
        assert_eq!(granted.len(), 19); // one block-initialization edge per block
        let audit = path(
            continuity(
                &repo,
                &document,
                AUDIT,
                "runs/continuous-learning-v2-20260909/cycle-000000/train-2",
                "runs/continuous-learning-v2-20260909/cycle-000001/train-2",
                Zstd::Builtin,
            )
            .unwrap(),
        );
        assert_eq!(audit.len(), 2);
        let refusals = [
            ((A100, "3000", "60000"), "3000 is not a retained checkpoint"),
            ((A100, "4000", "4477"), "missing lineage edge above 4477"),
            (
                (A100, "6000", "70000"),
                "70000 is not a retained checkpoint",
            ),
            (
                ("L-lineage-a-xbabe2", "a", "b"),
                "irrecoverable lineage supports no continuity",
            ),
            (("L-unknown", "a", "b"), "unknown lineage"),
        ];
        for ((lineage, ancestor, descendant), reason) in refusals {
            assert_refused(
                continuity(
                    &repo,
                    &document,
                    lineage,
                    ancestor,
                    descendant,
                    Zstd::Builtin,
                ),
                reason,
            );
        }
        for lineage in [A100, AUDIT, "L-lineage-a-xbabe2"] {
            assert_refused(
                exposure(&document, lineage),
                "exposure edges are not retained",
            );
        }
    }

    type Mutate = fn(&mut Value);

    #[test]
    fn recorded_lineage_facts_must_match_the_table() {
        let mutations: [(&str, Mutate, &str); 4] = [
            (
                "inflated retained edges",
                |d| {
                    let expected = at(lineage(d, A100), "expected");
                    set(expected, "retained_parent_edges", Value::Int(78));
                    set(expected, "unretained_parent_edges", Value::Int(0));
                },
                "recorded counts differ",
            ),
            (
                "hidden missing edge",
                |d| set(lineage(d, A100), "missing_edges", Value::List(vec![])),
                "recorded missing edges differ",
            ),
            (
                "table without columns",
                |d| drop(pop(lineage(d, AUDIT), "columns")),
                "lineage columns",
            ),
            (
                "irrecoverable table-less lineage relabelled",
                |d| {
                    set(
                        lineage(d, "L-lineage-a-xbabe2"),
                        "disposition",
                        elo_history::REPRODUCIBLE.into(),
                    )
                },
                "is irrecoverable and names what is missing",
            ),
        ];
        for (label, mutate, reason) in mutations {
            let mut document = fast(record());
            mutate(&mut document);
            let result = Copy::new(document).verify().map(|_| ());
            assert!(result.is_err(), "{label}: accepted");
            assert_refused(result, reason);
        }
    }

    #[test]
    fn inconsistent_or_ambiguous_tables_are_refused() {
        let header = "state,parent,step,parent_step,cycle,corpus\n";
        let tables = [
            (
                "inconsistent node",
                format!("{header}s1,p,1,0,0,c\ns1,q,1,0,0,c\n"),
                "recorded inconsistently",
            ),
            (
                "duplicate label",
                format!("{header}s1,p,1,0,0,c\ns2,s1,1,1,0,c\n"),
                "labels are not unique",
            ),
        ];
        for (label, text, reason) in tables {
            let directory = scratch();
            std::fs::write(directory.path().join("table.csv"), &text).unwrap();
            let entry = json(&format!(
                r#"{{"id": "L-test", "columns": {{"state": "state", "parent": "parent", "label": "step",
                    "parent_label": "parent_step", "block": "cycle", "corpus": "corpus"}},
                    "table": {{"path": "table.csv", "sha256": "{}"}}}}"#,
                py::sha256_hex(text.as_bytes())
            ));
            let result = lineage_graph(directory.path(), &entry, Zstd::Builtin).map(|_| ());
            assert!(result.is_err(), "{label}: accepted");
            assert_refused(result, reason);
        }
    }
}

mod derived {
    use super::*;

    fn plot(change: fn(&mut Vec<Dict>)) -> impl FnOnce(Vec<u8>) -> Vec<u8> {
        move |data| {
            let text = String::from_utf8(data).unwrap();
            let mut rows = pycsv::dict_reader(&text, Newlines::Universal).unwrap().rows;
            change(&mut rows);
            let fieldnames: Vec<String> = rows[0].keys().map(str::to_owned).collect();
            let mut out = String::new();
            pycsv::write_row(&mut out, &fieldnames);
            for row in &rows {
                let cells: Vec<String> = fieldnames
                    .iter()
                    .map(|f| row.get(f).map_or_else(String::new, pycsv::cell))
                    .collect();
                pycsv::write_row(&mut out, &cells);
            }
            out.into_bytes()
        }
    }

    #[test]
    fn plot_rows_must_repeat_reproducible_snapshots() {
        let document = record();
        let mut histories: HashMap<Key, &Value> = HashMap::new();
        for h in document.get("histories").and_then(Value::as_list).unwrap() {
            histories.insert(Key::of(h.get("id").unwrap()).unwrap(), h);
        }
        let view = document
            .get("derived_views")
            .and_then(Value::as_list)
            .unwrap()[0]
            .clone();
        let copy = Copy::new(document.clone());
        assert_eq!(
            derived_view(&copy.root, &view, &histories, Zstd::Builtin).unwrap(),
            661
        ); // positive control
        #[allow(clippy::ptr_arg)] // one signature for every plot change; `drop_row` pops
        fn raise_rating(rows: &mut Vec<Dict>) {
            rows[0].insert("elo", "1999.9".into());
        }
        fn drop_row(rows: &mut Vec<Dict>) {
            rows.pop();
        }
        type Change = fn(&mut Vec<Dict>);
        type Case = (
            &'static str,
            Option<Change>,
            Option<&'static str>,
            &'static str,
        );
        let cases: [Case; 3] = [
            (
                "raised rating",
                Some(raise_rating as Change),
                None,
                "differs from its snapshot",
            ),
            (
                "dropped row",
                Some(drop_row as Change),
                None,
                "only partly plotted",
            ),
            (
                "irrecoverable source",
                None,
                Some("H-legacy-round1-internal-elo"),
                "not backed by a reproducible history",
            ),
        ];
        for (label, change, source, reason) in cases {
            let mut changed = view.clone();
            let copy = Copy::new(document.clone());
            if let Some(change) = change {
                rewrite(&copy.root, at(&mut changed, "table"), plot(change));
            }
            if let Some(source) = source {
                set(
                    at(&mut changed, "snapshots"),
                    "same-cut-v1-deep",
                    source.into(),
                );
            }
            let result = derived_view(&copy.root, &changed, &histories, Zstd::Builtin);
            assert!(result.is_err(), "{label}: accepted");
            assert_refused(result, reason);
        }
    }
}

mod decoding {
    use super::*;

    #[test]
    fn compressed_inputs_are_decoded_and_bound_to_their_decoded_digest() {
        // Addition: `.gz` and `.zst` inputs decode in process and must match `decoded_sha256`;
        // without a zstd decoder a `.zst` input is unavailable, never trusted.
        let directory = scratch();
        let root = directory.path();
        let data = b"track_id,dataset_id\nt,d1\n".repeat(50);
        let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        gz.write_all(&data).unwrap();
        std::fs::write(root.join("x.csv.gz"), gz.finish().unwrap()).unwrap();
        let zst = ruzstd::encoding::compress_to_vec(
            &data[..],
            ruzstd::encoding::CompressionLevel::Fastest,
        );
        std::fs::write(root.join("x.csv.zst"), zst).unwrap();
        let item = |name: &str, decoded: &str| {
            json(&format!(
                r#"{{"path": "{name}", "sha256": "{}", "decoded_sha256": "{decoded}"}}"#,
                digest(&root.join(name))
            ))
        };
        let decoded = py::sha256_hex(&data);
        for name in ["x.csv.gz", "x.csv.zst"] {
            let loaded = elo_history::load(root, &item(name, &decoded), Zstd::Builtin).unwrap();
            assert_eq!(loaded, data, "{name}");
            assert_refused(
                elo_history::load(root, &item(name, &"0".repeat(64)), Zstd::Builtin),
                &format!("^{}: decoded digest mismatch$", regex::escape(name)),
            );
        }
        match elo_history::load(root, &item("x.csv.zst", &decoded), Zstd::Unavailable) {
            Err(jope_records::Error::Unavailable(reason)) => {
                assert_eq!(reason, "x.csv.zst: no zstd decoder on this host")
            }
            other => panic!("expected unavailable, got {other:?}"),
        }
        // A plain input needs no decoded digest.
        std::fs::write(root.join("plain.csv"), &data).unwrap();
        let plain = json(&format!(
            r#"{{"path": "plain.csv", "sha256": "{}"}}"#,
            digest(&root.join("plain.csv"))
        ));
        assert_eq!(
            elo_history::load(root, &plain, Zstd::Unavailable).unwrap(),
            data
        );
        assert_refused(
            elo_history::load(
                root,
                &json(r#"{"path": "plain.csv", "sha256": null}"#),
                Zstd::Builtin,
            ),
            r#"^\{'path': 'plain\.csv', 'sha256': None\}: input reference$"#,
        );
    }
}

mod sensitivity {
    use super::*;

    fn run(history: &str) -> String {
        let repo = repo();
        let value = elo_history::sensitivity_of(
            &repo.join(elo_history::RECORD),
            &repo,
            history,
            Zstd::Builtin,
        )
        .expect("sensitivity");
        pyjson::dumps(&value, true, None)
    }

    #[test]
    fn same_cut_sensitivity_matches_the_python_replay() {
        // Addition: the Python replay's `--sensitivity` outputs recorded in the slice-2 baseline.
        assert_eq!(
            run("H-same-cut-v1-screen"),
            r#"{"extended_converged": true, "extended_iterations": 123, "historical_cap": 250, "historical_converged": true, "history": "H-same-cut-v1-screen", "max_abs_rating_shift": 0.0, "positions_changed": 0, "top10_unchanged": true}"#
        );
        let repo = repo();
        assert_refused(
            elo_history::sensitivity_of(
                &repo.join(elo_history::RECORD),
                &repo,
                "H-legacy-round1-internal-elo",
                Zstd::Builtin,
            ),
            "^H-legacy-round1-internal-elo: no replayable history$",
        );
    }

    #[test]
    fn golden45_sensitivity_matches_the_python_replay_bit_for_bit() {
        // Addition: 2,000 synchronous Newton iterations must land on the same double as the
        // Python replay (0.30718335953474707), so the accumulation order is exactly preserved.
        assert_eq!(
            run("H-sotarena-golden45-published"),
            r#"{"extended_converged": false, "extended_iterations": 2000, "historical_cap": 250, "historical_converged": false, "history": "H-sotarena-golden45-published", "max_abs_rating_shift": 0.30718335953474707, "positions_changed": 6, "top10_unchanged": true}"#
        );
    }
}
