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

mod committed_record;

mod independent_fit;

mod substitution;

mod disposition;

mod lineage;

mod derived;

mod decoding;

mod sensitivity;
