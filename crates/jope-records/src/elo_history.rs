//! `elo-history`: replay retained historical Elo ladders and keep unrecoverable history open
//! (W12-T04).
//!
//! Every history in `elo-history/histories.json` carries exactly one disposition:
//!
//! - `reproducible-with-receipts`: the original scores, track and registries are retained with
//!   digests. This module re-implements the documented procedure independently (aggregated pair
//!   statistics, not the per-match port in `crates/jope-arena/src/elo.rs`, which it never calls)
//!   and must reproduce the committed leaderboard bytes and snapshot rows from those inputs alone.
//! - `scorer-parity-only`: a current scorer agrees with a reference implementation on fixtures.
//!   That is implementation parity; no historical match graph was recovered.
//! - `irrecoverable`: original matches, participants or settings are not retained. The retained
//!   observations may be quoted as attributable history; ranking, rating and continuity claims
//!   are refused.
//!
//! Every rating is historical-ranking-only and is refused to teaching, HWM, transfer and
//! promotion consumers. Ratings are relative to one method universe, so continuity across
//! histories is refused. Lineage tables are recomputed from their retained bytes: a parent edge
//! whose parent is not retained blocks continuity across it, and exposure totals that need
//! unretained corpus membership are refused. A digest authenticates retained bytes, not the truth
//! of the producer that wrote them.
//!
//! Numerics follow the replay this port replaces operation for operation: IEEE-754 doubles in
//! the same accumulation order, `exp`/`log` from the platform C library, CPython's `math.fsum`
//! for the centring mean, and Python's `round`/`repr` spellings in every written cell.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::io::Read as _;
use std::path::Path;

use crate::py::{self, Key, PosixPath};
use crate::pycsv::{self, Newlines};
use crate::pyjson::{self, Dict, Value};
use crate::{require, Error, Result};

pub const RECORD: &str = "docs/internal/core-ip/elo-history/histories.json";
pub const SCHEMA: &str = "jope.core-ip-elo-histories/v1";
pub const SNAPSHOT_SCHEMA: &str = "jope.arena-elo-snapshot/v2";
pub const CORRECTIONS_SCHEMA: &str = "jope.arena-protocol-corrections/v1";
pub const REPRODUCIBLE: &str = "reproducible-with-receipts";
pub const DISPOSITIONS: [&str; 3] = [REPRODUCIBLE, "scorer-parity-only", "irrecoverable"];
pub const CONSUMER: &str = "historical-ranking-only";
pub const REFUSED_CONSUMERS: [&str; 4] = ["teaching", "hwm", "transfer", "promotion"];
pub const CLAIMS: [&str; 3] = ["ranking", "rating", "continuity"];
pub const SCORE_COLUMNS: [&str; 13] = [
    "track_id",
    "dataset_id",
    "method_id",
    "protocol",
    "metric",
    "value",
    "status",
    "benchmark_valid",
    "source_kind",
    "source_revision",
    "source_sha256",
    "source_uri",
    "notes",
];
pub const LEADERBOARD_COLUMNS: [&str; 16] = [
    "elo_rank",
    "full_coverage_rank",
    "method_id",
    "method",
    "elo",
    "average_rank",
    "dataset_wins",
    "datasets_scored",
    "datasets_total",
    "coverage_pct",
    "full_coverage",
    "pending_cells",
    "coverage_status",
    "protocols",
    "benchmark_valid",
    "scope",
];
/// The documented procedure: monograph chapter 7, "How the retained Elo scale is constructed".
pub const CENTER: f64 = 1500.0;
pub const ITERATIONS: usize = 250;
pub const REGULARIZATION: f64 = 1e-4;
pub const TOLERANCE: f64 = 1e-8;
pub const CLAMP: f64 = 40.0;
pub const TIE: f64 = 1e-12;
const COMMON: [&str; 6] = [
    "id",
    "disposition",
    "subject",
    "scope",
    "anchors",
    "limitations",
];
const POSITION: [&str; 2] = ["elo_rank", "full_coverage_rank"];
const LINEAGE_COLUMNS: [&str; 6] = [
    "state",
    "parent",
    "label",
    "parent_label",
    "block",
    "corpus",
];

/// The scope sentence of each ladder kind.
pub fn scope_of(ladder: &str) -> Option<&'static str> {
    match ladder {
        "published" => Some("official unified benchmark comparison"),
        "benchmark-valid" => Some("benchmark-valid rows only; protocol corrections applied"),
        _ => None,
    }
}

/// `math.log(10) / 400`, computed at run time by the platform C library exactly as CPython's
/// `math.log` computes it (a compile-time constant could round differently).
fn scale() -> f64 {
    std::hint::black_box(10.0_f64).ln() / 400.0
}

/// Whether `.zst` inputs can be decoded. The decoder is built in (`ruzstd`, pure Rust), so
/// [`Zstd::Builtin`] is always available; [`Zstd::Unavailable`] keeps the "not replayed here"
/// path of a host without a decoder reachable and tested.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Zstd {
    Builtin,
    Unavailable,
}

fn shape(message: impl Into<String>) -> Error {
    Error::Shape(message.into())
}

/// `value[key]` on a dict, failing like Python's `KeyError`/`TypeError`.
fn index<'a>(value: &'a Value, key: &str) -> Result<&'a Value> {
    value.get(key).ok_or_else(|| {
        shape(format!(
            "KeyError: {} in {}",
            pyjson::str_repr(key),
            abbreviated(value)
        ))
    })
}

fn abbreviated(value: &Value) -> String {
    let text = value.repr();
    if text.chars().count() > 80 {
        format!("{}…", text.chars().take(80).collect::<String>())
    } else {
        text
    }
}

fn index_str<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    index(value, key)?
        .as_str()
        .ok_or_else(|| shape(format!("{key} is not a string")))
}

fn list_of<'a>(value: &'a Value, what: &str) -> Result<&'a Vec<Value>> {
    value
        .as_list()
        .ok_or_else(|| shape(format!("{what} is not a list")))
}

fn key_of(value: &Value) -> Result<Key> {
    Key::of(value).ok_or_else(|| shape(format!("unhashable key {}", abbreviated(value))))
}

fn decode_utf8(bytes: Vec<u8>, what: &str) -> Result<String> {
    String::from_utf8(bytes)
        .map_err(|e| Error::parse(Path::new(what), format!("'utf-8' codec can't decode: {e}")))
}

fn json_bytes(bytes: &[u8], what: &str) -> Result<Value> {
    pyjson::parse_bytes(bytes).map_err(|message| Error::parse(Path::new(what), message))
}

fn resolve(root: &Path, reference: &str) -> Result<std::path::PathBuf> {
    let relative = PosixPath::parse(reference);
    require!(
        !relative.is_absolute() && !relative.has_parent_reference(),
        "{reference}: path traversal"
    );
    Ok(relative.join_onto(root))
}

fn gunzip(raw: &[u8]) -> std::result::Result<Vec<u8>, String> {
    let mut data = Vec::new();
    flate2::read::MultiGzDecoder::new(raw)
        .read_to_end(&mut data)
        .map_err(|e| e.to_string())?;
    Ok(data)
}

/// Every frame of a zstd stream (skippable frames skipped, content checksums verified), as
/// `zstd -d -c` decodes it.
fn unzstd(raw: &[u8]) -> std::result::Result<Vec<u8>, String> {
    use ruzstd::decoding::errors::{FrameDecoderError, ReadFrameHeaderError};
    use ruzstd::decoding::{BlockDecodingStrategy, FrameDecoder};
    let mut decoder = FrameDecoder::new();
    let mut input = raw;
    let mut out = Vec::new();
    while !input.is_empty() {
        match decoder.reset(&mut input) {
            Ok(()) => {}
            Err(FrameDecoderError::ReadFrameHeaderError(ReadFrameHeaderError::SkipFrame {
                length,
                ..
            })) => {
                input = input
                    .get(length as usize..)
                    .ok_or("truncated skippable frame")?;
                continue;
            }
            Err(error) => return Err(error.to_string()),
        }
        while !decoder.is_finished() {
            decoder
                .decode_blocks(&mut input, BlockDecodingStrategy::UptoBytes(1 << 20))
                .map_err(|e| e.to_string())?;
            decoder
                .collect_to_writer(&mut out)
                .map_err(|e| e.to_string())?;
        }
        decoder
            .collect_to_writer(&mut out)
            .map_err(|e| e.to_string())?;
        if let Some(expected) = decoder.get_checksum_from_data() {
            if decoder.get_calculated_checksum() != Some(expected) {
                return Err("zstd content checksum mismatch".into());
            }
        }
    }
    Ok(out)
}

/// Digest-check one retained file and return its (decoded) bytes.
pub fn load(root: &Path, item: &Value, zstd: Zstd) -> Result<Vec<u8>> {
    let reference = item.get("path").and_then(Value::as_str);
    require!(
        item.as_dict().is_some() && reference.is_some() && py::is_hex(item.get("sha256"), 64),
        "{}: input reference",
        item.repr()
    );
    let reference = reference.expect("checked above");
    let path = resolve(root, reference)?;
    require!(
        path.is_file() && !py::is_symlink(&path),
        "{reference}: retained input missing"
    );
    let raw = py::read_bytes(&path)?;
    require!(
        item.get("sha256") == Some(&Value::from(py::sha256_hex(&raw))),
        "{reference}: digest mismatch"
    );
    let data = match PosixPath::parse(reference).suffix() {
        ".gz" => gunzip(&raw).map_err(|e| shape(format!("{reference}: {e}")))?,
        ".zst" => match zstd {
            Zstd::Unavailable => {
                return Err(Error::Unavailable(format!(
                    "{reference}: no zstd decoder on this host"
                )))
            }
            Zstd::Builtin => unzstd(&raw).map_err(|e| shape(format!("{reference}: {e}")))?,
        },
        _ => return Ok(raw),
    };
    require!(
        item.get("decoded_sha256") == Some(&Value::from(py::sha256_hex(&data))),
        "{reference}: decoded digest mismatch"
    );
    Ok(data)
}

/// One `scores.csv` row, cell by cell in [`SCORE_COLUMNS`] order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScoreRow(pub Vec<String>);

impl ScoreRow {
    fn cell(&self, column: &str) -> &str {
        let at = SCORE_COLUMNS
            .iter()
            .position(|c| *c == column)
            .expect("a scores.csv column");
        &self.0[at]
    }

    fn set(&mut self, column: &str, value: &str) {
        let at = SCORE_COLUMNS
            .iter()
            .position(|c| *c == column)
            .expect("a scores.csv column");
        self.0[at] = value.to_owned();
    }

    pub fn rankable(&self) -> bool {
        self.cell("status") == "ok" && !self.cell("value").is_empty()
    }
}

fn float_cell(text: &str) -> Result<f64> {
    py::float(text).ok_or_else(|| {
        Error::Invalid(format!(
            "could not convert string to float: {}",
            pyjson::str_repr(text)
        ))
    })
}

/// Rows of a `scores.csv`. A second delivery of a rankable cell is refused, never merged.
pub fn parse_scores(text: &str, track_id: &str) -> Result<Vec<ScoreRow>> {
    let mut records = pycsv::reader(text, Newlines::Universal)?.into_iter();
    let header = records.next().unwrap_or_default();
    require!(header == SCORE_COLUMNS, "scores.csv columns");
    let mut rows = Vec::new();
    let mut cells: HashSet<(String, String)> = HashSet::new();
    for (offset, values) in records.enumerate() {
        let line = offset + 2;
        require!(
            values.len() == SCORE_COLUMNS.len(),
            "scores.csv line {line}: column count"
        );
        let row = ScoreRow(values);
        if row.rankable() {
            require!(
                row.cell("track_id") == track_id,
                "scores.csv line {line}: row of another track"
            );
            let cell = (
                row.cell("dataset_id").to_owned(),
                row.cell("method_id").to_owned(),
            );
            require!(
                !cells.contains(&cell),
                "scores.csv line {line}: duplicate delivery of {} cannot add or replace a match",
                py::tuple_repr(&[Value::from(cell.0.as_str()), Value::from(cell.1.as_str())])
            );
            cells.insert(cell);
            require!(
                float_cell(row.cell("value"))?.is_finite(),
                "scores.csv line {line}: non-finite value"
            );
            require!(
                matches!(row.cell("benchmark_valid"), "True" | "False"),
                "scores.csv line {line}: validity"
            );
        }
        rows.push(row);
    }
    Ok(rows)
}

/// Demote the rows a digest-bound corrections file selects; corrections never promote.
/// Returns the applied corrections (`{"id", "rows"}`) and how many rows were demoted.
pub fn apply_corrections(
    rows: &mut [ScoreRow],
    document: &Value,
    track_id: &str,
    scores_sha256: &str,
) -> Result<(Vec<Value>, usize)> {
    require!(
        document.get("schema") == Some(&Value::from(CORRECTIONS_SCHEMA)),
        "corrections schema"
    );
    require!(
        document.get("track_id") == Some(&Value::from(track_id))
            && document.get("scores_sha256") == Some(&Value::from(scores_sha256)),
        "corrections bind another track or scores.csv"
    );
    let mut applied = Vec::new();
    let mut corrected = 0;
    let corrections = py::iterate(index(document, "corrections")?)
        .ok_or_else(|| shape("corrections is not iterable"))?;
    for correction in &corrections {
        // `is False`: only the JSON literal `false`, never `0`.
        require!(
            matches!(correction.get("benchmark_valid"), Some(Value::Bool(false)))
                && correction.get("evidence").is_some_and(Value::truthy),
            "{}: a correction only demotes and cites evidence",
            correction
                .get("id")
                .map_or_else(|| "None".to_owned(), Value::py_str)
        );
        let mut matched: usize = 0;
        for row in rows.iter_mut() {
            if !row.rankable() {
                continue;
            }
            if &Value::from(row.cell("method_id")) != index(correction, "method_id")? {
                continue;
            }
            let differs = ["source_kind", "source_revision"].iter().any(|field| {
                correction.get(field).is_some_and(|wanted| {
                    !wanted.is_null() && &Value::from(row.cell(field)) != wanted
                })
            });
            if differs {
                continue;
            }
            if let Some(datasets) = correction.get("dataset_ids").filter(|d| d.truthy()) {
                let member = py::contains(datasets, &Value::from(row.cell("dataset_id")))
                    .ok_or_else(|| shape("dataset_ids is not a container"))?;
                if !member {
                    continue;
                }
            }
            matched += 1;
            if row.cell("benchmark_valid") == "True" {
                row.set("benchmark_valid", "False");
                corrected += 1;
            }
        }
        let id = index(correction, "id")?;
        require!(matched > 0, "{}: selects no rankable row", id.py_str());
        let mut entry = Dict::new();
        entry.insert("id", id.clone());
        entry.insert("rows", matched.into());
        applied.push(Value::Dict(entry));
    }
    Ok((applied, corrected))
}

fn outcome(left: f64, right: f64, minimize: bool) -> f64 {
    let largest = left.abs().max(right.abs()).max(1.0);
    if (left - right).abs() <= TIE * largest {
        return 0.5;
    }
    if (left < right) == minimize {
        1.0
    } else {
        0.0
    }
}

/// Match count and outcome sum for each unordered method pair (the likelihood's sufficient
/// statistics), in the order Python's dict first saw each pair.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Pairs {
    order: Vec<(String, String)>,
    stats: HashMap<(String, String), (u64, f64)>,
}

impl Pairs {
    fn add(&mut self, a: &str, b: &str, outcome: f64) {
        let key = (a.to_owned(), b.to_owned());
        match self.stats.get_mut(&key) {
            Some((count, total)) => {
                *count += 1;
                *total += outcome;
            }
            None => {
                self.order.push(key.clone());
                self.stats.insert(key, (1, 0.0 + outcome));
            }
        }
    }

    /// `(count, total)` of the pair `(a, b)` with `a < b`.
    pub fn get(&self, a: &str, b: &str) -> Option<(u64, f64)> {
        self.stats.get(&(a.to_owned(), b.to_owned())).copied()
    }

    pub fn len(&self) -> usize {
        self.order.len()
    }

    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    /// Pairs in first-seen order.
    pub fn iter(&self) -> impl Iterator<Item = (&(String, String), (u64, f64))> {
        self.order.iter().map(|key| (key, self.stats[key]))
    }
}

/// One equal-weight match per dataset and pair of scored methods.
pub fn pair_statistics(
    by_dataset: &BTreeMap<String, BTreeMap<String, f64>>,
    minimize: bool,
) -> Pairs {
    let mut pairs = Pairs::default();
    for scores in by_dataset.values() {
        let scores: Vec<(&String, &f64)> = scores.iter().collect();
        for (i, (left, left_value)) in scores.iter().enumerate() {
            for (right, right_value) in &scores[i + 1..] {
                pairs.add(left, right, outcome(**left_value, **right_value, minimize));
            }
        }
    }
    pairs
}

/// Python's two-argument `min`/`max` (the first argument wins ties).
fn py_min(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

fn py_max(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}

/// Regularized Bradley–Terry on the Elo scale by synchronous diagonal Newton steps. Returns the
/// ratings by method, the iterations run and whether the tolerance was reached within `cap`.
pub fn fit(pairs: &Pairs, cap: usize) -> (BTreeMap<String, f64>, usize, bool) {
    let scale = scale();
    let methods: BTreeSet<&str> = pairs
        .order
        .iter()
        .flat_map(|(a, b)| [a.as_str(), b.as_str()])
        .collect();
    let methods: Vec<&str> = methods.into_iter().collect();
    let position: HashMap<&str, usize> = methods.iter().enumerate().map(|(i, m)| (*m, i)).collect();
    let mut keys: Vec<&(String, String)> = pairs.order.iter().collect();
    keys.sort();
    let table: Vec<(usize, usize, f64, f64)> = keys
        .into_iter()
        .map(|key| {
            let (count, total) = pairs.stats[key];
            (
                position[key.0.as_str()],
                position[key.1.as_str()],
                count as f64,
                total,
            )
        })
        .collect();
    let mut ratings = vec![CENTER; methods.len()];
    let mut iterations = 0;
    let mut converged = false;
    for _ in 0..cap {
        let mut gradient = vec![0.0_f64; methods.len()];
        let mut information = vec![0.0_f64; methods.len()];
        for &(a, b, count, total) in &table {
            let delta = py_max(-700.0, py_min(700.0, scale * (ratings[a] - ratings[b])));
            let probability = 1.0 / (1.0 + (-delta).exp());
            let residual = total - count * probability;
            let curvature = count * probability * (1.0 - probability);
            gradient[a] += residual;
            gradient[b] -= residual;
            information[a] += curvature;
            information[b] += curvature;
        }
        let steps: Vec<f64> = (0..methods.len())
            .map(|m| {
                let newton = (scale * gradient[m] - REGULARIZATION * (ratings[m] - CENTER))
                    / (scale * scale * information[m] + REGULARIZATION);
                py_max(-CLAMP, py_min(CLAMP, newton))
            })
            .collect();
        for (rating, step) in ratings.iter_mut().zip(&steps) {
            *rating += step;
        }
        iterations += 1;
        let largest = steps
            .iter()
            .fold(0.0_f64, |m, s| if s.abs() > m { s.abs() } else { m });
        if largest < TOLERANCE {
            converged = true;
            break;
        }
    }
    let shift = if ratings.is_empty() {
        0.0
    } else {
        CENTER - py::fsum(ratings.iter().copied()).expect("finite ratings") / ratings.len() as f64
    };
    let fitted = methods
        .iter()
        .zip(&ratings)
        .map(|(m, r)| ((*m).to_owned(), r + shift))
        .collect();
    (fitted, iterations, converged)
}

/// Methods with equal match counts against every other method and equal total score.
///
/// Bradley–Terry ratings depend on the data only through match counts and each method's total
/// score, so such methods have equal ratings in exact arithmetic even when their per-opponent
/// records differ. Their published order is decided by floating-point accumulation order and the
/// name tie-break, never by a comparison.
pub fn exchangeable_groups(pairs: &Pairs) -> Vec<Vec<String>> {
    let mut counts: HashMap<&str, HashMap<&str, u64>> = HashMap::new();
    let mut scores: Vec<(&str, f64)> = Vec::new();
    let mut score_at: HashMap<&str, usize> = HashMap::new();
    for ((a, b), (count, total)) in pairs.iter() {
        counts.entry(a).or_default().insert(b, count);
        counts.entry(b).or_default().insert(a, count);
        for (method, amount) in [(a.as_str(), None), (b.as_str(), Some(count))] {
            let slot = *score_at.entry(method).or_insert_with(|| {
                scores.push((method, 0.0));
                scores.len() - 1
            });
            let current = scores[slot].1;
            // scores[a] + total, and (scores[b] + count) - total, as Python evaluated them.
            scores[slot].1 = match amount {
                None => current + total,
                Some(count) => current + count as f64 - total,
            };
        }
    }
    let related = |m: &str, n: &str| {
        let (cm, cn) = (&counts[m], &counts[n]);
        cm.keys()
            .chain(cn.keys())
            .filter(|k| **k != m && **k != n)
            .all(|k| cm.get(k).copied().unwrap_or(0) == cn.get(k).copied().unwrap_or(0))
    };
    let mut distinct: Vec<f64> = scores.iter().map(|(_, s)| *s).collect();
    distinct.sort_by(|a, b| a.partial_cmp(b).expect("finite scores"));
    distinct.dedup_by(|a, b| a == b);
    let mut groups = Vec::new();
    for score in distinct {
        let mut members: Vec<&str> = scores
            .iter()
            .filter(|(_, s)| *s == score)
            .map(|(m, _)| *m)
            .collect();
        members.sort_unstable();
        let mut classes: Vec<Vec<&str>> = Vec::new();
        for method in members {
            match classes
                .iter_mut()
                .find(|class| class.iter().all(|n| related(method, n)))
            {
                Some(home) => home.push(method),
                None => classes.push(vec![method]),
            }
        }
        groups.extend(
            classes
                .into_iter()
                .filter(|c| c.len() > 1)
                .map(|c| c.into_iter().map(str::to_owned).collect::<Vec<String>>()),
        );
    }
    groups.sort();
    groups
}

/// Sizes of the connected components of the comparison graph. Order between components is
/// imposed by the regularizer and centring, not by any comparison.
pub fn components(pairs: &Pairs) -> Vec<usize> {
    let mut parent: HashMap<&str, &str> = HashMap::new();
    let mut order: Vec<&str> = Vec::new();
    fn find<'a>(
        parent: &mut HashMap<&'a str, &'a str>,
        order: &mut Vec<&'a str>,
        node: &'a str,
    ) -> &'a str {
        if !parent.contains_key(node) {
            parent.insert(node, node);
            order.push(node);
        }
        let mut node = node;
        while parent[node] != node {
            let grand = parent[parent[node]];
            parent.insert(node, grand);
            node = grand;
        }
        node
    }
    for (a, b) in &pairs.order {
        let rb = find(&mut parent, &mut order, b);
        let ra = find(&mut parent, &mut order, a);
        parent.insert(ra, rb);
    }
    let mut sizes: HashMap<&str, usize> = HashMap::new();
    for node in order.clone() {
        let root = find(&mut parent, &mut order, node);
        *sizes.entry(root).or_default() += 1;
    }
    let mut sizes: Vec<usize> = sizes.into_values().collect();
    sizes.sort_unstable();
    sizes
}

/// Settings of one ladder.
pub struct LadderOptions<'a> {
    pub ladder_kind: &'a str,
    pub minimize: bool,
    pub names: &'a HashMap<Key, Value>,
    pub pending: &'a HashMap<Key, i64>,
    pub total_datasets: usize,
}

/// A replayed ladder.
#[derive(Clone, Debug)]
pub struct Ladder {
    /// Leaderboard rows (every cell as Python wrote it; ranks and counts are integers).
    pub rows: Vec<Value>,
    /// The leaderboard CSV bytes as `csv.writer(lineterminator="\n")` wrote them.
    pub csv: String,
    pub ratings: BTreeMap<String, f64>,
    pub iterations: usize,
    pub converged: bool,
    pub rankable_rows: usize,
    pub excluded_rows: usize,
    pub datasets_scored: usize,
    pub matches: u64,
    pub components: Vec<usize>,
    pub exchangeable: Vec<Vec<String>>,
    pub identical_scores: Vec<Vec<String>>,
    pub pairs: Pairs,
}

/// The leaderboard of `rows` (every cell spelled as Python's `round`/`repr` write it).
pub fn ladder(rows: &[ScoreRow], options: &LadderOptions<'_>) -> Result<Ladder> {
    let scope = scope_of(options.ladder_kind).ok_or_else(|| shape("ladder kind"))?;
    let mut by_dataset: BTreeMap<String, BTreeMap<String, f64>> = BTreeMap::new();
    let mut protocols: HashMap<String, BTreeSet<String>> = HashMap::new();
    let mut valid: HashMap<String, bool> = HashMap::new();
    let mut rankable_rows = 0;
    let mut excluded = 0;
    for row in rows {
        if !row.rankable() {
            continue;
        }
        let is_valid = row.cell("benchmark_valid") == "True";
        if options.ladder_kind == "benchmark-valid" && !is_valid {
            excluded += 1;
            continue;
        }
        let method = row.cell("method_id").to_owned();
        by_dataset
            .entry(row.cell("dataset_id").to_owned())
            .or_default()
            .insert(method.clone(), float_cell(row.cell("value"))?);
        protocols
            .entry(method.clone())
            .or_default()
            .insert(row.cell("protocol").to_owned());
        let before = valid.get(&method).copied().unwrap_or(true);
        valid.insert(method, before && is_valid);
        rankable_rows += 1;
    }
    let pairs = pair_statistics(&by_dataset, options.minimize);
    let (ratings, iterations, converged) = fit(&pairs, ITERATIONS);
    let mut ranks: HashMap<&str, Vec<u64>> = HashMap::new();
    let mut wins: HashMap<&str, i64> = HashMap::new();
    for scores in by_dataset.values() {
        let mut ordered: Vec<(&String, f64)> = scores.iter().map(|(m, v)| (m, *v)).collect();
        ordered.sort_by(|a, b| {
            let (ka, kb) = if options.minimize {
                (a.1, b.1)
            } else {
                (-a.1, -b.1)
            };
            ka.partial_cmp(&kb)
                .expect("finite values")
                .then_with(|| a.0.cmp(b.0))
        });
        for (rank, (method, _)) in ordered.iter().enumerate() {
            ranks
                .entry(method.as_str())
                .or_default()
                .push(rank as u64 + 1);
        }
        *wins.entry(ordered[0].0.as_str()).or_default() += 1;
    }
    let mut ranked: Vec<(&String, f64)> = ratings.iter().map(|(m, r)| (m, *r)).collect();
    ranked.sort_by(|a, b| {
        (-a.1)
            .partial_cmp(&-b.1)
            .expect("finite ratings")
            .then_with(|| a.0.cmp(b.0))
    });
    let total = options.total_datasets;
    let mut board = Vec::new();
    let mut full = 0;
    for (position, (method, rating)) in ranked.iter().enumerate() {
        let method_ranks = ranks.get(method.as_str()).map_or(&[][..], Vec::as_slice);
        let scored = method_ranks.len();
        let covered = scored == total;
        full += usize::from(covered);
        let cells = options.pending.get(&Key::str(method)).copied().unwrap_or(0);
        if total == 0 {
            return Err(shape(
                "float division by zero: the track declares no datasets",
            ));
        }
        let average = method_ranks.iter().sum::<u64>() as f64 / scored as f64;
        let coverage = 100.0 * scored as f64 / total as f64;
        let name = options
            .names
            .get(&Key::str(method))
            .cloned()
            .unwrap_or_else(|| Value::from(method.as_str()));
        let mut status = format!("{scored}/{total}");
        if cells != 0 {
            status.push_str(&format!(" · {cells} pending"));
        }
        let protocol_list: Vec<&str> = protocols
            .get(method.as_str())
            .map(|p| p.iter().map(String::as_str).collect())
            .unwrap_or_default();
        let mut row = Dict::new();
        row.insert("elo_rank", (position + 1).into());
        row.insert(
            "full_coverage_rank",
            if covered {
                full.to_string()
            } else {
                String::new()
            }
            .into(),
        );
        row.insert("method_id", method.as_str().into());
        row.insert("method", name);
        row.insert("elo", pyjson::float_repr(py::round(*rating, 1)).into());
        row.insert(
            "average_rank",
            pyjson::float_repr(py::round(average, 3)).into(),
        );
        row.insert(
            "dataset_wins",
            Value::Int(wins.get(method.as_str()).copied().unwrap_or(0)),
        );
        row.insert("datasets_scored", scored.into());
        row.insert("datasets_total", total.into());
        row.insert(
            "coverage_pct",
            pyjson::float_repr(py::round(coverage, 1)).into(),
        );
        row.insert(
            "full_coverage",
            if covered { "True" } else { "False" }.into(),
        );
        row.insert("pending_cells", Value::Int(cells));
        row.insert("coverage_status", status.into());
        row.insert("protocols", protocol_list.join("|").into());
        row.insert(
            "benchmark_valid",
            if valid.get(method.as_str()).copied().unwrap_or(false) {
                "True"
            } else {
                "False"
            }
            .into(),
        );
        row.insert("scope", scope.into());
        board.push(Value::Dict(row));
    }
    let mut csv = String::new();
    pycsv::write_row(&mut csv, &LEADERBOARD_COLUMNS);
    for row in &board {
        let cells: Vec<String> = LEADERBOARD_COLUMNS
            .iter()
            .map(|c| pycsv::cell(row.get(c).expect("every column")))
            .collect();
        pycsv::write_row(&mut csv, &cells);
    }
    // Methods with identical score vectors (possible aliases of one another).
    let mut aliases: BTreeMap<Vec<(String, u64)>, Vec<String>> = BTreeMap::new();
    for method in ratings.keys() {
        let signature: Vec<(String, u64)> = by_dataset
            .iter()
            .filter_map(|(dataset, scores)| {
                scores
                    .get(method)
                    .map(|v| (dataset.clone(), (v + 0.0).to_bits()))
            })
            .collect();
        aliases.entry(signature).or_default().push(method.clone());
    }
    let mut identical: Vec<Vec<String>> = aliases.into_values().filter(|g| g.len() > 1).collect();
    identical.sort();
    Ok(Ladder {
        rows: board,
        csv,
        ratings: ratings.clone(),
        iterations,
        converged,
        rankable_rows,
        excluded_rows: excluded,
        datasets_scored: by_dataset.len(),
        matches: pairs.iter().map(|(_, (count, _))| count).sum(),
        components: components(&pairs),
        exchangeable: exchangeable_groups(&pairs),
        identical_scores: identical,
        pairs,
    })
}

/// A replayed history: its ladder, the digest of its decoded scores and its corrections.
#[derive(Clone, Debug)]
pub struct Replay {
    pub ladder: Ladder,
    pub scores_sha256: String,
    pub corrections: Value,
}

fn py_len(value: &Value) -> Result<usize> {
    match value {
        Value::List(list) => Ok(list.len()),
        Value::Dict(dict) => Ok(dict.len()),
        Value::Str(s) => Ok(s.chars().count()),
        other => Err(shape(format!(
            "object of type {} has no len()",
            abbreviated(other)
        ))),
    }
}

/// Recompute one ladder from its pinned inputs; the track's own declarations bind the paths.
pub fn replay(root: &Path, spec: &Value, zstd: Zstd) -> Result<Replay> {
    let base = PosixPath::parse(index_str(spec, "root")?);
    let inputs = index(spec, "inputs")?
        .as_dict()
        .ok_or_else(|| shape("replay inputs is not an object"))?;
    let track_id_value = index(spec, "track_id")?;
    let track_id = track_id_value
        .as_str()
        .ok_or_else(|| shape("track_id is not a string"))?;
    let input = |role: &str| -> Result<&Value> {
        inputs
            .get(role)
            .ok_or_else(|| shape(format!("KeyError: {}", pyjson::str_repr(role))))
    };
    let track_path = base
        .join("tracks")
        .join(track_id)
        .join("track.json")
        .to_string();
    require!(
        index(input("track")?, "path")? == &Value::from(track_path),
        "replay track path"
    );
    let track = json_bytes(&load(root, input("track")?, zstd)?, "track.json")?;
    require!(
        track.get("schema") == Some(&Value::from("sotarena.track.v1"))
            && track.get("id") == Some(track_id_value),
        "track identity"
    );
    let direction = track.get("metric_direction");
    require!(
        matches!(
            direction.and_then(Value::as_str),
            Some("minimize" | "maximize")
        ),
        "metric direction"
    );
    let default_methods = Value::from("registry/methods.json");
    let mut declared: Vec<(&str, &Value)> = vec![
        ("datasets", index(&track, "dataset_registry")?),
        (
            "methods",
            track.get("method_registry").unwrap_or(&default_methods),
        ),
        ("scores", index(&track, "scores")?),
    ];
    if let Some(ledger) = track.get("resolution_ledger").filter(|l| l.truthy()) {
        declared.push(("ledger", ledger));
    }
    let given: BTreeSet<&str> = inputs.keys().filter(|k| *k != "track").collect();
    let wanted: BTreeSet<&str> = declared.iter().map(|(role, _)| *role).collect();
    require!(given == wanted, "replay inputs differ from the track's");
    for (role, relative) in &declared {
        let path = index(input(role)?, "path")?;
        let relative = relative
            .as_str()
            .ok_or_else(|| shape(format!("the track's {role} is not a path")))?;
        let joined = base.join(relative).to_string();
        let allowed = [
            joined.clone(),
            format!("{joined}.zst"),
            format!("{joined}.gz"),
        ];
        require!(
            allowed.iter().any(|a| path == &Value::from(a.as_str())),
            "replay {role} path is not the track's declared input"
        );
    }
    let datasets = json_bytes(&load(root, input("datasets")?, zstd)?, "datasets registry")?;
    let total_datasets = py_len(index(&datasets, "datasets")?)?;
    let methods = json_bytes(&load(root, input("methods")?, zstd)?, "methods registry")?;
    let mut names: HashMap<Key, Value> = HashMap::new();
    for method in py::iterate(index(&methods, "methods")?).ok_or_else(|| shape("methods"))? {
        let id = index(&method, "id")?;
        let name = method.get("name").unwrap_or(id).clone();
        names.insert(key_of(id)?, name);
    }
    let scores_bytes = load(root, input("scores")?, zstd)?;
    let scores_sha256 = py::sha256_hex(&scores_bytes);
    let mut rows = parse_scores(&decode_utf8(scores_bytes, "scores.csv")?, track_id)?;
    let mut pending: HashMap<Key, i64> = HashMap::new();
    if inputs.contains_key("ledger") {
        let text = decode_utf8(load(root, input("ledger")?, zstd)?, "resolution ledger")?;
        for line in py::splitlines(&text) {
            if py::strip(line).is_empty() {
                continue;
            }
            let cell = pyjson::parse_str(line)
                .map_err(|message| Error::parse(Path::new("resolution ledger"), message))?;
            require!(
                matches!(cell.get("resolved"), Some(Value::Bool(_))),
                "ledger cell without `resolved`"
            );
            if cell.get("resolved") == Some(&Value::Bool(false)) {
                *pending
                    .entry(key_of(index(&cell, "method_id")?)?)
                    .or_default() += 1;
            }
        }
    }
    let mut corrections = Value::Null;
    if let Some(reference) = spec.get("corrections").filter(|c| c.truthy()) {
        let raw = load(root, reference, zstd)?;
        let document = json_bytes(&raw, "corrections")?;
        let (applied, corrected) =
            apply_corrections(&mut rows, &document, track_id, &scores_sha256)?;
        let mut summary = Dict::new();
        summary.insert(
            "file",
            PosixPath::parse(index_str(reference, "path")?)
                .name()
                .into(),
        );
        summary.insert("sha256", py::sha256_hex(&raw).into());
        summary.insert("applied", Value::List(applied));
        summary.insert("rows_corrected", corrected.into());
        corrections = Value::Dict(summary);
    }
    let ladder_kind = index(spec, "ladder")?
        .as_str()
        .filter(|kind| scope_of(kind).is_some());
    require!(ladder_kind.is_some(), "ladder kind");
    let ladder = ladder(
        &rows,
        &LadderOptions {
            ladder_kind: ladder_kind.expect("checked above"),
            minimize: direction.and_then(Value::as_str) == Some("minimize"),
            names: &names,
            pending: &pending,
            total_datasets,
        },
    )?;
    Ok(Replay {
        ladder,
        scores_sha256,
        corrections,
    })
}

/// Every cell must agree, except that methods of one exchangeable group may trade places.
///
/// Returns the groups whose committed order differs from the replay's; such an order was set by
/// the original implementation's floating-point accumulation, not by data.
pub fn compare(
    label: &str,
    committed: &[Value],
    replayed: &[Value],
    groups: &[Vec<String>],
) -> Result<Vec<Vec<String>>> {
    require!(
        committed.len() == replayed.len(),
        "{label}: ladder length differs from replay"
    );
    let mut group_of: HashMap<&str, usize> = HashMap::new();
    for (i, group) in groups.iter().enumerate() {
        for method in group {
            group_of.insert(method, i);
        }
    }
    let mut traded: BTreeSet<usize> = BTreeSet::new();
    for (offset, (theirs, ours)) in committed.iter().zip(replayed).enumerate() {
        let position = offset + 1;
        let mut same = true;
        for field in POSITION {
            if index(theirs, field)? != index(ours, field)? {
                same = false;
                break;
            }
        }
        require!(same, "{label}: rank {position} position cells");
        let (their_id, our_id) = (index(theirs, "method_id")?, index(ours, "method_id")?);
        if their_id != our_id {
            let group = our_id.as_str().and_then(|m| group_of.get(m));
            key_of(their_id)?;
            let their_group = their_id.as_str().and_then(|m| group_of.get(m));
            require!(
                group.is_some() && group == their_group,
                "{label}: rank {position} is {} in the committed ladder but {} in the independent \
                 replay",
                their_id.py_str(),
                our_id.py_str()
            );
            traded.insert(*group.expect("checked above"));
        }
    }
    let strip = |rows: &[Value]| -> Result<HashMap<Key, Value>> {
        let mut out = HashMap::new();
        for row in rows {
            let dict = row
                .as_dict()
                .ok_or_else(|| shape(format!("{label}: a ladder row is not an object")))?;
            let kept: Dict = dict
                .iter()
                .filter(|(k, _)| !POSITION.contains(k))
                .map(|(k, v)| (k, v.clone()))
                .collect();
            out.insert(key_of(index(row, "method_id")?)?, Value::Dict(kept));
        }
        Ok(out)
    };
    let (theirs, ours) = (strip(committed)?, strip(replayed)?);
    require!(
        theirs.len() == ours.len() && theirs.keys().all(|k| ours.contains_key(k)),
        "{label}: rated methods differ from replay"
    );
    let mut methods: Vec<&Key> = ours.keys().collect();
    methods.sort();
    for method in methods {
        let shown = method
            .as_str()
            .map_or_else(|| format!("{method:?}"), str::to_owned);
        require!(
            theirs[method] == ours[method],
            "{label}: {shown} differs from replay"
        );
    }
    Ok(traded.into_iter().map(|i| groups[i].clone()).collect())
}

/// What a replayed history reports.
#[derive(Clone, Debug, PartialEq)]
pub struct ReplaySummary {
    pub byte_identical: bool,
    pub exchangeable_groups: Vec<Vec<String>>,
    pub order_differs_from_replay: Vec<Vec<String>>,
    pub identical_score_groups: Vec<Vec<String>>,
    pub methods: usize,
    pub rankable_rows: usize,
    pub matches: u64,
    pub iterations: usize,
    pub converged: bool,
    pub components: Vec<usize>,
}

fn groups_value(groups: &[Vec<String>]) -> Value {
    Value::List(
        groups
            .iter()
            .map(|g| Value::List(g.iter().map(|m| Value::from(m.as_str())).collect()))
            .collect(),
    )
}

impl ReplaySummary {
    pub fn cross_component_order(&self) -> &'static str {
        if self.components.len() > 1 {
            "imposed-by-regularization"
        } else {
            "data-connected"
        }
    }

    pub fn to_json(&self) -> Value {
        let mut dict = Dict::new();
        dict.insert("byte_identical", self.byte_identical.into());
        dict.insert(
            "exchangeable_groups",
            groups_value(&self.exchangeable_groups),
        );
        dict.insert(
            "order_differs_from_replay",
            groups_value(&self.order_differs_from_replay),
        );
        dict.insert(
            "identical_score_groups",
            groups_value(&self.identical_score_groups),
        );
        dict.insert("methods", self.methods.into());
        dict.insert("rankable_rows", self.rankable_rows.into());
        dict.insert("matches", Value::Int(self.matches as i64));
        dict.insert("iterations", self.iterations.into());
        dict.insert("converged", self.converged.into());
        dict.insert(
            "components",
            Value::List(self.components.iter().map(|c| Value::from(*c)).collect()),
        );
        dict.insert("cross_component_order", self.cross_component_order().into());
        dict.insert("consumer", CONSUMER.into());
        Value::Dict(dict)
    }
}

/// Replay a reproducible history and compare it with its committed leaderboard and snapshot.
pub fn check_replay(root: &Path, history: &Value, zstd: Zstd) -> Result<ReplaySummary> {
    let spec = index(history, "replay")?;
    let hid = index(history, "id")?.py_str();
    let result = replay(root, spec, zstd)?;
    let outputs = index(spec, "outputs")?;
    let committed = load(root, index(outputs, "leaderboard")?, zstd)?;
    let committed_text = decode_utf8(committed.clone(), "leaderboard.csv")?;
    let committed_rows: Vec<Value> = pycsv::dict_reader(&committed_text, Newlines::Universal)?
        .rows
        .into_iter()
        .map(Value::Dict)
        .collect();
    let as_text: Vec<Value> = result
        .ladder
        .rows
        .iter()
        .map(|row| {
            let dict = row.as_dict().expect("ladder rows are objects");
            Value::Dict(
                dict.iter()
                    .map(|(k, v)| (k, Value::Str(v.py_str())))
                    .collect(),
            )
        })
        .collect();
    let groups = &result.ladder.exchangeable;
    let traded = compare(
        &format!("{hid} leaderboard"),
        &committed_rows,
        &as_text,
        groups,
    )?;
    let snapshot = json_bytes(&load(root, index(outputs, "snapshot")?, zstd)?, "snapshot")?;
    let expected: [(&str, Value); 10] = [
        ("schema", SNAPSHOT_SCHEMA.into()),
        ("track_id", index(spec, "track_id")?.clone()),
        ("ladder", index(spec, "ladder")?.clone()),
        ("scores_sha256", result.scores_sha256.as_str().into()),
        ("corrections", result.corrections.clone()),
        ("excluded_rows", result.ladder.excluded_rows.into()),
        ("leaderboard_sha256", py::sha256_hex(&committed).into()),
        ("rankable_rows", result.ladder.rankable_rows.into()),
        ("datasets_scored", result.ladder.datasets_scored.into()),
        ("methods", result.ladder.rows.len().into()),
    ];
    for (field, value) in &expected {
        require!(
            snapshot.get(field) == Some(value),
            "{hid}: snapshot {field} differs from replay"
        );
    }
    let empty = Value::List(Vec::new());
    let snapshot_rows = list_of(snapshot.get("rows").unwrap_or(&empty), "snapshot rows")?;
    let snapshot_traded = compare(
        &format!("{hid} snapshot"),
        snapshot_rows,
        &result.ladder.rows,
        groups,
    )?;
    require!(
        snapshot_traded == traded,
        "{hid}: snapshot and leaderboard orders differ"
    );
    Ok(ReplaySummary {
        byte_identical: result.ladder.csv.as_bytes() == committed.as_slice(),
        exchangeable_groups: result.ladder.exchangeable.clone(),
        order_differs_from_replay: traded,
        identical_score_groups: result.ladder.identical_scores.clone(),
        methods: result.ladder.rows.len(),
        rankable_rows: result.ladder.rankable_rows,
        matches: result.ladder.matches,
        iterations: result.ladder.iterations,
        converged: result.ladder.converged,
        components: result.ladder.components.clone(),
    })
}

fn by_rating(ratings: &BTreeMap<String, f64>) -> Vec<&str> {
    let mut ordered: Vec<(&String, f64)> = ratings.iter().map(|(m, r)| (m, *r)).collect();
    ordered.sort_by(|a, b| {
        (-a.1)
            .partial_cmp(&-b.1)
            .expect("finite ratings")
            .then_with(|| a.0.cmp(b.0))
    });
    ordered.into_iter().map(|(m, _)| m.as_str()).collect()
}

/// How far the retained ratings (the iterate at the historical cap) sit from a longer fit.
pub fn sensitivity(root: &Path, history: &Value, cap: usize, zstd: Zstd) -> Result<Value> {
    let result = replay(root, index(history, "replay")?, zstd)?;
    let (longer, iterations, converged) = fit(&result.ladder.pairs, cap);
    let retained = by_rating(&result.ladder.ratings);
    let extended = by_rating(&longer);
    require!(!longer.is_empty(), "max() arg is an empty sequence");
    let shift = longer
        .iter()
        .map(|(m, r)| (r - result.ladder.ratings[m]).abs())
        .fold(f64::NEG_INFINITY, |a, b| if b > a { b } else { a });
    let mut dict = Dict::new();
    dict.insert("history", index(history, "id")?.clone());
    dict.insert("historical_cap", ITERATIONS.into());
    dict.insert("historical_converged", result.ladder.converged.into());
    dict.insert("extended_iterations", iterations.into());
    dict.insert("extended_converged", converged.into());
    dict.insert("max_abs_rating_shift", Value::Float(shift));
    dict.insert(
        "positions_changed",
        retained
            .iter()
            .zip(&extended)
            .filter(|(a, b)| a != b)
            .count()
            .into(),
    );
    dict.insert(
        "top10_unchanged",
        (retained[..retained.len().min(10)] == extended[..extended.len().min(10)]).into(),
    );
    Ok(Value::Dict(dict))
}

fn truthy_field(value: &Value, key: &str) -> bool {
    value.get(key).is_some_and(Value::truthy)
}

/// A history's disposition rules and retained-file digests.
pub fn check_history(
    root: &Path,
    history: &Value,
    seen: &mut HashSet<Key>,
    zstd: Zstd,
) -> Result<()> {
    let complete = history.as_dict().is_some() && COMMON.iter().all(|f| truthy_field(history, f));
    require!(
        complete,
        "{}: incomplete history",
        if history.as_dict().is_some() {
            history
                .get("id")
                .map_or_else(|| "None".to_owned(), Value::py_str)
        } else {
            history.py_str()
        }
    );
    let id = index(history, "id")?;
    let hid = id.py_str();
    let key = key_of(id)?;
    require!(!seen.contains(&key), "{hid}: duplicate history");
    seen.insert(key);
    let disposition = index(history, "disposition")?;
    require!(
        disposition
            .as_str()
            .is_some_and(|d| DISPOSITIONS.contains(&d)),
        "{hid}: disposition"
    );
    let disposition = disposition.as_str().expect("checked above");
    require!(
        index(history, "limitations")?.as_list().is_some(),
        "{hid}: limitations"
    );
    if let Some(retained) = history.get("retained") {
        for item in py::iterate(retained).ok_or_else(|| shape(format!("{hid}: retained")))? {
            load(root, &item, zstd)?;
        }
    }
    if disposition == REPRODUCIBLE {
        require!(
            history.get("replay").and_then(Value::as_dict).is_some(),
            "{hid}: reproducible history needs a replay"
        );
        let receipts = history.get("receipts");
        require!(
            receipts.and_then(Value::as_dict).is_some()
                && receipts.is_some_and(
                    |r| truthy_field(r, "original_execution") && truthy_field(r, "algorithm")
                ),
            "{hid}: original execution and algorithm receipts"
        );
        return Ok(());
    }
    require!(
        history.get("replay").is_none(),
        "{hid}: {disposition} carries no replay"
    );
    require!(
        history
            .get("missing")
            .and_then(Value::as_list)
            .is_some_and(|m| !m.is_empty()),
        "{hid}: {disposition} must name what is missing"
    );
    require!(
        !["ratings", "ranking", "leaderboard"]
            .iter()
            .any(|k| history.get(k).is_some()),
        "{hid}: {disposition} history cannot carry ratings or rankings"
    );
    if disposition == "scorer-parity-only" {
        require!(
            truthy_field(history, "parity"),
            "{hid}: scorer parity needs its parity test"
        );
    }
    Ok(())
}

/// Checkpoint nodes and parent edges recomputed from a retained lineage table.
#[derive(Clone, Debug)]
pub struct LineageGraph {
    /// Nodes (role → cell) in first-seen order, keyed by checkpoint state.
    pub nodes: Vec<(Key, Dict)>,
    node_at: HashMap<Key, usize>,
    labels: HashMap<Key, usize>,
    pub counts: Dict,
    pub missing_edges: Value,
}

impl LineageGraph {
    fn node(&self, state: &Value) -> Option<&Dict> {
        Key::of(state)
            .and_then(|k| self.node_at.get(&k))
            .map(|&i| &self.nodes[i].1)
    }

    fn label(&self, label: &str) -> Option<&Dict> {
        self.labels.get(&Key::str(label)).map(|&i| &self.nodes[i].1)
    }
}

fn prefix12(value: &Value) -> Result<String> {
    value
        .as_str()
        .map(|s| s.chars().take(12).collect())
        .ok_or_else(|| shape(format!("{} is not subscriptable", abbreviated(value))))
}

/// Checkpoint nodes and parent edges from a retained table; one node per checkpoint state.
pub fn lineage_graph(root: &Path, lineage: &Value, zstd: Zstd) -> Result<LineageGraph> {
    let lid = index(lineage, "id")?.py_str();
    let columns = lineage
        .get("columns")
        .cloned()
        .unwrap_or(Value::Dict(Dict::new()));
    let columns_ok = columns.as_dict().is_some_and(|c| {
        c.len() == LINEAGE_COLUMNS.len() && LINEAGE_COLUMNS.iter().all(|k| c.contains_key(k))
    });
    require!(columns_ok, "{lid}: lineage columns");
    let columns = columns.as_dict().expect("checked above").clone();
    let text = decode_utf8(load(root, index(lineage, "table")?, zstd)?, "lineage table")?;
    let mut graph = LineageGraph {
        nodes: Vec::new(),
        node_at: HashMap::new(),
        labels: HashMap::new(),
        counts: Dict::new(),
        missing_edges: Value::Null,
    };
    for row in pycsv::dict_reader(&text, Newlines::Universal)?.rows {
        let mut node = Dict::new();
        for (role, column) in columns.iter() {
            let column = column
                .as_str()
                .ok_or_else(|| shape(format!("{lid}: lineage column names must be strings")))?;
            let cell = row
                .get(column)
                .ok_or_else(|| shape(format!("KeyError: {}", pyjson::str_repr(column))))?;
            node.insert(role, cell.clone());
        }
        let state = node.get("state").expect("columns checked").clone();
        let key = key_of(&state)?;
        match graph.node_at.get(&key) {
            Some(&i) => require!(
                graph.nodes[i].1 == node,
                "{lid}: checkpoint {}… is recorded inconsistently",
                prefix12(&state)?
            ),
            None => {
                graph.node_at.insert(key.clone(), graph.nodes.len());
                graph.nodes.push((key, node));
            }
        }
    }
    for (i, (_, node)) in graph.nodes.iter().enumerate() {
        graph
            .labels
            .insert(key_of(node.get("label").expect("columns checked"))?, i);
    }
    require!(
        graph.labels.len() == graph.nodes.len(),
        "{lid}: checkpoint labels are not unique"
    );
    let mut missing: Vec<((Value, Value), usize)> = Vec::new();
    for (_, node) in &graph.nodes {
        let parent = node.get("parent").expect("columns checked");
        if graph.node(parent).is_none() {
            let key = (
                parent.clone(),
                node.get("parent_label").expect("columns checked").clone(),
            );
            match missing.iter_mut().find(|(k, _)| *k == key) {
                Some(slot) => slot.1 += 1,
                None => missing.push((key, 1)),
            }
        }
    }
    let distinct = |role: &str| -> Result<usize> {
        let keys: Result<HashSet<Key>> = graph
            .nodes
            .iter()
            .map(|(_, n)| key_of(n.get(role).expect("columns checked")))
            .collect();
        Ok(keys?.len())
    };
    let unretained: usize = missing.iter().map(|(_, n)| n).sum();
    let mut counts = Dict::new();
    counts.insert("checkpoints", graph.nodes.len().into());
    counts.insert("blocks", distinct("block")?.into());
    counts.insert("corpus_identities", distinct("corpus")?.into());
    counts.insert(
        "retained_parent_edges",
        (graph.nodes.len() - unretained).into(),
    );
    counts.insert("unretained_parent_edges", unretained.into());
    if missing.len() > 1
        && missing
            .iter()
            .any(|((state, label), _)| state.as_str().is_none() || label.as_str().is_none())
    {
        return Err(shape(format!("{lid}: missing edges cannot be ordered")));
    }
    missing.sort_by(|((a, b), _), ((c, d), _)| {
        (a.as_str(), b.as_str()).cmp(&(c.as_str(), d.as_str()))
    });
    let edges = missing
        .into_iter()
        .map(|((state, label), children)| {
            let mut edge = Dict::new();
            edge.insert("parent_state_sha256", state);
            edge.insert("parent_label", label);
            edge.insert("children", children.into());
            Value::Dict(edge)
        })
        .collect();
    graph.counts = counts;
    graph.missing_edges = Value::List(edges);
    Ok(graph)
}

/// A lineage's disposition rules; its recomputed counts when it has a retained table.
pub fn check_lineage(
    root: &Path,
    lineage: &Value,
    seen: &mut HashSet<Key>,
    zstd: Zstd,
) -> Result<Value> {
    let lid = lineage.get("id");
    let shown = lid.map_or_else(|| "None".to_owned(), Value::py_str);
    let fresh = match lid {
        Some(id) if id.truthy() => !seen.contains(&key_of(id)?),
        _ => false,
    };
    require!(fresh, "{shown}: lineage id");
    seen.insert(key_of(lid.expect("checked above"))?);
    require!(
        lineage
            .get("disposition")
            .and_then(Value::as_str)
            .is_some_and(|d| DISPOSITIONS.contains(&d))
            && truthy_field(lineage, "limitations"),
        "{shown}: lineage disposition and limitations"
    );
    require!(
        matches!(
            lineage.get("exposure_edges").and_then(Value::as_str),
            Some("retained" | "not-retained")
        ),
        "{shown}: exposure edges"
    );
    if lineage.get("table").is_none() {
        require!(
            lineage.get("disposition") == Some(&Value::from("irrecoverable"))
                && truthy_field(lineage, "missing"),
            "{shown}: a lineage without a retained table is irrecoverable and names what is missing"
        );
        return Ok(Value::Null);
    }
    let graph = lineage_graph(root, lineage, zstd)?;
    require!(
        lineage.get("expected") == Some(&Value::Dict(graph.counts.clone())),
        "{shown}: recorded counts differ from the table"
    );
    require!(
        lineage.get("missing_edges") == Some(&graph.missing_edges),
        "{shown}: recorded missing edges differ from the table"
    );
    Ok(Value::Dict(graph.counts))
}

/// `next((e for e in document[collection] if e["id"] == id), None)`.
fn find_by_id<'a>(document: &'a Value, collection: &str, id: &str) -> Result<Option<&'a Value>> {
    let entries = list_of(index(document, collection)?, collection)?;
    for entry in entries {
        if index(entry, "id")? == &Value::from(id) {
            return Ok(Some(entry));
        }
    }
    Ok(None)
}

/// Return the permitted scope of a claim about one history, or refuse it.
pub fn authorize(document: &Value, history_id: &str, claim: &str, consumer: &str) -> Result<Value> {
    require!(
        !REFUSED_CONSUMERS.contains(&consumer),
        "{history_id}: Elo is historical-ranking-only; {consumer} consumers are refused"
    );
    require!(
        consumer == CONSUMER,
        "{history_id}: unknown consumer {}",
        pyjson::str_repr(consumer)
    );
    require!(
        CLAIMS.contains(&claim),
        "{history_id}: unknown claim {}",
        pyjson::str_repr(claim)
    );
    let history = find_by_id(document, "histories", history_id)?;
    require!(history.is_some(), "{history_id}: unknown history");
    let history = history.expect("checked above");
    require!(
        claim != "continuity",
        "{history_id}: a rating is relative to its own method universe; continuity across \
         ladders is not data-supported"
    );
    let disposition = index(history, "disposition")?;
    require!(
        disposition == &Value::from(REPRODUCIBLE),
        "{history_id}: {} history supports no {claim} claim",
        disposition.py_str()
    );
    let mut granted = Dict::new();
    granted.insert("history", history_id.into());
    granted.insert("claim", claim.into());
    granted.insert("consumer", CONSUMER.into());
    granted.insert("scope", index(history, "scope")?.clone());
    Ok(Value::Dict(granted))
}

/// Parameter-ancestry continuity between two retained checkpoints, through retained edges only.
pub fn continuity(
    root: &Path,
    document: &Value,
    lineage_id: &str,
    ancestor: &str,
    descendant: &str,
    zstd: Zstd,
) -> Result<Value> {
    let lineage = find_by_id(document, "lineages", lineage_id)?;
    require!(lineage.is_some(), "{lineage_id}: unknown lineage");
    let lineage = lineage.expect("checked above");
    require!(
        lineage.get("table").is_some(),
        "{lineage_id}: irrecoverable lineage supports no continuity claim"
    );
    let graph = lineage_graph(root, lineage, zstd)?;
    require!(
        graph.label(ancestor).is_some(),
        "{lineage_id}: {ancestor} is not a retained checkpoint"
    );
    let node = graph.label(descendant);
    require!(
        node.is_some(),
        "{lineage_id}: {descendant} is not a retained checkpoint"
    );
    let mut node = node.expect("checked above");
    let field = |node: &Dict, role: &str| node.get(role).expect("columns checked").clone();
    let mut path = vec![Value::from(descendant)];
    let mut visited: HashSet<Key> = HashSet::from([key_of(&field(node, "state"))?]);
    while field(node, "label") != Value::from(ancestor) {
        let parent = graph.node(&field(node, "parent"));
        require!(
            parent.is_some(),
            "{lineage_id}: missing lineage edge above {} (parent {}, {}… not retained)",
            field(node, "label").py_str(),
            field(node, "parent_label").py_str(),
            prefix12(&field(node, "parent"))?
        );
        let parent = parent.expect("checked above");
        let state = key_of(&field(parent, "state"))?;
        require!(
            !visited.contains(&state),
            "{lineage_id}: parent cycle at {}",
            field(parent, "label").py_str()
        );
        visited.insert(state);
        node = parent;
        path.push(field(node, "label"));
    }
    let mut granted = Dict::new();
    granted.insert("lineage", lineage_id.into());
    granted.insert("path", Value::List(path));
    granted.insert(
        "meaning",
        "parameter ancestry through recorded block-initialization edges; each block trains on \
         its own corpus"
            .into(),
    );
    Ok(Value::Dict(granted))
}

/// Exposure totals of a lineage, refused unless its corpus-to-dataset edges are retained.
pub fn exposure(document: &Value, lineage_id: &str) -> Result<Value> {
    let lineage = find_by_id(document, "lineages", lineage_id)?;
    require!(lineage.is_some(), "{lineage_id}: unknown lineage");
    let lineage = lineage.expect("checked above");
    require!(
        index(lineage, "exposure_edges")? == &Value::from("retained"),
        "{lineage_id}: corpus-to-dataset exposure edges are not retained; exposure totals refused"
    );
    Ok(index(lineage, "exposure")?.clone())
}

/// A committed plot table must repeat its snapshots' rows; it adds no history of its own.
/// Returns its row count.
pub fn derived_view(
    root: &Path,
    view: &Value,
    histories: &HashMap<Key, &Value>,
    zstd: Zstd,
) -> Result<usize> {
    let vid = index(view, "id")?.py_str();
    let text = decode_utf8(
        load(root, index(view, "table")?, zstd)?,
        "derived view table",
    )?;
    let rows = pycsv::dict_reader(&text, Newlines::LineFeed)?.rows;
    let mut snapshots: Vec<(String, Value, HashMap<Key, Value>)> = Vec::new();
    let named = index(view, "snapshots")?
        .as_dict()
        .ok_or_else(|| shape(format!("{vid}: snapshots is not an object")))?;
    for (name, history_id) in named.iter() {
        let history = Key::of(history_id).and_then(|k| histories.get(&k)).copied();
        require!(
            history.is_some_and(|h| h.get("disposition") == Some(&Value::from(REPRODUCIBLE))),
            "{vid}: {name} is not backed by a reproducible history"
        );
        let output = index(
            index(index(history.expect("checked above"), "replay")?, "outputs")?,
            "snapshot",
        )?;
        let snapshot = json_bytes(&load(root, output, zstd)?, "snapshot")?;
        let mut board = HashMap::new();
        for row in py::iterate(index(&snapshot, "rows")?).ok_or_else(|| shape("snapshot rows"))? {
            board.insert(key_of(index(&row, "method_id")?)?, row);
        }
        snapshots.push((name.to_owned(), index(output, "sha256")?.clone(), board));
    }
    let cell = |row: &Dict, column: &str| -> Result<Value> {
        row.get(column)
            .cloned()
            .ok_or_else(|| shape(format!("KeyError: {}", pyjson::str_repr(column))))
    };
    for (offset, row) in rows.iter().enumerate() {
        let number = offset + 2;
        let snapshot = cell(row, "snapshot")?;
        let backing = snapshots
            .iter()
            .find(|(name, _, _)| snapshot == Value::from(name.as_str()));
        require!(backing.is_some(), "{vid} line {number}: unbacked snapshot");
        let (_, digest, board) = backing.expect("checked above");
        require!(
            &cell(row, "source_sha256")? == digest,
            "{vid} line {number}: source digest"
        );
        let source = Key::of(&cell(row, "method")?).and_then(|k| board.get(&k));
        let mut same = source.is_some();
        if let Some(source) = source {
            for (k, v) in [
                ("elo_rank", "rank"),
                ("elo", "elo"),
                ("datasets_scored", "datasets_scored"),
                ("datasets_total", "datasets_total"),
                ("coverage_pct", "coverage_pct"),
            ] {
                if Value::from(index(source, k)?.py_str()) != cell(row, v)? {
                    same = false;
                    break;
                }
            }
        }
        require!(same, "{vid} line {number}: differs from its snapshot");
    }
    for (name, _, board) in &snapshots {
        let plotted = rows
            .iter()
            .filter(|row| row.get("snapshot") == Some(&Value::from(name.as_str())))
            .count();
        require!(
            plotted == board.len(),
            "{vid}: a snapshot is only partly plotted"
        );
    }
    Ok(rows.len())
}

/// What `elo-history` reports.
#[derive(Clone, Debug)]
pub struct Summary {
    pub replayed: Vec<String>,
    pub not_replayed_here: BTreeMap<String, String>,
    pub scorer_parity_only: Vec<String>,
    pub irrecoverable: Vec<String>,
    pub excluded_from_exact_totals: Vec<String>,
    pub replays: BTreeMap<String, ReplaySummary>,
    pub lineages: BTreeMap<String, Value>,
    pub derived_view_rows: BTreeMap<String, usize>,
}

impl Summary {
    pub fn exact_rankable_rows(&self) -> usize {
        self.replays.values().map(|r| r.rankable_rows).sum()
    }

    pub fn exact_pairwise_matches(&self) -> u64 {
        self.replays.values().map(|r| r.matches).sum()
    }

    pub fn to_json(&self) -> Value {
        let strs =
            |items: &[String]| Value::List(items.iter().map(|s| Value::from(s.as_str())).collect());
        let mut totals = Dict::new();
        totals.insert("histories", self.replays.len().into());
        totals.insert("rankable_rows", self.exact_rankable_rows().into());
        totals.insert(
            "pairwise_matches",
            Value::Int(self.exact_pairwise_matches() as i64),
        );
        let mut dict = Dict::new();
        dict.insert("replayed", strs(&self.replayed));
        dict.insert(
            "not_replayed_here",
            Value::Dict(
                self.not_replayed_here
                    .iter()
                    .map(|(k, v)| (k.clone(), Value::from(v.as_str())))
                    .collect(),
            ),
        );
        dict.insert("scorer_parity_only", strs(&self.scorer_parity_only));
        dict.insert("irrecoverable", strs(&self.irrecoverable));
        dict.insert("exact_totals", Value::Dict(totals));
        dict.insert(
            "excluded_from_exact_totals",
            strs(&self.excluded_from_exact_totals),
        );
        dict.insert(
            "replays",
            Value::Dict(
                self.replays
                    .iter()
                    .map(|(k, v)| (k.clone(), v.to_json()))
                    .collect(),
            ),
        );
        dict.insert(
            "lineages",
            Value::Dict(
                self.lineages
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect(),
            ),
        );
        dict.insert(
            "derived_view_rows",
            Value::Dict(
                self.derived_view_rows
                    .iter()
                    .map(|(k, v)| (k.clone(), Value::from(*v)))
                    .collect(),
            ),
        );
        dict.insert("consumer", CONSUMER.into());
        dict.insert(
            "refused_consumers",
            Value::List(REFUSED_CONSUMERS.iter().map(|c| Value::from(*c)).collect()),
        );
        Value::Dict(dict)
    }

    /// `--require-all`: fail when a reproducible history was not replayed on this host.
    pub fn require_all(&self) -> Result<()> {
        let missing: Vec<&str> = self.not_replayed_here.keys().map(String::as_str).collect();
        require!(
            missing.is_empty(),
            "not replayed here: {}",
            py::str_list_repr(&missing)
        );
        Ok(())
    }
}

/// Verify the record at `path`, resolving its pinned files under `root`.
pub fn verify(path: &Path, root: &Path, zstd: Zstd) -> Result<Summary> {
    let document = py::load_json(path)?;
    require!(
        document.get("schema") == Some(&Value::from(SCHEMA)),
        "histories schema"
    );
    require!(
        truthy_field(&document, "numerical_policy"),
        "numerical policy"
    );
    let mut seen = HashSet::new();
    let mut replays = BTreeMap::new();
    let mut unavailable = BTreeMap::new();
    let listed = match document.get("histories") {
        None => Vec::new(),
        Some(value) => py::iterate(value).ok_or_else(|| shape("histories is not iterable"))?,
    };
    for history in &listed {
        check_history(root, history, &mut seen, zstd)?;
        if history.get("disposition") == Some(&Value::from(REPRODUCIBLE)) {
            let hid = index(history, "id")?.py_str();
            match check_replay(root, history, zstd) {
                Ok(summary) => {
                    replays.insert(hid, summary);
                }
                Err(Error::Unavailable(reason)) => {
                    unavailable.insert(hid, reason);
                }
                Err(error) => return Err(error),
            }
        }
    }
    let all = list_of(index(&document, "histories")?, "histories")?;
    let mut histories: HashMap<Key, &Value> = HashMap::new();
    for history in all {
        histories.insert(key_of(index(history, "id")?)?, history);
    }
    let mut lineages = BTreeMap::new();
    let mut lineage_seen = HashSet::new();
    if let Some(value) = document.get("lineages") {
        for lineage in py::iterate(value).ok_or_else(|| shape("lineages is not iterable"))? {
            let counts = check_lineage(root, &lineage, &mut lineage_seen, zstd)?;
            lineages.insert(index(&lineage, "id")?.py_str(), counts);
        }
    }
    let mut views = BTreeMap::new();
    if let Some(value) = document.get("derived_views") {
        for view in py::iterate(value).ok_or_else(|| shape("derived_views is not iterable"))? {
            let id = index(&view, "id")?.py_str();
            views.insert(id, derived_view(root, &view, &histories, zstd)?);
        }
    }
    let with = |disposition: &str| -> Vec<String> {
        let mut ids: Vec<String> = all
            .iter()
            .filter(|h| h.get("disposition") == Some(&Value::from(disposition)))
            .filter_map(|h| h.get("id").map(Value::py_str))
            .collect();
        ids.sort();
        ids
    };
    let every: BTreeSet<String> = histories
        .values()
        .filter_map(|h| h.get("id").map(Value::py_str))
        .collect();
    Ok(Summary {
        replayed: replays.keys().cloned().collect(),
        not_replayed_here: unavailable,
        scorer_parity_only: with("scorer-parity-only"),
        irrecoverable: with("irrecoverable"),
        excluded_from_exact_totals: every
            .into_iter()
            .filter(|id| !replays.contains_key(id))
            .collect(),
        replays,
        lineages,
        derived_view_rows: views,
    })
}

/// `--sensitivity HISTORY`: the iteration-cap sensitivity of one replayable history.
pub fn sensitivity_of(record: &Path, root: &Path, history_id: &str, zstd: Zstd) -> Result<Value> {
    let document = py::load_json(record)?;
    let chosen = find_by_id(&document, "histories", history_id)?;
    require!(
        chosen.is_some_and(|h| h.get("replay").is_some()),
        "{history_id}: no replayable history"
    );
    sensitivity(root, chosen.expect("checked above"), 2000, zstd)
}
