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
