use crate::retention::parquet::{self, Cell, ColumnSpec, ColumnType, Table};
use anyhow::{bail, ensure, Context, Result};
use std::collections::{BTreeMap, BTreeSet};

/// Schema identifier for distilled values.
pub const DISTILLED_SCHEMA: &str = "gp_distilled/v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
/// `SplitRole` is an enum for the split role contract in this module: gpdistilled/v1: the permanent (T0) table that keeps every GP result's training signal after its raw bodies are packed and archived (strategy §8.3).
pub enum SplitRole {
    /// The `Train` case of `SplitRole`.
    Train,
    /// The `Val263` case of `SplitRole`.
    Val263,
    /// The `Sealed` case of `SplitRole`.
    Sealed,
    /// The `TestExposed` case of `SplitRole`.
    TestExposed,
    /// The `None` case of `SplitRole`.
    None,
    /// The `Unknown` case of `SplitRole`.
    Unknown,
}

impl SplitRole {
    /// Returns the stable string representation of this value.
    pub fn as_str(self) -> &'static str {
        match self {
            SplitRole::Train => "train",
            SplitRole::Val263 => "val263",
            SplitRole::Sealed => "sealed",
            SplitRole::TestExposed => "test-exposed",
            SplitRole::None => "none",
            SplitRole::Unknown => "unknown",
        }
    }
    /// Parses the supplied representation, returning an error when the input is invalid.
    pub fn parse(s: &str) -> Result<SplitRole> {
        Ok(match s {
            "train" => SplitRole::Train,
            "val263" => SplitRole::Val263,
            "sealed" => SplitRole::Sealed,
            "test-exposed" => SplitRole::TestExposed,
            "none" => SplitRole::None,
            "unknown" => SplitRole::Unknown,
            _ => bail!("unknown split_role {s:?}"),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
/// `DistilledOutcome` is an enum for the outcome contract in this module: gpdistilled/v1: the permanent (T0) table that keeps every GP result's training signal after its raw bodies are packed and archived (strategy §8.3).
pub enum DistilledOutcome {
    /// The `Ok` case of `DistilledOutcome`.
    Ok,
    /// The `Failed` case of `DistilledOutcome`.
    Failed,
    /// The `Timeout` case of `DistilledOutcome`.
    Timeout,
    /// Only part of the outcome mass is supported by measured outcomes.
    Partial,
}

/// Compatibility name for callers of the distilled outcome contract.
pub type Outcome = DistilledOutcome;

impl DistilledOutcome {
    /// Returns the stable string representation of this value.
    pub fn as_str(self) -> &'static str {
        match self {
            DistilledOutcome::Ok => "ok",
            DistilledOutcome::Failed => "failed",
            DistilledOutcome::Timeout => "timeout",
            DistilledOutcome::Partial => "partial",
        }
    }
    fn parse(s: &str) -> Result<DistilledOutcome> {
        Ok(match s {
            "ok" => DistilledOutcome::Ok,
            "failed" => DistilledOutcome::Failed,
            "timeout" => DistilledOutcome::Timeout,
            "partial" => DistilledOutcome::Partial,
            _ => bail!("unknown outcome {s:?}"),
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
/// `DistilledRow` is a record for the distilled row contract in this module: gpdistilled/v1: the permanent (T0) table that keeps every GP result's training signal after its raw bodies are packed and archived (strategy §8.3).
pub struct DistilledRow {
    /// Program sha recorded for this distilled row.
    pub program_sha: String,
    /// Optional program value; `None` means it is not available.
    pub program: Option<String>,
    /// Dataset recorded for this distilled row.
    pub dataset: String,
    /// SHA-256 digest identifying the associated context.
    pub context_sha256: Option<String>,
    /// Optional fold value; `None` means it is not available.
    pub fold: Option<i64>,
    /// Split role recorded for this distilled row.
    pub split_role: SplitRole,
    /// Distilled outcome recorded for this row.
    pub outcome: DistilledOutcome,
    /// Measured score for this outcome; absence denotes a failed draw.
    pub score: Option<f64>,
    /// Optional score name value; `None` means it is not available.
    pub score_name: Option<String>,
    /// Optional rank ctx value; `None` means it is not available.
    pub rank_ctx: Option<i64>,
    /// Identifier for the associated champion.
    pub champion_id: Option<String>,
    /// Optional superseded by value; `None` means it is not available.
    pub superseded_by: Option<String>,
    /// Optional seed packet value; `None` means it is not available.
    pub seed_packet: Option<String>,
    /// Optional policylm ckpt value; `None` means it is not available.
    pub policylm_ckpt: Option<String>,
    /// Optional campaign value; `None` means it is not available.
    pub campaign: Option<String>,
    /// Identifier for the associated run.
    pub run_id: String,
    /// Family recorded for this distilled row.
    pub family: String,
    /// Qualified recorded for this distilled row.
    pub qualified: bool,
    /// Optional l6 delta value; `None` means it is not available.
    pub l6_delta: Option<f64>,
    /// Source bucket recorded for this distilled row.
    pub source_bucket: String,
    /// Source key recorded for this distilled row.
    pub source_key: String,
    /// Identifier for the associated source version.
    pub source_version_id: Option<String>,
    /// SHA-256 digest identifying the associated source.
    pub source_sha256: String,
    /// Optional pack key value; `None` means it is not available.
    pub pack_key: Option<String>,
    /// Optional pack entry value; `None` means it is not available.
    pub pack_entry: Option<i64>,
}

/// The column order and types of `gp_distilled/v1`.
pub fn columns() -> Vec<ColumnSpec> {
    let c = |name: &str, ty, required| ColumnSpec {
        name: name.into(),
        ty,
        required,
    };
    use ColumnType::{Boolean, Double, Int64, Utf8};
    vec![
        c("program_sha", Utf8, true),
        c("program", Utf8, false),
        c("dataset", Utf8, true),
        c("context_sha256", Utf8, false),
        c("fold", Int64, false),
        c("split_role", Utf8, true),
        c("outcome", Utf8, true),
        c("score", Double, false),
        c("score_name", Utf8, false),
        c("rank_ctx", Int64, false),
        c("champion_id", Utf8, false),
        c("superseded_by", Utf8, false),
        c("seed_packet", Utf8, false),
        c("policylm_ckpt", Utf8, false),
        c("campaign", Utf8, false),
        c("run_id", Utf8, true),
        c("family", Utf8, true),
        c("qualified", Boolean, true),
        c("l6_delta", Double, false),
        c("source_bucket", Utf8, true),
        c("source_key", Utf8, true),
        c("source_version_id", Utf8, false),
        c("source_sha256", Utf8, true),
        c("pack_key", Utf8, false),
        c("pack_entry", Int64, false),
    ]
}

fn is_hex64(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Checks one row against the table's invariants.
pub fn validate(r: &DistilledRow, known_scores: &BTreeSet<String>) -> Result<()> {
    ensure!(
        is_hex64(&r.program_sha),
        "program_sha {:?} is not 64 hex digits",
        r.program_sha
    );
    ensure!(
        is_hex64(&r.source_sha256),
        "source_sha256 of {} is not 64 hex digits",
        r.source_key
    );
    if let Some(c) = &r.context_sha256 {
        ensure!(is_hex64(c), "context_sha256 {c:?} is not 64 hex digits");
    }
    ensure!(
        !r.run_id.is_empty() && !r.family.is_empty() && !r.dataset.is_empty(),
        "run_id, family and dataset are required"
    );
    ensure!(
        !r.source_bucket.is_empty() && !r.source_key.is_empty(),
        "source bucket and key are required"
    );
    match (r.score, &r.score_name) {
        (Some(s), Some(name)) => {
            ensure!(s.is_finite(), "score {s} is not finite");
            ensure!(
                known_scores.contains(name),
                "unknown score identity {name:?}"
            );
            ensure!(
                r.outcome == DistilledOutcome::Ok,
                "a scored row must have outcome ok, not {}",
                r.outcome.as_str()
            );
        }
        (None, None) => {}
        (Some(_), None) => bail!("score without score_name"),
        (None, Some(_)) => bail!("score_name without score"),
    }
    if let Some(d) = r.l6_delta {
        ensure!(d.is_finite(), "l6_delta is not finite");
        ensure!(
            r.score_name.as_deref() == Some("l6"),
            "l6_delta needs score_name l6"
        );
    }
    if r.qualified {
        ensure!(
            r.score_name.as_deref() == Some("l6") && r.l6_delta.is_some(),
            "qualified=true needs a paired L6 evaluation against frozen v1 (score_name l6 and l6_delta)"
        );
    }
    if r.superseded_by.is_some() {
        ensure!(r.champion_id.is_some(), "superseded_by needs champion_id");
    }
    ensure!(
        r.pack_entry.is_none() || r.pack_key.is_some(),
        "pack_entry needs pack_key"
    );
    Ok(())
}

fn s(v: &Option<String>) -> Cell {
    v.as_ref().map_or(Cell::Null, |x| Cell::Str(x.clone()))
}
fn i(v: Option<i64>) -> Cell {
    v.map_or(Cell::Null, Cell::I64)
}
fn f(v: Option<f64>) -> Cell {
    v.map_or(Cell::Null, Cell::F64)
}

fn to_cells(r: &DistilledRow) -> Vec<Cell> {
    vec![
        Cell::Str(r.program_sha.clone()),
        s(&r.program),
        Cell::Str(r.dataset.clone()),
        s(&r.context_sha256),
        i(r.fold),
        Cell::Str(r.split_role.as_str().into()),
        Cell::Str(r.outcome.as_str().into()),
        f(r.score),
        s(&r.score_name),
        i(r.rank_ctx),
        s(&r.champion_id),
        s(&r.superseded_by),
        s(&r.seed_packet),
        s(&r.policylm_ckpt),
        s(&r.campaign),
        Cell::Str(r.run_id.clone()),
        Cell::Str(r.family.clone()),
        Cell::Bool(r.qualified),
        f(r.l6_delta),
        Cell::Str(r.source_bucket.clone()),
        Cell::Str(r.source_key.clone()),
        s(&r.source_version_id),
        Cell::Str(r.source_sha256.clone()),
        s(&r.pack_key),
        i(r.pack_entry),
    ]
}

fn sort_key(r: &DistilledRow) -> (String, String, i64, String, String) {
    (
        r.program_sha.clone(),
        r.context_sha256.clone().unwrap_or_default(),
        r.fold.unwrap_or(-1),
        r.score_name.clone().unwrap_or_default(),
        r.source_key.clone(),
    )
}

/// Validates, sorts and writes the rows as a zstd-paged Parquet file. Duplicate identities
/// (program, context, fold, score identity, source key) are refused.
pub fn write(
    rows: &[DistilledRow],
    known_scores: &BTreeSet<String>,
    metadata: &[(String, String)],
) -> Result<Vec<u8>> {
    let mut sorted: Vec<&DistilledRow> = rows.iter().collect();
    for (n, r) in rows.iter().enumerate() {
        validate(r, known_scores).with_context(|| format!("distilled row {n}"))?;
    }
    sorted.sort_by_key(|r| sort_key(r));
    for w in sorted.windows(2) {
        ensure!(
            sort_key(w[0]) != sort_key(w[1]),
            "duplicate distilled row identity {:?}",
            sort_key(w[0])
        );
    }
    let mut meta = vec![
        ("schema".to_string(), DISTILLED_SCHEMA.to_string()),
        ("rows".to_string(), rows.len().to_string()),
    ];
    meta.extend(metadata.iter().cloned());
    let table = Table {
        columns: columns(),
        rows: sorted.into_iter().map(to_cells).collect(),
        metadata: meta,
    };
    parquet::write(&table, true, crate::TOOL_VERSION)
}

fn cell_str(c: &Cell, col: &str) -> Result<Option<String>> {
    Ok(match c {
        Cell::Null => None,
        Cell::Str(s) => Some(s.clone()),
        other => bail!("column {col}: expected a string, found {other:?}"),
    })
}

/// Reads a `gp_distilled/v1` file back; the schema must match exactly.
pub fn read(bytes: &[u8], known_scores: &BTreeSet<String>) -> Result<Vec<DistilledRow>> {
    let t = parquet::read(bytes)?;
    ensure!(
        t.columns == columns(),
        "file columns do not match gp_distilled/v1"
    );
    let meta: BTreeMap<_, _> = t.metadata.iter().cloned().collect();
    ensure!(
        meta.get("schema").map(String::as_str) == Some(DISTILLED_SCHEMA),
        "file schema metadata is not {DISTILLED_SCHEMA}"
    );
    let mut out = Vec::with_capacity(t.rows.len());
    for (n, row) in t.rows.iter().enumerate() {
        let st = |k: usize| cell_str(&row[k], &t.columns[k].name);
        let req = |k: usize| -> Result<String> {
            st(k)?.with_context(|| format!("row {n}: {} is null", t.columns[k].name))
        };
        let int = |k: usize| -> Result<Option<i64>> {
            Ok(match &row[k] {
                Cell::Null => None,
                Cell::I64(v) => Some(*v),
                other => bail!(
                    "row {n}: {} expected int64, found {other:?}",
                    t.columns[k].name
                ),
            })
        };
        let dbl = |k: usize| -> Result<Option<f64>> {
            Ok(match &row[k] {
                Cell::Null => None,
                Cell::F64(v) => Some(*v),
                other => bail!(
                    "row {n}: {} expected double, found {other:?}",
                    t.columns[k].name
                ),
            })
        };
        let r = DistilledRow {
            program_sha: req(0)?,
            program: st(1)?,
            dataset: req(2)?,
            context_sha256: st(3)?,
            fold: int(4)?,
            split_role: SplitRole::parse(&req(5)?)?,
            outcome: DistilledOutcome::parse(&req(6)?)?,
            score: dbl(7)?,
            score_name: st(8)?,
            rank_ctx: int(9)?,
            champion_id: st(10)?,
            superseded_by: st(11)?,
            seed_packet: st(12)?,
            policylm_ckpt: st(13)?,
            campaign: st(14)?,
            run_id: req(15)?,
            family: req(16)?,
            qualified: matches!(row[17], Cell::Bool(true)),
            l6_delta: dbl(18)?,
            source_bucket: req(19)?,
            source_key: req(20)?,
            source_version_id: st(21)?,
            source_sha256: req(22)?,
            pack_key: st(23)?,
            pack_entry: int(24)?,
        };
        validate(&r, known_scores).with_context(|| format!("row {n} read back"))?;
        out.push(r);
    }
    if let Some(declared) = meta.get("rows") {
        ensure!(
            *declared == out.len().to_string(),
            "metadata says {declared} rows, file holds {}",
            out.len()
        );
    }
    Ok(out)
}

/// The E3 precondition (§8.2): the distilled rows for a source cover every raw record exactly
/// once. `raw_keys` are the source object keys that were distilled.
pub fn reconcile(rows: &[DistilledRow], raw_keys: &BTreeSet<String>) -> Result<()> {
    let got: BTreeSet<String> = rows.iter().map(|r| r.source_key.clone()).collect();
    let missing: Vec<&String> = raw_keys.difference(&got).take(5).collect();
    let extra: Vec<&String> = got.difference(raw_keys).take(5).collect();
    ensure!(
        missing.is_empty() && extra.is_empty(),
        "distilled rows do not reconcile with the raw records: missing {missing:?}, extra {extra:?}"
    );
    Ok(())
}
