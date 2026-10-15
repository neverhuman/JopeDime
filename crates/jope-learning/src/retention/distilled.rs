//! `gp_distilled/v1`: the permanent (T0) table that keeps every GP result's training signal
//! after its raw bodies are packed and archived (strategy §8.3).
//!
//! One row per (program_sha, context, fold, score identity, source object). Failures and
//! partial results are rows too, with `outcome` set and no score: they are negative signal
//! (§8.5 E4). Validation refuses rows that would corrupt the table's meaning:
//! - a score must be finite and carry a known score identity; scores are never compared across
//!   identities;
//! - `qualified=true` only for a paired L6 evaluation against frozen v1 (`score_name = l6` with
//!   an `l6_delta`); L6 is a loss, lower is better, so `l6_delta < 0` is a candidate improvement;
//! - `superseded_by` needs a `champion_id`, and only qualified champions supersede (§8.5).

use super::parquet::{self, Cell, ColumnSpec, ColumnType, Table};
use anyhow::{bail, ensure, Context, Result};
use std::collections::{BTreeMap, BTreeSet};

pub const DISTILLED_SCHEMA: &str = "gp_distilled/v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum SplitRole {
    Train,
    Val263,
    Sealed,
    TestExposed,
    None,
    Unknown,
}

impl SplitRole {
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
    fn parse(s: &str) -> Result<SplitRole> {
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
pub enum Outcome {
    Ok,
    Failed,
    Timeout,
    Partial,
}

impl Outcome {
    pub fn as_str(self) -> &'static str {
        match self {
            Outcome::Ok => "ok",
            Outcome::Failed => "failed",
            Outcome::Timeout => "timeout",
            Outcome::Partial => "partial",
        }
    }
    fn parse(s: &str) -> Result<Outcome> {
        Ok(match s {
            "ok" => Outcome::Ok,
            "failed" => Outcome::Failed,
            "timeout" => Outcome::Timeout,
            "partial" => Outcome::Partial,
            _ => bail!("unknown outcome {s:?}"),
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DistilledRow {
    pub program_sha: String,
    pub program: Option<String>,
    pub dataset: String,
    pub context_sha256: Option<String>,
    pub fold: Option<i64>,
    pub split_role: SplitRole,
    pub outcome: Outcome,
    pub score: Option<f64>,
    pub score_name: Option<String>,
    pub rank_ctx: Option<i64>,
    pub champion_id: Option<String>,
    pub superseded_by: Option<String>,
    pub seed_packet: Option<String>,
    pub policylm_ckpt: Option<String>,
    pub campaign: Option<String>,
    pub run_id: String,
    pub family: String,
    pub qualified: bool,
    pub l6_delta: Option<f64>,
    pub source_bucket: String,
    pub source_key: String,
    pub source_version_id: Option<String>,
    pub source_sha256: String,
    pub pack_key: Option<String>,
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
                r.outcome == Outcome::Ok,
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
            outcome: Outcome::parse(&req(6)?)?,
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

/// One self-play or GP inner-cut observation to be distilled into `gp_distilled/v1`.
#[derive(Clone, Debug, PartialEq)]
pub struct SelfPlayDistillObservation {
    pub program_sha: String,
    pub program: Option<String>,
    pub outcome: Outcome,
    /// Measured quality `q` (higher is better; required iff `outcome == Outcome::Ok`).
    pub q: Option<f64>,
    /// Seed provenance label (e.g. `"ordinary_gp"`, `"bootstrap_warm_start"`, `"selfplay_archive_r1"`).
    pub seed_packet: Option<String>,
    pub source_key: String,
    pub source_version_id: Option<String>,
    pub source_sha256: String,
}

/// Context-level metadata for distilling a self-play / GP evaluation batch into `gp_distilled/v1`.
#[derive(Clone, Debug, PartialEq)]
pub struct SelfPlayDistillSpec {
    pub dataset: String,
    pub context_sha256: String,
    pub fold: Option<i64>,
    pub split_role: SplitRole,
    pub score_name: String,
    pub policylm_ckpt: Option<String>,
    pub campaign: Option<String>,
    pub run_id: String,
    pub family: String,
    pub source_bucket: String,
}

/// Distills a batch of self-play / GP inner-cut observations for one context into validated
/// `gp_distilled/v1` rows, assigning deterministic within-context rank (`rank_ctx = 1..=M`) and
/// recording the rank-1 `champion_id` (with `qualified = false` and `superseded_by = None`
/// because inner-cut self-play rewards are never paired L6 qualifications).
pub fn distill_selfplay_observations(
    spec: &SelfPlayDistillSpec,
    observations: &[SelfPlayDistillObservation],
    known_scores: &BTreeSet<String>,
) -> Result<Vec<DistilledRow>> {
    ensure!(
        !observations.is_empty(),
        "cannot distill an empty self-play observation batch"
    );
    ensure!(
        crate::canon::is_hex64(&spec.context_sha256),
        "context_sha256 {:?} must be lowercase 64-hex",
        spec.context_sha256
    );
    ensure!(
        known_scores.contains(&spec.score_name),
        "unknown score_name {:?}",
        spec.score_name
    );
    ensure!(
        spec.score_name != "l6",
        "self-play inner-cut distillation may not claim score_name \"l6\""
    );

    // Rank all successful observations by descending q, breaking ties by ascending program_sha.
    let mut ok_indices: Vec<usize> = Vec::new();
    for (idx, obs) in observations.iter().enumerate() {
        match (obs.outcome, obs.q) {
            (Outcome::Ok, Some(q)) => {
                ensure!(q.is_finite(), "observation {idx}: non-finite q");
                ok_indices.push(idx);
            }
            (Outcome::Ok, None) => bail!("observation {idx}: outcome ok requires finite q"),
            (_, Some(_)) => bail!("observation {idx}: failed/timeout observation must not carry q"),
            (_, None) => {}
        }
    }
    ok_indices.sort_by(|&a, &b| {
        let qa = observations[a].q.unwrap();
        let qb = observations[b].q.unwrap();
        qb.total_cmp(&qa).then_with(|| {
            observations[a]
                .program_sha
                .cmp(&observations[b].program_sha)
        })
    });

    let champion_id = ok_indices
        .first()
        .map(|&i| observations[i].program_sha.clone());
    let mut rank_by_idx = vec![None; observations.len()];
    for (rank_zero, &obs_idx) in ok_indices.iter().enumerate() {
        rank_by_idx[obs_idx] = Some((rank_zero + 1) as i64);
    }

    let mut out = Vec::with_capacity(observations.len());
    for (idx, obs) in observations.iter().enumerate() {
        let row = DistilledRow {
            program_sha: obs.program_sha.clone(),
            program: obs.program.clone(),
            dataset: spec.dataset.clone(),
            context_sha256: Some(spec.context_sha256.clone()),
            fold: spec.fold,
            split_role: spec.split_role,
            outcome: obs.outcome,
            score: obs.q,
            score_name: obs.q.map(|_| spec.score_name.clone()),
            rank_ctx: rank_by_idx[idx],
            champion_id: obs.q.as_ref().and(champion_id.clone()),
            superseded_by: None,
            seed_packet: obs.seed_packet.clone(),
            policylm_ckpt: spec.policylm_ckpt.clone(),
            campaign: spec.campaign.clone(),
            run_id: spec.run_id.clone(),
            family: spec.family.clone(),
            qualified: false,
            l6_delta: None,
            source_bucket: spec.source_bucket.clone(),
            source_key: obs.source_key.clone(),
            source_version_id: obs.source_version_id.clone(),
            source_sha256: obs.source_sha256.clone(),
            pack_key: None,
            pack_entry: None,
        };
        validate(&row, known_scores).with_context(|| format!("self-play distilled row {idx}"))?;
        out.push(row);
    }
    Ok(out)
}

fn program_expr_tokens(expr: &str) -> Vec<u32> {
    let mut tokens = vec![1u32];
    for part in expr
        .split(|c: char| c.is_whitespace() || c == '(' || c == ')')
        .filter(|s| !s.is_empty())
    {
        let mut h = 2166136261u32;
        for b in part.bytes() {
            h ^= u32::from(b);
            h = h.wrapping_mul(16777619);
        }
        tokens.push(3 + (h % 61));
    }
    tokens.push(2u32);
    tokens
}

/// Bridges validated `gp_distilled/v1` rows for a single score identity into
/// [`crate::pools::pool::MeasurementRow`] records for shadow pool freezing and cross-fitted
/// surrogate (`ContextualSurrogateQHat`) training.
///
/// Refuses mixed score identities, missing `context_sha256`, or `SplitRole::None` / `Unknown`.
/// Non-train roles (`Val263`, `Sealed`, `TestExposed`) preserve their split label so
/// `pools::pool::freeze` enforces its held-out leakage refusal.
pub fn distilled_rows_to_measurement_rows(
    rows: &[DistilledRow],
    score_name: &str,
    protocol_sha256: &str,
    known_scores: &BTreeSet<String>,
) -> Result<Vec<crate::pools::pool::MeasurementRow>> {
    ensure!(
        !rows.is_empty(),
        "cannot convert empty distilled row slice to measurement rows"
    );
    ensure!(
        crate::canon::is_hex64(protocol_sha256),
        "protocol_sha256 must be lowercase 64-hex"
    );
    ensure!(
        known_scores.contains(score_name),
        "unknown score identity {score_name:?}"
    );
    let mut out = Vec::with_capacity(rows.len());
    for (idx, r) in rows.iter().enumerate() {
        validate(r, known_scores).with_context(|| format!("distilled row {idx}"))?;
        if let Some(name) = &r.score_name {
            ensure!(
                name == score_name,
                "row {idx}: cannot mix score_name {name:?} with requested {score_name:?}"
            );
        }
        let ctx_sha = r
            .context_sha256
            .as_ref()
            .with_context(|| format!("row {idx}: context_sha256 is required for pool freezing"))?;
        ensure!(
            crate::canon::is_hex64(ctx_sha) && crate::canon::is_hex64(&r.program_sha),
            "row {idx}: context_sha256 and program_sha must be lowercase 64-hex"
        );
        let role = match r.split_role {
            SplitRole::Train => "train",
            SplitRole::Val263 => "val",
            SplitRole::Sealed => "sealed",
            SplitRole::TestExposed => "test",
            SplitRole::None | SplitRole::Unknown => {
                bail!(
                    "row {idx}: split_role {} may not enter a measurement row",
                    r.split_role.as_str()
                )
            }
        };
        let (status, q) = match r.outcome {
            Outcome::Ok => (
                crate::pools::pool::Status::Complete,
                Some(
                    r.score
                        .with_context(|| format!("row {idx}: complete row missing score"))?,
                ),
            ),
            Outcome::Failed | Outcome::Partial => (crate::pools::pool::Status::ProgramFailed, None),
            Outcome::Timeout => (crate::pools::pool::Status::Timeout, None),
        };
        let tokens = r.program.as_deref().map(program_expr_tokens);
        out.push(crate::pools::pool::MeasurementRow {
            context_id: ctx_sha.clone(),
            parent_id: r.dataset.clone(),
            role: role.into(),
            program_sha256: r.program_sha.clone(),
            status,
            q,
            incumbent_q: None,
            protocol_sha256: protocol_sha256.to_owned(),
            tokens,
        });
    }
    Ok(out)
}
