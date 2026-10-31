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
/// `Outcome` is an enum for the outcome contract in this module: gpdistilled/v1: the permanent (T0) table that keeps every GP result's training signal after its raw bodies are packed and archived (strategy §8.3).
pub enum Outcome {
    /// The `Ok` case of `Outcome`.
    Ok,
    /// The `Failed` case of `Outcome`.
    Failed,
    /// The `Timeout` case of `Outcome`.
    Timeout,
    /// Only part of the outcome mass is supported by measured outcomes.
    Partial,
}

impl Outcome {
    /// Returns the stable string representation of this value.
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
    /// Outcome recorded for this distilled row.
    pub outcome: Outcome,
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

/// Orientation of a registered `gp_distilled/v1` score identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScoreDirection {
    /// Loss or error metric where smaller finite values are better
    /// (`cv_geomean_rmse_ratio`, `best_rust_rmse`, `validation_rmse`, `l6`).
    LowerIsBetter,
    /// Reward or utility metric where larger finite values are better.
    HigherIsBetter,
}

/// Returns the explicit orientation of a registered score identity, refusing any identity
/// without a declared orientation.
pub fn score_direction(score_name: &str) -> Result<ScoreDirection> {
    match score_name {
        "cv_geomean_rmse_ratio" | "best_rust_rmse" | "validation_rmse" | "l6" => {
            Ok(ScoreDirection::LowerIsBetter)
        }
        "shadow_pool_q" => Ok(ScoreDirection::HigherIsBetter),
        other => bail!("score identity {other:?} has no declared ScoreDirection"),
    }
}

/// One self-play or GP inner-cut observation to be distilled into `gp_distilled/v1`.
#[derive(Clone, Debug, PartialEq)]
pub struct SelfPlayDistillObservation {
    /// Program sha recorded for this self play distill observation.
    pub program_sha: String,
    /// Optional program value; `None` means it is not available.
    pub program: Option<String>,
    /// Outcome recorded for this self play distill observation.
    pub outcome: Outcome,
    /// Measured score under `spec.score_name`, oriented according to
    /// [`score_direction`]`(&spec.score_name)` (required iff `outcome == Outcome::Ok`).
    pub score: Option<f64>,
    /// Seed provenance label (e.g. `"ordinary_gp"`, `"bootstrap_warm_start"`, `"selfplay_archive_r1"`).
    pub seed_packet: Option<String>,
    /// Source key recorded for this self play distill observation.
    pub source_key: String,
    /// Identifier for the associated source version.
    pub source_version_id: Option<String>,
    /// SHA-256 digest identifying the associated source.
    pub source_sha256: String,
}

/// Context-level metadata for distilling a self-play / GP evaluation batch into `gp_distilled/v1`.
#[derive(Clone, Debug, PartialEq)]
pub struct SelfPlayDistillSpec {
    /// Dataset recorded for this self play distill spec.
    pub dataset: String,
    /// SHA-256 digest identifying the associated context.
    pub context_sha256: String,
    /// Optional fold value; `None` means it is not available.
    pub fold: Option<i64>,
    /// Split role recorded for this self play distill spec.
    pub split_role: SplitRole,
    /// Score name recorded for this self play distill spec.
    pub score_name: String,
    /// Optional policylm ckpt value; `None` means it is not available.
    pub policylm_ckpt: Option<String>,
    /// Optional campaign value; `None` means it is not available.
    pub campaign: Option<String>,
    /// Identifier for the associated run.
    pub run_id: String,
    /// Family recorded for this self play distill spec.
    pub family: String,
    /// Source bucket recorded for this self play distill spec.
    pub source_bucket: String,
}

/// Distills a batch of self-play / GP inner-cut observations for one context into validated
/// `gp_distilled/v1` rows, assigning deterministic within-context rank (`rank_ctx = 1..=M`)
/// according to [`score_direction`]`(&spec.score_name)` and recording the rank-1 `champion_id`
/// (with `qualified = false` and `superseded_by = None` because inner-cut self-play measurements
/// are never paired L6 qualifications).
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
    let direction = score_direction(&spec.score_name)?;

    // Rank all successful observations by best score under `direction`, breaking ties by ascending program_sha.
    let mut ok_indices: Vec<usize> = Vec::new();
    for (idx, obs) in observations.iter().enumerate() {
        match (obs.outcome, obs.score) {
            (Outcome::Ok, Some(s)) => {
                ensure!(s.is_finite(), "observation {idx}: non-finite score");
                ok_indices.push(idx);
            }
            (Outcome::Ok, None) => bail!("observation {idx}: outcome ok requires finite score"),
            (_, Some(_)) => {
                bail!("observation {idx}: failed/timeout observation must not carry score")
            }
            (_, None) => {}
        }
    }
    ok_indices.sort_by(|&a, &b| {
        let sa = observations[a].score.unwrap();
        let sb = observations[b].score.unwrap();
        let primary = match direction {
            ScoreDirection::LowerIsBetter => sa.total_cmp(&sb),
            ScoreDirection::HigherIsBetter => sb.total_cmp(&sa),
        };
        primary.then_with(|| {
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
            score: obs.score,
            score_name: obs.score.map(|_| spec.score_name.clone()),
            rank_ctx: rank_by_idx[idx],
            champion_id: obs.score.as_ref().and(champion_id.clone()),
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

/// Bridges validated `gp_distilled/v1` rows for a single non-`l6` score identity into
/// [`crate::pools::pool::MeasurementRow`] records (where `q` is higher-is-better) for shadow pool
/// freezing and cross-fitted surrogate (`ContextualSurrogateQHat`) training.
///
/// Lower-is-better identities (`cv_geomean_rmse_ratio`, `best_rust_rmse`, `validation_rmse`) are
/// negated (`q = -score`) so lower loss/RMSE maps strictly monotonically to higher pool reward `q`.
/// Refuses `"l6"`, unknown/unoriented score identities, mixed score identities, mixed `fold`
/// values within one `context_sha256`, missing `context_sha256`, or `SplitRole::None` / `Unknown`.
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
    ensure!(
        score_name != "l6",
        "held-out l6 validation loss rows may not be bridged into shadow training pools"
    );
    let direction = score_direction(score_name)?;
    let mut fold_by_ctx: BTreeMap<&str, Option<i64>> = BTreeMap::new();
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
        if let Some(&prev_fold) = fold_by_ctx.get(ctx_sha.as_str()) {
            ensure!(
                prev_fold == r.fold,
                "row {idx}: context_sha256 {ctx_sha} mixes multiple fold values ({prev_fold:?} vs {:?})",
                r.fold
            );
        } else {
            fold_by_ctx.insert(ctx_sha.as_str(), r.fold);
        }
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
            Outcome::Ok => {
                let raw_score = r
                    .score
                    .with_context(|| format!("row {idx}: complete row missing score"))?;
                let oriented_q = match direction {
                    ScoreDirection::LowerIsBetter => {
                        if raw_score == 0.0 {
                            0.0
                        } else {
                            -raw_score
                        }
                    }
                    ScoreDirection::HigherIsBetter => raw_score,
                };
                (crate::pools::pool::Status::Complete, Some(oriented_q))
            }
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

/// Accounting summary for [`distill_gp_results_index_with_stats`].
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GpIndexDistillStats {
    /// Total index runs recorded for this GP index distill stats.
    pub total_index_runs: usize,
    /// Number of distilled rows represented here.
    pub distilled_rows: usize,
    /// Skipped unscored runs recorded for this GP index distill stats.
    pub skipped_unscored_runs: usize,
    /// Skipped incomplete provenance runs recorded for this GP index distill stats.
    pub skipped_incomplete_provenance_runs: usize,
    /// Reconciled s3 keys recorded for this GP index distill stats.
    pub reconciled_s3_keys: usize,
}

/// Additive complete archive. Unlike the v1 scored-only API, this preserves every supplied
/// occurrence and keeps missing producer/role/event evidence held. It grants no replay access.
pub fn import_gp_results_index_complete(
    index_bytes: &[u8],
    producer: Option<crate::replay::index_producer::ProducerInput<'_>>,
    known_scores: &BTreeSet<String>,
    dependencies: &BTreeMap<String, crate::replay::observation::DependencyEvidence>,
    historical: Option<&crate::replay::index_producer::HistoricalFixture>,
) -> Result<crate::replay::observation::ObservationArchive> {
    crate::replay::observation::import_index(
        index_bytes,
        producer,
        known_scores,
        dependencies,
        historical,
    )
}

/// Exact legacy compatibility projection, including original order/ranks/skip statistics and
/// refusals. Explicit caller role is diagnostic v1 metadata, never an I04 replay capability.
pub fn legacy_scored_projection(
    archive: &crate::replay::observation::ObservationArchive,
    diagnostic_split_role: SplitRole,
    known_scores: &BTreeSet<String>,
) -> Result<(Vec<DistilledRow>, BTreeSet<String>, GpIndexDistillStats)> {
    let index = crate::replay::index_producer::parse_index(archive.original_index.as_bytes())?;
    distill_gp_results_index_with_stats(&index, diagnostic_split_role, known_scores)
}

/// Distills all S3 context-level GP records from a `jopedime.gp-results-index/v1` document
/// (`GP_RESULTS_INDEX.json`) into validated `gp_distilled/v1` rows and returns both the rows and
/// the set of distilled S3 object keys (`raw_keys`) after checking [`reconcile`].
pub fn distill_gp_results_index(
    index: &serde_json::Value,
    default_split_role: SplitRole,
    known_scores: &BTreeSet<String>,
) -> Result<(Vec<DistilledRow>, BTreeSet<String>)> {
    let (rows, raw_keys, _) =
        distill_gp_results_index_with_stats(index, default_split_role, known_scores)?;
    Ok((rows, raw_keys))
}

/// Distills all S3 context-level GP records from a `jopedime.gp-results-index/v1` document
/// (`GP_RESULTS_INDEX.json`) into validated `gp_distilled/v1` rows and returns the rows,
/// reconciled S3 keys, and exact row-accounting statistics across all `runs` entries.
pub fn distill_gp_results_index_with_stats(
    index: &serde_json::Value,
    default_split_role: SplitRole,
    known_scores: &BTreeSet<String>,
) -> Result<(Vec<DistilledRow>, BTreeSet<String>, GpIndexDistillStats)> {
    ensure!(
        index.get("schema").and_then(serde_json::Value::as_str)
            == Some("jopedime.gp-results-index/v1"),
        "expected schema \"jopedime.gp-results-index/v1\""
    );
    let runs = index
        .get("runs")
        .and_then(serde_json::Value::as_array)
        .context("gp-results-index missing runs array")?;

    type GroupKey = (String, String, Option<i64>, String, String, String, String);
    let mut grouped: BTreeMap<GroupKey, Vec<SelfPlayDistillObservation>> = BTreeMap::new();
    let mut raw_keys = BTreeSet::new();
    let mut skipped_unscored_runs = 0usize;
    let mut skipped_incomplete_provenance_runs = 0usize;

    for (idx, r) in runs.iter().enumerate() {
        let Some(score_name) = r.get("score_name").and_then(serde_json::Value::as_str) else {
            skipped_unscored_runs += 1;
            continue;
        };
        let Some(score) = r.get("score").and_then(serde_json::Value::as_f64) else {
            skipped_unscored_runs += 1;
            continue;
        };
        let (Some(bucket), Some(key), Some(obj_sha), Some(prog_sha), Some(dataset), Some(ctx_sha)) = (
            r.get("bucket").and_then(serde_json::Value::as_str),
            r.get("key").and_then(serde_json::Value::as_str),
            r.get("object_sha256").and_then(serde_json::Value::as_str),
            r.get("program_sha").and_then(serde_json::Value::as_str),
            r.get("dataset").and_then(serde_json::Value::as_str),
            r.get("context_sha256").and_then(serde_json::Value::as_str),
        ) else {
            skipped_incomplete_provenance_runs += 1;
            continue;
        };
        ensure!(
            r.get("qualified")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false)
                == (score_name == "l6"),
            "gp-results-index row {idx}: qualified must equal (score_name == \"l6\")"
        );
        let fold = r.get("fold").and_then(serde_json::Value::as_i64);
        let family = r
            .get("family")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("gp-policy-dl");
        let run_id = r
            .get("run_id")
            .and_then(serde_json::Value::as_str)
            .or_else(|| r.get("eval").and_then(serde_json::Value::as_str))
            .or_else(|| r.get("grain").and_then(serde_json::Value::as_str))
            .unwrap_or("gp-hwm");
        let program = r
            .get("program")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned);
        let version_id = r
            .get("version_id")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned);
        let grain = r
            .get("grain")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned);

        ensure!(
            raw_keys.insert(key.to_owned()),
            "gp-results-index row {idx}: duplicate S3 key {key:?}"
        );
        grouped
            .entry((
                dataset.to_owned(),
                ctx_sha.to_owned(),
                fold,
                score_name.to_owned(),
                family.to_owned(),
                run_id.to_owned(),
                bucket.to_owned(),
            ))
            .or_default()
            .push(SelfPlayDistillObservation {
                program_sha: prog_sha.to_owned(),
                program,
                outcome: Outcome::Ok,
                score: Some(score),
                seed_packet: grain,
                source_key: key.to_owned(),
                source_version_id: version_id,
                source_sha256: obj_sha.to_owned(),
            });
    }

    ensure!(
        !grouped.is_empty(),
        "gp-results-index contained no distillable S3 context-level GP records"
    );
    let mut rows = Vec::with_capacity(raw_keys.len());
    for ((dataset, context_sha256, fold, score_name, family, run_id, source_bucket), obs) in grouped
    {
        let spec = SelfPlayDistillSpec {
            dataset,
            context_sha256,
            fold,
            split_role: default_split_role,
            score_name,
            policylm_ckpt: None,
            campaign: None,
            run_id,
            family,
            source_bucket,
        };
        let mut batch = distill_selfplay_observations(&spec, &obs, known_scores)?;
        rows.append(&mut batch);
    }
    reconcile(&rows, &raw_keys)?;
    let stats = GpIndexDistillStats {
        total_index_runs: runs.len(),
        distilled_rows: rows.len(),
        skipped_unscored_runs,
        skipped_incomplete_provenance_runs,
        reconciled_s3_keys: raw_keys.len(),
    };
    ensure!(
        stats.total_index_runs
            == stats.distilled_rows
                + stats.skipped_unscored_runs
                + stats.skipped_incomplete_provenance_runs,
        "gp-results-index row accounting mismatch"
    );
    Ok((rows, raw_keys, stats))
}
