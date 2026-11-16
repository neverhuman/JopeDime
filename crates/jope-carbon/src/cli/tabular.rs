//! Universal RFC-4180 CSV/TSV/semicolon/JSONL reader (`.gz`, `.zst`, UTF-16 BOM, `stdin`),
//! zero-leakage `FIT`-only tabular feature encoder (numeric, currency/percent, boolean, ISO-8601 date,
//! one-hot & smoothed leave-one-out categorical, missing-value imputation, custom objective `aux` separation),
//! fuzzy column suggestion, and instant local dataset preflight inspector (`carbon inspect`).

use crate::error::CarbonErrCtx as _;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Read;
use std::path::Path;

use jope_prep::{
    jmat_lite::split_and_encode_numeric_jmat_v1, FitLandmarkBundle, ObjectiveSpec, SplitPlan,
};
use serde::{Deserialize, Serialize};

use crate::{
    cli::client::CarbonClientPreparedDataset,
    modes::planner::{
        estimate_ms_per_candidate_1vcpu, plan_carbon_execution, CarbonExecutionMode,
        CarbonPlanInput,
    },
    role::{CarbonEffortTier, CarbonProductTier},
};

const MAX_DECOMPRESSED_TABLE_BYTES: u64 = 256 * 1024 * 1024;
const MAX_STDIN_TABLE_BYTES: u64 = 128 * 1024 * 1024;

/// `CarbonRawTable` is a struct holding parsed raw string/missing cells from a delimited or JSONL file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CarbonRawTable {
    pub headers: Vec<String>,
    pub rows: Vec<Vec<Option<String>>>,
    pub delimiter: char,
    pub had_header: bool,
    pub extra_na_tokens: Vec<String>,
    pub inf_as_nan: bool,
}

/// `CarbonFeatureTransform` is an enum describing how one raw input column maps to numeric features.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CarbonFeatureTransform {
    /// Clean or median-imputed numeric column (including `$1,200` / `12.5%`), optionally paired with `<col>__isna`.
    Numeric {
        source_column: String,
        median: f64,
        emit_missing_indicator: bool,
    },
    /// Boolean column (`true`/`false`, `yes`/`no`, `t`/`f`, `1`/`0`) imputed with `mode`.
    Boolean { source_column: String, mode: f64 },
    /// ISO-8601 date/timestamp column expanded into `FIT`-centered days plus calendar components.
    DateIso {
        source_column: String,
        median_epoch_days: f64,
        emit_calendar_parts: bool,
        emit_missing_indicator: bool,
    },
    /// Low-cardinality categorical expanded into deterministic one-hot indicators `<col>__<cat>`.
    OneHot {
        source_column: String,
        categories: Vec<String>,
    },
    /// High-cardinality categorical encoded via `FIT`-only smoothed target + frequency features.
    TargetFreq {
        source_column: String,
        global_mean: f64,
        smoothing: f64,
        fit_count: usize,
        category_stats: BTreeMap<String, (usize, f64)>,
    },
}

/// `CarbonTabularEncoder` is a struct storing the `FIT`-learned zero-leakage feature pipeline
/// serialized inside saved Carbon model handles for `carbon predict` on unlabeled CSVs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CarbonTabularEncoder {
    pub raw_feature_columns: Vec<String>,
    pub dropped_columns: Vec<(String, String)>,
    pub transforms: Vec<CarbonFeatureTransform>,
    pub encoded_feature_names: Vec<String>,
    pub target_column: String,
    pub delimiter: char,
    pub had_header: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub extra_na_tokens: Vec<String>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub inf_as_nan: bool,
}

/// `CarbonTrainingPrepOptions` configures split strategy, calibration, custom objectives, and hygiene checks.
#[derive(Clone, Copy, Debug)]
pub struct CarbonTrainingPrepOptions<'a> {
    pub requested_target: Option<&'a str>,
    pub seed: u64,
    pub holdout_fraction: f64,
    pub max_encoded_features: usize,
    pub objective_spec: Option<&'a ObjectiveSpec>,
    pub split_mode: Option<&'a str>,
    pub calibration_mode: Option<&'a str>,
    pub reject_constant_target: bool,
}

/// `CarbonEncodedTrainingSet` is a struct bundling the `CarbonClientPreparedDataset`, split matrices,
/// auxiliary objective columns, hygiene warnings, and reusable `CarbonTabularEncoder`.
#[derive(Clone, Debug)]
pub struct CarbonEncodedTrainingSet {
    pub prepared: CarbonClientPreparedDataset,
    pub encoder: CarbonTabularEncoder,
    pub total_valid_rows: usize,
    pub dropped_target_rows: usize,
    pub fit_x: Vec<Vec<f64>>,
    pub fit_y: Vec<f64>,
    pub holdout_x: Vec<Vec<f64>>,
    pub holdout_y: Vec<f64>,
    pub fit_aux_cols: BTreeMap<String, Vec<f64>>,
    pub holdout_aux_cols: BTreeMap<String, Vec<f64>>,
    pub hygiene_warnings: Vec<String>,
}

/// `CarbonColumnProfile` is a struct summarizing one column in `carbon inspect`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CarbonColumnProfile {
    pub name: String,
    pub role: String,
    pub detected_type: String,
    pub missing_count: usize,
    pub missing_pct: f64,
    pub distinct_count: usize,
    pub numeric_min: Option<f64>,
    pub numeric_median: Option<f64>,
    pub numeric_mean: Option<f64>,
    pub numeric_std: Option<f64>,
    pub numeric_max: Option<f64>,
    pub sample_categories: Vec<String>,
}

/// `CarbonEffortCostPreview` is a struct summarizing estimated latency, topology, and USD cost per tier.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CarbonEffortCostPreview {
    pub tier: String,
    pub effort: String,
    pub execution_mode: CarbonExecutionMode,
    pub eval_shards: u32,
    pub planned_waves: u32,
    pub estimated_wall_secs: f64,
    pub estimated_usd: f64,
    pub runs_per_1_usd: Option<f64>,
}

/// `CarbonTabularInspectReport` is a struct returned by `carbon inspect` for zero-network preflight.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CarbonTabularInspectReport {
    pub file_path: String,
    pub delimiter: String,
    pub has_header: bool,
    pub raw_rows: usize,
    pub valid_target_rows: usize,
    pub dropped_missing_target_rows: usize,
    pub raw_columns: usize,
    pub encoded_feature_count: usize,
    pub target_column: String,
    pub target_min: f64,
    pub target_median: f64,
    pub target_mean: f64,
    pub target_std: f64,
    pub target_max: f64,
    pub target_outliers_3sigma: usize,
    pub recommended_effort: String,
    pub warnings: Vec<String>,
    pub dropped_columns: Vec<(String, String)>,
    pub columns: Vec<CarbonColumnProfile>,
    pub effort_previews: Vec<CarbonEffortCostPreview>,
}

fn is_missing_token_with_extra(raw: &str, extra_na: &[String]) -> bool {
    let t = raw.trim();
    if t.is_empty() {
        return true;
    }
    let lower = t.to_ascii_lowercase();
    if matches!(
        lower.as_str(),
        "na" | "n/a"
            | "nan"
            | "-nan"
            | "+nan"
            | "null"
            | "none"
            | "nil"
            | "?"
            | "#n/a"
            | "#value!"
            | "#div/0!"
            | "#ref!"
            | "#name?"
            | "#num!"
            | "<na>"
            | "missing"
    ) {
        return true;
    }
    extra_na
        .iter()
        .any(|tok| !tok.is_empty() && t.eq_ignore_ascii_case(tok.trim()))
}

fn is_infinity_token(raw: &str) -> bool {
    matches!(
        raw.trim().to_ascii_lowercase().as_str(),
        "inf" | "+inf" | "-inf" | "infinity" | "+infinity" | "-infinity"
    )
}

fn parse_finite_f64(raw: &str) -> Option<f64> {
    raw.trim().parse::<f64>().ok().filter(|v| v.is_finite())
}

/// Parse plain finite `f64`, currency/grouped numbers (`$1,200.50`, `($450.00)`), and percentages (`12.5%`).
pub fn parse_flexible_numeric_f64(raw: &str) -> Option<f64> {
    if let Some(v) = parse_finite_f64(raw) {
        return Some(v);
    }
    let mut s = raw.trim();
    if s.len() < 2 {
        return None;
    }
    let mut sign = 1.0_f64;
    if s.starts_with('(') && s.ends_with(')') && s.len() >= 3 {
        sign = -1.0;
        s = s[1..s.len() - 1].trim();
    } else if let Some(rest) = s.strip_prefix('-') {
        sign = -1.0;
        s = rest.trim();
    } else if let Some(rest) = s.strip_prefix('+') {
        s = rest.trim();
    }

    let mut scale = 1.0_f64;
    let mut had_special = false;
    if let Some(rest) = s.strip_suffix('%') {
        scale = 0.01;
        had_special = true;
        s = rest.trim();
    }
    for sym in ['$', '€', '£', '¥'] {
        if let Some(rest) = s.strip_prefix(sym) {
            had_special = true;
            s = rest.trim();
            break;
        }
        if let Some(rest) = s.strip_suffix(sym) {
            had_special = true;
            s = rest.trim();
            break;
        }
    }
    if let Some(rest) = s.strip_prefix('-') {
        sign = -sign;
        s = rest.trim();
    }
    if s.contains(',') {
        let (int_part, frac_part) = match s.split_once('.') {
            Some((i, f)) => (i, Some(f)),
            None => (s, None),
        };
        let groups: Vec<&str> = int_part.split(',').collect();
        if groups.len() < 2
            || groups[0].is_empty()
            || groups[0].len() > 3
            || !groups[0].bytes().all(|b| b.is_ascii_digit())
            || !groups[1..]
                .iter()
                .all(|g| g.len() == 3 && g.bytes().all(|b| b.is_ascii_digit()))
        {
            return None;
        }
        if let Some(fp) = frac_part {
            if fp.contains(',') || !fp.bytes().all(|b| b.is_ascii_digit()) {
                return None;
            }
        }
        let cleaned = s.replace(',', "");
        let v = cleaned.parse::<f64>().ok().filter(|x| x.is_finite())?;
        return Some(sign * v * scale);
    }
    if !had_special {
        return None;
    }
    let v = s.parse::<f64>().ok().filter(|x| x.is_finite())?;
    Some(sign * v * scale)
}

fn parse_boolean_token(raw: &str) -> Option<f64> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "true" | "yes" | "y" | "t" => Some(1.0),
        "false" | "no" | "n" | "f" => Some(0.0),
        _ => None,
    }
}

fn is_leap_year(y: i32) -> bool {
    (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0)
}

fn days_in_month(y: i32, m: u32) -> u32 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_leap_year(y) {
                29
            } else {
                28
            }
        }
        _ => 0,
    }
}

fn days_from_civil(y: i32, m: u32, d: u32) -> i64 {
    let y_adj = i64::from(y) - if m <= 2 { 1 } else { 0 };
    let era = if y_adj >= 0 { y_adj } else { y_adj - 399 } / 400;
    let yoe = (y_adj - era * 400) as u64;
    let m_u64 = u64::from(m);
    let doy = (153 * (if m_u64 > 2 { m_u64 - 3 } else { m_u64 + 9 }) + 2) / 5 + u64::from(d) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + (doe as i64) - 719_468
}

/// Parse ISO-8601 `YYYY-MM-DD` or `YYYY-MM-DD[T ]HH:MM[:SS[.fff]][Z|±HH[:MM]]` into `(epoch_days, month, dow)`.
fn parse_iso8601_date(raw: &str) -> Option<(f64, f64, f64)> {
    let trimmed = raw.trim_matches(|c: char| c.is_ascii_whitespace());
    let bytes = trimmed.as_bytes();
    if !(10..=40).contains(&bytes.len()) || bytes[4] != b'-' || bytes[7] != b'-' {
        return None;
    }
    let year: i32 = trimmed.get(0..4)?.parse().ok()?;
    let month: u32 = trimmed.get(5..7)?.parse().ok()?;
    let day: u32 = trimmed.get(8..10)?.parse().ok()?;
    if !(1500..=2500).contains(&year)
        || !(1..=12).contains(&month)
        || day < 1
        || day > days_in_month(year, month)
    {
        return None;
    }
    let base_days = days_from_civil(year, month, day);
    let dow = base_days.rem_euclid(7) as f64;
    if bytes.len() == 10 {
        return Some((base_days as f64, month as f64, dow));
    }
    let sep = bytes[10];
    if !matches!(sep, b'T' | b't' | b' ') {
        return None;
    }
    let rest = trimmed.get(11..)?.trim();
    if rest.len() < 5 || rest.as_bytes()[2] != b':' {
        return None;
    }
    let hh: u32 = rest.get(0..2)?.parse().ok()?;
    let mm: u32 = rest.get(3..5)?.parse().ok()?;
    if hh > 23 || mm > 59 {
        return None;
    }
    let mut cursor = 5usize;
    let mut secs = (hh * 3600 + mm * 60) as f64;
    if cursor < rest.len() && rest.as_bytes()[cursor] == b':' {
        let ss: u32 = rest.get(cursor + 1..cursor + 3)?.parse().ok()?;
        if ss > 60 {
            return None;
        }
        secs += ss as f64;
        cursor += 3;
        if cursor < rest.len() && rest.as_bytes()[cursor] == b'.' {
            cursor += 1;
            let frac_start = cursor;
            while cursor < rest.len() && rest.as_bytes()[cursor].is_ascii_digit() {
                cursor += 1;
            }
            if cursor == frac_start {
                return None;
            }
            let frac_str = &rest[frac_start - 1..cursor];
            let frac: f64 = format!("0{frac_str}").parse().ok()?;
            secs += frac;
        }
    }
    let tz_part = rest.get(cursor..)?.trim();
    let tz_offset_secs: f64 = if tz_part.is_empty() || tz_part.eq_ignore_ascii_case("z") {
        0.0
    } else if tz_part.starts_with('+') || tz_part.starts_with('-') {
        let tz_sign = if tz_part.starts_with('-') { -1.0 } else { 1.0 };
        let tz_digits = &tz_part[1..];
        let (tzh, tzm): (u32, u32) = if let Some((h_s, m_s)) = tz_digits.split_once(':') {
            (h_s.parse().ok()?, m_s.parse().ok()?)
        } else if tz_digits.len() == 4 {
            (tz_digits[0..2].parse().ok()?, tz_digits[2..4].parse().ok()?)
        } else if tz_digits.len() == 2 {
            (tz_digits.parse().ok()?, 0)
        } else {
            return None;
        };
        if tzh > 23 || tzm > 59 {
            return None;
        }
        tz_sign * ((tzh * 3600 + tzm * 60) as f64)
    } else {
        return None;
    };
    let epoch_days = (base_days as f64) + ((secs - tz_offset_secs) / 86_400.0);
    Some((epoch_days, month as f64, dow))
}

fn count_unquoted_delim(line: &str, delim: char) -> usize {
    let mut in_quotes = false;
    let mut count = 0usize;
    let mut chars = line.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '"' {
            if in_quotes && chars.peek() == Some(&'"') {
                let _ = chars.next();
            } else {
                in_quotes = !in_quotes;
            }
        } else if ch == delim && !in_quotes {
            count += 1;
        }
    }
    count
}

fn detect_delimiter(first_line: &str) -> char {
    let candidates = [
        (',', count_unquoted_delim(first_line, ',')),
        ('\t', count_unquoted_delim(first_line, '\t')),
        (';', count_unquoted_delim(first_line, ';')),
        ('|', count_unquoted_delim(first_line, '|')),
    ];
    candidates
        .into_iter()
        .max_by_key(|&(_, c)| c)
        .filter(|&(_, c)| c > 0)
        .map(|(d, _)| d)
        .unwrap_or(',')
}

fn parse_rfc4180_records(text: &str, delim: char) -> Result<Vec<Vec<String>>, String> {
    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut current_row: Vec<String> = Vec::new();
    let mut cell = String::new();
    let mut in_quotes = false;
    let mut chars = text.chars().peekable();

    while let Some(ch) = chars.next() {
        if in_quotes {
            if ch == '"' {
                if chars.peek() == Some(&'"') {
                    let _ = chars.next();
                    cell.push('"');
                } else {
                    in_quotes = false;
                }
            } else {
                cell.push(ch);
            }
        } else if ch == '"' && cell.trim().is_empty() {
            cell.clear();
            in_quotes = true;
        } else if ch == delim {
            current_row.push(std::mem::take(&mut cell));
        } else if ch == '\r' {
            if chars.peek() == Some(&'\n') {
                let _ = chars.next();
            }
            current_row.push(std::mem::take(&mut cell));
            if !(current_row.len() == 1 && current_row[0].trim().is_empty()) {
                rows.push(std::mem::take(&mut current_row));
            } else {
                current_row.clear();
            }
        } else if ch == '\n' {
            current_row.push(std::mem::take(&mut cell));
            if !(current_row.len() == 1 && current_row[0].trim().is_empty()) {
                rows.push(std::mem::take(&mut current_row));
            } else {
                current_row.clear();
            }
        } else {
            cell.push(ch);
        }
    }
    if in_quotes {
        return Err("unclosed double-quote in CSV file".to_owned());
    }
    if !cell.is_empty() || !current_row.is_empty() {
        current_row.push(cell);
        if !(current_row.len() == 1 && current_row[0].trim().is_empty()) {
            rows.push(current_row);
        }
    }
    Ok(rows)
}

fn decode_utf8_or_utf16_bytes(bytes: Vec<u8>, label: &str) -> Result<String, String> {
    if bytes.starts_with(&[0xFF, 0xFE]) {
        let payload = &bytes[2..];
        if !payload.len().is_multiple_of(2) {
            return Err(format!(
                "UTF-16LE file {label} has odd byte length ({})",
                bytes.len()
            ));
        }
        let units = payload
            .chunks_exact(2)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]));
        char::decode_utf16(units)
            .collect::<Result<String, _>>()
            .map_err(|e| format!("decode UTF-16LE {label}: {e}"))
    } else if bytes.starts_with(&[0xFE, 0xFF]) {
        let payload = &bytes[2..];
        if !payload.len().is_multiple_of(2) {
            return Err(format!(
                "UTF-16BE file {label} has odd byte length ({})",
                bytes.len()
            ));
        }
        let units = payload
            .chunks_exact(2)
            .map(|pair| u16::from_be_bytes([pair[0], pair[1]]));
        char::decode_utf16(units)
            .collect::<Result<String, _>>()
            .map_err(|e| format!("decode UTF-16BE {label}: {e}"))
    } else {
        String::from_utf8(bytes).map_err(|e| {
            let offset = e.utf8_error().valid_up_to();
            format!(
                "file {label} contains invalid UTF-8 at byte offset {offset} (supported encodings: UTF-8, UTF-16LE BOM, UTF-16BE BOM)"
            )
        })
    }
}

fn decode_bounded_table_stream<R: Read>(
    reader: R,
    cap: u64,
    label: &str,
    stem: String,
) -> Result<(String, String), String> {
    let bytes = crate::storage::cas::read_bounded(reader, cap, label)?;
    let decoded = decode_utf8_or_utf16_bytes(bytes, label)?;
    Ok((decoded, stem))
}

fn read_xz_table_bytes(path: &Path, label: &str) -> Result<Vec<u8>, String> {
    use std::process::{Command, Stdio};
    let file = fs::File::open(path).map_err(|e| format!("open {label}: {e}"))?;
    let mut child = Command::new("xz")
        .arg("-dc")
        .stdin(Stdio::from(file))
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("spawn xz -dc for {label}: {e}"))?;
    let read_res = match child.stdout.take() {
        Some(mut stdout) => {
            crate::storage::cas::read_bounded(&mut stdout, MAX_DECOMPRESSED_TABLE_BYTES, label)
        }
        None => Err("xz -dc missing stdout pipe".to_owned()),
    };
    let bytes = match read_res {
        Ok(b) => b,
        Err(err) => {
            let _ = child.kill();
            let _ = child.wait();
            return Err(err);
        }
    };
    let status = child
        .wait()
        .map_err(|e| format!("wait xz -dc for {label}: {e}"))?;
    if !status.success() {
        return Err(format!("decompress xz table {label} failed ({status})"));
    }
    Ok(bytes)
}

fn read_decoded_table_text(path: &Path) -> Result<(String, String), String> {
    let label = path.display().to_string();
    if path == Path::new("-") {
        let stdin = std::io::stdin();
        return decode_bounded_table_stream(
            stdin.lock(),
            MAX_STDIN_TABLE_BYTES,
            "stdin ('-')",
            "-".to_owned(),
        );
    }
    let file_name_lc = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if let Some(stem) = file_name_lc.strip_suffix(".gz") {
        let file = fs::File::open(path).map_err(|e| format!("open {label}: {e}"))?;
        let decoder = flate2::read::GzDecoder::new(file);
        return decode_bounded_table_stream(
            decoder,
            MAX_DECOMPRESSED_TABLE_BYTES,
            &label,
            stem.to_owned(),
        );
    }
    if let Some(stem) = file_name_lc.strip_suffix(".zst") {
        let file = fs::File::open(path).map_err(|e| format!("open {label}: {e}"))?;
        let decoder = ruzstd::decoding::StreamingDecoder::new(file)
            .map_err(|e| format!("init zstd decoder for {label}: {e}"))?;
        return decode_bounded_table_stream(
            decoder,
            MAX_DECOMPRESSED_TABLE_BYTES,
            &label,
            stem.to_owned(),
        );
    }
    if let Some(stem) = file_name_lc.strip_suffix(".xz") {
        let bytes = read_xz_table_bytes(path, &label)?;
        let decoded = decode_utf8_or_utf16_bytes(bytes, &label)?;
        return Ok((decoded, stem.to_owned()));
    }
    let file = fs::File::open(path).map_err(|e| format!("read {label}: {e}"))?;
    decode_bounded_table_stream(file, MAX_DECOMPRESSED_TABLE_BYTES, &label, file_name_lc)
}

fn parse_jsonl_str(
    raw: &str,
    source_label: &str,
    extra_na: &[String],
    inf_as_nan: bool,
) -> Result<CarbonRawTable, String> {
    let stripped = raw.strip_prefix('\u{feff}').unwrap_or(raw);
    let mut headers: Vec<String> = Vec::new();
    let mut header_idx: BTreeMap<String, usize> = BTreeMap::new();
    let mut raw_maps: Vec<(usize, serde_json::Map<String, serde_json::Value>)> = Vec::new();

    for (idx, line) in stripped.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let line_no = idx + 1;
        let val: serde_json::Value = serde_json::from_str(trimmed)
            .map_err(|e| format!("JSONL {source_label} line {line_no} invalid JSON: {e}"))?;
        let serde_json::Value::Object(map) = val else {
            return Err(format!(
                "JSONL {source_label} line {line_no} must be a flat JSON object (`{{...}}`)"
            ));
        };
        for key in map.keys() {
            if !header_idx.contains_key(key) {
                let pos = headers.len();
                headers.push(key.clone());
                header_idx.insert(key.clone(), pos);
            }
        }
        raw_maps.push((line_no, map));
    }

    if headers.is_empty() || raw_maps.is_empty() {
        return Err(format!("JSONL {source_label} contains zero JSON records"));
    }

    let mut rows = Vec::with_capacity(raw_maps.len());
    for (line_no, map) in raw_maps {
        let mut row = vec![None; headers.len()];
        for (k, v) in map {
            let col_idx = header_idx[&k];
            let cell = match v {
                serde_json::Value::Null => None,
                serde_json::Value::Bool(b) => Some(b.to_string()),
                serde_json::Value::Number(n) => Some(n.to_string()),
                serde_json::Value::String(s) => {
                    let t = s.trim();
                    if is_missing_token_with_extra(t, extra_na) {
                        None
                    } else if is_infinity_token(t) {
                        if inf_as_nan {
                            None
                        } else {
                            return Err(format!(
                                "JSONL {source_label} line {line_no} column '{k}' contains non-finite value '{t}'; clean upstream or pass --inf-as-nan to treat ±Inf as missing"
                            ));
                        }
                    } else {
                        Some(t.to_owned())
                    }
                }
                serde_json::Value::Array(_) | serde_json::Value::Object(_) => {
                    return Err(format!(
                        "JSONL {source_label} line {line_no} column '{k}' is a nested array/object; flatten nested fields before running Carbon"
                    ));
                }
            };
            row[col_idx] = cell;
        }
        rows.push(row);
    }

    Ok(CarbonRawTable {
        headers,
        rows,
        delimiter: ',',
        had_header: true,
        extra_na_tokens: extra_na.to_vec(),
        inf_as_nan,
    })
}

/// Read an RFC-4180 CSV/TSV/semicolon/JSONL file (or `.gz`/`.zst`/`stdin`) into a `CarbonRawTable`.
pub fn load_delimited_table(
    path: &Path,
    delimiter_override: Option<char>,
    has_header: bool,
) -> Result<CarbonRawTable, String> {
    load_delimited_table_with_options(path, delimiter_override, has_header, &[], false)
}

/// Read a tabular file with custom missing-value sentinels (`--na-values`) and `--inf-as-nan` policy.
pub fn load_delimited_table_with_options(
    path: &Path,
    delimiter_override: Option<char>,
    has_header: bool,
    extra_na_values: &[String],
    inf_as_nan: bool,
) -> Result<CarbonRawTable, String> {
    let (raw, effective_name) = read_decoded_table_text(path)?;
    let label = path.display().to_string();
    let looks_jsonl = effective_name.ends_with(".jsonl")
        || effective_name.ends_with(".ndjson")
        || (effective_name == "-"
            && delimiter_override.is_none()
            && raw
                .lines()
                .map(str::trim)
                .find(|l| !l.is_empty())
                .is_some_and(|l| l.starts_with('{') && l.ends_with('}')));
    if looks_jsonl {
        return parse_jsonl_str(&raw, &label, extra_na_values, inf_as_nan);
    }
    parse_delimited_str_with_options(
        &raw,
        &label,
        delimiter_override,
        has_header,
        extra_na_values,
        inf_as_nan,
    )
}

/// Parse delimited text into a normalized `CarbonRawTable`.
pub fn parse_delimited_str(
    raw: &str,
    source_label: &str,
    delimiter_override: Option<char>,
    has_header: bool,
) -> Result<CarbonRawTable, String> {
    parse_delimited_str_with_options(
        raw,
        source_label,
        delimiter_override,
        has_header,
        &[],
        false,
    )
}

/// Parse delimited text with custom `--na-values` and explicit `±Inf` handling.
pub fn parse_delimited_str_with_options(
    raw: &str,
    source_label: &str,
    delimiter_override: Option<char>,
    has_header: bool,
    extra_na_values: &[String],
    inf_as_nan: bool,
) -> Result<CarbonRawTable, String> {
    let stripped = raw.strip_prefix('\u{feff}').unwrap_or(raw);
    let first_non_empty = stripped
        .lines()
        .map(|l| l.trim_matches(|c: char| c == '\u{feff}' || c == '\r'))
        .find(|l| !l.trim().is_empty())
        .ok_or_else(|| format!("CSV {source_label} is empty"))?;
    let delim = delimiter_override.unwrap_or_else(|| detect_delimiter(first_non_empty));
    let mut records = parse_rfc4180_records(stripped, delim)?;
    if records.is_empty() {
        return Err(format!("CSV {source_label} contains no records"));
    }

    let headers: Vec<String> = if has_header {
        let raw_hdr = records.remove(0);
        let mut seen = BTreeSet::new();
        let mut clean = Vec::with_capacity(raw_hdr.len());
        for (idx, h) in raw_hdr.into_iter().enumerate() {
            let base = h
                .trim()
                .trim_start_matches('\u{feff}')
                .trim_matches('"')
                .trim();
            let candidate = if base.is_empty() {
                format!("col_{idx}")
            } else {
                base.to_owned()
            };
            let unique = if seen.contains(&candidate) {
                let mut suffix = 2usize;
                loop {
                    let next = format!("{candidate}_{suffix}");
                    if !seen.contains(&next) {
                        break next;
                    }
                    suffix += 1;
                }
            } else {
                candidate
            };
            seen.insert(unique.clone());
            clean.push(unique);
        }
        clean
    } else {
        let width = records[0].len();
        (0..width).map(|i| format!("col_{i}")).collect()
    };

    if headers.is_empty() {
        return Err(format!("CSV {source_label} has zero columns"));
    }
    if records.is_empty() {
        return Err(format!(
            "CSV {source_label} has headers [{}] but zero data rows",
            headers.join(", ")
        ));
    }

    let expected_cols = headers.len();
    let mut norm_rows = Vec::with_capacity(records.len());
    for (r_idx, rec) in records.into_iter().enumerate() {
        let line_no = if has_header { r_idx + 2 } else { r_idx + 1 };
        if rec.len() != expected_cols {
            return Err(format!(
                "CSV {source_label} line {line_no} has {} columns, expected {expected_cols} (delimiter={:?})",
                rec.len(),
                delim
            ));
        }
        let mut row = Vec::with_capacity(expected_cols);
        for (c_idx, c) in rec.into_iter().enumerate() {
            let trimmed = c.trim();
            if is_missing_token_with_extra(trimmed, extra_na_values) {
                row.push(None);
            } else if is_infinity_token(trimmed) {
                if inf_as_nan {
                    row.push(None);
                } else {
                    return Err(format!(
                        "CSV {source_label} line {line_no} column '{}' contains non-finite value '{trimmed}'; clean upstream or pass --inf-as-nan to treat ±Inf as missing",
                        headers[c_idx]
                    ));
                }
            } else {
                row.push(Some(trimmed.to_owned()));
            }
        }
        norm_rows.push(row);
    }

    Ok(CarbonRawTable {
        headers,
        rows: norm_rows,
        delimiter: delim,
        had_header: has_header,
        extra_na_tokens: extra_na_values.to_vec(),
        inf_as_nan,
    })
}

fn levenshtein_distance(a: &str, b: &str) -> usize {
    let al: Vec<char> = a.to_ascii_lowercase().chars().collect();
    let bl: Vec<char> = b.to_ascii_lowercase().chars().collect();
    let mut prev: Vec<usize> = (0..=bl.len()).collect();
    let mut curr = vec![0usize; bl.len() + 1];
    for (i, &ca) in al.iter().enumerate() {
        curr[0] = i + 1;
        for (j, &cb) in bl.iter().enumerate() {
            let cost = usize::from(ca != cb);
            curr[j + 1] = (prev[j + 1] + 1).min(curr[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[bl.len()]
}

/// Resolve the target column name with smart defaults (`y`, `target`, `label`, `-1`/`last` on
/// `--no-header`) and fuzzy Levenshtein suggestions when a column name is misspelled.
pub fn resolve_target_column(
    headers: &[String],
    requested: Option<&str>,
    had_header: bool,
) -> Result<(usize, String, bool), String> {
    if headers.len() < 2 {
        return Err(
            "training CSV must contain at least 1 feature column and 1 target column".to_owned(),
        );
    }
    if let Some(req_raw) = requested {
        let req = req_raw.trim();
        if req == "-1" || req.eq_ignore_ascii_case("last") {
            let idx = headers.len() - 1;
            return Ok((idx, headers[idx].clone(), req != headers[idx]));
        }
        if let Some(idx) = headers.iter().position(|h| h == req) {
            return Ok((idx, headers[idx].clone(), false));
        }
        let ci_matches: Vec<usize> = headers
            .iter()
            .enumerate()
            .filter_map(|(i, h)| h.eq_ignore_ascii_case(req).then_some(i))
            .collect();
        if ci_matches.len() == 1 {
            let idx = ci_matches[0];
            return Ok((idx, headers[idx].clone(), true));
        }
        let fuzzy = headers
            .iter()
            .map(|h| (h, levenshtein_distance(req, h)))
            .min_by_key(|(_, d)| *d);
        let best = match fuzzy {
            Some((h, d)) if d <= 3 || d * 2 <= req.len().max(h.len()) => h.clone(),
            _ => headers
                .iter()
                .find(|h| matches!(h.to_ascii_lowercase().as_str(), "y" | "target" | "label"))
                .cloned()
                .unwrap_or_else(|| headers[headers.len() - 1].clone()),
        };
        return Err(format!(
            "target column '{req}' not found in CSV headers [{}]; did you mean '--target {best}'?",
            headers.join(", ")
        ));
    }

    if !had_header {
        let idx = headers.len() - 1;
        return Ok((idx, headers[idx].clone(), true));
    }

    for auto_name in ["y", "target", "label"] {
        if let Some(idx) = headers.iter().position(|h| h == auto_name) {
            return Ok((idx, headers[idx].clone(), auto_name != "y"));
        }
        let ci: Vec<usize> = headers
            .iter()
            .enumerate()
            .filter_map(|(i, h)| h.eq_ignore_ascii_case(auto_name).then_some(i))
            .collect();
        if ci.len() == 1 {
            let idx = ci[0];
            return Ok((idx, headers[idx].clone(), true));
        }
    }

    let last_col = &headers[headers.len() - 1];
    Err(format!(
        "target column 'y' not found in CSV headers [{}]; pass --target <COLUMN> (for example: --target {last_col})",
        headers.join(", ")
    ))
}

fn sorted_median(vals: &mut [f64]) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = vals.len() / 2;
    if vals.len() % 2 == 1 {
        vals[mid]
    } else {
        0.5 * (vals[mid - 1] + vals[mid])
    }
}

/// Condition & encode a `CarbonRawTable` into a zero-leakage `CarbonEncodedTrainingSet`
/// where all medians, category levels, and target/frequency statistics are learned strictly on `FIT`.
pub fn prepare_encoded_training_set(
    table: &CarbonRawTable,
    requested_target: Option<&str>,
    seed: u64,
    holdout_fraction: f64,
    max_encoded_features: usize,
) -> Result<CarbonEncodedTrainingSet, String> {
    prepare_encoded_training_set_with_options(
        table,
        &CarbonTrainingPrepOptions {
            requested_target,
            seed,
            holdout_fraction,
            max_encoded_features,
            objective_spec: None,
            split_mode: None,
            calibration_mode: None,
            reject_constant_target: false,
        },
    )
}

/// Condition & encode a `CarbonRawTable` with explicit split mode, calibration, custom objective
/// `aux_columns` separation, ISO-8601 date expansion, and pre-upload hygiene checks.
pub fn prepare_encoded_training_set_with_options(
    table: &CarbonRawTable,
    opts: &CarbonTrainingPrepOptions<'_>,
) -> Result<CarbonEncodedTrainingSet, String> {
    let (target_idx, target_col, _) =
        resolve_target_column(&table.headers, opts.requested_target, table.had_header)?;

    let normalized_split = match opts.split_mode.map(str::trim).filter(|s| !s.is_empty()) {
        None | Some("random") => None,
        Some("forward") => Some("forward".to_owned()),
        Some(other) => {
            return Err(format!(
                "unsupported --split mode '{other}' (expected 'random' or 'forward')"
            ));
        }
    };
    let normalized_cal = match opts
        .calibration_mode
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        None | Some("auto") => None,
        Some(m @ ("none" | "shift" | "affine")) => Some(m.to_owned()),
        Some(other) => {
            return Err(format!(
                "unsupported --calibration mode '{other}' (expected 'auto', 'none', 'shift', or 'affine')"
            ));
        }
    };

    // Resolve any custom-objective auxiliary columns (`col("<name>")`) so they stay out of `X` / `JLMB`
    // while remaining present in `fit.jmat` / `holdout.jmat` for fold-safe objective evaluation.
    let mut aux_indices: Vec<(usize, String)> = Vec::new();
    let mut aux_idx_set = BTreeSet::new();
    if let Some(spec) = opts.objective_spec {
        for aux_name in spec.columns() {
            let idx = table
                .headers
                .iter()
                .position(|h| h == &aux_name)
                .ok_or_else(|| {
                    format!(
                        "custom objective references auxiliary column '{aux_name}' which is missing from CSV headers [{}]",
                        table.headers.join(", ")
                    )
                })?;
            if idx == target_idx {
                return Err(format!(
                    "custom objective auxiliary column '{aux_name}' cannot be the target column '{target_col}'"
                ));
            }
            if aux_idx_set.insert(idx) {
                aux_indices.push((idx, aux_name));
            }
        }
    }

    let mut valid_rows: Vec<&Vec<Option<String>>> = Vec::with_capacity(table.rows.len());
    let mut valid_targets: Vec<f64> = Vec::with_capacity(table.rows.len());
    let mut valid_aux_rows: Vec<Vec<f64>> = Vec::with_capacity(table.rows.len());
    let mut dropped_target_rows = 0usize;

    for (r_idx, row) in table.rows.iter().enumerate() {
        let Some(y) = row[target_idx]
            .as_deref()
            .and_then(parse_flexible_numeric_f64)
        else {
            dropped_target_rows += 1;
            continue;
        };
        let mut aux_vals = Vec::with_capacity(aux_indices.len());
        for (a_idx, a_name) in &aux_indices {
            let line_no = if table.had_header {
                r_idx + 2
            } else {
                r_idx + 1
            };
            let val = row[*a_idx]
                .as_deref()
                .and_then(parse_flexible_numeric_f64)
                .ok_or_else(|| {
                    format!(
                        "line {line_no} auxiliary objective column '{a_name}' must contain a finite numeric value"
                    )
                })?;
            aux_vals.push(val);
        }
        valid_rows.push(row);
        valid_targets.push(y);
        valid_aux_rows.push(aux_vals);
    }

    if valid_rows.len() < 8 {
        return Err(format!(
            "need at least 8 rows with finite numeric target '{target_col}', got {} (dropped {dropped_target_rows} missing/non-numeric target rows)",
            valid_rows.len()
        ));
    }

    let total_rows = valid_rows.len();
    let split_plan = if normalized_split.as_deref() == Some("forward") {
        SplitPlan::chronological(total_rows, opts.seed, opts.holdout_fraction)
            .err_ctx("chronological split plan failed")?
    } else {
        SplitPlan::deterministic(total_rows, opts.seed, opts.holdout_fraction)
            .err_ctx("split plan failed")?
    };
    let fit_indices = split_plan.fit_rows();
    let fit_set: BTreeSet<usize> = fit_indices.iter().copied().collect();
    let n_fit = fit_indices.len().max(1);
    let global_fit_mean =
        fit_indices.iter().map(|&i| valid_targets[i]).sum::<f64>() / (n_fit as f64);
    let fit_y_var = fit_indices
        .iter()
        .map(|&i| {
            let d = valid_targets[i] - global_fit_mean;
            d * d
        })
        .sum::<f64>()
        / (n_fit as f64);
    if opts.reject_constant_target && fit_y_var.sqrt() <= 1e-12 {
        return Err(format!(
            "target column '{target_col}' is constant on the training split (value = {global_fit_mean}, std = 0); choose a non-constant `--target` column"
        ));
    }

    let raw_feature_indices: Vec<usize> = (0..table.headers.len())
        .filter(|&i| i != target_idx && !aux_idx_set.contains(&i))
        .collect();
    let raw_feature_columns: Vec<String> = raw_feature_indices
        .iter()
        .map(|&i| table.headers[i].clone())
        .collect();

    let mut transforms = Vec::new();
    let mut encoded_feature_names = Vec::new();
    let mut dropped_columns = Vec::new();
    let max_feats = opts.max_encoded_features.max(1);

    for &col_idx in &raw_feature_indices {
        let col_name = &table.headers[col_idx];
        let mut fit_present: Vec<(usize, &str)> = Vec::with_capacity(n_fit);
        for &r_i in fit_indices {
            if let Some(s) = valid_rows[r_i][col_idx].as_deref() {
                fit_present.push((r_i, s));
            }
        }
        if fit_present.is_empty() {
            dropped_columns.push((col_name.clone(), "all_missing_in_fit".to_owned()));
            continue;
        }
        if encoded_feature_names.len() >= max_feats {
            dropped_columns.push((col_name.clone(), "feature_cap_reached".to_owned()));
            continue;
        }

        let num_parsed_count = fit_present
            .iter()
            .filter(|&&(_, s)| parse_flexible_numeric_f64(s).is_some())
            .count();
        let mostly_numeric = num_parsed_count == fit_present.len()
            || (fit_present.len() >= 20 && num_parsed_count * 20 >= fit_present.len() * 19);

        if mostly_numeric {
            let ordered_nums: Vec<(usize, f64)> = fit_present
                .iter()
                .filter_map(|&(r_i, s)| parse_flexible_numeric_f64(s).map(|v| (r_i, v)))
                .collect();
            let missing_in_fit = n_fit.saturating_sub(ordered_nums.len());
            if raw_feature_indices.len() > 1 && missing_in_fit == 0 && !ordered_nums.is_empty() {
                let first_v = ordered_nums[0].1;
                if ordered_nums
                    .iter()
                    .all(|&(_, v)| (v - first_v).abs() <= 1e-15)
                {
                    dropped_columns.push((col_name.clone(), "constant_numeric".to_owned()));
                    continue;
                }
                let lower_col = col_name.to_ascii_lowercase();
                let looks_like_id_name = matches!(lower_col.as_str(), "id" | "row_id" | "index")
                    || lower_col.ends_with("_id");
                if looks_like_id_name
                    && n_fit >= 20
                    && ordered_nums.iter().all(|&(_, v)| v.fract() == 0.0)
                {
                    let mut by_row = ordered_nums.clone();
                    by_row.sort_by_key(|&(r_i, _)| r_i);
                    let strictly_increasing = by_row.windows(2).all(|w| w[1].1 > w[0].1);
                    if strictly_increasing {
                        dropped_columns.push((col_name.clone(), "monotonic_integer_id".to_owned()));
                        continue;
                    }
                }
            }
            let mut nums: Vec<f64> = ordered_nums.into_iter().map(|(_, v)| v).collect();
            let median = sorted_median(&mut nums);
            let emit_na = missing_in_fit > 0 && encoded_feature_names.len() + 2 <= max_feats;
            encoded_feature_names.push(col_name.clone());
            if emit_na {
                encoded_feature_names.push(format!("{col_name}__isna"));
            }
            transforms.push(CarbonFeatureTransform::Numeric {
                source_column: col_name.clone(),
                median,
                emit_missing_indicator: emit_na,
            });
            continue;
        }

        let all_bool = fit_present
            .iter()
            .all(|&(_, s)| parse_boolean_token(s).is_some());
        if all_bool {
            let ones = fit_present
                .iter()
                .filter(|&&(_, s)| parse_boolean_token(s) == Some(1.0))
                .count();
            let mode = if ones * 2 >= fit_present.len() {
                1.0
            } else {
                0.0
            };
            encoded_feature_names.push(col_name.clone());
            transforms.push(CarbonFeatureTransform::Boolean {
                source_column: col_name.clone(),
                mode,
            });
            continue;
        }

        let date_parsed: Vec<f64> = fit_present
            .iter()
            .filter_map(|&(_, s)| parse_iso8601_date(s).map(|(d, _, _)| d))
            .collect();
        let mostly_iso_date = date_parsed.len() == fit_present.len()
            || (fit_present.len() >= 10 && date_parsed.len() * 20 >= fit_present.len() * 19);
        if mostly_iso_date && !date_parsed.is_empty() {
            let mut days_sorted = date_parsed;
            let med_days = sorted_median(&mut days_sorted);
            let missing_in_fit = n_fit.saturating_sub(days_sorted.len());
            let emit_parts = encoded_feature_names.len() + 3 <= max_feats;
            let needed_after_parts = if emit_parts { 3 } else { 1 };
            let emit_na =
                missing_in_fit > 0 && encoded_feature_names.len() + needed_after_parts < max_feats;
            encoded_feature_names.push(format!("{col_name}__days"));
            if emit_parts {
                encoded_feature_names.push(format!("{col_name}__month"));
                encoded_feature_names.push(format!("{col_name}__dow"));
            }
            if emit_na {
                encoded_feature_names.push(format!("{col_name}__isna"));
            }
            transforms.push(CarbonFeatureTransform::DateIso {
                source_column: col_name.clone(),
                median_epoch_days: med_days,
                emit_calendar_parts: emit_parts,
                emit_missing_indicator: emit_na,
            });
            continue;
        }

        let mut cat_stats: BTreeMap<String, (usize, f64)> = BTreeMap::new();
        for &(r_i, s) in &fit_present {
            let entry = cat_stats.entry(s.to_owned()).or_insert((0, 0.0));
            entry.0 += 1;
            entry.1 += valid_targets[r_i];
        }
        let distinct = cat_stats.len();
        if n_fit >= 20 && distinct * 20 >= fit_present.len() * 19 {
            dropped_columns.push((col_name.clone(), "unique_string_id".to_owned()));
            continue;
        }
        if distinct <= 1 && fit_present.len() == n_fit {
            dropped_columns.push((col_name.clone(), "constant_categorical".to_owned()));
            continue;
        }

        if distinct <= 6 && encoded_feature_names.len() + distinct <= max_feats {
            let categories: Vec<String> = cat_stats.keys().cloned().collect();
            for cat in &categories {
                let safe_cat: String = cat
                    .chars()
                    .map(|c| {
                        if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                            c
                        } else {
                            '_'
                        }
                    })
                    .take(24)
                    .collect();
                encoded_feature_names.push(format!("{col_name}__{safe_cat}"));
            }
            transforms.push(CarbonFeatureTransform::OneHot {
                source_column: col_name.clone(),
                categories,
            });
        } else {
            encoded_feature_names.push(format!("{col_name}__te"));
            let emit_freq = encoded_feature_names.len() < max_feats;
            if emit_freq {
                encoded_feature_names.push(format!("{col_name}__freq"));
            }
            transforms.push(CarbonFeatureTransform::TargetFreq {
                source_column: col_name.clone(),
                global_mean: global_fit_mean,
                smoothing: 10.0,
                fit_count: if emit_freq { n_fit } else { 0 },
                category_stats: cat_stats,
            });
        }
    }

    if encoded_feature_names.is_empty() {
        return Err(
            "zero usable feature columns remained after removing target, auxiliary, empty, and string-ID columns"
                .to_owned(),
        );
    }

    let col_lookup: BTreeMap<&str, usize> = table
        .headers
        .iter()
        .enumerate()
        .map(|(i, h)| (h.as_str(), i))
        .collect();

    let n_enc = encoded_feature_names.len();
    let n_aux = aux_indices.len();
    let mut full_x_f32 = Vec::with_capacity(total_rows * n_enc);
    let mut full_y_f32 = Vec::with_capacity(total_rows);
    let mut jmat_numeric_rows: Vec<Vec<f64>> = Vec::with_capacity(total_rows);

    for (r_i, row) in valid_rows.iter().enumerate() {
        let is_fit_row = fit_set.contains(&r_i);
        let y_val = valid_targets[r_i];
        let enc_row = encode_single_row_internal(
            row,
            &col_lookup,
            &transforms,
            if is_fit_row { Some(y_val) } else { None },
        );
        for &v in &enc_row {
            full_x_f32.push(v as f32);
        }
        full_y_f32.push(y_val as f32);
        let mut jmat_row = enc_row;
        jmat_row.reserve(n_aux + 1);
        jmat_row.extend_from_slice(&valid_aux_rows[r_i]);
        jmat_row.push(y_val);
        jmat_numeric_rows.push(jmat_row);
    }

    let landmark_bundle = FitLandmarkBundle::extract_fit_only(
        &full_x_f32,
        total_rows,
        n_enc,
        &full_y_f32,
        &split_plan,
    )
    .err_ctx("FIT-only landmark conditioning failed")?;
    drop(full_x_f32);
    drop(full_y_f32);

    let aux_column_names: Vec<String> = aux_indices.iter().map(|(_, n)| n.clone()).collect();
    let mut jmat_headers = encoded_feature_names.clone();
    jmat_headers.extend(aux_column_names.iter().cloned());
    jmat_headers.push(target_col.clone());
    let artifacts =
        split_and_encode_numeric_jmat_v1(&jmat_headers, &jmat_numeric_rows, &split_plan)
            .err_ctx("JMAT v1 split encoding failed")?;

    let mut fit_x = Vec::with_capacity(split_plan.fit_rows().len());
    let mut fit_y = Vec::with_capacity(split_plan.fit_rows().len());
    let mut fit_aux_cols: BTreeMap<String, Vec<f64>> = aux_column_names
        .iter()
        .map(|n| (n.clone(), Vec::with_capacity(split_plan.fit_rows().len())))
        .collect();
    for &idx in split_plan.fit_rows() {
        fit_x.push(jmat_numeric_rows[idx][..n_enc].to_vec());
        fit_y.push(valid_targets[idx]);
        for (a_pos, a_name) in aux_column_names.iter().enumerate() {
            if let Some(vec) = fit_aux_cols.get_mut(a_name) {
                vec.push(valid_aux_rows[idx][a_pos]);
            }
        }
    }

    let mut holdout_x = Vec::with_capacity(split_plan.holdout_rows().len());
    let mut holdout_y = Vec::with_capacity(split_plan.holdout_rows().len());
    let mut holdout_aux_cols: BTreeMap<String, Vec<f64>> = aux_column_names
        .iter()
        .map(|n| {
            (
                n.clone(),
                Vec::with_capacity(split_plan.holdout_rows().len()),
            )
        })
        .collect();
    for &idx in split_plan.holdout_rows() {
        holdout_x.push(jmat_numeric_rows[idx][..n_enc].to_vec());
        holdout_y.push(valid_targets[idx]);
        for (a_pos, a_name) in aux_column_names.iter().enumerate() {
            if let Some(vec) = holdout_aux_cols.get_mut(a_name) {
                vec.push(valid_aux_rows[idx][a_pos]);
            }
        }
    }
    drop(jmat_numeric_rows);

    // Pre-upload target-leakage & duplicate-header hygiene diagnostics on `FIT`.
    let mut hygiene_warnings = Vec::new();
    if fit_x.len() >= 12 && fit_y_var > 1e-18 {
        let denom_y = (fit_y_var * (fit_x.len() as f64)).sqrt();
        for (f_idx, f_name) in encoded_feature_names.iter().enumerate() {
            let mean_x = fit_x.iter().map(|r| r[f_idx]).sum::<f64>() / (fit_x.len() as f64);
            let mut cov = 0.0_f64;
            let mut sxx = 0.0_f64;
            for (r, &yv) in fit_x.iter().zip(fit_y.iter()) {
                let dx = r[f_idx] - mean_x;
                let dy = yv - global_fit_mean;
                cov += dx * dy;
                sxx += dx * dx;
            }
            if sxx > 1e-18 {
                let r_abs = (cov / (sxx.sqrt() * denom_y)).abs();
                if r_abs > 0.9999 {
                    hygiene_warnings.push(format!(
                        "Potential target leakage: feature '{f_name}' has |Pearson r| = {r_abs:.5} with target '{target_col}' on FIT."
                    ));
                }
            }
        }
    }

    let prepared = CarbonClientPreparedDataset {
        fit_rows: split_plan.fit_rows().len() as u32,
        holdout_rows: split_plan.holdout_rows().len() as u32,
        split_plan,
        landmark_bundle,
        artifacts,
        feature_names: encoded_feature_names.clone(),
        target_name: target_col.clone(),
        aux_column_names,
        split_mode: normalized_split,
        calibration_mode: normalized_cal,
    };

    let encoder = CarbonTabularEncoder {
        raw_feature_columns,
        dropped_columns,
        transforms,
        encoded_feature_names,
        target_column: target_col,
        delimiter: table.delimiter,
        had_header: table.had_header,
        extra_na_tokens: table.extra_na_tokens.clone(),
        inf_as_nan: table.inf_as_nan,
    };

    Ok(CarbonEncodedTrainingSet {
        prepared,
        encoder,
        total_valid_rows: total_rows,
        dropped_target_rows,
        fit_x,
        fit_y,
        holdout_x,
        holdout_y,
        fit_aux_cols,
        holdout_aux_cols,
        hygiene_warnings,
    })
}

fn encode_single_row_internal(
    row: &[Option<String>],
    col_lookup: &BTreeMap<&str, usize>,
    transforms: &[CarbonFeatureTransform],
    loo_fit_y: Option<f64>,
) -> Vec<f64> {
    let mut out = Vec::with_capacity(transforms.len() * 2);
    for tf in transforms {
        match tf {
            CarbonFeatureTransform::Numeric {
                source_column,
                median,
                emit_missing_indicator,
            } => {
                let cell = col_lookup
                    .get(source_column.as_str())
                    .and_then(|&idx| row.get(idx))
                    .and_then(|v| v.as_deref())
                    .and_then(parse_flexible_numeric_f64);
                match cell {
                    Some(v) => {
                        out.push(v);
                        if *emit_missing_indicator {
                            out.push(0.0);
                        }
                    }
                    None => {
                        out.push(*median);
                        if *emit_missing_indicator {
                            out.push(1.0);
                        }
                    }
                }
            }
            CarbonFeatureTransform::Boolean {
                source_column,
                mode,
            } => {
                let val = col_lookup
                    .get(source_column.as_str())
                    .and_then(|&idx| row.get(idx))
                    .and_then(|v| v.as_deref())
                    .and_then(|s| parse_boolean_token(s).or_else(|| parse_flexible_numeric_f64(s)))
                    .unwrap_or(*mode);
                out.push(val);
            }
            CarbonFeatureTransform::DateIso {
                source_column,
                median_epoch_days,
                emit_calendar_parts,
                emit_missing_indicator,
            } => {
                let parsed = col_lookup
                    .get(source_column.as_str())
                    .and_then(|&idx| row.get(idx))
                    .and_then(|v| v.as_deref())
                    .and_then(parse_iso8601_date);
                match parsed {
                    Some((days, month, dow)) => {
                        out.push(days - *median_epoch_days);
                        if *emit_calendar_parts {
                            out.push(month);
                            out.push(dow);
                        }
                        if *emit_missing_indicator {
                            out.push(0.0);
                        }
                    }
                    None => {
                        out.push(0.0);
                        if *emit_calendar_parts {
                            out.push(6.0);
                            out.push(3.0);
                        }
                        if *emit_missing_indicator {
                            out.push(1.0);
                        }
                    }
                }
            }
            CarbonFeatureTransform::OneHot {
                source_column,
                categories,
            } => {
                let raw = col_lookup
                    .get(source_column.as_str())
                    .and_then(|&idx| row.get(idx))
                    .and_then(|v| v.as_deref());
                for cat in categories {
                    out.push(if raw == Some(cat.as_str()) { 1.0 } else { 0.0 });
                }
            }
            CarbonFeatureTransform::TargetFreq {
                source_column,
                global_mean,
                smoothing,
                fit_count,
                category_stats,
            } => {
                let raw = col_lookup
                    .get(source_column.as_str())
                    .and_then(|&idx| row.get(idx))
                    .and_then(|v| v.as_deref());
                let (te, freq) = match raw.and_then(|k| category_stats.get(k)) {
                    Some(&(cnt, sum_y)) => {
                        let (eff_cnt, eff_sum) = match loo_fit_y {
                            Some(y_self) if cnt > 1 => ((cnt - 1) as f64, sum_y - y_self),
                            Some(_) => (0.0, 0.0),
                            None => (cnt as f64, sum_y),
                        };
                        let te_val = (eff_sum + smoothing * global_mean) / (eff_cnt + smoothing);
                        let fq = if *fit_count > 0 {
                            (cnt as f64) / (*fit_count as f64)
                        } else {
                            0.0
                        };
                        (te_val, fq)
                    }
                    None => (*global_mean, 0.0),
                };
                out.push(te);
                if *fit_count > 0 {
                    out.push(freq);
                }
            }
        }
    }
    out
}

impl CarbonTabularEncoder {
    /// Encode an unlabeled (or labeled) prediction table using the exact `FIT`-learned schema.
    /// Automatically ignores any target column, auxiliary objective columns, or extra ID columns.
    pub fn encode_prediction_table(
        &self,
        table: &CarbonRawTable,
    ) -> Result<Vec<Vec<Option<f64>>>, String> {
        let mut col_lookup: BTreeMap<&str, usize> = BTreeMap::new();
        for (idx, h) in table.headers.iter().enumerate() {
            col_lookup.insert(h.as_str(), idx);
        }

        let positional_headers: Vec<String>;
        if !table.had_header && !self.had_header {
            positional_headers = self.raw_feature_columns.clone();
            col_lookup.clear();
            for (pos, raw_col) in positional_headers.iter().enumerate() {
                if pos < table.headers.len() {
                    col_lookup.insert(raw_col.as_str(), pos);
                }
            }
        } else {
            let mut missing_required = Vec::new();
            for tf in &self.transforms {
                let src = match tf {
                    CarbonFeatureTransform::Numeric { source_column, .. }
                    | CarbonFeatureTransform::Boolean { source_column, .. }
                    | CarbonFeatureTransform::DateIso { source_column, .. }
                    | CarbonFeatureTransform::OneHot { source_column, .. }
                    | CarbonFeatureTransform::TargetFreq { source_column, .. } => source_column,
                };
                if !col_lookup.contains_key(src.as_str()) {
                    missing_required.push(src.clone());
                }
            }
            if !missing_required.is_empty() {
                return Err(format!(
                    "prediction CSV is missing required feature column(s) [{}]; available columns: [{}]",
                    missing_required.join(", "),
                    table.headers.join(", ")
                ));
            }
        }

        let mut encoded = Vec::with_capacity(table.rows.len());
        for row in &table.rows {
            let vals = encode_single_row_internal(row, &col_lookup, &self.transforms, None);
            encoded.push(vals.into_iter().map(Some).collect());
        }
        Ok(encoded)
    }
}

/// Run a zero-network local preflight inspection on a CSV/TSV/JSONL dataset (`carbon inspect`).
pub fn inspect_tabular_file(
    path: &Path,
    requested_target: Option<&str>,
    delimiter_override: Option<char>,
    has_header: bool,
) -> Result<CarbonTabularInspectReport, String> {
    inspect_tabular_file_with_tier(
        path,
        requested_target,
        delimiter_override,
        has_header,
        CarbonProductTier::Pro,
    )
}

/// Run a zero-network local preflight inspection evaluated against a specific `CarbonProductTier`
/// (`--tier free|pro|enterprise`).
pub fn inspect_tabular_file_with_tier(
    path: &Path,
    requested_target: Option<&str>,
    delimiter_override: Option<char>,
    has_header: bool,
    requested_tier: CarbonProductTier,
) -> Result<CarbonTabularInspectReport, String> {
    let table = load_delimited_table(path, delimiter_override, has_header)?;
    let mut inferred_preview_target: Option<String> = None;
    let effective_target: Option<String> = match requested_target {
        Some(req) => Some(req.to_owned()),
        None => match resolve_target_column(&table.headers, None, table.had_header) {
            Ok((_, col, _)) => Some(col),
            Err(orig_err) => {
                let rightmost_numeric = (0..table.headers.len()).rev().find(|&c_idx| {
                    let mut present = 0usize;
                    let mut numeric = 0usize;
                    for r in &table.rows {
                        if let Some(s) = r[c_idx].as_deref() {
                            present += 1;
                            if parse_flexible_numeric_f64(s).is_some() {
                                numeric += 1;
                            }
                        }
                    }
                    present > 0 && present == numeric
                });
                match rightmost_numeric {
                    Some(idx) => {
                        let col = table.headers[idx].clone();
                        inferred_preview_target = Some(col.clone());
                        Some(col)
                    }
                    None => return Err(orig_err),
                }
            }
        },
    };
    let max_feats = requested_tier.max_features().unwrap_or(512).min(256);
    let encoded =
        prepare_encoded_training_set(&table, effective_target.as_deref(), 42, 0.20, max_feats)?;
    let (target_idx, target_col, _) = resolve_target_column(
        &table.headers,
        effective_target.as_deref(),
        table.had_header,
    )?;

    let mut y_vals: Vec<f64> = table
        .rows
        .iter()
        .filter_map(|r| {
            r[target_idx]
                .as_deref()
                .and_then(parse_flexible_numeric_f64)
        })
        .collect();
    let n_y = y_vals.len().max(1) as f64;
    let y_mean = y_vals.iter().sum::<f64>() / n_y;
    let y_var = y_vals
        .iter()
        .map(|&y| {
            let d = y - y_mean;
            d * d
        })
        .sum::<f64>()
        / n_y;
    let y_std = y_var.sqrt();
    let y_med = sorted_median(&mut y_vals);
    let y_min = y_vals.first().copied().unwrap_or(0.0);
    let y_max = y_vals.last().copied().unwrap_or(0.0);
    let outliers_3sigma = if y_std > 1e-12 {
        y_vals
            .iter()
            .filter(|&&y| (y - y_mean).abs() > 3.0 * y_std)
            .count()
    } else {
        0
    };

    let dropped_map: BTreeMap<&str, &str> = encoded
        .encoder
        .dropped_columns
        .iter()
        .map(|(c, r)| (c.as_str(), r.as_str()))
        .collect();

    let mut columns = Vec::with_capacity(table.headers.len());
    for (c_idx, name) in table.headers.iter().enumerate() {
        let present: Vec<&str> = table
            .rows
            .iter()
            .filter_map(|r| r[c_idx].as_deref())
            .collect();
        let missing_count = table.rows.len().saturating_sub(present.len());
        let missing_pct = 100.0 * (missing_count as f64) / (table.rows.len().max(1) as f64);
        let distinct_set: BTreeSet<&str> = present.iter().copied().collect();
        let all_num = !present.is_empty()
            && present
                .iter()
                .all(|s| parse_flexible_numeric_f64(s).is_some());
        let all_date = !present.is_empty()
            && !all_num
            && present.iter().all(|s| parse_iso8601_date(s).is_some());

        let (num_min, num_med, num_mean, num_std, num_max, sample_cats) = if all_num {
            let mut v: Vec<f64> = present
                .iter()
                .filter_map(|s| parse_flexible_numeric_f64(s))
                .collect();
            let nv = v.len().max(1) as f64;
            let mn = v.iter().sum::<f64>() / nv;
            let sd = (v.iter().map(|&x| (x - mn) * (x - mn)).sum::<f64>() / nv).sqrt();
            let med = sorted_median(&mut v);
            (
                v.first().copied(),
                Some(med),
                Some(mn),
                Some(sd),
                v.last().copied(),
                Vec::new(),
            )
        } else {
            let samples: Vec<String> = distinct_set
                .iter()
                .take(5)
                .map(|s| (*s).to_owned())
                .collect();
            (None, None, None, None, None, samples)
        };

        let role = if c_idx == target_idx {
            "target".to_owned()
        } else if let Some(reason) = dropped_map.get(name.as_str()) {
            format!("dropped ({reason})")
        } else {
            "feature".to_owned()
        };
        let detected_type = if present.is_empty() {
            "empty".to_owned()
        } else if all_num {
            "numeric".to_owned()
        } else if present.iter().all(|s| parse_boolean_token(s).is_some()) {
            "boolean".to_owned()
        } else if all_date {
            "datetime_iso8601".to_owned()
        } else {
            "categorical".to_owned()
        };

        columns.push(CarbonColumnProfile {
            name: name.clone(),
            role,
            detected_type,
            missing_count,
            missing_pct,
            distinct_count: distinct_set.len(),
            numeric_min: num_min,
            numeric_median: num_med,
            numeric_mean: num_mean,
            numeric_std: num_std,
            numeric_max: num_max,
            sample_categories: sample_cats,
        });
    }

    let mut warnings = Vec::new();
    if let Some(inf_col) = inferred_preview_target {
        warnings.push(format!(
            "Auto-previewed rightmost numeric column '{inf_col}' as candidate target; pass `--target {inf_col}` when fitting."
        ));
    }
    if encoded.dropped_target_rows > 0 {
        warnings.push(format!(
            "{} row(s) have missing/non-numeric target '{}' and will be excluded from training.",
            encoded.dropped_target_rows, target_col
        ));
    }
    if y_std <= 1e-12 {
        warnings.push(format!(
            "Target column '{target_col}' is constant (std = 0)."
        ));
    }
    if outliers_3sigma > 0 {
        warnings.push(format!(
            "Target column '{target_col}' has {outliers_3sigma} extreme outlier(s) (> 3 sigma); consider `--objective mae` or target inspection."
        ));
    }
    for (col, reason) in &encoded.encoder.dropped_columns {
        warnings.push(format!("Column '{col}' automatically excluded ({reason})."));
    }
    warnings.extend(encoded.hygiene_warnings);

    let fit_rows = encoded.prepared.fit_rows as usize;
    let feat_cnt = encoded.encoder.encoded_feature_names.len();
    if let Err(tier_err) = requested_tier.validate_admission(
        encoded.total_valid_rows,
        feat_cnt,
        CarbonEffortTier::Fast,
    ) {
        warnings.push(format!(
            "Selected --tier {} limit exceeded: {tier_err} (use `--tier pro` or `carbon serve` Enterprise).",
            requested_tier.as_str()
        ));
    }
    let est_ms = estimate_ms_per_candidate_1vcpu(fit_rows, feat_cnt);

    let mut effort_previews = Vec::new();
    let preview_specs: Vec<(CarbonProductTier, CarbonEffortTier)> = match requested_tier {
        CarbonProductTier::Free => vec![
            (CarbonProductTier::Free, CarbonEffortTier::Fast),
            (CarbonProductTier::Free, CarbonEffortTier::Medium),
            (CarbonProductTier::Pro, CarbonEffortTier::Fast),
            (CarbonProductTier::Pro, CarbonEffortTier::Medium),
        ],
        CarbonProductTier::Pro => vec![
            (CarbonProductTier::Free, CarbonEffortTier::Fast),
            (CarbonProductTier::Pro, CarbonEffortTier::Fast),
            (CarbonProductTier::Pro, CarbonEffortTier::Medium),
            (CarbonProductTier::Pro, CarbonEffortTier::High),
            (CarbonProductTier::Pro, CarbonEffortTier::Ultra),
        ],
        CarbonProductTier::Enterprise => vec![
            (CarbonProductTier::Enterprise, CarbonEffortTier::Fast),
            (CarbonProductTier::Enterprise, CarbonEffortTier::Medium),
            (CarbonProductTier::Enterprise, CarbonEffortTier::High),
            (CarbonProductTier::Enterprise, CarbonEffortTier::Ultra),
        ],
    };
    for (ptier, eff) in preview_specs {
        if let Ok(plan) = plan_carbon_execution(&CarbonPlanInput {
            product_tier: ptier,
            effort: eff,
            total_rows: encoded.total_valid_rows,
            fit_rows,
            features: feat_cnt,
            est_ms_per_candidate_1vcpu: est_ms,
        }) {
            let wall_ms = match plan.mode {
                CarbonExecutionMode::SoloSingleLambda => plan.solo_wall_ms,
                CarbonExecutionMode::HybridFanOut => plan.hybrid_wall_ms,
            };
            effort_previews.push(CarbonEffortCostPreview {
                tier: ptier.as_str().to_owned(),
                effort: eff.as_str().to_owned(),
                execution_mode: plan.mode,
                eval_shards: plan.shards_per_wave,
                planned_waves: plan.planned_waves,
                estimated_wall_secs: wall_ms / 1_000.0,
                estimated_usd: plan.full_dataset_cost_estimate.total_estimated_usd,
                runs_per_1_usd: plan.full_dataset_cost_estimate.runs_per_1_usd,
            });
        }
    }

    let recommended_effort = if encoded.total_valid_rows <= 500 {
        "medium".to_owned()
    } else if encoded.total_valid_rows <= 5_000 {
        "fast (iterative) or high (production)".to_owned()
    } else {
        "pro --effort medium".to_owned()
    };

    Ok(CarbonTabularInspectReport {
        file_path: path.display().to_string(),
        delimiter: match table.delimiter {
            '\t' => "\\t".to_owned(),
            other => other.to_string(),
        },
        has_header: table.had_header,
        raw_rows: table.rows.len(),
        valid_target_rows: encoded.total_valid_rows,
        dropped_missing_target_rows: encoded.dropped_target_rows,
        raw_columns: table.headers.len(),
        encoded_feature_count: feat_cnt,
        target_column: target_col,
        target_min: y_min,
        target_median: y_med,
        target_mean: y_mean,
        target_std: y_std,
        target_max: y_max,
        target_outliers_3sigma: outliers_3sigma,
        recommended_effort,
        warnings,
        dropped_columns: encoded.encoder.dropped_columns,
        columns,
        effort_previews,
    })
}
