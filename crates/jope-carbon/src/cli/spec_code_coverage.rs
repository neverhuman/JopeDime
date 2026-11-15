//! Compiler-instrumented (`rustc -C instrument-coverage` + `llvm-profdata` + `llvm-cov export`)
//! Before-vs-After analyzer and Section 14 HTML/SVG renderer for `carbon spec-report` and
//! cryptographic verifier for `carbon spec-verify`.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::cli::spec_code_coverage_rows::{
    build_tier_env_rows, resolve_anchor_span, SurfaceCoverageSpec, CLI_COMMAND_ROWS, HTTP_API_ROWS,
};

/// Canonical distilled receipt filename written into `target/carbon-spec/`.
pub const CODE_COVERAGE_SUMMARY_FILENAME: &str = "code-coverage-summary.json";

const BASELINE_JSON: &str =
    include_str!("../../../../docs/carbon/code-coverage-baseline-0ff23cc6.json");

/// Per-file or crate-wide coverage counters (`lines`, `regions`, `functions`).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CoverageCounters {
    pub lines_covered: u64,
    pub lines_total: u64,
    pub lines_pct: f64,
    pub regions_covered: u64,
    pub regions_total: u64,
    pub regions_pct: f64,
    pub functions_covered: u64,
    pub functions_total: u64,
    pub functions_pct: f64,
}

/// Per-file summary entry inside `code-coverage-summary.json`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FileCoverageDelta {
    pub file: String,
    #[serde(default)]
    pub before_functions_pct: f64,
    #[serde(default)]
    pub after_functions_covered: u64,
    #[serde(default)]
    pub after_functions_total: u64,
    #[serde(default)]
    pub after_functions_pct: f64,
    pub before_lines_pct: f64,
    pub after_lines_covered: u64,
    pub after_lines_total: u64,
    pub after_lines_pct: f64,
    pub before_regions_pct: f64,
    pub after_regions_covered: u64,
    pub after_regions_total: u64,
    pub after_regions_pct: f64,
}

/// Measured line and MIR branch-region coverage for one CLI command or HTTP API route.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvaluatedSurfaceRow {
    pub id: String,
    pub surface: String,
    pub bound_test: String,
    pub status_badge: String,
    pub source_span: String,
    pub lines_covered: u64,
    pub lines_total: u64,
    pub lines_pct: f64,
    pub regions_covered: u64,
    pub regions_total: u64,
    pub regions_pct: f64,
}

/// Measured line and MIR branch-region coverage for one `tier x env` cell.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvaluatedTierEnvRow {
    pub cell_id: String,
    pub environment: String,
    pub auth_tier: String,
    pub effort: String,
    pub expected_behavior: String,
    pub bound_test: String,
    pub status_badge: String,
    pub source_span: String,
    pub lines_covered: u64,
    pub lines_total: u64,
    pub lines_pct: f64,
    pub regions_covered: u64,
    pub regions_total: u64,
    pub regions_pct: f64,
}

/// Complete distilled coverage receipt bound by SHA-256 into `spec-pdf-receipt.json`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CodeCoverageSummaryReceipt {
    pub schema: String,
    pub source_sha: String,
    pub baseline_sha: String,
    pub before: CoverageCounters,
    pub after: CoverageCounters,
    pub files: Vec<FileCoverageDelta>,
    pub cli_rows: Vec<EvaluatedSurfaceRow>,
    pub api_rows: Vec<EvaluatedSurfaceRow>,
    pub tier_env_rows: Vec<EvaluatedTierEnvRow>,
}

#[derive(Deserialize)]
struct BaselineDoc {
    source_sha: String,
    lines_covered: u64,
    lines_total: u64,
    lines_pct: f64,
    regions_covered: u64,
    regions_total: u64,
    regions_pct: f64,
    functions_covered: u64,
    functions_total: u64,
    functions_pct: f64,
    files: Vec<BaselineFileEntry>,
}

#[derive(Deserialize)]
struct BaselineFileEntry {
    file: String,
    lines_pct: f64,
    regions_pct: f64,
    #[serde(default)]
    functions_pct: f64,
}

struct ParsedLlvmFile {
    summary: CoverageCounters,
    segments: Vec<[u64; 6]>,
}

fn round2(numer: u64, denom: u64) -> f64 {
    if denom == 0 {
        return 0.0;
    }
    ((numer as f64) * 10_000.0 / (denom as f64)).round() / 100.0
}

fn normalize_carbon_rel_path(filename: &str) -> Option<String> {
    let norm = filename.replace('\\', "/");
    let marker = "crates/jope-carbon/";
    let rel = if let Some(idx) = norm.find(marker) {
        &norm[idx + marker.len()..]
    } else if norm.starts_with("src/") {
        norm.as_str()
    } else {
        return None;
    };
    if rel.starts_with("src/") {
        Some(rel.to_owned())
    } else {
        None
    }
}

/// Parse either an `llvm-cov export` JSON file or a pre-distilled `CodeCoverageSummaryReceipt`.
pub fn load_and_evaluate_code_coverage(
    path: &Path,
    source_sha: &str,
) -> Result<CodeCoverageSummaryReceipt, String> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| format!("read code coverage JSON {}: {e}", path.display()))?;
    let val: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|e| format!("parse code coverage JSON {}: {e}", path.display()))?;
    if val.get("schema").and_then(|s| s.as_str())
        == Some("jopedime.carbon.code-coverage-summary.v1")
    {
        let mut rcpt: CodeCoverageSummaryReceipt = serde_json::from_value(val)
            .map_err(|e| format!("decode CodeCoverageSummaryReceipt: {e}"))?;
        if !source_sha.trim().is_empty() && rcpt.source_sha == "workspace" {
            rcpt.source_sha = source_sha.to_owned();
        }
        validate_code_coverage_receipt(&rcpt, Some(source_sha))?;
        return Ok(rcpt);
    }
    let rcpt = evaluate_llvm_cov_export(&val, source_sha)?;
    validate_code_coverage_receipt(&rcpt, Some(source_sha))?;
    Ok(rcpt)
}

fn evaluate_llvm_cov_export(
    val: &serde_json::Value,
    source_sha: &str,
) -> Result<CodeCoverageSummaryReceipt, String> {
    let base: BaselineDoc = serde_json::from_str(BASELINE_JSON)
        .map_err(|e| format!("parse embedded baseline coverage JSON: {e}"))?;
    let files_arr = val
        .get("data")
        .and_then(|d| d.get(0))
        .and_then(|d0| d0.get("files"))
        .and_then(|f| f.as_array())
        .ok_or_else(|| "llvm-cov export missing data[0].files array".to_owned())?;

    let mut by_file = BTreeMap::new();
    let mut after = CoverageCounters::default();
    for entry in files_arr {
        let Some(fname) = entry.get("filename").and_then(|s| s.as_str()) else {
            continue;
        };
        let Some(rel) = normalize_carbon_rel_path(fname) else {
            continue;
        };
        let parsed = parse_single_llvm_file(entry)?;
        after.lines_covered += parsed.summary.lines_covered;
        after.lines_total += parsed.summary.lines_total;
        after.regions_covered += parsed.summary.regions_covered;
        after.regions_total += parsed.summary.regions_total;
        after.functions_covered += parsed.summary.functions_covered;
        after.functions_total += parsed.summary.functions_total;
        by_file.insert(rel, parsed);
    }
    after.lines_pct = round2(after.lines_covered, after.lines_total);
    after.regions_pct = round2(after.regions_covered, after.regions_total);
    after.functions_pct = round2(after.functions_covered, after.functions_total);

    let before = CoverageCounters {
        lines_covered: base.lines_covered,
        lines_total: base.lines_total,
        lines_pct: base.lines_pct,
        regions_covered: base.regions_covered,
        regions_total: base.regions_total,
        regions_pct: base.regions_pct,
        functions_covered: base.functions_covered,
        functions_total: base.functions_total,
        functions_pct: base.functions_pct,
    };

    let files = build_file_deltas(&base, &by_file);
    let cli_rows = evaluate_surface_slice(CLI_COMMAND_ROWS, &by_file)?;
    let api_rows = evaluate_surface_slice(HTTP_API_ROWS, &by_file)?;
    let tier_env_rows = evaluate_tier_env_slice(&by_file)?;

    Ok(CodeCoverageSummaryReceipt {
        schema: "jopedime.carbon.code-coverage-summary.v1".to_owned(),
        source_sha: source_sha.to_owned(),
        baseline_sha: base.source_sha,
        before,
        after,
        files,
        cli_rows,
        api_rows,
        tier_env_rows,
    })
}

fn parse_single_llvm_file(entry: &serde_json::Value) -> Result<ParsedLlvmFile, String> {
    let sum = &entry["summary"];
    let lc = sum["lines"]["covered"].as_u64().unwrap_or(0);
    let lt = sum["lines"]["count"].as_u64().unwrap_or(0);
    let rc = sum["regions"]["covered"].as_u64().unwrap_or(0);
    let rt = sum["regions"]["count"].as_u64().unwrap_or(0);
    let fc = sum["functions"]["covered"].as_u64().unwrap_or(0);
    let ft = sum["functions"]["count"].as_u64().unwrap_or(0);
    let mut segments = Vec::new();
    if let Some(arr) = entry.get("segments").and_then(|s| s.as_array()) {
        for seg in arr {
            if let Some(items) = seg.as_array() {
                if items.len() >= 5 {
                    let ln = items[0].as_u64().unwrap_or(0);
                    let col = items[1].as_u64().unwrap_or(0);
                    let cnt = items[2].as_u64().unwrap_or(0);
                    let has_cnt =
                        u64::from(items[3].as_bool().unwrap_or(items[3].as_u64() == Some(1)));
                    let is_entry =
                        u64::from(items[4].as_bool().unwrap_or(items[4].as_u64() == Some(1)));
                    let is_gap = u64::from(
                        items
                            .get(5)
                            .is_some_and(|v| v.as_bool().unwrap_or(v.as_u64() == Some(1))),
                    );
                    segments.push([ln, col, cnt, has_cnt, is_entry, is_gap]);
                }
            }
        }
    }
    Ok(ParsedLlvmFile {
        summary: CoverageCounters {
            lines_covered: lc,
            lines_total: lt,
            lines_pct: round2(lc, lt),
            regions_covered: rc,
            regions_total: rt,
            regions_pct: round2(rc, rt),
            functions_covered: fc,
            functions_total: ft,
            functions_pct: round2(fc, ft),
        },
        segments,
    })
}

fn compute_span_metrics(
    file: &ParsedLlvmFile,
    start_line: usize,
    end_line: usize,
) -> (u64, u64, u64, u64) {
    let s_ln = start_line as u64;
    let e_ln = end_line as u64;
    let mut line_max: BTreeMap<u64, u64> = BTreeMap::new();
    let mut reg_tot = 0u64;
    let mut reg_cov = 0u64;
    let mut active: Option<(u64, bool)> = None;
    let mut prev_line = 1u64;

    for &[ln, col, cnt, has_cnt, is_entry, is_gap] in &file.segments {
        if let Some((act_cnt, act_gap)) = active {
            if !act_gap {
                let seg_end = if col == 1 { ln.saturating_sub(1) } else { ln };
                let lo = prev_line.max(s_ln);
                let hi = seg_end.min(e_ln);
                if lo <= hi {
                    for l in lo..=hi {
                        let cur = line_max.entry(l).or_insert(0);
                        if act_cnt > *cur {
                            *cur = act_cnt;
                        }
                    }
                }
            }
        }
        if ln >= s_ln && ln <= e_ln && has_cnt == 1 && is_entry == 1 && is_gap == 0 {
            reg_tot += 1;
            if cnt > 0 {
                reg_cov += 1;
            }
        }
        if has_cnt == 1 {
            active = Some((cnt, is_gap == 1));
            prev_line = ln;
            if is_gap == 0 && ln >= s_ln && ln <= e_ln {
                let cur = line_max.entry(ln).or_insert(0);
                if cnt > *cur {
                    *cur = cnt;
                }
            }
        } else {
            active = None;
        }
    }
    let ln_tot = line_max.len() as u64;
    let ln_cov = line_max.values().filter(|&&v| v > 0).count() as u64;
    (ln_cov, ln_tot, reg_cov, reg_tot)
}

fn build_file_deltas(
    base: &BaselineDoc,
    by_file: &BTreeMap<String, ParsedLlvmFile>,
) -> Vec<FileCoverageDelta> {
    let base_map: BTreeMap<&str, &BaselineFileEntry> =
        base.files.iter().map(|f| (f.file.as_str(), f)).collect();
    let mut out = Vec::with_capacity(by_file.len());
    for (rel, parsed) in by_file {
        let (bf, bl, br) = match base_map.get(rel.as_str()) {
            Some(b) => (b.functions_pct, b.lines_pct, b.regions_pct),
            None => (0.0, 0.0, 0.0),
        };
        out.push(FileCoverageDelta {
            file: rel.clone(),
            before_functions_pct: bf,
            after_functions_covered: parsed.summary.functions_covered,
            after_functions_total: parsed.summary.functions_total,
            after_functions_pct: parsed.summary.functions_pct,
            before_lines_pct: bl,
            after_lines_covered: parsed.summary.lines_covered,
            after_lines_total: parsed.summary.lines_total,
            after_lines_pct: parsed.summary.lines_pct,
            before_regions_pct: br,
            after_regions_covered: parsed.summary.regions_covered,
            after_regions_total: parsed.summary.regions_total,
            after_regions_pct: parsed.summary.regions_pct,
        });
    }
    out
}

fn evaluate_surface_slice(
    specs: &[SurfaceCoverageSpec],
    by_file: &BTreeMap<String, ParsedLlvmFile>,
) -> Result<Vec<EvaluatedSurfaceRow>, String> {
    let mut out = Vec::with_capacity(specs.len());
    for sp in specs {
        let (s_ln, e_ln) = resolve_anchor_span(sp.file, sp.anchor_start, sp.anchor_end)?;
        let pf = by_file
            .get(sp.file)
            .ok_or_else(|| format!("llvm-cov export missing file '{}'", sp.file))?;
        let (lc, lt, rc, rt) = compute_span_metrics(pf, s_ln, e_ln);
        out.push(EvaluatedSurfaceRow {
            id: sp.id.to_owned(),
            surface: sp.surface.to_owned(),
            bound_test: sp.bound_test.to_owned(),
            status_badge: sp.status_badge.to_owned(),
            source_span: format!("{}:{s_ln}-{e_ln}", sp.file),
            lines_covered: lc,
            lines_total: lt,
            lines_pct: round2(lc, lt),
            regions_covered: rc,
            regions_total: rt,
            regions_pct: round2(rc, rt),
        });
    }
    Ok(out)
}

fn evaluate_tier_env_slice(
    by_file: &BTreeMap<String, ParsedLlvmFile>,
) -> Result<Vec<EvaluatedTierEnvRow>, String> {
    let specs = build_tier_env_rows();
    let mut out = Vec::with_capacity(specs.len());
    for sp in specs {
        let (s_ln, e_ln) = resolve_anchor_span(sp.file, sp.anchor_start, sp.anchor_end)?;
        let pf = by_file
            .get(sp.file)
            .ok_or_else(|| format!("llvm-cov export missing file '{}'", sp.file))?;
        let (lc, lt, rc, rt) = compute_span_metrics(pf, s_ln, e_ln);
        out.push(EvaluatedTierEnvRow {
            cell_id: sp.cell_id,
            environment: sp.environment.to_owned(),
            auth_tier: sp.auth_tier.to_owned(),
            effort: sp.effort.to_owned(),
            expected_behavior: sp.expected_behavior.to_owned(),
            bound_test: sp.bound_test.to_owned(),
            status_badge: sp.status_badge.to_owned(),
            source_span: format!("{}:{s_ln}-{e_ln}", sp.file),
            lines_covered: lc,
            lines_total: lt,
            lines_pct: round2(lc, lt),
            regions_covered: rc,
            regions_total: rt,
            regions_pct: round2(rc, rt),
        });
    }
    Ok(out)
}

/// Fail-closed validation of `CodeCoverageSummaryReceipt` for both `spec-report` and `spec-verify`.
pub fn validate_code_coverage_receipt(
    rcpt: &CodeCoverageSummaryReceipt,
    expected_sha: Option<&str>,
) -> Result<(), String> {
    if rcpt.schema != "jopedime.carbon.code-coverage-summary.v1" {
        return Err(format!("unexpected code coverage schema '{}'", rcpt.schema));
    }
    if let Some(exp) = expected_sha {
        if !exp.trim().is_empty() && exp.trim() != "workspace" && rcpt.source_sha != exp.trim() {
            return Err(format!(
                "code coverage source_sha '{}' != expected '{}'",
                rcpt.source_sha,
                exp.trim()
            ));
        }
    }
    if rcpt.cli_rows.len() != CLI_COMMAND_ROWS.len()
        || rcpt.api_rows.len() != HTTP_API_ROWS.len()
        || rcpt.tier_env_rows.len() != 64
    {
        return Err(format!(
            "unexpected matrix row counts (cli={}, api={}, tier_env={})",
            rcpt.cli_rows.len(),
            rcpt.api_rows.len(),
            rcpt.tier_env_rows.len()
        ));
    }
    if rcpt.after.lines_pct <= rcpt.before.lines_pct
        || rcpt.after.regions_pct <= rcpt.before.regions_pct
        || rcpt.after.functions_pct <= rcpt.before.functions_pct
    {
        return Err(format!(
            "after coverage ({:.2}% fn / {:.2}% ln / {:.2}% br) did not improve over baseline ({:.2}% / {:.2}% / {:.2}%)",
            rcpt.after.functions_pct,
            rcpt.after.lines_pct,
            rcpt.after.regions_pct,
            rcpt.before.functions_pct,
            rcpt.before.lines_pct,
            rcpt.before.regions_pct,
        ));
    }
    for row in rcpt.cli_rows.iter().chain(rcpt.api_rows.iter()) {
        if row.lines_total == 0 || row.lines_covered == 0 {
            return Err(format!(
                "row {} ({}) has zero covered lines ({}/{})",
                row.id, row.surface, row.lines_covered, row.lines_total
            ));
        }
    }
    for cell in &rcpt.tier_env_rows {
        if cell.lines_total == 0 || cell.lines_covered == 0 {
            return Err(format!(
                "tier x env cell {} ({} {} {}) has zero covered lines",
                cell.cell_id, cell.environment, cell.auth_tier, cell.effort
            ));
        }
    }
    let fallback_cells = rcpt
        .tier_env_rows
        .iter()
        .filter(|c| c.status_badge == "AUTO-EMU-FALLBACK")
        .count();
    if fallback_cells != 8 {
        return Err(format!(
            "expected 8 AUTO-EMU-FALLBACK high/ultra Pro live cells, found {fallback_cells}"
        ));
    }
    Ok(())
}

fn esc(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len() + 8);
    for ch in raw.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(ch),
        }
    }
    out
}

fn badge_pill_html(badge: &str) -> String {
    if badge == "AUTO-EMU-FALLBACK" {
        "<span class=\"pill-gap\">AUTO-EMU-FALLBACK</span>".to_owned()
    } else {
        format!("<span class=\"pill-pass\">{}</span>", esc(badge))
    }
}

fn render_before_after_svg(
    base_sha_short: &str,
    before: &CoverageCounters,
    after: &CoverageCounters,
) -> String {
    let metrics = [
        ("Line Coverage (src/**)", before.lines_pct, after.lines_pct),
        ("MIR Branch-Regions", before.regions_pct, after.regions_pct),
        (
            "Function Coverage",
            before.functions_pct,
            after.functions_pct,
        ),
    ];
    let mut bars = String::new();
    for (idx, (label, b_pct, a_pct)) in metrics.iter().enumerate() {
        let y0 = 42.0 + (idx as f64) * 44.0;
        let wb = (b_pct * 5.2).clamp(2.0, 520.0);
        let wa = (a_pct * 5.2).clamp(2.0, 520.0);
        let delta = a_pct - b_pct;
        bars.push_str(&format!(
            r##"<text x="16" y="{ty:.1}" fill="#0f172a" font-size="9.5" font-weight="700" font-family="Liberation Sans, sans-serif">{label}</text>
<rect x="195" y="{y0:.1}" width="{wb:.1}" height="12" rx="2.5" fill="#94a3b8"/>
<text x="{tb:.1}" y="{tby:.1}" fill="#475569" font-size="8.4" font-family="Liberation Mono, monospace">Before: {b_pct:.2}%</text>
<rect x="195" y="{y1:.1}" width="{wa:.1}" height="13" rx="2.5" fill="#0284c7"/>
<text x="{ta:.1}" y="{tay:.1}" fill="#0f172a" font-size="8.6" font-weight="700" font-family="Liberation Mono, monospace">After: {a_pct:.2}% ({delta:+.2} pp)</text>"##,
            ty = y0 + 14.0,
            tb = 201.0 + wb,
            tby = y0 + 9.5,
            y1 = y0 + 15.0,
            ta = 201.0 + wa,
            tay = y0 + 25.0,
        ));
    }
    format!(
        r##"<svg viewBox="0 0 880 182" width="100%" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Before vs After Compiler Coverage">
  <rect x="1" y="1" width="878" height="180" rx="8" fill="#f8fafc" stroke="#cbd5e1"/>
  <text x="16" y="23" fill="#0f172a" font-size="11.0" font-weight="700" font-family="Liberation Sans, sans-serif">Figure 14A — rustc 1.97.1 -C instrument-coverage Before ({base_sha_short}) vs After Remaining Section 14 Gap Closure</text>
  {bars}
</svg>"##
    )
}

/// Render Section 14 HTML (KPIs, SVG chart, Table 14A CLI, Table 14B API, Table 14C `tier x env`, Table 14D file deltas).
pub fn render_code_coverage_section_html(rcpt: &CodeCoverageSummaryReceipt) -> String {
    let dl = rcpt.after.lines_pct - rcpt.before.lines_pct;
    let dr = rcpt.after.regions_pct - rcpt.before.regions_pct;
    let df = rcpt.after.functions_pct - rcpt.before.functions_pct;
    let base_sha_short = esc(&rcpt.baseline_sha[..rcpt.baseline_sha.len().min(10)]);
    let svg = render_before_after_svg(&base_sha_short, &rcpt.before, &rcpt.after);

    let cli_body = render_surface_rows_html(&rcpt.cli_rows);
    let api_body = render_surface_rows_html(&rcpt.api_rows);
    let te_body = render_tier_env_rows_html(&rcpt.tier_env_rows);
    let file_body = render_file_delta_rows_html(&rcpt.files);

    format!(
        r##"  <div class="page-break"></div>
  <h2>14. Compiler-Instrumented Full Product Code Coverage Matrix (<code>CLI</code>, <code>API</code> &amp; <code>Tier &times; Env</code>)</h2>
  <p>
    Section 14 binds every shipping Carbon CLI command and binary entrypoint (<strong>{cli_n}</strong> surfaces), every public Edge, Enterprise Serve, and internal Runtime/CAS HTTP endpoint (<strong>{api_n}</strong> routes), and every <code>effort &times; auth_tier &times; environment</code> product cell (<strong>64</strong> cells) to compiler-instrumented <code>rustc 1.97.1 -C instrument-coverage</code> + <code>llvm-cov export</code> function, line, and MIR branch-region counters, comparing baseline <code>origin/main</code> (<code>{base_sha}</code>; cumulative progression from <code>7988e2ecd306</code> at <code>45.52%</code> fn / <code>63.42%</code> ln &rarr; <code>4c4543295b</code> at <code>52.54%</code> fn / <code>73.11%</code> ln &rarr; <code>0ff23cc6ac</code> at <code>60.91%</code> fn / <code>78.46%</code> ln) against exact HEAD (<code>{head_sha}</code>). Live CI <code>high</code> and <code>ultra</code> Pro cells remain explicitly badged <code>AUTO-EMU-FALLBACK</code> under the zero-repo-secret posture:
  </p>
  <div class="kpi-strip">
    <div class="kpi"><div class="kpi-val">{al:.2}% ({dl:+.2} pp)</div><div class="kpi-lbl">Line Coverage: {alc}/{alt} (Before {bl:.2}%: {blc}/{blt})</div></div>
    <div class="kpi"><div class="kpi-val">{ar:.2}% ({dr:+.2} pp)</div><div class="kpi-lbl">MIR Branch-Regions: {arc}/{art} (Before {br:.2}%: {brc}/{brt})</div></div>
    <div class="kpi"><div class="kpi-val">{af:.2}% ({df:+.2} pp)</div><div class="kpi-lbl">Functions: {afc}/{aft} (Before {bf:.2}%: {bfc}/{bft})</div></div>
    <div class="kpi"><div class="kpi-val">{cli_n} / {api_n} / 64</div><div class="kpi-lbl">100% Mapped CLI, API &amp; Tier&times;Env Rows (0 Uncovered)</div></div>
  </div>
  <div class="svg-box">{svg}</div>

  <h3>Table 14A — All {cli_n} Carbon CLI Commands, Subcommands, Global Flags &amp; Lambda Binaries</h3>
  <table class="cov-table">
    <thead>
      <tr><th style="width:7%">ID</th><th style="width:24%">CLI Command / Binary Surface</th><th style="width:29%">Bound Automated Test(s)</th><th style="width:8%">Status</th><th class="num" style="width:16%">Line Cov</th><th class="num" style="width:16%">MIR Branch-Regions</th></tr>
    </thead>
    <tbody>{cli_body}</tbody>
  </table>

  <h3>Table 14B — All {api_n} Carbon HTTP API Endpoints (Public Edge, Enterprise Serve &amp; Internal Runtime/CAS)</h3>
  <table class="cov-table">
    <thead>
      <tr><th style="width:7%">ID</th><th style="width:26%">HTTP Method &amp; Endpoint</th><th style="width:27%">Bound Automated Test(s)</th><th style="width:8%">Status</th><th class="num" style="width:16%">Line Cov</th><th class="num" style="width:16%">MIR Branch-Regions</th></tr>
    </thead>
    <tbody>{api_body}</tbody>
  </table>

  <h3>Table 14C — Complete 64-Cell <code>Tier &times; Env</code> Matrix (<code>4</code> Efforts &times; <code>2</code> Auth Tiers &times; <code>8</code> Environments)</h3>
  <table class="cov-table">
    <thead>
      <tr><th style="width:6%">Cell</th><th style="width:15%">Environment</th><th style="width:9%">Tier/Eff</th><th style="width:19%">Expected Gate / Topology</th><th style="width:18%">Bound Test</th><th style="width:13%">Status</th><th class="num" style="width:10%">Line %</th><th class="num" style="width:10%">Branch %</th></tr>
    </thead>
    <tbody>{te_body}</tbody>
  </table>

  <h3>Table 14D — Per-Module Function, Line &amp; MIR Region Coverage Before (<code>{base_sha_short}</code>) vs After (<code>crates/jope-carbon/src/**</code>)</h3>
  <table class="cov-table">
    <thead>
      <tr><th style="width:24%">Source File</th><th class="num" style="width:8%">Before Fn</th><th class="num" style="width:14%">After Functions</th><th class="num" style="width:9%">Delta Fn</th><th class="num" style="width:8%">Before Ln</th><th class="num" style="width:14%">After Lines</th><th class="num" style="width:9%">Delta Ln</th><th class="num" style="width:14%">After MIR Br</th></tr>
    </thead>
    <tbody>{file_body}</tbody>
  </table>"##,
        base_sha_short = base_sha_short,
        cli_n = rcpt.cli_rows.len(),
        api_n = rcpt.api_rows.len(),
        base_sha = esc(&rcpt.baseline_sha[..rcpt.baseline_sha.len().min(12)]),
        head_sha = esc(&rcpt.source_sha[..rcpt.source_sha.len().min(12)]),
        al = rcpt.after.lines_pct,
        alc = rcpt.after.lines_covered,
        alt = rcpt.after.lines_total,
        bl = rcpt.before.lines_pct,
        blc = rcpt.before.lines_covered,
        blt = rcpt.before.lines_total,
        ar = rcpt.after.regions_pct,
        arc = rcpt.after.regions_covered,
        art = rcpt.after.regions_total,
        br = rcpt.before.regions_pct,
        brc = rcpt.before.regions_covered,
        brt = rcpt.before.regions_total,
        af = rcpt.after.functions_pct,
        afc = rcpt.after.functions_covered,
        aft = rcpt.after.functions_total,
        bf = rcpt.before.functions_pct,
        bfc = rcpt.before.functions_covered,
        bft = rcpt.before.functions_total,
    )
}

fn render_surface_rows_html(rows: &[EvaluatedSurfaceRow]) -> String {
    let mut buf = String::with_capacity(rows.len() * 260);
    for r in rows {
        buf.push_str(&format!(
            "<tr><td><code>{}</code></td><td><code class=\"wrap-any\">{}</code></td><td><code class=\"wrap-any\">{}</code></td><td>{}</td><td class=\"num\">{}/{} ({:.1}%)</td><td class=\"num\">{}/{} ({:.1}%)</td></tr>\n",
            esc(&r.id),
            esc(&r.surface),
            esc(&r.bound_test),
            badge_pill_html(&r.status_badge),
            r.lines_covered,
            r.lines_total,
            r.lines_pct,
            r.regions_covered,
            r.regions_total,
            r.regions_pct,
        ));
    }
    buf
}

fn render_tier_env_rows_html(rows: &[EvaluatedTierEnvRow]) -> String {
    let mut buf = String::with_capacity(rows.len() * 280);
    for r in rows {
        buf.push_str(&format!(
            "<tr><td><code>{}</code></td><td><code class=\"wrap-any\">{}</code></td><td><code>{}/{}</code></td><td>{}</td><td><code class=\"wrap-any\">{}</code></td><td>{}</td><td class=\"num\">{:.1}% ({}/{})</td><td class=\"num\">{:.1}% ({}/{})</td></tr>\n",
            esc(&r.cell_id),
            esc(&r.environment),
            esc(&r.auth_tier),
            esc(&r.effort),
            esc(&r.expected_behavior),
            esc(&r.bound_test),
            badge_pill_html(&r.status_badge),
            r.lines_pct,
            r.lines_covered,
            r.lines_total,
            r.regions_pct,
            r.regions_covered,
            r.regions_total,
        ));
    }
    buf
}

fn render_file_delta_rows_html(files: &[FileCoverageDelta]) -> String {
    let mut buf = String::with_capacity(files.len() * 280);
    for f in files {
        let d_fn = f.after_functions_pct - f.before_functions_pct;
        let d_ln = f.after_lines_pct - f.before_lines_pct;
        buf.push_str(&format!(
            "<tr><td><code class=\"wrap-any\">{}</code></td><td class=\"num\">{:.1}%</td><td class=\"num\">{}/{} ({:.1}%)</td><td class=\"num\">{:+.1} pp</td><td class=\"num\">{:.1}%</td><td class=\"num\">{}/{} ({:.1}%)</td><td class=\"num\">{:+.1} pp</td><td class=\"num\">{}/{} ({:.1}%)</td></tr>\n",
            esc(&f.file),
            f.before_functions_pct,
            f.after_functions_covered,
            f.after_functions_total,
            f.after_functions_pct,
            d_fn,
            f.before_lines_pct,
            f.after_lines_covered,
            f.after_lines_total,
            f.after_lines_pct,
            d_ln,
            f.after_regions_covered,
            f.after_regions_total,
            f.after_regions_pct,
        ));
    }
    buf
}
