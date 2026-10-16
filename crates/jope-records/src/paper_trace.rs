//! `paper-trace`: the white paper's and the README's printed numbers against the records they
//! come from.
//!
//! `docs/paper/TECHNICAL_WRITEUP.md` cites claims as `[Cnn]`; `docs/paper/claims.json` holds each
//! claim's statement, value, host-local source and pinned SHA-256; `docs/paper/NUMBERS_TRACE.md`
//! records where every claim was re-hashed and re-derived on xbabe2. The claim sources live on that
//! host, so their hashes are checked there (the trace says how), not here. This guard is the part a
//! clone can check, and it fails when a printed number drifts from its record:
//!
//! 1. the ledger: unique ids, a known status, a statement, value and source per claim, a 64-hex
//!    pin unless the claim is derived, and a declared root for every `ROOT:` prefix;
//! 2. citations: every explicitly cited id exists and every ledger id is cited;
//! 3. totals: the Appendix G sentence equals the ledger's status counts, and the Section 10.4
//!    table lists exactly the pending claims;
//! 4. numbers: every number the write-up prints is accounted for by a claim cited on its line or
//!    in its section (verbatim, or rounded at the precision printed), or by a row of the trace's
//!    exemption table whose text still occurs in the write-up;
//! 5. the trace: one row per claim, carrying the ledger's status;
//! 6. the README: the version and the "At a glance" counts, recomputed from the committed files
//!    their footnotes name;
//! 7. Figure 1: its data, the claims it draws and its rendering ([`crate::paper_figure`]);
//! 8. no unfilled marker (`...PLACEHOLDER...`, `TODO`, `TBD`) in the write-up or the trace.
//!
//! Numbers inside identifiers (`C26`, `R23`, `H1`, `v2`, `r3.1`, `p50`, `float32`, `SHA-256`),
//! inline code, author-year references, dates, clock times and single-digit integers are not
//! numbers here. A rounding match is accepted only for a printed number with decimals; integers
//! must match exactly.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::process::Command;
use std::sync::LazyLock;

use regex::Regex;

use crate::py;
use crate::pyjson::{self, Value};
use crate::{Error, Result};

pub const WRITEUP: &str = "docs/paper/TECHNICAL_WRITEUP.md";
pub const LEDGER: &str = "docs/paper/claims.json";
pub const TRACE: &str = "docs/paper/NUMBERS_TRACE.md";
pub const README: &str = "README.md";

const STATUSES: [&str; 4] = ["established", "preliminary", "non-qualifying", "pending"];

fn re(pattern: &str) -> Regex {
    Regex::new(pattern).expect("static pattern")
}

static CITATION: LazyLock<Regex> = LazyLock::new(|| re(r"\[C\d+[^\]]*\]"));
static CITED_ID: LazyLock<Regex> = LazyLock::new(|| re(r"C(\d+)(?:\s*[–-]\s*C?(\d+))?"));
static GROUPED: LazyLock<Regex> = LazyLock::new(|| re(r"(\d),(\d{3})\b"));
static SCIENTIFIC: LazyLock<Regex> =
    LazyLock::new(|| re(r"(\d+(?:\.\d+)?)\s*(?:\\times|×)\s*10\^\{?(-?\d+)\}?"));
static POWER: LazyLock<Regex> = LazyLock::new(|| re(r"\b10\^\{?(-?\d+)\}?"));
static NOT_NUMBERS: LazyLock<Regex> = LazyLock::new(|| {
    re(concat!(
        r"`[^`]*`",
        r"|\[C\d+[^\]]*\]",
        r"|\b[A-Z]{1,2}\d+\b",
        r"|\b[vr]\d+(?:\.\d+)*\b",
        r"|\bp\d{2}\b",
        r"|\bfloat\d+\b",
        r"|SHA-256",
        r"|\b(?:Sections?|Figure|Appendix|Phase|Equations?) \d+(?:\.\d+)*(?:(?:, | and | to |–|-)\d+(?:\.\d+)*)*",
        // Author-year references only in their citation shapes: "(Chen et al., 2021;", "Hutter,
        // 2017)" and the narrative "Parisi et al. (2019)". A bare capitalised word before a
        // number ("At 2,000", read as "At 2000" after normalisation) is not a reference.
        r"|[A-Z][\p{L}'-]+(?: et al\.| and [A-Z][\p{L}'-]+)?, (?:19|20)\d{2}[a-z]?[);]",
        r"|[A-Z][\p{L}'-]+(?: et al\.| and [A-Z][\p{L}'-]+) \((?:19|20)\d{2}[a-z]?\)",
    ))
});
/// Dates and clock times, removed before [`NOT_NUMBERS`] so that "The 2026-10-12 direction" is
/// not read as an author-year reference followed by `-10-12`.
static DATES: LazyLock<Regex> =
    LazyLock::new(|| re(r"\b20\d{2}-\d{2}-\d{2}(?:T[0-9:.]+Z?)?|\b\d{2}:\d{2}(?::\d{2})?Z?\b"));
static UNFILLED: LazyLock<Regex> =
    LazyLock::new(|| re(r"\b[A-Z0-9_]*PLACEHOLDER[A-Z0-9_]*\b|\bTODO\b|\bTBD\b"));
static NUMBER: LazyLock<Regex> = LazyLock::new(|| re(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?[MkK]?"));

/// Commas between digit groups, LaTeX thin spaces, Unicode minus and en dash, and `a×10^{b}` are
/// written one way on both sides of a comparison.
fn normalize(text: &str) -> String {
    let mut text = text
        .replace("{,}", "")
        .replace(['\u{2212}', '\u{2013}'], "-");
    loop {
        let next = GROUPED.replace_all(&text, "${1}${2}").into_owned();
        if next == text {
            break;
        }
        text = next;
    }
    let text = SCIENTIFIC.replace_all(&text, "${1}e${2}");
    POWER.replace_all(&text, "1e${1}").into_owned()
}

/// One printed number: its text, value and the half-width of the interval it rounds from (zero
/// for an integer, which must match exactly).
#[derive(Clone, Debug, PartialEq)]
struct Printed {
    text: String,
    value: f64,
    half_ulp: f64,
}

/// The numbers in already-normalized text. A sign directly after a digit, letter, `.` or `)` is a
/// range dash, not a sign; a number glued to a letter or `_` is part of an identifier.
fn numbers(text: &str) -> Vec<Printed> {
    let mut out = vec![];
    for m in NUMBER.find_iter(text) {
        let mut token = m.as_str();
        let before = text[..m.start()].chars().next_back();
        if before.is_some_and(|c| c.is_alphabetic() || c == '_') {
            continue;
        }
        if token.starts_with(['-', '+'])
            && before.is_some_and(|c| c.is_alphanumeric() || c == '.' || c == ')')
        {
            token = &token[1..];
        }
        let (body, scale) = match token.chars().last() {
            Some('M') => (&token[..token.len() - 1], 1e6),
            Some('k' | 'K') => (&token[..token.len() - 1], 1e3),
            _ => (token, 1.0),
        };
        let Ok(value) = body.parse::<f64>() else {
            continue;
        };
        let (mantissa, exponent) = match body.split_once('e') {
            Some((m, e)) => (m, e.parse::<i32>().unwrap_or(0)),
            None => (body, 0),
        };
        let decimals = mantissa.split_once('.').map_or(0, |(_, d)| d.len()) as i32;
        let integral = decimals == 0 && exponent == 0 && scale == 1.0;
        if integral && value.abs() < 10.0 {
            continue;
        }
        let half_ulp = if integral {
            0.0
        } else {
            0.5 * 10f64.powi(exponent - decimals) * scale
        };
        out.push(Printed {
            text: body.trim_start_matches('+').to_owned(),
            value: value * scale,
            half_ulp,
        });
    }
    out
}

/// The numbers a reader of the line sees, identifiers and references removed.
fn printed_numbers(line: &str) -> Vec<Printed> {
    let line = normalize(line);
    let line = DATES.replace_all(&line, " ");
    numbers(&NOT_NUMBERS.replace_all(&line, " "))
}

/// Whether a printed number is one of the record's numbers, or that number at the precision
/// printed.
fn accounted(printed: &Printed, record: &[Printed]) -> bool {
    record.iter().any(|r| {
        r.text == printed.text
            || (r.value - printed.value).abs() <= printed.half_ulp * (1.0 + 1e-9) + 1e-15
    })
}

/// Claim ids cited on a line: `[C01]`, `[C01, C02]`, `[C136–C140]` (ranges expand). The second
/// set holds only the explicitly written ids, which must exist.
fn cited(line: &str) -> (BTreeSet<String>, BTreeSet<String>) {
    let mut all = BTreeSet::new();
    let mut explicit = BTreeSet::new();
    for bracket in CITATION.find_iter(line) {
        for c in CITED_ID.captures_iter(bracket.as_str()) {
            let a: u32 = c[1].parse().unwrap_or(0);
            explicit.insert(format!("C{a:02}"));
            match c.get(2).and_then(|b| b.as_str().parse::<u32>().ok()) {
                Some(b) => {
                    explicit.insert(format!("C{b:02}"));
                    all.extend((a..=b).map(|i| format!("C{i:02}")));
                }
                None => {
                    all.insert(format!("C{a:02}"));
                }
            }
        }
    }
    (all, explicit)
}

#[derive(Debug)]
struct Claim {
    id: String,
    status: String,
    numbers: Vec<Printed>,
}

fn field<'a>(claim: &'a Value, key: &str) -> &'a str {
    claim.get(key).and_then(Value::as_str).unwrap_or("")
}

/// The ledger's claims, with the problems rule 1 finds.
fn ledger(text: &str, problems: &mut Vec<String>) -> Result<Vec<Claim>> {
    let doc = pyjson::parse_str(text).map_err(|m| Error::parse(Path::new(LEDGER), m))?;
    let roots: BTreeSet<String> = doc
        .get("source_conventions")
        .and_then(|c| c.get("roots"))
        .and_then(Value::as_dict)
        .map(|d| d.keys().map(str::to_owned).collect())
        .unwrap_or_default();
    let root_prefix = re(r"(?:^|;\s*)([A-Z][A-Z0-9]*):");
    let list = doc
        .get("claims")
        .and_then(Value::as_list)
        .ok_or_else(|| Error::Shape(format!("{LEDGER}: no claims list")))?;
    let mut claims = vec![];
    let mut seen = BTreeSet::new();
    for claim in list {
        let id = field(claim, "id").to_owned();
        if !seen.insert(id.clone()) {
            problems.push(format!("ledger: duplicate claim id {id}"));
        }
        let status = field(claim, "status").to_owned();
        if !STATUSES.contains(&status.as_str()) {
            problems.push(format!("ledger: {id} has unknown status {status:?}"));
        }
        for key in ["statement", "value", "source"] {
            if field(claim, key).trim().is_empty() {
                problems.push(format!("ledger: {id} has no {key}"));
            }
        }
        let source = field(claim, "source");
        let sha = field(claim, "sha256");
        let derived = source.starts_with("derived from");
        if !derived && !(sha.len() == 64 && sha.bytes().all(|b| b.is_ascii_hexdigit())) {
            problems.push(format!("ledger: {id} pins no 64-hex sha256"));
        }
        for c in root_prefix.captures_iter(source) {
            if !roots.contains(&c[1]) {
                problems.push(format!(
                    "ledger: {id} cites root {} that source_conventions.roots does not declare",
                    &c[1]
                ));
            }
        }
        let record = [
            field(claim, "statement"),
            field(claim, "value"),
            field(claim, "tex"),
        ]
        .join(" | ");
        claims.push(Claim {
            id,
            status,
            numbers: numbers(&normalize(&record)),
        });
    }
    Ok(claims)
}

/// `| <text> | <numbers> | <reason> |` rows under the trace's `## Exempt numbers` heading.
fn exemptions(trace: &str) -> Vec<(String, Vec<String>)> {
    let mut rows = vec![];
    let mut inside = false;
    for line in trace.lines() {
        if line.starts_with("## ") {
            inside = line.trim() == "## Exempt numbers";
            continue;
        }
        let cells: Vec<&str> = line.split('|').map(str::trim).collect();
        if !inside || cells.len() < 4 || cells[1].starts_with("---") || cells[1] == "Text" {
            continue;
        }
        let text = cells[1].trim_matches('`').to_owned();
        let listed = cells[2]
            .split(',')
            .map(|n| n.trim().to_owned())
            .filter(|n| !n.is_empty())
            .collect();
        rows.push((text, listed));
    }
    rows
}

/// `| Cnn | <status> | ...` rows of the trace's `## Claim trace` table.
fn trace_rows(trace: &str) -> BTreeMap<String, String> {
    let mut rows = BTreeMap::new();
    let mut inside = false;
    let id = re(r"^C\d+$");
    for line in trace.lines() {
        if line.starts_with("## ") {
            inside = line.trim() == "## Claim trace";
            continue;
        }
        let cells: Vec<&str> = line.split('|').map(str::trim).collect();
        if inside && cells.len() > 3 && id.is_match(cells[1]) {
            rows.insert(cells[1].to_owned(), cells[2].to_owned());
        }
    }
    rows
}

/// Rules 1-5 and 8 over the three paper documents; the problems found and a one-line summary.
pub fn check_paper(writeup: &str, ledger_text: &str, trace: &str) -> Result<(Vec<String>, String)> {
    let mut problems = vec![];
    // A section left as a marker to fill in later is not a record.
    for (name, text) in [(WRITEUP, writeup), (TRACE, trace)] {
        for (i, line) in text.lines().enumerate() {
            if UNFILLED.is_match(line) {
                problems.push(format!(
                    "{name}:{}: unfilled placeholder: {}",
                    i + 1,
                    line.trim()
                ));
            }
        }
    }
    let claims = ledger(ledger_text, &mut problems)?;
    let by_id: BTreeMap<&str, &Claim> = claims.iter().map(|c| (c.id.as_str(), c)).collect();

    // Sections are delimited by headings of level 2 to 4; a number may be accounted for by any
    // claim cited in its section.
    let lines: Vec<&str> = writeup.lines().collect();
    let mut section = vec![0usize; lines.len()];
    let mut current = 0;
    for (i, line) in lines.iter().enumerate() {
        if line.starts_with("## ") || line.starts_with("### ") || line.starts_with("#### ") {
            current += 1;
        }
        section[i] = current;
    }
    let mut section_ids: BTreeMap<usize, BTreeSet<String>> = BTreeMap::new();
    let mut cited_anywhere = BTreeSet::new();
    for (i, line) in lines.iter().enumerate() {
        let (all, explicit) = cited(line);
        for id in &explicit {
            if !by_id.contains_key(id.as_str()) {
                problems.push(format!(
                    "{WRITEUP}:{}: cites {id}, which is not in the ledger",
                    i + 1
                ));
            }
        }
        cited_anywhere.extend(all.iter().cloned());
        section_ids.entry(section[i]).or_default().extend(all);
    }
    for claim in &claims {
        if !cited_anywhere.contains(&claim.id) {
            problems.push(format!(
                "{WRITEUP}: {} is in the ledger but never cited",
                claim.id
            ));
        }
    }

    // Appendix G totals and the Section 10.4 pending table.
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for claim in &claims {
        *counts.entry(claim.status.as_str()).or_default() += 1;
    }
    let count = |s: &str| counts.get(s).copied().unwrap_or(0);
    let totals = format!(
        "`claims.json` holds {} claims: {} established, {} preliminary, {} non-qualifying and {} pending.",
        claims.len(),
        count("established"),
        count("preliminary"),
        count("non-qualifying"),
        count("pending")
    );
    if !writeup.contains(&totals) {
        problems.push(format!(
            "{WRITEUP}: Appendix G must state the ledger's totals: {totals}"
        ));
    }
    let pending: BTreeSet<&str> = claims
        .iter()
        .filter(|c| c.status == "pending")
        .map(|c| c.id.as_str())
        .collect();
    let mut table = BTreeSet::new();
    let mut inside = false;
    let row = re(r"^\| (C\d+) \|.*\| pending \|$");
    for line in &lines {
        if line.starts_with("### ") {
            inside = line.starts_with("### 10.4 ");
            continue;
        }
        if let Some(c) = inside.then(|| row.captures(line)).flatten() {
            table.insert(c.get(1).map_or("", |m| m.as_str()));
        }
    }
    if table != pending {
        problems.push(format!(
            "{WRITEUP}: the Section 10.4 table lists {:?}; the ledger's pending claims are {:?}",
            table, pending
        ));
    }

    // Every printed number.
    let exempt = exemptions(trace);
    for (text, _) in &exempt {
        if !writeup.contains(text.as_str()) {
            problems.push(format!(
                "{TRACE}: the exempted text {text:?} no longer occurs in {WRITEUP}"
            ));
        }
    }
    let mut checked = 0;
    for (i, line) in lines.iter().enumerate() {
        // Headings, and the totals sentence the rule above already compared with the ledger.
        if line.starts_with('#') || line.contains(&totals) {
            continue;
        }
        let scope: Vec<Printed> = section_ids
            .get(&section[i])
            .into_iter()
            .flatten()
            .filter_map(|id| by_id.get(id.as_str()))
            .flat_map(|c| c.numbers.iter().cloned())
            .collect();
        let excused: BTreeSet<&str> = exempt
            .iter()
            .filter(|(text, _)| line.contains(text.as_str()))
            .flat_map(|(_, listed)| listed.iter().map(String::as_str))
            .collect();
        for number in printed_numbers(line) {
            checked += 1;
            if excused.contains(number.text.as_str()) || accounted(&number, &scope) {
                continue;
            }
            problems.push(format!(
                "{WRITEUP}:{}: {} is not in any claim cited in its section, nor exempted in {TRACE}",
                i + 1,
                number.text
            ));
        }
    }

    // The trace has one row per claim, with the ledger's status.
    let rows = trace_rows(trace);
    for claim in &claims {
        match rows.get(&claim.id) {
            None => problems.push(format!("{TRACE}: no Claim trace row for {}", claim.id)),
            Some(status) if *status != claim.status => problems.push(format!(
                "{TRACE}: {} is {status:?} in the trace but {:?} in the ledger",
                claim.id, claim.status
            )),
            Some(_) => {}
        }
    }
    for id in rows.keys() {
        if !by_id.contains_key(id.as_str()) {
            problems.push(format!("{TRACE}: row {id} has no ledger claim"));
        }
    }

    let summary = format!(
        "paper: {} claims ({} established, {} preliminary, {} non-qualifying, {} pending); {} printed numbers checked; {} exemptions",
        claims.len(),
        count("established"),
        count("preliminary"),
        count("non-qualifying"),
        count("pending"),
        checked,
        exempt.len()
    );
    Ok((problems, summary))
}

/// `1234567` as `1,234,567`.
fn grouped(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, d) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(d);
    }
    out
}

/// Small counts the README spells out ("nine jobs", "forty ... fixtures").
fn words(n: usize) -> String {
    const ONES: [&str; 20] = [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    ];
    const TENS: [&str; 10] = [
        "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    ];
    match n {
        0..=19 => ONES[n].to_owned(),
        20..=99 if n.is_multiple_of(10) => TENS[n / 10].to_owned(),
        20..=99 => format!("{}-{}", TENS[n / 10], ONES[n % 10]),
        _ => n.to_string(),
    }
}

/// The values the README's "At a glance" footnotes name, recomputed from committed files.
#[derive(Debug, Default, PartialEq)]
pub struct Glance {
    pub version: String,
    pub members: usize,
    pub default_members: usize,
    pub versions: BTreeMap<String, usize>,
    pub vendored: usize,
    pub vendored_version: String,
    pub emit_targets: usize,
    pub rust_tests: usize,
    pub rust_ignored: usize,
    pub rust_packages: usize,
    pub python_tests: usize,
    pub ci_jobs: usize,
    pub full_families: usize,
    pub coverage_items: usize,
    pub coverage_lanes: usize,
    pub matrix: Vec<usize>,
    pub synthetic: usize,
    pub test_datasets: usize,
    pub python_files: usize,
    pub python_under_python: usize,
    pub adrs: usize,
}

fn json(root: &Path, rel: &str) -> Result<Value> {
    let path = root.join(rel);
    pyjson::parse_str_big(&py::read_text(&path)?).map_err(|m| Error::parse(&path, m))
}

fn len_at(doc: &Value, path: &[&str]) -> usize {
    let mut v = Some(doc);
    for key in path {
        v = v.and_then(|x| x.get(key));
    }
    match v {
        Some(Value::List(l)) => l.len(),
        Some(Value::Dict(d)) => d.len(),
        _ => 0,
    }
}

fn int_at(doc: &Value, path: &[&str]) -> usize {
    let mut v = Some(doc);
    for key in path {
        v = v.and_then(|x| x.get(key));
    }
    v.and_then(Value::as_int)
        .and_then(|i| usize::try_from(i).ok())
        .unwrap_or(0)
}

fn git_ls(root: &Path, pathspec: &str) -> Result<usize> {
    let output = Command::new("git")
        .args(["ls-files", "--", pathspec])
        .current_dir(root)
        .output()
        .map_err(|e| Error::io(Path::new("git"), e))?;
    if !output.status.success() {
        return Err(Error::Command(format!("git ls-files {pathspec} failed")));
    }
    Ok(String::from_utf8_lossy(&output.stdout).lines().count())
}

fn toml_doc(path: &Path) -> Result<toml::Table> {
    py::read_text(path)?
        .parse::<toml::Table>()
        .map_err(|e| Error::parse(path, e.to_string()))
}

/// Recompute every "At a glance" value whose footnote reads a committed file.
pub fn glance(root: &Path) -> Result<Glance> {
    let mut g = Glance::default();
    let manifest = toml_doc(&root.join("Cargo.toml"))?;
    let workspace = manifest
        .get("workspace")
        .and_then(toml::Value::as_table)
        .ok_or_else(|| Error::Shape("Cargo.toml: no [workspace]".into()))?;
    g.version = workspace
        .get("package")
        .and_then(|p| p.get("version"))
        .and_then(toml::Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let list = |key: &str| -> Vec<String> {
        workspace
            .get(key)
            .and_then(toml::Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|m| m.as_str().map(str::to_owned))
                    .collect()
            })
            .unwrap_or_default()
    };
    let members = list("members");
    g.members = members.len();
    g.default_members = list("default-members").len();
    for member in &members {
        let package = toml_doc(&root.join(member).join("Cargo.toml"))?;
        let version = match package.get("package").and_then(|p| p.get("version")) {
            Some(toml::Value::String(v)) => v.clone(),
            _ => g.version.clone(),
        };
        if member.starts_with("crates/smartcluster/") {
            g.vendored += 1;
            g.vendored_version = version.clone();
        }
        *g.versions.entry(version).or_default() += 1;
    }

    let kernel = py::read_text(&root.join("crates/jope-cli/src/kernel.rs"))?;
    let arm = re(r#"^\s+"([a-z0-9+-]+)"(?: \| "[a-z]+")? => vec!\["#);
    g.emit_targets = kernel
        .split("fn languages")
        .nth(1)
        .unwrap_or("")
        .lines()
        .take_while(|l| !l.starts_with('}'))
        .filter_map(|l| arm.captures(l))
        .filter(|c| &c[1] != "all")
        .count();

    let inventory = json(root, "docs/parity/test-inventory.json")?;
    g.rust_tests = int_at(&inventory, &["rust", "count"]);
    g.rust_ignored = int_at(&inventory, &["rust", "ignored"]);
    g.rust_packages = len_at(&inventory, &["rust", "packages"]);
    g.python_tests = int_at(&inventory, &["python", "count"]);

    let ci = py::read_text(&root.join(".github/workflows/ci.yml"))?;
    let job = re(r"^  [A-Za-z0-9_-]+:\s*$");
    let mut in_jobs = false;
    for line in ci.lines() {
        if !line.starts_with(' ') && !line.is_empty() {
            in_jobs = line.starts_with("jobs:");
            continue;
        }
        if in_jobs && job.is_match(line) {
            g.ci_jobs += 1;
        }
    }
    let coverage = json(root, "deploy/ci/coverage.json")?;
    g.coverage_items = len_at(&coverage, &["items"]);
    g.coverage_lanes = len_at(&coverage, &["lanes"]);
    g.full_families = coverage
        .get("lanes")
        .and_then(Value::as_dict)
        .map_or(0, |d| d.keys().filter(|k| k.starts_with("full-")).count());

    let matrix = py::read_text(&root.join("docs/parity/capability-matrix.md"))?;
    let total = re(r"^\| \*\*total\*\* \|((?: \*\*\d+\*\* \|)+)\s*$");
    let cell = re(r"\d+");
    g.matrix = matrix
        .lines()
        .find_map(|l| total.captures(l))
        .map(|c| {
            cell.find_iter(&c[1])
                .filter_map(|m| m.as_str().parse().ok())
                .collect()
        })
        .unwrap_or_default();

    g.synthetic = len_at(
        &json(root, "datasets/synthetic/catalog.json")?,
        &["datasets"],
    );
    g.test_datasets = len_at(&json(root, "datasets/manifest.json")?, &["members"]);
    g.python_files = git_ls(root, "*.py")?;
    g.python_under_python = git_ls(root, "python/*.py")?;
    g.adrs = git_ls(root, "docs/adr/ADR-*.md")?;
    Ok(g)
}

/// Rule 6: each README statement that prints one of the recomputed values.
pub fn check_readme(readme: &str, g: &Glance) -> Vec<String> {
    let mut problems = vec![];
    let main_crates = g.versions.get(&g.version).copied().unwrap_or(0);
    let expect_row = |row: &str, parts: Vec<String>, problems: &mut Vec<String>| {
        let Some(line) = readme.lines().find(|l| l.starts_with(row)) else {
            problems.push(format!("{README}: no row starting {row:?}"));
            return;
        };
        for part in parts {
            if !line.contains(&part) {
                problems.push(format!("{README}: {row:?} must say {part:?}"));
            }
        }
    };
    expect_row(
        "| What |",
        vec![format!("| {} |", g.version)],
        &mut problems,
    );
    expect_row(
        "| `jope` with no arguments |",
        vec![format!("prints `jope {}`", g.version)],
        &mut problems,
    );
    expect_row(
        "| Workspace |",
        vec![
            format!("{} members", g.members),
            format!("{main_crates} crates at {}", g.version),
            format!(
                "{} vendored SmartCluster crates at {}",
                g.vendored, g.vendored_version
            ),
            format!("{} are `default-members`", g.default_members),
        ],
        &mut problems,
    );
    expect_row(
        "| Emit targets |",
        vec![format!("| {}: ", g.emit_targets)],
        &mut problems,
    );
    expect_row(
        "| Rust tests |",
        vec![
            format!(
                "{} listed in {} packages",
                grouped(g.rust_tests),
                g.rust_packages
            ),
            format!("{} of them ignored", g.rust_ignored),
        ],
        &mut problems,
    );
    expect_row(
        "| Python tests |",
        vec![format!("| {}, ", grouped(g.python_tests))],
        &mut problems,
    );
    expect_row(
        "| CI |",
        vec![
            format!("{} jobs per pull request", g.ci_jobs),
            format!("{} `full-main-cpu` families", g.full_families),
        ],
        &mut problems,
    );
    expect_row(
        "| Coverage map |",
        vec![format!(
            "{} items over {} lanes",
            grouped(g.coverage_items),
            g.coverage_lanes
        )],
        &mut problems,
    );
    let m = |i: usize| g.matrix.get(i).copied().unwrap_or(0);
    expect_row(
        "| Capability matrix |",
        vec![format!(
            "{} rows: {} moved, {} new, {} pending, {} dropped, {} lost",
            m(0),
            m(1),
            m(2),
            m(3),
            m(4),
            m(5)
        )],
        &mut problems,
    );
    expect_row(
        "| Datasets in Git |",
        vec![
            format!("{} synthetic fixtures", g.synthetic),
            format!("{} test datasets", g.test_datasets),
        ],
        &mut problems,
    );
    expect_row(
        "| Python still tracked |",
        vec![format!(
            "{} `.py` files, {} of them under `python/`",
            grouped(g.python_files),
            g.python_under_python
        )],
        &mut problems,
    );
    expect_row(
        "| Architecture decision records |",
        vec![format!("| {} ", g.adrs)],
        &mut problems,
    );
    for prose in [
        format!("# jope {}", g.version),
        format!("source of v{};", g.version),
        format!(
            "CI is `.github/workflows/ci.yml`: {} jobs",
            words(g.ci_jobs)
        ),
        format!(
            "{} generated synthetic fixtures and {} tiny test datasets",
            words(g.synthetic),
            words(g.test_datasets)
        ),
        format!("The {} workspace members", g.members),
    ] {
        if !readme.contains(&prose) {
            problems.push(format!("{README}: must say {prose:?}"));
        }
    }
    problems
}

/// The file a `ROOT:path[:lines|#jq]` source part names, or `None` when its root is undeclared
/// or the part names no single file (a commit, a brace range, a glob).
fn source_file(part: &str, roots: &BTreeMap<String, String>) -> Option<std::path::PathBuf> {
    let (root, rest) = part.split_once(':')?;
    let base = roots.get(root)?;
    let end = rest.find([':', '#', ' ']).unwrap_or(rest.len());
    let path = &rest[..end];
    if path.is_empty() || path.contains(['{', '*']) {
        return None;
    }
    Some(Path::new(base).join(path))
}

/// Host mode: re-hash every claim's resolvable source files against the pinned SHA-256. A claim
/// passes when one of its files matches; it is unresolvable when none of its parts names a file
/// present here (a derived value, a directory). A declared root that is absent fails: this host
/// does not hold the evidence.
pub fn check_sources(ledger_text: &str) -> Result<(Vec<String>, String)> {
    use sha2::{Digest, Sha256};
    let doc = pyjson::parse_str(ledger_text).map_err(|m| Error::parse(Path::new(LEDGER), m))?;
    let roots: BTreeMap<String, String> = doc
        .get("source_conventions")
        .and_then(|c| c.get("roots"))
        .and_then(Value::as_dict)
        .map(|d| {
            d.iter()
                .filter_map(|(k, v)| {
                    let path = v.as_str()?.split(' ').next()?;
                    Some((k.to_owned(), path.to_owned()))
                })
                .collect()
        })
        .unwrap_or_default();
    let mut problems = vec![];
    for (name, path) in &roots {
        if !Path::new(path).is_dir() {
            problems.push(format!(
                "sources: root {name} ({path}) is not a directory on this host"
            ));
        }
    }
    let (mut matched, mut unresolvable) = (0, vec![]);
    let mut hashes: BTreeMap<std::path::PathBuf, String> = BTreeMap::new();
    for claim in doc
        .get("claims")
        .and_then(Value::as_list)
        .into_iter()
        .flatten()
    {
        let id = field(claim, "id");
        let pin = field(claim, "sha256");
        let mut resolved = 0;
        let mut ok = false;
        for part in field(claim, "source").split(';').map(str::trim) {
            let Some(file) = source_file(part, &roots).filter(|f| f.is_file()) else {
                continue;
            };
            resolved += 1;
            let digest = match hashes.get(&file) {
                Some(d) => d.clone(),
                None => {
                    let bytes = std::fs::read(&file).map_err(|e| Error::io(&file, e))?;
                    let d: String = Sha256::digest(&bytes)
                        .iter()
                        .map(|b| format!("{b:02x}"))
                        .collect();
                    hashes.insert(file.clone(), d.clone());
                    d
                }
            };
            ok |= digest == pin;
        }
        match (resolved, ok) {
            (0, _) => unresolvable.push(id.to_owned()),
            (_, true) => matched += 1,
            (_, false) => problems.push(format!(
                "sources: {id} pins {pin}, which none of its {resolved} source file(s) here hashes to"
            )),
        }
    }
    let summary = format!(
        "sources: {matched} claims re-hashed to their pin; {} unresolvable here ({})",
        unresolvable.len(),
        unresolvable.join(", ")
    );
    Ok((problems, summary))
}

/// The report and whether the paper, its Figure 1 and the README agree with their records; with
/// `sources`, also whether every claim source on this host still hashes to its pin. With
/// `write_figure`, Figure 1 is re-rendered first.
pub fn check(root: &Path, sources: bool, write_figure: bool) -> Result<(String, bool)> {
    let writeup = py::read_text(&root.join(WRITEUP))?;
    let ledger_text = py::read_text(&root.join(LEDGER))?;
    let trace = py::read_text(&root.join(TRACE))?;
    let (mut problems, mut summary) = check_paper(&writeup, &ledger_text, &trace)?;
    problems.extend(crate::paper_figure::check(
        root,
        &writeup,
        &ledger_text,
        write_figure,
    )?);
    summary = format!("{summary}; Figure 1 against its data and C26, C27 and C76");
    if sources {
        let (found, line) = check_sources(&ledger_text)?;
        problems.extend(found);
        summary = format!("{summary}\n{line}");
    }
    let g = glance(root)?;
    problems.extend(check_readme(&py::read_text(&root.join(README))?, &g));
    let mut out = String::new();
    for p in &problems {
        out.push_str(p);
        out.push('\n');
    }
    out.push_str(&format!(
        "{summary}; README: version {} and the At a glance counts\n",
        g.version
    ));
    out.push_str(if problems.is_empty() {
        "paper-trace: OK\n"
    } else {
        "paper-trace: FAIL\n"
    });
    Ok((out, problems.is_empty()))
}

#[cfg(test)]
mod tests {
    use super::*;

    const LEDGER_OK: &str = r#"{"source_conventions": {"roots": {"LANE": "/x"}}, "claims": [
      {"id": "C01", "statement": "PolicyLM has 893,957 parameters", "value": "893,957", "tex": "893{,}957",
       "source": "LANE:a.json#.p", "sha256": "0000000000000000000000000000000000000000000000000000000000000000", "status": "established"},
      {"id": "C02", "statement": "Learning rate 3e-4, floor 1e-6; KL 0.004-0.011 over 8 epochs", "value": "0.0912345",
       "source": "LANE:b.json", "sha256": "1111111111111111111111111111111111111111111111111111111111111111", "status": "non-qualifying"},
      {"id": "C03", "statement": "E1 primary", "value": "pending", "source": "LANE:c.json",
       "sha256": "2222222222222222222222222222222222222222222222222222222222222222", "status": "pending"}
    ]}"#;

    const WRITEUP_OK: &str = "\
# Title

## 1. Model

PolicyLM has 893,957 parameters [C01]. Section 4.4 says so (Chen et al., 2021).
The rate starts at $3\\times10^{-4}$ and stops at $10^{-6}$; KL stayed 0.004–0.011 (R23, v2) [C02].
| Arm | KL |
|---|---|
| anchored | 0.09 |

### 10.4 Pending (pre-registered or planned)

E1 is pre-registered [C03].

| Id | Result | Status |
|---|---|---|
| C03 | E1 primary | pending |

## Appendix G

`claims.json` holds 3 claims: 1 established, 0 preliminary, 1 non-qualifying and 1 pending.
Illustrative: q = log 10 ≈ 2.303.
";

    const TRACE_OK: &str = "\
## Claim trace

| Id | Status | Printed | Source |
|---|---|---|---|
| C01 | established | 893,957 | LANE:a.json |
| C02 | non-qualifying | 0.004–0.011 | LANE:b.json |
| C03 | pending | — | LANE:c.json |

## Exempt numbers

| Text | Numbers | Why |
|---|---|---|
| `q = log 10 ≈ 2.303` | 10, 2.303 | illustrative arithmetic |
";

    fn problems(writeup: &str, ledger: &str, trace: &str) -> Vec<String> {
        check_paper(writeup, ledger, trace).expect("parses").0
    }

    #[test]
    fn consistent_documents_pass() {
        assert_eq!(
            problems(WRITEUP_OK, LEDGER_OK, TRACE_OK),
            Vec::<String>::new()
        );
    }

    #[test]
    fn a_drifted_printed_number_fails() {
        let w = WRITEUP_OK.replace("893,957 parameters", "893,958 parameters");
        let p = problems(&w, LEDGER_OK, TRACE_OK);
        assert!(
            p.iter().any(|m| m.contains("893958 is not in any claim")),
            "{p:?}"
        );
    }

    #[test]
    fn a_drifted_ledger_value_fails() {
        let l = LEDGER_OK
            .replace("893,957 parameters", "894,000 parameters")
            .replace("\"893,957\"", "\"894,000\"")
            .replace("893{,}957", "894{,}000");
        let p = problems(WRITEUP_OK, &l, TRACE_OK);
        assert!(
            p.iter().any(|m| m.contains("893957 is not in any claim")),
            "{p:?}"
        );
    }

    #[test]
    fn rounding_is_accepted_only_at_the_printed_precision() {
        // 0.0912345 prints as 0.09; 0.10 would not.
        assert_eq!(
            problems(WRITEUP_OK, LEDGER_OK, TRACE_OK),
            Vec::<String>::new()
        );
        let w = WRITEUP_OK.replace("| anchored | 0.09 |", "| anchored | 0.10 |");
        assert!(!problems(&w, LEDGER_OK, TRACE_OK).is_empty());
    }

    #[test]
    fn a_range_dash_is_not_a_minus_sign() {
        let printed: Vec<String> = printed_numbers("0.004–0.011 and −0.0625 to 3.75\\times10^{-5}")
            .into_iter()
            .map(|p| p.text)
            .collect();
        assert_eq!(printed, ["0.004", "0.011", "-0.0625", "3.75e-5"]);
    }

    #[test]
    fn identifiers_and_references_are_not_numbers() {
        let line = "R23 and C26 (Section 10.3.1; Loshchilov and Hutter, 2017) at 2026-10-13T00:06Z, `v2/r3.1` p50 float32 SHA-256 [C140–C143]. The 2026-10-12 direction";
        assert_eq!(printed_numbers(line), vec![]);
    }

    /// "At 2,000" (read as "At 2000" once grouping commas go) is a number to check, not an
    /// author-year reference; the citation shapes around it stay excluded.
    #[test]
    fn a_capitalised_word_before_a_number_is_not_a_reference() {
        let line = "At 2,000 evaluations per hour (Loshchilov and Hutter, 2017; Chen et al., 2021), as Parisi et al. (2019) note.";
        let texts: Vec<String> = printed_numbers(line).into_iter().map(|p| p.text).collect();
        assert_eq!(texts, ["2000"]);
        let w = WRITEUP_OK.replace(
            "PolicyLM has 893,957 parameters [C01].",
            "PolicyLM has 893,957 parameters [C01]. At 2,048 evaluations per hour [C01].",
        );
        let p = problems(&w, LEDGER_OK, TRACE_OK);
        assert!(
            p.iter().any(|m| m.contains("2048 is not in any claim")),
            "{p:?}"
        );
    }

    #[test]
    fn totals_and_pending_table_must_match_the_ledger() {
        let w = WRITEUP_OK.replace("1 non-qualifying", "2 non-qualifying");
        let p = problems(&w, LEDGER_OK, TRACE_OK);
        assert!(p.iter().any(|m| m.contains("Appendix G")), "{p:?}");
        let w = WRITEUP_OK.replace("| C03 | E1 primary | pending |\n", "");
        let p = problems(&w, LEDGER_OK, TRACE_OK);
        assert!(p.iter().any(|m| m.contains("Section 10.4")), "{p:?}");
    }

    #[test]
    fn citations_must_resolve_and_cover_the_ledger() {
        let w = WRITEUP_OK.replace("[C01]", "[C09]");
        let p = problems(&w, LEDGER_OK, TRACE_OK);
        assert!(p.iter().any(|m| m.contains("cites C09")), "{p:?}");
        assert!(
            p.iter()
                .any(|m| m.contains("C01 is in the ledger but never cited")),
            "{p:?}"
        );
    }

    #[test]
    fn the_trace_must_follow_the_ledger() {
        let t = TRACE_OK.replace("| C02 | non-qualifying |", "| C02 | established |");
        let p = problems(WRITEUP_OK, LEDGER_OK, &t);
        assert!(
            p.iter()
                .any(|m| m.contains("C02 is \"established\" in the trace")),
            "{p:?}"
        );
        let t = TRACE_OK.replace("| C03 | pending | — | LANE:c.json |\n", "");
        let p = problems(WRITEUP_OK, LEDGER_OK, &t);
        assert!(
            p.iter().any(|m| m.contains("no Claim trace row for C03")),
            "{p:?}"
        );
        let t = format!("{TRACE_OK}\n## Fresh-clone run\n\nFRESH_RUN_PLACEHOLDER\n");
        let p = problems(WRITEUP_OK, LEDGER_OK, &t);
        assert!(
            p.iter()
                .any(|m| m.contains("unfilled placeholder: FRESH_RUN_PLACEHOLDER")),
            "{p:?}"
        );
    }

    #[test]
    fn a_stale_exemption_fails() {
        let w = WRITEUP_OK.replace("Illustrative: q = log 10 ≈ 2.303.\n", "");
        let p = problems(&w, LEDGER_OK, TRACE_OK);
        assert!(p.iter().any(|m| m.contains("no longer occurs")), "{p:?}");
    }

    #[test]
    fn ledger_rules() {
        let l = LEDGER_OK
            .replace("\"source\": \"LANE:c.json\"", "\"source\": \"NOPE:c.json\"")
            .replace(
                "\"sha256\": \"2222222222222222222222222222222222222222222222222222222222222222\"",
                "\"sha256\": \"\"",
            );
        let p = problems(WRITEUP_OK, &l, TRACE_OK);
        assert!(p.iter().any(|m| m.contains("root NOPE")), "{p:?}");
        assert!(
            p.iter().any(|m| m.contains("C03 pins no 64-hex sha256")),
            "{p:?}"
        );
    }

    #[test]
    fn readme_glance_drift_fails() {
        let g = Glance {
            version: "0.3.0".into(),
            members: 26,
            default_members: 8,
            versions: [("0.3.0".to_owned(), 19), ("8.0.0".to_owned(), 7)].into(),
            vendored: 7,
            vendored_version: "8.0.0".into(),
            emit_targets: 6,
            rust_tests: 4189,
            rust_ignored: 66,
            rust_packages: 26,
            python_tests: 182,
            ci_jobs: 9,
            full_families: 11,
            coverage_items: 678,
            coverage_lanes: 24,
            matrix: vec![824, 353, 61, 1, 402, 7],
            synthetic: 40,
            test_datasets: 10,
            python_files: 1184,
            python_under_python: 159,
            adrs: 6,
        };
        let readme = "\
| What | 0.3.0 |
| `jope` with no arguments | prints `jope 0.3.0` <sup>1</sup> |
| Workspace | 26 members: 19 crates at 0.3.0 and 7 vendored SmartCluster crates at 8.0.0; 8 are `default-members` |
| Emit targets | 6: `python` |
| Rust tests | 4,189 listed in 26 packages, 66 of them ignored |
| Python tests | 182, found by a static scan |
| CI | 9 jobs per pull request in `ci.yml`; 11 `full-main-cpu` families |
| Coverage map | 678 items over 24 lanes |
| Capability matrix | 824 rows: 353 moved, 61 new, 1 pending, 402 dropped, 7 lost |
| Datasets in Git | 40 synthetic fixtures (10 each) and 10 test datasets |
| Python still tracked | 1,184 `.py` files, 159 of them under `python/` |
| Architecture decision records | 6 <sup>10</sup> |
# jope 0.3.0
source of v0.3.0; the release
Pull-request CI is `.github/workflows/ci.yml`: nine jobs on runners.
forty generated synthetic fixtures and ten tiny test datasets.
The 26 workspace members, by directory.
";
        assert_eq!(check_readme(readme, &g), Vec::<String>::new());
        let stale = readme.replace("4,189 listed", "4,100 listed");
        assert!(!check_readme(&stale, &g).is_empty());
        let stale = readme.replace("nine jobs", "eight jobs");
        assert!(!check_readme(&stale, &g).is_empty());
    }

    #[test]
    fn number_words_and_grouping() {
        assert_eq!(words(9), "nine");
        assert_eq!(words(40), "forty");
        assert_eq!(words(26), "twenty-six");
        assert_eq!(grouped(4189), "4,189");
        assert_eq!(grouped(678), "678");
        assert_eq!(grouped(1234567), "1,234,567");
    }

    /// The committed paper, ledger, trace, Figure 1 and README agree with each other and with the
    /// files the README's footnotes name. A number edited on one side only fails here.
    #[test]
    fn the_committed_documents_agree() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let (report, ok) = check(&root, false, false).expect("readable");
        assert!(ok, "{report}");

        // C140's old TeX bound widened the accepted set despite the corrected record.
        // Keep the recorded bound accepted, and reject the old double-rounded value.
        let ledger_text = std::fs::read_to_string(root.join(LEDGER)).expect("ledger readable");
        let mut issues = Vec::new();
        let claims = ledger(&ledger_text, &mut issues).expect("ledger parses");
        assert!(issues.is_empty(), "{issues:?}");
        let c140 = claims
            .iter()
            .find(|claim| claim.id == "C140")
            .expect("C140");
        assert!(accounted(&printed_numbers("0.8963")[0], &c140.numbers));
        assert!(!accounted(&printed_numbers("0.8964")[0], &c140.numbers));
    }

    #[test]
    fn source_parts_resolve_to_single_files_only() {
        let roots: BTreeMap<String, String> = [("LANE".to_owned(), "/lane".to_owned())].into();
        let file = |p: &str| source_file(p, &roots).map(|f| f.display().to_string());
        assert_eq!(
            file("LANE:a/b.json#.x.y").as_deref(),
            Some("/lane/a/b.json")
        );
        assert_eq!(file("LANE:c.md:4-5,9").as_deref(), Some("/lane/c.md"));
        assert_eq!(
            file("LANE:d.jsonl (rows with x)").as_deref(),
            Some("/lane/d.jsonl")
        );
        assert_eq!(file("LANE:e-0000{10..40}.json#.loss"), None);
        assert_eq!(file("LANE:r2*.json"), None);
        assert_eq!(file("NOPE:a.json"), None);
        assert_eq!(file("derived from C10 and C23"), None);
    }
}
