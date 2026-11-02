use super::*;

pub const WRITEUP: &str = "docs/paper/TECHNICAL_WRITEUP.md";
pub const LEDGER: &str = "docs/paper/claims.json";
pub const TRACE: &str = "docs/paper/NUMBERS_TRACE.md";
pub const README: &str = "README.md";
pub const EXPERIMENTS_JSONL: &str = "docs/experiments/experiments.jsonl";
pub const EXPERIMENTS_README: &str = "docs/experiments/README.md";
pub const EXPERIMENTS_SYNTHESIS: &str = "docs/experiments/SYNTHESIS_AND_NEXT_EXPERIMENTS.md";
pub(super) const REPLAY_JSON: &str =
    "docs/internal/clover/exp-008-quantile-parity-replay.20261017.json";
pub(super) const SAMECUT_CSV: &str = "assets/same-cut/v2/leaderboard-jope-same-cut-v1-deep.csv";
pub(super) const SVG_PATHS: [&str; 4] = [
    "docs/paper/figures/why-it-matters.svg",
    "docs/paper/figures/clover-why-it-matters-3panel.svg",
    "docs/paper/figures/parity-census-and-exp008-replay.svg",
    "docs/paper/figures/readme-clover-overview.svg",
];

const STATUSES: [&str; 4] = ["established", "preliminary", "non-qualifying", "pending"];

pub(super) fn re(pattern: &str) -> Regex {
    Regex::new(pattern).expect("static pattern")
}

static CITATION: LazyLock<Regex> = LazyLock::new(|| re(r"\[C\d+[^\]]*\]"));
static CITED_ID: LazyLock<Regex> = LazyLock::new(|| re(r"C(\d+)(?:\s*[–-]\s*C?(\d+))?"));
static GROUPED: LazyLock<Regex> = LazyLock::new(|| re(r"(\d),(\d{3})\b"));
static SCIENTIFIC: LazyLock<Regex> =
    LazyLock::new(|| re(r"(\d+(?:\.\d+)?)\s*(?:\\times|×)\s*10\^\{?(-?\d+)\}?"));
static POWER: LazyLock<Regex> = LazyLock::new(|| re(r"\b10\^\{?(-?\d+)\}?"));
/// `±10`, `$\pm 10$` and `+/-10` print both bounds.
static PLUS_MINUS: LazyLock<Regex> =
    LazyLock::new(|| re(r"(?:\\pm|±|\+/-)\s*(\d+(?:\.\d+)?(?:e-?\d+)?)"));
static NOT_NUMBERS: LazyLock<Regex> = LazyLock::new(|| {
    re(concat!(
        r"`[^`]*`",
        r"|\[C\d+[^\]]*\]",
        r"|\b[A-Z]{1,2}\d+\b",
        r"|\b[vr]\d+(?:\.\d+)*\b",
        r"|\bp\d{2}\b",
        r"|\bfloat\d+\b",
        r"|SHA-256",
        r"|\b(?:Sections?|Figures?|Appendix|Phase|Equations?) \d+(?:\.\d+)*(?:\([a-z]\))?(?:(?:, | and | to |–|-)\d+(?:\.\d+)*(?:\([a-z]\))?)*",
        // Author-year references only in their citation shapes: "(Chen et al., 2021;", "Hutter,
        // 2017)" and the narrative "Parisi et al. (2019)". A bare capitalised word before a
        // number ("At 2,000", read as "At 2000" once grouping commas go) is not a reference.
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

/// Commas between digit groups, LaTeX thin spaces, Unicode minus and en dash, `a×10^{b}` (also
/// `a{\times}10^{b}`) and `±a` are written one way on both sides of a comparison.
pub(super) fn normalize(text: &str) -> String {
    let mut text = text
        .replace("{,}", "")
        .replace("{\\times}", "\\times")
        .replace(['\u{2212}', '\u{2013}'], "-");
    loop {
        let next = GROUPED.replace_all(&text, "${1}${2}").into_owned();
        if next == text {
            break;
        }
        text = next;
    }
    let text = SCIENTIFIC.replace_all(&text, "${1}e${2}");
    let text = POWER.replace_all(&text, "1e${1}");
    PLUS_MINUS.replace_all(&text, "-${1} ${1}").into_owned()
}

/// One printed number: its text, value and the half-width of the interval it rounds from (zero
/// for an integer, which must match exactly).
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Printed {
    pub(super) text: String,
    pub(super) value: f64,
    pub(super) half_ulp: f64,
}

/// The numbers in already-normalized text. A sign directly after a digit, letter, `.` or `)` is a
/// range dash, not a sign; a number glued to a letter or `_` is part of an identifier.
pub(super) fn numbers(text: &str) -> Vec<Printed> {
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
pub(super) fn printed_numbers(line: &str) -> Vec<Printed> {
    let line = normalize(line);
    let line = DATES.replace_all(&line, " ");
    numbers(&NOT_NUMBERS.replace_all(&line, " "))
}

/// Whether a printed number is one of the record's numbers, or that number at the precision
/// printed.
pub(super) fn accounted(printed: &Printed, record: &[Printed]) -> bool {
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
pub(super) struct Claim {
    pub(super) id: String,
    pub(super) status: String,
    pub(super) numbers: Vec<Printed>,
}

pub(super) fn field<'a>(claim: &'a Value, key: &str) -> &'a str {
    claim.get(key).and_then(Value::as_str).unwrap_or("")
}

/// The ledger's claims, with the problems rule 1 finds.
pub(super) fn ledger(text: &str, problems: &mut Vec<String>) -> Result<Vec<Claim>> {
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
        // The write-up is checked against the statement and value only. `tex` is the LaTeX
        // display of the same claim, so every number it prints must be one of theirs at the
        // precision it prints; a stale `tex` is reported rather than accepted.
        let record = numbers(&normalize(
            &[field(claim, "statement"), field(claim, "value")].join(" | "),
        ));
        for n in numbers(&normalize(field(claim, "tex"))) {
            if !accounted(&n, &record) {
                problems.push(format!(
                    "ledger: {id} tex prints {}, which is not in its statement or value",
                    n.text
                ));
            }
        }
        claims.push(Claim {
            id,
            status,
            numbers: record,
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
