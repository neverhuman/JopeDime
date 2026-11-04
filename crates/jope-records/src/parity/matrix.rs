use super::*;

pub const ARCHIVE: &str = "s3://veox-jopedime-archive-use2/git/20260918-jope-dime-all-refs.bundle";
pub const MATRIX: &str = "docs/parity/capability-matrix.md";
const COUNT_CLASSES: [&str; 5] = ["moved", "new", "pending", "dropped", "lost"];

/// Why the `## Counts` table is gated at all. Required by the check, so that regenerating the
/// matrix cannot silently drop the rule and hand-editing cannot either.
pub const COUNTS_PREAMBLE: [&str; 4] = [
    "Recomputed from the rows below and checked by `deploy/ci/parity-matrix.sh`: a tally that disagrees",
    "with them fails the gate. Two lanes adding a row of the same kind each bump the same cell to the",
    "same number, so git reports no conflict — update this table from the rows, not from the last value",
    "you saw.",
];

/// One `## Rows` line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Row {
    pub kind: String,
    pub old: String,
    pub new: String,
    pub test: String,
    pub status: String,
    pub notes: String,
}

pub fn split_row(line: &str) -> Vec<String> {
    let mut parts = vec![];
    let mut current = String::new();
    let mut chars = line.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\\' && chars.peek() == Some(&'|') {
            current.push('|');
            chars.next();
        } else if ch == '|' {
            parts.push(std::mem::take(&mut current));
        } else {
            current.push(ch);
        }
    }
    parts.push(current);
    if parts.len() < 2 {
        return vec![];
    }
    parts[1..parts.len() - 1]
        .iter()
        .map(|p| py::strip(p).to_owned())
        .collect()
}

pub fn read_matrix(path: &Path) -> Result<Vec<Row>> {
    Ok(parse_matrix(&py::read_text(path)?))
}

/// The six-cell table rows of a matrix text.
pub fn parse_matrix(text: &str) -> Vec<Row> {
    let mut rows = vec![];
    for line in py::splitlines(text) {
        if !line.starts_with("| ") || line.starts_with("| kind ") || line.starts_with("|---") {
            continue;
        }
        let cells = split_row(line);
        if cells.len() != 6 {
            continue;
        }
        let mut cells = cells.into_iter();
        let mut next = || cells.next().expect("six cells");
        rows.push(Row {
            kind: next(),
            old: next(),
            new: next(),
            test: next(),
            status: next(),
            notes: next(),
        });
    }
    rows
}

/// `PENDING_RE.match(status)` (`^pending-WP(\d+)$`, Python's Unicode `\d`): classifying a status
/// reads no number and never refused.
fn is_pending(status: &str) -> bool {
    status
        .strip_prefix("pending-WP")
        .is_some_and(py::is_decimal_run)
}

/// `int(PENDING_RE.match(status).group(1))` of a pending status through the strict number grammar
/// ([`py::decimal_int`]): the work-package number as ASCII decimal, or the refusal. The check
/// crashed where Python 3.12 refused more than 4300 digits; a number written in other Unicode
/// digits, which Python read, is refused as well (stricter).
pub(super) fn pending_int(status: &str, place: &str) -> Result<String> {
    let digits = status.strip_prefix("pending-WP").expect("a pending status");
    py::decimal_int(digits)
        .map_err(|e| Error::Shape(format!("{place}: pending-WP<n>: ValueError: {e}")))
}

pub fn status_class(status: &str) -> &'static str {
    match status {
        "moved" => "moved",
        "new" => "new",
        "lost" => "lost",
        s if is_pending(s) => "pending",
        s if s.starts_with("dropped:") => "dropped",
        _ => "invalid",
    }
}

/// The body of the `## Counts` table and the covering-test tally, recomputed from `rows`.
pub fn counts_lines(rows: &[Row]) -> (Vec<String>, usize) {
    let mut by_kind: BTreeMap<&str, BTreeMap<&str, usize>> = BTreeMap::new();
    let mut statuses: BTreeMap<&str, usize> = BTreeMap::new();
    let mut no_test = 0;
    for r in rows {
        let class = status_class(&r.status);
        *by_kind
            .entry(&r.kind)
            .or_default()
            .entry(class)
            .or_default() += 1;
        *statuses.entry(class).or_default() += 1;
        if r.test == "none" && (class == "moved" || class == "new") {
            no_test += 1;
        }
    }
    let mut lines = vec![];
    for (kind, counts) in &by_kind {
        let total: usize = counts.values().sum();
        let cells: Vec<String> = COUNT_CLASSES
            .iter()
            .map(|c| counts.get(c).copied().unwrap_or(0).to_string())
            .collect();
        lines.push(format!("| {kind} | {total} | {} |", cells.join(" | ")));
    }
    let cells: Vec<String> = COUNT_CLASSES
        .iter()
        .map(|c| format!("**{}**", statuses.get(c).copied().unwrap_or(0)))
        .collect();
    lines.push(format!(
        "| **total** | **{}** | {} |",
        rows.len(),
        cells.join(" | ")
    ));
    (lines, no_test)
}

pub(super) fn counts_row_kind(line: &str) -> String {
    split_row(line)
        .into_iter()
        .next()
        .unwrap_or_else(|| line.to_owned())
}

/// The `## Counts` section as committed: table lines, covering-test tally (`int()` of what `\d+`
/// captured through the strict number grammar, [`py::decimal_int`]: ASCII decimal, more than 4300
/// digits refused as Python 3.12's `int()` raised, other Unicode digits refused though Python read
/// them), every other line.
pub(super) fn committed_counts(text: &str) -> Result<(Vec<String>, Option<String>, Vec<String>)> {
    let lines = py::splitlines(text);
    let Some(start) = lines.iter().position(|l| *l == "## Counts") else {
        return Ok((vec![], None, vec![]));
    };
    let tally = py::regex(r"^`moved`/`new` rows without a covering test: (\d+)\.$");
    let (mut table, mut prose, mut no_test) = (vec![], vec![], None);
    for line in &lines[start + 1..] {
        if line.starts_with("## ") {
            break;
        }
        let stripped = line.trim_end_matches(py::is_space);
        if stripped.starts_with("| ") && !stripped.starts_with("| kind ") {
            table.push(stripped.to_owned());
            continue;
        }
        if let Some(m) = tally.captures(stripped) {
            no_test = Some(py::decimal_int(&m[1]).map_err(|e| {
                Error::Shape(format!(
                    "## Counts `moved`/`new` rows without a covering test: ValueError: {e}"
                ))
            })?);
            continue;
        }
        if !stripped.is_empty() {
            prose.push(stripped.to_owned());
        }
    }
    Ok((table, no_test, prose))
}

pub(super) fn head_of(new: &str) -> &str {
    py::strip(new.split(" (").next().unwrap_or(""))
}
