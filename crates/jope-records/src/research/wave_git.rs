//! Deterministic, byte-accounted public projections for archived wave evidence.
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

pub const MANIFEST_LIMIT: usize = 12 * 1024;
pub const CARD_LIMIT: usize = 4 * 1024;
pub const CONTEXT_LIMIT: usize = 4 * 1024;
pub const METRICS_LIMIT: usize = 8 * 1024;
pub const INDEX_ROW_LIMIT: usize = 1024;
pub const TOTAL_LIMIT: usize = 32 * 1024;
const CONTEXT_SLOTS: [&str; 10] = [
    "question",
    "hypothesis",
    "controls",
    "execution",
    "effect",
    "stage",
    "changed",
    "counterevidence",
    "falsifier",
    "reconstruction",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Synthetic,
    Shadow,
    Adaptive,
    Confirmatory,
}
impl Stage {
    fn text(self) -> &'static str {
        match self {
            Self::Synthetic => "synthetic",
            Self::Shadow => "shadow",
            Self::Adaptive => "adaptive",
            Self::Confirmatory => "confirmatory",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvidenceRoot {
    pub sha256: String,
    pub member_sha256: String,
    pub version: String,
    pub physical_version: Option<String>,
    pub bytes: u64,
    pub codec: String,
    pub rows: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Number {
    Integer(i64),
    Unsigned(u64),
    Float(f64),
}
impl Number {
    fn json(&self) -> Option<String> {
        match self {
            Self::Integer(v) => Some(v.to_string()),
            Self::Unsigned(v) => Some(v.to_string()),
            Self::Float(v) if v.is_finite() => Some(v.to_string()),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NumericAssertion {
    pub id: String,
    pub value: Number,
    pub unit: String,
    pub numerator: u64,
    pub denominator: u64,
    pub status: String,
    pub source_field: String,
    pub expression: String,
    pub evidence_sha256: String,
    pub evidence_version: String,
    pub export_allowed: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DisclosurePolicy {
    pub role: String,
    pub policy_id: String,
    pub sha256: String,
    pub allowed_fields: BTreeSet<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WaveGitInput {
    pub wave_id: String,
    pub version: String,
    pub source_commit: Option<String>,
    pub training_identity_sha256: String,
    pub archive_manifest_sha256: String,
    pub archive_status: String,
    pub capture_status: String,
    pub stage: Stage,
    pub disposition: String,
    pub ancestry: Vec<String>,
    pub evidence_roots: Vec<EvidenceRoot>,
    pub context: BTreeMap<String, String>,
    pub assertions: Vec<NumericAssertion>,
    pub disclosure: DisclosurePolicy,
    pub additional_review_bytes: usize,
    pub budget_exception: Option<(String, String)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Projection {
    pub manifest: Vec<u8>,
    pub card: Vec<u8>,
    pub context: Vec<u8>,
    pub metrics: Vec<u8>,
    pub index_row: Vec<u8>,
    pub hashes: BTreeMap<String, String>,
    pub byte_counts: BTreeMap<String, usize>,
    pub total_bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectionError(pub String);
impl std::fmt::Display for ProjectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for ProjectionError {}

pub fn compile(input: &WaveGitInput) -> Result<Projection, ProjectionError> {
    validate(input)?;
    let mut roots = input.evidence_roots.clone();
    roots.sort_by(|a, b| a.sha256.cmp(&b.sha256));
    let ancestry = input.ancestry.clone();
    let mut assertions = input.assertions.clone();
    assertions.sort_by(|a, b| a.id.cmp(&b.id));
    let mut context = String::new();
    for key in CONTEXT_SLOTS {
        writeln!(context, "{}: {}", key, input.context[key]).map_err(fmt_err)?;
    }
    let context = context.into_bytes();
    let mut metrics = String::from("{\"assertions\":[");
    for (i, a) in assertions.iter().enumerate() {
        if i > 0 {
            metrics.push(',')
        }
        write!(metrics,"{{\"denominator\":{},\"evidence_sha256\":{},\"evidence_version\":{},\"expression\":{},\"id\":{},\"numerator\":{},\"source_field\":{},\"status\":{},\"unit\":{},\"value\":{}}}",a.denominator,q(&a.evidence_sha256),q(&a.evidence_version),q(&a.expression),q(&a.id),a.numerator,q(&a.source_field),q(&a.status),q(&a.unit),a.value.json().ok_or_else(||ProjectionError("non-finite numeric assertion".into()))?).map_err(fmt_err)?;
    }
    metrics.push_str("],\"schema_version\":1}");
    let metrics = metrics.into_bytes();
    let roots_json = roots
        .iter()
        .map(|r| {
            format!(
                "{{\"bytes\":{},\"codec\":{},\"member_sha256\":{},\"physical_version\":{},\"rows\":{},\"sha256\":{},\"version\":{}}}",
                r.bytes,
                q(&r.codec),
                q(&r.member_sha256),
                opt(&r.physical_version),
                r.rows,
                q(&r.sha256),
                q(&r.version),
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let ancestry_json = ancestry.iter().map(|x| q(x)).collect::<Vec<_>>().join(",");
    let exception = input
        .budget_exception
        .as_ref()
        .map(|(id, reason)| format!("{{\"review_id\":{},\"reason\":{}}}", q(id), q(reason)))
        .unwrap_or_else(|| "null".into());
    let manifest=format!("{{\"archive_manifest_sha256\":{},\"archive_status\":{},\"ancestry\":[{}],\"budget_exception\":{},\"capture_status\":{},\"disclosure_policy\":{{\"policy_id\":{},\"role\":{},\"sha256\":{}}},\"disposition\":{},\"evidence_roots\":[{}],\"git_status\":\"GIT_PUBLICATION_PENDING\",\"schema_version\":1,\"source_commit\":{},\"stage\":{},\"training_identity_sha256\":{},\"version\":{},\"wave_id\":{}}}",q(&input.archive_manifest_sha256),q(&input.archive_status),ancestry_json,exception,q(&input.capture_status),q(&input.disclosure.policy_id),q(&input.disclosure.role),q(&input.disclosure.sha256),q(&input.disposition),roots_json,opt(&input.source_commit),q(input.stage.text()),q(&input.training_identity_sha256),q(&input.version),q(&input.wave_id));
    let card=format!("{{\"archive_status\":{},\"assertion_count\":{},\"capture_status\":{},\"disposition\":{},\"evidence_root_count\":{},\"interpretation\":\"descriptive; no promotion authority\",\"schema_version\":1,\"stage\":{},\"version\":{},\"wave_id\":{}}}",q(&input.archive_status),assertions.len(),q(&input.capture_status),q(&input.disposition),roots.len(),q(input.stage.text()),q(&input.version),q(&input.wave_id)).into_bytes();
    let manifest = manifest.into_bytes();
    let mut hashes = BTreeMap::new();
    let mut byte_counts = BTreeMap::new();
    for (name, bytes) in [
        ("wave.manifest.json", &manifest),
        ("card.json", &card),
        ("context.md", &context),
        ("metrics.json", &metrics),
    ] {
        hashes.insert(name.into(), hash(bytes));
        byte_counts.insert(name.into(), bytes.len());
    }
    let mut index=format!("{{\"archive_status\":{},\"capture_status\":{},\"card_sha256\":{},\"context_sha256\":{},\"git_status\":\"GIT_PUBLICATION_PENDING\",\"manifest_sha256\":{},\"metrics_sha256\":{},\"version\":{},\"wave_id\":{}}}",q(&input.archive_status),q(&input.capture_status),q(&hashes["card.json"]),q(&hashes["context.md"]),q(&hashes["wave.manifest.json"]),q(&hashes["metrics.json"]),q(&input.version),q(&input.wave_id));
    index.push('\n');
    let index_row = index.into_bytes();
    let total_bytes = manifest.len()
        + card.len()
        + context.len()
        + metrics.len()
        + index_row.len()
        + input.additional_review_bytes;
    let over = manifest.len() > MANIFEST_LIMIT
        || card.len() > CARD_LIMIT
        || context.len() > CONTEXT_LIMIT
        || metrics.len() > METRICS_LIMIT
        || index_row.len() > INDEX_ROW_LIMIT
        || total_bytes > TOTAL_LIMIT;
    if over
        && input
            .budget_exception
            .as_ref()
            .is_none_or(|(id, reason)| id.trim().is_empty() || reason.trim().is_empty())
    {
        return Err(ProjectionError(format!("byte budget exceeded: manifest={} card={} context={} metrics={} index={} total={total_bytes}; reviewed exception required",manifest.len(),card.len(),context.len(),metrics.len(),index_row.len())));
    }
    if manifest.len() > MANIFEST_LIMIT
        || card.len() > CARD_LIMIT
        || context.len() > CONTEXT_LIMIT
        || metrics.len() > METRICS_LIMIT
        || index_row.len() > INDEX_ROW_LIMIT
    {
        return Err(ProjectionError(
            "reviewed exception cannot exceed per-file hard ceiling".into(),
        ));
    }
    if total_bytes > TOTAL_LIMIT && input.budget_exception.is_none() {
        return Err(ProjectionError("total byte cap exceeded".into()));
    }
    Ok(Projection {
        manifest,
        card,
        context,
        metrics,
        index_row,
        hashes,
        byte_counts,
        total_bytes,
    })
}

pub fn validate_index_rows(rows: &[Vec<u8>]) -> Result<(), ProjectionError> {
    let mut previous = (String::new(), String::new());
    let mut seen = BTreeSet::new();
    for row in rows {
        if row.len() > INDEX_ROW_LIMIT || !row.ends_with(b"\n") {
            return Err(ProjectionError("index row exceeds cap or lacks LF".into()));
        }
        let line = std::str::from_utf8(row)
            .map_err(|_| ProjectionError("index row is not UTF-8".into()))?;
        let id = extract_prop(line, "wave_id")
            .ok_or_else(|| ProjectionError("index row has no wave_id".into()))?;
        let version = extract_prop(line, "version")
            .ok_or_else(|| ProjectionError("index row has no version".into()))?;
        let key = (id.to_owned(), version.to_owned());
        if !safe_segment(id)
            || !safe_segment(version)
            || !seen.insert(key.clone())
            || (!previous.0.is_empty() && key < previous)
        {
            return Err(ProjectionError(
                "index rows are unsafe, duplicated or not sorted".into(),
            ));
        }
        previous = key;
    }
    Ok(())
}

fn validate(i: &WaveGitInput) -> Result<(), ProjectionError> {
    if !safe_segment(&i.wave_id) || !safe_segment(&i.version) {
        return Err(ProjectionError("unsafe wave id or version".into()));
    }
    for digest in [
        &i.training_identity_sha256,
        &i.archive_manifest_sha256,
        &i.disclosure.sha256,
    ] {
        if !hex64(digest) {
            return Err(ProjectionError("invalid identity digest".into()));
        }
    }
    if i.disclosure.role != "public" || i.disclosure.policy_id.trim().is_empty() {
        return Err(ProjectionError(
            "public disclosure policy is missing or has wrong role".into(),
        ));
    }
    for field in [
        "wave_index",
        "wave_id",
        "version",
        "source_commit",
        "training_identity",
        "archive_manifest",
        "archive_status",
        "capture_status",
        "stage",
        "disposition",
        "ancestry",
        "evidence_roots",
    ] {
        if !i.disclosure.allowed_fields.contains(field) {
            return Err(ProjectionError(format!(
                "{field} is not authorized for public export"
            )));
        }
    }
    if i.archive_status != "LOCAL_COMMITTED_REMOTE_PENDING"
        && i.archive_status != "REMOTE_VERIFIED"
        && i.archive_status != "HISTORICAL_PARTIAL"
    {
        return Err(ProjectionError("unknown archive status".into()));
    }
    if !["COMPLETE", "INCOMPLETE_BLOCKED", "HISTORICAL_PARTIAL"]
        .contains(&i.capture_status.as_str())
    {
        return Err(ProjectionError("unknown capture status".into()));
    }
    for slot in CONTEXT_SLOTS {
        if !i.context.contains_key(slot) {
            return Err(ProjectionError(format!("missing context slot {slot}")));
        }
        if !i.disclosure.allowed_fields.contains(slot) {
            return Err(ProjectionError(format!(
                "context slot {slot} is not allowlisted"
            )));
        }
        if has_numeric_claim(&i.context[slot]) {
            return Err(ProjectionError(format!("numeric prose is forbidden in context slot {slot}; place numeric claims in metrics with evidence")));
        }
    }
    for a in &i.assertions {
        if !safe_segment(&a.id)
            || !a.export_allowed
            || !i
                .disclosure
                .allowed_fields
                .contains(&format!("metric:{}", a.id))
        {
            return Err(ProjectionError(
                "numeric assertion is not allowlisted".into(),
            ));
        }
        if !a.value.json().is_some()
            || a.unit.is_empty()
            || a.denominator == 0
            || a.status.is_empty()
            || a.source_field.is_empty()
            || a.expression.is_empty()
            || !hex64(&a.evidence_sha256)
            || a.evidence_version.is_empty()
        {
            return Err(ProjectionError(format!(
                "numeric assertion {} lacks value/unit/denominator/status/source/evidence closure",
                a.id
            )));
        }
    }
    for root in &i.evidence_roots {
        if !hex64(&root.sha256)
            || !hex64(&root.member_sha256)
            || root.version.is_empty()
            || root.codec.is_empty()
        {
            return Err(ProjectionError(
                "evidence root lacks hash/version/codec".into(),
            ));
        }
        if i.archive_status == "REMOTE_VERIFIED"
            && root
                .physical_version
                .as_ref()
                .is_none_or(|v| v.trim().is_empty())
        {
            return Err(ProjectionError(
                "remote verified root lacks physical version".into(),
            ));
        }
    }
    if i.source_commit.as_ref().is_some_and(|s| !hex40(s)) {
        return Err(ProjectionError(
            "source commit must be a full lowercase SHA or unknown".into(),
        ));
    }
    Ok(())
}
fn safe_segment(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 128
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b))
        && s != "."
        && s != ".."
}
fn has_numeric_claim(s: &str) -> bool {
    const WORDS: &[&str] = &[
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
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "sixty",
        "seventy",
        "eighty",
        "ninety",
        "hundred",
        "thousand",
        "million",
        "billion",
        "first",
        "second",
        "third",
        "fourth",
        "fifth",
        "sixth",
        "seventh",
        "eighth",
        "ninth",
        "tenth",
    ];
    s.chars().any(|c| c.is_ascii_digit())
        || s.split(|c: char| !c.is_ascii_alphabetic())
            .any(|token| WORDS.contains(&token.to_ascii_lowercase().as_str()))
}
fn hex64(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn hex40(s: &str) -> bool {
    s.len() == 40
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn q(v: &str) -> String {
    let mut s = String::from("\"");
    for c in v.chars() {
        match c {
            '"' => s.push_str("\\\""),
            '\\' => s.push_str("\\\\"),
            '\n' => s.push_str("\\n"),
            '\r' => s.push_str("\\r"),
            '\t' => s.push_str("\\t"),
            c if c < ' ' => {
                write!(s, "\\u{:04x}", c as u32).ok();
            }
            c => s.push(c),
        }
    }
    s.push('"');
    s
}
fn opt(v: &Option<String>) -> String {
    v.as_deref().map(q).unwrap_or_else(|| "null".into())
}
fn fmt_err(e: std::fmt::Error) -> ProjectionError {
    ProjectionError(e.to_string())
}
fn extract_prop<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let marker = format!("\"{key}\":\"");
    let rest = line.split_once(&marker)?.1;
    rest.split_once('"').map(|(value, _)| value)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stage_names_are_stable() {
        assert_eq!(Stage::Shadow.text(), "shadow");
    }
}
