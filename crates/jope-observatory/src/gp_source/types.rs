use super::prelude::*;

/// Version of the append-only GP source event log.
pub const EVENT_SCHEMA: &str = "jope.gp-source/v1";
/// Maximum serialized size of one source event line.
pub const MAX_LINE: usize = 32 * 1024 * 1024;
/// Language targets emitted and accounted for by this store.
pub const LANGUAGES: [&str; 4] = feat_math::source_stream::LANGUAGES;

/// Return the lowercase SHA-256 digest of `bytes`.
pub fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
pub(crate) fn valid_hash(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
}
pub(crate) fn extension(language: &str) -> Result<&'static str> {
    Ok(match language {
        "rust" => "rs",
        "python" => "py",
        "cpp17" => "cpp",
        "javascript" => "mjs",
        _ => bail!("unknown source language"),
    })
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
/// Lifecycle state of one emitted source artifact.
pub enum Status {
    /// Source was emitted and has not yet been checked.
    Pending,
    /// Source passed the configured syntax/type check.
    Passed,
    /// Emission or source check failed.
    Failed,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// One emitted source file and its lexical-token accounting.
pub struct Artifact {
    /// Target language identifier.
    pub language: String,
    /// Emitted source text, absent when emission failed.
    pub source: Option<String>,
    /// Lowercase SHA-256 of `source`, when emitted.
    pub sha256: Option<String>,
    /// Lexical code-token count.
    pub code_tokens: u64,
    /// Documentation-word count, accounted separately from code tokens.
    pub doc_tokens: u64,
    /// UTF-8 source length in bytes.
    pub bytes: u64,
    /// Current emission/check state.
    pub status: Status,
    /// Emission failure detail, when emission failed.
    pub error: Option<String>,
}
impl Artifact {
    /// Return code and documentation tokens combined.
    pub fn tokens(&self) -> u64 {
        self.code_tokens + self.doc_tokens
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
/// Persisted event payload; source publication precedes its compiler result.
pub enum Payload {
    /// A source occurrence and its per-language emitted artifacts.
    Source {
        occurrence: Occurrence,
        program_sha256: String,
        emit_ns: u64,
        artifacts: Vec<Artifact>,
    },
    /// The result of checking one previously published artifact.
    Checked {
        occurrence_id: String,
        language: String,
        source_sha256: String,
        status: Status,
        tool: String,
        diagnostic: String,
        elapsed_ns: u64,
    },
}
#[derive(Clone, Debug, Serialize, Deserialize)]
/// One sequenced event in the append-only GP source log.
pub struct Event {
    /// Event schema version.
    pub schema: String,
    /// One-based event sequence number.
    pub sequence: u64,
    /// Event creation time in Unix nanoseconds.
    pub at_unix_ns: u64,
    #[serde(flatten)]
    pub payload: Payload,
}

#[derive(Clone, Debug, Default, Serialize)]
/// Cumulative emission and validation counters for one language.
pub struct LanguageMetrics {
    /// Number of artifacts emitted.
    pub artifacts: u64,
    /// Code plus documentation tokens emitted.
    pub emitted_tokens: u64,
    /// Lexical code tokens emitted.
    pub code_tokens: u64,
    /// Documentation words emitted.
    pub doc_tokens: u64,
    /// Tokens belonging to artifacts that passed checking.
    pub passed_tokens: u64,
    /// Tokens belonging to artifacts that failed checking.
    pub lost_tokens: u64,
    /// Tokens awaiting checking.
    pub pending_tokens: u64,
    /// Number of source emission failures.
    pub emission_errors: u64,
    /// Number of compiler/check failures.
    pub compile_errors: u64,
    /// Number of artifacts awaiting checking.
    pub pending_artifacts: u64,
    /// Number of artifacts that passed checking.
    pub passed_artifacts: u64,
}
impl LanguageMetrics {
    pub(crate) fn add(&mut self, other: &Self) {
        self.artifacts += other.artifacts;
        self.emitted_tokens += other.emitted_tokens;
        self.code_tokens += other.code_tokens;
        self.doc_tokens += other.doc_tokens;
        self.passed_tokens += other.passed_tokens;
        self.lost_tokens += other.lost_tokens;
        self.pending_tokens += other.pending_tokens;
        self.emission_errors += other.emission_errors;
        self.compile_errors += other.compile_errors;
        self.pending_artifacts += other.pending_artifacts;
        self.passed_artifacts += other.passed_artifacts;
    }
}
