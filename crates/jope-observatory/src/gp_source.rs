//! Append-only all-language source events and compilation-qualified token accounting.
//! This module never fits a model or runs generated programs.
use anyhow::{bail, ensure, Context, Result};
use feat_math::{
    source_stream::{Occurrence, SCHEMA},
    tokens::{count_tokens, Lang},
    transpile::{self, EmitOptions},
    GenomeIr,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

pub const EVENT_SCHEMA: &str = "jope.gp-source/v1";
pub const MAX_LINE: usize = 32 * 1024 * 1024;
pub const LANGUAGES: [&str; 4] = ["rust", "python", "cpp17", "javascript"];

pub fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn valid_hash(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
}
fn extension(language: &str) -> Result<&'static str> {
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
pub enum Status {
    Pending,
    Passed,
    Failed,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Artifact {
    pub language: String,
    pub source: Option<String>,
    pub sha256: Option<String>,
    pub code_tokens: u64,
    pub doc_tokens: u64,
    pub bytes: u64,
    pub status: Status,
    pub error: Option<String>,
}
impl Artifact {
    pub fn tokens(&self) -> u64 {
        self.code_tokens + self.doc_tokens
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Payload {
    Source {
        occurrence: Occurrence,
        program_sha256: String,
        emit_ns: u64,
        artifacts: Vec<Artifact>,
    },
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
pub struct Event {
    pub schema: String,
    pub sequence: u64,
    pub at_unix_ns: u64,
    #[serde(flatten)]
    pub payload: Payload,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct LanguageMetrics {
    pub artifacts: u64,
    pub emitted_tokens: u64,
    pub code_tokens: u64,
    pub doc_tokens: u64,
    pub passed_tokens: u64,
    pub lost_tokens: u64,
    pub pending_tokens: u64,
    pub emission_errors: u64,
    pub compile_errors: u64,
    pub pending_artifacts: u64,
    pub passed_artifacts: u64,
}
impl LanguageMetrics {
    fn add(&mut self, other: &Self) {
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
#[derive(Clone, Debug)]
struct Pending {
    occurrence_id: String,
    artifact: Artifact,
    inflight: bool,
}

pub struct Store {
    pub root: PathBuf,
    log: File,
    _lock: File,
    pub sequence: u64,
    seen: BTreeMap<String, String>,
    pending: BTreeMap<(String, String), Pending>,
    offsets: Vec<u64>,
    programs: BTreeSet<String>,
    pub metrics: BTreeMap<String, LanguageMetrics>,
    pub emit_ns: u64,
    first_unix_ns: Option<u64>,
    failed: bool,
}
impl Store {
    pub fn open(root: &Path) -> Result<Self> {
        fs::create_dir_all(root.join("sources"))?;
        fs::create_dir_all(root.join("checks"))?;
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(root.join("writer.lock"))?;
        lock.try_lock()
            .context("another GP source service owns this store")?;
        let log_path = root.join("events.jsonl");
        let log = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(&log_path)?;
        let mut store = Self {
            root: root.into(),
            log,
            _lock: lock,
            sequence: 0,
            seen: BTreeMap::new(),
            pending: BTreeMap::new(),
            offsets: vec![0],
            programs: BTreeSet::new(),
            metrics: LANGUAGES
                .into_iter()
                .map(|l| (l.into(), LanguageMetrics::default()))
                .collect(),
            emit_ns: 0,
            first_unix_ns: None,
            failed: false,
        };
        let mut reader = BufReader::new(File::open(log_path)?);
        while let Some(bytes) = complete_line(&mut reader, false)? {
            let event: Event =
                serde_json::from_slice(&bytes).context("invalid GP event log; preserve it")?;
            store.apply(&event)?;
            store.offsets.push(reader.stream_position()?);
        }
        Ok(store)
    }
    fn apply(&mut self, event: &Event) -> Result<()> {
        ensure!(
            event.schema == EVENT_SCHEMA && event.sequence == self.sequence + 1,
            "GP event schema/sequence mismatch"
        );
        ensure!(event.at_unix_ns > 0, "missing event time");
        match &event.payload {
            Payload::Source {
                occurrence,
                program_sha256,
                emit_ns,
                artifacts,
            } => {
                ensure!(
                    occurrence.schema == SCHEMA && !occurrence.id.is_empty(),
                    "invalid occurrence"
                );
                ensure!(
                    *program_sha256 == hash(occurrence.program.as_bytes()),
                    "program hash mismatch"
                );
                ensure!(
                    !self.seen.contains_key(&occurrence.id),
                    "duplicate occurrence in event log"
                );
                ensure!(
                    artifacts.len() == LANGUAGES.len(),
                    "incomplete source target set"
                );
                let languages: BTreeSet<_> =
                    artifacts.iter().map(|a| a.language.as_str()).collect();
                ensure!(
                    languages == LANGUAGES.into_iter().collect(),
                    "invalid source target set"
                );
                for artifact in artifacts {
                    let m = self
                        .metrics
                        .get_mut(&artifact.language)
                        .context("unknown language")?;
                    m.artifacts += 1;
                    match (&artifact.source, &artifact.sha256, &artifact.status) {
                        (Some(source), Some(digest), Status::Pending) => {
                            ensure!(
                                *digest == hash(source.as_bytes())
                                    && artifact.bytes == source.len() as u64
                                    && artifact.error.is_none(),
                                "source bytes/hash/status mismatch"
                            );
                            let count = count_tokens(source, language_kind(&artifact.language)?);
                            ensure!(
                                artifact.code_tokens == count.code_tokens as u64
                                    && artifact.doc_tokens == count.doc_tokens as u64,
                                "source token count mismatch"
                            );
                            m.emitted_tokens += artifact.tokens();
                            m.code_tokens += artifact.code_tokens;
                            m.doc_tokens += artifact.doc_tokens;
                            m.pending_tokens += artifact.tokens();
                            m.pending_artifacts += 1;
                            let mut descriptor = artifact.clone();
                            descriptor.source = None;
                            self.pending.insert(
                                (occurrence.id.clone(), artifact.language.clone()),
                                Pending {
                                    occurrence_id: occurrence.id.clone(),
                                    artifact: descriptor,
                                    inflight: false,
                                },
                            );
                        }
                        (None, None, Status::Failed) => {
                            ensure!(
                                artifact.tokens() == 0
                                    && artifact.bytes == 0
                                    && artifact.error.as_ref().is_some_and(|e| !e.is_empty()),
                                "invalid emission error"
                            );
                            m.emission_errors += 1;
                        }
                        _ => bail!("invalid initial artifact state"),
                    }
                }
                self.programs.insert(program_sha256.clone());
                self.emit_ns += emit_ns;
                self.seen.insert(
                    occurrence.id.clone(),
                    hash(&serde_json::to_vec(occurrence)?),
                );
            }
            Payload::Checked {
                occurrence_id,
                language,
                source_sha256,
                status,
                tool,
                ..
            } => {
                ensure!(
                    *status != Status::Pending && !tool.is_empty(),
                    "invalid compiler result"
                );
                let key = (occurrence_id.clone(), language.clone());
                let pending = self
                    .pending
                    .get(&key)
                    .context("compiler result without pending artifact")?;
                ensure!(
                    pending.artifact.sha256.as_ref() == Some(source_sha256),
                    "compiler checked wrong source"
                );
                let m = self.metrics.get_mut(language).context("unknown language")?;
                m.pending_tokens -= pending.artifact.tokens();
                m.pending_artifacts -= 1;
                if *status == Status::Passed {
                    m.passed_tokens += pending.artifact.tokens();
                    m.passed_artifacts += 1;
                } else {
                    m.lost_tokens += pending.artifact.tokens();
                    m.compile_errors += 1;
                }
                self.pending.remove(&key);
            }
        }
        self.first_unix_ns.get_or_insert(event.at_unix_ns);
        self.sequence = event.sequence;
        Ok(())
    }
    fn append(&mut self, payload: Payload) -> Result<()> {
        ensure!(
            !self.failed,
            "source store stopped after earlier publication failure"
        );
        let at_unix_ns = u64::try_from(SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos())?;
        let event = Event {
            schema: EVENT_SCHEMA.into(),
            sequence: self
                .sequence
                .checked_add(1)
                .context("event sequence overflow")?,
            at_unix_ns,
            payload,
        };
        // Validate before publishing. Apply errors are fatal; never keep serving partial state.
        if let Err(error) = self.apply(&event) {
            self.failed = true;
            return Err(error);
        }
        let mut bytes = serde_json::to_vec(&event)?;
        bytes.push(b'\n');
        if let Err(error) = self
            .log
            .write_all(&bytes)
            .and_then(|_| self.log.flush())
            .and_then(|_| self.log.sync_data())
        {
            self.failed = true;
            return Err(error.into());
        }
        self.offsets.push(self.log.metadata()?.len());
        Ok(())
    }
    pub fn offset_after(&self, sequence: u64) -> Result<u64> {
        self.offsets
            .get(usize::try_from(sequence)?)
            .copied()
            .context("future sequence cursor")
    }
    pub fn contains(&self, occurrence: &Occurrence) -> Result<bool> {
        match self.seen.get(&occurrence.id) {
            None => Ok(false),
            Some(digest) => {
                ensure!(
                    *digest == hash(&serde_json::to_vec(occurrence)?),
                    "occurrence identity reused with different content"
                );
                Ok(true)
            }
        }
    }
    pub fn publish(
        &mut self,
        occurrence: Occurrence,
        artifacts: Vec<Artifact>,
        emit_ns: u64,
    ) -> Result<()> {
        ensure!(!self.contains(&occurrence)?, "occurrence already published");
        for artifact in &artifacts {
            if let (Some(source), Some(digest)) = (&artifact.source, &artifact.sha256) {
                ensure!(
                    valid_hash(digest) && *digest == hash(source.as_bytes()),
                    "invalid artifact digest"
                );
                let path = self
                    .root
                    .join("sources")
                    .join(format!("{digest}.{}", extension(&artifact.language)?));
                if path.exists() {
                    ensure!(
                        fs::read(&path)? == source.as_bytes(),
                        "cached source mismatch"
                    );
                } else {
                    let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
                    let staging = self
                        .root
                        .join("sources")
                        .join(format!(".{digest}-{stamp}.partial"));
                    let mut file = OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(&staging)?;
                    file.write_all(source.as_bytes())?;
                    file.sync_all()?;
                    fs::rename(staging, &path)?;
                }
            }
        }
        let digest = hash(occurrence.program.as_bytes());
        self.append(Payload::Source {
            occurrence,
            program_sha256: digest,
            emit_ns,
            artifacts,
        })
    }
    pub fn next_check(&mut self) -> Option<(String, Artifact, PathBuf)> {
        let next = self.pending.values_mut().find(|p| !p.inflight)?;
        next.inflight = true;
        Some((
            next.occurrence_id.clone(),
            next.artifact.clone(),
            self.root.clone(),
        ))
    }
    pub fn checked(&mut self, id: String, artifact: &Artifact, check: Check) -> Result<()> {
        self.append(Payload::Checked {
            occurrence_id: id,
            language: artifact.language.clone(),
            source_sha256: artifact.sha256.clone().context("missing source hash")?,
            status: check.status,
            tool: check.tool,
            diagnostic: check.diagnostic,
            elapsed_ns: check.elapsed_ns,
        })
    }
    pub fn snapshot(&self) -> serde_json::Value {
        let mut total = LanguageMetrics::default();
        for metrics in self.metrics.values() {
            total.add(metrics);
        }
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let wall = self
            .first_unix_ns
            .map(|first| now.saturating_sub(first as u128) as f64 / 1e9)
            .unwrap_or(0.0)
            .max(1e-9);
        let stats = |m: &LanguageMetrics| {
            serde_json::json!({
                "counts":m, "token_yield":if m.emitted_tokens == 0 {None} else {Some(m.passed_tokens as f64 / m.emitted_tokens as f64)},
                "artifact_yield":if m.artifacts == 0 {None} else {Some(m.passed_artifacts as f64 / m.artifacts as f64)},
                "emitted_tokens_per_second":m.emitted_tokens as f64 / wall,
            })
        };
        serde_json::json!({"schema":"jope.gp-source-metrics/v1", "tokenizer":"feat-math lexical code tokens plus documentation words; not model BPE", "sequence":self.sequence, "occurrences":self.seen.len(), "unique_programs":self.programs.len(), "elapsed_seconds_including_downtime":wall, "store_failed":self.failed, "cumulative_emit_ns":self.emit_ns, "assembly_tokens_per_second":if self.emit_ns == 0 {None} else {Some(total.emitted_tokens as f64 * 1e9 / self.emit_ns as f64)}, "total":stats(&total), "languages":self.metrics.iter().map(|(l,m)| (l.clone(),stats(m))).collect::<BTreeMap<_,_>>()})
    }
}

pub fn language_kind(language: &str) -> Result<Lang> {
    Ok(match language {
        "rust" => Lang::Rust,
        "python" => Lang::Python,
        "cpp17" => Lang::Cpp,
        "javascript" => Lang::JavaScript,
        _ => bail!("unknown language"),
    })
}
pub fn emit(occurrence: &Occurrence) -> Result<(Vec<Artifact>, u64)> {
    ensure!(
        occurrence.schema == SCHEMA
            && occurrence.program.len() <= feat_math::source_stream::MAX_PROGRAM_BYTES,
        "invalid/oversized occurrence"
    );
    let started = Instant::now();
    let parsed = GenomeIr::parse(&occurrence.program);
    let pair = parsed
        .as_ref()
        .map_err(|e| format!("{e:#}"))
        .and_then(|g| transpile::emit(g, &EmitOptions::default()).map_err(|e| format!("{e:#}")));
    let mut artifacts = Vec::new();
    for language in LANGUAGES {
        let source: std::result::Result<String, String> = match language {
            "rust" => pair.as_ref().map(|p| p.rust.clone()).map_err(Clone::clone),
            "python" => pair
                .as_ref()
                .map(|p| p.python.clone())
                .map_err(Clone::clone),
            "cpp17" => parsed.as_ref().map_err(|e| format!("{e:#}")).and_then(|g| {
                transpile::emit_cpp17(g, &EmitOptions::default()).map_err(|e| format!("{e:#}"))
            }),
            "javascript" => parsed.as_ref().map_err(|e| format!("{e:#}")).and_then(|g| {
                transpile::emit_javascript(g, &EmitOptions::default()).map_err(|e| format!("{e:#}"))
            }),
            _ => unreachable!(),
        };
        artifacts.push(match source {
            Ok(source) => {
                let count = count_tokens(&source, language_kind(language)?);
                Artifact {
                    language: language.into(),
                    sha256: Some(hash(source.as_bytes())),
                    source: Some(source),
                    code_tokens: count.code_tokens as u64,
                    doc_tokens: count.doc_tokens as u64,
                    bytes: count.bytes as u64,
                    status: Status::Pending,
                    error: None,
                }
            }
            Err(error) => Artifact {
                language: language.into(),
                source: None,
                sha256: None,
                code_tokens: 0,
                doc_tokens: 0,
                bytes: 0,
                status: Status::Failed,
                error: Some(error),
            },
        });
    }
    Ok((
        artifacts,
        u64::try_from(started.elapsed().as_nanos())
            .unwrap_or(u64::MAX)
            .max(1),
    ))
}

#[derive(Clone)]
pub struct Check {
    pub status: Status,
    pub tool: String,
    pub diagnostic: String,
    pub elapsed_ns: u64,
}
/// GNU timeout bounds the compiler process group. No shell or generated-code execution.
pub fn check(root: &Path, artifact: &Artifact, timeout_seconds: u64) -> Result<Check> {
    ensure!(
        (1..=300).contains(&timeout_seconds),
        "compiler timeout outside 1..300 seconds"
    );
    let digest = artifact.sha256.as_ref().context("no source digest")?;
    ensure!(valid_hash(digest), "invalid source digest");
    let source = root
        .join("sources")
        .join(format!("{digest}.{}", extension(&artifact.language)?));
    ensure!(
        hash(&fs::read(&source)?) == *digest,
        "source changed before compilation"
    );
    let scratch = root
        .join("checks")
        .join(format!("{digest}-{}", artifact.language));
    fs::create_dir_all(&scratch)?;
    let diagnostics = scratch.join("diagnostic.txt");
    let (tool, mut args): (&str, Vec<String>) = match artifact.language.as_str() {
        "rust" => ("rustc", vec!["--edition=2021".into(), "--crate-name=gp_source".into(), "--crate-type=lib".into(), "--emit=metadata".into(), "-o".into(), scratch.join("checked.rmeta").to_string_lossy().into_owned()]),
        "python" => ("python3", vec!["-I".into(), "-c".into(), "import pathlib,sys; compile(pathlib.Path(sys.argv[1]).read_bytes(), sys.argv[1], 'exec')".into()]),
        "cpp17" => ("g++", vec!["-std=c++17".into(), "-fsyntax-only".into()]),
        "javascript" => ("node", vec!["--check".into()]),
        _ => bail!("unknown language"),
    };
    let variable = match artifact.language.as_str() {
        "rust" => "JOPE_GP_RUSTC",
        "python" => "JOPE_GP_PYTHON",
        "cpp17" => "JOPE_GP_CXX",
        "javascript" => "JOPE_GP_NODE",
        _ => unreachable!(),
    };
    let tool = std::env::var(variable).unwrap_or_else(|_| tool.into());
    args.push(source.to_string_lossy().into_owned());
    let version_path = scratch.join("tool-version.txt");
    let _ = Command::new("prlimit")
        .args([
            "--fsize=65536",
            "--",
            "timeout",
            "--signal=KILL",
            "2s",
            &tool,
            "--version",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::from(File::create(&version_path)?))
        .stderr(Stdio::null())
        .status();
    let mut version = String::new();
    File::open(version_path)?
        .take(65536)
        .read_to_string(&mut version)?;
    let started = Instant::now();
    let status = Command::new("prlimit")
        .args([
            "--fsize=4194304",
            "--",
            "timeout",
            "--signal=KILL",
            &format!("{timeout_seconds}s"),
            &tool,
        ])
        .args(&args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::from(File::create(&diagnostics)?))
        .status();
    let (passed, error) = match status {
        Ok(status) => (status.success(), format!("exit={status}")),
        Err(error) => (false, format!("compiler launch failed: {error}")),
    };
    let mut bytes = Vec::new();
    File::open(diagnostics)?
        .take(65536)
        .read_to_end(&mut bytes)?;
    let mut diagnostic = String::from_utf8_lossy(&bytes).into_owned();
    diagnostic.push_str(&format!("\n{error}"));
    Ok(Check { status:if passed {Status::Passed} else {Status::Failed}, tool:format!("{tool}; version={version:?}; validation=syntax-and-types-v1; output_file_limit=4194304; timeout={timeout_seconds}s; argv={args:?}"), diagnostic, elapsed_ns:u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX) })
}

/// Read a bounded complete line. A producer's in-progress final record waits for completion.
pub fn complete_line<R: BufRead + Seek>(
    reader: &mut R,
    allow_partial: bool,
) -> Result<Option<Vec<u8>>> {
    let position = reader.stream_position()?;
    let mut bytes = Vec::new();
    reader
        .take(MAX_LINE as u64 + 1)
        .read_until(b'\n', &mut bytes)?;
    ensure!(
        bytes.len() <= MAX_LINE,
        "source journal/event line exceeds limit"
    );
    if bytes.is_empty() {
        return Ok(None);
    }
    if bytes.last() != Some(&b'\n') {
        ensure!(
            allow_partial,
            "incomplete event log; preserve original before recovery"
        );
        reader.seek(SeekFrom::Start(position))?;
        return Ok(None);
    }
    Ok(Some(bytes))
}
