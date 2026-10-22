//! Complete, opt-in GP occurrence capture. No sampling, ranking or numerical authority.
//!
//! Set `JOPE_GP_SOURCE_DIR` before starting a producer. Each process appends to its own
//! create-only journal; the source service emits and checks every captured occurrence.
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

pub const SCHEMA: &str = "jope.gp-occurrence/v1";
pub const MAX_PROGRAM_BYTES: usize = 4 * 1024 * 1024;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Occurrence {
    pub schema: String,
    pub id: String,
    pub engine: String,
    pub coordinate: String,
    pub program: String,
}

/// A single producer's append-only journal. Never share a journal between processes.
pub struct Journal {
    file: File,
    name: String,
    sequence: u64,
    pub path: PathBuf,
}
impl Journal {
    pub fn create(root: &Path) -> Result<Self> {
        fs::create_dir_all(root).context("create GP source journal directory")?;
        let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let name = format!("{stamp}-{}", std::process::id());
        let path = root.join(format!("{name}.jsonl"));
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        Ok(Self {
            file,
            name,
            sequence: 0,
            path,
        })
    }
    pub fn append(&mut self, engine: &str, coordinate: &str, program: &str) -> Result<Occurrence> {
        ensure!(!engine.is_empty(), "GP source engine is empty");
        ensure!(
            program.len() <= MAX_PROGRAM_BYTES,
            "GP program exceeds journal bound"
        );
        let sequence = self
            .sequence
            .checked_add(1)
            .context("GP journal sequence overflow")?;
        let occurrence = Occurrence {
            schema: SCHEMA.into(),
            id: format!("{}:{sequence}", self.name),
            engine: engine.into(),
            coordinate: coordinate.into(),
            program: program.into(),
        };
        let mut bytes = serde_json::to_vec(&occurrence)?;
        bytes.push(b'\n');
        self.file
            .write_all(&bytes)
            .context("append complete GP occurrence")?;
        self.file.flush()?;
        self.sequence = sequence;
        Ok(occurrence)
    }
}

static JOURNAL: OnceLock<std::result::Result<Option<Mutex<Journal>>, String>> = OnceLock::new();

fn state() -> &'static std::result::Result<Option<Mutex<Journal>>, String> {
    JOURNAL.get_or_init(|| match std::env::var_os("JOPE_GP_SOURCE_DIR") {
        None => Ok(None),
        Some(root) => Journal::create(Path::new(&root))
            .map(|j| Some(Mutex::new(j)))
            .map_err(|e| format!("{e:#}")),
    })
}

/// The emitted source languages of the source-output ledger, in ledger order.
pub const LANGUAGES: [&str; 4] = ["rust", "python", "cpp17", "javascript"];

/// `program` emitted in each of [`LANGUAGES`], in that order, under default options. A row is
/// the language's complete source, or the error text when the program does not parse or that
/// emitter does not support it. Emission only: nothing is compiled or run.
pub fn emit_languages(program: &str) -> [std::result::Result<String, String>; 4] {
    use crate::transpile::{self, EmitOptions};
    let parsed = crate::GenomeIr::parse(program);
    let pair = parsed
        .as_ref()
        .map_err(|e| format!("{e:#}"))
        .and_then(|g| transpile::emit(g, &EmitOptions::default()).map_err(|e| format!("{e:#}")));
    [
        pair.as_ref().map(|p| p.rust.clone()).map_err(Clone::clone),
        pair.as_ref()
            .map(|p| p.python.clone())
            .map_err(Clone::clone),
        parsed.as_ref().map_err(|e| format!("{e:#}")).and_then(|g| {
            transpile::emit_cpp17(g, &EmitOptions::default()).map_err(|e| format!("{e:#}"))
        }),
        parsed.as_ref().map_err(|e| format!("{e:#}")).and_then(|g| {
            transpile::emit_javascript(g, &EmitOptions::default()).map_err(|e| format!("{e:#}"))
        }),
    ]
}

/// The lexical token rules ([`crate::tokens::count_tokens`]) of a [`LANGUAGES`] entry.
pub fn language_kind(language: &str) -> Option<crate::tokens::Lang> {
    use crate::tokens::Lang;
    Some(match language {
        "rust" => Lang::Rust,
        "python" => Lang::Python,
        "cpp17" => Lang::Cpp,
        "javascript" => Lang::JavaScript,
        _ => return None,
    })
}

/// Code plus documentation tokens of `program`'s emitted source per [`LANGUAGES`] entry, as the
/// source-output ledger counts them. A language that fails to emit contributes 0: no guessed
/// count is credited.
pub fn emitted_tokens(program: &str) -> [u64; 4] {
    count_emitted(&emit_languages(program))
}

/// [`emitted_tokens`] of sources already emitted by [`emit_languages`].
pub fn count_emitted(sources: &[std::result::Result<String, String>; 4]) -> [u64; 4] {
    let mut tokens = [0u64; 4];
    for ((slot, language), source) in tokens.iter_mut().zip(LANGUAGES).zip(sources) {
        if let (Ok(source), Some(kind)) = (source, language_kind(language)) {
            let count = crate::tokens::count_tokens(source, kind);
            *slot = (count.code_tokens + count.doc_tokens) as u64;
        }
    }
    tokens
}

/// Whether this process enables capture. Initialization failures remain visible on capture.
pub fn enabled() -> bool {
    !matches!(state(), Ok(None))
}

/// Capture an occurrence before filtering or scoring. Enabled IO failures stop the producer.
pub fn capture(engine: &str, coordinate: &str, program: &str) {
    match state() {
        Ok(None) => {}
        Ok(Some(journal)) => {
            journal
                .lock()
                .expect("GP source journal mutex poisoned")
                .append(engine, coordinate, program)
                .expect("GP source capture failed; producer stopped");
        }
        Err(error) => panic!("GP source capture initialization failed: {error}"),
    };
}

/// Avoid formatting/canonicalization work entirely when occurrence capture is disabled.
#[macro_export]
macro_rules! capture_gp_source {
    ($engine:expr, $coordinate:expr, $program:expr) => {
        if $crate::source_stream::enabled() {
            $crate::source_stream::capture($engine, $coordinate, $program);
        }
    };
}
