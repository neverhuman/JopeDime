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
