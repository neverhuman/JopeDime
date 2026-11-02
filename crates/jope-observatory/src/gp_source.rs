//! Append-only all-language source events and compilation-qualified token accounting.
//! This module never fits a model or runs generated programs.

mod prelude {
    pub(super) use anyhow::{bail, ensure, Context, Result};
    pub(super) use feat_math::{
        source_stream::{emit_languages, Occurrence, SCHEMA},
        tokens::{count_tokens, Lang},
    };
    pub(super) use serde::{Deserialize, Serialize};
    pub(super) use sha2::{Digest, Sha256};
    pub(super) use std::{
        collections::{BTreeMap, BTreeSet, VecDeque},
        fs::{self, File, OpenOptions},
        io::{BufRead, BufReader, Read, Seek, SeekFrom, Write},
        path::{Path, PathBuf},
        process::{Command, Stdio},
        time::{Instant, SystemTime, UNIX_EPOCH},
    };
}

mod line;
mod operations;
mod store;
mod types;

pub use line::complete_line;
pub use operations::{check, emit, language_kind, Check};
pub use store::Store;
pub use types::{
    hash, Artifact, Event, LanguageMetrics, Payload, Status, EVENT_SCHEMA, LANGUAGES, MAX_LINE,
};
