//! The durable copy: an `object_store` URL (`file://` in tests, `s3://` via `JOPE_LEARNING_URL`
//! in production) written through `jope_datasets::store::Store`.
//!
//! Thin adapter, not a re-implementation: immutable objects (blobs, ledger batches) use the
//! create-only `put_if_absent` (`If-None-Match: *`); pointers (`current.json`, `HEAD.json`) are
//! read back and compared against an expected-previous value before being replaced.
//!
//! That expected-previous value must never be an *unchecked* read of the key about to be written:
//! that makes the precondition self-satisfying and the write last-writer-wins, which is how a
//! second host's ledger batch gets orphaned. Nor can it simply be the writer's own pre-write state
//! — the `HEAD.json` captured under the ledger lock before the batch was appended — because a
//! precondition with no way to satisfy it is a wedge: one local-only ingest and the mirror can
//! never be written again.
//!
//! [`crate::sync`] resolves both: it reads the mirror's `HEAD.json`, *proves against the local
//! chain* that the tip it found is a batch this host already holds, uploads the batches in
//! between, and only then hands those exact bytes to [`Mirror::put_pointer`] as the witness. A
//! mirror that is merely behind catches up; a mirror whose tip this host cannot produce is
//! refused, untouched.
//!
//! A true server-side conditional PUT (`PutMode::Update` with the expected version) is not
//! reachable through the `Store` API today and `object_store`'s local filesystem backend does not
//! implement it either, so the compare-then-put is still TOCTOU between the GET and the PUT, and
//! is serialised only by the local `$JOPE_STATE` lock that every writer takes first — a lock that
//! does not reach across hosts. What the precondition does buy is that a *diverged* mirror is
//! detected and refused rather than overwritten.

use anyhow::{bail, Result};
use jope_datasets::store::{PutOutcome, Store};
use serde::Serialize;

pub struct Mirror {
    store: Store,
    rt: tokio::runtime::Runtime,
    pub url: String,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct MirrorReport {
    pub url: String,
    pub created: u64,
    pub existed: u64,
    pub pointers: u64,
    /// Ledger batch objects this push uploaded. More than one means the mirror was behind and has
    /// been caught up; `created` vs `existed` says how many of them were genuinely new.
    pub batches: u64,
}

impl MirrorReport {
    pub fn count(&mut self, outcome: PutOutcome) {
        match outcome {
            PutOutcome::Created => self.created += 1,
            PutOutcome::Exists => self.existed += 1,
        }
    }
}

/// `--store`, else `JOPE_LEARNING_URL`, else none.
pub fn mirror_url(arg: Option<String>) -> Option<String> {
    arg.or_else(|| std::env::var("JOPE_LEARNING_URL").ok())
        .filter(|s| !s.is_empty())
}

impl Mirror {
    pub fn open(url: &str) -> Result<Mirror> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        let store = Store::open(url)?;
        Ok(Mirror {
            url: store.url.clone(),
            store,
            rt,
        })
    }

    /// Create-only write; an existing object must already hold the same bytes.
    pub fn put_immutable(&self, rel: &str, bytes: &[u8]) -> Result<PutOutcome> {
        let outcome = self
            .rt
            .block_on(self.store.put_if_absent(rel, bytes.to_vec()))?;
        if outcome == PutOutcome::Exists {
            let existing = self.rt.block_on(self.store.get_bytes(rel))?;
            if existing.as_deref() != Some(bytes) {
                bail!("{rel}: exists in {} with different bytes", self.url);
            }
        }
        Ok(outcome)
    }

    /// Replace a pointer only if the mirror still holds `expected_prev` (None = absent).
    pub fn put_pointer(&self, rel: &str, bytes: &[u8], expected_prev: Option<&[u8]>) -> Result<()> {
        let existing = self.rt.block_on(self.store.get_bytes(rel))?;
        if existing.as_deref() != expected_prev {
            bail!(
                "{rel}: mirror pointer changed (expected {} bytes, found {})",
                expected_prev.map_or("no".to_string(), |b| b.len().to_string()),
                existing.map_or("none".to_string(), |b| format!("{} bytes", b.len()))
            );
        }
        self.rt.block_on(self.store.put(rel, bytes.to_vec()))
    }

    pub fn get(&self, rel: &str) -> Result<Option<Vec<u8>>> {
        self.rt.block_on(self.store.get_bytes(rel))
    }

    pub fn list(&self, prefix: &str) -> Result<Vec<String>> {
        self.rt.block_on(self.store.list(prefix))
    }
}
