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
//! The observation also retains the version returned with those bytes. Remote pointer
//! writes use that version as a server-side precondition; local mirrors revalidate it
//! under a shared directory lock. A competing publisher fails rather than losing the
//! winner's update. This protects individual pointers, not a transaction across the
//! ledger head and every derived index. Live S3 qualification remains separate.

use anyhow::Result;
use jope_datasets::store::{ObservedObject, PutOutcome, Store};
use serde::Serialize;

/// `Mirror` is a record for the mirror contract in this module: The durable copy: an objectstore URL (file:// in tests, s3:// via JOPELEARNINGURL in production) written through jopedatasets::store::Store.
pub struct Mirror {
    store: Store,
    rt: tokio::runtime::Runtime,
    /// Url recorded for this mirror.
    pub url: String,
}

#[derive(Clone, Debug, Default, Serialize)]
/// `MirrorReport` is a record for the mirror report contract in this module: The durable copy: an objectstore URL (file:// in tests, s3:// via JOPELEARNINGURL in production) written through jopedatasets::store::Store.
pub struct MirrorReport {
    /// Url recorded for this mirror report.
    pub url: String,
    /// Created recorded for this mirror report.
    pub created: u64,
    /// Existed recorded for this mirror report.
    pub existed: u64,
    /// Pointers recorded for this mirror report.
    pub pointers: u64,
    /// Ledger batch objects this push uploaded. More than one means the mirror was behind and has
    /// been caught up; `created` vs `existed` says how many of them were genuinely new.
    pub batches: u64,
    /// `current.json` pointers still owed to the mirror after this push: written locally, and
    /// named by batches the mirror's `HEAD.json` may already reach, but not yet on the mirror.
    pub pointers_owed: u64,
}

impl MirrorReport {
    /// Performs the count operation and returns an error if it cannot be completed.
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
    /// Constructs or opens a open value from the supplied inputs.
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

    /// Create-only write verified by full SHA-256 content and byte length on readback;
    /// an existing object must already hold the exact same bytes.
    pub fn put_immutable(&self, rel: &str, bytes: &[u8]) -> Result<PutOutcome> {
        Ok(self.put_immutable_verified(rel, bytes)?.put_outcome())
    }

    /// Create-only write returning the full [`jope_datasets::verified_object::VerifiedObject`]
    /// witness after readback verification.
    pub fn put_immutable_verified(
        &self,
        rel: &str,
        bytes: &[u8],
    ) -> Result<jope_datasets::verified_object::VerifiedObject> {
        let expected_sha256 = jope_datasets::hash::sha256_hex(bytes);
        let expected_bytes = bytes.len() as u64;
        self.rt
            .block_on(jope_datasets::verified_object::publish_verified_object(
                &self.store,
                rel,
                bytes,
                &expected_sha256,
                expected_bytes,
            ))
    }

    /// Replace only the version whose bytes the caller checked against its history.
    /// `None` means the pointer was absent; it must still be absent at creation.
    pub fn put_pointer(
        &self,
        rel: &str,
        bytes: &[u8],
        observed: Option<&ObservedObject>,
    ) -> Result<()> {
        self.rt
            .block_on(self.store.put_conditional(rel, bytes.to_vec(), observed))
    }

    /// Looks up observed and returns the matching value when present.
    pub fn get_observed(&self, rel: &str) -> Result<Option<ObservedObject>> {
        self.rt.block_on(self.store.get_observed(rel))
    }

    /// Performs the get operation and returns an error if it cannot be completed.
    pub fn get(&self, rel: &str) -> Result<Option<Vec<u8>>> {
        self.rt.block_on(self.store.get_bytes(rel))
    }

    /// Performs the list operation and returns an error if it cannot be completed.
    pub fn list(&self, prefix: &str) -> Result<Vec<String>> {
        self.rt.block_on(self.store.list(prefix))
    }
}
