//! The remote side of W8 delivery and publication.
//!
//! A trait over exactly the operations W8 uses. The product implementation is
//! [`crate::mirror::Mirror`] (an `object_store` URL given explicitly by `--store` or
//! `JOPE_LEARNING_URL`). The offline tests wrap it to inject faults: disconnects, lost responses,
//! and conflicting or corrupt objects. There is deliberately no unconditional `put`: an immutable
//! write is create-only, and a pointer write always names the version it replaces.

use crate::mirror::Mirror;
use anyhow::Result;
use jope_datasets::store::{ObservedObject, PutOutcome};

pub trait Remote {
    /// The store URL, as recorded in acknowledgements (never credentials).
    fn url(&self) -> &str;

    /// Create-only write (`If-None-Match: *`). An existing object must already hold these exact
    /// bytes, otherwise this fails and nothing is overwritten.
    fn put_immutable(&self, key: &str, bytes: &[u8]) -> Result<PutOutcome>;

    fn get(&self, key: &str) -> Result<Option<Vec<u8>>>;

    /// Bytes and the version that belongs to them, from one read.
    fn get_observed(&self, key: &str) -> Result<Option<ObservedObject>>;

    /// Replace only the observed version (`None`: create only if still absent). A backend that
    /// cannot express the precondition fails; there is no unconditional fallback.
    fn put_pointer(&self, key: &str, bytes: &[u8], observed: Option<&ObservedObject>)
        -> Result<()>;

    fn list(&self, prefix: &str) -> Result<Vec<String>>;
}

impl Remote for Mirror {
    fn url(&self) -> &str {
        &self.url
    }

    fn put_immutable(&self, key: &str, bytes: &[u8]) -> Result<PutOutcome> {
        Mirror::put_immutable(self, key, bytes)
    }

    fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        Mirror::get(self, key)
    }

    fn get_observed(&self, key: &str) -> Result<Option<ObservedObject>> {
        Mirror::get_observed(self, key)
    }

    fn put_pointer(
        &self,
        key: &str,
        bytes: &[u8],
        observed: Option<&ObservedObject>,
    ) -> Result<()> {
        Mirror::put_pointer(self, key, bytes, observed)
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>> {
        Mirror::list(self, prefix)
    }
}
