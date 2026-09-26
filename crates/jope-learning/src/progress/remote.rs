//! The remote side of W8 delivery and publication.
//!
//! A trait over exactly the operations W8 uses. The product implementation is
//! [`crate::mirror::Mirror`] (an `object_store` URL given explicitly by `--store` or
//! `JOPE_LEARNING_URL`). The offline tests wrap it to inject faults: disconnects, lost responses,
//! and conflicting or corrupt objects. There is deliberately no unconditional `put`: an immutable
//! write is create-only, and a pointer write always names the version it replaces.

use crate::mirror::Mirror;
use anyhow::{bail, Context, Result};
use jope_datasets::store::{ObservedObject, PutOutcome};

/// The environment a progress publisher reads to reach an `s3://` store. Nothing here is a
/// default: the store URL must be named explicitly, and the credential is the operator's.
///
/// | Variable | Meaning |
/// |---|---|
/// | `JOPE_LEARNING_URL` (or `--store`) | `s3://<bucket>/<run-scoped prefix>`; a bare bucket is refused |
/// | `JOPE_AWS_PROFILE` (or `AWS_PROFILE`) | the scoped publisher profile, evaluated through `aws configure export-credentials` |
/// | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN` | a static or STS key instead of a profile |
/// | `AWS_REGION` | the bucket's region (`us-east-2` when unset) |
/// | `AWS_CONDITIONAL_PUT` | unset or `etag`: `If-None-Match`/`If-Match` conditional writes |
/// | `JOPE_DATASETS_ANON` | must not be `1`: unsigned requests cannot publish |
/// | `JOPE_STATE` (or `--state`) | the local state tree holding the outbox and HWM ledger |
pub const S3_ENVIRONMENT: &str = "JOPE_LEARNING_URL=s3://<bucket>/<run-scoped prefix>; \
     JOPE_AWS_PROFILE=<scoped publisher profile> (or AWS_PROFILE, or AWS_ACCESS_KEY_ID + \
     AWS_SECRET_ACCESS_KEY [+ AWS_SESSION_TOKEN]); AWS_REGION (us-east-2 when unset); \
     AWS_CONDITIONAL_PUT unset or `etag`; JOPE_DATASETS_ANON not `1`";

/// Refuse, before any store is opened or anything is written, a publishing store whose
/// configuration could not keep W8's guarantees. Local (`file://`, bare path) and `memory:///`
/// stores need nothing. An `s3://` store must:
/// - name a prefix under the bucket (a run-scoped prefix; never the bucket root);
/// - sign its requests (`JOPE_DATASETS_ANON` is not `1`);
/// - keep object_store's conditional writes on (`AWS_CONDITIONAL_PUT` unset or `etag`): every
///   immutable write is `If-None-Match: *` and every pointer write `If-Match`, with no
///   unconditional fallback.
///
/// `env` reads one environment variable; the CLI passes the process environment.
pub fn preflight_publishing_store(url: &str, env: impl Fn(&str) -> Option<String>) -> Result<()> {
    let Some((scheme, rest)) = url.split_once("://") else {
        return Ok(());
    };
    if !matches!(scheme, "s3" | "s3a") {
        return Ok(());
    }
    let (bucket, prefix) = rest.split_once('/').unwrap_or((rest, ""));
    if bucket.is_empty() || prefix.trim_matches('/').is_empty() {
        bail!(
            "{url}: a progress store on S3 names a run-scoped prefix under the bucket \
             (s3://<bucket>/<prefix>); the bucket root is refused. Nothing was written"
        );
    }
    if env("JOPE_DATASETS_ANON").is_some_and(|v| v == "1") {
        bail!(
            "{url}: JOPE_DATASETS_ANON=1 makes S3 requests unsigned, which can never publish \
             progress. Nothing was written"
        );
    }
    if let Some(mode) = env("AWS_CONDITIONAL_PUT") {
        if mode.trim() != "etag" {
            bail!(
                "{url}: AWS_CONDITIONAL_PUT={mode:?} turns off the conditional writes W8 \
                 publication relies on (If-None-Match for immutable objects, If-Match for the run \
                 pointer); unset it or set it to `etag`. Nothing was written"
            );
        }
    }
    Ok(())
}

/// Open the store progress is published to, after [`preflight_publishing_store`].
pub fn open_publishing_store(url: &str) -> Result<Mirror> {
    preflight_publishing_store(url, |name| std::env::var(name).ok())?;
    Mirror::open(url).with_context(|| {
        format!("open the progress store {url} (it needs: {S3_ENVIRONMENT} for s3://)")
    })
}

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
