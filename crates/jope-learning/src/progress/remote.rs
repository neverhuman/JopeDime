//! The remote side of W8 delivery and publication.
//!
//! A trait over exactly the operations W8 uses. The product implementation is
//! [`crate::mirror::Mirror`] (an `object_store` URL given explicitly by `--store` or
//! `JOPE_LEARNING_URL`). The offline tests wrap it to inject faults: disconnects, lost responses,
//! and conflicting or corrupt objects. There is deliberately no unconditional `put`: an immutable
//! write is create-only, and a pointer write always names the version it replaces.

use crate::mirror::Mirror;
use anyhow::{bail, Context, Result};
use jope_datasets::store::{ObservedObject, PutOutcome, StoreLocation};

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
/// configuration could not keep W8's guarantees. The URL is read exactly as the store opens it
/// (`jope_datasets::store::StoreLocation`, the reading `Store::open_with` uses), so every
/// spelling the store treats as S3 (`s3`/`s3a` in any case) is checked, at the prefix the store
/// would write under after normalization (`.` and `..` segments, percent-encoded ones included,
/// removed). Local (`file://`, bare path) and `memory:///` stores need nothing. An S3 store must:
/// - name a prefix under the bucket (a run-scoped prefix; never the bucket root);
/// - sign its requests (`JOPE_DATASETS_ANON` is not `1`);
/// - keep object_store's conditional writes on (`AWS_CONDITIONAL_PUT` unset or `etag`): every
///   immutable write is `If-None-Match: *` and every pointer write `If-Match`, with no
///   unconditional fallback.
///
/// `env` reads one environment variable; the CLI passes the process environment.
pub fn preflight_publishing_store(url: &str, env: impl Fn(&str) -> Option<String>) -> Result<()> {
    let location = StoreLocation::parse(url)
        .with_context(|| format!("{url}: not a store URL. Nothing was written"))?;
    let Some(s3) = &location.s3 else {
        return Ok(());
    };
    if s3.bucket.is_empty() || s3.prefix.as_ref().is_empty() {
        bail!(
            "{url}: a progress store on S3 names a run-scoped prefix under the bucket \
             (s3://<bucket>/<prefix>); the bucket root is refused (this URL opens the store at \
             bucket {:?}, prefix {:?}). Nothing was written",
            s3.bucket,
            s3.prefix.as_ref()
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

#[cfg(test)]
mod tests {
    use super::preflight_publishing_store as preflight;
    use jope_datasets::store::StoreLocation;

    type Env = fn(&str) -> Option<String>;
    fn signed(_: &str) -> Option<String> {
        None
    }
    fn unsigned(name: &str) -> Option<String> {
        (name == "JOPE_DATASETS_ANON").then(|| "1".to_owned())
    }
    fn unconditional(name: &str) -> Option<String> {
        (name == "AWS_CONDITIONAL_PUT").then(|| "disabled".to_owned())
    }
    fn both(name: &str) -> Option<String> {
        unsigned(name).or_else(|| unconditional(name))
    }

    /// `url` under `env` is refused with `needle`; otherwise the bypass is recorded.
    fn expect_refusal(url: &str, env: (&str, Env), needle: &str, bypasses: &mut Vec<String>) {
        match preflight(url, env.1) {
            Ok(()) => bypasses.push(format!("{url} ({}): passed the preflight", env.0)),
            Err(e) if !format!("{e:#}").contains(needle) => bypasses.push(format!(
                "{url} ({}): refused without {needle:?}: {e:#}",
                env.0
            )),
            Err(_) => {}
        }
    }

    /// PR #114 review round 1 (the reviewer's probe, as a refusal): the preflight reads a store
    /// URL exactly as the store opens it. Every spelling the store reads as an S3 bucket root is
    /// refused whatever the environment: an upper-case scheme, `.` and `..` segments (also
    /// percent-encoded), an empty segment `object_store` cannot parse as a prefix, no bucket.
    /// Every S3 spelling with a run-scoped prefix still goes through the unsigned-request and
    /// conditional-write checks. Pure URL parsing: no store is opened, and no credential or
    /// network is used.
    #[test]
    fn every_s3_spelling_the_store_opens_goes_through_the_preflight() {
        let mut bypasses = Vec::new();
        let root = [
            "s3://example-bucket",
            "S3://example-bucket",
            "s3://example-bucket/",
            "s3://example-bucket//",
            "s3://example-bucket/.",
            "s3://example-bucket/run/..",
            "s3://example-bucket/run/%2e%2e",
            "s3://example-bucket/%2E",
            "S3A://example-bucket/run/./../",
            "s3://example-bucket/run//1",
            "s3:///run-1",
        ];
        for url in root {
            for env in [("signed", signed as Env), ("unsigned, unconditional", both)] {
                expect_refusal(url, env, "the bucket root is refused", &mut bypasses);
            }
        }
        let prefixed = [
            "s3://example-bucket/run-1",
            "S3://example-bucket/run-1",
            "s3a://example-bucket/runs/../run-1",
            "S3://example-bucket/run-1/./",
        ];
        for url in prefixed {
            if let Err(e) = preflight(url, signed) {
                bypasses.push(format!("{url} (signed): refused: {e:#}"));
            }
            expect_refusal(
                url,
                ("unsigned", unsigned),
                "JOPE_DATASETS_ANON=1",
                &mut bypasses,
            );
            expect_refusal(
                url,
                ("unconditional", unconditional),
                "AWS_CONDITIONAL_PUT",
                &mut bypasses,
            );
        }
        for url in [
            "memory:///",
            "file:///var/lib/jope/progress",
            "progress-store",
        ] {
            if let Err(e) = preflight(url, both) {
                bypasses.push(format!("{url}: a non-S3 store was refused: {e:#}"));
            }
        }
        assert!(bypasses.is_empty(), "{}", bypasses.join("\n"));

        // The preflight and the store read each URL the same way (`StoreLocation`, which
        // `Store::open_with` uses): the root spellings open the store at the bucket root (or name
        // no bucket), the prefixed ones at `run-1` in `example-bucket`, and the others are not S3.
        let s3 = |url: &str| {
            StoreLocation::parse(url)
                .unwrap()
                .s3
                .map(|s3| (s3.bucket, s3.prefix.as_ref().to_owned()))
        };
        for url in root {
            let (bucket, prefix) = s3(url).expect(url);
            assert!(
                bucket.is_empty() || prefix.is_empty(),
                "{url}: {bucket} {prefix}"
            );
        }
        for url in prefixed {
            assert_eq!(
                s3(url),
                Some(("example-bucket".into(), "run-1".into())),
                "{url}"
            );
        }
        for url in [
            "memory:///",
            "file:///var/lib/jope/progress",
            "progress-store",
        ] {
            assert_eq!(s3(url), None, "{url}");
        }
    }
}
