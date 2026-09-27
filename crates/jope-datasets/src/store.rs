//! One transport for every backend through `object_store`: `s3://` (native AWS providers or selected CLI profile, anonymous
//! with `JOPE_DATASETS_ANON=1`), `file://` (tests, local mirrors) and `memory://`.

use anyhow::{anyhow, ensure, Context, Result};
use futures_util::StreamExt;
use object_store::path::Path as ObjPath;
use object_store::{
    GetOptions, GetRange, ObjectStore, ObjectStoreExt, PutMode, PutOptions, PutPayload,
    UpdateVersion,
};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::Path;
use url::Url;

pub struct Store {
    inner: Box<dyn ObjectStore>,
    prefix: ObjPath,
    pub url: String,
    local_root: Option<std::path::PathBuf>,
}

/// Bytes and their precondition from the same GET. Keep this observation until
/// the conditional write; a second GET would discard the original precondition.
#[derive(Debug)]
pub struct ObservedObject {
    pub bytes: Vec<u8>,
    version: UpdateVersion,
    key: ObjPath,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PutOutcome {
    Created,
    Exists,
}

/// A store URL as [`Store::open_with`] reads it. The store and every caller that must agree with
/// it about where it writes (the W8 publishing preflight,
/// `jope_learning::progress::remote::preflight_publishing_store`) take it from here, so the two
/// cannot disagree:
/// - a bare filesystem path becomes a `file://` URL;
/// - `url::Url` parses and normalizes it: the scheme is lower-cased, and `.` and `..` segments
///   (percent-encoded ones included) are removed;
/// - an `s3://` or `s3a://` URL names its bucket and the object prefix every key is written under.
#[derive(Clone, Debug)]
pub struct StoreLocation {
    /// The URL the store opens and records ([`Store::url`]).
    pub url: String,
    pub parsed: Url,
    /// The bucket and effective object prefix of an S3 store; `None` for every other backend.
    pub s3: Option<S3Location>,
}

/// Where an S3 store writes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S3Location {
    pub bucket: String,
    /// Empty when the store writes at the bucket root. A path `object_store` cannot parse as a
    /// prefix (an empty segment, for one) is the root as well: that is where the store writes.
    pub prefix: ObjPath,
}

impl StoreLocation {
    pub fn parse(url: &str) -> Result<StoreLocation> {
        let url_string = if url.contains("://") {
            url.to_string()
        } else {
            let abs =
                std::path::absolute(url).with_context(|| format!("absolute path of {url}"))?;
            format!("file://{}", abs.display())
        };
        let parsed = Url::parse(&url_string).with_context(|| format!("parse {url_string}"))?;
        let s3 = matches!(parsed.scheme(), "s3" | "s3a").then(|| S3Location {
            bucket: parsed.host_str().unwrap_or_default().to_owned(),
            prefix: ObjPath::parse(parsed.path().trim_matches('/')).unwrap_or_default(),
        });
        Ok(StoreLocation {
            url: url_string,
            parsed,
            s3,
        })
    }
}

/// `JOPE_DATASETS_URL`, else the default bucket.
pub fn datasets_url() -> String {
    std::env::var("JOPE_DATASETS_URL")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| crate::DEFAULT_DATASETS_URL.to_string())
}

impl Store {
    pub fn from_env() -> Result<Store> {
        Store::open(&datasets_url())
    }

    /// Open `s3://bucket[/prefix]`, `file:///dir`, `memory:///` or a bare filesystem path;
    /// S3 requests are signed unless `JOPE_DATASETS_ANON=1`.
    pub fn open(url: &str) -> Result<Store> {
        let anonymous = std::env::var("JOPE_DATASETS_ANON").is_ok_and(|v| v == "1");
        Store::open_with(url, anonymous)
    }

    /// [`Store::open`] with an explicit choice of anonymous (unsigned) S3 requests. The URL is read
    /// by [`StoreLocation::parse`].
    pub fn open_with(url: &str, anonymous: bool) -> Result<Store> {
        let StoreLocation {
            url: url_string,
            parsed,
            s3,
        } = StoreLocation::parse(url)?;
        let local_root = if parsed.scheme() == "file" {
            Some(
                parsed
                    .to_file_path()
                    .map_err(|()| anyhow!("invalid file URL"))?,
            )
        } else {
            None
        };
        let (inner, prefix): (Box<dyn ObjectStore>, ObjPath) = match s3 {
            Some(S3Location { prefix, .. }) => {
                // object_store's default request timeout is 30 s, which a 200–250 MB blob GET
                // exceeds on a shared link (seen against the real bucket: "request or response
                // body error" on the three largest starforge/hyperion blobs); allow up to an
                // hour per request, keep a short connect timeout.
                let client_options = object_store::ClientOptions::new()
                    .with_timeout(std::time::Duration::from_secs(3600))
                    .with_connect_timeout(std::time::Duration::from_secs(30));
                let mut builder = object_store::aws::AmazonS3Builder::from_env()
                    .with_url(url_string.clone())
                    .with_client_options(client_options);
                if std::env::var_os("AWS_REGION").is_none()
                    && std::env::var_os("AWS_DEFAULT_REGION").is_none()
                {
                    builder = builder.with_region("us-east-2");
                }
                if anonymous {
                    builder = builder.with_skip_signature(true);
                } else if let Some(credentials) = crate::credentials::resolve()? {
                    // `from_env` above already consumed any credentials the environment carried;
                    // this covers the profile case it cannot see, and fails by name rather than
                    // through the IMDS fallback. See `crate::credentials`.
                    builder = credentials.apply_to(builder);
                }
                let store = builder.build().context("build S3 store")?;
                (Box::new(store), prefix)
            }
            None => {
                object_store::parse_url(&parsed).with_context(|| format!("open {url_string}"))?
            }
        };
        Ok(Store {
            inner,
            prefix,
            url: url_string,
            local_root,
        })
    }

    pub fn key(&self, rel: &str) -> ObjPath {
        if self.prefix.as_ref().is_empty() {
            ObjPath::from(rel)
        } else {
            ObjPath::from(format!("{}/{rel}", self.prefix.as_ref()))
        }
    }

    /// `None` when the key does not exist.
    pub async fn get_bytes(&self, rel: &str) -> Result<Option<Vec<u8>>> {
        Ok(self.get_observed(rel).await?.map(|object| object.bytes))
    }

    /// Read a pointer and the version belonging to those exact bytes.
    pub async fn get_observed(&self, rel: &str) -> Result<Option<ObservedObject>> {
        match self.inner.get(&self.key(rel)).await {
            Ok(r) => {
                let version = UpdateVersion {
                    e_tag: r.meta.e_tag.clone(),
                    version: r.meta.version.clone(),
                };
                let bytes = r
                    .bytes()
                    .await
                    .with_context(|| format!("read {rel}"))?
                    .to_vec();
                Ok(Some(ObservedObject {
                    bytes,
                    version,
                    key: self.key(rel),
                }))
            }
            Err(object_store::Error::NotFound { .. }) => Ok(None),
            Err(e) => Err(anyhow!("get {rel}: {e}")),
        }
    }

    pub async fn get_string(&self, rel: &str) -> Result<Option<String>> {
        Ok(self
            .get_bytes(rel)
            .await?
            .map(|b| String::from_utf8_lossy(&b).into_owned()))
    }

    pub async fn size(&self, rel: &str) -> Result<Option<u64>> {
        match self.inner.head(&self.key(rel)).await {
            Ok(m) => Ok(Some(m.size)),
            Err(object_store::Error::NotFound { .. }) => Ok(None),
            Err(e) => Err(anyhow!("head {rel}: {e}")),
        }
    }

    /// Create-only put (`If-None-Match: *`): an existing key is never rewritten.
    pub async fn put_if_absent(&self, rel: &str, bytes: Vec<u8>) -> Result<PutOutcome> {
        let opts = PutOptions {
            mode: PutMode::Create,
            ..Default::default()
        };
        match self
            .inner
            .put_opts(&self.key(rel), PutPayload::from(bytes), opts)
            .await
        {
            Ok(_) => Ok(PutOutcome::Created),
            Err(object_store::Error::AlreadyExists { .. }) => Ok(PutOutcome::Exists),
            Err(e) => Err(anyhow!("put {rel}: {e}")),
        }
    }

    pub async fn put(&self, rel: &str, bytes: Vec<u8>) -> Result<()> {
        self.inner
            .put(&self.key(rel), PutPayload::from(bytes))
            .await
            .map_err(|e| anyhow!("put {rel}: {e}"))?;
        Ok(())
    }

    /// Create when the observed pointer was absent, otherwise replace only the
    /// observed version. Unsupported remote preconditions fail without fallback.
    /// Local mirrors serialize cooperating pointer publishers on the root directory
    /// inode, then revalidate under that lock: object_store's local backend does
    /// not implement Update. Raw filesystem writers do not participate in this lock.
    pub async fn put_conditional(
        &self,
        rel: &str,
        bytes: Vec<u8>,
        observed: Option<&ObservedObject>,
    ) -> Result<()> {
        if let Some(object) = observed {
            ensure!(
                object.key == self.key(rel),
                "pointer observation belongs to another key"
            );
            ensure!(
                object.version.e_tag.is_some() || object.version.version.is_some(),
                "{rel}: backend returned no pointer version"
            );
        }
        if let Some(root) = &self.local_root {
            let root = root.clone();
            // Never block the async executor while another writer holds the lock.
            // Opening the directory separately makes independent Store instances
            // and processes coordinate without polluting the object namespace.
            let _lock = tokio::task::spawn_blocking(move || -> Result<std::fs::File> {
                std::fs::create_dir_all(&root)?;
                let lock = std::fs::File::open(&root)?;
                lock.lock()?;
                Ok(lock)
            })
            .await
            .context("pointer lock task")??;
            let current = self.get_observed(rel).await?;
            ensure!(
                match (&current, observed) {
                    (None, None) => true,
                    (Some(current), Some(expected)) =>
                        current.version == expected.version && current.bytes == expected.bytes,
                    _ => false,
                },
                "{rel}: mirror pointer changed since observation"
            );
            return self.put(rel, bytes).await;
        }
        let opts = PutOptions {
            mode: observed.map_or(PutMode::Create, |object| {
                PutMode::Update(object.version.clone())
            }),
            ..Default::default()
        };
        self.inner
            .put_opts(&self.key(rel), PutPayload::from(bytes), opts)
            .await
            .with_context(|| format!("{rel}: conditional pointer write failed"))?;
        Ok(())
    }

    pub async fn list(&self, rel_prefix: &str) -> Result<Vec<String>> {
        let prefix = self.key(rel_prefix);
        let mut stream = self.inner.list(Some(&prefix));
        let mut out = Vec::new();
        while let Some(item) = stream.next().await {
            let meta = item.map_err(|e| anyhow!("list {rel_prefix}: {e}"))?;
            let full = meta.location.as_ref().to_string();
            let rel = if self.prefix.as_ref().is_empty() {
                full
            } else {
                full.strip_prefix(&format!("{}/", self.prefix.as_ref()))
                    .unwrap_or(&full)
                    .to_string()
            };
            out.push(rel);
        }
        out.sort();
        Ok(out)
    }

    /// Download `rel` to `dest` via `dest.partial`, hashing as it streams; rename only when the
    /// sha256 matches `expected_sha256`, otherwise delete the partial and fail. A partial of at
    /// least `resume_threshold` bytes is resumed with a ranged GET.
    pub async fn download_verified(
        &self,
        rel: &str,
        dest: &Path,
        expected_sha256: &str,
        resume_threshold: u64,
    ) -> Result<u64> {
        let partial = dest.with_file_name(format!(
            "{}.partial",
            dest.file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default()
        ));
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }
        // One writer per destination across threads AND processes: an exclusive advisory lock on
        // `<dest>.partial.lock`. Without it two fetches of the same blob (two services sharing one
        // $JOPE_HOME, or a retried CI job) appended to one `.partial` and produced a blob that was
        // the content twice (seen against the real bucket). The lock is released when `_lock`
        // drops; the lock file itself is harmless to leave behind.
        let lock_path = dest.with_file_name(format!(
            "{}.partial.lock",
            dest.file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default()
        ));
        let lock_file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&lock_path)?;
        // `File::lock` blocks its thread until the lock is granted, so it runs on the blocking
        // pool: taken inline it would block the runtime thread polling this future, and when
        // several `ensure` futures share one polling thread (`join_all` under `block_on`) the
        // holder is never polled again to release the lock (a deadlock, seen in the concurrency
        // test).
        let _lock = tokio::task::spawn_blocking(move || lock_file.lock().map(|()| lock_file))
            .await
            .context("lock task")?
            .with_context(|| format!("lock {}", lock_path.display()))?;
        let total = self
            .size(rel)
            .await?
            .ok_or_else(|| anyhow!("{rel}: not found in {}", self.url))?;
        // Another writer may have completed this destination while we waited for the lock.
        if let Ok(m) = std::fs::metadata(dest) {
            if m.len() == total && digest_file_hex(dest)? == expected_sha256 {
                return Ok(total);
            }
            std::fs::remove_file(dest)?;
        }
        let mut hasher = Sha256::new();
        let mut start = 0u64;
        if let Ok(m) = std::fs::metadata(&partial) {
            if m.len() >= resume_threshold && m.len() <= total {
                let mut f = std::fs::File::open(&partial)?;
                let mut buf = vec![0u8; 1 << 20];
                loop {
                    let n = f.read(&mut buf)?;
                    if n == 0 {
                        break;
                    }
                    hasher.update(&buf[..n]);
                }
                start = m.len();
            } else {
                let _ = std::fs::remove_file(&partial);
            }
        }
        // Resume appends to the verified prefix; a fresh download truncates whatever is there.
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(start > 0)
            .write(true)
            .truncate(start == 0)
            .open(&partial)?;
        if start < total {
            let opts = GetOptions {
                range: (start > 0).then_some(GetRange::Offset(start)),
                ..Default::default()
            };
            let result = self
                .inner
                .get_opts(&self.key(rel), opts)
                .await
                .map_err(|e| anyhow!("get {rel}: {e}"))?;
            let mut stream = result.into_stream();
            while let Some(chunk) = stream.next().await {
                let chunk = chunk.map_err(|e| anyhow!("stream {rel}: {e}"))?;
                hasher.update(&chunk);
                file.write_all(&chunk)?;
            }
        }
        file.flush()?;
        drop(file);
        let got = hex::encode(hasher.finalize());
        if got == expected_sha256 {
            std::fs::rename(&partial, dest)?;
            Ok(total)
        } else {
            let _ = std::fs::remove_file(&partial);
            Err(anyhow!(
                "{rel}: sha256 mismatch (expected {expected_sha256}, got {got}); partial deleted"
            ))
        }
    }
}

/// sha256 (hex) of a local file, streamed.
fn digest_file_hex(path: &Path) -> Result<String> {
    let mut f = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}

#[cfg(test)]
mod tests;
