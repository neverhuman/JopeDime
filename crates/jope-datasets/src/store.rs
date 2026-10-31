//! One transport for every backend through `object_store`: `s3://` (native AWS providers or selected CLI profile, anonymous
//! with `JOPE_DATASETS_ANON=1`), `file://` (tests, local mirrors) and `memory://`.

use anyhow::{anyhow, ensure, Context, Result};
use futures_util::StreamExt;
use object_store::{
    GetOptions, GetRange, ObjectStore, ObjectStoreExt, PutMode, PutOptions, PutPayload,
};
use sha2::{Digest, Sha256};
use std::io::Write;
use std::path::Path;
use url::Url;

pub use object_store::path::Path as ObjPath;
pub use object_store::{ObjectStoreScheme, UpdateVersion};

/// Deterministic injected backend seam for testing versioned object stores (`VersionId` vs `ETag`),
/// concurrent create races, readback corruption, and lost-response reconciliation without live
/// remote network calls.
pub trait InjectedBackend: Send + Sync {
    /// Read the latest version (`required_version == None`) or a specific physical `VersionId`
    /// (`required_version == Some(v)`), returning `(bytes, UpdateVersion)` from one atomic read.
    fn get_versioned(
        &self,
        key: &ObjPath,
        required_version: Option<&str>,
    ) -> Result<Option<(Vec<u8>, UpdateVersion)>>;

    /// Create-only put (`If-None-Match: *`).
    fn put_create(
        &self,
        key: &ObjPath,
        bytes: Vec<u8>,
    ) -> Result<(PutOutcome, Option<UpdateVersion>)>;

    /// Unconditional overwrite (used by tests to simulate adversarial version replacement or
    /// local pointer updates).
    fn put_overwrite(&self, key: &ObjPath, bytes: Vec<u8>) -> Result<UpdateVersion>;

    /// List all object keys under `prefix`.
    fn list_keys(&self, prefix: &ObjPath) -> Result<Vec<ObjPath>>;
}

pub struct Store {
    inner: Box<dyn ObjectStore>,
    prefix: ObjPath,
    pub url: String,
    local_root: Option<std::path::PathBuf>,
}

type InjectedRegistry =
    std::collections::BTreeMap<usize, (ObjectStoreScheme, std::sync::Arc<dyn InjectedBackend>)>;

fn injected_registry() -> &'static std::sync::Mutex<InjectedRegistry> {
    static REGISTRY: std::sync::OnceLock<std::sync::Mutex<InjectedRegistry>> =
        std::sync::OnceLock::new();
    REGISTRY.get_or_init(|| std::sync::Mutex::new(std::collections::BTreeMap::new()))
}

fn injected_registry_guard() -> std::sync::MutexGuard<'static, InjectedRegistry> {
    injected_registry()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Bytes and their precondition from the same GET. Keep this observation until
/// the conditional write; a second GET would discard the original precondition.
#[derive(Debug)]
pub struct ObservedObject {
    pub bytes: Vec<u8>,
    version: UpdateVersion,
    key: ObjPath,
}

impl ObservedObject {
    /// Opaque concurrency token (`ETag`) returned with `bytes` in the same GET.
    pub fn e_tag(&self) -> Option<&str> {
        self.version.e_tag.as_deref()
    }

    /// Physical backend object version (`VersionId`) returned with `bytes` in the same GET.
    pub fn version(&self) -> Option<&str> {
        self.version.version.as_deref()
    }

    /// Full resolved object-store key (`ObjPath`) for this observation.
    pub fn key_str(&self) -> &str {
        self.key.as_ref()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PutOutcome {
    Created,
    Exists,
}

/// How [`Store::open_with`] builds the store for a [`StoreLocation`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StoreRoute {
    /// An `s3://` or `s3a://` URL: object_store's S3 builder, configured from the environment
    /// (credentials, region, request signing), writing under the location's prefix.
    S3,
    /// Every other URL: `object_store::parse_url`, which opens the backend its classifier
    /// ([`ObjectStoreScheme::parse`]) names, at the prefix it names.
    ParseUrl,
}

/// A store URL as [`Store::open_with`] reads it. The store and every caller that must agree with
/// it about where it writes (the W8 publishing preflight,
/// `jope_learning::progress::remote::preflight_publishing_store`) take it from here, so the two
/// cannot disagree:
/// - a bare filesystem path becomes a `file://` URL;
/// - `url::Url` parses and normalizes it: the scheme is lower-cased, and `.` and `..` segments
///   (percent-encoded ones included) are removed;
/// - an `s3://` or `s3a://` URL takes the S3 route: its host is the bucket and its path the
///   object prefix every key is written under;
/// - every other URL takes object_store's own route, and the backend and prefix are the ones
///   object_store's classifier names, exactly as `object_store::parse_url` opens it. That route
///   reaches S3 too: AWS virtual-host (`https://<bucket>.s3.<region>.amazonaws.com/<prefix>`) and
///   path-style (`https://s3.<region>.amazonaws.com/<bucket>/<prefix>`) URLs and R2 URLs
///   (`https://<account>.r2.cloudflarestorage.com/<bucket>/<prefix>`) are S3 locations. A URL the
///   classifier does not recognise is not a store location.
#[derive(Clone, Debug)]
pub struct StoreLocation {
    /// The URL the store opens and records ([`Store::url`]).
    pub url: String,
    pub parsed: Url,
    pub route: StoreRoute,
    /// The backend the store opens.
    pub backend: ObjectStoreScheme,
    /// The object prefix every key is written under; empty at the root of the backend.
    pub prefix: ObjPath,
    /// The bucket and effective object prefix of an S3 store, whichever route reaches S3; `None`
    /// for every other backend.
    pub s3: Option<S3Location>,
}

/// Where an S3 store writes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S3Location {
    /// The bucket the URL names: the host of an `s3://` or `s3a://` URL (empty when it names none);
    /// for an HTTPS URL, where object_store's classifier splits it off the prefix (the first path
    /// segment of a path-style AWS or an R2 URL, else the first label of the host).
    pub bucket: String,
    /// The location's prefix. Empty when the store writes at the bucket root. On the S3 route, a
    /// path `object_store` cannot parse as a prefix (an empty segment, for one) is the root as
    /// well: that is where the store writes.
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
        let (route, backend, prefix) = if matches!(parsed.scheme(), "s3" | "s3a") {
            let prefix = ObjPath::parse(parsed.path().trim_matches('/')).unwrap_or_default();
            (StoreRoute::S3, ObjectStoreScheme::AmazonS3, prefix)
        } else {
            // The two steps `object_store::parse_url` takes before it builds the store.
            let (backend, path) =
                ObjectStoreScheme::parse(&parsed).with_context(|| format!("open {url_string}"))?;
            let prefix = ObjPath::parse(path).with_context(|| format!("open {url_string}"))?;
            (StoreRoute::ParseUrl, backend, prefix)
        };
        let s3 = (backend == ObjectStoreScheme::AmazonS3).then(|| S3Location {
            bucket: match route {
                StoreRoute::S3 => parsed.host_str().unwrap_or_default().to_owned(),
                StoreRoute::ParseUrl => endpoint_bucket(&parsed),
            },
            prefix: prefix.clone(),
        });
        Ok(StoreLocation {
            url: url_string,
            parsed,
            route,
            backend,
            prefix,
            s3,
        })
    }
}

/// The bucket of an HTTPS URL object_store's classifier sends to S3, split where the classifier
/// splits it off the prefix: path-style AWS hosts (`s3…`) and R2 hosts name it in the first path
/// segment, virtual-host AWS hosts in their first label.
fn endpoint_bucket(parsed: &Url) -> String {
    let host = parsed.host_str().unwrap_or_default();
    if host.starts_with("s3") || host.ends_with("r2.cloudflarestorage.com") {
        let path = parsed.path();
        let path = path.strip_prefix('/').unwrap_or(path);
        path.split('/').next().unwrap_or_default().to_owned()
    } else {
        host.split('.').next().unwrap_or_default().to_owned()
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
    /// by [`StoreLocation::parse`]: the store takes that location's route and writes under its
    /// prefix.
    pub fn open_with(url: &str, anonymous: bool) -> Result<Store> {
        let StoreLocation {
            url: url_string,
            parsed,
            route,
            backend,
            prefix,
            ..
        } = StoreLocation::parse(url)?;
        let local_root = if backend == ObjectStoreScheme::Local {
            Some(
                parsed
                    .to_file_path()
                    .map_err(|()| anyhow!("invalid file URL"))?,
            )
        } else {
            None
        };
        let inner: Box<dyn ObjectStore> = match route {
            StoreRoute::S3 => {
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
                Box::new(store)
            }
            StoreRoute::ParseUrl => {
                let (store, opened_at) = object_store::parse_url(&parsed)
                    .with_context(|| format!("open {url_string}"))?;
                // parse_url classifies the URL as the location did. Were they ever to disagree,
                // the store would not write where its callers checked, so it is not opened.
                ensure!(
                    opened_at == prefix,
                    "open {url_string}: object_store opens it at prefix {:?}, not {:?}",
                    opened_at.as_ref(),
                    prefix.as_ref()
                );
                store
            }
        };
        Ok(Store {
            inner,
            prefix,
            url: url_string,
            local_root,
        })
    }

    fn inner_id(&self) -> usize {
        (&*self.inner as *const dyn ObjectStore).cast::<()>() as usize
    }

    fn injected_backend(&self) -> Option<std::sync::Arc<dyn InjectedBackend>> {
        injected_registry_guard()
            .get(&self.inner_id())
            .map(|(_, b)| b.clone())
    }

    /// Construct a [`Store`] with an explicitly injected [`ObjectStore`] backend (for deterministic
    /// versioned/fault-injecting tests without live remote network access).
    pub fn from_object_store(
        url: impl Into<String>,
        prefix: ObjPath,
        _backend: ObjectStoreScheme,
        inner: Box<dyn ObjectStore>,
    ) -> Store {
        Store {
            inner,
            prefix,
            url: url.into(),
            local_root: None,
        }
    }

    /// Construct a [`Store`] backed by a deterministic [`InjectedBackend`] implementation (for
    /// testing versioned stores, create races, and lost-response reconciliation without live
    /// remote network calls).
    pub fn from_injected_backend(
        url: impl Into<String>,
        prefix: ObjPath,
        backend: ObjectStoreScheme,
        injected: std::sync::Arc<dyn InjectedBackend>,
    ) -> Store {
        let inner: Box<dyn ObjectStore> = Box::new(object_store::memory::InMemory::new());
        let id = (&*inner as *const dyn ObjectStore).cast::<()>() as usize;
        injected_registry_guard().insert(id, (backend, injected));
        Store {
            inner,
            prefix,
            url: url.into(),
            local_root: None,
        }
    }

    /// The classified [`ObjectStoreScheme`] for this store.
    pub fn backend(&self) -> ObjectStoreScheme {
        if let Some((scheme, _)) = injected_registry_guard().get(&self.inner_id()) {
            return scheme.clone();
        }
        StoreLocation::parse(&self.url)
            .map(|l| l.backend)
            .unwrap_or(ObjectStoreScheme::Local)
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
        self.get_observed_version(rel, None).await
    }

    /// Read an object (optionally pinned to a specific backend physical `version`) and capture the
    /// [`UpdateVersion`] (`e_tag` and physical `version`) returned with those exact bytes in one GET.
    pub async fn get_observed_version(
        &self,
        rel: &str,
        required_version: Option<&str>,
    ) -> Result<Option<ObservedObject>> {
        let key = self.key(rel);
        if let Some(injected) = self.injected_backend() {
            let Some((bytes, version)) = injected.get_versioned(&key, required_version)? else {
                return Ok(None);
            };
            return Ok(Some(ObservedObject {
                bytes,
                version,
                key,
            }));
        }
        let opts = GetOptions {
            version: required_version.map(str::to_owned),
            ..Default::default()
        };
        match self.inner.get_opts(&key, opts).await {
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
                    key,
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
        if let Some(injected) = self.injected_backend() {
            return Ok(injected
                .get_versioned(&self.key(rel), None)?
                .map(|(b, _)| b.len() as u64));
        }
        match self.inner.head(&self.key(rel)).await {
            Ok(m) => Ok(Some(m.size)),
            Err(object_store::Error::NotFound { .. }) => Ok(None),
            Err(e) => Err(anyhow!("head {rel}: {e}")),
        }
    }

    /// Create-only put (`If-None-Match: *`) returning the [`UpdateVersion`] (`e_tag` and physical
    /// `version`) reported by the backend when created. A raw [`PutOutcome`] alone is not a
    /// content-verified receipt; callers requiring content/version verification use
    /// [`crate::verified_object`].
    pub async fn put_if_absent_with_result(
        &self,
        rel: &str,
        bytes: Vec<u8>,
    ) -> Result<(PutOutcome, Option<UpdateVersion>)> {
        if let Some(injected) = self.injected_backend() {
            return injected.put_create(&self.key(rel), bytes);
        }
        let opts = PutOptions {
            mode: PutMode::Create,
            ..Default::default()
        };
        match self
            .inner
            .put_opts(&self.key(rel), PutPayload::from(bytes), opts)
            .await
        {
            Ok(res) => Ok((
                PutOutcome::Created,
                Some(UpdateVersion {
                    e_tag: res.e_tag,
                    version: res.version,
                }),
            )),
            Err(object_store::Error::AlreadyExists { .. }) => Ok((PutOutcome::Exists, None)),
            Err(e) => Err(anyhow!("put {rel}: {e}")),
        }
    }

    /// Create-only put (`If-None-Match: *`): an existing key is never rewritten.
    pub async fn put_if_absent(&self, rel: &str, bytes: Vec<u8>) -> Result<PutOutcome> {
        Ok(self.put_if_absent_with_result(rel, bytes).await?.0)
    }

    pub async fn put(&self, rel: &str, bytes: Vec<u8>) -> Result<()> {
        if let Some(injected) = self.injected_backend() {
            injected.put_overwrite(&self.key(rel), bytes)?;
            return Ok(());
        }
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
        if let Some(injected) = self.injected_backend() {
            let mut out = Vec::new();
            for location in injected.list_keys(&prefix)? {
                let full = location.as_ref().to_string();
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
            return Ok(out);
        }
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
            if m.len() == total
                && crate::hash::digest_content_async(dest.to_path_buf())
                    .await?
                    .sha256
                    == expected_sha256
            {
                return Ok(total);
            }
            std::fs::remove_file(dest)?;
        }
        let mut hasher = Sha256::new();
        let mut start = 0u64;
        if let Ok(m) = std::fs::metadata(&partial) {
            if m.len() >= resume_threshold && m.len() <= total {
                let (prefix_hasher, prefix_bytes) =
                    crate::hash::content_hasher_async(partial.clone()).await?;
                ensure!(prefix_bytes == m.len(), "partial changed while hashing");
                hasher = prefix_hasher;
                start = prefix_bytes;
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

#[cfg(test)]
mod tests;
