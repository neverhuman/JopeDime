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

mod location;
mod transfer;
#[cfg(test)]
use location::injected_registry;
use location::injected_registry_guard;
pub use location::{
    datasets_url, InjectedBackend, ObservedObject, PutOutcome, S3Location, Store, StoreLocation,
    StoreRoute,
};

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
}

#[cfg(test)]
mod tests;
