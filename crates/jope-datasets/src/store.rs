//! One transport for every backend through `object_store`: `s3://` (env credentials, anonymous
//! with `JOPE_DATASETS_ANON=1`), `file://` (tests, local mirrors) and `memory://`.

use anyhow::{anyhow, Context, Result};
use futures_util::StreamExt;
use object_store::path::Path as ObjPath;
use object_store::{
    GetOptions, GetRange, ObjectStore, ObjectStoreExt, PutMode, PutOptions, PutPayload,
};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::Path;
use url::Url;

pub struct Store {
    inner: Box<dyn ObjectStore>,
    prefix: ObjPath,
    pub url: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PutOutcome {
    Created,
    Exists,
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

    /// [`Store::open`] with an explicit choice of anonymous (unsigned) S3 requests.
    pub fn open_with(url: &str, anonymous: bool) -> Result<Store> {
        let url_string = if url.contains("://") {
            url.to_string()
        } else {
            let abs =
                std::path::absolute(url).with_context(|| format!("absolute path of {url}"))?;
            format!("file://{}", abs.display())
        };
        let parsed = Url::parse(&url_string).with_context(|| format!("parse {url_string}"))?;
        let (inner, prefix): (Box<dyn ObjectStore>, ObjPath) = match parsed.scheme() {
            "s3" | "s3a" => {
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
                }
                let store = builder.build().context("build S3 store")?;
                let prefix = ObjPath::parse(parsed.path().trim_matches('/')).unwrap_or_default();
                (Box::new(store), prefix)
            }
            _ => object_store::parse_url(&parsed).with_context(|| format!("open {url_string}"))?,
        };
        Ok(Store {
            inner,
            prefix,
            url: url_string,
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
        match self.inner.get(&self.key(rel)).await {
            Ok(r) => Ok(Some(
                r.bytes()
                    .await
                    .with_context(|| format!("read {rel}"))?
                    .to_vec(),
            )),
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
        let _lock = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&lock_path)?;
        _lock
            .lock()
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
