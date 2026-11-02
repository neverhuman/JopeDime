use super::*;

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
    pub(crate) inner: Box<dyn ObjectStore>,
    pub(crate) prefix: ObjPath,
    pub url: String,
    pub(crate) local_root: Option<std::path::PathBuf>,
}

pub(crate) type InjectedRegistry =
    std::collections::BTreeMap<usize, (ObjectStoreScheme, std::sync::Arc<dyn InjectedBackend>)>;

fn injected_registry() -> &'static std::sync::Mutex<InjectedRegistry> {
    static REGISTRY: std::sync::OnceLock<std::sync::Mutex<InjectedRegistry>> =
        std::sync::OnceLock::new();
    REGISTRY.get_or_init(|| std::sync::Mutex::new(std::collections::BTreeMap::new()))
}

pub(crate) fn injected_registry_guard() -> std::sync::MutexGuard<'static, InjectedRegistry> {
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
    /// The location's prefix. Empty when the store writes at the bucket root. Invalid S3 prefixes
    /// are rejected so a malformed path cannot silently redirect writes to the bucket root.
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
            let prefix = ObjPath::parse(parsed.path().trim_matches('/'))
                .with_context(|| format!("parse S3 object prefix in {url_string}"))?;
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
