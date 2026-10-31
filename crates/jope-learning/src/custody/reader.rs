//! Injected single-attempt read transport. The custody CLI supplies no live S3 adapter.
use super::{identity, unsigned};
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::{
    collections::BTreeSet,
    io::Read,
    time::{Duration, Instant},
};

/// `GetResponse` is a record for the get response contract in this module: Injected single-attempt read transport. The custody CLI supplies no live S3 adapter.
pub struct GetResponse {
    /// Identifier for the associated version.
    pub version_id: String,
    /// Content length recorded for this get response.
    pub content_length: u64,
    /// Body recorded for this get response.
    pub body: Box<dyn Read>,
}

/// Each invocation represents ONE physical send; retries must return through the guard.
pub trait ReadTransport {
    /// Performs the list versions operation and returns an error if it cannot be completed.
    fn list_versions(&mut self, params: &Value) -> Result<Value>;
    /// Looks up version and returns the matching value when present.
    fn get_version(&mut self, params: &Value) -> Result<GetResponse>;
}

/// `Reader` is a record for the reader contract in this module: Injected single-attempt read transport. The custody CLI supplies no live S3 adapter.
pub struct Reader {
    bucket: String,
    prefix: String,
    deadline: Instant,
    max_requests: usize,
    metadata_cap: u64,
    body_cap: u64,
    metadata_bytes: u64,
    body_bytes: u64,
    requests: Vec<Value>,
}

impl Reader {
    /// Constructs or opens a new value from the supplied inputs.
    pub fn new(
        bucket: &str,
        prefix: &str,
        max_requests: usize,
        metadata_cap: u64,
        body_cap: u64,
        seconds: f64,
    ) -> Result<Self> {
        ensure!(
            !bucket.is_empty() && !prefix.is_empty() && prefix.ends_with('/'),
            "explicit bucket/prefix"
        );
        ensure!(
            seconds.is_finite() && seconds > 0. && seconds <= 900.,
            "finite work deadline"
        );
        ensure!(
            max_requests <= 1500
                && metadata_cap <= 32 * 1024 * 1024
                && body_cap <= 64 * 1024 * 1024,
            "bounded read transport envelope"
        );
        let duration = Duration::try_from_secs_f64(seconds)?;
        Ok(Self {
            bucket: bucket.into(),
            prefix: prefix.into(),
            deadline: Instant::now()
                .checked_add(duration)
                .context("deadline overflow")?,
            max_requests,
            metadata_cap,
            body_cap,
            metadata_bytes: 0,
            body_bytes: 0,
            requests: Vec::new(),
        })
    }

    fn check(&self) -> Result<()> {
        ensure!(
            Instant::now() < self.deadline,
            "original read window expired"
        );
        Ok(())
    }

    fn send(&mut self, operation: &str, params: &Value) -> Result<()> {
        self.check()?;
        ensure!(params["Bucket"] == self.bucket, "bucket scope");
        let field = match operation {
            "ListObjectVersions" => "Prefix",
            "GetObject" => "Key",
            _ => anyhow::bail!("read-only operation allowlist"),
        };
        ensure!(
            params[field]
                .as_str()
                .is_some_and(|p| p.starts_with(&self.prefix)),
            "key/prefix scope"
        );
        if operation == "GetObject" {
            ensure!(
                params["VersionId"]
                    .as_str()
                    .is_some_and(|v| !v.is_empty() && v != "null"),
                "exact immutable GET version"
            );
        }
        ensure!(
            self.requests.len() < self.max_requests,
            "physical HTTP request cap"
        );
        self.requests
            .push(json!({"operation":operation,"outcome":null,"response_bytes":null}));
        Ok(())
    }

    /// Performs the census operation and returns an error if it cannot be completed.
    pub fn census(&mut self, transport: &mut impl ReadTransport) -> Result<Value> {
        let mut params = json!({"Bucket":self.bucket,"Prefix":self.prefix,"MaxKeys":1000});
        let mut rows = Vec::new();
        let mut markers = Vec::new();
        let mut seen = BTreeSet::new();
        let mut cursors = BTreeSet::new();
        loop {
            self.send("ListObjectVersions", &params)?;
            let response = transport.list_versions(&params)?;
            let bytes = serde_json::to_vec(&response)?.len() as u64;
            self.metadata_bytes = self
                .metadata_bytes
                .checked_add(bytes)
                .context("metadata overflow")?;
            ensure!(
                self.metadata_bytes <= self.metadata_cap,
                "metadata byte cap"
            );
            for (key, destination) in [("Versions", &mut rows), ("DeleteMarkers", &mut markers)] {
                let entries = response
                    .get(key)
                    .map(|v| v.as_array().context("listed entries array"))
                    .transpose()?;
                for item in entries.into_iter().flatten() {
                    let name = item["Key"].as_str().context("listed key")?;
                    ensure!(name.starts_with(&self.prefix), "listed key outside prefix");
                    let mut row = json!({"bucket":self.bucket,"key":name,"version_id":item["VersionId"],
                        "is_latest":item["IsLatest"],"last_modified":item["LastModified"]});
                    ensure!(
                        row["is_latest"].is_boolean() && seen.insert(identity(&row)?),
                        "duplicate or invalid marker/version identity"
                    );
                    if key == "Versions" {
                        row["bytes"] = json!(unsigned(&item["Size"], "listed bytes")?);
                        row["storage_class"] = json!(item["StorageClass"]
                            .as_str()
                            .context("listed storage class")?);
                        row["etag"] = json!(item["ETag"].as_str().context("listed ETag")?);
                        row["sha256"] = Value::Null;
                    }
                    destination.push(row);
                }
            }
            let truncated = response["IsTruncated"]
                .as_bool()
                .context("explicit pagination completion")?;
            if let Some(request) = self.requests.last_mut() {
                request["outcome"] = json!("completed");
                request["response_bytes"] = json!(bytes);
            }
            if !truncated {
                break;
            }
            let key = response["NextKeyMarker"]
                .as_str()
                .filter(|s| !s.is_empty())
                .context("missing or repeated pagination cursor")?;
            let version = response["NextVersionIdMarker"].as_str();
            ensure!(
                response["NextVersionIdMarker"].is_null() || version.is_some(),
                "pagination version cursor"
            );
            ensure!(
                cursors.insert((key.to_owned(), version.map(str::to_owned))),
                "missing or repeated pagination cursor"
            );
            params["KeyMarker"] = json!(key);
            if let Some(version) = version {
                params["VersionIdMarker"] = json!(version);
            } else {
                params
                    .as_object_mut()
                    .context("params")?
                    .remove("VersionIdMarker");
            }
        }
        self.check()?;
        rows.sort_by_key(|r| identity(r).unwrap_or_default());
        markers.sort_by_key(|r| identity(r).unwrap_or_default());
        Ok(
            json!({"rows":rows,"delete_markers":markers,"complete_listing":true,"point_in_time_snapshot":false,"qualified":false}),
        )
    }

    /// Performs the capture operation and returns an error if it cannot be completed.
    pub fn capture(&mut self, transport: &mut impl ReadTransport, row: &Value) -> Result<Vec<u8>> {
        let id = identity(row)?;
        ensure!(
            id[0] == self.bucket && row["storage_class"] == "STANDARD",
            "standard scoped capture"
        );
        let expected = unsigned(&row["bytes"], "capture bytes")?;
        ensure!(
            self.body_bytes
                .checked_add(expected)
                .is_some_and(|n| n <= self.body_cap),
            "body byte admission cap"
        );
        self.send(
            "GetObject",
            &json!({"Bucket":id[0],"Key":id[1],"VersionId":id[2]}),
        )?;
        let mut response =
            transport.get_version(&json!({"Bucket":id[0],"Key":id[1],"VersionId":id[2]}))?;
        // A refused identity drops the body without consuming any byte.
        ensure!(
            response.version_id == id[2] && response.content_length == expected,
            "GET identity/size joins listed immutable version"
        );
        let mut body = Vec::new();
        let mut buffer = [0u8; 64 * 1024];
        loop {
            self.check()?;
            let left = expected.saturating_sub(body.len() as u64).saturating_add(1);
            let max = buffer
                .len()
                .min(usize::try_from(left).unwrap_or(usize::MAX));
            let n = response.body.read(&mut buffer[..max])?;
            if n == 0 {
                break;
            }
            self.body_bytes = self
                .body_bytes
                .checked_add(n as u64)
                .context("body overflow")?;
            ensure!(
                body.len() as u64 + n as u64 <= expected && self.body_bytes <= self.body_cap,
                "body size/read cap"
            );
            body.extend_from_slice(&buffer[..n]);
        }
        self.check()?;
        ensure!(body.len() as u64 == expected, "short object body");
        if let Some(request) = self.requests.last_mut() {
            request["outcome"] = json!("completed");
            request["response_bytes"] = json!(body.len());
        }
        Ok(body)
    }

    /// Performs the receipt operation and returns an error if it cannot be completed.
    pub fn receipt(&self) -> Value {
        json!({"http_requests":self.requests,"physical_request_count":self.requests.len(),
            "metadata_response_bytes":self.metadata_bytes,"object_body_bytes":self.body_bytes,
            "remote_mutation_calls":0,"etag_is_sha256":false,"qualified":false,
            "transport":"injected single-attempt reader; no bundled live adapter"})
    }
}
