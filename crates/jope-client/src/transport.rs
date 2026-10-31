//! Shared registry-driven request boundary. No engine or executor dependency.
use jope_api::{
    error::Problem,
    ids::*,
    registry::{
        AlgorithmPage, DatasetPage, DatasetRegistration, DeleteReceipt, EventPage, GcReport,
        GcRequest, HttpMethod, ObjectiveValidation, ObjectiveValidationRequest, RunControlRequest,
        RunForkRequest, RunPage, TrainingFeedback, OPERATIONS,
    },
    Algorithm, DatasetProfile, ExportManifest, ExportRequest, Plan, PredictRequest, RunSnapshot,
    RunSpec, SystemInfo,
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{
    fmt,
    io::{self, Read},
};
pub use ureq::http::{header, HeaderMap, HeaderValue};

pub type Result<T> = std::result::Result<T, Error>;
pub const MAX_JSON_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug)]
pub enum Error {
    InvalidRequest(&'static str),
    Protocol(&'static str),
    Io(io::Error),
    Json(serde_json::Error),
    Http(String),
    Problem(Box<Problem>),
    Status(u16),
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRequest(s) | Self::Protocol(s) => f.write_str(s),
            Self::Io(e) => e.fmt(f),
            Self::Json(e) => e.fmt(f),
            Self::Http(e) => f.write_str(e),
            Self::Problem(p) => write!(f, "{}: {} ({})", p.title, p.detail, p.remedy),
            Self::Status(s) => write!(f, "HTTP status {s}"),
        }
    }
}
impl std::error::Error for Error {}
impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}
impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}

/// Bodies are bounded JSON here; file-upload streaming is the CLT-03 integration.
#[derive(Clone, Debug)]
pub struct Request {
    pub(crate) method: HttpMethod,
    pub(crate) path: String,
    pub headers: HeaderMap,
    pub(crate) body: Vec<u8>,
    requires_key: bool,
}
impl Request {
    pub fn method(&self) -> HttpMethod {
        self.method
    }
    pub fn target(&self) -> &str {
        &self.path
    }
    pub fn body(&self) -> &[u8] {
        &self.body
    }
    pub fn new(operation: u8, route: usize, parameters: &[(&str, &str)]) -> Result<Self> {
        let op = OPERATIONS
            .iter()
            .find(|o| o.id == operation)
            .ok_or(Error::InvalidRequest("unknown registry operation"))?;
        let mut path = op
            .paths
            .get(route)
            .ok_or(Error::InvalidRequest("unknown registry route"))?
            .to_string();
        for (name, value) in parameters {
            let placeholder = format!("{{{name}}}");
            if !path.contains(&placeholder) {
                return Err(Error::InvalidRequest("unexpected path parameter"));
            }
            path = path.replace(&placeholder, &encode(value));
        }
        if path.contains(['{', '}']) {
            return Err(Error::InvalidRequest("missing path parameter"));
        }
        Ok(Self {
            method: op.method,
            path,
            headers: HeaderMap::new(),
            body: Vec::new(),
            requires_key: op.requires_idempotency_key,
        })
    }
    pub fn query(mut self, values: &[(&str, &str)]) -> Self {
        for (name, value) in values {
            self.path
                .push(if self.path.contains('?') { '&' } else { '?' });
            self.path.push_str(&encode(name));
            self.path.push('=');
            self.path.push_str(&encode(value));
        }
        self
    }
    pub fn key(mut self, value: Option<&str>) -> Result<Self> {
        if let Some(value) = value {
            if value.is_empty()
                || value.len() > 128
                || !value.bytes().all(|b| (32..=126).contains(&b))
            {
                return Err(Error::InvalidRequest(
                    "Idempotency-Key must be 1-128 printable ASCII bytes",
                ));
            }
            self.headers.insert(
                "idempotency-key",
                HeaderValue::from_str(value)
                    .map_err(|_| Error::InvalidRequest("invalid Idempotency-Key"))?,
            );
        }
        Ok(self)
    }
    pub fn json<T: Serialize + ?Sized>(mut self, value: &T) -> Result<Self> {
        self.body = serde_json::to_vec(value)?;
        if self.body.len() > MAX_JSON_BYTES {
            return Err(Error::InvalidRequest("JSON request exceeds 8 MiB"));
        }
        self.headers.insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );
        Ok(self)
    }
    pub fn if_match(mut self, revision: jope_api::decimal::DecimalU64) -> Self {
        self.headers.insert(
            header::IF_MATCH,
            HeaderValue::from_str(&format!("\"{}\"", revision.0)).unwrap(),
        );
        self
    }
    pub(crate) fn validate(&self) -> Result<()> {
        if !self.path.starts_with("/v1/")
            || self.path.contains('#')
            || self.path.contains('\r')
            || self.path.contains('\n')
        {
            return Err(Error::InvalidRequest("invalid API request target"));
        }
        if self.headers.contains_key(header::HOST)
            || self.headers.contains_key(header::AUTHORIZATION)
        {
            return Err(Error::InvalidRequest(
                "transport owns Host and Authorization",
            ));
        }
        let key = self
            .headers
            .get("idempotency-key")
            .map(|v| v.to_str())
            .transpose()
            .map_err(|_| Error::InvalidRequest("invalid Idempotency-Key"))?;
        if self.headers.get_all("idempotency-key").iter().count() > 1 {
            return Err(Error::InvalidRequest("duplicate Idempotency-Key"));
        }
        if self.requires_key && key.is_none() {
            return Err(Error::InvalidRequest("Idempotency-Key is required"));
        }
        if let Some(key) = key {
            if key.is_empty() || key.len() > 128 || !key.bytes().all(|b| (32..=126).contains(&b)) {
                return Err(Error::InvalidRequest("invalid Idempotency-Key"));
            }
        }
        Ok(())
    }
}
pub(crate) fn encode(value: &str) -> String {
    let mut out = String::new();
    for b in value.bytes() {
        if b.is_ascii_alphanumeric() || b"-._~".contains(&b) {
            out.push(char::from(b));
        } else {
            use fmt::Write;
            write!(out, "%{b:02X}").unwrap();
        }
    }
    out
}

pub struct Response {
    pub status: u16,
    pub headers: HeaderMap,
    pub body: Box<dyn Read + Send>,
}
impl Response {
    pub fn new(status: u16, headers: HeaderMap, body: impl Read + Send + 'static) -> Self {
        Self {
            status,
            headers,
            body: Box::new(body),
        }
    }
    fn media_type(&self) -> &str {
        self.headers
            .get(header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .split(';')
            .next()
            .unwrap_or("")
            .trim()
    }
    pub fn checked(mut self, media: &str) -> Result<Self> {
        let version = self
            .headers
            .get("jope-api-version")
            .and_then(|v| v.to_str().ok())
            .ok_or(Error::Protocol("missing Jope-Api-Version"))?;
        if !version
            .strip_prefix("1.")
            .is_some_and(|v| !v.is_empty() && v.bytes().all(|b| b.is_ascii_digit()))
        {
            return Err(Error::Protocol("unsupported API version"));
        }
        if !(200..300).contains(&self.status) {
            if self
                .media_type()
                .eq_ignore_ascii_case(Problem::CONTENT_TYPE)
            {
                let problem: Problem = serde_json::from_slice(&read_json(&mut self.body)?)?;
                if problem.status != self.status {
                    return Err(Error::Protocol("Problem status disagrees with HTTP status"));
                }
                return Err(Error::Problem(Box::new(problem)));
            }
            return Err(Error::Status(self.status));
        }
        if !self.media_type().eq_ignore_ascii_case(media) {
            return Err(Error::Protocol("unexpected response Content-Type"));
        }
        Ok(self)
    }
    pub fn json<T: DeserializeOwned>(self) -> Result<T> {
        let mut response = self.checked("application/json")?;
        Ok(serde_json::from_slice(&read_json(&mut response.body)?)?)
    }
}
fn read_json(reader: &mut dyn Read) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    reader
        .take((MAX_JSON_BYTES + 1) as u64)
        .read_to_end(&mut bytes)?;
    if bytes.len() > MAX_JSON_BYTES {
        return Err(Error::Protocol("JSON response exceeds 8 MiB"));
    }
    Ok(bytes)
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum RunResult {
    Plan(Box<Plan>),
    Started(Box<RunSnapshot>),
}

// Method bodies are generated by these macros; method/path/key facts come from
// OPERATIONS at request construction; exchange types are the canonical API DTOs.
macro_rules! get_method {
    ($name:ident, $op:literal, $route:literal, $out:ty $(, $arg:ident:$ty:ty)*) => {
        fn $name(&self, $($arg: &$ty),*) -> Result<$out> {
            let request = Request::new($op, $route, &[$((stringify!($arg), &$arg.to_string())),*])?;
            self.execute(request)?.json()
        }
    };
}
macro_rules! post_method {
    ($name:ident, $op:literal, $body:ty, $out:ty $(, $arg:ident:$ty:ty)*) => {
        fn $name(&self, $($arg: &$ty,)* body: &$body, key: Option<&str>) -> Result<$out> {
            let request = Request::new($op, 0, &[$((stringify!($arg), &$arg.to_string())),*])?.key(key)?.json(body)?;
            self.execute(request)?.json()
        }
    };
}

pub trait Transport: Send + Sync {
    fn execute(&self, request: Request) -> Result<Response>;
    /// Stream a CSV upload through the canonical dataset registration route.
    /// Transports without streaming support keep compiling and fail explicitly.
    fn dataset_register_stream(
        &self,
        _name: &str,
        _body: Box<dyn Read + Send>,
        _idempotency_key: &str,
    ) -> Result<Response> {
        Err(Error::InvalidRequest(
            "streaming dataset registration is unsupported by this transport",
        ))
    }
    get_method!(system_info, 1, 0, SystemInfo);
    post_method!(dataset_register, 2, DatasetRegistration, DatasetProfile);
    get_method!(datasets, 3, 0, DatasetPage);
    get_method!(dataset_inspect, 3, 1, DatasetProfile, id:DatasetId);
    post_method!(
        objective_validate,
        5,
        ObjectiveValidationRequest,
        ObjectiveValidation
    );
    post_method!(run_start, 6, RunSpec, RunResult);
    get_method!(runs, 7, 0, RunPage);
    get_method!(run_status, 7, 1, RunSnapshot, id:RunId);
    post_method!(run_control, 8, RunControlRequest, RunSnapshot, id:RunId);
    post_method!(run_fork, 9, RunForkRequest, RunSnapshot, id:RunId);
    get_method!(algorithms, 12, 0, AlgorithmPage, id:RunId);
    get_method!(algorithm_get, 12, 1, Algorithm, id:RunId, alg:AlgorithmId);
    post_method!(model_export, 13, ExportRequest, ExportManifest, id:RunId);
    get_method!(feedback_export, 16, 0, TrainingFeedback, id:RunId);
    post_method!(store_gc, 17, GcRequest, GcReport);

    fn dataset_delete(&self, id: &DatasetId, key: &str) -> Result<DeleteReceipt> {
        self.execute(Request::new(4, 0, &[("id", &id.to_string())])?.key(Some(key))?)?
            .json()
    }
    fn run_delete(&self, id: &RunId, key: &str) -> Result<DeleteReceipt> {
        self.execute(Request::new(10, 0, &[("id", &id.to_string())])?.key(Some(key))?)?
            .json()
    }
    fn run_events(
        &self,
        id: &RunId,
        after: jope_api::decimal::DecimalU64,
        limit: u16,
    ) -> Result<EventPage> {
        if limit == 0 || limit > 256 {
            return Err(Error::InvalidRequest("event page limit must be 1-256"));
        }
        let request = Request::new(11, 0, &[("id", &id.to_string())])?.query(&[
            ("after", &after.0.to_string()),
            ("limit", &limit.to_string()),
        ]);
        self.execute(request)?.json()
    }
    fn artifact_download(&self, id: &ArtifactId) -> Result<Response> {
        self.execute(Request::new(14, 0, &[("id", &id.to_string())])?)?
            .checked("application/octet-stream")
    }
    fn model_predict(&self, body: &PredictRequest, key: &str) -> Result<Response> {
        self.execute(Request::new(15, 0, &[])?.key(Some(key))?.json(body)?)?
            .checked("text/csv")
    }
    fn follow_events(
        &self,
        run: RunId,
        after: Option<jope_api::decimal::DecimalU64>,
    ) -> Result<crate::sse::EventStream<'_, Self>>
    where
        Self: Sized,
    {
        crate::sse::EventStream::new(self, run, after)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    struct GoldenTransport<'a> {
        request: Option<&'a Value>,
        response: &'a Value,
    }
    impl Transport for GoldenTransport<'_> {
        fn execute(&self, request: Request) -> Result<Response> {
            request.validate()?;
            if let Some(expected) = self.request {
                assert_eq!(serde_json::from_slice::<Value>(request.body())?, *expected);
            } else {
                assert!(request.body().is_empty());
            }
            let mut headers = HeaderMap::new();
            headers.insert("jope-api-version", HeaderValue::from_static("1.0"));
            headers.insert(
                header::CONTENT_TYPE,
                HeaderValue::from_static("application/json"),
            );
            Ok(Response::new(
                200,
                headers,
                io::Cursor::new(serde_json::to_vec(self.response)?),
            ))
        }
    }

    fn exchange_goldens() -> Value {
        serde_json::from_str(include_str!(
            "../../jope-api/tests/snapshots/exchanges.json"
        ))
        .unwrap()
    }

    #[test]
    fn canonical_request_dtos_preserve_api_owner_golden_wire_values() {
        let goldens = exchange_goldens();
        let positive = &goldens["positive"];
        let profile = &positive["DatasetPage"][0]["datasets"][0];
        let snapshot = &positive["RunPage"][0]["runs"][0];
        let run: RunId = "run_01ARZ3NDEKTSV4RRFFQ69G5FAV".parse().unwrap();
        macro_rules! request_cases {
            ($input:ty, $output:ty, $reply:expr, $method:ident $(, $arg:expr)*) => {
                for value in positive[stringify!($input)].as_array().unwrap() {
                    let body: $input = serde_json::from_value(value.clone()).unwrap();
                    let transport = GoldenTransport { request: Some(value), response: $reply };
                    let result: $output = transport.$method($($arg,)* &body, Some("golden-retry")).unwrap();
                    assert_eq!(serde_json::to_value(result).unwrap(), *$reply);
                }
            };
        }
        request_cases!(
            DatasetRegistration,
            DatasetProfile,
            profile,
            dataset_register
        );
        request_cases!(
            ObjectiveValidationRequest,
            ObjectiveValidation,
            &positive["ObjectiveValidation"][0],
            objective_validate
        );
        request_cases!(RunControlRequest, RunSnapshot, snapshot, run_control, &run);
        request_cases!(RunForkRequest, RunSnapshot, snapshot, run_fork, &run);
        request_cases!(GcRequest, GcReport, &positive["GcReport"][0], store_gc);
    }

    #[test]
    fn canonical_response_dtos_round_trip_and_reject_api_owner_negative_goldens() {
        let goldens = exchange_goldens();
        let run: RunId = "run_01ARZ3NDEKTSV4RRFFQ69G5FAV".parse().unwrap();
        let dataset: DatasetId = "ds_00112233445566778899aabb".parse().unwrap();
        macro_rules! response_cases {
            ($output:ty, $method:ident $(, $arg:expr)*) => {
                for value in goldens["positive"][stringify!($output)].as_array().unwrap() {
                    let transport = GoldenTransport { request: None, response: value };
                    let result: $output = transport.$method($($arg),*).unwrap();
                    assert_eq!(serde_json::to_value(result).unwrap(), *value);
                }
                for value in goldens["negative"][stringify!($output)].as_array().unwrap() {
                    let transport = GoldenTransport { request: None, response: value };
                    assert!(matches!(transport.$method($($arg),*), Err(Error::Json(_))));
                }
            };
        }
        response_cases!(DatasetPage, datasets);
        response_cases!(RunPage, runs);
        response_cases!(AlgorithmPage, algorithms, &run);
        response_cases!(DeleteReceipt, dataset_delete, &dataset, "delete-dataset");
        response_cases!(DeleteReceipt, run_delete, &run, "delete-run");
        response_cases!(
            EventPage,
            run_events,
            &run,
            jope_api::decimal::DecimalU64(0),
            256
        );
        response_cases!(TrainingFeedback, feedback_export, &run);

        // POST responses have request bodies as well as the newly typed results.
        let validation: ObjectiveValidationRequest =
            serde_json::from_value(goldens["positive"]["ObjectiveValidationRequest"][0].clone())
                .unwrap();
        for name in ["positive", "negative"] {
            for value in goldens[name]["ObjectiveValidation"].as_array().unwrap() {
                let transport = GoldenTransport {
                    request: Some(&goldens["positive"]["ObjectiveValidationRequest"][0]),
                    response: value,
                };
                let result: Result<ObjectiveValidation> =
                    transport.objective_validate(&validation, None);
                if name == "positive" {
                    assert_eq!(serde_json::to_value(result.unwrap()).unwrap(), *value);
                } else {
                    assert!(matches!(result, Err(Error::Json(_))));
                }
            }
        }
        let gc: GcRequest =
            serde_json::from_value(goldens["positive"]["GcRequest"][0].clone()).unwrap();
        for value in goldens["negative"]["GcReport"].as_array().unwrap() {
            let transport = GoldenTransport {
                request: Some(&goldens["positive"]["GcRequest"][0]),
                response: value,
            };
            assert!(matches!(
                transport.store_gc(&gc, Some("gc-retry")),
                Err(Error::Json(_))
            ));
        }
    }

    #[test]
    fn registry_routes_and_headers_are_validated_before_dispatch() {
        for op in OPERATIONS {
            for (route, template) in op.paths.iter().enumerate() {
                let mut params = vec![];
                if template.contains("{id}") {
                    params.push(("id", "a/b ü"));
                }
                if template.contains("{alg}") {
                    params.push(("alg", "alg_0001"));
                }
                let request = Request::new(op.id, route, &params).unwrap();
                assert_eq!(request.method, op.method);
                assert!(!request.path.contains('{'));
                assert!(!request.path.contains("a/b"));
                assert_eq!(request.validate().is_err(), op.requires_idempotency_key);
                request.key(Some("cli_test")).unwrap().validate().unwrap();
            }
        }
        assert_eq!(encode("a/b ü"), "a%2Fb%20%C3%BC");
        assert_eq!(
            OPERATIONS.len(),
            17,
            "new operations need typed client bindings"
        );
        assert_eq!(OPERATIONS.iter().map(|o| o.paths.len()).sum::<usize>(), 20);
        assert!(Request::new(7, 1, &[]).is_err());
        assert!(Request::new(1, 0, &[("id", "x")]).is_err());
        for key in ["", "line\nbreak", "\u{7f}", &"x".repeat(129)] {
            assert!(Request::new(6, 0, &[]).unwrap().key(Some(key)).is_err());
        }
        let mut duplicate = Request::new(6, 0, &[]).unwrap().key(Some("one")).unwrap();
        duplicate
            .headers
            .append("idempotency-key", HeaderValue::from_static("two"));
        assert!(duplicate.validate().is_err());
    }
    #[test]
    fn response_requires_media_version_and_matching_problem_status() {
        let make = |status, media, bytes: &[u8]| {
            let mut headers = HeaderMap::new();
            headers.insert("jope-api-version", HeaderValue::from_static("1.0"));
            headers.insert(header::CONTENT_TYPE, HeaderValue::from_static(media));
            Response::new(status, headers, io::Cursor::new(bytes.to_vec()))
        };
        assert_eq!(
            make(200, "application/json", b"{\"ok\":true}")
                .json::<Value>()
                .unwrap()["ok"],
            true
        );
        assert!(make(200, "text/plain", b"{}").json::<Value>().is_err());
        let mut response = make(200, "application/json", b"{}");
        response.headers.remove("jope-api-version");
        assert!(response.json::<Value>().is_err());
        assert!(matches!(
            make(503, "text/plain", b"busy").json::<Value>(),
            Err(Error::Status(503))
        ));
        let problem = Problem {
            r#type: "about:blank".into(),
            title: "Invalid input".into(),
            status: 400,
            detail: "bad field".into(),
            instance: None,
            code: jope_api::error::ErrorCode::InvalidInput,
            retriable: false,
            remedy: "supply a valid field".into(),
            field: Some("field".into()),
        };
        let bytes = serde_json::to_vec(&problem).unwrap();
        assert!(
            matches!(make(400,Problem::CONTENT_TYPE,&bytes).json::<Value>(),
            Err(Error::Problem(p)) if *p==problem)
        );
        assert!(matches!(
            make(409, Problem::CONTENT_TYPE, &bytes).json::<Value>(),
            Err(Error::Protocol(_))
        ));
        assert!(make(200, "Application/JSON", b"{}").json::<Value>().is_ok());
    }
}
