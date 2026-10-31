//! Local engine request boundary (spec §4.6). Token generation and daemon-file
//! permissions belong to the daemon owner; this module never displays a token.

use axum::{
    extract::{Request, State},
    http::{header, HeaderMap, HeaderName, HeaderValue, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use jope_api::error::{ErrorCode, Problem};
use std::{fmt, sync::Arc};
use subtle::ConstantTimeEq;

/// The daemon's 32 random bytes encoded as 64 hex characters, without
/// normalization. Generation remains the caller's responsibility.
#[derive(Clone)]
pub struct BearerToken(Arc<[u8; 64]>);

/// Stable identity for the single authenticated local-engine authority.
/// It deliberately does not contain or derive from the bearer credential.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AuthenticatedPrincipal(Arc<str>);

impl AuthenticatedPrincipal {
    fn local_engine() -> Self {
        Self(Arc::from("local-engine"))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl BearerToken {
    pub fn from_hex(value: &str) -> Result<Self, InvalidLocalAuth> {
        if value.len() != 64 || !value.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(InvalidLocalAuth::Token);
        }
        let mut bytes = [0; 64];
        bytes.copy_from_slice(value.as_bytes());
        Ok(Self(Arc::new(bytes)))
    }
}

impl fmt::Debug for BearerToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("BearerToken([REDACTED])")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InvalidLocalAuth {
    Token,
    Port,
}

impl fmt::Display for InvalidLocalAuth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Token => "bearer token must contain 64 hex characters",
            Self::Port => "local engine authority requires a nonzero port",
        })
    }
}

impl std::error::Error for InvalidLocalAuth {}

/// Authority context for both HTTP and in-process requests. In-process clients
/// choose a local authority for their requests without binding any socket.
#[derive(Clone, Debug)]
pub struct LocalAuth {
    token: BearerToken,
    port: u16,
}

impl LocalAuth {
    pub fn new(token: BearerToken, port: u16) -> Result<Self, InvalidLocalAuth> {
        if port == 0 {
            return Err(InvalidLocalAuth::Port);
        }
        Ok(Self { token, port })
    }

    fn allows_authority(&self, value: &str) -> bool {
        value == format!("127.0.0.1:{}", self.port)
            || value.eq_ignore_ascii_case(&format!("localhost:{}", self.port))
    }

    fn authenticates(&self, value: &HeaderValue) -> bool {
        let Ok(value) = value.to_str() else {
            return false;
        };
        let Some((scheme, credential)) = value.split_once(' ') else {
            return false;
        };
        scheme.eq_ignore_ascii_case("Bearer")
            && bool::from(credential.as_bytes().ct_eq(&self.token.0[..]))
    }
}

fn single_header(headers: &HeaderMap, name: HeaderName) -> Option<&HeaderValue> {
    let mut values = headers.get_all(name).iter();
    let value = values.next()?;
    values.next().is_none().then_some(value)
}

pub(super) async fn authenticate(
    State(auth): State<LocalAuth>,
    request: Request,
    next: Next,
) -> Response {
    let mut response = if request.headers().contains_key(header::ORIGIN) {
        problem(
            ErrorCode::InvalidInput,
            "Browser-origin requests are not accepted by the local engine.",
            "Use a JopeDime client without an Origin header.",
            Some("Origin"),
        )
    } else if !single_header(request.headers(), header::HOST)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| auth.allows_authority(v))
        || request
            .uri()
            .authority()
            .is_some_and(|v| !auth.allows_authority(v.as_str()))
    {
        problem(
            ErrorCode::InvalidInput,
            "Host must match the local engine address and port.",
            "Use the local engine URL from daemon.json.",
            Some("Host"),
        )
    } else if !single_header(request.headers(), header::AUTHORIZATION)
        .is_some_and(|v| auth.authenticates(v))
    {
        problem(
            ErrorCode::Unauthorized,
            "A valid engine bearer token is required.",
            "Supply the token from the engine's owner-only daemon.json.",
            None,
        )
    } else {
        let mut request = request;
        request
            .extensions_mut()
            .insert(AuthenticatedPrincipal::local_engine());
        next.run(request).await
    };
    response
        .headers_mut()
        .insert("jope-api-version", HeaderValue::from_static("1.0"));
    // The local engine never exposes a browser CORS surface, including when a
    // downstream handler happens to supply such a header.
    let cors: Vec<_> = response
        .headers()
        .keys()
        .filter(|name| name.as_str().starts_with("access-control-"))
        .cloned()
        .collect();
    for name in cors {
        response.headers_mut().remove(name);
    }
    response
}

pub(super) fn problem(
    code: ErrorCode,
    detail: &str,
    remedy: &str,
    field: Option<&str>,
) -> Response {
    let status = StatusCode::from_u16(code.http_status().expect("HTTP catalog code"))
        .expect("valid catalog status");
    let document = Problem {
        r#type: "about:blank".into(),
        title: status.canonical_reason().unwrap_or("Request failed").into(),
        status: status.as_u16(),
        detail: detail.into(),
        instance: None,
        code,
        retriable: false,
        remedy: remedy.into(),
        field: field.map(str::to_owned),
    };
    let mut response = (status, Json(document)).into_response();
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static(Problem::CONTENT_TYPE),
    );
    if code == ErrorCode::Unauthorized {
        response
            .headers_mut()
            .insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
    }
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::{to_bytes, Body},
        http::Method,
        middleware,
        routing::get,
        Router,
    };
    use tower::ServiceExt;

    const PORT: u16 = 31234;

    fn app() -> Router {
        let auth = LocalAuth::new(BearerToken::from_hex(&"ab".repeat(32)).unwrap(), PORT).unwrap();
        Router::new()
            .route(
                "/probe",
                get(|| async { ([("access-control-allow-origin", "*")], "ok") }),
            )
            .layer(middleware::from_fn_with_state(auth, authenticate))
    }

    fn request(method: Method, host: &str, token: Option<&str>) -> Request {
        let mut builder = Request::builder()
            .method(method)
            .uri("/probe")
            .header(header::HOST, host);
        if let Some(token) = token {
            builder = builder.header(header::AUTHORIZATION, token);
        }
        builder.body(Body::empty()).unwrap()
    }

    async fn check(response: Response, status: StatusCode, code: ErrorCode) {
        assert_eq!(response.status(), status);
        assert_eq!(
            response.headers()[header::CONTENT_TYPE],
            Problem::CONTENT_TYPE
        );
        assert_eq!(response.headers()["jope-api-version"], "1.0");
        assert!(!response
            .headers()
            .keys()
            .any(|k| k.as_str().starts_with("access-control-")));
        let bytes = to_bytes(response.into_body(), 8192).await.unwrap();
        let document: Problem = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(document.code, code);
        assert_eq!(document.status, status.as_u16());
        assert!(!document.retriable);
        assert!(!String::from_utf8(bytes.to_vec())
            .unwrap()
            .contains(&"ab".repeat(32)));
    }

    #[test]
    fn secret_and_authority_configuration_are_validated_and_redacted() {
        for bad in [
            String::new(),
            "a".repeat(63),
            "a".repeat(65),
            "z".repeat(64),
        ] {
            assert_eq!(
                BearerToken::from_hex(&bad).unwrap_err(),
                InvalidLocalAuth::Token
            );
        }
        let secret = "ab".repeat(32);
        let token = BearerToken::from_hex(&secret).unwrap();
        assert!(!format!("{token:?}").contains(&secret));
        assert_eq!(
            LocalAuth::new(token, 0).unwrap_err(),
            InvalidLocalAuth::Port
        );
    }

    #[tokio::test]
    async fn missing_wrong_and_malformed_tokens_are_unauthorized() {
        let good = format!("Bearer {}", "ab".repeat(32));
        let wrong = format!("Bearer {}", "cd".repeat(32));
        for value in [
            None,
            Some(wrong.as_str()),
            Some("Bearer"),
            Some("Basic x"),
            Some("Bearer "),
            Some(good.trim_end_matches('b')),
        ] {
            check(
                app()
                    .oneshot(request(Method::GET, "localhost:31234", value))
                    .await
                    .unwrap(),
                StatusCode::UNAUTHORIZED,
                ErrorCode::Unauthorized,
            )
            .await;
        }
        let mut duplicate = request(Method::GET, "localhost:31234", Some(&good));
        duplicate
            .headers_mut()
            .append(header::AUTHORIZATION, HeaderValue::from_str(&good).unwrap());
        check(
            app().oneshot(duplicate).await.unwrap(),
            StatusCode::UNAUTHORIZED,
            ErrorCode::Unauthorized,
        )
        .await;
        let mut query_only = request(Method::GET, "localhost:31234", None);
        *query_only.uri_mut() = format!("/probe?token={}", "ab".repeat(32)).parse().unwrap();
        check(
            app().oneshot(query_only).await.unwrap(),
            StatusCode::UNAUTHORIZED,
            ErrorCode::Unauthorized,
        )
        .await;
    }

    #[tokio::test]
    async fn exact_local_authorities_and_valid_bearer_reach_a_successful_handler() {
        for host in ["localhost:31234", "LOCALHOST:31234", "127.0.0.1:31234"] {
            let token = format!("bEaReR {}", "ab".repeat(32));
            let response = app()
                .oneshot(request(Method::GET, host, Some(&token)))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            assert_eq!(response.headers()["jope-api-version"], "1.0");
            assert!(!response
                .headers()
                .contains_key("access-control-allow-origin"));
            assert_eq!(to_bytes(response.into_body(), 16).await.unwrap(), "ok");
        }
    }

    #[tokio::test]
    async fn aliases_wrong_ports_and_duplicate_hosts_are_rejected() {
        let token = format!("Bearer {}", "ab".repeat(32));
        for host in [
            "evil.example",
            "evil.example:31234",
            "localhost.evil.example:31234",
            "localhost.:31234",
            "127.0.0.2:31234",
            "127.1:31234",
            "[::1]:31234",
            "2130706433:31234",
            "127.0.0.1",
            "localhost:1",
            "localhost:031234",
            "user@localhost:31234",
        ] {
            check(
                app()
                    .oneshot(request(Method::GET, host, Some(&token)))
                    .await
                    .unwrap(),
                StatusCode::BAD_REQUEST,
                ErrorCode::InvalidInput,
            )
            .await;
        }
        let mut req = request(Method::GET, "localhost:31234", Some(&token));
        req.headers_mut()
            .append(header::HOST, HeaderValue::from_static("localhost:31234"));
        check(
            app().oneshot(req).await.unwrap(),
            StatusCode::BAD_REQUEST,
            ErrorCode::InvalidInput,
        )
        .await;
        let mut req = request(Method::GET, "localhost:31234", Some(&token));
        *req.uri_mut() = "http://evil.example:31234/probe".parse().unwrap();
        check(
            app().oneshot(req).await.unwrap(),
            StatusCode::BAD_REQUEST,
            ErrorCode::InvalidInput,
        )
        .await;
        let mut req = request(Method::GET, "localhost:31234", Some(&token));
        req.headers_mut().remove(header::HOST);
        check(
            app().oneshot(req).await.unwrap(),
            StatusCode::BAD_REQUEST,
            ErrorCode::InvalidInput,
        )
        .await;
    }

    #[tokio::test]
    async fn browser_methods_and_preflights_cannot_bypass_origin_policy() {
        let token = format!("Bearer {}", "ab".repeat(32));
        for method in [Method::GET, Method::HEAD, Method::POST, Method::OPTIONS] {
            for origin in ["https://evil.example", "null", "http://localhost:31234"] {
                let mut req = request(method.clone(), "localhost:31234", Some(&token));
                req.headers_mut()
                    .insert(header::ORIGIN, HeaderValue::from_str(origin).unwrap());
                let response = app().oneshot(req).await.unwrap();
                if method == Method::HEAD {
                    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
                    assert_eq!(
                        response.headers()[header::CONTENT_TYPE],
                        Problem::CONTENT_TYPE
                    );
                    assert_eq!(response.headers()["jope-api-version"], "1.0");
                    assert!(to_bytes(response.into_body(), 8192)
                        .await
                        .unwrap()
                        .is_empty());
                } else {
                    check(response, StatusCode::BAD_REQUEST, ErrorCode::InvalidInput).await;
                }
            }
        }
    }
}
