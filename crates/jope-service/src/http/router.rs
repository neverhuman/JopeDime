//! Registry-driven local router (spec §3.1, §4). Engine handlers are separate
//! integrations; registered operations currently return an explicit capability error.

use super::{
    auth::{authenticate, problem, BearerToken, LocalAuth},
    system,
};
use crate::engine::DatasetState;
use axum::extract::Extension;
use axum::{
    extract::DefaultBodyLimit,
    http::{header, HeaderValue},
    middleware,
    response::Response,
    routing::{get, on, MethodFilter},
    Router,
};
use jope_api::{
    error::ErrorCode,
    registry::{HttpMethod, OPERATIONS},
};
use std::{io, net::Ipv4Addr};
use tokio::net::TcpListener;

const OPENAPI_DOCUMENT: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../schemas/jope.api.v1.openapi.json"
));

async fn get_openapi() -> Response {
    let mut response = Response::new(axum::body::Body::from(OPENAPI_DOCUMENT));
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    response
}

/// The same request boundary is used by a listener and by in-process clients.
/// In-process clients configure their request authority without opening a socket.
pub fn router(auth: LocalAuth) -> Router {
    build_router(auth)
}

/// Engine-owned store/config state is passed by the live engine instance.
pub(crate) fn router_with_state(auth: LocalAuth, state: DatasetState) -> Router {
    build_router(auth).layer(Extension(state))
}

fn build_router(auth: LocalAuth) -> Router {
    let mut app = Router::new();
    for operation in OPERATIONS {
        let method = match operation.method {
            HttpMethod::Get => MethodFilter::GET,
            HttpMethod::Post => MethodFilter::POST,
            HttpMethod::Delete => MethodFilter::DELETE,
        };
        for path in operation.paths {
            app = if operation.name == "system_info" {
                app.route(path, get(system::get))
            } else {
                app.route(
                    path,
                    on(method, move || async move {
                        problem(
                            ErrorCode::CapabilityUnavailable,
                            &format!(
                                "Operation {} is not implemented in this build.",
                                operation.name
                            ),
                            "Use a build with this operation implemented.",
                            None,
                        )
                    }),
                )
            };
        }
    }
    app = app.route("/v1/openapi.json", get(get_openapi));
    app.fallback(|| async {
        problem(
            ErrorCode::NotFound,
            "No operation matches this path.",
            "Use an API v1 operation path.",
            None,
        )
    })
    .method_not_allowed_fallback(|| async {
        problem(
            ErrorCode::InvalidInput,
            "This method is not supported for this path.",
            "Use the operation's declared HTTP method.",
            None,
        )
    })
    .layer(DefaultBodyLimit::max(8 * 1024 * 1024))
    .layer(middleware::from_fn_with_state(auth, authenticate))
}

/// Bind the only network endpoint allowed by v1. Token generation and daemon
/// publication remain with the engine owner; this function does not spawn tasks.
pub async fn bind_loopback(token: BearerToken) -> io::Result<(TcpListener, Router)> {
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await?;
    let auth = LocalAuth::new(token, listener.local_addr()?.port())
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
    Ok((listener, router(auth)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::{to_bytes, Body},
        http::{header, Method, Request, StatusCode},
    };
    use jope_api::error::Problem;
    use std::collections::BTreeSet;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tower::ServiceExt;

    fn token() -> BearerToken {
        BearerToken::from_hex(&"ab".repeat(32)).unwrap()
    }
    fn request(method: Method, path: &str) -> Request<Body> {
        Request::builder()
            .method(method)
            .uri(path)
            .header(header::HOST, "127.0.0.1:31234")
            .header(header::AUTHORIZATION, format!("Bearer {}", "ab".repeat(32)))
            .body(Body::empty())
            .unwrap()
    }
    fn app() -> Router {
        router(LocalAuth::new(token(), 31234).unwrap())
    }

    #[tokio::test]
    async fn openapi_route_returns_the_approved_document() {
        let response = app()
            .oneshot(request(Method::GET, "/v1/openapi.json"))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()[header::CONTENT_TYPE], "application/json");
        assert_eq!(response.headers()["jope-api-version"], "1.0");
        assert!(!response
            .headers()
            .keys()
            .any(|key| key.as_str().starts_with("access-control-")));

        let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        assert_eq!(bytes.as_ref(), OPENAPI_DOCUMENT);
        let document: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(document.get("x-jope-pending-exchanges").is_none());
        let operation_ids: BTreeSet<_> = document["paths"]
            .as_object()
            .unwrap()
            .values()
            .flat_map(|path| path.as_object().unwrap().values())
            .filter_map(|operation| operation["operationId"].as_str())
            .collect();
        let registered: BTreeSet<_> = OPERATIONS.iter().map(|operation| operation.name).collect();
        assert_eq!(operation_ids, registered);
    }

    #[tokio::test]
    async fn openapi_route_keeps_auth_origin_and_host_rejections() {
        let mut unauthorized = request(Method::GET, "/v1/openapi.json");
        unauthorized.headers_mut().remove(header::AUTHORIZATION);
        let mut browser = request(Method::GET, "/v1/openapi.json");
        browser
            .headers_mut()
            .insert(header::ORIGIN, "https://example.invalid".parse().unwrap());
        let mut hostile_host = request(Method::GET, "/v1/openapi.json");
        hostile_host
            .headers_mut()
            .insert(header::HOST, "attacker.invalid:31234".parse().unwrap());

        for (request, expected) in [
            (unauthorized, StatusCode::UNAUTHORIZED),
            (browser, StatusCode::BAD_REQUEST),
            (hostile_host, StatusCode::BAD_REQUEST),
        ] {
            let response = app().oneshot(request).await.unwrap();
            assert_eq!(response.status(), expected);
            assert_eq!(response.headers()["jope-api-version"], "1.0");
            assert!(!response
                .headers()
                .keys()
                .any(|key| key.as_str().starts_with("access-control-")));
        }
    }

    #[tokio::test]
    async fn all_registry_method_path_pairs_reach_explicit_capability_stubs() {
        let app = app();
        let mut count = 0;
        for operation in OPERATIONS {
            // Consume the API's wire spelling independently of the router's
            // MethodFilter conversion, so swapped method bindings fail here.
            let method: Method = operation.method.as_str().parse().unwrap();
            for path in operation.paths {
                let path = path
                    .replace("{id}", "run_test")
                    .replace("{alg}", "alg_0001");
                let response = app
                    .clone()
                    .oneshot(request(method.clone(), &path))
                    .await
                    .unwrap();
                assert_eq!(response.headers()["jope-api-version"], "1.0");
                assert!(!response
                    .headers()
                    .keys()
                    .any(|k| k.as_str().starts_with("access-control-")));
                if operation.name == "system_info" {
                    assert_eq!(response.status(), StatusCode::OK, "{method} {path}");
                    assert_eq!(response.headers()[header::CONTENT_TYPE], "application/json");
                    let doc: jope_api::SystemInfo = serde_json::from_slice(
                        &to_bytes(response.into_body(), 8192).await.unwrap(),
                    )
                    .unwrap();
                    assert_eq!(doc.version, env!("CARGO_PKG_VERSION"));
                    assert_eq!(doc.api_version, "1.0");
                    assert_eq!(doc.health, "ok");
                    assert!(doc.local_paths);
                    assert!(doc.capabilities.tasks.is_empty());
                    assert!(doc.capabilities.languages.is_empty());
                    assert!(doc.capabilities.devices.is_empty());
                    assert!(!doc.capabilities.learned_guidance);
                    assert!(doc.warnings.is_empty());
                    assert!(doc.telemetry.is_empty());
                } else {
                    assert_eq!(
                        response.status(),
                        StatusCode::UNPROCESSABLE_ENTITY,
                        "{method} {path}"
                    );
                    assert_eq!(
                        response.headers()[header::CONTENT_TYPE],
                        Problem::CONTENT_TYPE
                    );
                    let doc: Problem = serde_json::from_slice(
                        &to_bytes(response.into_body(), 8192).await.unwrap(),
                    )
                    .unwrap();
                    assert_eq!(doc.code, ErrorCode::CapabilityUnavailable);
                    assert_eq!(doc.status, 422);
                    assert!(doc.detail.contains(operation.name));
                }
                count += 1;
            }
        }
        assert_eq!(OPERATIONS.len(), 17);
        assert_eq!(count, 20);
    }

    #[tokio::test]
    async fn fallbacks_have_typed_errors_and_remain_authenticated() {
        let unknown = request(Method::GET, "/v1/unknown");
        let response = app().oneshot(unknown).await.unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        assert_eq!(
            response.headers()[header::CONTENT_TYPE],
            Problem::CONTENT_TYPE
        );
        let response = app()
            .oneshot(request(Method::PATCH, "/v1/system"))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            response.headers()[header::CONTENT_TYPE],
            Problem::CONTENT_TYPE
        );
        for path in ["/v1/system", "/v1/unknown"] {
            let mut req = request(Method::GET, path);
            req.headers_mut().remove(header::AUTHORIZATION);
            assert_eq!(
                app().oneshot(req).await.unwrap().status(),
                StatusCode::UNAUTHORIZED
            );
        }
    }

    #[tokio::test]
    async fn listener_uses_ephemeral_ipv4_loopback_and_the_same_security_boundary() {
        let (listener, app) = bind_loopback(token()).await.unwrap();
        let address = listener.local_addr().unwrap();
        assert_eq!(address.ip(), Ipv4Addr::LOCALHOST);
        assert_ne!(address.port(), 0);
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let exercise = async {
            for authorized in [false, true] {
                let mut socket = tokio::net::TcpStream::connect(address).await.unwrap();
                let auth = if authorized {
                    format!("Authorization: Bearer {}\r\n", "ab".repeat(32))
                } else {
                    String::new()
                };
                let request = format!("GET /v1/system HTTP/1.1\r\nHost: localhost:{}\r\n{auth}Connection: close\r\n\r\n", address.port());
                socket.write_all(request.as_bytes()).await.unwrap();
                let mut response = Vec::new();
                socket.read_to_end(&mut response).await.unwrap();
                let response = String::from_utf8(response).unwrap();
                if authorized {
                    assert!(response.starts_with("HTTP/1.1 200"));
                    assert!(response
                        .to_lowercase()
                        .contains("content-type: application/json"));
                    let body = response.split_once("\r\n\r\n").unwrap().1;
                    let doc: jope_api::SystemInfo = serde_json::from_str(body).unwrap();
                    assert_eq!(doc.version, env!("CARGO_PKG_VERSION"));
                    assert_eq!(doc.api_version, "1.0");
                    assert_eq!(doc.health, "ok");
                    assert!(doc.local_paths);
                    assert!(doc.warnings.is_empty());
                    assert!(doc.telemetry.is_empty());
                } else {
                    assert!(response.starts_with("HTTP/1.1 401"));
                    assert!(response
                        .to_lowercase()
                        .contains("content-type: application/problem+json"));
                }
                assert!(response.to_lowercase().contains("jope-api-version: 1.0"));
                assert!(!response.to_lowercase().contains("access-control-"));
            }
        };
        let result = tokio::time::timeout(std::time::Duration::from_secs(5), exercise).await;
        server.abort();
        let _ = server.await;
        result.unwrap();
    }
}
