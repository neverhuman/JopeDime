use super::*;

fn local_router() -> axum::Router {
    let token = BearerToken::from_hex(&"ab".repeat(32)).unwrap();
    router::router(LocalAuth::new(token, 31234).unwrap())
}

fn authenticated_request(method: Method, uri: &str, body: String) -> Request<Body> {
    Request::builder()
        .method(method)
        .uri(uri)
        .header(header::HOST, "127.0.0.1:31234")
        .header(header::AUTHORIZATION, format!("Bearer {}", "ab".repeat(32)))
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body))
        .unwrap()
}

#[tokio::test]
async fn live_system_info_http_response_does_not_echo_private_request_metadata() {
    let query = format!(
        "dsl={}&operator={}&lineage={}&round={}",
        SENTINELS[0], SENTINELS[1], SENTINELS[2], SENTINELS[3]
    );
    let request = Request::builder()
        .method(Method::GET)
        .uri(format!("/v1/system?{query}"))
        .header(header::HOST, "127.0.0.1:31234")
        .header(header::AUTHORIZATION, format!("Bearer {}", "ab".repeat(32)))
        .header("x-internal-dsl", SENTINELS[0])
        .header("x-internal-operator", SENTINELS[1])
        .header("x-internal-lineage", SENTINELS[2])
        .header("x-internal-round", SENTINELS[3])
        .body(Body::empty())
        .unwrap();
    let response = local_router().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
    let info: jope_api::SystemInfo = serde_json::from_slice(&body).unwrap();
    assert_eq!(info.health, "ok");
    scan_surfaces(&[(Surface::Http, String::from_utf8(body.to_vec()).unwrap())]).unwrap();
}

#[tokio::test]
async fn current_dataset_route_refusals_do_not_echo_private_payloads() {
    for operation in jope_api::registry::OPERATIONS
        .iter()
        .filter(|operation| operation.name.starts_with("dataset_"))
    {
        for path in operation.paths {
            let path = path.replace("{id}", "ds_00112233445566778899aabb");
            let query = format!(
                "?dsl={}&operator={}&lineage={}&round={}",
                SENTINELS[0], SENTINELS[1], SENTINELS[2], SENTINELS[3]
            );
            let body = serde_json::json!({
                "internal": {
                    "dsl": SENTINELS[0],
                    "operator": SENTINELS[1],
                    "lineage": SENTINELS[2],
                    "round": SENTINELS[3]
                }
            })
            .to_string();
            let method: Method = operation.method.as_str().parse().unwrap();
            let request = authenticated_request(method, &format!("{path}{query}"), body);
            let response = local_router().oneshot(request).await.unwrap();
            assert_eq!(
                response.status(),
                StatusCode::UNPROCESSABLE_ENTITY,
                "{} {path}",
                operation.name
            );
            let bytes = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
            let problem: jope_api::error::Problem = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(
                problem.code,
                jope_api::error::ErrorCode::CapabilityUnavailable
            );
            scan_surfaces(&[(Surface::Error, String::from_utf8(bytes.to_vec()).unwrap())]).unwrap();
        }
    }
}

#[tokio::test]
async fn current_run_and_model_responses_do_not_echo_private_payloads() {
    let operations = jope_api::registry::OPERATIONS.iter().filter(|operation| {
        operation.name.starts_with("run_")
            || operation.name.starts_with("model_")
            || matches!(
                operation.name,
                "algorithm_get" | "artifact_download" | "feedback_export"
            )
    });
    let mut checked_paths = 0;

    for operation in operations {
        for path in operation.paths {
            let path = path.replace("{id}", "run_01J00000000000000000000000");
            let query = format!(
                "?dsl={}&operator={}&lineage={}&round={}",
                SENTINELS[0], SENTINELS[1], SENTINELS[2], SENTINELS[3]
            );
            let body = serde_json::json!({
                "internal": {
                    "dsl": SENTINELS[0],
                    "operator": SENTINELS[1],
                    "lineage": SENTINELS[2],
                    "round": SENTINELS[3]
                }
            })
            .to_string();
            let method: Method = operation.method.as_str().parse().unwrap();
            let mut request = authenticated_request(method, &format!("{path}{query}"), body);
            if operation.name == "run_events" {
                request.headers_mut().insert(
                    header::ACCEPT,
                    "text/event-stream".parse().expect("static Accept header"),
                );
            }
            let response = local_router().oneshot(request).await.unwrap();
            let status = response.status();
            assert_eq!(
                status,
                StatusCode::UNPROCESSABLE_ENTITY,
                "{} {path}",
                operation.name
            );
            let is_sse = response
                .headers()
                .get(header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .is_some_and(|value| value.starts_with("text/event-stream"));
            let response_headers = response
                .headers()
                .iter()
                .map(|(name, value)| {
                    format!("{}: {}", name, String::from_utf8_lossy(value.as_bytes()))
                })
                .collect::<Vec<_>>()
                .join("\n");
            let bytes = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
            let problem: jope_api::error::Problem = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(
                problem.code,
                jope_api::error::ErrorCode::CapabilityUnavailable,
                "{} {path}",
                operation.name
            );
            let output = format!("{response_headers}\n{}", String::from_utf8_lossy(&bytes));
            let surface = if is_sse {
                Surface::Sse
            } else if status.is_success() {
                Surface::Http
            } else {
                Surface::Error
            };
            scan_surfaces(&[(surface, output)])
                .unwrap_or_else(|leak| panic!("{} {path} leaked {leak:?}", operation.name));
            checked_paths += 1;
        }
    }

    assert!(checked_paths >= 10, "run/model operation inventory changed");
}

// Build the same tiny real native worker used by the service's supervised-worker
// tests. Keeping it here lets this integration test exercise a real bounded run
// without depending on a separately built appliance binary.
