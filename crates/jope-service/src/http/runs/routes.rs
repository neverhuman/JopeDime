use super::*;

pub(crate) async fn start(
    Extension(service): Extension<Arc<RunService>>,
    request: Request<Body>,
) -> Response {
    let headers = request.headers().clone();
    #[cfg(all(feature = "failpoints", debug_assertions))]
    let pub06_values = match (
        headers.get("x-pub06-test-dsl"),
        headers.get("x-pub06-test-operator"),
        headers.get("x-pub06-test-lineage"),
        headers.get("x-pub06-test-round"),
    ) {
        (None, None, None, None) => None,
        (Some(dsl), Some(operator), Some(lineage), Some(round)) => {
            let parse = |value: &axum::http::HeaderValue| value.to_str().ok().map(str::to_owned);
            match (parse(dsl), parse(operator), parse(lineage), parse(round)) {
                (Some(dsl), Some(operator), Some(lineage), Some(round)) => {
                    Some(Pub06InternalValues {
                        dsl,
                        operator,
                        lineage,
                        round,
                        fail_after_seed: headers
                            .get("x-pub06-test-fail-after-seed")
                            .is_some_and(|value| value == "1"),
                    })
                }
                _ => return response(&invalid("Invalid PUB-06 test injection values.")),
            }
        }
        _ => {
            return response(&invalid(
                "PUB-06 test injection values must be supplied together.",
            ))
        }
    };
    // The in-process bridge uses an absolute URI; a listener receives origin
    // form. Authority is authenticated separately and is not command identity.
    let target = request
        .uri()
        .path_and_query()
        .map(|v| v.as_str().to_owned())
        .unwrap_or_else(|| "/v1/runs".into());
    let mut keys = headers.get_all("idempotency-key").iter();
    let Some(key) = keys.next().and_then(|v| v.to_str().ok()) else {
        return response(&invalid("Idempotency-Key is required."));
    };
    if keys.next().is_some() {
        return response(&invalid("Idempotency-Key must occur once."));
    }
    let Json(spec) = match Json::<RunSpec>::from_request(request, &()).await {
        Ok(value) => value,
        Err(_) => return response(&invalid("Expected a valid JSON RunSpec.")),
    };
    let body = match serde_json::to_value(&spec) {
        Ok(value) => value,
        Err(error) => return response(&io(error)),
    };
    let identity = match RequestIdentity::new(
        "local-engine-owner",
        key,
        6,
        &target,
        headers.get("if-match").and_then(|v| v.to_str().ok()),
        &body,
    ) {
        Ok(value) => value,
        Err(error) => return error.into_response(),
    };
    #[cfg(all(feature = "failpoints", debug_assertions))]
    let leak_value = std::env::var_os("JOPE_PUB06_TEST_LEAK")
        .and_then(|_| pub06_values.as_ref().map(|values| values.dsl.clone()));
    #[cfg(not(all(feature = "failpoints", debug_assertions)))]
    let pub06_values = ();
    let result = match tokio::task::spawn_blocking(move || {
        service.start(spec, identity, pub06_values)
    })
    .await
    {
        Ok(value) => value,
        Err(error) => response(&io(error)),
    };
    #[cfg(all(feature = "failpoints", debug_assertions))]
    if let Some(value) = leak_value {
        if let Ok(value) = axum::http::HeaderValue::from_str(&value) {
            let mut result = result;
            result.headers_mut().insert("x-pub06-test-leak", value);
            return result;
        }
    }
    result
}

pub(crate) async fn control(
    Extension(service): Extension<Arc<RunService>>,
    path: std::result::Result<axum::extract::Path<String>, axum::extract::rejection::PathRejection>,
    request: Request<Body>,
) -> Response {
    let Ok(axum::extract::Path(run)) = path else {
        return response(&invalid("Invalid run path."));
    };
    let Ok(run) = run.parse::<jope_api::ids::RunId>() else {
        return response(&invalid("Invalid run id."));
    };
    let headers = request.headers().clone();
    let target = request
        .uri()
        .path_and_query()
        .map(|v| v.as_str().to_owned())
        .unwrap_or_default();
    let mut keys = headers.get_all("idempotency-key").iter();
    let Some(key) = keys.next().and_then(|v| v.to_str().ok()) else {
        return response(&invalid("Idempotency-Key is required."));
    };
    if keys.next().is_some() {
        return response(&invalid("Idempotency-Key must occur once."));
    }
    let content_type = headers
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.split(';').next())
        .map(|v| v.trim().to_ascii_lowercase());
    if !content_type.as_deref().is_some_and(|v| {
        v == "application/json" || (v.starts_with("application/") && v.ends_with("+json"))
    }) {
        return response(&invalid("Expected JSON content type."));
    }
    let bytes = match axum::body::to_bytes(request.into_body(), 1024 * 1024).await {
        Ok(value) => value,
        Err(_) => return response(&invalid("Invalid control body.")),
    };
    // Preserve explicit nulls in retry identity; reject duplicate JSON keys.
    let value: Value = match jope_runtime::strict_json::decode(&bytes) {
        Ok(value) => value,
        Err(_) => return response(&invalid("Invalid control JSON.")),
    };
    let body: jope_api::registry::RunControlRequest = match serde_json::from_value(value.clone()) {
        Ok(value) => value,
        Err(_) => return response(&invalid("Expected a valid run control request.")),
    };
    let jope_api::registry::RunControlRequest::Continue { expected_revision } = body else {
        return response(&unavailable(
            "This bounded backend currently supports explicit continue only.",
        ));
    };
    let mut preconditions = headers.get_all("if-match").iter();
    let raw_if_match = preconditions.next();
    if preconditions.next().is_some() {
        return response(&invalid("If-Match must occur once."));
    }
    let if_match = match raw_if_match.map(|v| {
        v.to_str()
            .ok()
            .and_then(|s| s.trim_matches('"').parse::<u64>().ok())
    }) {
        None => None,
        Some(Some(value)) => Some(value),
        Some(None) => return response(&invalid("Invalid If-Match revision.")),
    };
    if if_match
        .zip(expected_revision)
        .is_some_and(|(a, b)| a != b.0)
    {
        return response(&invalid("Revision preconditions disagree."));
    }
    let identity = match RequestIdentity::new(
        "local-engine-owner",
        key,
        8,
        &target,
        raw_if_match.and_then(|v| v.to_str().ok()),
        &value,
    ) {
        Ok(v) => v,
        Err(e) => return e.into_response(),
    };
    match tokio::task::spawn_blocking(move || {
        service.continue_run(run, if_match.or(expected_revision.map(|v| v.0)), identity)
    })
    .await
    {
        Ok(value) => value,
        Err(error) => response(&io(error)),
    }
}

/// Execute one Q1 request and return the same compact bytes durably retained by the engine.
pub(crate) async fn q1_evaluate(
    Extension(service): Extension<Arc<RunService>>,
    request: Request<Body>,
) -> Response {
    let Json(body) = match Json::<Value>::from_request(request, &()).await {
        Ok(value) => value,
        Err(_) => return response(&invalid("Expected a valid JSON Q1 request.")),
    };
    match tokio::task::spawn_blocking(move || service.q1_evaluate(body)).await {
        Ok(Ok(bytes)) => Response::builder()
            .status(axum::http::StatusCode::OK)
            .header(axum::http::header::CONTENT_TYPE, "application/json")
            .body(Body::from(bytes))
            .unwrap_or_else(|_| response(&io("Could not construct Q1 response."))),
        Ok(Err(error)) => response(&error),
        Err(error) => response(&io(error)),
    }
}

#[derive(Default, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ListQuery {
    after: Option<String>,
    limit: Option<usize>,
}

pub(crate) async fn list(
    Extension(service): Extension<Arc<RunService>>,
    query: std::result::Result<Query<ListQuery>, axum::extract::rejection::QueryRejection>,
) -> Response {
    let Ok(Query(query)) = query else {
        return response(&invalid("Invalid run list query."));
    };
    let limit = query.limit.unwrap_or(100);
    if !(1..=256).contains(&limit) {
        return response(&invalid("limit must be between1 and256."));
    }
    let after = match query
        .after
        .map(|v| v.parse::<jope_api::ids::RunId>())
        .transpose()
    {
        Ok(value) => value,
        Err(_) => return response(&invalid("Invalid run list cursor.")),
    };
    let result = tokio::task::spawn_blocking(move || -> Result<jope_api::registry::RunPage> {
        let mut ids = service
            .store
            .run_ids_after(after.as_ref(), limit + 1)
            .map_err(io)?;
        let more = ids.len() > limit;
        ids.truncate(limit);
        let next_after = if more {
            ids.last().map(ToString::to_string)
        } else {
            None
        };
        let runs = ids
            .iter()
            .map(|id| {
                service
                    .journal
                    .snapshot(id)
                    .map_err(io)?
                    .ok_or_else(|| io("Listed run disappeared."))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(jope_api::registry::RunPage { runs, next_after })
    })
    .await;
    match result {
        Ok(Ok(page)) => Json(page).into_response(),
        Ok(Err(error)) => response(&error),
        Err(error) => response(&io(error)),
    }
}
