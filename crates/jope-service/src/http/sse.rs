//! Functional journal routes with the existing local-engine security boundary.
//! Engine composition must mount these instead of the SVC05 capability stubs.

use super::auth::{authenticate, problem, LocalAuth};
use crate::{
    events::{event_kind, EventLog, EventPage, EVENT_KINDS},
    store::{Result, StoreError},
};
use axum::{
    extract::{
        rejection::{PathRejection, QueryRejection},
        Path, Query, State,
    },
    http::{header, HeaderMap},
    middleware,
    response::{
        sse::{Event, KeepAlive},
        IntoResponse, Response, Sse,
    },
    routing::get,
    Json, Router,
};
use jope_api::{decimal::DecimalU64, error::ErrorCode, ids::RunId, PublicEvent, TelemetryEvent};
use serde::Deserialize;
use std::{collections::VecDeque, sync::Arc, time::Duration};
use tokio::{
    sync::{broadcast, watch},
    time::Instant,
};

/// The caller uses the same router for a loopback listener and in-process HTTP.
pub fn event_routes(log: Arc<EventLog>, auth: LocalAuth) -> Router {
    Router::new()
        .route("/v1/runs/{id}/events", get(deliver_events))
        .route("/v1/runs/{id}", get(long_poll_status))
        .fallback(|| async {
            problem(
                ErrorCode::NotFound,
                "No event operation matches this path.",
                "Use a run status or events path.",
                None,
            )
        })
        .method_not_allowed_fallback(|| async { invalid("Use GET for journal reads.", None) })
        .with_state(log)
        .layer(middleware::from_fn_with_state(auth, authenticate))
}

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct EventsQuery {
    after: Option<String>,
    limit: Option<usize>,
    kinds: Option<String>,
}

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct StatusQuery {
    wait_seconds: Option<f64>,
    after_event: Option<String>,
}

fn invalid(detail: &str, field: Option<&str>) -> Response {
    problem(
        ErrorCode::InvalidInput,
        detail,
        "Use valid API v1 query arguments.",
        field,
    )
}

fn journal_failure(error: StoreError) -> Response {
    match error {
        StoreError::Invalid(detail) => invalid(detail, None),
        _ => problem(
            ErrorCode::IoFailure,
            "The committed journal could not be read.",
            "Inspect the local engine/store diagnostics before retrying.",
            None,
        ),
    }
}

fn missing_run() -> Response {
    problem(
        ErrorCode::NotFound,
        "The run does not exist.",
        "Use an existing run ID.",
        Some("run_id"),
    )
}

fn cursor(value: &str) -> Option<u64> {
    serde_json::from_value::<DecimalU64>(serde_json::Value::String(value.into()))
        .ok()
        .map(|n| n.0)
}

fn requested_kinds(value: Option<&str>) -> Option<Vec<String>> {
    let Some(value) = value else {
        return Some(vec![]);
    };
    let values: Vec<_> = value.split(',').map(str::to_owned).collect();
    values
        .iter()
        .all(|v| EVENT_KINDS.contains(&v.as_str()))
        .then_some(values)
}

async fn read_page(
    log: Arc<EventLog>,
    run: RunId,
    after: u64,
    limit: usize,
    kinds: Vec<String>,
) -> Result<Option<EventPage>> {
    tokio::task::spawn_blocking(move || log.page(&run, after, limit, &kinds))
        .await
        .map_err(|_| StoreError::Integrity("event page reader failed"))?
}

async fn deliver_events(
    State(log): State<Arc<EventLog>>,
    path: std::result::Result<Path<String>, PathRejection>,
    query: std::result::Result<Query<EventsQuery>, QueryRejection>,
    headers: HeaderMap,
) -> Response {
    let Ok(Path(id)) = path else {
        return invalid("Invalid run path.", Some("run_id"));
    };
    let Ok(run) = id.parse::<RunId>() else {
        return invalid("Invalid run ID.", Some("run_id"));
    };
    let Ok(Query(query)) = query else {
        return invalid("Invalid events query.", None);
    };
    let limit = query.limit.unwrap_or(100);
    if !(1..=256).contains(&limit) {
        return invalid("Event limit must be 1 through 256.", Some("limit"));
    }
    let Some(kinds) = requested_kinds(query.kinds.as_deref()) else {
        return invalid("Unknown durable event kind.", Some("kinds"));
    };
    let streaming = headers
        .get(header::ACCEPT)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| {
            v.split(',').any(|part| {
                let mut fields = part.split(';');
                fields
                    .next()
                    .is_some_and(|mime| mime.trim().eq_ignore_ascii_case("text/event-stream"))
                    && fields.all(|parameter| {
                        let Some((name, value)) = parameter.split_once('=') else {
                            return true;
                        };
                        !name.trim().eq_ignore_ascii_case("q")
                            || value
                                .trim()
                                .parse::<f32>()
                                .is_ok_and(|quality| quality > 0.0 && quality <= 1.0)
                    })
            })
        });
    let mut header_ids = headers.get_all("last-event-id").iter();
    let last_id = if streaming { header_ids.next() } else { None };
    if streaming && header_ids.next().is_some() {
        return invalid("Last-Event-ID must occur once.", Some("Last-Event-ID"));
    }
    let effective = match last_id {
        Some(value) => match value.to_str() {
            Ok(v) => v,
            Err(_) => return invalid("Invalid Last-Event-ID.", Some("Last-Event-ID")),
        },
        None => query.after.as_deref().unwrap_or("0"),
    };
    let Some(after) = cursor(effective) else {
        return invalid(
            "Event cursor must be a canonical u64 decimal.",
            Some("after"),
        );
    };
    // Register both channels BEFORE reading the first committed page.
    let wake = log.subscribe();
    let telemetry = log.subscribe_telemetry();
    let page = match read_page(Arc::clone(&log), run.clone(), after, limit, kinds.clone()).await {
        Ok(Some(page)) => page,
        Ok(None) => return missing_run(),
        Err(error) => return journal_failure(error),
    };
    if !streaming {
        return Json(page).into_response();
    }
    let mut state = Delivery {
        log,
        run,
        cursor: after,
        limit,
        kinds,
        pending: VecDeque::new(),
        wake,
        telemetry,
        last_telemetry: None,
    };
    state.accept_page(page);
    let stream = futures_util::stream::try_unfold(state, |mut state| async move {
        let event = state.next_frame().await?;
        Ok::<_, StoreError>(Some((event, state)))
    });
    Sse::new(stream)
        .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
        .into_response()
}

async fn long_poll_status(
    State(log): State<Arc<EventLog>>,
    path: std::result::Result<Path<String>, PathRejection>,
    query: std::result::Result<Query<StatusQuery>, QueryRejection>,
) -> Response {
    let Ok(Path(id)) = path else {
        return invalid("Invalid run path.", Some("run_id"));
    };
    let Ok(run) = id.parse::<RunId>() else {
        return invalid("Invalid run ID.", Some("run_id"));
    };
    let Ok(Query(query)) = query else {
        return invalid("Invalid status query.", None);
    };
    let seconds = query.wait_seconds.unwrap_or(0.0);
    if !seconds.is_finite() || !(0.0..=60.0).contains(&seconds) {
        return invalid(
            "wait_seconds must be finite and between 0 and 60.",
            Some("wait_seconds"),
        );
    }
    let Some(after) = cursor(query.after_event.as_deref().unwrap_or("0")) else {
        return invalid(
            "after_event must be a canonical u64 decimal.",
            Some("after_event"),
        );
    };
    match log
        .wait_after(&run, after, Duration::from_secs_f64(seconds))
        .await
    {
        Ok(Some(snapshot)) => Json(snapshot).into_response(),
        Ok(None) => missing_run(),
        Err(error) => journal_failure(error),
    }
}

struct Delivery {
    log: Arc<EventLog>,
    run: RunId,
    cursor: u64,
    limit: usize,
    kinds: Vec<String>,
    pending: VecDeque<PublicEvent>,
    wake: watch::Receiver<()>,
    telemetry: broadcast::Receiver<TelemetryEvent>,
    last_telemetry: Option<Instant>,
}

impl Delivery {
    fn accept_page(&mut self, page: EventPage) {
        if page.watermark.saturating_sub(self.cursor) > 1000 {
            // Durable rows remain authoritative; discard accumulated telemetry.
            self.telemetry = self.telemetry.resubscribe();
        }
        self.cursor = page.next_after.0;
        self.pending = page.events.into();
    }

    async fn next_frame(&mut self) -> Result<Event> {
        loop {
            if let Some(event) = self.pending.pop_front() {
                let data = serde_json::to_string(&event)?;
                return Ok(Event::default()
                    .id(event.sequence.0.to_string())
                    .event(event_kind(&event.event))
                    .data(data));
            }
            let page = read_page(
                Arc::clone(&self.log),
                self.run.clone(),
                self.cursor,
                self.limit,
                self.kinds.clone(),
            )
            .await?
            .ok_or(StoreError::Integrity("stream run disappeared"))?;
            let scanned = page.next_after.0 > self.cursor;
            self.accept_page(page);
            if scanned || !self.pending.is_empty() {
                continue;
            }
            tokio::select! {
                biased;
                changed = self.wake.changed() => changed.map_err(|_| StoreError::Integrity("stream notifier closed"))?,
                sample = self.telemetry.recv() => match sample {
                    Ok(sample) if sample.run_id == self.run => {
                        let now = Instant::now();
                        if self.last_telemetry.is_none_or(|last| now.duration_since(last) >= Duration::from_secs(1)) {
                            self.last_telemetry = Some(now);
                            return Ok(Event::default().event("telemetry").data(serde_json::to_string(&sample)?));
                        }
                    },
                    Ok(_) | Err(broadcast::error::RecvError::Lagged(_)) => {},
                    Err(broadcast::error::RecvError::Closed) => return Err(StoreError::Integrity("telemetry channel closed")),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        events::tests::{append, fixture, test_body},
        http::auth::BearerToken,
    };
    use axum::{
        body::{to_bytes, Body},
        http::{Method, Request, StatusCode},
    };
    use futures_util::StreamExt;
    use jope_api::{error::Problem, RunSnapshot, TelemetryBody};
    use serde_json::Value;
    use tower::ServiceExt;

    fn app(log: Arc<EventLog>) -> Router {
        event_routes(
            log,
            LocalAuth::new(BearerToken::from_hex(&"ab".repeat(32)).unwrap(), 31234).unwrap(),
        )
    }
    fn request(path: &str, sse: bool, last_id: Option<&str>) -> Request<Body> {
        let mut builder = Request::builder()
            .method(Method::GET)
            .uri(path)
            .header(header::HOST, "127.0.0.1:31234")
            .header(header::AUTHORIZATION, format!("Bearer {}", "ab".repeat(32)));
        if sse {
            builder = builder.header(header::ACCEPT, "text/event-stream");
        }
        if let Some(id) = last_id {
            builder = builder.header("last-event-id", id);
        }
        builder.body(Body::empty()).unwrap()
    }
    async fn document(response: Response) -> Value {
        serde_json::from_slice(&to_bytes(response.into_body(), 1 << 20).await.unwrap()).unwrap()
    }
    async fn frame(
        stream: &mut (impl futures_util::Stream<Item = std::result::Result<axum::body::Bytes, axum::Error>>
                  + Unpin),
    ) -> String {
        String::from_utf8(
            tokio::time::timeout(Duration::from_secs(2), stream.next())
                .await
                .unwrap()
                .unwrap()
                .unwrap()
                .to_vec(),
        )
        .unwrap()
    }
    fn decoded_event(frame: &str) -> jope_api::PublicEvent {
        let data = frame
            .lines()
            .find_map(|line| line.strip_prefix("data: "))
            .unwrap();
        serde_json::from_str(data).unwrap()
    }

    #[tokio::test]
    async fn authenticated_pages_have_decimal_cursors_and_match_sse_bytes_across_reconnect() {
        let (_dir, log, snapshot) = fixture();
        let current = append(
            &log,
            &snapshot,
            vec![
                test_body("first"),
                test_body("second"),
                test_body("third\nquoted \"body\""),
            ],
        );
        let root = format!("/v1/runs/{}/events", current.run_id);
        let response = app(Arc::clone(&log))
            .oneshot(request(&format!("{root}?limit=2"), false, None))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()["jope-api-version"], "1.0");
        let page = document(response).await;
        assert_eq!(page["next_after"], "2");
        assert_eq!(page["events"].as_array().unwrap().len(), 2);
        assert!(page.get("watermark").is_none());
        let response = app(Arc::clone(&log))
            .oneshot(request(&root, true, None))
            .await
            .unwrap();
        assert!(response.headers()[header::CONTENT_TYPE]
            .to_str()
            .unwrap()
            .starts_with("text/event-stream"));
        let mut first = response.into_body().into_data_stream();
        let a = decoded_event(&frame(&mut first).await);
        let b = decoded_event(&frame(&mut first).await);
        assert_eq!((a.sequence.0, b.sequence.0), (1, 2));
        assert_eq!(serde_json::to_value(a).unwrap(), page["events"][0]);
        drop(first);
        let response = app(Arc::clone(&log))
            .oneshot(request(&format!("{root}?after=0&limit=1"), true, Some("2")))
            .await
            .unwrap();
        let mut resumed = response.into_body().into_data_stream();
        let c = decoded_event(&frame(&mut resumed).await);
        assert_eq!(c.sequence.0, 3);
        assert_eq!(event_kind(&c.event), "reasoning");
        drop(resumed);
        let next = append(&log, &current, vec![test_body("fourth")]);
        let response = app(Arc::clone(&log))
            .oneshot(request(&format!("{root}?after=0"), true, Some("3")))
            .await
            .unwrap();
        let mut reconnected = response.into_body().into_data_stream();
        let d = decoded_event(&frame(&mut reconnected).await);
        assert_eq!(d.sequence.0, 4);
        let page = log.page(&next.run_id, 2, 256, &[]).unwrap().unwrap();
        assert_eq!(page.events, vec![c, d]);
    }

    #[tokio::test]
    async fn telemetry_is_unidentified_ephemeral_and_cannot_advance_resume_or_long_poll() {
        let (_dir, log, snapshot) = fixture();
        let current = append(&log, &snapshot, vec![test_body("durable")]);
        let response = app(Arc::clone(&log))
            .oneshot(request(
                &format!("/v1/runs/{}/events?after=1", current.run_id),
                true,
                None,
            ))
            .await
            .unwrap();
        let mut stream = response.into_body().into_data_stream();
        let other: RunId = "run_01JBD8Q6N4V2T5Y8K3M7P9R1S1".parse().unwrap();
        log.publish_telemetry(TelemetryEvent {
            run_id: other,
            time: "2026-10-30T00:00:00Z".into(),
            event: TelemetryBody::Telemetry(vec![]),
        });
        log.publish_telemetry(TelemetryEvent {
            run_id: current.run_id.clone(),
            time: "2026-10-30T00:00:01Z".into(),
            event: TelemetryBody::Telemetry(vec![]),
        });
        let ephemeral = frame(&mut stream).await;
        assert!(ephemeral.contains("event: telemetry"));
        assert!(!ephemeral.lines().any(|l| l.starts_with("id:")));
        assert!(ephemeral.contains(&current.run_id.to_string()));
        assert_eq!(
            log.snapshot(&current.run_id)
                .unwrap()
                .unwrap()
                .event_watermark
                .0,
            1
        );
        let next = append(&log, &current, vec![test_body("after telemetry")]);
        let durable = decoded_event(&frame(&mut stream).await);
        assert_eq!(durable.sequence.0, 2);
        assert_eq!(
            log.page(&next.run_id, 0, 256, &[])
                .unwrap()
                .unwrap()
                .events
                .len(),
            2
        );
    }

    #[tokio::test]
    async fn telemetry_delivery_is_limited_to_one_frame_per_second() {
        let (_dir, log, snapshot) = fixture();
        let response = app(Arc::clone(&log))
            .oneshot(request(
                &format!("/v1/runs/{}/events", snapshot.run_id),
                true,
                None,
            ))
            .await
            .unwrap();
        let mut stream = response.into_body().into_data_stream();
        let sample = TelemetryEvent {
            run_id: snapshot.run_id,
            time: "2026-10-30T00:00:00Z".into(),
            event: TelemetryBody::Telemetry(vec![]),
        };
        log.publish_telemetry(sample.clone());
        frame(&mut stream).await;
        log.publish_telemetry(sample.clone());
        assert!(
            tokio::time::timeout(Duration::from_millis(20), stream.next())
                .await
                .is_err()
        );
        tokio::time::sleep(Duration::from_secs(1)).await;
        log.publish_telemetry(sample);
        assert!(frame(&mut stream).await.contains("event: telemetry"));
    }

    #[tokio::test]
    async fn http_long_poll_wakes_under_50ms_and_honors_the_timeout() {
        let (_dir, log, snapshot) = fixture();
        let path = format!("/v1/runs/{}", snapshot.run_id);
        let router = app(Arc::clone(&log));
        let waiting = tokio::spawn(router.oneshot(request(
            &format!("{path}?wait_seconds=60&after_event=0"),
            false,
            None,
        )));
        tokio::time::sleep(Duration::from_millis(5)).await;
        let writer = Arc::clone(&log);
        let old = snapshot.clone();
        let (committed, start) = tokio::task::spawn_blocking(move || {
            let snapshot = append(&writer, &old, vec![test_body("wake")]);
            (snapshot, std::time::Instant::now())
        })
        .await
        .unwrap();
        let response = tokio::time::timeout(
            Duration::from_millis(50).saturating_sub(start.elapsed()),
            waiting,
        )
        .await
        .unwrap()
        .unwrap()
        .unwrap();
        assert!(start.elapsed() < Duration::from_millis(50));
        let actual: RunSnapshot = serde_json::from_value(document(response).await).unwrap();
        assert_eq!(actual, committed);
        let start = std::time::Instant::now();
        // Keep database teardown outside the HTTP timeout measurement.
        let response = app(Arc::clone(&log))
            .oneshot(request(
                &format!("{path}?wait_seconds=0.04&after_event=1"),
                false,
                None,
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert!(start.elapsed() >= Duration::from_millis(40));
        assert!(start.elapsed() < Duration::from_millis(250));
    }

    #[tokio::test]
    async fn journal_routes_preserve_auth_host_origin_and_problem_boundaries() {
        let (_dir, log, snapshot) = fixture();
        let path = format!("/v1/runs/{}/events", snapshot.run_id);
        for mode in ["token", "host", "origin"] {
            let mut req = request(&path, false, None);
            match mode {
                "token" => {
                    req.headers_mut().remove(header::AUTHORIZATION);
                }
                "host" => {
                    req.headers_mut()
                        .insert(header::HOST, "evil.example".parse().unwrap());
                }
                _ => {
                    req.headers_mut()
                        .insert(header::ORIGIN, "https://evil.example".parse().unwrap());
                }
            }
            let response = app(Arc::clone(&log)).oneshot(req).await.unwrap();
            assert_eq!(
                response.status(),
                if mode == "token" {
                    StatusCode::UNAUTHORIZED
                } else {
                    StatusCode::BAD_REQUEST
                }
            );
            assert_eq!(
                response.headers()[header::CONTENT_TYPE],
                Problem::CONTENT_TYPE
            );
            assert_eq!(response.headers()["jope-api-version"], "1.0");
            assert!(!response
                .headers()
                .keys()
                .any(|h| h.as_str().starts_with("access-control-")));
        }
        let missing = app(log)
            .oneshot(request(
                "/v1/runs/run_01JBD8Q6N4V2T5Y8K3M7P9R1S1/events",
                false,
                None,
            ))
            .await
            .unwrap();
        assert_eq!(missing.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn malformed_cursors_waits_and_filters_are_typed_refusals() {
        let (_dir, log, snapshot) = fixture();
        let path = format!("/v1/runs/{}", snapshot.run_id);
        for query in [
            "events?limit=0",
            "events?limit=257",
            "events?after=01",
            "events?after=-1",
            "events?after=18446744073709551616",
            "events?kinds=telemetry",
            "events?after=1",
        ] {
            let response = app(Arc::clone(&log))
                .oneshot(request(&format!("{path}/{query}"), false, None))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{query}");
            assert_eq!(
                response.headers()[header::CONTENT_TYPE],
                Problem::CONTENT_TYPE
            );
        }
        for query in [
            "wait_seconds=61",
            "wait_seconds=-1",
            "wait_seconds=NaN",
            "wait_seconds=inf",
            "after_event=01",
            "wait_seconds=no",
        ] {
            let response = app(Arc::clone(&log))
                .oneshot(request(&format!("{path}?{query}"), false, None))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{query}");
        }
        let mut req = request(&format!("{path}/events"), true, Some("0"));
        req.headers_mut()
            .append("last-event-id", "0".parse().unwrap());
        assert_eq!(
            app(Arc::clone(&log)).oneshot(req).await.unwrap().status(),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            app(log)
                .oneshot(request(&format!("{path}/events"), true, Some("01")))
                .await
                .unwrap()
                .status(),
            StatusCode::BAD_REQUEST
        );
    }

    #[tokio::test]
    async fn undecodable_path_parameters_keep_the_problem_json_contract() {
        let (_dir, log, _snapshot) = fixture();
        for path in ["/v1/runs/%FF", "/v1/runs/%FF/events"] {
            let response = app(Arc::clone(&log))
                .oneshot(request(path, false, None))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            assert_eq!(
                response.headers()[header::CONTENT_TYPE],
                Problem::CONTENT_TYPE
            );
        }
    }

    #[tokio::test]
    async fn an_explicitly_unacceptable_sse_media_type_returns_the_json_page() {
        let (_dir, log, snapshot) = fixture();
        let mut req = request(&format!("/v1/runs/{}/events", snapshot.run_id), true, None);
        req.headers_mut().insert(
            header::ACCEPT,
            "application/json, text/event-stream;q=0".parse().unwrap(),
        );
        let response = app(log).oneshot(req).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()[header::CONTENT_TYPE], "application/json");
        assert_eq!(document(response).await["next_after"], "0");
    }

    #[tokio::test]
    async fn separate_observers_and_filtered_pages_leave_the_durable_projection_unchanged() {
        let (_dir, log, snapshot) = fixture();
        let current = append(&log, &snapshot, vec![test_body("unchanged")]);
        let path = format!("/v1/runs/{}/events", current.run_id);
        let before = log
            .page(&current.run_id, 0, 256, &[])
            .unwrap()
            .unwrap()
            .events;
        for _ in 0..2 {
            let response = app(Arc::clone(&log))
                .oneshot(request(&path, true, None))
                .await
                .unwrap();
            let mut stream = response.into_body().into_data_stream();
            assert_eq!(decoded_event(&frame(&mut stream).await), before[0]);
        }
        let response = app(Arc::clone(&log))
            .oneshot(request(&format!("{path}?kinds=warning"), false, None))
            .await
            .unwrap();
        let filtered = document(response).await;
        assert_eq!(filtered["events"], serde_json::json!([]));
        assert_eq!(filtered["next_after"], "1");
        assert_eq!(log.snapshot(&current.run_id).unwrap().unwrap(), current);
        assert_eq!(
            log.page(&current.run_id, 0, 256, &[])
                .unwrap()
                .unwrap()
                .events,
            before
        );
    }
}
