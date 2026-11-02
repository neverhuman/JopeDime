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
pub(super) struct EventsQuery {
    after: Option<String>,
    limit: Option<usize>,
    kinds: Option<String>,
}

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct StatusQuery {
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

pub(super) async fn deliver_events(
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

pub(super) async fn long_poll_status(
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
mod tests;
