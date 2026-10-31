//! Fixture-level R4 parity between terminal SSE attach and MCP event paging.
//! This does not qualify live CLI/MCP dispatch or service handlers.

use super::{durable, fixture_events, frame};
use jope_api::{decimal::DecimalU64, PublicEvent};
use jope_client::{
    transport::{header, HeaderMap, HeaderValue},
    EventStream, Request, Response, Transport,
};
use serde_json::json;
use std::{io::Cursor, sync::Mutex};

struct ParityTransport {
    events: Vec<PublicEvent>,
    initial_cut: usize,
    stream_cursors: Mutex<Vec<Option<DecimalU64>>>,
    page_cursors: Mutex<Vec<u64>>,
}

impl ParityTransport {
    fn new(events: Vec<PublicEvent>, initial_cut: usize) -> Self {
        Self {
            events,
            initial_cut,
            stream_cursors: Mutex::new(Vec::new()),
            page_cursors: Mutex::new(Vec::new()),
        }
    }

    fn query_u64(target: &str, name: &str) -> u64 {
        target
            .split_once('?')
            .expect("event page has a query")
            .1
            .split('&')
            .find_map(|pair| {
                let (key, value) = pair.split_once('=')?;
                (key == name).then(|| value.parse().expect("decimal event cursor"))
            })
            .expect("event page query field exists")
    }
}

impl Transport for ParityTransport {
    fn execute(&self, request: Request) -> jope_client::Result<Response> {
        let mut headers = HeaderMap::new();
        headers.insert("jope-api-version", HeaderValue::from_static("1.0"));

        let is_stream = request.headers.get(header::ACCEPT)
            == Some(&HeaderValue::from_static("text/event-stream"));
        let body = if is_stream {
            headers.insert(
                header::CONTENT_TYPE,
                HeaderValue::from_static("text/event-stream"),
            );
            let after = request
                .headers
                .get("last-event-id")
                .map(|value| DecimalU64(value.to_str().unwrap().parse().unwrap()));
            let first = {
                let mut cursors = self.stream_cursors.lock().unwrap();
                cursors.push(after);
                cursors.len() == 1
            };
            // Deliberately replay the last acknowledged event on reconnect.
            let events = self
                .events
                .iter()
                .filter(|event| after.is_none_or(|cursor| event.sequence >= cursor))
                .collect::<Vec<_>>();
            let mut text = String::new();
            for event in events.iter().take(if first {
                self.initial_cut
            } else {
                events.len()
            }) {
                text.push_str(&frame(event));
            }
            if first {
                if let Some(event) = events.get(self.initial_cut) {
                    let partial = frame(event);
                    text.push_str(&partial[..partial.len() / 2]);
                }
            }
            text.into_bytes()
        } else {
            headers.insert(
                header::CONTENT_TYPE,
                HeaderValue::from_static("application/json"),
            );
            let after = Self::query_u64(request.target(), "after");
            let limit = usize::try_from(Self::query_u64(request.target(), "limit")).unwrap();
            assert!((1..=256).contains(&limit));
            self.page_cursors.lock().unwrap().push(after);
            let events = self
                .events
                .iter()
                .filter(|event| event.sequence.0 > after)
                .take(limit)
                .cloned()
                .collect::<Vec<_>>();
            let next_after = events
                .last()
                .map_or(DecimalU64(after), |event| event.sequence);
            serde_json::to_vec(&json!({"events": events, "next_after": next_after})).unwrap()
        };

        Ok(Response::new(200, headers, Cursor::new(body)))
    }
}

#[test]
fn r4_terminal_reconnect_matches_mcp_pages_without_gaps_or_duplicates() {
    let expected = fixture_events();
    let run = expected[0].run_id.clone();
    let cut = 2;

    // The terminal closes after two complete events and an incomplete third
    // frame. Reattach from the last accepted ID; the server replays that ID.
    let terminal_transport = ParityTransport::new(expected.clone(), cut);
    let mut terminal_stream = EventStream::new(&terminal_transport, run.clone(), None).unwrap();
    let mut terminal_events = terminal_stream.by_ref().map(durable).collect::<Vec<_>>();
    assert_eq!(terminal_events, expected[..cut]);
    let terminal_cursor = terminal_stream.last_event_id().unwrap();
    assert_eq!(terminal_cursor, DecimalU64(u64::try_from(cut).unwrap()));
    drop(terminal_stream);
    let terminal_stream =
        EventStream::new(&terminal_transport, run.clone(), Some(terminal_cursor)).unwrap();
    terminal_events.extend(terminal_stream.map(durable));
    assert_eq!(
        *terminal_transport.stream_cursors.lock().unwrap(),
        vec![None, Some(DecimalU64(u64::try_from(cut).unwrap()))]
    );

    // The MCP observer resumes its paged run_events cursor after disconnect.
    let mcp_transport = ParityTransport::new(expected.clone(), expected.len());
    let mut mcp_events = Vec::new();
    let first_page = mcp_transport.run_events(&run, DecimalU64(0), 2).unwrap();
    assert_eq!(first_page.events, expected[..2]);
    let mut mcp_cursor = first_page.next_after;
    mcp_events.extend(first_page.events);
    while mcp_cursor.0 < expected.last().unwrap().sequence.0 {
        let page = mcp_transport.run_events(&run, mcp_cursor, 2).unwrap();
        assert!(page.next_after.0 > mcp_cursor.0);
        mcp_cursor = page.next_after;
        mcp_events.extend(page.events);
    }

    assert_eq!(terminal_events, expected);
    assert_eq!(mcp_events, expected);
    assert_eq!(terminal_events, mcp_events);
    let sequences = terminal_events
        .iter()
        .map(|event| event.sequence.0)
        .collect::<Vec<_>>();
    let last = u64::try_from(expected.len()).unwrap();
    assert_eq!(sequences, (1..=last).collect::<Vec<_>>());
    assert_eq!(*mcp_transport.page_cursors.lock().unwrap(), vec![0, 2, 4]);
}
