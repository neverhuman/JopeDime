use std::{collections::BTreeMap, fmt::Debug, str::FromStr};

use jain_smartcluster::{FrameCodec, FrameError};
use jain_smartcluster_core::{
    AttemptNumber, ClientFrame, ControlMessage, ErrorCode, EventEnvelope, EventOffset, EventRecord,
    FailureClass, FenceToken, FencedAttempt, GetRequest, JobEvent, JobId, ProtocolVersion, Request,
    RequestEnvelope, RequestId, ResponseEnvelope, ServerError, ServerFrame, StreamGap, StreamId,
    StructuredProgress, Timestamp, WorkerControlFrame, WorkerReport, WorkerReportFrame,
};
use serde::{de::DeserializeOwned, Serialize};

fn request_id() -> RequestId {
    RequestId::from_str("00000000000000000000000001").expect("fixed request ULID")
}

fn job_id() -> JobId {
    JobId::from_str("00000000000000000000000002").expect("fixed job ULID")
}

fn stream_id() -> StreamId {
    StreamId::from_str("00000000000000000000000003").expect("fixed stream ULID")
}

fn attempt() -> FencedAttempt {
    FencedAttempt::new(
        job_id(),
        AttemptNumber::new(2).expect("nonzero attempt"),
        FenceToken::new(9).expect("nonzero fence"),
    )
}

fn request_frame() -> ClientFrame {
    ClientFrame::request(RequestEnvelope {
        version: ProtocolVersion::CURRENT,
        request_id: request_id(),
        request: Request::Get(GetRequest { job_id: job_id() }),
    })
}

fn response_frame() -> ServerFrame {
    let mut details = BTreeMap::new();
    details.insert("scope".to_owned(), "web".to_owned());
    ServerFrame::response(ResponseEnvelope {
        version: ProtocolVersion::CURRENT,
        request_id: request_id(),
        result: Err(ServerError {
            code: ErrorCode::Forbidden,
            message: "owner mismatch".to_owned(),
            retry_after_ms: None,
            details,
        }),
    })
}

fn event_frame() -> ServerFrame {
    let mut fields = BTreeMap::new();
    fields.insert("fold".to_owned(), "2".to_owned());
    ServerFrame::Event(EventEnvelope {
        stream_id: stream_id(),
        event: EventRecord {
            offset: EventOffset::new(41),
            job_id: job_id(),
            attempt: Some(attempt()),
            at: Timestamp::from_unix_millis(1_700_000_000_123),
            event: JobEvent::Progress(StructuredProgress {
                phase: "train".to_owned(),
                completed: 3,
                total: Some(10),
                message: Some("fitting".to_owned()),
                fields,
            }),
        },
    })
}

fn stream_gap_frame() -> ServerFrame {
    ServerFrame::StreamGap {
        stream_id: stream_id(),
        gap: StreamGap {
            requested: EventOffset::new(5),
            available_from: EventOffset::new(17),
            latest: EventOffset::new(42),
            reason: "compacted".to_owned(),
        },
    }
}

fn worker_control_frame() -> WorkerControlFrame {
    WorkerControlFrame {
        version: ProtocolVersion::CURRENT,
        attempt: attempt(),
        sequence: 7,
        control: ControlMessage::Checkpoint { deadline_ms: 5_000 },
    }
}

fn worker_report_frame() -> WorkerReportFrame {
    WorkerReportFrame {
        version: ProtocolVersion::CURRENT,
        attempt: attempt(),
        sequence: 8,
        report: WorkerReport::Failure {
            class: FailureClass::GpuTransient,
            message: "device reset".to_owned(),
        },
    }
}

fn fixture(source: &str) -> Vec<u8> {
    let compact: String = source
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();
    assert_eq!(compact.len() % 2, 0, "fixture must contain whole bytes");
    compact
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let digits = std::str::from_utf8(pair).expect("fixture is ASCII");
            u8::from_str_radix(digits, 16).expect("fixture contains hex bytes")
        })
        .collect()
}

fn assert_golden<T>(value: &T, source: &str)
where
    T: Debug + DeserializeOwned + PartialEq + Serialize,
{
    let codec = FrameCodec::default();
    let wire = fixture(source);
    let length = u32::from_be_bytes(wire[..4].try_into().expect("four-byte length prefix"));
    let payload = &wire[4..];
    assert_eq!(
        usize::try_from(length).expect("u32 fits usize"),
        payload.len()
    );
    assert_eq!(codec.encode(value).expect("encode frame"), payload);
    assert_eq!(codec.decode::<T>(payload).expect("decode fixture"), *value);
}

fn assert_truncated_fixture_rejected<T>(source: &str)
where
    T: DeserializeOwned,
{
    let codec = FrameCodec::default();
    let mut wire = fixture(source);
    wire.pop().expect("fixture is nonempty");
    assert!(matches!(
        codec.decode::<T>(&wire[4..]),
        Err(FrameError::Serialization(_))
    ));
}

#[test]
fn postcard_wire_contract_matches_golden_fixtures() {
    assert_golden(
        &request_frame(),
        include_str!("fixtures/client_request_v1.hex"),
    );
    assert_golden(
        &response_frame(),
        include_str!("fixtures/server_response_v1.hex"),
    );
    assert_golden(&event_frame(), include_str!("fixtures/server_event_v1.hex"));
    assert_golden(
        &stream_gap_frame(),
        include_str!("fixtures/server_stream_gap_v1.hex"),
    );
    assert_golden(
        &worker_control_frame(),
        include_str!("fixtures/worker_control_v1.hex"),
    );
    assert_golden(
        &worker_report_frame(),
        include_str!("fixtures/worker_report_v1.hex"),
    );
}

#[test]
fn truncated_golden_frames_are_rejected() {
    assert_truncated_fixture_rejected::<ClientFrame>(include_str!(
        "fixtures/client_request_v1.hex"
    ));
    assert_truncated_fixture_rejected::<ServerFrame>(include_str!(
        "fixtures/server_response_v1.hex"
    ));
    assert_truncated_fixture_rejected::<ServerFrame>(include_str!("fixtures/server_event_v1.hex"));
    assert_truncated_fixture_rejected::<ServerFrame>(include_str!(
        "fixtures/server_stream_gap_v1.hex"
    ));
    assert_truncated_fixture_rejected::<WorkerControlFrame>(include_str!(
        "fixtures/worker_control_v1.hex"
    ));
    assert_truncated_fixture_rejected::<WorkerReportFrame>(include_str!(
        "fixtures/worker_report_v1.hex"
    ));
}

#[test]
fn invalid_client_frame_discriminant_is_rejected() {
    let codec = FrameCodec::default();
    let mut bytes = fixture(include_str!("fixtures/client_request_v1.hex"));
    bytes[4] = 3;
    assert!(matches!(
        codec.decode::<ClientFrame>(&bytes[4..]),
        Err(FrameError::Serialization(_))
    ));
}
