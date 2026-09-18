use jain_smartcluster::*;
use serde::Serialize;
#[derive(Serialize)]
#[allow(dead_code)] // Preserve every old discriminant in this compatibility fixture.
enum LegacyRequest {
    Status(StatusRequest),
    Submit(Box<SubmitRequest>),
    Get(GetRequest),
    List(ListRequest),
    Wait(WaitRequest),
    Cancel(CancelRequest),
    Pause(PauseRequest),
    Resume(ResumeRequest),
    SendControl(SendControlRequest),
    Subscribe(SubscribeRequest),
    Attach(AttachRequest),
}
#[test]
fn appended_request_preserves_legacy_unary_and_stream_postcard_bytes() {
    let id = JobId::new();
    let pairs = [
        (
            Request::Status(StatusRequest),
            LegacyRequest::Status(StatusRequest),
        ),
        (
            Request::Get(GetRequest { job_id: id }),
            LegacyRequest::Get(GetRequest { job_id: id }),
        ),
        (
            Request::Wait(WaitRequest { job_id: id }),
            LegacyRequest::Wait(WaitRequest { job_id: id }),
        ),
        (
            Request::Pause(PauseRequest { job_id: id }),
            LegacyRequest::Pause(PauseRequest { job_id: id }),
        ),
        (
            Request::Resume(ResumeRequest { job_id: id }),
            LegacyRequest::Resume(ResumeRequest { job_id: id }),
        ),
        (
            Request::Attach(AttachRequest {
                job_id: id,
                offset: EventOffset::default(),
                read_only: true,
            }),
            LegacyRequest::Attach(AttachRequest {
                job_id: id,
                offset: EventOffset::default(),
                read_only: true,
            }),
        ),
    ];
    for (new, old) in pairs {
        assert_eq!(
            postcard::to_allocvec(&new).unwrap(),
            postcard::to_allocvec(&old).unwrap()
        );
    }
}
fn request() -> AttachTrustedFitChildRequest {
    AttachTrustedFitChildRequest {
        authority_attempt: FencedAttempt::new(
            JobId::new(),
            AttemptNumber::new(1).unwrap(),
            FenceToken::new(1).unwrap(),
        ),
        target_attempt: FencedAttempt::new(
            JobId::new(),
            AttemptNumber::new(1).unwrap(),
            FenceToken::new(2).unwrap(),
        ),
        child_pid: 42,
        child_birth_ticks: 123,
        child_boot_id: "11111111-1111-1111-1111-111111111111".into(),
        task_manifest_sha256: "a".repeat(64),
        descriptor_sha256: "b".repeat(64),
        candidate_index: 0,
        idempotency_key: IdempotencyKey::new("candidate-0").unwrap(),
        child_executable_sha256: "c".repeat(64),
    }
}
#[test]
fn placement_roundtrip_is_appended_and_bad_identity_is_refused() {
    let req = request();
    req.validate().unwrap();
    let value = Request::AttachTrustedFitChild(Box::new(req.clone()));
    let bytes = postcard::to_allocvec(&value).unwrap();
    assert_eq!(bytes[0], 11);
    let decoded: Request = postcard::from_bytes(&bytes).unwrap();
    assert_eq!(value, decoded);
    let mut bad = req.clone();
    bad.child_pid = 0;
    assert!(bad.validate().is_err());
    bad = req.clone();
    bad.child_boot_id = "wrong-boot".into();
    assert!(bad.validate().is_err());
    bad = req;
    bad.child_executable_sha256 = "A".repeat(64);
    assert!(bad.validate().is_err());
}
#[test]
fn attached_response_keeps_legacy_index_and_payload() {
    let opened = StreamOpened {
        stream_id: StreamId::new(),
        next_offset: EventOffset::default(),
    };
    let mut expected = vec![8];
    expected.extend(postcard::to_allocvec(&opened).unwrap());
    assert_eq!(
        postcard::to_allocvec(&Response::Attached(opened)).unwrap(),
        expected
    );
}

#[test]
fn attempt_inspection_appends_wire_variants_and_roundtrips_exact_fence() {
    let attempt = request().target_attempt;
    let query = Request::InspectAttempt(InspectAttemptRequest { attempt });
    let bytes = postcard::to_stdvec(&query).unwrap();
    assert_eq!(bytes[0], 12);
    assert_eq!(postcard::from_bytes::<Request>(&bytes).unwrap(), query);
    let response = Response::AttemptInspected(None);
    let bytes = postcard::to_stdvec(&response).unwrap();
    assert_eq!(bytes, vec![10, 0]);
    assert_eq!(postcard::from_bytes::<Response>(&bytes).unwrap(), response);
}
