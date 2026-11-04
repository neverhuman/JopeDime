use super::*;

use std::fs::OpenOptions;
use std::io::{Seek, SeekFrom, Write};

use serde::{Deserialize, Serialize};
use tempfile::TempDir;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct Event {
    message: String,
}

fn options() -> EventLogOptions {
    EventLogOptions {
        max_event_bytes: 32,
        max_segment_bytes: HEADER_LEN + (RECORD_PREFIX_LEN + 24) * 2,
        max_segments: 2,
    }
}

fn event(index: u64) -> Event {
    Event {
        message: format!("event-{index:04}"),
    }
}

#[test]
fn append_reopen_and_resume_preserve_offsets() {
    let directory = TempDir::new().expect("tempdir");
    {
        let mut log = EventLog::<Event>::open(directory.path(), options()).expect("open");
        assert_eq!(log.append(&event(0)).expect("append"), 0);
        assert_eq!(log.append(&event(1)).expect("append"), 1);
    }

    let mut reopened = EventLog::<Event>::open(directory.path(), options()).expect("reopen");
    assert_eq!(reopened.next_offset(), 2);
    assert_eq!(reopened.append(&event(2)).expect("append"), 2);
    let batch = reopened.read_from(1, 10).expect("read");
    assert_eq!(batch.resume_offset, 3);
    assert_eq!(
        batch.items,
        vec![
            LogItem::Event(LoggedEvent {
                offset: 1,
                event: event(1),
            }),
            LogItem::Event(LoggedEvent {
                offset: 2,
                event: event(2),
            }),
        ]
    );
}

#[test]
fn retention_returns_a_resumable_gap() {
    let directory = TempDir::new().expect("tempdir");
    let mut log = EventLog::<Event>::open(directory.path(), options()).expect("open");
    for index in 0..8 {
        log.append(&event(index)).expect("append");
    }
    assert!(log.first_available_offset() > 0);
    let batch = log.read_from(0, 20).expect("read");
    assert!(matches!(
        batch.items.first(),
        Some(LogItem::Gap(StreamGap {
            first_missing: 0,
            next_available,
        })) if *next_available == log.first_available_offset()
    ));
    assert_eq!(batch.resume_offset, log.next_offset());
}

#[test]
fn corrupt_tail_is_truncated_and_unacknowledged_offset_is_reused() {
    let directory = TempDir::new().expect("tempdir");
    let path;
    {
        let mut log = EventLog::<Event>::open(directory.path(), options()).expect("open");
        log.append(&event(0)).expect("append");
        path = log.segments.last().expect("segment").path.clone();
    }
    let original_len = fs::metadata(&path).expect("metadata").len();
    let mut file = OpenOptions::new()
        .append(true)
        .open(&path)
        .expect("open tail");
    file.write_all(&[7_u8; 9]).expect("write torn prefix");
    file.sync_all().expect("sync tail");

    let mut recovered = EventLog::<Event>::open(directory.path(), options()).expect("recover");
    assert_eq!(recovered.recovery_report().truncated_segments, 1);
    assert_eq!(fs::metadata(&path).expect("metadata").len(), original_len);
    assert_eq!(recovered.append(&event(1)).expect("append"), 1);
}

#[test]
fn checksum_failure_drops_only_the_bad_tail() {
    let directory = TempDir::new().expect("tempdir");
    let path;
    {
        let mut log = EventLog::<Event>::open(directory.path(), options()).expect("open");
        log.append(&event(0)).expect("append");
        log.append(&event(1)).expect("append");
        path = log.segments.last().expect("segment").path.clone();
    }
    let first_payload_len = {
        let mut file = File::open(&path).expect("open");
        file.seek(SeekFrom::Start(HEADER_LEN)).expect("seek");
        let mut length = [0_u8; 4];
        file.read_exact(&mut length).expect("read length");
        u64::from(u32::from_le_bytes(length))
    };
    let second_checksum = HEADER_LEN + RECORD_PREFIX_LEN + first_payload_len + 12;
    let mut file = OpenOptions::new()
        .write(true)
        .open(&path)
        .expect("open write");
    file.seek(SeekFrom::Start(second_checksum)).expect("seek");
    file.write_all(&0_u32.to_le_bytes()).expect("corrupt crc");
    file.sync_all().expect("sync corruption");

    let mut recovered = EventLog::<Event>::open(directory.path(), options()).expect("recover");
    assert_eq!(recovered.next_offset(), 1);
    assert_eq!(recovered.append(&event(2)).expect("append"), 1);
    let batch = recovered.read_from(0, 10).expect("read");
    assert_eq!(batch.items.len(), 2);
}

#[test]
fn zero_limit_does_not_advance_subscriber() {
    let directory = TempDir::new().expect("tempdir");
    let mut log = EventLog::<Event>::open(directory.path(), options()).expect("open");
    log.append(&event(0)).expect("append");
    let batch = log.read_from(0, 0).expect("read");
    assert!(batch.items.is_empty());
    assert_eq!(batch.resume_offset, 0);
    assert_eq!(batch.end_offset, 1);
}

#[test]
fn oversized_event_is_rejected_without_consuming_an_offset() {
    let directory = TempDir::new().expect("tempdir");
    let mut log = EventLog::<Event>::open(directory.path(), options()).expect("open");
    let oversized = Event {
        message: "x".repeat(256),
    };
    assert!(matches!(
        log.append(&oversized),
        Err(EventLogError::EventTooLarge { .. })
    ));
    assert_eq!(log.next_offset(), 0);
}

#[test]
fn side_effect_free_parser_validates_bounds_and_checksum() {
    let payload = b"hello";
    let offset = 42_u64;
    let mut frame = Vec::new();
    frame.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    frame.extend_from_slice(&offset.to_le_bytes());
    frame.extend_from_slice(&record_checksum(offset, payload).to_le_bytes());
    frame.extend_from_slice(payload);
    frame.extend_from_slice(b"next");
    let decoded = decode_record_bytes(&frame, 16).expect("decode");
    assert_eq!(decoded.offset, 42);
    assert_eq!(decoded.payload, payload);
    assert_eq!(decoded.consumed, RECORD_PREFIX_LEN as usize + payload.len());

    frame[12] ^= 1;
    assert_eq!(
        decode_record_bytes(&frame, 16),
        Err(RecordDecodeError::Checksum)
    );
    assert_eq!(
        decode_record_bytes(&frame[..4], 16),
        Err(RecordDecodeError::Incomplete)
    );
}
