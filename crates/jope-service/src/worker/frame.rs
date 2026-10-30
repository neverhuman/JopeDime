//! Length-prefixed JSON frames used by the worker protocol.

use std::fmt;
use std::io::{self, Read};
use std::str::{self, Utf8Error};

use serde_json::Value;

/// The maximum JSON payload size, in bytes. The four-byte prefix is not part of the payload.
pub const MAX_FRAME_BYTES: usize = 8 * 1024 * 1024;
const LENGTH_PREFIX_BYTES: usize = std::mem::size_of::<u32>();

/// Errors raised while encoding or decoding a worker frame.
#[derive(Debug)]
pub enum FrameError {
    /// The length prefix or encoded JSON exceeds the protocol's maximum payload size.
    Oversized { length: usize, maximum: usize },
    /// The stream ended before all four bytes of the length prefix arrived.
    TruncatedHeader { received: usize },
    /// The stream ended before the declared JSON payload arrived.
    TruncatedPayload { expected: usize, received: usize },
    /// The payload is not valid UTF-8.
    InvalidUtf8(Utf8Error),
    /// The UTF-8 payload is not valid JSON.
    InvalidJson(serde_json::Error),
    /// Reading the stream failed for a reason other than EOF.
    Io(io::Error),
}

impl fmt::Display for FrameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Oversized { length, maximum } => {
                write!(f, "frame payload is {length} bytes; maximum is {maximum}")
            }
            Self::TruncatedHeader { received } => write!(
                f,
                "truncated frame header: received {received} of {LENGTH_PREFIX_BYTES} bytes"
            ),
            Self::TruncatedPayload { expected, received } => write!(
                f,
                "truncated frame payload: received {received} of {expected} bytes"
            ),
            Self::InvalidUtf8(error) => write!(f, "frame payload is not UTF-8: {error}"),
            Self::InvalidJson(error) => write!(f, "frame payload is not valid JSON: {error}"),
            Self::Io(error) => write!(f, "failed to read frame: {error}"),
        }
    }
}

impl std::error::Error for FrameError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidUtf8(error) => Some(error),
            Self::InvalidJson(error) => Some(error),
            Self::Io(error) => Some(error),
            Self::Oversized { .. }
            | Self::TruncatedHeader { .. }
            | Self::TruncatedPayload { .. } => None,
        }
    }
}

/// Encode a JSON value as a little-endian `u32` payload length followed by JSON bytes.
pub fn encode_frame(value: &Value) -> Result<Vec<u8>, FrameError> {
    let payload = serde_json::to_vec(value).map_err(FrameError::InvalidJson)?;
    if payload.len() > MAX_FRAME_BYTES {
        return Err(FrameError::Oversized {
            length: payload.len(),
            maximum: MAX_FRAME_BYTES,
        });
    }

    let length = u32::try_from(payload.len()).map_err(|_| FrameError::Oversized {
        length: payload.len(),
        maximum: MAX_FRAME_BYTES,
    })?;
    let mut frame = Vec::with_capacity(LENGTH_PREFIX_BYTES + payload.len());
    frame.extend_from_slice(&length.to_le_bytes());
    frame.extend_from_slice(&payload);
    Ok(frame)
}

/// Read and decode one frame, leaving any following frame bytes in the stream.
pub fn decode_frame<R: Read>(reader: &mut R) -> Result<Value, FrameError> {
    let mut header = [0; LENGTH_PREFIX_BYTES];
    let header_read = read_available(reader, &mut header).map_err(FrameError::Io)?;
    if header_read != LENGTH_PREFIX_BYTES {
        return Err(FrameError::TruncatedHeader {
            received: header_read,
        });
    }

    let payload_length = u32::from_le_bytes(header) as usize;
    if payload_length > MAX_FRAME_BYTES {
        return Err(FrameError::Oversized {
            length: payload_length,
            maximum: MAX_FRAME_BYTES,
        });
    }

    let mut payload = vec![0; payload_length];
    let payload_read = read_available(reader, &mut payload).map_err(FrameError::Io)?;
    if payload_read != payload_length {
        return Err(FrameError::TruncatedPayload {
            expected: payload_length,
            received: payload_read,
        });
    }

    let json = str::from_utf8(&payload).map_err(FrameError::InvalidUtf8)?;
    serde_json::from_str(json).map_err(FrameError::InvalidJson)
}

fn read_available<R: Read>(reader: &mut R, buffer: &mut [u8]) -> io::Result<usize> {
    let mut received = 0;
    while received < buffer.len() {
        match reader.read(&mut buffer[received..]) {
            Ok(0) => break,
            Ok(count) => received += count,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        }
    }
    Ok(received)
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use serde_json::{Map, Number, Value};

    use super::{decode_frame, encode_frame, FrameError, MAX_FRAME_BYTES};

    fn json_values() -> BoxedStrategy<Value> {
        let leaves = prop_oneof![
            any::<bool>().prop_map(Value::Bool),
            any::<i64>().prop_map(|value| Value::Number(Number::from(value))),
            any::<String>().prop_map(Value::String),
        ];

        leaves
            .prop_recursive(3, 64, 8, |inner| {
                prop_oneof![
                    prop::collection::vec(inner.clone(), 0..8).prop_map(Value::Array),
                    prop::collection::btree_map("[a-z]{0,12}", inner, 0..8)
                        .prop_map(|values| Value::Object(Map::from_iter(values))),
                ]
            })
            .boxed()
    }

    #[test]
    fn round_trips_json_value() {
        let value = serde_json::json!({"attempt": 7, "dsl": "x + 1", "ok": true});
        let frame = encode_frame(&value).unwrap();
        assert_eq!(
            u32::from_le_bytes(frame[..4].try_into().unwrap()) as usize,
            frame.len() - 4
        );
        assert_eq!(decode_frame(&mut frame.as_slice()).unwrap(), value);
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 64, ..ProptestConfig::default() })]

        #[test]
        fn random_json_values_round_trip(value in json_values()) {
            let frame = encode_frame(&value).unwrap();
            prop_assert_eq!(decode_frame(&mut frame.as_slice()).unwrap(), value);
        }
    }

    #[test]
    fn truncated_header_is_an_error_without_panicking() {
        for received in 0..4 {
            let bytes = [0x01, 0x02, 0x03][..received.min(3)].to_vec();
            assert!(matches!(
                decode_frame(&mut bytes.as_slice()),
                Err(FrameError::TruncatedHeader { received: actual }) if actual == received
            ));
        }
    }

    #[test]
    fn truncated_payload_reports_expected_and_received_lengths() {
        let mut bytes = (5_u32).to_le_bytes().to_vec();
        bytes.extend_from_slice(b"{} ");
        assert!(matches!(
            decode_frame(&mut bytes.as_slice()),
            Err(FrameError::TruncatedPayload {
                expected: 5,
                received: 3
            })
        ));
    }

    #[test]
    fn invalid_utf8_and_invalid_json_have_distinct_errors() {
        let mut invalid_utf8 = 1_u32.to_le_bytes().to_vec();
        invalid_utf8.push(0xff);
        assert!(matches!(
            decode_frame(&mut invalid_utf8.as_slice()),
            Err(FrameError::InvalidUtf8(_))
        ));

        let mut invalid_json = 1_u32.to_le_bytes().to_vec();
        invalid_json.push(b"x"[0]);
        assert!(matches!(
            decode_frame(&mut invalid_json.as_slice()),
            Err(FrameError::InvalidJson(_))
        ));
    }

    #[test]
    fn rejects_oversize_before_reading_or_allocating_payload() {
        let header = ((MAX_FRAME_BYTES as u32) + 1).to_le_bytes();
        assert!(matches!(
            decode_frame(&mut header.as_slice()),
            Err(FrameError::Oversized { length, maximum })
                if length == MAX_FRAME_BYTES + 1 && maximum == MAX_FRAME_BYTES
        ));
    }

    #[test]
    fn encoder_rejects_oversize_json() {
        let value = Value::String("x".repeat(MAX_FRAME_BYTES));
        assert!(matches!(
            encode_frame(&value),
            Err(FrameError::Oversized { maximum, .. }) if maximum == MAX_FRAME_BYTES
        ));
    }

    #[test]
    fn decodes_one_frame_and_preserves_the_next() {
        let first = encode_frame(&Value::Bool(true)).unwrap();
        let second = encode_frame(&Value::Bool(false)).unwrap();
        let bytes = first.into_iter().chain(second).collect::<Vec<_>>();
        let mut stream = bytes.as_slice();
        assert_eq!(decode_frame(&mut stream).unwrap(), Value::Bool(true));
        assert_eq!(decode_frame(&mut stream).unwrap(), Value::Bool(false));
        assert!(stream.is_empty());
    }
}
