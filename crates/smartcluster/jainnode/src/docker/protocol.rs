use super::*;

#[derive(Deserialize)]
pub(super) struct CreateResponse {
    #[serde(rename = "Id")]
    pub(super) id: String,
}

#[derive(Deserialize)]
pub(super) struct WaitResponse {
    #[serde(rename = "StatusCode")]
    pub(super) status_code: i32,
}

#[derive(Deserialize)]
pub(super) struct InspectResponse {
    #[serde(rename = "State")]
    pub(super) state: InspectState,
}

#[derive(Deserialize)]
pub(super) struct InspectState {
    #[serde(rename = "OOMKilled")]
    pub(super) oom_killed: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum HttpBodyFraming {
    Chunked,
    ContentLength(usize),
    Empty,
}

pub(super) fn decode_http_body(
    header: &str,
    body: &[u8],
    status: u16,
) -> Result<Vec<u8>, DockerError> {
    match parse_http_body_framing(header, status)? {
        HttpBodyFraming::Chunked => decode_chunked(body),
        HttpBodyFraming::ContentLength(length) if length == body.len() => Ok(body.to_vec()),
        HttpBodyFraming::ContentLength(_) => Err(DockerError::MalformedResponse),
        HttpBodyFraming::Empty if body.is_empty() => Ok(Vec::new()),
        HttpBodyFraming::Empty => Err(DockerError::MalformedResponse),
    }
}

pub(super) fn parse_http_body_framing(
    header: &str,
    status: u16,
) -> Result<HttpBodyFraming, DockerError> {
    let mut transfer_encoding = None;
    let mut content_length = None;
    let mut lines = header.split("\r\n");
    let status_line = lines.next().ok_or(DockerError::MalformedResponse)?;
    if status_line.is_empty() || status_line.contains(['\r', '\n']) {
        return Err(DockerError::MalformedResponse);
    }
    for line in lines {
        if line.is_empty() || line.contains(['\r', '\n']) {
            return Err(DockerError::MalformedResponse);
        }
        let (name, value) = line.split_once(':').ok_or(DockerError::MalformedResponse)?;
        if name.is_empty()
            || !name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
        {
            return Err(DockerError::MalformedResponse);
        }
        let value = value.trim();
        if name.eq_ignore_ascii_case("transfer-encoding") {
            if transfer_encoding.replace(value).is_some() {
                return Err(DockerError::MalformedResponse);
            }
        } else if name.eq_ignore_ascii_case("content-length") {
            if content_length.is_some()
                || value.is_empty()
                || !value.bytes().all(|b| b.is_ascii_digit())
            {
                return Err(DockerError::MalformedResponse);
            }
            content_length = Some(
                value
                    .parse::<usize>()
                    .map_err(|_| DockerError::MalformedResponse)?,
            );
            if content_length.is_some_and(|length| length > MAX_DOCKER_RESPONSE) {
                return Err(DockerError::OversizedResponse);
            }
        }
    }
    let framing = match (transfer_encoding, content_length) {
        (Some(_), Some(_)) => Err(DockerError::MalformedResponse),
        (Some(value), None) if value.eq_ignore_ascii_case("chunked") => {
            Ok(HttpBodyFraming::Chunked)
        }
        (Some(_), None) => Err(DockerError::MalformedResponse),
        (None, Some(0)) if matches!(status, 204 | 304) => Ok(HttpBodyFraming::Empty),
        (None, Some(_)) if matches!(status, 204 | 304) => Err(DockerError::MalformedResponse),
        (None, Some(length)) => Ok(HttpBodyFraming::ContentLength(length)),
        (None, None) if matches!(status, 204 | 304) => Ok(HttpBodyFraming::Empty),
        (None, None) => Err(DockerError::MalformedResponse),
    }?;
    Ok(framing)
}

pub(super) fn decode_chunked(mut body: &[u8]) -> Result<Vec<u8>, DockerError> {
    let mut output = Vec::new();
    loop {
        let line_end = find_subsequence(body, b"\r\n").ok_or(DockerError::MalformedResponse)?;
        let size_text = std::str::from_utf8(&body[..line_end])
            .map_err(|_| DockerError::MalformedResponse)?
            .split(';')
            .next()
            .ok_or(DockerError::MalformedResponse)?;
        let size =
            usize::from_str_radix(size_text, 16).map_err(|_| DockerError::MalformedResponse)?;
        body = &body[line_end + 2..];
        if size == 0 {
            return (body == b"\r\n")
                .then_some(output)
                .ok_or(DockerError::MalformedResponse);
        }
        let framed = size.checked_add(2).ok_or(DockerError::OversizedResponse)?;
        if body.len() < framed || &body[size..framed] != b"\r\n" {
            return Err(DockerError::MalformedResponse);
        }
        output.extend_from_slice(&body[..size]);
        if output.len() > MAX_DOCKER_RESPONSE {
            return Err(DockerError::OversizedResponse);
        }
        body = &body[size + 2..];
    }
}

async fn read_stream_header(
    stream: &mut UnixStream,
    timeout: Duration,
) -> Result<(String, Vec<u8>), DockerError> {
    let mut response = Vec::new();
    let mut buffer = [0_u8; 4096];
    loop {
        let count = tokio::time::timeout(timeout, stream.read(&mut buffer))
            .await
            .map_err(|_| DockerError::Transport("Docker log header timed out".to_owned()))?
            .map_err(|error| DockerError::Transport(error.to_string()))?;
        if count == 0 {
            return Err(DockerError::MalformedResponse);
        }
        response.extend_from_slice(&buffer[..count]);
        if response.len() > MAX_HEADER_BYTES + 64 * 1024 {
            return Err(DockerError::OversizedResponse);
        }
        if let Some(header_end) = find_subsequence(&response, b"\r\n\r\n") {
            if header_end > MAX_HEADER_BYTES {
                return Err(DockerError::OversizedResponse);
            }
            let header = std::str::from_utf8(&response[..header_end])
                .map_err(|_| DockerError::MalformedResponse)?
                .to_owned();
            let status = header
                .lines()
                .next()
                .and_then(|line| line.split_ascii_whitespace().nth(1))
                .and_then(|status| status.parse::<u16>().ok())
                .ok_or(DockerError::MalformedResponse)?;
            if status != 200 {
                return Err(DockerError::Rejected(status));
            }
            return Ok((header, response[header_end + 4..].to_vec()));
        }
    }
}

pub(super) fn drain_chunked(wire: &mut Vec<u8>, output: &mut Vec<u8>) -> Result<bool, DockerError> {
    loop {
        let Some(line_end) = find_subsequence(wire, b"\r\n") else {
            return Ok(false);
        };
        let size_text = std::str::from_utf8(&wire[..line_end])
            .map_err(|_| DockerError::MalformedResponse)?
            .split(';')
            .next()
            .ok_or(DockerError::MalformedResponse)?;
        let size =
            usize::from_str_radix(size_text, 16).map_err(|_| DockerError::MalformedResponse)?;
        if size == 0 {
            let terminator_end = line_end
                .checked_add(4)
                .ok_or(DockerError::OversizedResponse)?;
            if wire.len() < terminator_end {
                return Ok(false);
            }
            if wire.len() != terminator_end || &wire[line_end + 2..terminator_end] != b"\r\n" {
                return Err(DockerError::MalformedResponse);
            }
            wire.clear();
            return Ok(true);
        }
        let payload_start = line_end + 2;
        let frame_end = payload_start
            .checked_add(size)
            .ok_or(DockerError::OversizedResponse)?;
        if size > MAX_DOCKER_RESPONSE {
            return Err(DockerError::OversizedResponse);
        }
        if wire.len() < frame_end + 2 {
            return Ok(false);
        }
        if &wire[frame_end..frame_end + 2] != b"\r\n" {
            return Err(DockerError::MalformedResponse);
        }
        output.extend_from_slice(&wire[payload_start..frame_end]);
        if output.len() > MAX_DOCKER_RESPONSE {
            return Err(DockerError::OversizedResponse);
        }
        wire.drain(..frame_end + 2);
    }
}

pub(super) fn drain_content_length(
    wire: &mut Vec<u8>,
    output: &mut Vec<u8>,
    remaining: &mut usize,
) -> Result<bool, DockerError> {
    if wire.len() > *remaining {
        return Err(DockerError::MalformedResponse);
    }
    *remaining -= wire.len();
    output.append(wire);
    Ok(*remaining == 0)
}

pub(super) fn take_multiplexed_logs(
    body: &mut Vec<u8>,
) -> Result<Vec<DockerLogChunk>, DockerError> {
    let mut chunks = Vec::new();
    loop {
        if body.len() < 8 {
            return Ok(chunks);
        }
        let stream = body[0];
        if !matches!(stream, 1 | 2) || body[1..4] != [0, 0, 0] {
            return Err(DockerError::MalformedResponse);
        }
        let length = u32::from_be_bytes([body[4], body[5], body[6], body[7]]) as usize;
        if length > 64 * 1024 {
            return Err(DockerError::MalformedResponse);
        }
        if body.len() < 8 + length {
            return Ok(chunks);
        }
        chunks.push(DockerLogChunk {
            stderr: stream == 2,
            bytes: body[8..8 + length].to_vec(),
        });
        body.drain(..8 + length);
    }
}

pub(super) fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

pub(crate) fn percent_encode(value: &str) -> String {
    let mut encoded = String::new();
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.') {
            encoded.push(char::from(byte));
        } else {
            encoded.push_str(&format!("%{byte:02X}"));
        }
    }
    encoded
}
