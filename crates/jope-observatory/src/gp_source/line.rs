use super::prelude::*;
use super::MAX_LINE;

/// Read a bounded complete line. A producer's in-progress final record waits for completion.
pub fn complete_line<R: BufRead + Seek>(
    reader: &mut R,
    allow_partial: bool,
) -> Result<Option<Vec<u8>>> {
    let position = reader.stream_position()?;
    let mut bytes = Vec::new();
    reader
        .take(MAX_LINE as u64 + 1)
        .read_until(b'\n', &mut bytes)?;
    ensure!(
        bytes.len() <= MAX_LINE,
        "source journal/event line exceeds limit"
    );
    if bytes.is_empty() {
        return Ok(None);
    }
    if bytes.last() != Some(&b'\n') {
        ensure!(
            allow_partial,
            "incomplete event log; preserve original before recovery"
        );
        reader.seek(SeekFrom::Start(position))?;
        return Ok(None);
    }
    Ok(Some(bytes))
}
