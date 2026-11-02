use super::*;
use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReadPage {
    /// Identifier of the journal run.
    pub run_id: String,
    /// Validated records returned by this page.
    pub events: Vec<Envelope>,
    /// Cursor to pass as `after` for the next page, if more records may exist.
    pub next: Option<Cursor>,
    /// Whether the file ended in an incomplete trailing record.
    pub partial_tail: bool,
}

/// Bounded offline reader. Live services maintain their own validated offset index using
/// `Envelope::validate`, so polling a stream never requires rescanning historical records.
/// This helper validates through the returned page; the terminal watermark count/hash
/// is checked only when the read reaches that watermark. Use the observatory index
/// for full committed-prefix validation before serving a finite page.
pub fn read_journal(
    path: &Path,
    after: Option<&Cursor>,
    limit: usize,
    max_bytes: usize,
) -> Result<ReadPage> {
    ensure!(
        (1..=1024).contains(&limit) && (1..=64 * 1024 * 1024).contains(&max_bytes),
        "invalid page limits"
    );
    let after_seq = after
        .map(|c| decimal_u64(&c.sequence))
        .transpose()?
        .unwrap_or(0);
    let committed = read_commit(path)?;
    let committed_bytes = decimal_u64(&committed.bytes)?;
    let file = File::open(path)?;
    let observed = file.metadata()?.len();
    ensure!(
        observed >= committed_bytes,
        "journal truncated below watermark"
    );
    use std::io::Read;
    let mut reader = BufReader::new(file.take(committed_bytes));
    let mut run_id = committed.run_id.clone();
    if let Some(cursor) = after {
        ensure!(cursor.run_id == run_id, "cursor run mismatch");
    }
    let mut expected = 1u64;
    let mut previous = String::new();
    let mut events = Vec::new();
    let mut bytes = 0usize;
    let partial_tail = observed > committed_bytes;
    loop {
        let Some((line, complete)) = bounded_line(&mut reader, MAX_FRAME_BYTES)? else {
            break;
        };
        ensure!(complete, "committed record lacks newline");
        let record: Envelope =
            serde_json::from_slice(&line).context("invalid complete narration record")?;
        if run_id.is_empty() {
            run_id = record.run_id.clone();
            if let Some(cursor) = after {
                ensure!(cursor.run_id == run_id, "cursor run mismatch");
            }
        }
        record.validate(&previous, expected, &run_id)?;
        previous = record.sha256.clone();
        expected = expected.checked_add(1).context("sequence exhausted")?;
        if expected - 1 > after_seq {
            if line.len() > max_bytes.saturating_sub(bytes) {
                ensure!(!events.is_empty(), "page budget smaller than next record");
                break;
            }
            bytes += line.len();
            events.push(record);
            if events.len() >= limit {
                break;
            }
        }
    }
    if reader.fill_buf()?.is_empty() {
        ensure!(
            expected - 1 == decimal_u64(&committed.records)?,
            "watermark record count mismatch"
        );
        ensure!(previous == committed.sha256, "watermark hash mismatch");
    }
    ensure!(after_seq < expected, "future cursor");
    if run_id.is_empty() {
        if let Some(cursor) = after {
            ensure!(cursor.sequence == "0", "future cursor");
            run_id = cursor.run_id.clone();
        }
    }
    let next = events
        .last()
        .map(|e| e.cursor.clone())
        .or_else(|| after.cloned());
    Ok(ReadPage {
        run_id,
        events,
        next,
        partial_tail,
    })
}

fn bounded_line(reader: &mut impl BufRead, max: usize) -> Result<Option<(Vec<u8>, bool)>> {
    let mut line = Vec::new();
    loop {
        let buffer = reader.fill_buf()?;
        if buffer.is_empty() {
            return Ok(if line.is_empty() {
                None
            } else {
                Some((line, false))
            });
        }
        let end = buffer.iter().position(|b| *b == b'\n');
        let take = end.map(|p| p + 1).unwrap_or(buffer.len());
        ensure!(
            take <= max.saturating_add(1).saturating_sub(line.len()),
            "narration frame exceeds limit"
        );
        line.extend_from_slice(&buffer[..take]);
        reader.consume(take);
        if end.is_some() {
            line.pop();
            return Ok(Some((line, true)));
        }
    }
}
