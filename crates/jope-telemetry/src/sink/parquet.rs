//! Deterministic bounded-memory rollup from the append-only event JSONL source.

use anyhow::{anyhow, Context, Result};
use arrow_array::{ArrayRef, Float64Array, RecordBatch, StringArray, UInt16Array, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use serde::Deserialize;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;

const BATCH_ROWS: usize = 4096;
const MAX_JSONL_LINE: usize = 16 * 1024;

#[derive(Deserialize)]
struct EventLine {
    kind: String,
    #[serde(default)]
    event: Option<EventPayload>,
}

#[derive(Deserialize)]
struct EventPayload {
    run_id: String,
    producer_sequence: u64,
    monotonic_ns: u64,
    step: u64,
    source_id: u16,
    metric_id: u16,
    group_id: u16,
    flags: u8,
    value: Option<f64>,
    nonfinite: Option<String>,
    availability_reason_id: Option<u16>,
}

struct Columns {
    run_id: Vec<String>,
    producer_sequence: Vec<u64>,
    monotonic_ns: Vec<u64>,
    step: Vec<u64>,
    source_id: Vec<u16>,
    metric_id: Vec<u16>,
    group_id: Vec<u16>,
    flags: Vec<u16>,
    value: Vec<Option<f64>>,
    nonfinite: Vec<Option<String>>,
    availability_reason_id: Vec<Option<u16>>,
}

impl Columns {
    fn new() -> Self {
        Self {
            run_id: Vec::with_capacity(BATCH_ROWS),
            producer_sequence: Vec::with_capacity(BATCH_ROWS),
            monotonic_ns: Vec::with_capacity(BATCH_ROWS),
            step: Vec::with_capacity(BATCH_ROWS),
            source_id: Vec::with_capacity(BATCH_ROWS),
            metric_id: Vec::with_capacity(BATCH_ROWS),
            group_id: Vec::with_capacity(BATCH_ROWS),
            flags: Vec::with_capacity(BATCH_ROWS),
            value: Vec::with_capacity(BATCH_ROWS),
            nonfinite: Vec::with_capacity(BATCH_ROWS),
            availability_reason_id: Vec::with_capacity(BATCH_ROWS),
        }
    }

    fn push(&mut self, event: EventPayload) {
        self.run_id.push(event.run_id);
        self.producer_sequence.push(event.producer_sequence);
        self.monotonic_ns.push(event.monotonic_ns);
        self.step.push(event.step);
        self.source_id.push(event.source_id);
        self.metric_id.push(event.metric_id);
        self.group_id.push(event.group_id);
        self.flags.push(event.flags as u16);
        self.value.push(event.value);
        self.nonfinite.push(event.nonfinite);
        self.availability_reason_id
            .push(event.availability_reason_id);
    }

    fn len(&self) -> usize {
        self.step.len()
    }

    fn write(&mut self, writer: &mut ArrowWriter<File>) -> Result<()> {
        if self.len() == 0 {
            return Ok(());
        }
        let schema = schema();
        let arrays: Vec<ArrayRef> = vec![
            Arc::new(StringArray::from(std::mem::take(&mut self.run_id))),
            Arc::new(UInt64Array::from(std::mem::take(
                &mut self.producer_sequence,
            ))),
            Arc::new(UInt64Array::from(std::mem::take(&mut self.monotonic_ns))),
            Arc::new(UInt64Array::from(std::mem::take(&mut self.step))),
            Arc::new(UInt16Array::from(std::mem::take(&mut self.source_id))),
            Arc::new(UInt16Array::from(std::mem::take(&mut self.metric_id))),
            Arc::new(UInt16Array::from(std::mem::take(&mut self.group_id))),
            Arc::new(UInt16Array::from(std::mem::take(&mut self.flags))),
            Arc::new(Float64Array::from(std::mem::take(&mut self.value))),
            Arc::new(StringArray::from(std::mem::take(&mut self.nonfinite))),
            Arc::new(UInt16Array::from(std::mem::take(
                &mut self.availability_reason_id,
            ))),
        ];
        writer.write(&RecordBatch::try_new(schema, arrays)?)?;
        *self = Self::new();
        Ok(())
    }
}

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("run_id", DataType::Utf8, false),
        Field::new("producer_sequence", DataType::UInt64, false),
        Field::new("monotonic_ns", DataType::UInt64, false),
        Field::new("step", DataType::UInt64, false),
        Field::new("source_id", DataType::UInt16, false),
        Field::new("metric_id", DataType::UInt16, false),
        Field::new("group_id", DataType::UInt16, false),
        Field::new("flags", DataType::UInt16, false),
        Field::new("value", DataType::Float64, true),
        Field::new("nonfinite", DataType::Utf8, true),
        Field::new("availability_reason_id", DataType::UInt16, true),
    ]))
}

/// Regenerate a sibling temporary and atomically replace the destination on success.
pub fn regenerate(jsonl_path: &Path, parquet_path: &Path) -> Result<u64> {
    let parent = parquet_path
        .parent()
        .ok_or_else(|| anyhow!("Parquet path has no parent"))?;
    let file_name = parquet_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("metrics.parquet");
    let temp = parent.join(format!(".{file_name}.{}.tmp", std::process::id()));
    let result = regenerate_inner(jsonl_path, &temp);
    match result {
        Ok(rows) => {
            fs::rename(&temp, parquet_path).context("atomically replace Parquet rollup")?;
            if let Ok(dir) = File::open(parent) {
                let _ = dir.sync_all();
            }
            Ok(rows)
        }
        Err(error) => {
            let _ = fs::remove_file(&temp);
            Err(error)
        }
    }
}

fn regenerate_inner(jsonl_path: &Path, temp: &Path) -> Result<u64> {
    let schema = schema();
    let props = WriterProperties::builder().build();
    let file = File::create(temp).context("create temporary Parquet rollup")?;
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    let file = File::open(jsonl_path).with_context(|| format!("open {}", jsonl_path.display()))?;
    let mut reader = BufReader::new(file);
    let mut line = Vec::with_capacity(1024);
    let mut columns = Columns::new();
    let mut rows = 0_u64;
    loop {
        line.clear();
        let read = reader.read_until(b'\n', &mut line)?;
        if read == 0 {
            break;
        }
        if line.len() > MAX_JSONL_LINE {
            return Err(anyhow!(
                "JSONL event exceeds {MAX_JSONL_LINE}-byte line cap"
            ));
        }
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        let record: EventLine = match serde_json::from_slice(&line) {
            Ok(record) => record,
            Err(_) if line.last() != Some(&b'\n') => break,
            Err(error) => return Err(error).context("parse complete JSONL line"),
        };
        if record.kind == "event" {
            if let Some(event) = record.event {
                if event.run_id.len() != 32
                    || !event
                        .run_id
                        .bytes()
                        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
                {
                    return Err(anyhow!(
                        "event run_id must be 32 lowercase hexadecimal bytes"
                    ));
                }
                columns.push(event);
                rows += 1;
                if columns.len() >= BATCH_ROWS {
                    columns.write(&mut writer)?;
                }
            }
        }
    }
    columns.write(&mut writer)?;
    writer.close()?;
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::Event;
    use crate::sink::jsonl::JsonlWriter;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    #[test]
    fn rollup_is_readable_and_ignores_low_rate_records() {
        let dir = tempfile::tempdir().unwrap();
        let mut jsonl = JsonlWriter::create(dir.path()).unwrap();
        jsonl
            .append_event(&Event::scalar([1; 16], 0, 1, 1, 1, 0x0101, 0, 1.5, false))
            .unwrap();
        jsonl
            .append_record_line(b"{\"kind\":\"test/v1\",\"body\":{}}\n")
            .unwrap();
        jsonl.flush().unwrap();
        drop(jsonl);
        let path = dir.path().join("metrics.parquet");
        assert_eq!(
            regenerate(&dir.path().join("events.jsonl"), &path).unwrap(),
            1
        );
        let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(path).unwrap())
            .unwrap()
            .build()
            .unwrap();
        let batches: Vec<_> = reader.map(Result::unwrap).collect();
        assert_eq!(
            batches.iter().map(|batch| batch.num_rows()).sum::<usize>(),
            1
        );
        assert_eq!(batches[0].num_columns(), 11);
    }
}
