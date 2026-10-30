//! Run identity, metric registry and atomic status manifest.

use crate::alarm::utc_now_rfc3339;
use crate::event::EVENT_SCHEMA_VERSION;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

/// Availability reason ids, as listed in every manifest's `availability_reasons`.
pub const REASON_NO_S3_PATH: u16 = 3;
pub const REASON_NO_OBJECT_CACHE: u16 = 4;
pub const REASON_NO_NVML_OR_GPU: u16 = 7;
pub const REASON_PROCFS_UNAVAILABLE: u16 = 8;
pub const REASON_NO_TEXT_TOKEN_PATH: u16 = 10;
pub const REASON_STARTUP_READS_NOT_OBSERVED: u16 = 11;
pub const REASON_CUDA_EVENT_UNAVAILABLE: u16 = 12;
use ulid::Ulid;

pub const MANIFEST_SCHEMA: &str = "jope.telemetry.manifest/v1";
pub const MAX_MANIFEST_BYTES: usize = 1 << 20;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SourceDef {
    pub id: u16,
    pub name: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Running,
    Complete,
    Interrupted,
    SinkFailed,
}

#[derive(Clone, Debug, Serialize)]
pub struct RunManifest {
    pub schema: String,
    pub event_schema_version: u8,
    pub run_id: String,
    pub start_utc: String,
    pub end_utc: Option<String>,
    pub status: RunStatus,
    pub sources: Vec<SourceDef>,
    pub metrics: Vec<crate::train::MetricDef>,
    pub metric_groups: BTreeMap<u16, BTreeMap<u16, String>>,
    pub metadata: BTreeMap<String, Value>,
    pub requested_sampling: BTreeMap<String, Value>,
    pub effective_sampling: BTreeMap<String, Value>,
    pub availability_reasons: BTreeMap<u16, String>,
    pub event_queue_capacity: usize,
    pub event_queue_producers: usize,
    pub record_queue_capacity: usize,
    pub max_record_bytes: usize,
    pub events_enqueued: u64,
    pub events_written: u64,
    pub records_written: u64,
    pub event_drops: u64,
    pub late_event_drops: u64,
    pub event_queue_high_water: u64,
    pub record_drops: u64,
    pub oversized_record_drops: u64,
    pub sink_errors: u64,
    pub recovery_truncated_bytes: u64,
    pub sink_error: Option<String>,
}

#[derive(Clone, Debug)]
pub struct RunMetadata {
    pub metadata: BTreeMap<String, Value>,
    pub requested_sampling: BTreeMap<String, Value>,
    pub sources: Vec<SourceDef>,
    pub metric_groups: BTreeMap<u16, BTreeMap<u16, String>>,
}

impl Default for RunMetadata {
    fn default() -> Self {
        let sources = vec![SourceDef {
            id: 1,
            name: "jope-telemetry".to_owned(),
        }];
        Self {
            metadata: BTreeMap::new(),
            requested_sampling: BTreeMap::new(),
            sources,
            metric_groups: BTreeMap::new(),
        }
    }
}

impl RunManifest {
    pub fn new(run_id: Ulid, metadata: RunMetadata) -> Self {
        Self {
            schema: MANIFEST_SCHEMA.to_owned(),
            event_schema_version: EVENT_SCHEMA_VERSION,
            run_id: run_id.to_string(),
            start_utc: utc_now_rfc3339(),
            end_utc: None,
            status: RunStatus::Running,
            sources: metadata.sources,
            metrics: crate::train::METRICS.to_vec(),
            metric_groups: metadata.metric_groups,
            metadata: metadata.metadata,
            requested_sampling: metadata.requested_sampling,
            effective_sampling: BTreeMap::new(),
            availability_reasons: BTreeMap::from([
                (1, "not_host_visible_without_extra_transfer".to_owned()),
                (2, "no_prediction_signatures".to_owned()),
                (3, "no_active_s3_get_client".to_owned()),
                (4, "no_explicit_object_cache".to_owned()),
                (5, "no_data_loader_or_pin_path".to_owned()),
                (6, "no_fused_dope_trainer".to_owned()),
                (7, "no_nvml_or_no_visible_gpu".to_owned()),
                (8, "host_procfs_unavailable".to_owned()),
                (9, "nvml_unavailable".to_owned()),
                (10, "no_text_token_path".to_owned()),
                (11, "startup_reads_not_observed".to_owned()),
                (12, "cuda_event_unavailable".to_owned()),
                (13, "no_family_grouped_validation_producer".to_owned()),
            ]),
            event_queue_capacity: 0,
            event_queue_producers: 0,
            record_queue_capacity: 0,
            max_record_bytes: 0,
            events_enqueued: 0,
            events_written: 0,
            records_written: 0,
            event_drops: 0,
            late_event_drops: 0,
            event_queue_high_water: 0,
            record_drops: 0,
            oversized_record_drops: 0,
            sink_errors: 0,
            recovery_truncated_bytes: 0,
            sink_error: None,
        }
    }

    pub fn finalize(&mut self, status: RunStatus) {
        self.status = status;
        self.end_utc = Some(utc_now_rfc3339());
    }

    pub fn write_atomic(&self, path: &Path) -> io::Result<()> {
        let parent = path
            .parent()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "manifest has no parent"))?;
        let temp = temp_sibling(path);
        let result = (|| {
            let mut file = File::create(&temp)?;
            {
                let mut capped = CappedWriter {
                    inner: &mut file,
                    written: 0,
                };
                serde_json::to_writer_pretty(&mut capped, self).map_err(io::Error::other)?;
                capped.write_all(b"\n")?;
            }
            file.sync_all()?;
            fs::rename(&temp, path)?;
            if let Ok(dir) = File::open(parent) {
                let _ = dir.sync_all();
            }
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(&temp);
        }
        result
    }
}

struct CappedWriter<'a> {
    inner: &'a mut File,
    written: usize,
}

impl Write for CappedWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.written.saturating_add(bytes.len()) > MAX_MANIFEST_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "telemetry manifest exceeds 1 MiB registry cap",
            ));
        }
        let written = self.inner.write(bytes)?;
        self.written += written;
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

fn temp_sibling(path: &Path) -> PathBuf {
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("manifest.json");
    path.with_file_name(format!(".{name}.{}.tmp", std::process::id()))
}
