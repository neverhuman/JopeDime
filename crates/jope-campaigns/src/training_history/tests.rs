use super::*;
use tempfile::TempDir;

fn row(epoch: i64, train: f64, val: f64) -> String {
    format!(
            "{{\"epoch\":{epoch},\"phase\":\"A\",\"train_loss\":{train},\"val_loss\":{val},\"model_version\":\"policy_v3.3-fused-heads\"}}"
        )
}

fn key() -> SegmentKey {
    SegmentKey {
        run_id: "jope-selfplay-train-001805437-a-g9".into(),
        attempt: "1".into(),
        watermark: "watermark-001805437".into(),
    }
}

#[path = "tests/ingest.rs"]
mod ingest;

#[path = "tests/remote.rs"]
mod remote;
