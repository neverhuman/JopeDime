pub(super) const RECOVERY_SCHEMA: &str = "JopeRecoveryReport.v2";
pub(super) const CONTINUITY_SCHEMA: &str = "JopeRecoveryContinuity.v1";
pub(super) const REQUIRED_GREEN_SECONDS: u64 = 86_400;
pub(super) const MAXIMUM_OBSERVATION_GAP_SECONDS: u64 = 120;

use super::*;

mod component_check;
mod materialize;

pub(super) use component_check::recovery_component_check;
pub(super) use materialize::{materialize_recovery, verify_continuity_observation};

#[derive(Clone, Copy)]
pub(super) struct RecoveryComponent {
    pub(super) name: &'static str,
    pub(super) path: &'static str,
    pub(super) schema: &'static str,
    pub(super) freshness_budget_seconds: u64,
}

pub(super) const RECOVERY_COMPONENTS: &[RecoveryComponent] = &[
    RecoveryComponent {
        name: "controller",
        path: "ops/task-learning-controller.v2.json",
        schema: "TaskLearningControllerStatus.v2",
        freshness_budget_seconds: 120,
    },
    RecoveryComponent {
        name: "ingestion",
        path: "ops/last-ingest.json",
        schema: "JopeSelfplayIngestReceipt.v1",
        freshness_budget_seconds: 300,
    },
    RecoveryComponent {
        name: "queue",
        path: "ops/queue-keeper-v3.json",
        schema: "JopeSelfplayQueueKeeper.v3",
        freshness_budget_seconds: 120,
    },
    RecoveryComponent {
        name: "supply",
        path: "ops/micro-wave-supply-v2.json",
        schema: "JopeMicroWaveSupplyStatus.v2",
        freshness_budget_seconds: 300,
    },
    RecoveryComponent {
        name: "freezer",
        path: "ops/corpus-freezer-v1.json",
        schema: "JopeSelfplayCorpusFreezer.v1",
        freshness_budget_seconds: 120,
    },
    RecoveryComponent {
        name: "trainer",
        path: "ops/task-learning-controller.v2.json",
        schema: "TaskLearningControllerStatus.v2",
        freshness_budget_seconds: 120,
    },
    RecoveryComponent {
        name: "promotion_executor",
        path: "ops/canary-executor.v2.json",
        schema: "CanaryExecutorStatus.v2",
        freshness_budget_seconds: 120,
    },
    RecoveryComponent {
        name: "repair",
        path: "ops/causal-pair-repair-executor.v1.json",
        schema: "CausalPairRepairExecutor.v1",
        freshness_budget_seconds: 120,
    },
    RecoveryComponent {
        name: "causal_evidence",
        path: "ops/causal-evidence-audit.v2.json",
        schema: "CausalEvidenceAudit.v3",
        freshness_budget_seconds: 900,
    },
    RecoveryComponent {
        name: "freshness",
        path: "ops/freshness-status.json",
        schema: "JopeSelfplayFreshnessStatus.v1",
        freshness_budget_seconds: 31 * 86_400,
    },
    RecoveryComponent {
        name: "liveness_watchdog",
        path: "ops/flywheel-liveness.v1.json",
        schema: "FlywheelLiveness.v1",
        freshness_budget_seconds: 120,
    },
    RecoveryComponent {
        name: "effective_unit_audit",
        path: "ops/effective-unit-audit.v1.json",
        schema: "JopeEffectiveUnitAudit.v1",
        freshness_budget_seconds: 86_400,
    },
];

pub(super) fn json_files(directory: &Path) -> Vec<PathBuf> {
    let Ok(entries) = fs::read_dir(directory) else {
        return Vec::new();
    };
    let mut paths = entries
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|value| value.to_str()) == Some("json"))
        .collect::<Vec<_>>();
    paths.sort();
    paths
}

pub(super) fn collect_named(directory: &Path, suffix: &str, output: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(directory) else {
        return;
    };
    for entry in entries.filter_map(|entry| entry.ok()) {
        let path = entry.path();
        if path.is_dir() {
            collect_named(&path, suffix, output);
        } else if path
            .file_name()
            .and_then(|value| value.to_str())
            .is_some_and(|name| name.ends_with(suffix))
        {
            output.push(path);
        }
    }
}

pub(super) fn read_json(path: &Path) -> Result<Value> {
    Ok(serde_json::from_reader(std::io::BufReader::new(
        File::open(path).with_context(|| format!("open {}", path.display()))?,
    ))?)
}

pub(super) fn read_optional_json(path: &Path) -> Result<Value> {
    let file = match File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(json!({})),
        Err(error) => return Err(error).with_context(|| format!("open {}", path.display())),
    };
    serde_json::from_reader(std::io::BufReader::new(file))
        .with_context(|| format!("parse {}", path.display()))
}

pub(super) fn native_timestamp(value: &Value) -> Option<u128> {
    [
        "heartbeat_at_ms",
        "last_heartbeat_at_ms",
        "generated_at_ms",
        "updated_at_ms",
        "activated_at_ms",
    ]
    .into_iter()
    .find_map(|key| value.get(key).and_then(Value::as_u64).map(u128::from))
}

pub(super) fn useful_queue_depth(value: &Value) -> u64 {
    value
        .get("useful_work")
        .and_then(Value::as_u64)
        .unwrap_or_else(|| {
            ["queued", "starting", "dispatching", "validated_ready"]
                .into_iter()
                .filter_map(|key| value.get(key).and_then(Value::as_u64))
                .sum()
        })
}
