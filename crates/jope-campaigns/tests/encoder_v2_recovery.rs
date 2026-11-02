use anyhow::bail;
use jope_dime_regressor::encoder_v2_authority::{
    eight_slot_sampling_weights, BenchmarkLaneMetrics, EncoderBenchmarkReportV2,
    EncoderLiftReportV2, LiftTaskResult, OptionalTaskReadinessV2, OptionalTaskRow,
    BENCHMARK_REPORT_SCHEMA, LIFT_REPORT_SCHEMA, OPTIONAL_READINESS_SCHEMA,
};
use jope_dime_regressor::encoder_v2_capacity::{
    CapacityDecision, EncoderCapacityObservation, CAPACITY_OBSERVATION_SCHEMA,
};
use jope_dime_regressor::encoder_v2_execution::{
    validate_output_bytes, EncoderAttemptReceiptV2, EncoderExecutionReceiptV2, EncoderStageTiming,
    ObservedEncoderRuntime, ATTEMPT_SCHEMA, EXECUTION_SCHEMA,
};
use jope_dime_regressor::encoder_v2_identity::{
    cache_key, ArtifactIdentity, BuildIdentity, DatasetReadIdentity, FoundationDatasetInputV2,
    FoundationEncoderIdentityV2, NumericalRuntimeIdentity, OutputContractIdentity,
    RuntimePackageIdentity, DATASET_INPUT_SCHEMA, ENCODER_IDENTITY_SCHEMA, OUTPUT_BYTES,
    OUTPUT_BYTE_ORDER, OUTPUT_DTYPE, OUTPUT_FEATURES, OUTPUT_LAYOUT,
};
use jope_dime_regressor::encoder_v2_store::{EncoderV2Store, PublishOutcome, PublishStep};
use jope_dime_regressor::encoder_v2_worker::{
    plan_chunks, ChunkMode, EncoderWorkItem, ResidentWorkerProgress,
};
use jope_dime_regressor::paths;
use jope_dime_regressor::recovery_contracts::{
    immutable_flywheel_root_str, relocate_v1_quarantine, runtime_current_root_str,
    MutableScriptObservation, QuarantineFileIdentity, QuarantinePolicyFlags,
    QuarantinedEncodingTree, ReferenceExclusionProof, RollbackCapsuleV1, RollbackNodeReceipt,
    RollbackUnitIdentity, V1EncodingQuarantineV1, REQUIRED_NODES, ROLLBACK_CAPSULE_SCHEMA,
    ROLLBACK_RUNTIME_BUNDLE_SHA256, V1_QUARANTINE_SCHEMA,
};
use jope_dime_regressor::selfplay_data::canonical_json_sha256;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::os::unix::fs::PermissionsExt;

#[path = "encoder_v2_recovery/mod.rs"]
mod encoder_v2_recovery;
