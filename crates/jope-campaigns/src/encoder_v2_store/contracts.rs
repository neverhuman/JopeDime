use super::*;

pub const EXPANDED_CACHE_SCHEMA: &str = "JopeExpandedEncodingCache.v2";
pub const EXPANDED_POINTER_SCHEMA: &str = "JopeExpandedCachePointer.v2";
pub const CONTEXT_BINDING_SCHEMA: &str = "JopeContextTransformBinding.v2";
pub const CONFLICT_SCHEMA: &str = "JopeEncoderNondeterminismConflict.v1";
pub const GC_ROOT_SET_SCHEMA: &str = "JopeEncoderV2GcRootSet.v1";

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EncoderV2GcRootSetV1 {
    pub schema: String,
    pub generated_at_unix_ms: u64,
    pub freshness_budget_seconds: u64,
    pub phase6_mark_only: bool,
    pub binding_sha256: Vec<String>,
    pub context_transform_sha256: Vec<String>,
    pub benchmark_plan_sha256: Vec<String>,
    pub lift_manifest_sha256: Vec<String>,
    pub quarantine_sha256: String,
    pub rollback_evidence_sha256: String,
}

impl EncoderV2GcRootSetV1 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == GC_ROOT_SET_SCHEMA,
            "invalid Encoder V2 GC-root schema"
        );
        ensure!(
            self.generated_at_unix_ms > 0 && self.freshness_budget_seconds > 0,
            "Encoder V2 GC roots lack freshness metadata"
        );
        ensure!(
            self.phase6_mark_only,
            "Encoder V2 GC is not mark-only in Phase 6"
        );
        for (values, field) in [
            (&self.binding_sha256, "binding"),
            (&self.context_transform_sha256, "context transform"),
            (&self.benchmark_plan_sha256, "benchmark plan"),
            (&self.lift_manifest_sha256, "lift manifest"),
        ] {
            ensure!(
                values.windows(2).all(|pair| pair[0] < pair[1]),
                "Encoder V2 GC {field} roots are not unique and sorted"
            );
            ensure!(
                values.iter().all(|value| is_sha256(value)),
                "Encoder V2 GC {field} root is invalid"
            );
        }
        ensure!(
            is_sha256(&self.quarantine_sha256) && is_sha256(&self.rollback_evidence_sha256),
            "Encoder V2 GC does not bind quarantine and rollback roots"
        );
        Ok(())
    }

    pub fn permits_sweep(&self) -> bool {
        false
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExpandedEncodingCacheV2 {
    pub schema: String,
    pub cache_key_sha256: String,
    pub encoder_identity_sha256: String,
    pub input_identity_sha256: String,
    pub output_sha256: String,
    pub output_size_bytes: u64,
    pub object_relative_path: String,
    pub execution_receipt_sha256: String,
    pub scq_job_identity_sha256: String,
}

impl ExpandedEncodingCacheV2 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == EXPANDED_CACHE_SCHEMA,
            "invalid V2 cache schema"
        );
        for (value, field) in [
            (&self.cache_key_sha256, "cache key"),
            (&self.encoder_identity_sha256, "encoder identity"),
            (&self.input_identity_sha256, "input identity"),
            (&self.output_sha256, "output"),
            (&self.execution_receipt_sha256, "execution receipt"),
            (&self.scq_job_identity_sha256, "SCQ job identity"),
        ] {
            ensure!(is_sha256(value), "{field} is not a lowercase SHA-256");
        }
        ensure!(
            self.output_size_bytes == OUTPUT_BYTES,
            "V2 cache output size drift"
        );
        ensure!(
            self.object_relative_path == object_relative_path(&self.output_sha256),
            "V2 cache object path is noncanonical"
        );
        Ok(())
    }

    pub fn digest(&self) -> Result<String> {
        self.validate()?;
        canonical_json_sha256(&serde_json::to_value(self)?)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExpandedCachePointerV2 {
    pub schema: String,
    pub encoder_identity_sha256: String,
    pub input_identity_sha256: String,
    pub cache_key_sha256: String,
    pub binding_sha256: String,
    pub binding_relative_path: String,
    pub output_sha256: String,
}

impl ExpandedCachePointerV2 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == EXPANDED_POINTER_SCHEMA,
            "invalid V2 cache pointer schema"
        );
        for value in [
            &self.encoder_identity_sha256,
            &self.input_identity_sha256,
            &self.cache_key_sha256,
            &self.binding_sha256,
            &self.output_sha256,
        ] {
            ensure!(is_sha256(value), "V2 pointer has an invalid digest");
        }
        ensure!(
            self.binding_relative_path == binding_relative_path(&self.cache_key_sha256),
            "V2 pointer binding path is noncanonical"
        );
        Ok(())
    }
}
