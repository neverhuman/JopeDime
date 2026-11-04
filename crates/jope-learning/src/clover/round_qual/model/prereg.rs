use super::*;

/// Schema identifier for the offline self-play round qualification report.
pub const ROUND_QUAL_SCHEMA: &str = "jope.selfplay-round-qualification/v1";

/// Expected schema identifier for per-context self-play ratchet outputs.
pub const RATCHET_SCHEMA: &str = "clover.selfplay-ratchet/v1";

/// Expected schema identifier for the Round-1 pre-registration packet.
pub const PREREG_SCHEMA: &str = "jope.selfplay-round1-prereg/v1";

/// Expected `endpoints.secondary_status` disclosure string in `jope.selfplay-round1-prereg/v1`.
pub const PREREG_SECONDARY_STATUS: &str =
    "DIAGNOSTIC_VERIFIER_ONLY_PRODUCER_SCRATCH_UNEQUAL_RESOURCES_SINGLE_ENDPOINT_ALPHA_0_025";

/// F10 tie tolerance on log-RMSE ratios: `|ln(cand) - ln(ref)| <= 1e-12` is treated as zero (`0.0`).
pub const F10_TIE_TOLERANCE: f64 = 1e-12;

/// Maximum single-endpoint alpha permitted while joint Holm-Bonferroni multiplicity is unimplemented.
pub const MAX_SINGLE_ENDPOINT_ALPHA: f64 = 0.025;

/// Provenance binding disclosure string emitted when no `--prereg` packet is supplied.
pub const PROVENANCE_BINDING_STATUS: &str = "unverified_caller_asserted_inputs";

/// Provenance binding disclosure string emitted when `--prereg` matches `plan.prereg_sha256`
/// (raw or canonical) and all pinned contract fields match, but one or more fields remain
/// `"UNPINNED"` or `execution_grant.authorized == false`.
pub const PREREG_UNPINNED_BINDING_STATUS: &str =
    "prereg_contract_fields_checked_unpinned_no_execution_grant";

/// Provenance binding disclosure string emitted when `--prereg` matches `plan.prereg_sha256`
/// (raw or canonical), all pre-registration contract fields are syntactically pinned with
/// `execution_grant.authorized == true`, and per-round search/observation digest fields pass
/// 64-hex format checks, while underlying search/observation artifacts and the execution grant
/// remain caller-asserted and unverified by this offline tool.
pub const PREREG_PINNED_BINDING_STATUS: &str =
    "prereg_contract_fields_checked_receipts_and_grant_caller_asserted_unverified";

/// Disqualification reason appended on every `--prereg` report because underlying search/observation
/// artifacts and the execution grant are caller-asserted rather than externally verified.
pub const PREREG_CALLER_ASSERTED_REASON: &str =
    "prereg_receipts_and_execution_grant_caller_asserted_unverified";

/// Multiplicity adjustment disclosure string emitted on every report in this build.
pub const MULTIPLICITY_ADJUSTMENT_STATUS: &str = "unimplemented_single_endpoint_alpha_cap_0_025";

pub(super) fn default_alpha() -> f64 {
    MAX_SINGLE_ENDPOINT_ALPHA
}

fn value_contains_placeholder_sentinel(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => is_placeholder_sentinel(s).then(|| s.clone()),
        Value::Array(arr) => arr.iter().find_map(value_contains_placeholder_sentinel),
        Value::Object(map) => map.values().find_map(value_contains_placeholder_sentinel),
        _ => None,
    }
}

/// Recursive `serde_json::Value` wrapper whose `Deserialize` implementation rejects duplicate
/// object keys at every nesting depth (including inside arrays and nested maps) before
/// constructing a `serde_json::Map`.
struct StrictJsonValue(Value);

impl<'de> Deserialize<'de> for StrictJsonValue {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        struct StrictValueVisitor;

        impl<'de> Visitor<'de> for StrictValueVisitor {
            type Value = StrictJsonValue;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a valid JSON value without duplicate object keys")
            }

            fn visit_bool<E>(self, v: bool) -> std::result::Result<Self::Value, E> {
                Ok(StrictJsonValue(Value::Bool(v)))
            }

            fn visit_i64<E>(self, v: i64) -> std::result::Result<Self::Value, E> {
                Ok(StrictJsonValue(Value::Number(v.into())))
            }

            fn visit_u64<E>(self, v: u64) -> std::result::Result<Self::Value, E> {
                Ok(StrictJsonValue(Value::Number(v.into())))
            }

            fn visit_f64<E: de::Error>(self, v: f64) -> std::result::Result<Self::Value, E> {
                serde_json::Number::from_f64(v)
                    .map(|n| StrictJsonValue(Value::Number(n)))
                    .ok_or_else(|| de::Error::custom("non-finite float in JSON"))
            }

            fn visit_str<E>(self, v: &str) -> std::result::Result<Self::Value, E> {
                Ok(StrictJsonValue(Value::String(v.to_owned())))
            }

            fn visit_string<E>(self, v: String) -> std::result::Result<Self::Value, E> {
                Ok(StrictJsonValue(Value::String(v)))
            }

            fn visit_none<E>(self) -> std::result::Result<Self::Value, E> {
                Ok(StrictJsonValue(Value::Null))
            }

            fn visit_unit<E>(self) -> std::result::Result<Self::Value, E> {
                Ok(StrictJsonValue(Value::Null))
            }

            fn visit_some<D: Deserializer<'de>>(
                self,
                deserializer: D,
            ) -> std::result::Result<Self::Value, D::Error> {
                Deserialize::deserialize(deserializer)
            }

            fn visit_seq<A: SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut vec = Vec::new();
                while let Some(elem) = seq.next_element::<StrictJsonValue>()? {
                    vec.push(elem.0);
                }
                Ok(StrictJsonValue(Value::Array(vec)))
            }

            fn visit_map<M: MapAccess<'de>>(
                self,
                mut access: M,
            ) -> std::result::Result<Self::Value, M::Error> {
                let mut map = serde_json::Map::new();
                while let Some((k, v)) = access.next_entry::<String, StrictJsonValue>()? {
                    if map.contains_key(&k) {
                        return Err(de::Error::custom(format!("duplicate JSON object key: {k}")));
                    }
                    map.insert(k, v.0);
                }
                Ok(StrictJsonValue(Value::Object(map)))
            }
        }

        deserializer.deserialize_any(StrictValueVisitor)
    }
}

/// Parse a JSON document into [`serde_json::Value`], rejecting duplicate object keys at all
/// nesting depths and rejecting trailing characters after the root value.
pub fn parse_strict_json_value(bytes: &[u8], what: &str) -> Result<Value> {
    let mut de = serde_json::Deserializer::from_slice(bytes);
    let StrictJsonValue(val) = StrictJsonValue::deserialize(&mut de)
        .with_context(|| format!("{what}: invalid JSON or duplicate object key"))?;
    de.end()
        .with_context(|| format!("{what}: trailing characters after JSON value"))?;
    Ok(val)
}

/// Pre-registration field that is either explicitly marked `"UNPINNED"` or bound to a value `T`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UnpinnedOr<T> {
    /// The `Unpinned` case of `UnpinnedOr`.
    Unpinned,
    /// The `Pinned` case of `UnpinnedOr`.
    Pinned(T),
}

impl<T> UnpinnedOr<T> {
    /// Returns whether unpinned.
    pub fn is_unpinned(&self) -> bool {
        matches!(self, Self::Unpinned)
    }

    /// Performs the as pinned operation.
    pub fn as_pinned(&self) -> Option<&T> {
        match self {
            Self::Unpinned => None,
            Self::Pinned(v) => Some(v),
        }
    }
}

impl<T: Serialize> Serialize for UnpinnedOr<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        match self {
            Self::Unpinned => serializer.serialize_str("UNPINNED"),
            Self::Pinned(v) => v.serialize(serializer),
        }
    }
}

impl<'de, T: serde::de::DeserializeOwned> Deserialize<'de> for UnpinnedOr<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let StrictJsonValue(val) = StrictJsonValue::deserialize(deserializer)?;
        if val.as_str() == Some("UNPINNED") {
            return Ok(Self::Unpinned);
        }
        if let Some(bad) = value_contains_placeholder_sentinel(&val) {
            return Err(de::Error::custom(format!(
                "invalid placeholder or case/whitespace sentinel {bad:?}; use exact \"UNPINNED\" or a concrete pinned value"
            )));
        }
        serde_json::from_value(val)
            .map(Self::Pinned)
            .map_err(de::Error::custom)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `PreregEndpoints` is a record for the prereg endpoints contract in this module: Offline self-play round qualification diagnostic report (jope.selfplay-round-qualification/v1).
pub struct PreregEndpoints {
    /// Primary recorded for this prereg endpoints.
    pub primary: Endpoint,
    /// Secondary recorded for this prereg endpoints.
    pub secondary: Endpoint,
    /// Secondary status recorded for this prereg endpoints.
    pub secondary_status: String,
    /// Multiplicity procedure recorded for this prereg endpoints.
    pub multiplicity_procedure: String,
    /// Cli single endpoint alpha cap recorded for this prereg endpoints.
    pub cli_single_endpoint_alpha_cap: f64,
    /// Alpha total recorded for this prereg endpoints.
    pub alpha_total: f64,
    /// Alpha primary strict recorded for this prereg endpoints.
    pub alpha_primary_strict: f64,
    /// F10 tie tolerance log ratio recorded for this prereg endpoints.
    pub f10_tie_tolerance_log_ratio: f64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `PreregRosterAndFamilies` is a record for the prereg roster and families contract in this module: Offline self-play round qualification diagnostic report (jope.selfplay-round-qualification/v1).
pub struct PreregRosterAndFamilies {
    /// Unit of inference recorded for this prereg roster and families.
    pub unit_of_inference: String,
    /// Roster role recorded for this prereg roster and families.
    pub roster_role: String,
    /// SHA-256 digest identifying the associated sealed roster.
    pub sealed_roster_sha256: UnpinnedOr<String>,
    /// SHA-256 digest identifying the associated parent cluster map.
    pub parent_cluster_map_sha256: UnpinnedOr<String>,
    /// G23 sealed boundary ruling recorded for this prereg roster and families.
    pub g23_sealed_boundary_ruling: UnpinnedOr<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `PreregSampleSizeAndMargin` is a record for the prereg sample size and margin contract in this module: Offline self-play round qualification diagnostic report (jope.selfplay-round-qualification/v1).
pub struct PreregSampleSizeAndMargin {
    /// Lower bound on parent clusters.
    pub min_parent_clusters: UnpinnedOr<usize>,
    /// Practical margin log ratio recorded for this prereg sample size and margin.
    pub practical_margin_log_ratio: UnpinnedOr<f64>,
    /// C158 g judge binding receipt recorded for this prereg sample size and margin.
    pub c158_g_judge_binding_receipt: UnpinnedOr<String>,
    /// Bootstrap resamples recorded for this prereg sample size and margin.
    pub bootstrap_resamples: u32,
    /// Bootstrap seed recorded for this prereg sample size and margin.
    pub bootstrap_seed: u64,
    /// Bootstrap method recorded for this prereg sample size and margin.
    pub bootstrap_method: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `PreregKlGuard` is a record for the prereg KL guard contract in this module: Offline self-play round qualification diagnostic report (jope.selfplay-round-qualification/v1).
pub struct PreregKlGuard {
    /// Upper bound on conditional base KL.
    pub max_conditional_base_kl: f64,
    /// Upper bound on new support mass.
    pub max_new_support_mass: f64,
    /// Allow kl frozen rounds recorded for this prereg KL guard.
    pub allow_kl_frozen_rounds: bool,
    /// Collection of allowlisted measured sources values carried by this prereg KL guard.
    pub allowlisted_measured_sources: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `PreregArmsAndRounds` is a record for the prereg arms and rounds contract in this module: Offline self-play round qualification diagnostic report (jope.selfplay-round-qualification/v1).
pub struct PreregArmsAndRounds {
    /// Rounds recorded for this prereg arms and rounds.
    pub rounds: usize,
    /// Seed cap b recorded for this prereg arms and rounds.
    pub seed_cap_b: usize,
    /// Equal attempts per round required recorded for this prereg arms and rounds.
    pub equal_attempts_per_round_required: bool,
    /// Equal resource comparison required recorded for this prereg arms and rounds.
    pub equal_resource_comparison_required: bool,
    /// Measured tilt policy recorded for this prereg arms and rounds.
    pub measured_tilt_policy: MeasuredTiltPolicyConfig,
    /// Kl guard recorded for this prereg arms and rounds.
    pub kl_guard: PreregKlGuard,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `PreregSequentialLooks` is a record for the prereg sequential looks contract in this module: Offline self-play round qualification diagnostic report (jope.selfplay-round-qualification/v1).
pub struct PreregSequentialLooks {
    /// Collection of reference earlier looks values carried by this prereg sequential looks.
    pub reference_earlier_looks: Vec<String>,
    /// Collection of candidate earlier looks when distinct values carried by this prereg sequential looks.
    pub candidate_earlier_looks_when_distinct: Vec<String>,
    /// Reject omitted earlier looks field recorded for this prereg sequential looks.
    pub reject_omitted_earlier_looks_field: bool,
    /// Reject registry reset both empty when distinct recorded for this prereg sequential looks.
    pub reject_registry_reset_both_empty_when_distinct: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `PreregHeldOutAndParityRules` is a record for the prereg held out and parity rules contract in this module: Offline self-play round qualification diagnostic report (jope.selfplay-round-qualification/v1).
pub struct PreregHeldOutAndParityRules {
    /// Test score schema recorded for this prereg held out and parity rules.
    pub test_score_schema: String,
    /// Pinned test score registry recorded for this prereg held out and parity rules.
    pub pinned_test_score_registry: UnpinnedOr<String>,
    /// Sequential same registry looks recorded for this prereg held out and parity rules.
    pub sequential_same_registry_looks: PreregSequentialLooks,
    /// Require feeds selection false recorded for this prereg held out and parity rules.
    pub require_feeds_selection_false: bool,
    /// R5 parity failure disposition recorded for this prereg held out and parity rules.
    pub r5_parity_failure_disposition: String,
    /// Reject unreferenced score or parity records recorded for this prereg held out and parity rules.
    pub reject_unreferenced_score_or_parity_records: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `PreregExecutionGrant` is a record for the prereg execution grant contract in this module: Offline self-play round qualification diagnostic report (jope.selfplay-round-qualification/v1).
pub struct PreregExecutionGrant {
    /// Authorized recorded for this prereg execution grant.
    pub authorized: bool,
    /// Owner admission locator recorded for this prereg execution grant.
    pub owner_admission_locator: UnpinnedOr<String>,
    /// Hosts recorded for this prereg execution grant.
    pub hosts: UnpinnedOr<Vec<String>>,
    /// Cpu ceiling percent recorded for this prereg execution grant.
    pub cpu_ceiling_percent: UnpinnedOr<u32>,
    /// Memory floor free percent recorded for this prereg execution grant.
    pub memory_floor_free_percent: u32,
    /// Gpu seconds budget recorded for this prereg execution grant.
    pub gpu_seconds_budget: u64,
    /// Deadline utc recorded for this prereg execution grant.
    pub deadline_utc: UnpinnedOr<String>,
}

/// Pre-registration packet (`jope.selfplay-round1-prereg/v1`) for Round-1 self-play qualification.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelfPlayPrereg {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Identifier for the associated study.
    pub study_id: String,
    /// Outcome or lifecycle status recorded for this value.
    pub status: String,
    /// Qualification report schema recorded for this self play prereg.
    pub qualification_report_schema: String,
    /// Provenance binding status recorded for this self play prereg.
    pub provenance_binding_status: String,
    /// SHA-256 digest identifying the associated experiment identity.
    pub experiment_identity_sha256: UnpinnedOr<String>,
    /// Source commit at draft recorded for this self play prereg.
    pub source_commit_at_draft: String,
    /// Endpoints recorded for this self play prereg.
    pub endpoints: PreregEndpoints,
    /// Roster and families recorded for this self play prereg.
    pub roster_and_families: PreregRosterAndFamilies,
    /// Sample size and margin recorded for this self play prereg.
    pub sample_size_and_margin: PreregSampleSizeAndMargin,
    /// Arms and rounds recorded for this self play prereg.
    pub arms_and_rounds: PreregArmsAndRounds,
    /// Held out and parity rules recorded for this self play prereg.
    pub held_out_and_parity_rules: PreregHeldOutAndParityRules,
    /// Execution grant recorded for this self play prereg.
    pub execution_grant: PreregExecutionGrant,
}

impl SelfPlayPrereg {
    /// True if any required pre-registration field remains `"UNPINNED"`.
    pub fn has_unpinned_fields(&self) -> bool {
        self.experiment_identity_sha256.is_unpinned()
            || self.roster_and_families.sealed_roster_sha256.is_unpinned()
            || self
                .roster_and_families
                .parent_cluster_map_sha256
                .is_unpinned()
            || self
                .roster_and_families
                .g23_sealed_boundary_ruling
                .is_unpinned()
            || self
                .sample_size_and_margin
                .min_parent_clusters
                .is_unpinned()
            || self
                .sample_size_and_margin
                .practical_margin_log_ratio
                .is_unpinned()
            || self
                .sample_size_and_margin
                .c158_g_judge_binding_receipt
                .is_unpinned()
            || self
                .held_out_and_parity_rules
                .pinned_test_score_registry
                .is_unpinned()
            || self.execution_grant.owner_admission_locator.is_unpinned()
            || self.execution_grant.hosts.is_unpinned()
            || self.execution_grant.cpu_ceiling_percent.is_unpinned()
            || self.execution_grant.deadline_utc.is_unpinned()
    }
}
