use crate::pathway::ThinkingLevel;
use anyhow::{bail, ensure, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub const OPTIONS_SCHEMA: &str = "jope.model-build-options/v1";

/// Where a resolved value came from. Recorded per field in the receipt.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Source {
    Default,
    File,
    Flag,
}

/// An untrusted checkpoint reference from a configuration file. It is not evidence: it becomes
/// a [`VerifiedNeuralCheckpoint`] only through [`verify_checkpoint`] against a P02 closure.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointClaim {
    pub release_id: String,
    pub source_commit: String,
    pub checkpoint_sha256: String,
    pub runtime_sha256: String,
}

/// Configuration file. An absent field inherits; an explicit `false` disables an inherited
/// `true`, so the switches are `Option<bool>`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelBuildFile {
    pub schema: String,
    #[serde(default)]
    pub automl: Option<bool>,
    #[serde(default)]
    pub ensemble: Option<bool>,
    /// `low`, `med`, `medium`, `high` or `ultra`. Independent of the two switches above.
    #[serde(default)]
    pub thinking: Option<String>,
    #[serde(default)]
    pub neural_checkpoint: Option<CheckpointClaim>,
}

/// Explicit switch values. `None` means the value was not given on the command line.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ModelBuildFlags {
    pub automl: Option<bool>,
    pub ensemble: Option<bool>,
    pub thinking: Option<ThinkingLevel>,
}

/// Command-line surface, flattened into `jope pipeline run`.
#[derive(Clone, Debug, Default, clap::Args)]
pub struct ModelBuildArgs {
    /// Opt in to the native estimator CV bake-off.
    #[arg(long, conflicts_with = "no_automl")]
    pub automl: bool,
    /// Disable an inherited `automl = true` from the configuration file.
    #[arg(long = "no-automl")]
    pub no_automl: bool,
    /// Opt in to the post-search winner ensemble.
    #[arg(long, conflicts_with = "no_ensemble")]
    pub ensemble: bool,
    /// Disable an inherited `ensemble = true` from the configuration file.
    #[arg(long = "no-ensemble")]
    pub no_ensemble: bool,
    /// Model-build options file (`jope.model-build-options/v1`).
    #[arg(long)]
    pub model_build_config: Option<PathBuf>,
    /// Independently trusted source commit that a checkpoint claim must name. It comes from the
    /// operator, never from the claim itself.
    #[arg(long)]
    pub expected_source_commit: Option<String>,
    /// Whole-request Thinking level: low (64), med or medium (256), high (1024), ultra (4096).
    #[arg(long, value_parser = parse_thinking)]
    pub thinking: Option<String>,
}

impl ModelBuildArgs {
    pub fn flags(&self) -> Result<ModelBuildFlags> {
        Ok(ModelBuildFlags {
            automl: switch(self.automl, self.no_automl),
            ensemble: switch(self.ensemble, self.no_ensemble),
            thinking: match self.thinking.as_deref() {
                Some(text) => Some(thinking_level(text)?),
                None => None,
            },
        })
    }
}

fn switch(on: bool, off: bool) -> Option<bool> {
    if on {
        Some(true)
    } else if off {
        Some(false)
    } else {
        None
    }
}

/// Accepts `low`, `med`, `medium`, `high` and `ultra`. Returns the input unchanged.
pub fn parse_thinking(text: &str) -> std::result::Result<String, String> {
    match text {
        "low" | "med" | "medium" | "high" | "ultra" => Ok(text.to_owned()),
        other => Err(format!("unknown thinking level {other:?}")),
    }
}

/// `med` and `medium` resolve to the same enum value and therefore the same identity.
fn thinking_level(text: &str) -> Result<ThinkingLevel> {
    Ok(match parse_thinking(text) {
        Ok(canonical) => match canonical.as_str() {
            "low" => ThinkingLevel::Low,
            "med" | "medium" => ThinkingLevel::Medium,
            "high" => ThinkingLevel::High,
            _ => ThinkingLevel::Ultra,
        },
        Err(reason) => bail!(reason),
    })
}

/// The resolved configuration, with the source of each value.
///
/// `thinking == None` with `thinking_source == Default` means no Thinking request. The fast
/// default never selects `Direct`, which is not implemented.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedModelBuild {
    pub automl: bool,
    pub automl_source: Source,
    pub ensemble: bool,
    pub ensemble_source: Source,
    pub thinking: Option<ThinkingLevel>,
    pub thinking_source: Source,
}

/// Reads a `jope.model-build-options/v1` JSON file. Unknown fields are refused by the schema.
pub fn load_file(path: &std::path::Path) -> Result<ModelBuildFile> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| anyhow::anyhow!("reading {}: {error}", path.display()))?;
    serde_json::from_str(&text)
        .map_err(|error| anyhow::anyhow!("parsing {}: {error}", path.display()))
}

/// Precedence for every field: explicit flag, then configuration file, then the built-in default.
pub fn resolve(
    file: Option<&ModelBuildFile>,
    flags: ModelBuildFlags,
) -> Result<ResolvedModelBuild> {
    if let Some(file) = file {
        ensure!(
            file.schema == OPTIONS_SCHEMA,
            "model-build options schema {:?} is not {OPTIONS_SCHEMA}",
            file.schema
        );
    }
    let pick_bool = |flag: Option<bool>, from_file: Option<bool>| match (flag, from_file) {
        (Some(value), _) => (value, Source::Flag),
        (None, Some(value)) => (value, Source::File),
        (None, None) => (false, Source::Default),
    };
    let (automl, automl_source) = pick_bool(flags.automl, file.and_then(|f| f.automl));
    let (ensemble, ensemble_source) = pick_bool(flags.ensemble, file.and_then(|f| f.ensemble));
    let (thinking, thinking_source) =
        match (flags.thinking, file.and_then(|f| f.thinking.as_deref())) {
            (Some(level), _) => (Some(level), Source::Flag),
            (None, Some(text)) => (Some(thinking_level(text)?), Source::File),
            (None, None) => (None, Source::Default),
        };
    Ok(ResolvedModelBuild {
        automl,
        automl_source,
        ensemble,
        ensemble_source,
        thinking,
        thinking_source,
    })
}

/// The four routes of the execution table. The switches are independent: `automl = false` does
/// not disable an explicitly requested `ensemble`, and `ensemble = false` does not disable an
/// explicitly requested `automl`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Route {
    /// `(false, false)`: fast checkpoint-bound neural to GP, one arm.
    FastNeuralGp,
    /// `(true, false)`: neural to GP plus the native estimator CV bake-off.
    NeuralGpAutomlBakeoff,
    /// `(false, true)`: neural to GP pool, then the requested post-search winner ensemble.
    NeuralGpWinnerEnsemble,
    /// `(true, true)`: combined roster bake-off, then the post-search winner ensemble.
    CombinedRosterWinnerEnsemble,
}

pub fn route(options: &ResolvedModelBuild) -> Route {
    match (options.automl, options.ensemble) {
        (false, false) => Route::FastNeuralGp,
        (true, false) => Route::NeuralGpAutomlBakeoff,
        (false, true) => Route::NeuralGpWinnerEnsemble,
        (true, true) => Route::CombinedRosterWinnerEnsemble,
    }
}

/// The P02 release closure a checkpoint must match. Produced by the P02 verifier; this module
/// only consumes it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReleaseClosure {
    pub release_id: String,
    pub source_commit: String,
    pub checkpoint_sha256: String,
    pub runtime_sha256: String,
}

/// A checkpoint that matched a P02 closure. Fields are private, so this is only constructed by
/// [`verify_checkpoint`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct VerifiedNeuralCheckpoint {
    release_id: String,
    source_commit: String,
    checkpoint_sha256: String,
    runtime_sha256: String,
}

impl VerifiedNeuralCheckpoint {
    pub fn checkpoint_sha256(&self) -> &str {
        &self.checkpoint_sha256
    }
    pub fn release_id(&self) -> &str {
        &self.release_id
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub enum CheckpointRefusal {
    /// No P02 release closure is available to verify against.
    Absent,
    /// A claim was given without an independently trusted source expectation.
    NoSourceExpectation,
    Malformed(String),
    WrongSource {
        expected: String,
        found: String,
    },
    Mismatch(String),
}

impl std::fmt::Display for CheckpointRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Absent => write!(f, "no verified P02 release closure is present"),
            Self::NoSourceExpectation => write!(
                f,
                "no independently trusted source expectation was supplied"
            ),
            Self::Malformed(reason) => write!(f, "checkpoint claim is malformed: {reason}"),
            Self::WrongSource { expected, found } => {
                write!(
                    f,
                    "checkpoint claim is for source {found}, expected {expected}"
                )
            }
            Self::Mismatch(reason) => {
                write!(f, "checkpoint claim does not match its closure: {reason}")
            }
        }
    }
}

fn is_hex(text: &str, len: usize) -> bool {
    text.len() == len
        && text
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

/// Verifies an untrusted claim against a P02 closure. Refuses without a closure, on a malformed
/// field, on a source other than `expected_source_commit`, or on any field that differs from the
/// closure. No fallback is substituted.
pub fn verify_checkpoint(
    claim: &CheckpointClaim,
    closure: Option<&ReleaseClosure>,
    expected_source_commit: &str,
) -> std::result::Result<VerifiedNeuralCheckpoint, CheckpointRefusal> {
    if !is_hex(&claim.checkpoint_sha256, 64) || !is_hex(&claim.runtime_sha256, 64) {
        return Err(CheckpointRefusal::Malformed(
            "sha256 fields must be 64 lowercase hex characters".into(),
        ));
    }
    if !is_hex(&claim.source_commit, 40) {
        return Err(CheckpointRefusal::Malformed(
            "source_commit must be a 40-character lowercase hex commit".into(),
        ));
    }
    if claim.source_commit != expected_source_commit {
        return Err(CheckpointRefusal::WrongSource {
            expected: expected_source_commit.into(),
            found: claim.source_commit.clone(),
        });
    }
    let closure = closure.ok_or(CheckpointRefusal::Absent)?;
    if closure.release_id != claim.release_id
        || closure.source_commit != claim.source_commit
        || closure.checkpoint_sha256 != claim.checkpoint_sha256
        || closure.runtime_sha256 != claim.runtime_sha256
    {
        return Err(CheckpointRefusal::Mismatch(format!(
            "release {} does not match the closure for release {}",
            claim.release_id, closure.release_id
        )));
    }
    Ok(VerifiedNeuralCheckpoint {
        release_id: claim.release_id.clone(),
        source_commit: claim.source_commit.clone(),
        checkpoint_sha256: claim.checkpoint_sha256.clone(),
        runtime_sha256: claim.runtime_sha256.clone(),
    })
}

/// The checkpoint input every route needs.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub enum NeuralCheckpoint {
    Verified(VerifiedNeuralCheckpoint),
    Refused(CheckpointRefusal),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Disposition {
    /// The route cannot start. No fallback is substituted.
    Refused { route: Route, reason: String },
    /// The checkpoint verified, but the named stages are not implemented yet.
    Pending {
        route: Route,
        missing: Vec<&'static str>,
    },
}

/// Never returns a completed disposition: source-level routing cannot execute a fit.
pub fn disposition(route: Route, checkpoint: &NeuralCheckpoint) -> Disposition {
    match checkpoint {
        NeuralCheckpoint::Refused(refusal) => Disposition::Refused {
            route,
            reason: format!(
                "checkpoint-bound neural inference is unavailable: {refusal}; no generic CPU GP or ridge fallback is substituted"
            ),
        },
        NeuralCheckpoint::Verified(_) => {
            let missing = match route {
                Route::FastNeuralGp => vec!["final neural-to-GP executor"],
                Route::NeuralGpAutomlBakeoff => {
                    vec!["final neural-to-GP executor", "JA-03 native estimator CV bake-off"]
                }
                Route::NeuralGpWinnerEnsemble => {
                    vec!["final neural-to-GP executor", "JA-04 OOF winner ensemble"]
                }
                Route::CombinedRosterWinnerEnsemble => vec![
                    "final neural-to-GP executor",
                    "JA-03 native estimator CV bake-off",
                    "JA-04 OOF winner ensemble",
                ],
            };
            Disposition::Pending { route, missing }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(automl: Option<bool>, ensemble: Option<bool>) -> ModelBuildFile {
        ModelBuildFile {
            schema: OPTIONS_SCHEMA.into(),
            automl,
            ensemble,
            thinking: None,
            neural_checkpoint: None,
        }
    }

    const SOURCE: &str = "9c9202aa39357bc64758c0550ad509e7541c4bb4";

    fn claim() -> CheckpointClaim {
        CheckpointClaim {
            release_id: "native-policy-lm-282-r1".into(),
            source_commit: SOURCE.into(),
            checkpoint_sha256: "a".repeat(64),
            runtime_sha256: "b".repeat(64),
        }
    }

    fn closure_for(claim: &CheckpointClaim) -> ReleaseClosure {
        ReleaseClosure {
            release_id: claim.release_id.clone(),
            source_commit: claim.source_commit.clone(),
            checkpoint_sha256: claim.checkpoint_sha256.clone(),
            runtime_sha256: claim.runtime_sha256.clone(),
        }
    }

    fn verified() -> NeuralCheckpoint {
        let c = claim();
        NeuralCheckpoint::Verified(verify_checkpoint(&c, Some(&closure_for(&c)), SOURCE).unwrap())
    }

    #[test]
    fn defaults_are_the_fast_route_with_default_sources() {
        let r = resolve(None, ModelBuildFlags::default()).unwrap();
        assert!(!r.automl && !r.ensemble);
        assert_eq!(r.automl_source, Source::Default);
        assert_eq!(r.ensemble_source, Source::Default);
        assert_eq!(r.thinking, None);
        assert_eq!(r.thinking_source, Source::Default);
        assert_eq!(route(&r), Route::FastNeuralGp);
    }

    #[test]
    fn flag_overrides_the_file_and_a_negative_flag_disables_an_inherited_true() {
        let inherited = file(Some(true), Some(true));
        let overridden = resolve(
            Some(&inherited),
            ModelBuildFlags {
                automl: Some(false),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(!overridden.automl);
        assert_eq!(overridden.automl_source, Source::Flag);
        assert!(overridden.ensemble);
        assert_eq!(overridden.ensemble_source, Source::File);
        assert_eq!(route(&overridden), Route::NeuralGpWinnerEnsemble);
    }

    #[test]
    fn an_explicit_false_in_the_file_is_distinct_from_an_absent_field() {
        let explicit = resolve(Some(&file(Some(false), None)), ModelBuildFlags::default()).unwrap();
        assert_eq!(explicit.automl_source, Source::File);
        let absent = resolve(Some(&file(None, None)), ModelBuildFlags::default()).unwrap();
        assert_eq!(absent.automl_source, Source::Default);
    }

    #[test]
    fn cli_thinking_overrides_the_file_and_records_its_source() {
        let mut inherited = file(None, None);
        inherited.thinking = Some("low".into());
        let flags = ModelBuildFlags {
            thinking: Some(ThinkingLevel::High),
            ..Default::default()
        };
        let r = resolve(Some(&inherited), flags).unwrap();
        assert_eq!(r.thinking, Some(ThinkingLevel::High));
        assert_eq!(r.thinking_source, Source::Flag);
        let from_file = resolve(Some(&inherited), ModelBuildFlags::default()).unwrap();
        assert_eq!(from_file.thinking, Some(ThinkingLevel::Low));
        assert_eq!(from_file.thinking_source, Source::File);
    }

    #[test]
    fn med_and_medium_resolve_to_the_same_identity_from_flag_and_file() {
        let flag_med = ModelBuildArgs {
            thinking: Some("med".into()),
            ..Default::default()
        };
        let flag_medium = ModelBuildArgs {
            thinking: Some("medium".into()),
            ..Default::default()
        };
        let a = resolve(None, flag_med.flags().unwrap()).unwrap();
        let b = resolve(None, flag_medium.flags().unwrap()).unwrap();
        assert_eq!(a.thinking, b.thinking);
        assert_eq!(
            serde_json::to_string(&a).unwrap(),
            serde_json::to_string(&b).unwrap()
        );
        let mut from_file = file(None, None);
        from_file.thinking = Some("med".into());
        assert_eq!(
            resolve(Some(&from_file), ModelBuildFlags::default())
                .unwrap()
                .thinking,
            a.thinking
        );
    }

    #[derive(clap::Parser)]
    struct Harness {
        #[command(flatten)]
        model: ModelBuildArgs,
    }

    #[test]
    fn the_positive_and_negative_switch_together_is_a_usage_error() {
        use clap::Parser;
        assert!(Harness::try_parse_from(["jope", "--automl", "--no-automl"]).is_err());
        assert!(Harness::try_parse_from(["jope", "--ensemble", "--no-ensemble"]).is_err());
        let alone = Harness::try_parse_from(["jope", "--no-automl"]).unwrap();
        assert_eq!(alone.model.flags().unwrap().automl, Some(false));
        let none = Harness::try_parse_from(["jope"]).unwrap();
        assert_eq!(none.model.flags().unwrap(), ModelBuildFlags::default());
    }

    #[test]
    fn the_thinking_flag_is_parsed_and_resolved_through_clap() {
        use clap::Parser;
        let parsed = Harness::try_parse_from(["jope", "--thinking", "medium"]).unwrap();
        assert_eq!(
            parsed.model.flags().unwrap().thinking,
            Some(ThinkingLevel::Medium)
        );
        assert!(Harness::try_parse_from(["jope", "--thinking", "huge"]).is_err());
    }

    #[test]
    fn a_file_with_a_foreign_schema_is_refused() {
        let mut wrong = file(None, None);
        wrong.schema = "jope.other/v1".into();
        assert!(resolve(Some(&wrong), ModelBuildFlags::default()).is_err());
    }

    #[test]
    fn unknown_fields_in_the_file_are_refused_at_parse_time() {
        let text = r#"{"schema":"jope.model-build-options/v1","automl":true,"extra":1}"#;
        assert!(serde_json::from_str::<ModelBuildFile>(text).is_err());
    }

    #[test]
    fn the_four_routes_match_the_execution_table() {
        let make = |automl, ensemble| ResolvedModelBuild {
            automl,
            automl_source: Source::Default,
            ensemble,
            ensemble_source: Source::Default,
            thinking: None,
            thinking_source: Source::Default,
        };
        assert_eq!(route(&make(false, false)), Route::FastNeuralGp);
        assert_eq!(route(&make(true, false)), Route::NeuralGpAutomlBakeoff);
        assert_eq!(route(&make(false, true)), Route::NeuralGpWinnerEnsemble);
        assert_eq!(
            route(&make(true, true)),
            Route::CombinedRosterWinnerEnsemble
        );
    }

    #[test]
    fn a_missing_closure_refuses_every_route_without_fallback() {
        let c = claim();
        let refused = NeuralCheckpoint::Refused(verify_checkpoint(&c, None, SOURCE).unwrap_err());
        for r in [
            Route::FastNeuralGp,
            Route::NeuralGpAutomlBakeoff,
            Route::NeuralGpWinnerEnsemble,
            Route::CombinedRosterWinnerEnsemble,
        ] {
            match disposition(r, &refused) {
                Disposition::Refused { reason, .. } => {
                    assert!(
                        reason.contains("no generic CPU GP or ridge fallback"),
                        "{reason}"
                    );
                }
                other => panic!("{r:?} did not refuse: {other:?}"),
            }
        }
    }

    #[test]
    fn malformed_claims_are_refused_before_any_closure_is_consulted() {
        let c = claim();
        let closure = closure_for(&c);
        let mut upper = c.clone();
        upper.checkpoint_sha256 = "A".repeat(64);
        assert!(matches!(
            verify_checkpoint(&upper, Some(&closure), SOURCE),
            Err(CheckpointRefusal::Malformed(_))
        ));
        let mut short = c.clone();
        short.runtime_sha256 = "b".repeat(63);
        assert!(matches!(
            verify_checkpoint(&short, Some(&closure), SOURCE),
            Err(CheckpointRefusal::Malformed(_))
        ));
        let mut bad_commit = c.clone();
        bad_commit.source_commit = "xyz".into();
        assert!(matches!(
            verify_checkpoint(&bad_commit, Some(&closure), SOURCE),
            Err(CheckpointRefusal::Malformed(_))
        ));
    }

    #[test]
    fn a_claim_for_another_source_is_refused() {
        let c = claim();
        let other = "1".repeat(40);
        assert!(matches!(
            verify_checkpoint(&c, Some(&closure_for(&c)), &other),
            Err(CheckpointRefusal::WrongSource { .. })
        ));
    }

    #[test]
    fn a_claim_that_differs_from_its_closure_is_refused() {
        let c = claim();
        let mut closure = closure_for(&c);
        closure.checkpoint_sha256 = "c".repeat(64);
        assert!(matches!(
            verify_checkpoint(&c, Some(&closure), SOURCE),
            Err(CheckpointRefusal::Mismatch(_))
        ));
    }

    #[test]
    fn a_matching_claim_verifies_and_routes_stay_pending_never_completed() {
        assert!(matches!(verified(), NeuralCheckpoint::Verified(_)));
        for r in [
            Route::FastNeuralGp,
            Route::NeuralGpAutomlBakeoff,
            Route::NeuralGpWinnerEnsemble,
            Route::CombinedRosterWinnerEnsemble,
        ] {
            match disposition(r, &verified()) {
                Disposition::Pending { missing, .. } => assert!(!missing.is_empty()),
                other => panic!("{r:?} must stay pending, got {other:?}"),
            }
        }
    }

    #[test]
    fn resolved_options_round_trip_through_json() {
        let r = resolve(
            Some(&file(Some(true), Some(false))),
            ModelBuildFlags::default(),
        )
        .unwrap();
        let text = serde_json::to_string(&r).unwrap();
        assert_eq!(
            serde_json::from_str::<ResolvedModelBuild>(&text).unwrap(),
            r
        );
    }
}
