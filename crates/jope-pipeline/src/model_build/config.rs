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

/// Configuration file. An absent field inherits; an explicit `false` disables an inherited
/// `true`, so the field is `Option<bool>`.
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
}

/// Explicit switch values. `None` means the switch was not given.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ModelBuildFlags {
    pub automl: Option<bool>,
    pub ensemble: Option<bool>,
}

/// Command-line surface, flattened into a subcommand when wiring is approved.
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
    /// Whole-request Thinking level: low (64), med or medium (256), high (1024), ultra (4096).
    #[arg(long, value_parser = parse_thinking)]
    pub thinking: Option<String>,
}

impl ModelBuildArgs {
    pub fn flags(&self) -> ModelBuildFlags {
        ModelBuildFlags {
            automl: switch(self.automl, self.no_automl),
            ensemble: switch(self.ensemble, self.no_ensemble),
        }
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

/// Accepts `low`, `med`, `medium`, `high` and `ultra`. Returns the canonical spelling.
pub fn parse_thinking(text: &str) -> std::result::Result<String, String> {
    match text {
        "low" | "med" | "medium" | "high" | "ultra" => Ok(text.to_owned()),
        other => Err(format!("unknown thinking level {other:?}")),
    }
}

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

/// The resolved configuration, with the source of each switch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedModelBuild {
    pub automl: bool,
    pub automl_source: Source,
    pub ensemble: bool,
    pub ensemble_source: Source,
    pub thinking: Option<ThinkingLevel>,
}

/// Precedence: explicit flag, then configuration file, then the built-in default (false, false).
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
    let pick = |flag: Option<bool>, from_file: Option<bool>| match (flag, from_file) {
        (Some(value), _) => (value, Source::Flag),
        (None, Some(value)) => (value, Source::File),
        (None, None) => (false, Source::Default),
    };
    let (automl, automl_source) = pick(flags.automl, file.and_then(|f| f.automl));
    let (ensemble, ensemble_source) = pick(flags.ensemble, file.and_then(|f| f.ensemble));
    let thinking = match file.and_then(|f| f.thinking.as_deref()) {
        Some(text) => Some(thinking_level(text)?),
        None => None,
    };
    Ok(ResolvedModelBuild {
        automl,
        automl_source,
        ensemble,
        ensemble_source,
        thinking,
    })
}

/// The four routes of the execution table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Route {
    /// `(false, false)`: fast checkpoint-bound neural to GP.
    FastNeuralGp,
    /// `(true, false)`: neural to GP plus the native estimator CV bake-off.
    NeuralGpAutomlBakeoff,
    /// `(false, true)`: fast neural to GP, then the post-search winner ensemble attempt.
    NeuralGpWinnerEnsemble,
    /// `(true, true)`: combined roster bake-off, then the post-search winner ensemble attempt.
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

/// Checkpoint-bound neural input. Every route needs it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum NeuralCheckpoint {
    Available { checkpoint_sha256: String },
    Unavailable { reason: String },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Disposition {
    /// The route cannot start. No fallback is substituted.
    Refused { route: Route, reason: String },
    /// The checkpoint is present, but the named stages are not implemented yet.
    Pending {
        route: Route,
        missing: Vec<&'static str>,
    },
}

/// Never returns a completed disposition: source-level routing cannot execute a fit.
pub fn disposition(route: Route, checkpoint: &NeuralCheckpoint) -> Disposition {
    match checkpoint {
        NeuralCheckpoint::Unavailable { reason } => Disposition::Refused {
            route,
            reason: format!(
                "checkpoint-bound neural inference is unavailable ({reason}); no generic CPU GP or ridge fallback is substituted"
            ),
        },
        NeuralCheckpoint::Available { .. } => {
            let missing = match route {
                Route::FastNeuralGp => vec!["final neural-to-GP executor (P02 runtime)"],
                Route::NeuralGpAutomlBakeoff => vec![
                    "final neural-to-GP executor (P02 runtime)",
                    "JA-03 native estimator CV bake-off",
                ],
                Route::NeuralGpWinnerEnsemble => vec![
                    "final neural-to-GP executor (P02 runtime)",
                    "JA-04 OOF winner ensemble",
                ],
                Route::CombinedRosterWinnerEnsemble => vec![
                    "final neural-to-GP executor (P02 runtime)",
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
        }
    }

    fn available() -> NeuralCheckpoint {
        NeuralCheckpoint::Available {
            checkpoint_sha256: "a".repeat(64),
        }
    }

    #[test]
    fn defaults_are_the_fast_route_with_default_sources() {
        let r = resolve(None, ModelBuildFlags::default()).unwrap();
        assert!(!r.automl && !r.ensemble);
        assert_eq!(r.automl_source, Source::Default);
        assert_eq!(r.ensemble_source, Source::Default);
        assert_eq!(route(&r), Route::FastNeuralGp);
    }

    #[test]
    fn flag_overrides_the_file_and_a_negative_flag_disables_an_inherited_true() {
        let inherited = file(Some(true), Some(true));
        let overridden = resolve(
            Some(&inherited),
            ModelBuildFlags {
                automl: Some(false),
                ensemble: None,
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
        let explicit_false =
            resolve(Some(&file(Some(false), None)), ModelBuildFlags::default()).unwrap();
        assert_eq!(explicit_false.automl_source, Source::File);
        let absent = resolve(Some(&file(None, None)), ModelBuildFlags::default()).unwrap();
        assert_eq!(absent.automl_source, Source::Default);
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
        assert_eq!(alone.model.flags().automl, Some(false));
        let none = Harness::try_parse_from(["jope"]).unwrap();
        assert_eq!(none.model.flags(), ModelBuildFlags::default());
    }

    #[test]
    fn the_thinking_flag_accepts_med_through_clap() {
        use clap::Parser;
        let parsed = Harness::try_parse_from(["jope", "--thinking", "med"]).unwrap();
        assert_eq!(parsed.model.thinking.as_deref(), Some("med"));
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
    fn thinking_accepts_med_and_medium_as_the_same_cap() {
        assert_eq!(thinking_level("med").unwrap(), ThinkingLevel::Medium);
        assert_eq!(thinking_level("medium").unwrap(), ThinkingLevel::Medium);
        assert!(thinking_level("huge").is_err());
    }

    #[test]
    fn the_four_routes_match_the_execution_table() {
        let make = |automl, ensemble| ResolvedModelBuild {
            automl,
            automl_source: Source::Default,
            ensemble,
            ensemble_source: Source::Default,
            thinking: None,
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
    fn a_missing_checkpoint_refuses_every_route_without_fallback() {
        let missing = NeuralCheckpoint::Unavailable {
            reason: "P02 release set not published".into(),
        };
        for r in [
            Route::FastNeuralGp,
            Route::NeuralGpAutomlBakeoff,
            Route::NeuralGpWinnerEnsemble,
            Route::CombinedRosterWinnerEnsemble,
        ] {
            match disposition(r, &missing) {
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
    fn an_available_checkpoint_never_reports_a_completed_route() {
        for r in [
            Route::FastNeuralGp,
            Route::NeuralGpAutomlBakeoff,
            Route::NeuralGpWinnerEnsemble,
            Route::CombinedRosterWinnerEnsemble,
        ] {
            match disposition(r, &available()) {
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
