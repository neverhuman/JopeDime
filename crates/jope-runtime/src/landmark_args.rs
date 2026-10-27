//! Shared opt-in flags for Dope encoding. No options preserves the caller's existing path.
use anyhow::{ensure, Result};
use clap::{Args, ValueEnum};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum LandmarkMode {
    None,
    Fast,
    Progressive,
    Full,
}

#[derive(Args, Clone, Debug, Default)]
pub struct LandmarkArgs {
    /// Opt into Dope512 encoding. Progressive starts without landmarks and defaults to 300 rows.
    #[arg(long, value_enum)]
    pub landmark_mode: Option<LandmarkMode>,
    /// Rows used only for landmark evidence. Full mode otherwise uses all training rows.
    #[arg(long, value_parser = clap::value_parser!(u32).range(2..))]
    pub landmark_subsample: Option<u32>,
    /// Checksummed Dope weights; defaults to DOPE_WEIGHTS. Never downloaded at runtime.
    #[arg(long, env = "DOPE_WEIGHTS")]
    pub dope_weights: Option<PathBuf>,
    /// Write encoding snapshots as JSON lines while retaining the command's final stdout format.
    #[arg(long)]
    pub landmark_events: Option<PathBuf>,
    /// Companion landmark encoding receipt.
    #[arg(long)]
    pub landmark_receipt: Option<PathBuf>,
}

impl LandmarkArgs {
    pub fn enabled(&self) -> bool {
        self.landmark_mode.is_some() || self.landmark_subsample.is_some()
    }

    pub fn mode(&self) -> LandmarkMode {
        self.landmark_mode.unwrap_or(LandmarkMode::Full)
    }

    pub fn subsample(&self) -> Option<usize> {
        self.landmark_subsample
            .map(|n| n as usize)
            .or_else(|| (self.mode() == LandmarkMode::Progressive).then_some(300))
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.enabled() || (self.landmark_events.is_none() && self.landmark_receipt.is_none()),
            "landmark output paths require a landmark mode or subsample"
        );
        ensure!(
            self.landmark_subsample.is_none_or(|n| n >= 2),
            "landmark subsample must be at least 2"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        options: LandmarkArgs,
    }

    #[test]
    fn omitted_options_do_not_select_a_new_encoder() {
        let options = LandmarkArgs {
            dope_weights: Some("resident.safetensors".into()),
            ..Default::default()
        };
        assert!(!options.enabled());
        options.validate().unwrap();
    }

    #[test]
    fn full_is_unsampled_and_progressive_defaults_to_300() {
        let full = Cli::try_parse_from(["test", "--landmark-mode", "full"])
            .unwrap()
            .options;
        let progressive = Cli::try_parse_from(["test", "--landmark-mode", "progressive"])
            .unwrap()
            .options;
        assert!(full.enabled());
        assert_eq!(full.subsample(), None);
        assert_eq!(progressive.subsample(), Some(300));
    }

    #[test]
    fn explicit_sample_selects_dope_and_overrides_progressive_default() {
        let full = Cli::try_parse_from(["test", "--landmark-subsample", "100"])
            .unwrap()
            .options;
        assert!(full.enabled());
        assert_eq!(full.mode(), LandmarkMode::Full);
        assert_eq!(full.subsample(), Some(100));
        assert!(Cli::try_parse_from(["test", "--landmark-subsample", "0"]).is_err());
    }
}
