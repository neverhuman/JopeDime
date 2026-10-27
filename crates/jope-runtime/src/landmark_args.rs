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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum LandmarkConsumer {
    Direct,
    JopeDl,
}

#[derive(Args, Clone, Debug, Default)]
pub struct LandmarkArgs {
    /// Select the raw Dope encoding or the learned Jope context. Explicit selection opts in.
    #[arg(long, value_enum)]
    pub landmark_consumer: Option<LandmarkConsumer>,
    /// References to controller-admitted, FIT-scoped foundation and target components.
    #[arg(long)]
    pub landmark_components: Option<PathBuf>,
    /// Verified policy-only asset directory; no runtime download or full-bundle loading.
    #[arg(long)]
    pub jope_policy_assets: Option<PathBuf>,
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
        self.landmark_mode.is_some()
            || self.landmark_subsample.is_some()
            || self.landmark_consumer.is_some()
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
        let learned = self.landmark_consumer == Some(LandmarkConsumer::JopeDl);
        ensure!(
            learned || (self.landmark_components.is_none() && self.jope_policy_assets.is_none()),
            "component and policy asset inputs require --landmark-consumer jope-dl"
        );
        ensure!(
            !learned || self.landmark_components.is_some(),
            "jope-dl requires --landmark-components with controller-admitted FIT inputs"
        );
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

    pub fn protected_inputs(&self) -> Vec<PathBuf> {
        if !self.enabled() {
            return Vec::new();
        }
        let mut paths = self
            .dope_weights
            .iter()
            .chain(&self.landmark_components)
            .chain(&self.jope_policy_assets)
            .cloned()
            .collect::<Vec<_>>();
        if self.landmark_consumer == Some(LandmarkConsumer::JopeDl) {
            paths.push(jope_paths::jope_state());
        }
        paths
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
        assert!(options.protected_inputs().is_empty());
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

    #[test]
    fn explicit_consumers_activate_encoding_and_require_scoped_inputs() {
        let direct = Cli::try_parse_from(["test", "--landmark-consumer", "direct"])
            .unwrap()
            .options;
        assert!(direct.enabled());
        assert_eq!(direct.mode(), LandmarkMode::Full);
        assert_eq!(direct.subsample(), None);
        direct.validate().unwrap();
        let mut learned = Cli::try_parse_from(["test", "--landmark-consumer", "jope-dl"])
            .unwrap()
            .options;
        assert!(learned.enabled());
        assert!(learned.validate().is_err());
        learned.landmark_components = Some("components.json".into());
        learned.jope_policy_assets = Some("selected-assets".into());
        learned.validate().unwrap();
        assert!(learned
            .protected_inputs()
            .contains(&jope_paths::jope_state()));
        assert!(learned
            .protected_inputs()
            .contains(&"components.json".into()));
        for selector in [None, Some(LandmarkConsumer::Direct)] {
            learned.landmark_consumer = selector;
            assert!(learned.validate().is_err());
        }
    }
}
