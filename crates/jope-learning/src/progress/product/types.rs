use crate::progress::cadence::{CheckpointDue, Clock, FlushDue, Policy, Tick};
use crate::progress::inventory::Head;
use crate::progress::outbox::Limits;
use crate::progress::publish::Publication;
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// The approval state of the default [`Config`] values: the outbox limits (W8-T02) and the
/// cadence (W8-T05) were adopted by owner delegation on 2026-09-26 (owner decision D6). This is
/// reported beside them; it is not a setting, and no configuration file can change it.
pub const APPROVAL: &str = "adopted-by-owner-delegation-2026-09-26";

/// The approval state reported when a configuration file sets values other than the adopted
/// defaults: they are the operator's, not the owner's.
pub const OVERRIDDEN: &str = "operator-configured-not-the-adopted-defaults";

/// The publisher's explicit configuration.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Outbox capacity: pause new work, then refuse everything (W8-T02).
    pub limits: Limits,
    /// Flush and checkpoint cadence (W8-T05).
    pub cadence: Policy,
}

impl Config {
    /// The documented defaults, adopted by owner delegation on 2026-09-26 ([`APPROVAL`]):
    /// - outbox: pause new work at 7,500 entries or 768 MiB pending; refuse every append at
    ///   10,000 entries or 1 GiB;
    /// - cadence: flush every 60 s or at 1 MiB pending; checkpoint every 100 committed updates
    ///   or 5 minutes, and at controlled shutdown.
    pub fn adopted() -> Config {
        Config {
            schema: CONFIG_SCHEMA.into(),
            limits: Limits::ADOPTED,
            cadence: Policy::ADOPTED,
        }
    }

    /// Performs the check operation and returns an error if it cannot be completed.
    pub fn check(&self) -> Result<()> {
        ensure!(
            self.schema == CONFIG_SCHEMA,
            "config schema {:?} is not {CONFIG_SCHEMA}",
            self.schema
        );
        self.limits.check()?;
        let p = &self.cadence;
        ensure!(
            p.flush_interval_ms > 0
                && p.flush_bytes > 0
                && p.checkpoint_updates > 0
                && p.checkpoint_interval_ms > 0,
            "every cadence value must be positive"
        );
        Ok(())
    }

    /// Read an explicit configuration file. Unknown and missing fields are refused: there is no
    /// partial override that silently keeps some defaults.
    pub fn load(path: &Path) -> Result<Config> {
        let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        let config: Config = serde_json::from_slice(&bytes)
            .with_context(|| format!("{} is not a {CONFIG_SCHEMA} file", path.display()))?;
        config.check()?;
        Ok(config)
    }

    /// The configuration as reported: its values, whether they are the adopted defaults, and
    /// their approval state.
    pub fn report(&self) -> ConfigReport {
        let adopted_defaults = *self == Config::adopted();
        ConfigReport {
            values: self.clone(),
            adopted_defaults,
            approval: if adopted_defaults {
                APPROVAL
            } else {
                OVERRIDDEN
            },
        }
    }
}

#[derive(Clone, Debug, Serialize)]
/// `ConfigReport` is a record for the config report contract in this module: W8 product integration: the publisher that turns the engine's durable events into committed W8 progress.
pub struct ConfigReport {
    /// Values recorded for this config report.
    pub values: Config,
    /// Adopted defaults recorded for this config report.
    pub adopted_defaults: bool,
    /// Approval recorded for this config report.
    pub approval: &'static str,
}

/// The wall clock, in milliseconds since the Unix epoch.
pub struct SystemClock;

impl Clock for SystemClock {
    fn now_ms(&self) -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
    }
}

/// What the checkpoint cadence decided at an observed engine checkpoint.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case", tag = "decision")]
pub enum CheckpointStep {
    /// No checkpoint is published yet, or none newer than the last one counted.
    Nothing,
    /// Not due: the engine's newer optimizer steps are counted and wait for the next W8
    /// checkpoint.
    /// Field value recorded by this public API type.
    /// Field value recorded by this public API type.
    Deferred {
        /// Step recorded for this outcome.
        step: u64,
        /// Uncheckpointed recorded for this outcome.
        uncheckpointed: u64,
    },
    /// Mapped as the next W8 update.
    Mapped {
        /// Step recorded for this checkpoint step.
        step: u64,
        /// Update index recorded for this checkpoint step.
        update_index: u64,
        /// Number of consumed rows represented here.
        consumed_rows: u64,
        /// Due recorded for this checkpoint step.
        due: CheckpointDue,
    },
}

/// What one driver operation did.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Step {
    /// Outbox keys this operation appended.
    pub appended: Vec<String>,
    /// Keys it offered that the outbox already held (a replay after a crash, or the same bytes).
    pub existing: Vec<String>,
    /// Optional checkpoint value; `None` means it is not available.
    pub checkpoint: Option<CheckpointStep>,
    /// Why a flush was due, if one was.
    pub due: Option<FlushDue>,
    /// Optional publication value; `None` means it is not available.
    pub publication: Option<Publication>,
    /// Optional tick value; `None` means it is not available.
    pub tick: Option<Tick>,
    /// A flush was due and left nothing staged and nothing pending: everything appended so far
    /// is committed and acknowledged.
    pub flushed: bool,
    /// Objects appended but not yet committed after this operation.
    pub staged: u64,
    /// Optional head value; `None` means it is not available.
    pub head: Option<Head>,
}
