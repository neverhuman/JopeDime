//! Four-island generations 20-29 continuation of the immutable Golden42
//! test-adaptive campaign.

use super::*;

const ISLANDS: usize = 4;
const ISLAND_POPULATION: usize = 30;
const TOTAL_POPULATION: usize = ISLANDS * ISLAND_POPULATION;
const ISLAND_ELITES: usize = 4;
const GENERATION_20_PARENT_POOL: usize = 30;
const ISLAND_PARENT_POOL: usize = 15;
const ISLAND_MIN_VALID: usize = 15;
const GLOBAL_MIN_VALID: usize = 60;
const START_GENERATION: usize = 20;
const GENERATIONS: usize = 10;
const END_GENERATION: usize = START_GENERATION + GENERATIONS;
const EXISTING_SURPASSED: usize = 35;
const MIGRATION_GENERATIONS: &[usize] = &[21, 23, 25, 27];
const MANIFEST_SCHEMA: &str = "JopeTestAdaptiveIslandExtensionManifest.v1";
const CAMPAIGN_SCHEMA: &str = "JopeTestAdaptiveIslandDatasetCampaign.v1";
const CHECKPOINT_SCHEMA: &str = "JopeTestAdaptiveIslandGenerationCheckpoint.v1";
const MIGRATION_SCHEMA: &str = "JopeTestAdaptiveIslandMigration.v1";
const RESULT_SCHEMA: &str = "JopeTestAdaptiveIslandDatasetResult.v1";
const SUMMARY_SCHEMA: &str = "JopeTestAdaptiveIslandExtensionSummary.v1";
const MANIFEST_FILE: &str = "island-extension-manifest.v1.json";
const SUMMARY_FILE: &str = "island-extension-summary.v1.json";
const STAGING_FILE: &str = "scq-staging-receipt.v1.json";
const BEST_SOLUTIONS_FILE: &str = "golden42-best-solutions.v1.json";
const BEST_SOLUTIONS_SCHEMA: &str = "JopeGolden42BestSolutions.v1";
const CHAMPION_START_GENERATION: usize = 30;
const CHAMPION_MANIFEST_SCHEMA: &str = "JopeGolden42ChampionContinuationManifest.v1";
const CHAMPION_CAMPAIGN_SCHEMA: &str = "JopeGolden42ChampionDatasetCampaign.v1";
const CHAMPION_CHECKPOINT_SCHEMA: &str = "JopeGolden42ChampionGenerationCheckpoint.v1";
const CHAMPION_RESULT_SCHEMA: &str = "JopeGolden42ChampionDatasetResult.v1";
const CHAMPION_SUMMARY_SCHEMA: &str = "JopeGolden42ChampionContinuationSummary.v1";
const CHAMPION_MANIFEST_FILE: &str = "champion-continuation-manifest.v1.json";
const CHAMPION_SUMMARY_FILE: &str = "champion-continuation-summary.v1.json";

mod island_types;
use island_types::*;
mod champion_types;
use champion_types::*;
mod support;
use support::*;
mod population;
use population::*;
mod campaign;
use campaign::*;
mod checkpoint;
use checkpoint::*;
mod lineage;
use lineage::*;
mod dataset;
use dataset::*;
mod summary;
use summary::*;
mod runner;
use runner::*;
mod champion_manifest;
use champion_manifest::*;
mod champion_checkpoints;
use champion_checkpoints::*;
mod champion_run;
use champion_run::*;
mod champion_summary;
use champion_summary::*;

pub(crate) use champion_summary::{run_champion_continuation, validate_champion_continuation};
pub(crate) use runner::{run_island_extension, validate_island_extension};

#[cfg(test)]
#[path = "island_extension/tests.rs"]
mod tests;
