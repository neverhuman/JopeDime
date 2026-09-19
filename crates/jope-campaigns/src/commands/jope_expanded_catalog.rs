//! Production CLI for Rust-only expanded cache/catalog operations.

use anyhow::{bail, ensure, Context, Result};
use clap::{Args, Parser, Subcommand};
use crate::paths;
use crate::selfplay_expansion::{
    activate_tranche, adaptive_inventory, audit_discover, freeze_sealed, install_sidecars,
    load_inventory_file, materialize_cache, merge_live, select_unactivated_cached, status,
    write_conflict_summary, write_inventory_file, ExpansionPaths, InventoryEntry, TASKS,
    TRANCHE_MAX,
};
use serde_json::{json, Value};
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(
    name = "jope-expanded-catalog",
    about = "Rust-only expanded dataset/cache/catalog controller"
)]
struct Cli {
    #[command(flatten)]
    paths: PathArgs,
    #[command(subcommand)]
    command: Action,
}

#[derive(Clone, Debug, Args)]
struct PathArgs {
    #[arg(long, default_value_os_t = paths::quant_root())]
    source_root: PathBuf,
    #[arg(long, default_value_os_t = paths::quant_mirror_root())]
    mirror_root: PathBuf,
    #[arg(long, default_value_os_t = paths::flywheel_encodings_root())]
    cache_root: PathBuf,
    #[arg(
        long,
        default_value_os_t = paths::flywheel_root().join("catalog/v2")
    )]
    catalog_root: PathBuf,
    #[arg(
        long,
        default_value_os_t = paths::flywheel_root().join("receipts/mirror-conflicts")
    )]
    receipt_root: PathBuf,
    #[arg(
        long,
        default_value_os_t = paths::flywheel_root().join("catalog/private/sealed-expanded-v1.json")
    )]
    sealed_path: PathBuf,
    #[arg(
        long,
        default_value_os_t = paths::flywheel_encodings_root().join("foundation-v1/standardization/sha256/daac224cf7c6d13a1bf74fc3bb7defd36f33bc3f30db35d3e881c16ede976d1f.json")
    )]
    standardization: PathBuf,
}

impl From<PathArgs> for ExpansionPaths {
    fn from(value: PathArgs) -> Self {
        Self {
            source_root: value.source_root,
            mirror_root: value.mirror_root,
            cache_root: value.cache_root,
            catalog_root: value.catalog_root,
            receipt_root: value.receipt_root,
            sealed_path: value.sealed_path,
            standardization: value.standardization,
        }
    }
}

#[derive(Debug, Subcommand)]
enum Action {
    /// Hash and mirror-audit the authoritative candidate tree.
    Audit {
        #[arg(long)]
        task: Option<String>,
        #[arg(long)]
        limit: Option<usize>,
        #[arg(long)]
        output: PathBuf,
        /// Freeze the private 256-per-task sealed cohort from this inventory.
        #[arg(long)]
        freeze_sealed: bool,
        #[arg(long, default_value_t = 20_260_726)]
        sealed_seed: u64,
    },
    /// Materialize immutable caches from a previously audited inventory.
    Materialize {
        #[arg(long)]
        inventory: PathBuf,
        #[arg(long)]
        task: Option<String>,
        #[arg(long)]
        dataset_id: Option<String>,
        #[arg(long)]
        limit: Option<usize>,
        /// Optional native ELF encoder. Scripts and Python are rejected.
        #[arg(long)]
        rust_encoder: Option<PathBuf>,
    },
    /// Activate one immutable, fully encoded adaptive tranche.
    Activate {
        #[arg(long)]
        inventory: PathBuf,
        #[arg(long)]
        tranche_id: String,
        #[arg(long, default_value_t = TRANCHE_MAX)]
        limit: usize,
    },
    /// Validate caches, private sealed receipt, pointer, blob, and catalog.
    Status,
    /// Fold every activated v2 tranche into the live v1-schema catalog pointer.
    MergeLive {
        #[arg(
            long,
            default_value_os_t = paths::flywheel_root().join("catalog")
        )]
        live_catalog_root: PathBuf,
        /// Tranche ids to leave out of the merged catalog (selective rollback).
        #[arg(long)]
        exclude_tranche: Vec<String>,
        /// Validate and report without writing anything.
        #[arg(long)]
        dry_run: bool,
    },
    /// Additively install encoding sidecars into the authoritative registry.
    InstallSidecars {
        #[arg(long)]
        from: PathBuf,
        #[arg(long)]
        task: String,
        #[arg(long)]
        receipts: PathBuf,
    },
}

fn task_valid(task: Option<&str>) -> bool {
    task.is_none_or(|task| TASKS.contains(&task))
}

fn filter_inventory(
    rows: Vec<InventoryEntry>,
    task: Option<&str>,
    dataset_id: Option<&str>,
    limit: Option<usize>,
) -> Vec<InventoryEntry> {
    rows.into_iter()
        .filter(|row| task.is_none_or(|task| row.task == task))
        .filter(|row| dataset_id.is_none_or(|id| row.dataset_id == id))
        .take(limit.unwrap_or(usize::MAX))
        .collect()
}

fn print_json(value: &Value) -> Result<()> {
    println!("{}", serde_json::to_string(value)?);
    Ok(())
}

pub fn main(args: Vec<String>) -> Result<()> {
    let cli = Cli::parse_from(&args);
    let paths: ExpansionPaths = cli.paths.into();
    match cli.command {
        Action::Audit {
            task,
            limit,
            output,
            freeze_sealed: should_freeze,
            sealed_seed,
        } => {
            ensure!(task_valid(task.as_deref()), "unsupported task filter");
            ensure!(
                !should_freeze || (task.is_none() && limit.is_none()),
                "sealed cohort requires a complete all-task inventory"
            );
            let audit = audit_discover(&paths, task.as_deref(), limit)?;
            if !audit.conflicts.is_empty() {
                let summary = write_conflict_summary(&paths, &audit)?;
                bail!(
                    "mirror audit found {} conflicts across {} datasets; inventory and sealed cohort were not published; summary={}",
                    audit.conflicts.len(),
                    audit.inspected,
                    summary.display()
                );
            }
            let discovered = audit.entries;
            let sealed = if should_freeze {
                Some(freeze_sealed(&paths, &discovered, sealed_seed)?)
            } else {
                None
            };
            let entries = if paths.sealed_path.is_file() {
                adaptive_inventory(&paths, &discovered)?
            } else {
                discovered
            };
            let mut summary = write_inventory_file(&output, &entries)?;
            if let Some(sealed) = sealed {
                summary["sealed"] = sealed;
            }
            print_json(&summary)
        }
        Action::Materialize {
            inventory,
            task,
            dataset_id,
            limit,
            rust_encoder,
        } => {
            ensure!(task_valid(task.as_deref()), "unsupported task filter");
            let rows = filter_inventory(
                load_inventory_file(&inventory)?,
                task.as_deref(),
                dataset_id.as_deref(),
                limit,
            );
            ensure!(!rows.is_empty(), "inventory selection is empty");
            let mut counts = json!({"materialized": 0_u64, "reused": 0_u64});
            for row in rows {
                let destination = paths
                    .cache_root
                    .join(format!("{}-expanded", row.task))
                    .join(&row.dataset_id);
                let existed = destination.is_dir();
                materialize_cache(&paths, &row, rust_encoder.as_deref())?;
                let field = if existed { "reused" } else { "materialized" };
                counts[field] = Value::from(
                    counts[field]
                        .as_u64()
                        .context("internal count is invalid")?
                        + 1,
                );
            }
            counts["schema"] = Value::String("JopeExpandedMaterializationResult.v2".to_owned());
            print_json(&counts)
        }
        Action::Activate {
            inventory,
            tranche_id,
            limit,
        } => {
            ensure!((1..=TRANCHE_MAX).contains(&limit), "limit must be 1..=2048");
            ensure!(
                !tranche_id.contains('/') && !tranche_id.contains(".."),
                "tranche id is not path-safe"
            );
            let rows = load_inventory_file(&inventory)?;
            let selected = select_unactivated_cached(&paths, &rows, limit)?;
            print_json(&activate_tranche(&paths, &tranche_id, &selected)?)
        }
        Action::Status => print_json(&status(&paths)?),
        Action::MergeLive {
            live_catalog_root,
            exclude_tranche,
            dry_run,
        } => print_json(&merge_live(
            &paths,
            &live_catalog_root,
            &exclude_tranche,
            dry_run,
        )?),
        Action::InstallSidecars {
            from,
            task,
            receipts,
        } => {
            ensure!(TASKS.contains(&task.as_str()), "unsupported task: {task}");
            print_json(&install_sidecars(&paths, &from, &task, &receipts)?)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn inventory_filter_is_exact_and_bounded() {
        let rows = (0..3)
            .map(|index| InventoryEntry {
                task: if index == 2 {
                    "binary".to_owned()
                } else {
                    "regression".to_owned()
                },
                dataset_id: format!("{index:016x}"),
                dataset_hash: format!("{index:016x}"),
                source: Path::new("/source").join(index.to_string()),
                source_family: format!("family-{index}"),
                train_sha256: "a".repeat(64),
                test_sha256: "b".repeat(64),
                meta_sha256: "c".repeat(64),
                mirror: None,
            })
            .collect();
        let selected = filter_inventory(rows, Some("regression"), None, Some(1));
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].task, "regression");
    }

    #[test]
    fn task_filter_is_fail_closed() {
        assert!(task_valid(None));
        assert!(task_valid(Some("binary")));
        assert!(task_valid(Some("timeseries")));
        assert!(task_valid(Some("multiclass")));
        // The on-disk compatibility directory name is NOT a task string.
        assert!(!task_valid(Some("multi_label")));
    }
}
