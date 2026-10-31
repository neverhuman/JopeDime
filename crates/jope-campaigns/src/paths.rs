//! Campaign-level locations built on the ADR-0003 roots (`jope_paths`).
//!
//! Each function names the absolute path the historical tree hard-coded
//! (written `~/…`: the `ubuntu` account's home on the xbabe hosts) and the
//! root it now hangs off. Source-tree inputs (scripts, manifests, seeds,
//! built binaries) resolve under [`crate::paths::jope_home`]; mutable run state (runs,
//! queues, sqlite stores, pointers, logs) under [`crate::paths::jope_state`]; retired
//! per-host trees keep their names under [`crate::paths::legacy_root`]
//! (`<jope_state>/legacy`). Everything here is resolved at call time;
//! binaries that need a `&'static str` wrap these in `jope_paths::once_str!`.

use std::path::PathBuf;

pub use jope_paths::{
    jope_home, jope_registry, jope_state, legacy_root, quant_root, scq_root,
    sotarena_reporter_root, sotarena_root, var,
};

// ---- retired per-host trees (mutable; under <jope_state>/legacy) ---------

/// A retired single-host tree by its historical directory name
/// (`~/<name>` → `<jope_state>/legacy/<name>`; the `jope-flywheel` host tree
/// is the exception and contributes its children directly, see below).
pub fn legacy_tree(name: &str) -> PathBuf {
    legacy_root().join(name)
}

/// `~/jope-flywheel/test_exact_gen0` — the lineage-A self-play tree
/// (corpus, catalog, receipts, ops, runtime-bundles).
pub fn flywheel_root() -> PathBuf {
    legacy_tree("test_exact_gen0")
}

/// `~/jope-flywheel/training` — trainer rounds and rotation inputs.
pub fn flywheel_training_root() -> PathBuf {
    legacy_tree("training")
}

/// `~/jope-flywheel/training/round1/checkpoints` — lineage-A
/// warm-start checkpoints (`<sha256>.model.pt`).
pub fn legacy_checkpoints_root() -> PathBuf {
    flywheel_training_root().join("round1/checkpoints")
}

/// `~/jope-flywheel/encodings` — the encoding cache
/// (`foundation-v1/`, `expanded-v2/`, `_driver`, `_fetch/`, `_batch_manifests/`).
pub fn flywheel_encodings_root() -> PathBuf {
    legacy_tree("encodings")
}

/// `~/jope-flywheel/scripts/fweval_runner.sh` — the bundled
/// evaluation runner.
pub fn fweval_runner() -> PathBuf {
    legacy_tree("scripts/fweval_runner.sh")
}

/// `~/jope-flywheel/artifacts-v1` — the general artifact CAS.
pub fn artifact_cas_root() -> PathBuf {
    legacy_tree("artifacts-v1")
}

/// `~/jope-flywheel/test_exact_gen0/corpus/strict-test-exact.sqlite`.
pub fn corpus_db() -> PathBuf {
    flywheel_root().join("corpus/strict-test-exact.sqlite")
}

/// `~/jope-flywheel/test_exact_gen0/catalog/current.json`.
pub fn catalog_pointer() -> PathBuf {
    flywheel_root().join("catalog/current.json")
}

/// `~/jope-flywheel/test_exact_gen0/runtime-bundles`.
pub fn runtime_bundles_root() -> PathBuf {
    flywheel_root().join("runtime-bundles")
}

/// `~/jope-flywheel/test_exact_gen0/runtime-bundles/current`.
pub fn runtime_bundle_current() -> PathBuf {
    runtime_bundles_root().join("current")
}

/// `~/jope-flywheel/test_exact_gen0/runtime-bundles/current/pytorch`
/// — the active bundle's Python tree.
pub fn runtime_bundle_pytorch() -> PathBuf {
    runtime_bundle_current().join("pytorch")
}

/// `~/quant` — the local mirror of the authoritative dataset tree
/// ([`quant_root`]).
pub fn quant_mirror_root() -> PathBuf {
    legacy_tree("quant")
}

// ---- the scheduler checkout (JOPE_SCQ_ROOT; was ~/smart_cluster) --

/// `~/smart_cluster/target/release/scq`.
pub fn scq_binary() -> PathBuf {
    scq_root().join("target/release/scq")
}

/// `~/smart_cluster/target/release/scqd`.
pub fn scqd_binary() -> PathBuf {
    scq_root().join("target/release/scqd")
}

/// `~/smart_cluster/var/runs`.
pub fn scq_runs_root() -> PathBuf {
    scq_root().join("var/runs")
}

/// `~/smart_cluster/var/scqd.sock`.
pub fn scq_socket() -> PathBuf {
    scq_root().join("var/scqd.sock")
}

/// `~/smart_cluster/var/events.jsonl`.
pub fn scq_events() -> PathBuf {
    scq_root().join("var/events.jsonl")
}

/// `~/smart_cluster/etc/projects.toml`.
pub fn scq_projects() -> PathBuf {
    scq_root().join("etc/projects.toml")
}

/// `~/super_cluster/scq` — the second (all-node) scq build the
/// test-adaptive campaigns drove; a retired per-host tree.
pub fn super_cluster_scq_binary() -> PathBuf {
    legacy_tree("super_cluster/scq")
}

// ---- source-tree inputs (under JOPE_HOME; was ~/jope-dime) -----

/// `~/jope-prime` — the Python package checkout, now `python/`.
pub fn python_root() -> PathBuf {
    jope_home().join("python")
}

/// `~/jope-prime/src` — the `PYTHONPATH` entry that exposes `jope`.
pub fn python_src() -> PathBuf {
    python_root().join("src")
}

/// `~/jope-dime/scripts` — the self-play Python scripts, now
/// `python/src/jope/research`.
pub fn research_scripts_root() -> PathBuf {
    python_src().join("jope/research")
}

/// `~/jope-dime/scripts/<name>`.
pub fn research_script(name: &str) -> PathBuf {
    research_scripts_root().join(name)
}

/// `~/remote_super/fable_god` — the foundation-expert tree encode
/// jobs import as `python.elo_finetune.model_io`, now `python/legacy/fable_god`.
pub fn foundation_expert_root() -> PathBuf {
    python_root().join("legacy/fable_god")
}

/// `~/jope-dime/manifests` — the campaign manifests, now
/// `crates/jope-campaigns/manifests`.
pub fn campaign_manifests_root() -> PathBuf {
    jope_home().join("crates/jope-campaigns/manifests")
}

/// `~/jope-dime/manifests/<relative>`.
pub fn campaign_manifest(relative: &str) -> PathBuf {
    campaign_manifests_root().join(relative)
}

/// `~/jope-dime/target/release/<name>` — a binary built from the
/// checkout.
pub fn release_binary(name: &str) -> PathBuf {
    jope_home().join("target/release").join(name)
}

/// `~/jope-dime/target/release/corpus_eval`.
pub fn corpus_eval_binary() -> PathBuf {
    release_binary("corpus_eval")
}

// ---- mutable campaign state (under JOPE_STATE) ----------------------------

/// `~/jope-dime/runs`.
pub fn runs_root() -> PathBuf {
    jope_state().join("runs")
}

/// `~/jope-dime/campaigns`.
pub fn campaigns_root() -> PathBuf {
    jope_state().join("campaigns")
}

/// `~/jope-dime/campaigns/test-adaptive`.
pub fn test_adaptive_root() -> PathBuf {
    campaigns_root().join("test-adaptive")
}

/// `~/ratchet-shadow/causal-v2` — the causal shadow store.
pub fn shadow_root() -> PathBuf {
    jope_state().join("ratchet-shadow/causal-v2")
}

/// `~/.config/systemd/user` — the invoking user's unit directory
/// (`$XDG_CONFIG_HOME/systemd/user`, else `~/.config/systemd/user`).
pub fn user_unit_root() -> PathBuf {
    let config = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .filter(|path| path.is_absolute())
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))
        .unwrap_or_else(|| jope_state().join("config"));
    config.join("systemd/user")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derived_locations_hang_off_the_roots() {
        let state = jope_state();
        let home = jope_home();
        assert!(flywheel_root().starts_with(&state));
        assert!(flywheel_root().ends_with("legacy/test_exact_gen0"));
        assert!(legacy_checkpoints_root().ends_with("legacy/training/round1/checkpoints"));
        assert!(corpus_db().ends_with("corpus/strict-test-exact.sqlite"));
        assert!(scq_binary().starts_with(scq_root()));
        assert!(scq_projects().ends_with("etc/projects.toml"));
        assert!(research_script("x.py").starts_with(&home));
        assert_eq!(
            research_script("x.py"),
            home.join("python/src/jope/research/x.py")
        );
        assert_eq!(
            corpus_eval_binary(),
            home.join("target/release/corpus_eval")
        );
        assert_eq!(
            campaign_manifest("a/b.json"),
            home.join("crates/jope-campaigns/manifests/a/b.json")
        );
        assert_eq!(test_adaptive_root(), state.join("campaigns/test-adaptive"));
        assert!(user_unit_root().ends_with("systemd/user"));
    }
}
