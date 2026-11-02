//! Explicit continuation-objective corpus. Direct-score ledgers are preserved, never rewritten.
use crate::{
    manifest::{hash, resolve},
    native_genome::{self, Arm, Candidate, ContinuationTarget},
    native_graph::Artifact,
};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Write,
    path::{Path, PathBuf},
};
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetSet {
    pub schema: String,
    pub run: String,
    pub generation: u64,
    pub source_graph_state: PathBuf,
    pub qualification: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preparation_overlay: Option<Artifact>,
    pub parent_corpus: Artifact,
    pub vocabulary: Artifact,
    pub admission_policy: Artifact,
    pub frozen_scientific_policy: Artifact,
    pub targets: Vec<Artifact>,
    pub training_arms: BTreeSet<Arm>,
}
fn immutable(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().context("corpus parent absent")?;
    fs::create_dir_all(parent)?;
    let (mut f, tmp) = tempfile::Builder::new()
        .prefix(".continuation-pending-")
        .tempfile_in(parent)?
        .keep()?;
    f.write_all(bytes)?;
    f.sync_all()?;
    match crate::native_persistence::publish_staged(&tmp, path) {
        Ok(()) => (),
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => ensure!(
            fs::read(path)? == bytes,
            "conflicting continuation corpus bytes"
        ),
        Err(e) => return Err(e.into()),
    }
    fs::File::open(parent)?.sync_all()?;
    crate::native_persistence::discard_staging(&tmp)?;
    fs::File::open(parent)?.sync_all()?;
    Ok(())
}
/// Reconstruct source closure from immutable native admissions and producer receipts.
/// Original archived admissions remain mandatory; no graph edge is replaced with a success claim.
mod input_closure;
mod render;
pub use input_closure::input_closure;
pub use render::{build, render};

/// plain_seeded_continuation_v1 corpus minimum: undersized contexts are refused, never padded.
pub(crate) const MINIMUM_ADMITTED_TARGETS_PER_CONTEXT: usize = 8;
/// Refused continuation targets stay in the target set and provenance but never become labels.
fn training_refusal(target: &ContinuationTarget) -> Option<String> {
    if target.training_admission != native_genome::TRAINING_ADMITTED {
        Some(target.training_admission.clone())
    } else if target.quality_rmse_cut_normalized.is_none() {
        Some("no_measured_quality".into())
    } else {
        None
    }
}
/// Deterministic split by context key; eligible contexts keep all of their admitted targets.
type ContextPartition<T> = (BTreeMap<String, Vec<T>>, Vec<(String, Vec<T>)>);
fn eligible_contexts<T>(groups: BTreeMap<String, Vec<T>>, minimum: usize) -> ContextPartition<T> {
    let mut eligible = BTreeMap::new();
    let mut undersized = Vec::new();
    for (key, group) in groups {
        if group.len() >= minimum {
            eligible.insert(key, group);
        } else {
            undersized.push((key, group));
        }
    }
    (eligible, undersized)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn output_conflicts_preserve_original_bytes() {
        let t = tempfile::tempdir().unwrap();
        let p = t.path().join("x");
        immutable(&p, b"old").unwrap();
        assert!(immutable(&p, b"new").is_err());
        assert_eq!(fs::read(p).unwrap(), b"old");
    }
    #[test]
    fn large_original_index_transitive_edges_are_retained_and_corruption_refused() {
        let t = tempfile::tempdir().unwrap();
        let write = |name: &str, bytes: &[u8]| {
            fs::write(t.path().join(name), bytes).unwrap();
            Artifact {
                path: name.into(),
                sha256: hash(bytes),
            }
        };
        let leaf = write("original.bin", b"original dependency bytes");
        let index = write(
            "index.json",
            &serde_json::to_vec(&json!({"schema":"parallel.teaching_shards.v1",
            "retained_metadata":"x".repeat(17 * 1024 * 1024),"shards":[leaf]}))
            .unwrap(),
        );
        let other = write("policy.json", b"{}");
        let set = TargetSet {
            preparation_overlay: None,
            schema: "jope.continuation_target_set.v1".into(),
            run: "r".into(),
            generation: 0,
            source_graph_state: "archived".into(),
            qualification: false,
            parent_corpus: index,
            vocabulary: other.clone(),
            admission_policy: other.clone(),
            frozen_scientific_policy: other,
            targets: vec![],
            training_arms: BTreeSet::from([Arm::Model]),
        };
        let manifest = write("targets.json", &serde_json::to_vec(&set).unwrap());
        let closure = input_closure(t.path(), &manifest, &t.path().join("archived")).unwrap();
        assert_eq!(closure.get(&leaf.path), Some(&leaf.sha256));
        fs::write(t.path().join(&leaf.path), b"changed dependency").unwrap();
        assert!(input_closure(t.path(), &manifest, &t.path().join("archived")).is_err());
    }
    #[test]
    fn undersized_contexts_are_refused_while_eligible_contexts_pack() {
        let groups = BTreeMap::from([
            ("a".to_string(), vec![0; 8]),
            ("b".to_string(), vec![0; 7]),
            ("c".to_string(), vec![0; 12]),
        ]);
        let (eligible, undersized) =
            eligible_contexts(groups, MINIMUM_ADMITTED_TARGETS_PER_CONTEXT);
        assert_eq!(eligible.keys().cloned().collect::<Vec<_>>(), vec!["a", "c"]);
        assert_eq!(eligible["c"].len(), 12);
        assert_eq!(undersized.len(), 1);
        assert_eq!((undersized[0].0.as_str(), undersized[0].1.len()), ("b", 7));
        let only_small = BTreeMap::from([("z".to_string(), vec![0; 3])]);
        let (eligible, undersized) = eligible_contexts(only_small, 8);
        assert!(eligible.is_empty());
        assert_eq!(undersized.len(), 1);
    }
    #[test]
    fn refused_continuation_targets_never_become_training_labels() {
        let a = Artifact {
            path: "target.json".into(),
            sha256: "a".repeat(64),
        };
        let mut target = ContinuationTarget {
            schema: "jope.continuation_target.v2".into(),
            protocol: native_genome::CONTINUATION_PROTOCOL.into(),
            candidate: a.clone(),
            direct: a.clone(),
            continuation: a.clone(),
            parent_genome_sha256: "b".repeat(64),
            winning_original_observation: Some(a.clone()),
            winning_row: Some(0),
            quality_rmse_cut_normalized: Some(0.5),
            fit_runtime_seconds: 31.,
            completed_evaluations: 1,
            failed_evaluations: 1,
            completed_children: 0,
            training_admission: native_genome::REFUSED_NO_COMPLETED_CHILD.into(),
            evaluation_budget: 24,
            search_seconds_budget: 54,
            target_kind: "best_observed_outcome_after_plain_seeded_continuation_v1".into(),
        };
        assert_eq!(
            training_refusal(&target).as_deref(),
            Some(native_genome::REFUSED_NO_COMPLETED_CHILD)
        );
        target.completed_children = 1;
        target.training_admission = native_genome::TRAINING_ADMITTED.into();
        assert_eq!(training_refusal(&target), None);
        target.quality_rmse_cut_normalized = None;
        assert_eq!(
            training_refusal(&target).as_deref(),
            Some("no_measured_quality")
        );
    }
}
