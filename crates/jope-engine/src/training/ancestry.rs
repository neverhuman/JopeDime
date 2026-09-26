//! Accepted ancestry and exposure reconciliation (W6-T07).
//!
//! The genealogy is rebuilt only from retained checkpoint manifests and the
//! consumption journals they bind. Nothing is inferred from step labels,
//! directory names or hardware.
//!
//! **Nodes and lineages.** Every retained checkpoint is a node identified by
//! the SHA-256 of its `state.json`. Checkpoints of one lineage share the
//! lineage record (`initialization`) byte for byte. A manifest without a
//! lineage record is its own orphan lineage.
//!
//! **Parents** are two separate relationships, weights and optimizer state:
//! - within a lineage, the previous retained checkpoint is both parents;
//! - a lineage start names its weight parent by state digest. It has an
//!   optimizer parent only when it carried the optimizer.
//!
//! **Unresolved ancestry.** A parent that is not retained stays unresolved.
//! It is listed, contributes no exposure, and a later run never bridges it.
//!
//! **Refusals:** two checkpoints of one lineage at one step, a lineage
//! record claiming exact resume, a claim about a retained parent that the
//! parent contradicts, a cycle, a bound journal that was altered, and
//! journal exposure that disagrees with committed steps.
//!
//! **Exposure** has three quantities, never merged:
//! - `possible`: committed optimizer steps × global batch. It follows from
//!   verified step counters alone; record identities are unknown.
//! - `exact`: committed presentations, supervised positions and unique
//!   records, taken only from byte-bound consumption journals.
//! - `availability`: the frozen training-input manifest of each segment.
//!
//! A union of several targets counts each shared ancestor once.
use super::consumption::{self, Binding, Units};
use crate::manifest::{file_hash, hash, resolve};
use anyhow::{ensure, Context, Result};
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

pub const SCHEMA: &str = "jope.training_ancestry.v1";

#[derive(Clone, Debug)]
pub struct Node {
    pub digest: String,
    pub dir: PathBuf,
    pub step: u64,
    pub state: Value,
    pub weights_retained: bool,
    pub optimizer_retained: bool,
}

/// Read one retained checkpoint. Objects that are absent or differ from
/// their manifest digest are reported as not retained; the manifest itself
/// must be a readable checkpoint manifest.
pub fn load_node(dir: &Path) -> Result<Node> {
    let bytes = fs::read(dir.join("state.json"))
        .with_context(|| format!("checkpoint manifest missing: {}", dir.display()))?;
    let state: Value = serde_json::from_slice(&bytes).context("malformed checkpoint manifest")?;
    ensure!(
        state["schema"] == "parallel.checkpoint.v1",
        "not a checkpoint manifest: {}",
        dir.display()
    );
    let step = state["step"].as_u64().context("checkpoint step absent")?;
    let retained = |name: &str, field: &str| {
        let path = dir.join(name);
        path.is_file()
            && file_hash(&path)
                .ok()
                .is_some_and(|digest| state[field] == digest.as_str())
    };
    Ok(Node {
        digest: hash(&bytes),
        dir: dir.to_path_buf(),
        step,
        weights_retained: retained("model.ot", "model_sha256"),
        optimizer_retained: retained("optimizer.ot", "optimizer_sha256")
            && retained("optimizer.json", "optimizer_metadata_sha256"),
        state,
    })
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Link {
    /// No parent: cold initialization, or an optimizer reset.
    None,
    Resolved {
        digest: String,
    },
    /// A claimed or unknowable parent that is not retained.
    Unresolved {
        digest: Option<String>,
    },
}

#[derive(Clone, Debug, Serialize)]
pub struct Edge {
    pub child: String,
    pub child_step: u64,
    /// `same_lineage` or `lineage_start`.
    pub kind: &'static str,
    pub mode: String,
    /// Optimizer step at which this segment's updates begin; `None` when the
    /// lineage record cannot be interpreted.
    pub from_step: Option<u64>,
    pub weights: Link,
    pub optimizer: Link,
    /// `none`, `reset`, `retained`, `not_retained` or `unknown`.
    pub optimizer_state: &'static str,
}

#[derive(Debug)]
pub struct Genealogy {
    pub nodes: BTreeMap<String, Node>,
    pub edges: BTreeMap<String, Edge>,
}

fn lineage_key(node: &Node) -> Result<String> {
    let init = &node.state["initialization"];
    Ok(if init.is_object() {
        hash(&serde_json::to_vec(init)?)
    } else {
        format!("orphan:{}", node.digest)
    })
}

fn lineage_mode(node: &Node) -> String {
    node.state["initialization"]
        .get("continuation_mode")
        .and_then(Value::as_str)
        .unwrap_or("unknown_legacy")
        .to_string()
}

fn resolve_link(nodes: &BTreeMap<String, Node>, digest: Option<&str>) -> Link {
    match digest {
        Some(d) if nodes.contains_key(d) => Link::Resolved { digest: d.into() },
        other => Link::Unresolved {
            digest: other.map(String::from),
        },
    }
}

/// The start of a lineage as its record states it. Claims about a retained
/// parent must agree with that parent.
fn lineage_start(nodes: &BTreeMap<String, Node>, child: &Node) -> Result<Edge> {
    let init = &child.state["initialization"];
    ensure!(
        init["exact_same_experiment_resume"] != true,
        "a lineage record claims exact resume; only an attempt can resume exactly"
    );
    let edge = |from_step, weights, optimizer, optimizer_state| Edge {
        child: child.digest.clone(),
        child_step: child.step,
        kind: "lineage_start",
        mode: lineage_mode(child),
        from_step,
        weights,
        optimizer,
        optimizer_state,
    };
    match init["kind"].as_str() {
        Some("fresh") => Ok(edge(Some(0), Link::None, Link::None, "none")),
        Some("warm_start") => {
            let claimed = init["state_sha256"]
                .as_str()
                .context("warm start names no parent state")?;
            let weights = resolve_link(nodes, Some(claimed));
            let carried = init
                .get("optimizer_parent_step")
                .filter(|v| !v.is_null())
                .map(|v| v.as_u64().context("optimizer parent step malformed"))
                .transpose()?;
            if let Link::Resolved { digest } = &weights {
                let parent = &nodes[digest];
                ensure!(
                    init["parent_step"].as_u64() == Some(parent.step)
                        && carried.is_none_or(|step| step == parent.step),
                    "lineage record claims a parent step the retained parent does not have"
                );
                let continuity = &init["continuity"];
                if !continuity.is_null() {
                    let parent_objective = parent
                        .state
                        .get("objective_sha256")
                        .cloned()
                        .unwrap_or(Value::Null);
                    ensure!(
                        continuity["parent_step"].as_u64() == Some(parent.step)
                            && continuity["objective"]["parent_sha256"] == parent_objective
                            && continuity["corpus"]["parent"] == parent.state["corpus_sha256"],
                        "continuity record contradicts the retained parent"
                    );
                }
            }
            Ok(match carried {
                Some(step) => {
                    let optimizer_state = match &weights {
                        Link::Resolved { digest } if nodes[digest].optimizer_retained => "retained",
                        Link::Resolved { .. } => "not_retained",
                        _ => "unknown",
                    };
                    edge(Some(step), weights.clone(), weights, optimizer_state)
                }
                None => edge(Some(0), weights, Link::None, "reset"),
            })
        }
        _ => Ok(edge(
            None,
            Link::Unresolved { digest: None },
            Link::Unresolved { digest: None },
            "unknown",
        )),
    }
}

/// Build and validate the genealogy of `nodes`. The same checkpoint
/// ingested more than once is one node.
pub fn build(nodes: impl IntoIterator<Item = Node>) -> Result<Genealogy> {
    let mut by_digest = BTreeMap::new();
    for node in nodes {
        by_digest.entry(node.digest.clone()).or_insert(node);
    }
    let mut lineages = BTreeMap::<String, Vec<&Node>>::new();
    for node in by_digest.values() {
        lineages.entry(lineage_key(node)?).or_default().push(node);
    }
    let mut edges = BTreeMap::new();
    for members in lineages.values_mut() {
        members.sort_by_key(|n| n.step);
        for pair in members.windows(2) {
            ensure!(
                pair[0].step != pair[1].step,
                "contradictory checkpoints claim step {} of one lineage",
                pair[0].step
            );
        }
        for (i, node) in members.iter().enumerate() {
            let edge = if i == 0 {
                lineage_start(&by_digest, node)?
            } else {
                let parent = members[i - 1];
                Edge {
                    child: node.digest.clone(),
                    child_step: node.step,
                    kind: "same_lineage",
                    mode: lineage_mode(node),
                    from_step: Some(parent.step),
                    weights: Link::Resolved {
                        digest: parent.digest.clone(),
                    },
                    optimizer: Link::Resolved {
                        digest: parent.digest.clone(),
                    },
                    optimizer_state: if parent.optimizer_retained {
                        "retained"
                    } else {
                        "not_retained"
                    },
                }
            };
            edges.insert(node.digest.clone(), edge);
        }
    }
    // Digest-addressed claims cannot form a cycle; refuse one defensively.
    for start in edges.keys() {
        let mut seen = BTreeSet::new();
        let mut current = start.clone();
        while let Link::Resolved { digest } = &edges[&current].weights {
            ensure!(
                seen.insert(current.clone()),
                "ancestry cycle through checkpoint {current}"
            );
            current = digest.clone();
        }
    }
    Ok(Genealogy {
        nodes: by_digest,
        edges,
    })
}

/// Load every checkpoint below `run/checkpoints/step-*` of each run.
pub fn load_runs(runs: &[PathBuf]) -> Result<Vec<Node>> {
    let mut nodes = Vec::new();
    for run in runs {
        for entry in fs::read_dir(run.join("checkpoints"))? {
            let entry = entry?;
            if entry.file_type()?.is_dir()
                && entry.file_name().to_string_lossy().starts_with("step-")
            {
                nodes.push(load_node(&entry.path())?);
            }
        }
    }
    Ok(nodes)
}

#[derive(Clone, Debug, Serialize)]
pub struct Segment {
    pub child: String,
    pub from_step: Option<u64>,
    pub to_step: u64,
    pub corpus_sha256: Value,
    pub possible_presentations: Option<u64>,
    pub exact: Option<Units>,
    /// `journal_bound`, `no_consumption_journal`, `journal_not_retained`,
    /// `journal_starts_after_segment` or `unknown_origin`.
    pub exact_status: &'static str,
    /// The segment's frozen training-input availability, when retained.
    pub availability: Value,
}

#[derive(Clone, Debug, Serialize)]
pub struct Exposure {
    pub schema: &'static str,
    pub targets: Vec<String>,
    pub segments: Vec<Segment>,
    /// Sum over segments with a known step window; complete only when
    /// `possible_complete` holds.
    pub possible_presentations: u64,
    pub possible_complete: bool,
    /// Sum over journal-bound segments; complete only when `exact_complete`
    /// holds.
    pub exact: Units,
    pub exact_complete: bool,
    /// Distinct `(corpus, record)` pairs among journal-bound segments.
    pub unique_committed_records: u64,
    /// Claimed or unknowable ancestors that are not retained: their exposure is
    /// unknown, not zero.
    pub unresolved_ancestors: Vec<Option<String>>,
}

fn add(total: &mut Units, units: &Units) -> Result<()> {
    let sum = |a: u64, b: u64| a.checked_add(b).context("exposure overflow");
    *total = Units {
        microbatches: sum(total.microbatches, units.microbatches)?,
        presentations: sum(total.presentations, units.presentations)?,
        supervised_positions: sum(total.supervised_positions, units.supervised_positions)?,
    };
    Ok(())
}

fn subtract(a: &Units, b: &Units) -> Result<Units> {
    let sub = |x: u64, y: u64| {
        x.checked_sub(y)
            .context("journal prefix exceeds its extension")
    };
    Ok(Units {
        microbatches: sub(a.microbatches, b.microbatches)?,
        presentations: sub(a.presentations, b.presentations)?,
        supervised_positions: sub(a.supervised_positions, b.supervised_positions)?,
    })
}

/// One rank's committed consumption as bound in a checkpoint.
enum Bound {
    /// The manifest binds no consumption journal (written before W6-T04).
    Absent,
    /// The manifest binds a journal that is no longer retained.
    NotRetained,
    Verified {
        path: String,
        units: Units,
        records: BTreeSet<u64>,
        initial: u64,
    },
}

fn bound_journal(root: &Path, node: &Node, rank: usize) -> Result<Bound> {
    let row = &node.state["committed_exposure_by_rank"][rank]["consumption_journal"];
    if row.is_null() {
        return Ok(Bound::Absent);
    }
    let path = row["path"]
        .as_str()
        .context("consumption journal path")?
        .to_string();
    let binding: Binding = serde_json::from_value(row["binding"].clone())?;
    let Ok(bytes) = fs::read(resolve(root, Path::new(&path))?) else {
        return Ok(Bound::NotRetained);
    };
    // A retained journal that no longer holds its bound prefix is altered
    // evidence: an error, never an unknown.
    consumption::verify_binding(&bytes, &binding)?;
    let prefix = &bytes[..binding.bytes as usize];
    let r = consumption::reconcile(prefix, Some(node.step))?;
    Ok(Bound::Verified {
        path,
        units: r.committed,
        records: r.committed_records,
        initial: r.header.initial_optimizer_step,
    })
}

fn availability(root: &Path, node: &Node) -> Value {
    let (Some(expected), Some(receipt)) = (
        node.state["training_inputs_sha256"].as_str(),
        node.state["rank_receipts"][0]["path"].as_str(),
    ) else {
        return Value::Null;
    };
    let Some(launch) = Path::new(receipt).parent() else {
        return Value::Null;
    };
    let Ok(path) = resolve(root, &launch.join("training-inputs.json")) else {
        return Value::Null;
    };
    match fs::read(&path) {
        Ok(bytes) if hash(&bytes) == expected => serde_json::from_slice::<Value>(&bytes)
            .map(|frozen| frozen["availability"].clone())
            .unwrap_or(Value::Null),
        _ => Value::Null,
    }
}

fn segment(root: &Path, g: &Genealogy, edge: &Edge) -> Result<(Segment, BTreeSet<(String, u64)>)> {
    let node = &g.nodes[&edge.child];
    let corpus = node.state["corpus_sha256"].clone();
    let mut segment = Segment {
        child: edge.child.clone(),
        from_step: edge.from_step,
        to_step: node.step,
        corpus_sha256: corpus.clone(),
        possible_presentations: None,
        exact: None,
        exact_status: "unknown_origin",
        availability: availability(root, node),
    };
    let Some(from) = edge.from_step else {
        return Ok((segment, BTreeSet::new()));
    };
    ensure!(
        node.step >= from,
        "checkpoint step {} precedes its segment origin {from}",
        node.step
    );
    let global = node.state["global_batch"]
        .as_u64()
        .context("checkpoint global batch")?;
    let possible = (node.step - from)
        .checked_mul(global)
        .context("possible exposure overflow")?;
    segment.possible_presentations = Some(possible);
    let world = usize::try_from(node.state["world"].as_u64().context("checkpoint world")?)?;
    let parent = match &edge.weights {
        Link::Resolved { digest } if edge.kind == "same_lineage" => Some(&g.nodes[digest]),
        _ => None,
    };
    let mut total = Units::default();
    let mut records = BTreeSet::new();
    for rank in 0..world {
        let (path, units, committed, initial) = match bound_journal(root, node, rank)? {
            Bound::Absent => {
                segment.exact_status = "no_consumption_journal";
                return Ok((segment, BTreeSet::new()));
            }
            Bound::NotRetained => {
                segment.exact_status = "journal_not_retained";
                return Ok((segment, BTreeSet::new()));
            }
            Bound::Verified {
                path,
                units,
                records,
                initial,
            } => (path, units, records, initial),
        };
        // Every record committed in a same-lineage journal prefix was
        // committed on this path, so the union takes the full committed set;
        // only the units are split per segment.
        let units = if initial == from {
            units
        } else {
            // The segment began inside this journal: subtract what the same
            // journal had already committed at the retained parent.
            match parent.map(|p| bound_journal(root, p, rank)).transpose()? {
                Some(Bound::Verified {
                    path: parent_path,
                    units: parent_units,
                    initial: parent_initial,
                    ..
                }) if parent_path == path && parent_initial == initial => {
                    subtract(&units, &parent_units)?
                }
                _ => {
                    segment.exact_status = "journal_starts_after_segment";
                    return Ok((segment, BTreeSet::new()));
                }
            }
        };
        add(&mut total, &units)?;
        let corpus = corpus.as_str().unwrap_or_default().to_string();
        records.extend(committed.into_iter().map(|r| (corpus.clone(), r)));
    }
    ensure!(
        total.presentations == possible,
        "journal-bound exposure {} disagrees with committed steps × batch {possible}",
        total.presentations
    );
    segment.exact = Some(total);
    segment.exact_status = "journal_bound";
    Ok((segment, records))
}

/// Exposure of the accepted ancestry of `targets`: every segment on a path
/// from a target back to its roots, each counted once.
pub fn exposure(root: &Path, g: &Genealogy, targets: &[&str]) -> Result<Exposure> {
    let mut on_path = BTreeSet::new();
    let mut unresolved = Vec::new();
    for &target in targets {
        ensure!(
            g.edges.contains_key(target),
            "target {target} is not a retained checkpoint"
        );
        let mut current = target.to_string();
        while on_path.insert(current.clone()) {
            match &g.edges[&current].weights {
                Link::Resolved { digest } => current = digest.clone(),
                Link::Unresolved { digest } => {
                    if !unresolved.contains(digest) {
                        unresolved.push(digest.clone());
                    }
                    break;
                }
                Link::None => break,
            }
        }
    }
    let mut segments = Vec::new();
    let mut records = BTreeSet::new();
    let mut exact = Units::default();
    let mut possible = 0u64;
    for child in &on_path {
        let (segment, segment_records) = segment(root, g, &g.edges[child])?;
        if let Some(p) = segment.possible_presentations {
            possible = possible
                .checked_add(p)
                .context("possible exposure overflow")?;
        }
        if let Some(units) = &segment.exact {
            add(&mut exact, units)?;
        }
        records.extend(segment_records);
        segments.push(segment);
    }
    let resolved = unresolved.is_empty();
    Ok(Exposure {
        schema: SCHEMA,
        targets: targets.iter().map(|t| t.to_string()).collect(),
        possible_complete: resolved && segments.iter().all(|s| s.possible_presentations.is_some()),
        exact_complete: resolved && segments.iter().all(|s| s.exact.is_some()),
        possible_presentations: possible,
        exact,
        unique_committed_records: records.len() as u64,
        unresolved_ancestors: unresolved,
        segments,
    })
}

/// A compact view of the genealogy whose every digest can be checked
/// against the retained files independently.
pub fn describe(g: &Genealogy) -> Value {
    json!({"schema":SCHEMA,
        "nodes":g.nodes.values().map(|n| json!({"digest":n.digest,"step":n.step,
            "weights_retained":n.weights_retained,"optimizer_retained":n.optimizer_retained}))
            .collect::<Vec<_>>(),
        "edges":g.edges.values().collect::<Vec<_>>()})
}
