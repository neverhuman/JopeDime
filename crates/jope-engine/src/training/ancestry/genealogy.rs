use super::*;

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

/// A compact view of the genealogy whose every digest can be checked
/// against the retained files independently.
pub fn describe(g: &Genealogy) -> Value {
    json!({"schema":SCHEMA,
        "nodes":g.nodes.values().map(|n| json!({"digest":n.digest,"step":n.step,
            "weights_retained":n.weights_retained,"optimizer_retained":n.optimizer_retained}))
            .collect::<Vec<_>>(),
        "edges":g.edges.values().collect::<Vec<_>>()})
}
