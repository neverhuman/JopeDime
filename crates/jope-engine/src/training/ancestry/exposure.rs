use super::*;

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
