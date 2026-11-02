/// Checkpoint nodes and parent edges recomputed from a retained lineage table.
#[derive(Clone, Debug)]
pub struct LineageGraph {
    /// Nodes (role → cell) in first-seen order, keyed by checkpoint state.
    pub nodes: Vec<(Key, Dict)>,
    node_at: HashMap<Key, usize>,
    labels: HashMap<Key, usize>,
    pub counts: Dict,
    pub missing_edges: Value,
}

impl LineageGraph {
    fn node(&self, state: &Value) -> Option<&Dict> {
        Key::of(state)
            .and_then(|k| self.node_at.get(&k))
            .map(|&i| &self.nodes[i].1)
    }

    fn label(&self, label: &str) -> Option<&Dict> {
        self.labels.get(&Key::str(label)).map(|&i| &self.nodes[i].1)
    }
}

fn prefix12(value: &Value) -> Result<String> {
    value
        .as_str()
        .map(|s| s.chars().take(12).collect())
        .ok_or_else(|| shape(format!("{} is not subscriptable", abbreviated(value))))
}

/// Checkpoint nodes and parent edges from a retained table; one node per checkpoint state.
pub fn lineage_graph(root: &Path, lineage: &Value, zstd: Zstd) -> Result<LineageGraph> {
    let lid = index(lineage, "id")?.py_str();
    let columns = lineage
        .get("columns")
        .cloned()
        .unwrap_or(Value::Dict(Dict::new()));
    let columns_ok = columns.as_dict().is_some_and(|c| {
        c.len() == LINEAGE_COLUMNS.len() && LINEAGE_COLUMNS.iter().all(|k| c.contains_key(k))
    });
    require!(columns_ok, "{lid}: lineage columns");
    let columns = columns.as_dict().expect("checked above").clone();
    let text = decode_utf8(load(root, index(lineage, "table")?, zstd)?, "lineage table")?;
    let mut graph = LineageGraph {
        nodes: Vec::new(),
        node_at: HashMap::new(),
        labels: HashMap::new(),
        counts: Dict::new(),
        missing_edges: Value::Null,
    };
    for row in pycsv::dict_reader(&text, Newlines::Universal)?.rows {
        let mut node = Dict::new();
        for (role, column) in columns.iter() {
            let column = column
                .as_str()
                .ok_or_else(|| shape(format!("{lid}: lineage column names must be strings")))?;
            let cell = row
                .get(column)
                .ok_or_else(|| shape(format!("KeyError: {}", pyjson::str_repr(column))))?;
            node.insert(role, cell.clone());
        }
        let state = node.get("state").expect("columns checked").clone();
        let key = key_of(&state)?;
        match graph.node_at.get(&key) {
            Some(&i) => require!(
                graph.nodes[i].1 == node,
                "{lid}: checkpoint {}… is recorded inconsistently",
                prefix12(&state)?
            ),
            None => {
                graph.node_at.insert(key.clone(), graph.nodes.len());
                graph.nodes.push((key, node));
            }
        }
    }
    for (i, (_, node)) in graph.nodes.iter().enumerate() {
        graph
            .labels
            .insert(key_of(node.get("label").expect("columns checked"))?, i);
    }
    require!(
        graph.labels.len() == graph.nodes.len(),
        "{lid}: checkpoint labels are not unique"
    );
    let mut missing: Vec<((Value, Value), usize)> = Vec::new();
    for (_, node) in &graph.nodes {
        let parent = node.get("parent").expect("columns checked");
        if graph.node(parent).is_none() {
            let key = (
                parent.clone(),
                node.get("parent_label").expect("columns checked").clone(),
            );
            match missing.iter_mut().find(|(k, _)| *k == key) {
                Some(slot) => slot.1 += 1,
                None => missing.push((key, 1)),
            }
        }
    }
    let distinct = |role: &str| -> Result<usize> {
        let keys: Result<HashSet<Key>> = graph
            .nodes
            .iter()
            .map(|(_, n)| key_of(n.get(role).expect("columns checked")))
            .collect();
        Ok(keys?.len())
    };
    let unretained: usize = missing.iter().map(|(_, n)| n).sum();
    let mut counts = Dict::new();
    counts.insert("checkpoints", graph.nodes.len().into());
    counts.insert("blocks", distinct("block")?.into());
    counts.insert("corpus_identities", distinct("corpus")?.into());
    counts.insert(
        "retained_parent_edges",
        (graph.nodes.len() - unretained).into(),
    );
    counts.insert("unretained_parent_edges", unretained.into());
    if missing.len() > 1
        && missing
            .iter()
            .any(|((state, label), _)| state.as_str().is_none() || label.as_str().is_none())
    {
        return Err(shape(format!("{lid}: missing edges cannot be ordered")));
    }
    missing.sort_by(|((a, b), _), ((c, d), _)| {
        (a.as_str(), b.as_str()).cmp(&(c.as_str(), d.as_str()))
    });
    let edges = missing
        .into_iter()
        .map(|((state, label), children)| {
            let mut edge = Dict::new();
            edge.insert("parent_state_sha256", state);
            edge.insert("parent_label", label);
            edge.insert("children", children.into());
            Value::Dict(edge)
        })
        .collect();
    graph.counts = counts;
    graph.missing_edges = Value::List(edges);
    Ok(graph)
}

/// A lineage's disposition rules; its recomputed counts when it has a retained table.
pub fn check_lineage(
    root: &Path,
    lineage: &Value,
    seen: &mut HashSet<Key>,
    zstd: Zstd,
) -> Result<Value> {
    let lid = lineage.get("id");
    let shown = lid.map_or_else(|| "None".to_owned(), Value::py_str);
    let fresh = match lid {
        Some(id) if id.truthy() => !seen.contains(&key_of(id)?),
        _ => false,
    };
    require!(fresh, "{shown}: lineage id");
    seen.insert(key_of(lid.expect("checked above"))?);
    require!(
        lineage
            .get("disposition")
            .and_then(Value::as_str)
            .is_some_and(|d| DISPOSITIONS.contains(&d))
            && truthy_field(lineage, "limitations"),
        "{shown}: lineage disposition and limitations"
    );
    require!(
        matches!(
            lineage.get("exposure_edges").and_then(Value::as_str),
            Some("retained" | "not-retained")
        ),
        "{shown}: exposure edges"
    );
    if lineage.get("table").is_none() {
        require!(
            lineage.get("disposition") == Some(&Value::from("irrecoverable"))
                && truthy_field(lineage, "missing"),
            "{shown}: a lineage without a retained table is irrecoverable and names what is missing"
        );
        return Ok(Value::Null);
    }
    let graph = lineage_graph(root, lineage, zstd)?;
    require!(
        lineage.get("expected") == Some(&Value::Dict(graph.counts.clone())),
        "{shown}: recorded counts differ from the table"
    );
    require!(
        lineage.get("missing_edges") == Some(&graph.missing_edges),
        "{shown}: recorded missing edges differ from the table"
    );
    Ok(Value::Dict(graph.counts))
}
