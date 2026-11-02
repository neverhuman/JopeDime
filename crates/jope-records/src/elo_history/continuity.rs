/// `next((e for e in document[collection] if e["id"] == id), None)`.
fn find_by_id<'a>(document: &'a Value, collection: &str, id: &str) -> Result<Option<&'a Value>> {
    let entries = list_of(index(document, collection)?, collection)?;
    for entry in entries {
        if index(entry, "id")? == &Value::from(id) {
            return Ok(Some(entry));
        }
    }
    Ok(None)
}

/// Return the permitted scope of a claim about one history, or refuse it.
pub fn authorize(document: &Value, history_id: &str, claim: &str, consumer: &str) -> Result<Value> {
    require!(
        !REFUSED_CONSUMERS.contains(&consumer),
        "{history_id}: Elo is historical-ranking-only; {consumer} consumers are refused"
    );
    require!(
        consumer == CONSUMER,
        "{history_id}: unknown consumer {}",
        pyjson::str_repr(consumer)
    );
    require!(
        CLAIMS.contains(&claim),
        "{history_id}: unknown claim {}",
        pyjson::str_repr(claim)
    );
    let history = find_by_id(document, "histories", history_id)?;
    require!(history.is_some(), "{history_id}: unknown history");
    let history = history.expect("checked above");
    require!(
        claim != "continuity",
        "{history_id}: a rating is relative to its own method universe; continuity across \
         ladders is not data-supported"
    );
    let disposition = index(history, "disposition")?;
    require!(
        disposition == &Value::from(REPRODUCIBLE),
        "{history_id}: {} history supports no {claim} claim",
        disposition.py_str()
    );
    let mut granted = Dict::new();
    granted.insert("history", history_id.into());
    granted.insert("claim", claim.into());
    granted.insert("consumer", CONSUMER.into());
    granted.insert("scope", index(history, "scope")?.clone());
    Ok(Value::Dict(granted))
}

/// Parameter-ancestry continuity between two retained checkpoints, through retained edges only.
pub fn continuity(
    root: &Path,
    document: &Value,
    lineage_id: &str,
    ancestor: &str,
    descendant: &str,
    zstd: Zstd,
) -> Result<Value> {
    let lineage = find_by_id(document, "lineages", lineage_id)?;
    require!(lineage.is_some(), "{lineage_id}: unknown lineage");
    let lineage = lineage.expect("checked above");
    require!(
        lineage.get("table").is_some(),
        "{lineage_id}: irrecoverable lineage supports no continuity claim"
    );
    let graph = lineage_graph(root, lineage, zstd)?;
    require!(
        graph.label(ancestor).is_some(),
        "{lineage_id}: {ancestor} is not a retained checkpoint"
    );
    let node = graph.label(descendant);
    require!(
        node.is_some(),
        "{lineage_id}: {descendant} is not a retained checkpoint"
    );
    let mut node = node.expect("checked above");
    let field = |node: &Dict, role: &str| node.get(role).expect("columns checked").clone();
    let mut path = vec![Value::from(descendant)];
    let mut visited: HashSet<Key> = HashSet::from([key_of(&field(node, "state"))?]);
    while field(node, "label") != Value::from(ancestor) {
        let parent = graph.node(&field(node, "parent"));
        require!(
            parent.is_some(),
            "{lineage_id}: missing lineage edge above {} (parent {}, {}… not retained)",
            field(node, "label").py_str(),
            field(node, "parent_label").py_str(),
            prefix12(&field(node, "parent"))?
        );
        let parent = parent.expect("checked above");
        let state = key_of(&field(parent, "state"))?;
        require!(
            !visited.contains(&state),
            "{lineage_id}: parent cycle at {}",
            field(parent, "label").py_str()
        );
        visited.insert(state);
        node = parent;
        path.push(field(node, "label"));
    }
    let mut granted = Dict::new();
    granted.insert("lineage", lineage_id.into());
    granted.insert("path", Value::List(path));
    granted.insert(
        "meaning",
        "parameter ancestry through recorded block-initialization edges; each block trains on \
         its own corpus"
            .into(),
    );
    Ok(Value::Dict(granted))
}

/// Exposure totals of a lineage, refused unless its corpus-to-dataset edges are retained.
pub fn exposure(document: &Value, lineage_id: &str) -> Result<Value> {
    let lineage = find_by_id(document, "lineages", lineage_id)?;
    require!(lineage.is_some(), "{lineage_id}: unknown lineage");
    let lineage = lineage.expect("checked above");
    require!(
        index(lineage, "exposure_edges")? == &Value::from("retained"),
        "{lineage_id}: corpus-to-dataset exposure edges are not retained; exposure totals refused"
    );
    Ok(index(lineage, "exposure")?.clone())
}

/// A committed plot table must repeat its snapshots' rows; it adds no history of its own.
/// Returns its row count.
pub fn derived_view(
    root: &Path,
    view: &Value,
    histories: &HashMap<Key, &Value>,
    zstd: Zstd,
) -> Result<usize> {
    let vid = index(view, "id")?.py_str();
    let text = decode_utf8(
        load(root, index(view, "table")?, zstd)?,
        "derived view table",
    )?;
    let rows = pycsv::dict_reader(&text, Newlines::LineFeed)?.rows;
    let mut snapshots: Vec<(String, Value, HashMap<Key, Value>)> = Vec::new();
    let named = index(view, "snapshots")?
        .as_dict()
        .ok_or_else(|| shape(format!("{vid}: snapshots is not an object")))?;
    for (name, history_id) in named.iter() {
        let history = Key::of(history_id).and_then(|k| histories.get(&k)).copied();
        require!(
            history.is_some_and(|h| h.get("disposition") == Some(&Value::from(REPRODUCIBLE))),
            "{vid}: {name} is not backed by a reproducible history"
        );
        let output = index(
            index(index(history.expect("checked above"), "replay")?, "outputs")?,
            "snapshot",
        )?;
        let snapshot = json_bytes(&load(root, output, zstd)?, "snapshot")?;
        let mut board = HashMap::new();
        for row in py::iterate(index(&snapshot, "rows")?).ok_or_else(|| shape("snapshot rows"))? {
            board.insert(key_of(index(&row, "method_id")?)?, row);
        }
        snapshots.push((name.to_owned(), index(output, "sha256")?.clone(), board));
    }
    let cell = |row: &Dict, column: &str| -> Result<Value> {
        row.get(column)
            .cloned()
            .ok_or_else(|| shape(format!("KeyError: {}", pyjson::str_repr(column))))
    };
    for (offset, row) in rows.iter().enumerate() {
        let number = offset + 2;
        let snapshot = cell(row, "snapshot")?;
        let backing = snapshots
            .iter()
            .find(|(name, _, _)| snapshot == Value::from(name.as_str()));
        require!(backing.is_some(), "{vid} line {number}: unbacked snapshot");
        let (_, digest, board) = backing.expect("checked above");
        require!(
            &cell(row, "source_sha256")? == digest,
            "{vid} line {number}: source digest"
        );
        let source = Key::of(&cell(row, "method")?).and_then(|k| board.get(&k));
        let mut same = source.is_some();
        if let Some(source) = source {
            for (k, v) in [
                ("elo_rank", "rank"),
                ("elo", "elo"),
                ("datasets_scored", "datasets_scored"),
                ("datasets_total", "datasets_total"),
                ("coverage_pct", "coverage_pct"),
            ] {
                if Value::from(index(source, k)?.py_str()) != cell(row, v)? {
                    same = false;
                    break;
                }
            }
        }
        require!(same, "{vid} line {number}: differs from its snapshot");
    }
    for (name, _, board) in &snapshots {
        let plotted = rows
            .iter()
            .filter(|row| row.get("snapshot") == Some(&Value::from(name.as_str())))
            .count();
        require!(
            plotted == board.len(),
            "{vid}: a snapshot is only partly plotted"
        );
    }
    Ok(rows.len())
}

/// What `elo-history` reports.
#[derive(Clone, Debug)]
pub struct Summary {
    pub replayed: Vec<String>,
    pub not_replayed_here: BTreeMap<String, String>,
    pub scorer_parity_only: Vec<String>,
    pub irrecoverable: Vec<String>,
    pub excluded_from_exact_totals: Vec<String>,
    pub replays: BTreeMap<String, ReplaySummary>,
    pub lineages: BTreeMap<String, Value>,
    pub derived_view_rows: BTreeMap<String, usize>,
}

impl Summary {
    pub fn exact_rankable_rows(&self) -> usize {
        self.replays.values().map(|r| r.rankable_rows).sum()
    }

    pub fn exact_pairwise_matches(&self) -> u64 {
        self.replays.values().map(|r| r.matches).sum()
    }

    pub fn to_json(&self) -> Value {
        let strs =
            |items: &[String]| Value::List(items.iter().map(|s| Value::from(s.as_str())).collect());
        let mut totals = Dict::new();
        totals.insert("histories", self.replays.len().into());
        totals.insert("rankable_rows", self.exact_rankable_rows().into());
        totals.insert(
            "pairwise_matches",
            Value::Int(self.exact_pairwise_matches() as i64),
        );
        let mut dict = Dict::new();
        dict.insert("replayed", strs(&self.replayed));
        dict.insert(
            "not_replayed_here",
            Value::Dict(
                self.not_replayed_here
                    .iter()
                    .map(|(k, v)| (k.clone(), Value::from(v.as_str())))
                    .collect(),
            ),
        );
        dict.insert("scorer_parity_only", strs(&self.scorer_parity_only));
        dict.insert("irrecoverable", strs(&self.irrecoverable));
        dict.insert("exact_totals", Value::Dict(totals));
        dict.insert(
            "excluded_from_exact_totals",
            strs(&self.excluded_from_exact_totals),
        );
        dict.insert(
            "replays",
            Value::Dict(
                self.replays
                    .iter()
                    .map(|(k, v)| (k.clone(), v.to_json()))
                    .collect(),
            ),
        );
        dict.insert(
            "lineages",
            Value::Dict(
                self.lineages
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect(),
            ),
        );
        dict.insert(
            "derived_view_rows",
            Value::Dict(
                self.derived_view_rows
                    .iter()
                    .map(|(k, v)| (k.clone(), Value::from(*v)))
                    .collect(),
            ),
        );
        dict.insert("consumer", CONSUMER.into());
        dict.insert(
            "refused_consumers",
            Value::List(REFUSED_CONSUMERS.iter().map(|c| Value::from(*c)).collect()),
        );
        Value::Dict(dict)
    }

    /// `--require-all`: fail when a reproducible history was not replayed on this host.
    pub fn require_all(&self) -> Result<()> {
        let missing: Vec<&str> = self.not_replayed_here.keys().map(String::as_str).collect();
        require!(
            missing.is_empty(),
            "not replayed here: {}",
            py::str_list_repr(&missing)
        );
        Ok(())
    }
}

/// Verify the record at `path`, resolving its pinned files under `root`.
pub fn verify(path: &Path, root: &Path, zstd: Zstd) -> Result<Summary> {
    let document = py::load_json(path)?;
    require!(
        document.get("schema") == Some(&Value::from(SCHEMA)),
        "histories schema"
    );
    require!(
        truthy_field(&document, "numerical_policy"),
        "numerical policy"
    );
    let mut seen = HashSet::new();
    let mut replays = BTreeMap::new();
    let mut unavailable = BTreeMap::new();
    let listed = match document.get("histories") {
        None => Vec::new(),
        Some(value) => py::iterate(value).ok_or_else(|| shape("histories is not iterable"))?,
    };
    for history in &listed {
        check_history(root, history, &mut seen, zstd)?;
        if history.get("disposition") == Some(&Value::from(REPRODUCIBLE)) {
            let hid = index(history, "id")?.py_str();
            match check_replay(root, history, zstd) {
                Ok(summary) => {
                    replays.insert(hid, summary);
                }
                Err(Error::Unavailable(reason)) => {
                    unavailable.insert(hid, reason);
                }
                Err(error) => return Err(error),
            }
        }
    }
    let all = list_of(index(&document, "histories")?, "histories")?;
    let mut histories: HashMap<Key, &Value> = HashMap::new();
    for history in all {
        histories.insert(key_of(index(history, "id")?)?, history);
    }
    let mut lineages = BTreeMap::new();
    let mut lineage_seen = HashSet::new();
    if let Some(value) = document.get("lineages") {
        for lineage in py::iterate(value).ok_or_else(|| shape("lineages is not iterable"))? {
            let counts = check_lineage(root, &lineage, &mut lineage_seen, zstd)?;
            lineages.insert(index(&lineage, "id")?.py_str(), counts);
        }
    }
    let mut views = BTreeMap::new();
    if let Some(value) = document.get("derived_views") {
        for view in py::iterate(value).ok_or_else(|| shape("derived_views is not iterable"))? {
            let id = index(&view, "id")?.py_str();
            views.insert(id, derived_view(root, &view, &histories, zstd)?);
        }
    }
    let with = |disposition: &str| -> Vec<String> {
        let mut ids: Vec<String> = all
            .iter()
            .filter(|h| h.get("disposition") == Some(&Value::from(disposition)))
            .filter_map(|h| h.get("id").map(Value::py_str))
            .collect();
        ids.sort();
        ids
    };
    let every: BTreeSet<String> = histories
        .values()
        .filter_map(|h| h.get("id").map(Value::py_str))
        .collect();
    Ok(Summary {
        replayed: replays.keys().cloned().collect(),
        not_replayed_here: unavailable,
        scorer_parity_only: with("scorer-parity-only"),
        irrecoverable: with("irrecoverable"),
        excluded_from_exact_totals: every
            .into_iter()
            .filter(|id| !replays.contains_key(id))
            .collect(),
        replays,
        lineages,
        derived_view_rows: views,
    })
}

/// `--sensitivity HISTORY`: the iteration-cap sensitivity of one replayable history.
pub fn sensitivity_of(record: &Path, root: &Path, history_id: &str, zstd: Zstd) -> Result<Value> {
    let document = py::load_json(record)?;
    let chosen = find_by_id(&document, "histories", history_id)?;
    require!(
        chosen.is_some_and(|h| h.get("replay").is_some()),
        "{history_id}: no replayable history"
    );
    sensitivity(root, chosen.expect("checked above"), 2000, zstd)
}
