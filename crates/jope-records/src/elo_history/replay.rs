/// A replayed history: its ladder, the digest of its decoded scores and its corrections.
#[derive(Clone, Debug)]
pub struct Replay {
    pub ladder: Ladder,
    pub scores_sha256: String,
    pub corrections: Value,
}

fn py_len(value: &Value) -> Result<usize> {
    match value {
        Value::List(list) => Ok(list.len()),
        Value::Dict(dict) => Ok(dict.len()),
        Value::Str(s) => Ok(s.chars().count()),
        other => Err(shape(format!(
            "object of type {} has no len()",
            abbreviated(other)
        ))),
    }
}

/// Recompute one ladder from its pinned inputs; the track's own declarations bind the paths.
pub fn replay(root: &Path, spec: &Value, zstd: Zstd) -> Result<Replay> {
    let base = PosixPath::parse(index_str(spec, "root")?);
    let inputs = index(spec, "inputs")?
        .as_dict()
        .ok_or_else(|| shape("replay inputs is not an object"))?;
    let track_id_value = index(spec, "track_id")?;
    let track_id = track_id_value
        .as_str()
        .ok_or_else(|| shape("track_id is not a string"))?;
    let input = |role: &str| -> Result<&Value> {
        inputs
            .get(role)
            .ok_or_else(|| shape(format!("KeyError: {}", pyjson::str_repr(role))))
    };
    let track_path = base
        .join("tracks")
        .join(track_id)
        .join("track.json")
        .to_string();
    require!(
        index(input("track")?, "path")? == &Value::from(track_path),
        "replay track path"
    );
    let track = json_bytes(&load(root, input("track")?, zstd)?, "track.json")?;
    require!(
        track.get("schema") == Some(&Value::from("sotarena.track.v1"))
            && track.get("id") == Some(track_id_value),
        "track identity"
    );
    let direction = track.get("metric_direction");
    require!(
        matches!(
            direction.and_then(Value::as_str),
            Some("minimize" | "maximize")
        ),
        "metric direction"
    );
    let default_methods = Value::from("registry/methods.json");
    let mut declared: Vec<(&str, &Value)> = vec![
        ("datasets", index(&track, "dataset_registry")?),
        (
            "methods",
            track.get("method_registry").unwrap_or(&default_methods),
        ),
        ("scores", index(&track, "scores")?),
    ];
    if let Some(ledger) = track.get("resolution_ledger").filter(|l| l.truthy()) {
        declared.push(("ledger", ledger));
    }
    let given: BTreeSet<&str> = inputs.keys().filter(|k| *k != "track").collect();
    let wanted: BTreeSet<&str> = declared.iter().map(|(role, _)| *role).collect();
    require!(given == wanted, "replay inputs differ from the track's");
    for (role, relative) in &declared {
        let path = index(input(role)?, "path")?;
        let relative = relative
            .as_str()
            .ok_or_else(|| shape(format!("the track's {role} is not a path")))?;
        let joined = base.join(relative).to_string();
        let allowed = [
            joined.clone(),
            format!("{joined}.zst"),
            format!("{joined}.gz"),
        ];
        require!(
            allowed.iter().any(|a| path == &Value::from(a.as_str())),
            "replay {role} path is not the track's declared input"
        );
    }
    let datasets = json_bytes(&load(root, input("datasets")?, zstd)?, "datasets registry")?;
    let total_datasets = py_len(index(&datasets, "datasets")?)?;
    let methods = json_bytes(&load(root, input("methods")?, zstd)?, "methods registry")?;
    let mut names: HashMap<Key, Value> = HashMap::new();
    for method in py::iterate(index(&methods, "methods")?).ok_or_else(|| shape("methods"))? {
        let id = index(&method, "id")?;
        let name = method.get("name").unwrap_or(id).clone();
        names.insert(key_of(id)?, name);
    }
    let scores_bytes = load(root, input("scores")?, zstd)?;
    let scores_sha256 = py::sha256_hex(&scores_bytes);
    let mut rows = parse_scores(&decode_utf8(scores_bytes, "scores.csv")?, track_id)?;
    let mut pending: HashMap<Key, i64> = HashMap::new();
    if inputs.contains_key("ledger") {
        let text = decode_utf8(load(root, input("ledger")?, zstd)?, "resolution ledger")?;
        for line in py::splitlines(&text) {
            if py::strip(line).is_empty() {
                continue;
            }
            let cell = pyjson::parse_str(line)
                .map_err(|message| Error::parse(Path::new("resolution ledger"), message))?;
            require!(
                matches!(cell.get("resolved"), Some(Value::Bool(_))),
                "ledger cell without `resolved`"
            );
            if cell.get("resolved") == Some(&Value::Bool(false)) {
                *pending
                    .entry(key_of(index(&cell, "method_id")?)?)
                    .or_default() += 1;
            }
        }
    }
    let mut corrections = Value::Null;
    if let Some(reference) = spec.get("corrections").filter(|c| c.truthy()) {
        let raw = load(root, reference, zstd)?;
        let document = json_bytes(&raw, "corrections")?;
        let (applied, corrected) =
            apply_corrections(&mut rows, &document, track_id, &scores_sha256)?;
        let mut summary = Dict::new();
        summary.insert(
            "file",
            PosixPath::parse(index_str(reference, "path")?)
                .name()
                .into(),
        );
        summary.insert("sha256", py::sha256_hex(&raw).into());
        summary.insert("applied", Value::List(applied));
        summary.insert("rows_corrected", corrected.into());
        corrections = Value::Dict(summary);
    }
    let ladder_kind = index(spec, "ladder")?
        .as_str()
        .filter(|kind| scope_of(kind).is_some());
    require!(ladder_kind.is_some(), "ladder kind");
    let ladder = ladder(
        &rows,
        &LadderOptions {
            ladder_kind: ladder_kind.expect("checked above"),
            minimize: direction.and_then(Value::as_str) == Some("minimize"),
            names: &names,
            pending: &pending,
            total_datasets,
        },
    )?;
    Ok(Replay {
        ladder,
        scores_sha256,
        corrections,
    })
}

/// Every cell must agree, except that methods of one exchangeable group may trade places.
///
/// Returns the groups whose committed order differs from the replay's; such an order was set by
/// the original implementation's floating-point accumulation, not by data.
pub fn compare(
    label: &str,
    committed: &[Value],
    replayed: &[Value],
    groups: &[Vec<String>],
) -> Result<Vec<Vec<String>>> {
    require!(
        committed.len() == replayed.len(),
        "{label}: ladder length differs from replay"
    );
    let mut group_of: HashMap<&str, usize> = HashMap::new();
    for (i, group) in groups.iter().enumerate() {
        for method in group {
            group_of.insert(method, i);
        }
    }
    let mut traded: BTreeSet<usize> = BTreeSet::new();
    for (offset, (theirs, ours)) in committed.iter().zip(replayed).enumerate() {
        let position = offset + 1;
        let mut same = true;
        for field in POSITION {
            if index(theirs, field)? != index(ours, field)? {
                same = false;
                break;
            }
        }
        require!(same, "{label}: rank {position} position cells");
        let (their_id, our_id) = (index(theirs, "method_id")?, index(ours, "method_id")?);
        if their_id != our_id {
            let group = our_id.as_str().and_then(|m| group_of.get(m));
            key_of(their_id)?;
            let their_group = their_id.as_str().and_then(|m| group_of.get(m));
            require!(
                group.is_some() && group == their_group,
                "{label}: rank {position} is {} in the committed ladder but {} in the independent \
                 replay",
                their_id.py_str(),
                our_id.py_str()
            );
            traded.insert(*group.expect("checked above"));
        }
    }
    let strip = |rows: &[Value]| -> Result<HashMap<Key, Value>> {
        let mut out = HashMap::new();
        for row in rows {
            let dict = row
                .as_dict()
                .ok_or_else(|| shape(format!("{label}: a ladder row is not an object")))?;
            let kept: Dict = dict
                .iter()
                .filter(|(k, _)| !POSITION.contains(k))
                .map(|(k, v)| (k, v.clone()))
                .collect();
            out.insert(key_of(index(row, "method_id")?)?, Value::Dict(kept));
        }
        Ok(out)
    };
    let (theirs, ours) = (strip(committed)?, strip(replayed)?);
    require!(
        theirs.len() == ours.len() && theirs.keys().all(|k| ours.contains_key(k)),
        "{label}: rated methods differ from replay"
    );
    let mut methods: Vec<&Key> = ours.keys().collect();
    methods.sort();
    for method in methods {
        let shown = method
            .as_str()
            .map_or_else(|| format!("{method:?}"), str::to_owned);
        require!(
            theirs[method] == ours[method],
            "{label}: {shown} differs from replay"
        );
    }
    Ok(traded.into_iter().map(|i| groups[i].clone()).collect())
}

/// What a replayed history reports.
#[derive(Clone, Debug, PartialEq)]
pub struct ReplaySummary {
    pub byte_identical: bool,
    pub exchangeable_groups: Vec<Vec<String>>,
    pub order_differs_from_replay: Vec<Vec<String>>,
    pub identical_score_groups: Vec<Vec<String>>,
    pub methods: usize,
    pub rankable_rows: usize,
    pub matches: u64,
    pub iterations: usize,
    pub converged: bool,
    pub components: Vec<usize>,
}

fn groups_value(groups: &[Vec<String>]) -> Value {
    Value::List(
        groups
            .iter()
            .map(|g| Value::List(g.iter().map(|m| Value::from(m.as_str())).collect()))
            .collect(),
    )
}

impl ReplaySummary {
    pub fn cross_component_order(&self) -> &'static str {
        if self.components.len() > 1 {
            "imposed-by-regularization"
        } else {
            "data-connected"
        }
    }

    pub fn to_json(&self) -> Value {
        let mut dict = Dict::new();
        dict.insert("byte_identical", self.byte_identical.into());
        dict.insert(
            "exchangeable_groups",
            groups_value(&self.exchangeable_groups),
        );
        dict.insert(
            "order_differs_from_replay",
            groups_value(&self.order_differs_from_replay),
        );
        dict.insert(
            "identical_score_groups",
            groups_value(&self.identical_score_groups),
        );
        dict.insert("methods", self.methods.into());
        dict.insert("rankable_rows", self.rankable_rows.into());
        dict.insert("matches", Value::Int(self.matches as i64));
        dict.insert("iterations", self.iterations.into());
        dict.insert("converged", self.converged.into());
        dict.insert(
            "components",
            Value::List(self.components.iter().map(|c| Value::from(*c)).collect()),
        );
        dict.insert("cross_component_order", self.cross_component_order().into());
        dict.insert("consumer", CONSUMER.into());
        Value::Dict(dict)
    }
}

/// Replay a reproducible history and compare it with its committed leaderboard and snapshot.
pub fn check_replay(root: &Path, history: &Value, zstd: Zstd) -> Result<ReplaySummary> {
    let spec = index(history, "replay")?;
    let hid = index(history, "id")?.py_str();
    let result = replay(root, spec, zstd)?;
    let outputs = index(spec, "outputs")?;
    let committed = load(root, index(outputs, "leaderboard")?, zstd)?;
    let committed_text = decode_utf8(committed.clone(), "leaderboard.csv")?;
    let committed_rows: Vec<Value> = pycsv::dict_reader(&committed_text, Newlines::Universal)?
        .rows
        .into_iter()
        .map(Value::Dict)
        .collect();
    let as_text: Vec<Value> = result
        .ladder
        .rows
        .iter()
        .map(|row| {
            let dict = row.as_dict().expect("ladder rows are objects");
            Value::Dict(
                dict.iter()
                    .map(|(k, v)| (k, Value::Str(v.py_str())))
                    .collect(),
            )
        })
        .collect();
    let groups = &result.ladder.exchangeable;
    let traded = compare(
        &format!("{hid} leaderboard"),
        &committed_rows,
        &as_text,
        groups,
    )?;
    let snapshot = json_bytes(&load(root, index(outputs, "snapshot")?, zstd)?, "snapshot")?;
    let expected: [(&str, Value); 10] = [
        ("schema", SNAPSHOT_SCHEMA.into()),
        ("track_id", index(spec, "track_id")?.clone()),
        ("ladder", index(spec, "ladder")?.clone()),
        ("scores_sha256", result.scores_sha256.as_str().into()),
        ("corrections", result.corrections.clone()),
        ("excluded_rows", result.ladder.excluded_rows.into()),
        ("leaderboard_sha256", py::sha256_hex(&committed).into()),
        ("rankable_rows", result.ladder.rankable_rows.into()),
        ("datasets_scored", result.ladder.datasets_scored.into()),
        ("methods", result.ladder.rows.len().into()),
    ];
    for (field, value) in &expected {
        require!(
            snapshot.get(field) == Some(value),
            "{hid}: snapshot {field} differs from replay"
        );
    }
    let empty = Value::List(Vec::new());
    let snapshot_rows = list_of(snapshot.get("rows").unwrap_or(&empty), "snapshot rows")?;
    let snapshot_traded = compare(
        &format!("{hid} snapshot"),
        snapshot_rows,
        &result.ladder.rows,
        groups,
    )?;
    require!(
        snapshot_traded == traded,
        "{hid}: snapshot and leaderboard orders differ"
    );
    Ok(ReplaySummary {
        byte_identical: result.ladder.csv.as_bytes() == committed.as_slice(),
        exchangeable_groups: result.ladder.exchangeable.clone(),
        order_differs_from_replay: traded,
        identical_score_groups: result.ladder.identical_scores.clone(),
        methods: result.ladder.rows.len(),
        rankable_rows: result.ladder.rankable_rows,
        matches: result.ladder.matches,
        iterations: result.ladder.iterations,
        converged: result.ladder.converged,
        components: result.ladder.components.clone(),
    })
}
