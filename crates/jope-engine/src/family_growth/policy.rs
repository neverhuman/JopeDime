use super::*;

// ---- the policy ------------------------------------------------------------------------------

/// Everything `assign_new_components` is called with on the verification path.
pub struct Request<'a> {
    pub families: &'a Value,
    pub live_families: &'a Value,
    pub sealed_test_families: &'a Value,
    pub root: &'a Value,
    pub graph: &'a Value,
    pub parent_registry: &'a Value,
    pub decisions: &'a Value,
}

struct Graph {
    /// alias -> canonical component id
    aliases: BTreeMap<String, String>,
    components: BTreeMap<String, Value>,
}

impl Graph {
    fn canonical(&self, name: &Value) -> Outcome<String> {
        key(name)?;
        match name {
            Value::String(s) => match self.aliases.get(s) {
                Some(cid) => Ok(cid.clone()),
                None => Err(refused(format!("unresolved component alias: {s}"))),
            },
            _ => Err(malformed("component alias is not a string")),
        }
    }
}

fn load_graph(graph: &Value) -> Outcome<Graph> {
    let mut aliases: BTreeMap<String, String> = BTreeMap::new();
    let mut components: BTreeMap<String, Value> = BTreeMap::new();
    for component in iterate(index(graph, "components", "graph")?)? {
        let cid = index(&component, "component_id", "component")?.clone();
        if !truthy(&cid) {
            return Err(refused("duplicate or empty canonical component"));
        }
        let cid_key = key(&cid)?;
        let duplicate = matches!(&cid_key, Key::Str(s) if components.contains_key(s));
        if duplicate || !truthy(get(object(&component, "component")?, "aliases")) {
            return Err(refused("duplicate or empty canonical component"));
        }
        let listed = index(&component, "aliases", "component")?;
        let mut names: BTreeSet<Key> = BTreeSet::new();
        let mut values: BTreeMap<Key, Value> = BTreeMap::new();
        for name in iterate(listed)?.into_iter().chain([cid.clone()]) {
            let k = key(&name)?;
            values.entry(k.clone()).or_insert(name);
            names.insert(k);
        }
        if names.len() != length(listed)? || !contains(listed, &cid)? {
            return Err(refused(
                "component aliases must be unique and include canonical id",
            ));
        }
        let Key::Str(cid) = cid_key else {
            return Err(refused("ambiguous component alias"));
        };
        for name in values.values() {
            match name {
                Value::String(s) if !s.is_empty() && !aliases.contains_key(s) => {
                    aliases.insert(s.clone(), cid.clone());
                }
                _ => return Err(refused("ambiguous component alias")),
            }
        }
        components.insert(cid, component);
    }
    Ok(Graph {
        aliases,
        components,
    })
}

fn reviewer_listed(reviewers: &Value, reviewer: &Value) -> Outcome<bool> {
    match reviewers {
        Value::Array(_) | Value::Object(_) | Value::String(_) => contains(reviewers, reviewer),
        _ => Err(malformed("policy reviewers are not a collection")),
    }
}

/// Authenticate one decision receipt and return its component id. The arguments mirror the
/// retired Python's `authenticate` one for one.
#[allow(clippy::too_many_arguments)]
fn authenticate(
    root: &Path,
    graph: &Graph,
    decided: &BTreeMap<String, Value>,
    policy: &Map<String, Value>,
    policy_sha: &str,
    graph_ref: &Value,
    parent_ref: &Value,
    reference: &Value,
) -> Outcome<String> {
    // `set(ref) != {"path", "sha256", "signature"}`: a dict compares its keys; a list or string
    // compares its elements, and a list that happens to equal the key set fails on `ref[key]`.
    let wanted: BTreeSet<Key> = ["path", "sha256", "signature"]
        .map(|k| Key::Str(k.into()))
        .into();
    let fields = match reference {
        Value::Object(o)
            if o.keys()
                .map(|k| Key::Str(k.clone()))
                .collect::<BTreeSet<_>>()
                == wanted =>
        {
            o
        }
        Value::Array(_) | Value::String(_) => {
            let elements = iterate(reference)?
                .iter()
                .map(key)
                .collect::<Outcome<BTreeSet<_>>>()?;
            return Err(if elements == wanted {
                malformed("decision receipt is not an object")
            } else {
                refused("authenticated decision receipt required")
            });
        }
        Value::Object(_) => return Err(refused("authenticated decision receipt required")),
        _ => return Err(malformed("decision receipt is not iterable")),
    };
    let raw = artifact(
        root,
        &json!({"path": fields["path"], "sha256": fields["sha256"]}),
    )?;
    let decision_value = parse(&raw, "decision receipt")?;
    let decision = object(&decision_value, "decision")?;
    let cid = get(decision, "component_id");
    key(cid)?;
    let cid = match cid {
        Value::String(s) if graph.components.contains_key(s) && !decided.contains_key(s) => {
            s.clone()
        }
        _ => return Err(refused("duplicate or unresolved component decision")),
    };
    let reviewer = get(decision, "reviewer");
    let listed = match policy.get("reviewers") {
        None => false,
        Some(reviewers) => reviewer_listed(reviewers, reviewer)?,
    };
    let reviewer = match reviewer {
        Value::String(name)
            if listed && {
                let stripped: String = name.chars().filter(|c| *c != '-' && *c != '_').collect();
                !stripped.is_empty() && stripped.chars().all(|c| c.is_ascii_alphanumeric())
            } =>
        {
            name.clone()
        }
        _ => return Err(refused("unauthorized provenance reviewer")),
    };
    let key_path = root.join(REVIEWER_KEYS).join(format!("{reviewer}.key"));
    if escapes(&key_path, root)? {
        return Err(refused("reviewer key escaped local custody"));
    }
    let secret = read(&key_path)?;
    let sha = fields["sha256"]
        .as_str()
        .ok_or_else(|| malformed("decision sha256 is not a string"))?;
    let expected = hmac_sha256_hex(&secret, format!("{SIGNATURE_DOMAIN}{sha}").as_bytes());
    if secret.len() < 32 {
        return Err(refused("unauthenticated provenance decision"));
    }
    let signature = fields["signature"]
        .as_str()
        .filter(|s| s.is_ascii())
        .ok_or_else(|| malformed("signature is not an ASCII string"))?;
    if !compare_digest(&expected, signature) {
        return Err(refused("unauthenticated provenance decision"));
    }
    if !py_eq(get(decision, "schema"), &json!(DECISION_SCHEMA))
        || !py_eq(get(decision, "admission"), &json!("approved"))
    {
        return Err(refused("unsupported or unapproved provenance decision"));
    }
    for (field, expected) in [
        ("graph_sha256", index(graph_ref, "sha256", "graph")?.clone()),
        (
            "parent_registry_sha256",
            index(parent_ref, "sha256", "parent registry")?.clone(),
        ),
        ("policy_sha256", json!(policy_sha)),
        ("task", json!("regression")),
    ] {
        if !py_eq(get(decision, field), &expected) {
            return Err(refused(format!(
                "stale or conflicting provenance decision: {field}"
            )));
        }
    }
    if !["target", "units"].iter().all(|field| {
        get(decision, field)
            .as_str()
            .is_some_and(|s| !py_strip(s).is_empty())
    }) {
        return Err(refused("missing target or units contract"));
    }
    let predictors = get(decision, "predictors");
    if !truthy(predictors)
        || !iterate(predictors)?
            .iter()
            .all(|p| p.as_str().is_some_and(|s| !s.is_empty()))
    {
        return Err(refused("missing predictor contract"));
    }
    let contract = match decision.get("entity_time_contract") {
        None => &Map::new(),
        Some(value) => object(value, "entity/time contract")?,
    };
    let strategy = get(contract, "strategy");
    let strategy = ["iid", "grouped", "temporal"]
        .into_iter()
        .find(|s| py_eq(strategy, &json!(s)))
        .ok_or_else(|| refused("missing entity/time contract"))?;
    if strategy == "grouped" && !truthy(get(contract, "entity")) {
        return Err(refused("grouped source lacks entity contract"));
    }
    if strategy == "temporal"
        && !["entity", "order", "horizon", "gap"]
            .iter()
            .all(|k| !get(contract, k).is_null())
    {
        return Err(refused("temporal source lacks order/horizon/gap contract"));
    }
    let sources = get(
        object(&graph.components[&cid], "component")?,
        "source_objects",
    );
    if !truthy(sources) || !py_eq(get(decision, "source_objects"), sources) {
        return Err(refused("provenance source identity mismatch"));
    }
    for source in iterate(sources)? {
        artifact(root, &source)?;
    }
    let prior = match decision.get("prior_use_findings") {
        None => &Map::new(),
        Some(value) => object(value, "prior-use findings")?,
    };
    if !py_eq(get(prior, "status"), &json!("unused"))
        || !py_eq(
            get(prior, "parent_registry_sha256"),
            index(parent_ref, "sha256", "parent registry")?,
        )
        || !truthy(get(prior, "evidence"))
    {
        return Err(refused("original prior-use evidence required"));
    }
    for evidence in iterate(index(&Value::Object(prior.clone()), "evidence", "prior")?)? {
        artifact(root, &evidence)?;
    }
    Ok(cid)
}

/// `assign_new_components` on the verification path: authenticate every decision before
/// assigning genuinely new canonical components. Policy and reviewer keys are read from the
/// workspace custody under `root`; no caller boolean, receipt summary or alias map grants
/// admission. The returned lineage does not apply contexts or activate execution.
pub fn assign_new_components(request: &Request<'_>) -> Outcome<Value> {
    if request.root.is_null()
        || request.graph.is_null()
        || request.parent_registry.is_null()
        || !truthy(request.decisions)
    {
        return Err(refused(
            "durable provenance graph, parent registry and decision receipts required",
        ));
    }
    let families = iterate(request.families)?;
    let root = resolve_strict(&path_text(request.root, "root")?)?;
    let root_text = root
        .to_str()
        .ok_or_else(|| malformed("root is not UTF-8"))?
        .to_owned();
    let policy_path = root.join(POLICY_PATH);
    if escapes(&policy_path, &root)? {
        return Err(refused(
            "pinned provenance policy escaped workspace custody",
        ));
    }
    let policy_data = read(&policy_path)?;
    let policy_sha = hash(&policy_data);
    let policy_value = parse(&policy_data, "policy")?;
    let policy = object(&policy_value, "policy")?;
    if !py_eq(get(policy, "schema"), &json!(POLICY_SCHEMA))
        || !py_eq(get(policy, "seed"), &json!(SEED))
    {
        return Err(refused("unsupported pinned provenance policy"));
    }
    let graph_value = parse(&artifact(&root, request.graph)?, "graph")?;
    let parent_value = parse(
        &artifact(&root, request.parent_registry)?,
        "parent registry",
    )?;
    // `graph.get(...) != ... or parent.get(...) != ...` short-circuits on the graph.
    if !py_eq(
        get(object(&graph_value, "graph")?, "schema"),
        &json!(GRAPH_SCHEMA),
    ) || !py_eq(
        get(object(&parent_value, "parent registry")?, "schema"),
        &json!(REGISTRY_SCHEMA),
    ) {
        return Err(refused("unsupported graph or parent registry"));
    }
    let graph = load_graph(&graph_value)?;
    let mut memberships: BTreeMap<String, BTreeSet<Key>> = BTreeMap::new();
    for (name, assignment) in object(
        index(&parent_value, "families", "parent registry")?,
        "parent families",
    )? {
        // `[name, *assignment["source_keys"]]` is built before any alias is resolved.
        let keys = iterate(index(assignment, "source_keys", "parent family")?)?;
        let mut ids = BTreeSet::new();
        for alias in std::iter::once(json!(name)).chain(keys) {
            ids.insert(graph.canonical(&alias)?);
        }
        if ids.len() != 1 {
            return Err(refused("parent family split across canonical components"));
        }
        let partition = key(index(assignment, "partition", "parent family")?)?;
        memberships
            .entry(ids.pop_first().expect("one id"))
            .or_default()
            .insert(partition);
    }
    let blocked_live = iterate(request.live_families)?
        .iter()
        .map(|name| graph.canonical(name))
        .collect::<Outcome<BTreeSet<_>>>()?;
    let blocked_sealed = iterate(request.sealed_test_families)?
        .iter()
        .map(|name| graph.canonical(name))
        .collect::<Outcome<BTreeSet<_>>>()?;
    let mut decisions: BTreeMap<String, Value> = BTreeMap::new();
    for reference in iterate(request.decisions)? {
        let cid = authenticate(
            &root,
            &graph,
            &decisions,
            policy,
            &policy_sha,
            request.graph,
            request.parent_registry,
            &reference,
        )?;
        decisions.insert(cid, reference);
    }
    let requested = families
        .iter()
        .map(|name| graph.canonical(name))
        .collect::<Outcome<BTreeSet<_>>>()?;
    let (mut rows, mut live, mut sealed, mut quarantined, mut missing) =
        (vec![], vec![], vec![], vec![], vec![]);
    let meta_test = Key::Str("meta_test".into());
    for cid in requested {
        let partitions = memberships.get(&cid).cloned().unwrap_or_default();
        if partitions.len() > 1 {
            quarantined.push(cid);
        } else if partitions.contains(&meta_test) || blocked_sealed.contains(&cid) {
            sealed.push(cid);
        } else if !partitions.is_empty() || blocked_live.contains(&cid) {
            live.push(cid);
        } else if let Some(decision) = decisions.get(&cid) {
            let (digest, bucket, partition) = stable_assignment_v2(&cid);
            let component = object(&graph.components[&cid], "component")?;
            let mut aliases = iterate(&component["aliases"])?
                .into_iter()
                .map(|a| a.as_str().map(str::to_owned))
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| malformed("component alias is not a string"))?;
            aliases.sort();
            rows.push(json!({
                "family": cid, "component_id": cid, "aliases": aliases, "partition": partition,
                "assignment_hash": digest, "bucket": bucket,
                "assignment_method": "stable_sha256_component_v2", "decision": decision,
            }));
        } else {
            missing.push(cid);
        }
    }
    let sorted_names = |value: &Value| -> Outcome<Vec<String>> {
        let set = iterate(value)?
            .into_iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| malformed("family name is not a string"))
            })
            .collect::<Outcome<BTreeSet<_>>>()?;
        Ok(set.into_iter().collect())
    };
    let count = |name: &str| rows.iter().filter(|r| r["partition"] == name).count();
    Ok(json!({
        "schema": ASSIGNMENT_SCHEMA,
        "root": root_text,
        "seed": SEED,
        "requested_families": sorted_names(&Value::Array(families.clone()))?,
        "additional_live_families": sorted_names(request.live_families)?,
        "additional_sealed_families": sorted_names(request.sealed_test_families)?,
        "graph": request.graph,
        "parent_registry": request.parent_registry,
        "policy": {"path": format!("{root_text}/{POLICY_PATH}"), "sha256": policy_sha},
        "decisions": request.decisions,
        "n_train": count("meta_train"),
        "n_development_validation": count("development_validation"),
        "n_test": 0,
        "rows": rows,
        "excluded_live_components": live,
        "sealed_test_components": sealed,
        "quarantined_components": quarantined,
        "missing_decision_components": missing,
        "runtime_closure_qualified": false,
        "execution_eligible": false,
        "apply_contexts_ran": false,
        "final_tests_opened": false,
    }))
}

/// `verify_assignment`: recompute a supplied assignment from its own recorded inputs and
/// refuse it unless the recomputation equals it (Python `==`). Returns the recomputation.
pub fn verify_assignment(supplied: &Value) -> Outcome<Value> {
    let fields = object(supplied, "growth assignment")?;
    if !py_eq(get(fields, "schema"), &json!(ASSIGNMENT_SCHEMA)) {
        return Err(refused("unsupported growth assignment"));
    }
    let field = |name: &str| {
        fields
            .get(name)
            .ok_or_else(|| malformed(format!("growth assignment has no {name:?}")))
    };
    let expected = assign_new_components(&Request {
        families: field("requested_families")?,
        live_families: field("additional_live_families")?,
        sealed_test_families: field("additional_sealed_families")?,
        root: field("root")?,
        graph: field("graph")?,
        parent_registry: field("parent_registry")?,
        decisions: field("decisions")?,
    })?;
    if !py_eq(&expected, supplied) {
        return Err(refused("growth assignment differs from original evidence"));
    }
    Ok(expected)
}

/// `verify_assignment(path)`: read and parse the file, then verify it.
pub fn verify_assignment_file(path: &Path) -> Outcome<Value> {
    verify_assignment(&parse(&read(path)?, "growth assignment")?)
}
