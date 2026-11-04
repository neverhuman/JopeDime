use super::*;

fn is_one_of(value: Option<&Value>, allowed: &[&str]) -> bool {
    value.is_some_and(|v| allowed.iter().any(|a| v == &Value::from(*a)))
}

/// `mapping.get(name, {})` used as a dict (`.keys()` or `.items()`): absent is empty, anything
/// but an object raises as it did in Python.
fn section<'a>(mapping: &'a Dict, name: &str, method: &str, empty: &'a Dict) -> Result<&'a Dict> {
    match mapping.get(name) {
        None => Ok(empty),
        Some(Value::Dict(dict)) => Ok(dict),
        Some(other) => Err(raised(
            format_args!("coverage map {name}"),
            py::exception::no_attribute(other, method),
        )),
    }
}

/// `for x in value: if x not in known` over an item's lane or disposition assignment, as the
/// Python did it: a list's items, a dict's keys or a string's characters are each looked up, an
/// unknown one is reported as `str(x)`, and a value Python could not iterate or hash raises.
/// Nothing is filtered out.
fn unknown_members(
    assigned: &Value,
    known: &Dict,
    context: &str,
    report: &mut dyn FnMut(String),
) -> Result<()> {
    for member in iterate(assigned, context)? {
        let key = py::Key::of(&member)
            .ok_or_else(|| raised(context, py::exception::unhashable(&member)))?;
        if !key.as_str().is_some_and(|name| known.contains_key(name)) {
            report(member.py_str());
        }
    }
    Ok(())
}

/// Every error of `mapping` against `discovered`, in `coverage.py validate` order.
///
/// Where the Python raised on a malformed map (a section, lane, disposition or item that is not an
/// object, an assignment it could not iterate, an assignment member it could not hash) it stopped
/// with a traceback and printed nothing; this returns that refusal as an error instead of a report.
/// Malformed assignments are never filtered: every member of a list, every key of an object and
/// every character of a string is looked up exactly as the Python looked it up.
pub fn validate(mapping: &Value, discovered: &Dict) -> Result<Vec<String>> {
    let empty = Dict::new();
    let map = mapping
        .as_dict()
        .ok_or_else(|| raised("coverage map", py::exception::no_attribute(mapping, "get")))?;
    let mut errors = vec![];
    if map.get("schema") != Some(&Value::from(SCHEMA)) {
        errors.push("unsupported coverage schema".to_owned());
    }
    let recorded = section(map, "items", "keys", &empty)?;
    let recorded_keys: BTreeSet<String> = recorded.keys().map(str::to_owned).collect();
    let discovered_keys: BTreeSet<String> = discovered.keys().map(str::to_owned).collect();
    for key in discovered_keys.difference(&recorded_keys) {
        errors.push(format!("unassigned addition: {key}"));
    }
    for key in recorded_keys.difference(&discovered_keys) {
        errors.push(format!("stale assignment: {key}"));
    }
    let lanes = section(map, "lanes", "items", &empty)?;
    for (name, lane) in lanes.iter() {
        let Some(lane) = lane.as_dict() else {
            // `field not in lane` works on a string or a list and then `lane.get` fails; on
            // anything else `in` itself fails.
            let exception = match lane {
                Value::Str(_) | Value::List(_) => py::exception::no_attribute(lane, "get"),
                other => py::exception::not_container(other),
            };
            return Err(raised(
                format_args!("coverage map lanes[{}]", pyjson::str_repr(name)),
                exception,
            ));
        };
        for field in LANE_FIELDS {
            if lane.get(field).is_none() {
                errors.push(format!("lane {name} missing {field}"));
            }
        }
        if !lane.get("command").is_some_and(Value::truthy)
            || !lane.get("inputs").is_some_and(Value::truthy)
            || !lane.get("evidence").is_some_and(Value::truthy)
        {
            errors.push(format!("lane {name} has empty command/inputs/evidence"));
        }
        if !matches!(lane.get("requires_nonzero_tests"), Some(Value::Bool(_))) {
            errors.push(format!(
                "lane {name} requires_nonzero_tests must be boolean"
            ));
        }
        if !is_one_of(lane.get("stage"), &["static", "collection", "execution"]) {
            errors.push(format!("lane {name} has invalid stage"));
        }
        if !is_one_of(
            lane.get("feature_selection"),
            &["package-defaults", "explicit", "not-applicable"],
        ) {
            errors.push(format!("lane {name} must declare feature selection"));
        }
        if !is_one_of(
            lane.get("result_backend"),
            &["none", "rust", "nextest", "pytest", "unittest"],
        ) {
            errors.push(format!("lane {name} must declare result backend"));
        }
    }
    let dispositions = section(map, "dispositions", "items", &empty)?;
    for (name, disposition) in dispositions.iter() {
        let Some(disposition) = disposition.as_dict() else {
            return Err(raised(
                format_args!("coverage map dispositions[{}]", pyjson::str_repr(name)),
                py::exception::no_attribute(disposition, "get"),
            ));
        };
        if !is_one_of(disposition.get("status"), &["deferred", "historical"]) {
            errors.push(format!("disposition {name} has invalid status"));
        }
        for field in ["reason", "dependencies", "owner_phase"] {
            if !disposition.get(field).is_some_and(Value::truthy) {
                errors.push(format!("disposition {name} missing {field}"));
            }
        }
    }
    let default_lanes: BTreeSet<&str> = lanes
        .iter()
        .filter(|(_, lane)| lane.get("feature_selection") == Some(&Value::from("package-defaults")))
        .map(|(name, _)| name)
        .collect();
    let no_list = Value::List(vec![]);
    let full_cpu_default = Value::from("full-cpu-default");
    for key in recorded_keys.intersection(&discovered_keys) {
        let at = format!("coverage map items[{}]", pyjson::str_repr(key));
        let item = recorded.get(key).expect("recorded key");
        let item = item
            .as_dict()
            .ok_or_else(|| raised(&at, py::exception::no_attribute(item, "get")))?;
        let found = discovered.get(key).expect("discovered key");
        if item.get("discovery") != Some(found) {
            errors.push(format!("changed discovery: {key}; review its assignment"));
        }
        // `item.get("lanes", [])` and `item.get("dispositions", [])`, kept as the values they are.
        let assigned = item.get("lanes").unwrap_or(&no_list);
        let deferred = item.get("dispositions").unwrap_or(&no_list);
        let lanes_at = format!("{at}.lanes");
        let dispositions_at = format!("{at}.dispositions");
        if let Some(rest) = key.strip_prefix("target:") {
            let package = rest.split('/').next().unwrap_or_default();
            // `set(discovered[key]["required_features"])`
            let required = field(found, "required_features", &format!("discovered {key}"))?;
            let required: BTreeSet<String> =
                strings(required, &format!("discovered {key} required_features"))?
                    .into_iter()
                    .collect();
            let active = default_features(discovered, package)?;
            let missing: Vec<String> = required.difference(&active).cloned().collect();
            if !missing.is_empty() {
                // `default_lanes.intersection(assigned) or "full-cpu-default" in deferred`: the
                // intersection iterates and hashes every member; `in` runs only when it is empty.
                let mut default_execution = false;
                for member in iterate(assigned, &lanes_at)? {
                    let name = py::Key::of(&member)
                        .ok_or_else(|| raised(&lanes_at, py::exception::unhashable(&member)))?;
                    default_execution |= name.as_str().is_some_and(|n| default_lanes.contains(n));
                }
                if !default_execution {
                    default_execution =
                        py::contains(deferred, &full_cpu_default).ok_or_else(|| {
                            raised(&dispositions_at, py::exception::not_container(deferred))
                        })?;
                }
                if default_execution {
                    errors.push(format!(
                        "non-default required features {} assigned to default execution: {key}",
                        py::str_list_repr(&missing)
                    ));
                }
            }
        }
        if !assigned.truthy() && !deferred.truthy() {
            errors.push(format!("unassigned item: {key}"));
        }
        unknown_members(assigned, lanes, &lanes_at, &mut |lane| {
            errors.push(format!("unknown lane {lane} for {key}"));
        })?;
        unknown_members(
            deferred,
            dispositions,
            &dispositions_at,
            &mut |disposition| {
                errors.push(format!("unknown disposition {disposition} for {key}"));
            },
        )?;
        if !item.get("scope").is_some_and(Value::truthy) {
            errors.push(format!("missing execution scope: {key}"));
        }
    }
    Ok(errors)
}

/// The per-kind counts `validate` summarises, sorted by kind (`i["kind"]` of every discovered
/// item, as the Python counted them).
pub fn counts(discovered: &Dict) -> Result<Dict> {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for (key, item) in discovered.iter() {
        let kind = field(item, "kind", &format!("discovered {key}"))?;
        let kind = kind.as_str().ok_or_else(|| {
            Error::Shape(format!(
                "discovered {key} kind: expected a string, found {}",
                kind.repr()
            ))
        })?;
        *counts.entry(kind.to_owned()).or_default() += 1;
    }
    Ok(counts
        .into_iter()
        .map(|(k, n)| (k, Value::from(n)))
        .collect())
}
