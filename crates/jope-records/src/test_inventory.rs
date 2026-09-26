//! `verify-test-inventory`: the maintained regression test inventory against the CI coverage
//! map (W0).
//!
//! Every test-bearing coverage item has exactly one role. Regression items name their
//! subsystems, every required regression subsystem has an executing regression item, and every
//! preserved baseline failure has one triage record. This is a static map: it neither collects
//! nor executes tests and does not upgrade any failure into a pass.

use std::collections::{BTreeMap, HashSet};
use std::path::Path;

use crate::py::{self, Key};
use crate::pyjson::{Dict, Value};
use crate::{require, Error, Result};

pub const SCHEMA: &str = "jope.regression-test-inventory/v1";
pub const ROLES: [&str; 6] = [
    "regression",
    "regression-real-corpus",
    "deferred-classification",
    "historical",
    "research",
    "infrastructure",
];
pub const SUBSYSTEM_ROLES: [&str; 2] = ["regression", "regression-real-corpus"];
pub const REQUIRED: [&str; 9] = [
    "preparation",
    "encoders",
    "policy",
    "gp",
    "fitting",
    "teaching",
    "training",
    "storage",
    "export",
];
pub const TRIAGE: [&str; 7] = [
    "fit-authority-fence",
    "product-defect",
    "stale-expectation",
    "missing-real-corpus-input",
    "deferred-classification",
    "lane-timeout",
    "untriaged",
];
pub const STATUS: [&str; 4] = ["open", "fixed-locally", "moved", "ci-verified"];
/// Triage classes that can never be marked fixed.
const NEVER_FIXED: [&str; 4] = [
    "missing-real-corpus-input",
    "deferred-classification",
    "fit-authority-fence",
    "untriaged",
];
pub const SCOPE: &str = "static classification only; no collection, execution or qualification";

/// The summary `verify-test-inventory` prints.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Summary {
    pub items: usize,
    pub roles: BTreeMap<String, usize>,
    pub executing_regression_subsystems: BTreeMap<String, usize>,
    pub baseline_failures: usize,
    pub triage_status: BTreeMap<String, usize>,
}

fn counts(map: &BTreeMap<String, usize>) -> Value {
    Value::Dict(
        map.iter()
            .map(|(k, v)| (k.clone(), Value::from(*v)))
            .collect(),
    )
}

impl Summary {
    pub fn to_json(&self) -> Value {
        let mut dict = Dict::new();
        dict.insert("items", self.items.into());
        dict.insert("roles", counts(&self.roles));
        dict.insert(
            "executing_regression_subsystems",
            counts(&self.executing_regression_subsystems),
        );
        dict.insert("baseline_failures", self.baseline_failures.into());
        dict.insert("triage_status", counts(&self.triage_status));
        dict.insert("scope", SCOPE.into());
        Value::Dict(dict)
    }
}

fn dict_field<'a>(value: &'a Value, name: &str, context: &str) -> Result<&'a Dict> {
    value
        .get(name)
        .and_then(Value::as_dict)
        .ok_or_else(|| Error::Shape(format!("{context}: {name} is not an object")))
}

/// Coverage items that carry collected tests: test-enabled targets and Python families.
fn test_items(coverage: &Value) -> Result<Vec<String>> {
    let mut keys = Vec::new();
    for (key, item) in dict_field(coverage, "items", "coverage")?.iter() {
        let discovery = item
            .get("discovery")
            .ok_or_else(|| Error::Shape(format!("coverage item {key}: no discovery")))?;
        let kind = discovery.get("kind").and_then(Value::as_str);
        if kind == Some("python-family")
            || (kind == Some("target") && discovery.get("test").is_some_and(Value::truthy))
        {
            keys.push(key.to_owned());
        }
    }
    Ok(keys)
}

fn executing(coverage: &Value, key: &str) -> Result<bool> {
    let lanes = dict_field(coverage, "lanes", "coverage")?;
    let item = dict_field(coverage, "items", "coverage")?
        .get(key)
        .ok_or_else(|| Error::Shape(format!("coverage has no item {key}")))?;
    let item_lanes = item
        .get("lanes")
        .and_then(py::iterate)
        .ok_or_else(|| Error::Shape(format!("coverage item {key}: lanes")))?;
    for lane in &item_lanes {
        let name = lane.as_str().unwrap_or_default();
        let stage = lanes
            .get(name)
            .ok_or_else(|| {
                Error::Shape(format!("coverage item {key}: unknown lane {}", lane.repr()))
            })?
            .get("stage");
        if stage == Some(&Value::from("execution")) && name != "rust-check" {
            return Ok(true);
        }
    }
    Ok(false)
}

fn one_of(value: Option<&Value>, set: &[&str]) -> bool {
    value
        .and_then(Value::as_str)
        .is_some_and(|s| set.contains(&s))
}

/// A baseline failure identity `(lane, class, test)`.
type Identity = [Value; 3];

fn identity_key(identity: &Identity) -> Option<[Key; 3]> {
    Some([
        Key::of(&identity[0])?,
        Key::of(&identity[1])?,
        Key::of(&identity[2])?,
    ])
}

/// Verify the inventory at `inventory_path` against `root`'s `deploy/ci/coverage.json` and
/// `docs/internal/core-ip/findings/baseline.json`.
pub fn verify(inventory_path: &Path, root: &Path) -> Result<Summary> {
    let coverage = py::load_json(&root.join("deploy/ci/coverage.json"))?;
    let baseline = py::load_json(&root.join("docs/internal/core-ip/findings/baseline.json"))?;
    let inventory = py::load_json(inventory_path)?;
    require!(
        inventory.get("schema") == Some(&Value::from(SCHEMA)),
        "inventory schema"
    );
    let empty = Dict::new();
    let subsystems = match inventory.get("subsystems") {
        None => &empty,
        Some(value) => value
            .as_dict()
            .ok_or_else(|| Error::Shape("inventory: subsystems is not an object".into()))?,
    };
    require!(
        REQUIRED.iter().all(|name| subsystems.contains_key(name)),
        "required subsystem definitions"
    );
    let packages: HashSet<String> = (0..13).map(|i| format!("W{i}")).collect();
    for (name, value) in subsystems.iter() {
        let work_packages = value.get("work_packages");
        let ok = work_packages.is_some_and(Value::truthy)
            && work_packages.and_then(py::iterate).is_some_and(|wp| {
                wp.iter()
                    .all(|p| p.as_str().is_some_and(|p| packages.contains(p)))
            })
            && value.get("description").is_some_and(Value::truthy);
        require!(ok, "subsystem {name} definition");
    }
    let items = match inventory.get("items") {
        None => &empty,
        Some(value) => value
            .as_dict()
            .ok_or_else(|| Error::Shape("inventory: items is not an object".into()))?,
    };
    let expected = test_items(&coverage)?;
    let expected_set: HashSet<&str> = expected.iter().map(String::as_str).collect();
    let mut unclassified: Vec<&str> = expected_set
        .iter()
        .copied()
        .filter(|key| !items.contains_key(key))
        .collect();
    unclassified.sort_unstable();
    require!(
        unclassified.is_empty(),
        "unclassified: {}",
        py::str_list_repr(&unclassified[..unclassified.len().min(5)])
    );
    let mut stale: Vec<&str> = items
        .keys()
        .filter(|key| !expected_set.contains(key))
        .collect();
    stale.sort_unstable();
    require!(
        stale.is_empty(),
        "stale: {}",
        py::str_list_repr(&stale[..stale.len().min(5)])
    );
    let mut covered: BTreeMap<String, usize> = BTreeMap::new();
    for (key, item) in items.iter() {
        let role = item.get("role");
        require!(one_of(role, &ROLES), "{key}: role");
        let role = role.and_then(Value::as_str).expect("checked above");
        if SUBSYSTEM_ROLES.contains(&role) {
            let names = item.get("subsystems").and_then(Value::as_list);
            let ok = names.is_some_and(|names| {
                let unique: HashSet<Option<Key>> = names.iter().map(Key::of).collect();
                !names.is_empty()
                    && unique.len() == names.len()
                    && !unique.contains(&None)
                    && names
                        .iter()
                        .all(|n| n.as_str().is_some_and(|n| subsystems.contains_key(n)))
            });
            require!(ok, "{key}: subsystems");
            if role == "regression" && executing(&coverage, key)? {
                for name in names.expect("checked above") {
                    let name = name.as_str().expect("checked above");
                    *covered.entry(name.to_owned()).or_default() += 1;
                }
            }
        } else {
            require!(
                !item.get("subsystems").is_some_and(Value::truthy)
                    && item.get("reason").is_some_and(Value::truthy),
                "{key}: reason"
            );
        }
    }
    let mut uncovered: Vec<&str> = REQUIRED
        .iter()
        .copied()
        .filter(|name| !covered.contains_key(*name))
        .collect();
    uncovered.sort_unstable();
    require!(
        uncovered.is_empty(),
        "no executing regression item for {}",
        py::str_list_repr(&uncovered)
    );

    let mut failures: Vec<Identity> = Vec::new();
    let mut failure_keys: HashSet<[Key; 3]> = HashSet::new();
    let families = baseline
        .get("families")
        .and_then(Value::as_list)
        .ok_or_else(|| Error::Shape("baseline: families is not a list".into()))?;
    for family in families {
        let family_failures = family
            .get("failures")
            .and_then(Value::as_list)
            .ok_or_else(|| Error::Shape("baseline: family failures is not a list".into()))?;
        for failure in family_failures {
            let identity: Identity = [
                family.get("lane").cloned().unwrap_or(Value::Null),
                failure.get("class").cloned().unwrap_or(Value::Null),
                failure.get("test").cloned().unwrap_or(Value::Null),
            ];
            let key = identity_key(&identity)
                .ok_or_else(|| Error::Shape("baseline: unhashable failure identity".into()))?;
            if failure_keys.insert(key) {
                failures.push(identity);
            }
        }
    }
    let triage = match inventory.get("baseline_triage") {
        None => Vec::new(),
        Some(value) => py::iterate(value)
            .ok_or_else(|| Error::Shape("inventory: baseline_triage is not iterable".into()))?,
    };
    let mut seen: HashSet<[Key; 3]> = HashSet::new();
    for record in &triage {
        let identity: Identity = [
            record.get("lane").cloned().unwrap_or(Value::Null),
            record.get("class").cloned().unwrap_or(Value::Null),
            record.get("test").cloned().unwrap_or(Value::Null),
        ];
        let shown = py::tuple_repr(&identity);
        let key = identity_key(&identity)
            .ok_or_else(|| Error::Shape(format!("unhashable triage identity {shown}")))?;
        require!(
            failure_keys.contains(&key),
            "triage for unknown baseline failure {shown}"
        );
        require!(!seen.contains(&key), "duplicate triage {shown}");
        seen.insert(key);
        require!(
            record
                .get("item")
                .and_then(Value::as_str)
                .is_some_and(|item| items.contains_key(item)),
            "{shown}: inventory item"
        );
        require!(one_of(record.get("triage"), &TRIAGE), "{shown}: triage");
        require!(one_of(record.get("status"), &STATUS), "{shown}: status");
        require!(
            record.get("note").is_some_and(Value::truthy),
            "{shown}: note"
        );
        let status = record
            .get("status")
            .and_then(Value::as_str)
            .expect("checked above");
        let triage_class = record
            .get("triage")
            .and_then(Value::as_str)
            .expect("checked above");
        if status != "open" {
            require!(
                record.get("evidence").is_some_and(Value::truthy),
                "{shown}: {status} without evidence"
            );
        }
        if status == "moved" {
            require!(
                record.get("moved_to").is_some_and(Value::truthy),
                "{shown}: moved without destination"
            );
        }
        if NEVER_FIXED.contains(&triage_class) {
            require!(
                status == "open" || status == "moved",
                "{shown}: {triage_class} cannot be marked fixed"
            );
        }
    }
    if seen != failure_keys {
        let mut missing: Vec<&Identity> = failures
            .iter()
            .filter(|identity| identity_key(identity).is_some_and(|key| !seen.contains(&key)))
            .collect();
        missing.sort_by(|a, b| {
            let ka = identity_key(a);
            let kb = identity_key(b);
            ka.cmp(&kb)
        });
        let shown: Vec<String> = missing
            .iter()
            .take(3)
            .map(|i| py::tuple_repr(&i[..]))
            .collect();
        return Err(Error::Invalid(format!(
            "untriaged baseline failures: [{}]",
            shown.join(", ")
        )));
    }

    let mut roles: BTreeMap<String, usize> = BTreeMap::new();
    for item in items.values() {
        let role = item
            .get("role")
            .map_or_else(|| "None".to_owned(), Value::py_str);
        *roles.entry(role).or_default() += 1;
    }
    let mut triage_status: BTreeMap<String, usize> = BTreeMap::new();
    for record in &triage {
        let status = record
            .get("status")
            .map_or_else(|| "None".to_owned(), Value::py_str);
        *triage_status.entry(status).or_default() += 1;
    }
    Ok(Summary {
        items: items.len(),
        roles,
        executing_regression_subsystems: covered,
        baseline_failures: failure_keys.len(),
        triage_status,
    })
}
