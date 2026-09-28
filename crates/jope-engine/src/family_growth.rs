//! Authenticated grown-family (component) assignment and its verifier.
//!
//! Native port of the retired `crates/jope-engine/policy/family_growth_policy.py`
//! (`assign_new_components` / `verify_assignment`, archived at `8a55c4fef`; see
//! `docs/archive/python-archive-manifest.json`). `families::freeze_growth` used to hash-check
//! that helper against the compiled release and execute it with `/usr/bin/python3`; the compiled
//! release now *is* the verifier, so no helper file in the workspace is consulted at all.
//!
//! The rules, their order and every refusal message are the Python ones, so a receipt accepted
//! or refused there is accepted or refused here for the same reason. Python's own semantics are
//! reproduced where the policy relies on them: truthiness, `==` across ints/floats/bools, `in`
//! on lists/strings/dicts, iteration of dicts/strings, set de-duplication and `str.strip()`.
//! Differential proof: `tests/families.rs` replays the outputs captured from the Python module
//! before it was deleted (`tests/fixtures/family-growth/`) on the existing fixtures and on a
//! fixed-seed random scenario set.
//!
//! Deliberate, fail-closed differences (inputs Python accepted that are refused here): JSON
//! Python's `json` module reads but RFC 8259 does not (`NaN`, `Infinity`, UTF-16/32, a BOM);
//! integers of about 310 to 4300 digits (`serde_json` reads none a double cannot hold; beyond 4300
//! digits Python 3.12 refused them too);
//! non-ASCII reviewer names (Python's `str.isalnum` admitted them; the reviewer key is a path
//! component); paths spelled with exactly two leading slashes, which Python's `pathlib`
//! keeps as a distinct root; and a policy or reviewer key whose resolved name is too long to
//! resolve but lies inside the root (Python built the name and read the file; the custody check
//! here cannot establish custody without resolving it, so it refuses as an escape). Custody that
//! cannot be established at all (an `lstat` or resolution error) is never taken as custody kept.
//!
//! Growth publishes membership only: the assignment never applies contexts, never opens a final
//! test and never makes anything execution eligible.

use crate::manifest::{hash, SEED};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Path, PathBuf};

pub const ASSIGNMENT_SCHEMA: &str = "parallel.family_growth_assignment.v3";
pub const POLICY_SCHEMA: &str = "parallel.family_growth_policy.v3";
pub const POLICY_PATH: &str = "manifests/family-growth-policy.json";
pub const REVIEWER_KEYS: &str = "credentials/provenance-review";
/// Domain separator of the reviewer HMAC over a decision receipt's SHA-256.
pub const SIGNATURE_DOMAIN: &str = "parallel.family_growth.v3:";
const GRAPH_SCHEMA: &str = "parallel.component_graph.v1";
const REGISTRY_SCHEMA: &str = "parallel.family_registry.v1";
const DECISION_SCHEMA: &str = "parallel.provenance_decision.v3";

/// Why an assignment was not produced or not verified.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GrowthError {
    /// A policy refusal. Python raised `ValueError` with exactly this message.
    Refused(String),
    /// Malformed input or an I/O failure. Python raised some other exception
    /// (`KeyError`, `TypeError`, `AttributeError`, `OSError`, ...).
    Malformed(String),
}

impl fmt::Display for GrowthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GrowthError::Refused(message) => write!(f, "{message}"),
            GrowthError::Malformed(message) => write!(f, "malformed growth evidence: {message}"),
        }
    }
}

impl std::error::Error for GrowthError {}

type Outcome<T> = std::result::Result<T, GrowthError>;

fn refused(message: impl Into<String>) -> GrowthError {
    GrowthError::Refused(message.into())
}

fn malformed(message: impl Into<String>) -> GrowthError {
    GrowthError::Malformed(message.into())
}

// ---- Python value semantics ------------------------------------------------------------------

/// Python truthiness of a JSON value.
pub fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => match (n.as_i64(), n.as_u64(), n.as_f64()) {
            (Some(i), _, _) => i != 0,
            (_, Some(u), _) => u != 0,
            (_, _, Some(f)) => f != 0.0,
            _ => true,
        },
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// A number as Python compares it: `True == 1 == 1.0`.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Num {
    Int(i128),
    Float(f64),
}

fn number(value: &Value) -> Option<Num> {
    match value {
        Value::Bool(b) => Some(Num::Int(i128::from(*b))),
        Value::Number(n) => Some(if let Some(i) = n.as_i64() {
            Num::Int(i128::from(i))
        } else if let Some(u) = n.as_u64() {
            Num::Int(i128::from(u))
        } else {
            let f = n.as_f64()?;
            if f.is_finite() && f.fract() == 0.0 && f.abs() < 1e38 {
                Num::Int(f as i128)
            } else {
                Num::Float(f)
            }
        }),
        _ => None,
    }
}

/// Python `==` between two JSON values.
pub fn py_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Null, Value::Null) => true,
        (Value::String(x), Value::String(y)) => x == y,
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| py_eq(p, q))
        }
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len() && x.iter().all(|(k, v)| y.get(k).is_some_and(|w| py_eq(v, w)))
        }
        _ => match (number(a), number(b)) {
            (Some(x), Some(y)) => x == y,
            _ => false,
        },
    }
}

/// A hashable JSON scalar as a Python `set`/`dict` key (`1`, `1.0` and `True` are one key).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Key {
    Null,
    Int(i128),
    Float(u64),
    Str(String),
}

fn key(value: &Value) -> Outcome<Key> {
    Ok(match value {
        Value::Null => Key::Null,
        Value::String(s) => Key::Str(s.clone()),
        Value::Array(_) | Value::Object(_) => {
            return Err(malformed("unhashable list or object used as a key"))
        }
        _ => match number(value) {
            Some(Num::Int(i)) => Key::Int(i),
            Some(Num::Float(f)) => Key::Float(f.to_bits()),
            None => return Err(malformed("unhashable value")),
        },
    })
}

/// `iter(value)`: list elements, dict keys, or the characters of a string.
fn iterate(value: &Value) -> Outcome<Vec<Value>> {
    match value {
        Value::Array(items) => Ok(items.clone()),
        Value::Object(map) => Ok(map.keys().cloned().map(Value::String).collect()),
        Value::String(s) => Ok(s.chars().map(|c| Value::String(c.to_string())).collect()),
        _ => Err(malformed("value is not iterable")),
    }
}

/// `len(value)` for the iterable shapes.
fn length(value: &Value) -> Outcome<usize> {
    match value {
        Value::Array(items) => Ok(items.len()),
        Value::Object(map) => Ok(map.len()),
        Value::String(s) => Ok(s.chars().count()),
        _ => Err(malformed("value has no length")),
    }
}

/// `item in container` for the container shapes Python allows.
fn contains(container: &Value, item: &Value) -> Outcome<bool> {
    match container {
        Value::Array(items) => Ok(items.iter().any(|x| py_eq(x, item))),
        Value::Object(map) => {
            key(item)?;
            Ok(item.as_str().is_some_and(|k| map.contains_key(k)))
        }
        Value::String(haystack) => match item {
            Value::String(needle) => Ok(haystack.contains(needle.as_str())),
            _ => Err(malformed("'in <string>' requires a string")),
        },
        _ => Err(malformed("argument is not iterable")),
    }
}

/// `dict.get(name)`; a missing key is `None`.
fn get<'a>(object: &'a Map<String, Value>, name: &str) -> &'a Value {
    object.get(name).unwrap_or(&Value::Null)
}

fn object<'a>(value: &'a Value, what: &str) -> Outcome<&'a Map<String, Value>> {
    value
        .as_object()
        .ok_or_else(|| malformed(format!("{what} is not an object")))
}

fn index<'a>(value: &'a Value, name: &str, what: &str) -> Outcome<&'a Value> {
    object(value, what)?
        .get(name)
        .ok_or_else(|| malformed(format!("{what} has no {name:?}")))
}

/// Python's `str.isspace` set, which adds U+001C..U+001F to Unicode `White_Space`.
fn py_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

fn py_strip(s: &str) -> &str {
    s.trim_matches(py_space)
}

// ---- digests ---------------------------------------------------------------------------------

/// HMAC-SHA256 (RFC 2104). Hand-written over `sha2` so the engine takes no new dependency;
/// checked against the RFC 4231 vectors in the unit tests.
pub fn hmac_sha256_hex(key: &[u8], message: &[u8]) -> String {
    const BLOCK: usize = 64;
    let mut block = [0u8; BLOCK];
    if key.len() > BLOCK {
        block[..32].copy_from_slice(&Sha256::digest(key));
    } else {
        block[..key.len()].copy_from_slice(key);
    }
    let mut inner = Sha256::new();
    inner.update(block.map(|b| b ^ 0x36));
    inner.update(message);
    let mut outer = Sha256::new();
    outer.update(block.map(|b| b ^ 0x5c));
    outer.update(inner.finalize());
    hex::encode(outer.finalize())
}

/// `hmac.compare_digest` on two ASCII strings: no early exit on the first differing byte.
fn compare_digest(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    let mut differ = u8::from(a.len() != b.len());
    for (i, x) in a.iter().enumerate() {
        differ |= x ^ b.get(i).copied().unwrap_or(!x);
    }
    differ == 0
}

/// `stable_assignment_v2`: the 80/20 bucket on the canonical component id; no sealed test.
pub fn stable_assignment_v2(component: &str) -> (String, u64, &'static str) {
    let digest = hash(format!("{SEED}:component:{component}").as_bytes());
    let bucket = u64::from_str_radix(&digest[..16], 16).expect("SHA-256 hex prefix") % 1_000_000;
    let partition = if bucket < 800_000 {
        "meta_train"
    } else {
        "development_validation"
    };
    (digest, bucket, partition)
}

// ---- paths -----------------------------------------------------------------------------------

fn path_text(value: &Value, what: &str) -> Outcome<PathBuf> {
    let text = value
        .as_str()
        .ok_or_else(|| malformed(format!("{what} is not a string path")))?;
    if text.starts_with("//") && !text.starts_with("///") {
        return Err(refused(format!(
            "{what} spelled with a '//' root is not supported"
        )));
    }
    Ok(PathBuf::from(if text.is_empty() { "." } else { text }))
}

fn resolve_strict(path: &Path) -> Outcome<PathBuf> {
    path.canonicalize()
        .map_err(|e| malformed(format!("resolve {}: {e}", path.display())))
}

/// `path.is_symlink() or not path.resolve().is_relative_to(root)`, failing closed: custody that
/// cannot be established is never taken as custody kept.
///
/// - The path itself cannot be inspected (`lstat` fails: missing, a loop, a directory that may not
///   be searched): malformed, as Python raised there or on the read that followed
///   (`PermissionError`, `FileNotFoundError`, pathlib's `RuntimeError` for a loop).
/// - A symlink escapes.
/// - It resolves: it escapes unless the resolved path is inside `root`.
/// - It cannot be resolved because the resolved name is too long (`ENAMETOOLONG`): it escapes.
///   Python's non-strict `resolve()` built that name anyway and found it outside the root (a
///   symlink out of the workspace to a deep directory); inside the root it would have been read,
///   so refusing it is stricter there.
/// - Any other resolution error: malformed.
fn escapes(path: &Path, root: &Path) -> Outcome<bool> {
    const ENAMETOOLONG: i32 = 36; // Linux
    let custody = |e: std::io::Error| malformed(format!("custody of {}: {e}", path.display()));
    if std::fs::symlink_metadata(path)
        .map_err(custody)?
        .file_type()
        .is_symlink()
    {
        return Ok(true);
    }
    match path.canonicalize() {
        Ok(resolved) => Ok(!resolved.starts_with(root)),
        Err(e) if e.raw_os_error() == Some(ENAMETOOLONG) => Ok(true),
        Err(e) => Err(custody(e)),
    }
}

fn read(path: &Path) -> Outcome<Vec<u8>> {
    std::fs::read(path).map_err(|e| malformed(format!("read {}: {e}", path.display())))
}

fn parse(bytes: &[u8], what: &str) -> Outcome<Value> {
    serde_json::from_slice(bytes).map_err(|e| malformed(format!("parse {what}: {e}")))
}

/// Read durable evidence named by `{path, sha256}` inside `root` and return its bytes.
fn artifact(root: &Path, reference: &Value) -> Outcome<Vec<u8>> {
    let Some(fields) = reference
        .as_object()
        .filter(|o| o.len() == 2 && o.contains_key("path") && o.contains_key("sha256"))
    else {
        return Err(refused("durable evidence path and hash required"));
    };
    let mut path = path_text(&fields["path"], "evidence path")?;
    if !path.is_absolute() {
        path = root.join(path);
    }
    let path = resolve_strict(&path)?;
    if !path.starts_with(root) {
        return Err(refused("growth evidence escapes workspace"));
    }
    let data = read(&path)?;
    if !py_eq(&Value::String(hash(&data)), &fields["sha256"]) {
        return Err(refused("growth evidence content changed"));
    }
    Ok(data)
}

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

#[cfg(test)]
mod tests {
    use super::*;

    /// RFC 4231 test cases 1, 2 and 6 (short key, short ASCII key, key longer than a block).
    #[test]
    fn hmac_matches_rfc_4231() {
        assert_eq!(
            hmac_sha256_hex(&[0x0b; 20], b"Hi There"),
            "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"
        );
        assert_eq!(
            hmac_sha256_hex(b"Jefe", b"what do ya want for nothing?"),
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
        assert_eq!(
            hmac_sha256_hex(
                &[0xaa; 131],
                b"Test Using Larger Than Block-Size Key - Hash Key First"
            ),
            "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54"
        );
    }

    #[test]
    fn compare_digest_needs_equal_length_and_bytes() {
        assert!(compare_digest("abc", "abc"));
        assert!(!compare_digest("abc", "abd"));
        assert!(!compare_digest("abc", "ab"));
        assert!(!compare_digest("", "a"));
        assert!(compare_digest("", ""));
    }

    #[test]
    fn python_equality_truthiness_and_membership() {
        assert!(py_eq(&json!(1), &json!(1.0)));
        assert!(py_eq(&json!(true), &json!(1)));
        assert!(!py_eq(&json!("1"), &json!(1)));
        assert!(py_eq(&json!({"a": [1, "x"]}), &json!({"a": [1.0, "x"]})));
        assert!(!py_eq(&json!({"a": 1}), &json!({"a": 1, "b": 2})));
        for falsy in [
            json!(null),
            json!(false),
            json!(0),
            json!(0.0),
            json!(""),
            json!([]),
            json!({}),
        ] {
            assert!(!truthy(&falsy), "{falsy}");
        }
        assert!(contains(&json!(["a", "b"]), &json!("b")).unwrap());
        assert!(contains(&json!("reviewer-a"), &json!("view")).unwrap());
        assert!(contains(&json!({"k": 1}), &json!("k")).unwrap());
        assert!(contains(&json!(3), &json!("k")).is_err());
        assert_eq!(iterate(&json!("ab")).unwrap(), vec![json!("a"), json!("b")]);
        assert_eq!(py_strip("\u{1c} x \u{a0}"), "x");
    }

    /// Python 3.12's `json.loads` refused an integer literal of more than 4300 digits, of either
    /// sign, even in a field nothing reads. `serde_json` reads no integer that a double cannot
    /// hold (about 310 digits and up), so the native reader refuses those too.
    #[test]
    fn integers_python_refused_to_convert_are_refused() {
        for sign in ["", "-"] {
            let receipt = format!(
                r#"{{"schema": "{ASSIGNMENT_SCHEMA}", "unused": {sign}{}}}"#,
                "9".repeat(4301)
            );
            assert!(matches!(
                parse(receipt.as_bytes(), "growth assignment"),
                Err(GrowthError::Malformed(_))
            ));
        }
        let receipt =
            format!(r#"{{"schema": "{ASSIGNMENT_SCHEMA}", "unused": 12345678901234567890123}}"#);
        assert!(parse(receipt.as_bytes(), "growth assignment").is_ok());
    }

    /// Same seed and bucket rule as `families::stable_component_assignment`.
    #[test]
    fn component_bucket_is_the_registry_rule() {
        for id in ["canonical-cc", "alias-string", "development-0"] {
            let (digest, bucket, partition) = stable_assignment_v2(id);
            let (d2, b2, p2) = crate::families::stable_component_assignment(id);
            assert_eq!((digest, bucket, partition.to_owned()), (d2, b2, p2));
        }
    }
}
