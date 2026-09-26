//! `verify-claims`: scoped publication claims against an evidence dependency graph
//! (W12-T01/T05/T07).
//!
//! Evidence nodes bind a subject, scope, source commit, kind and parents. A claim must match the
//! subject, scope and source of every node it cites, and its stage must be backed by a node of
//! an admissible kind; a synthetic fixture never backs live, measurement, promotion or
//! historical claims. Revoking a node revokes exactly its descendants; any claim citing a
//! revoked node must itself be recorded as revoked. History is append-only. Digests
//! authenticate bytes, not the truth of a producer.
//!
//! Synthetic identity (W12-T07) is read from the evidence itself, not from its declared kind:
//! a file under the public conformance tree (by path or by resolved location), or verified JSON
//! bytes carrying a `synthetic_identity` of the conformance schema, can never be declared
//! real-corpus, live, measurement, decision or historical evidence, and a live, measurement,
//! promotion or historical claim can never cite it.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::py::{self, Key, PosixPath};
use crate::pyjson::{self, Dict, Value};
use crate::{require, synthetic, Error, Result};

pub const SCHEMA: &str = "jope.core-ip-claims/v1";
pub const KINDS: [&str; 8] = [
    "source-trace",
    "synthetic-fixture",
    "real-corpus-fixture",
    "hosted-ci",
    "live-host",
    "measurement",
    "decision",
    "historical-record",
];
pub const CLAIM_FIELDS: [&str; 16] = [
    "id",
    "statement",
    "subject",
    "metric",
    "units",
    "denominator",
    "comparison",
    "baseline",
    "scope",
    "source_commit",
    "observed_utc",
    "uncertainty",
    "limitations",
    "stage",
    "evidence",
    "status",
];
pub const STATUSES: [&str; 3] = ["active", "revoked", "superseded"];
/// Claim stages that synthetic evidence can never support.
pub const NOT_SYNTHETIC_STAGES: [&str; 4] = ["live", "measurement", "promotion", "historical"];
/// Kinds that attest real data, live operation, measurement, a decision or retained history,
/// which evidence of synthetic identity can never be.
pub const NOT_SYNTHETIC_KINDS: [&str; 5] = [
    "real-corpus-fixture",
    "live-host",
    "measurement",
    "decision",
    "historical-record",
];
pub const SCOPE: &str =
    "claim-to-evidence binding and revocation closure; evidence truth needs independent review";

/// Evidence kinds each claim stage accepts (`None`: not a stage).
pub fn stage_kinds(stage: &str) -> Option<&'static [&'static str]> {
    Some(match stage {
        "implementation" => &["source-trace"],
        "fixture" => &["synthetic-fixture", "real-corpus-fixture"],
        "hosted-ci" => &["hosted-ci"],
        "live" => &["live-host"],
        "measurement" => &["measurement"],
        "promotion" => &["decision"],
        "historical" => &["historical-record"],
        _ => return None,
    })
}

/// The summary `verify-claims` prints.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Summary {
    pub evidence: usize,
    pub verified_bytes: usize,
    pub custody_unverified: usize,
    pub revoked_evidence: Vec<String>,
    pub claims: BTreeMap<String, usize>,
}

impl Summary {
    pub fn to_json(&self) -> Value {
        let mut dict = Dict::new();
        dict.insert("evidence", self.evidence.into());
        dict.insert("verified_bytes", self.verified_bytes.into());
        dict.insert("custody_unverified", self.custody_unverified.into());
        dict.insert(
            "revoked_evidence",
            Value::List(
                self.revoked_evidence
                    .iter()
                    .map(|n| n.as_str().into())
                    .collect(),
            ),
        );
        dict.insert(
            "claims",
            Value::Dict(
                self.claims
                    .iter()
                    .map(|(k, v)| (k.clone(), Value::from(*v)))
                    .collect(),
            ),
        );
        dict.insert("scope", SCOPE.into());
        Value::Dict(dict)
    }
}

/// Repository evidence is always verified; custody evidence when present or required.
/// Returns the evidence file when its bytes were verified on this host.
fn check_bytes(
    root: &Path,
    node_id: &str,
    node: &Value,
    require_custody: bool,
) -> Result<Option<PathBuf>> {
    let reference = node.get("path").and_then(Value::as_str);
    require!(
        reference.is_some() && py::is_hex(node.get("sha256"), 64),
        "{node_id}: evidence path and digest"
    );
    let reference = reference.expect("checked above");
    let path: PathBuf = if node.get("location") == Some(&Value::from("repository")) {
        let relative = PosixPath::parse(reference);
        require!(
            !relative.is_absolute() && !relative.has_parent_reference(),
            "{node_id}: repository path traversal"
        );
        relative.join_onto(root)
    } else {
        require!(
            node.get("location") == Some(&Value::from("custody"))
                && PosixPath::parse(reference).is_absolute(),
            "{node_id}: custody evidence needs an absolute path"
        );
        let path = PathBuf::from(reference);
        // An inaccessible custody root (another user's home on a CI runner) makes the lookup
        // fail with EACCES rather than report absence: that is unavailable custody, never a
        // verifier crash.
        let present = std::fs::symlink_metadata(&path).is_ok();
        if !present {
            require!(!require_custody, "{node_id}: custody evidence unavailable");
            return Ok(None);
        }
        path
    };
    require!(
        path.is_file() && !py::is_symlink(&path),
        "{node_id}: evidence file missing"
    );
    require!(
        node.get("sha256") == Some(&Value::from(py::sha256_hex(&py::read_bytes(&path)?))),
        "{node_id}: evidence digest mismatch"
    );
    Ok(Some(path))
}

/// Whether verified evidence has synthetic identity, by location or by its own bytes.
///
/// Unavailable custody has an unknown identity on this host, not a synthetic one.
fn is_synthetic(root: &Path, node: &Value, verified: Option<&Path>) -> bool {
    let Some(path) = verified else {
        return false;
    };
    if node.get("location") == Some(&Value::from("repository")) {
        let reference = node.get("path").and_then(Value::as_str).unwrap_or_default();
        if PosixPath::parse(reference).starts_with(&synthetic::CONFORMANCE_TREE) {
            return true;
        }
    }
    // The resolved location too: a directory symlink can lead into the conformance tree.
    let tree = synthetic::CONFORMANCE_TREE
        .iter()
        .fold(root.to_path_buf(), |dir, part| dir.join(part));
    if let (Ok(resolved), Ok(tree)) = (path.canonicalize(), tree.canonicalize()) {
        if resolved.starts_with(tree) {
            return true;
        }
    }
    std::fs::read(path)
        .ok()
        .and_then(|bytes| pyjson::parse_bytes(&bytes).ok())
        .is_some_and(|body| synthetic::marked(&body))
}

fn parents(node: &Value) -> Vec<Value> {
    node.get("depends_on")
        .and_then(Value::as_list)
        .cloned()
        .unwrap_or_default()
}

fn revoked_closure(nodes: &Dict, revocations: &[Value]) -> HashSet<String> {
    let mut revoked: HashSet<String> = revocations
        .iter()
        .filter_map(|r| r.get("node").and_then(Value::as_str).map(str::to_owned))
        .collect();
    let mut changed = true;
    while changed {
        changed = false;
        for (node_id, node) in nodes.iter() {
            if !revoked.contains(node_id)
                && parents(node)
                    .iter()
                    .any(|p| p.as_str().is_some_and(|p| revoked.contains(p)))
            {
                revoked.insert(node_id.to_owned());
                changed = true;
            }
        }
    }
    revoked
}

fn visit(
    nodes: &Dict,
    node_id: &str,
    visiting: &mut HashSet<String>,
    done: &mut HashSet<String>,
) -> Result<()> {
    require!(!visiting.contains(node_id), "{node_id}: dependency cycle");
    if done.contains(node_id) {
        return Ok(());
    }
    visiting.insert(node_id.to_owned());
    let node = nodes.get(node_id).expect("parents were checked to exist");
    for parent in parents(node) {
        visit(
            nodes,
            parent.as_str().expect("parents are node IDs"),
            visiting,
            done,
        )?;
    }
    visiting.remove(node_id);
    done.insert(node_id.to_owned());
    Ok(())
}

/// Verify the claims record at `path`. `root` (the repository evidence paths are relative to)
/// defaults to the fifth ancestor of `path`, i.e. the repository root for the committed
/// `docs/internal/core-ip/claims/claims.json`.
pub fn verify(path: &Path, root: Option<&Path>, require_custody: bool) -> Result<Summary> {
    let root: PathBuf = match root {
        Some(root) => root.to_path_buf(),
        None => path
            .ancestors()
            .nth(5)
            .map(Path::to_path_buf)
            .ok_or_else(|| {
                Error::Shape(format!(
                    "{}: no repository root above the claims file",
                    path.display()
                ))
            })?,
    };
    let document = py::load_json(path)?;
    require!(
        document.get("schema") == Some(&Value::from(SCHEMA)),
        "claims schema"
    );
    let empty = Value::Dict(Dict::new());
    let nodes = document.get("evidence").unwrap_or(&empty);
    require!(nodes.as_dict().is_some(), "evidence map");
    let nodes = nodes.as_dict().expect("checked above");
    let mut verified = 0;
    let mut synthetic_nodes: HashSet<&str> = HashSet::new();
    for (node_id, node) in nodes.iter() {
        require!(
            node.get("kind")
                .and_then(Value::as_str)
                .is_some_and(|k| KINDS.contains(&k)),
            "{node_id}: kind"
        );
        require!(
            py::is_hex(node.get("source_commit"), 40),
            "{node_id}: source commit"
        );
        require!(
            node.get("subject").is_some_and(Value::truthy)
                && node.get("scope").is_some_and(Value::truthy),
            "{node_id}: subject and scope"
        );
        let known = node
            .get("depends_on")
            .and_then(Value::as_list)
            .is_some_and(|list| {
                list.iter()
                    .all(|p| p.as_str().is_some_and(|p| nodes.contains_key(p)))
            });
        require!(known, "{node_id}: unknown parent");
        let present = check_bytes(&root, node_id, node, require_custody)?;
        if present.is_some() {
            verified += 1;
        }
        if is_synthetic(&root, node, present.as_deref()) {
            let kind = node
                .get("kind")
                .and_then(Value::as_str)
                .expect("kind checked above");
            require!(
                !NOT_SYNTHETIC_KINDS.contains(&kind),
                "{node_id}: synthetic identity cannot be {kind} evidence"
            );
            synthetic_nodes.insert(node_id);
        }
    }
    let mut visiting = HashSet::new();
    let mut done = HashSet::new();
    for node_id in nodes.keys() {
        visit(nodes, node_id, &mut visiting, &mut done)?;
    }
    let revocations = match document.get("revocations") {
        None => Vec::new(),
        Some(value) => {
            py::iterate(value).ok_or_else(|| Error::Shape("revocations is not iterable".into()))?
        }
    };
    let mut last = 0;
    for revocation in &revocations {
        require!(
            revocation
                .get("node")
                .and_then(Value::as_str)
                .is_some_and(|n| nodes.contains_key(n)),
            "revocation of unknown evidence"
        );
        require!(
            ["issuer", "cause", "effective_utc"]
                .iter()
                .all(|f| revocation.get(f).is_some_and(Value::truthy)),
            "revocation issuer, cause and time"
        );
        let sequence = revocation.get("sequence").and_then(Value::as_int);
        require!(
            sequence.is_some_and(|s| s > last),
            "revocation sequence must increase (append-only)"
        );
        last = sequence.expect("checked above");
    }
    let revoked = revoked_closure(nodes, &revocations);
    let claims = match document.get("claims") {
        None => Vec::new(),
        Some(value) => {
            py::iterate(value).ok_or_else(|| Error::Shape("claims is not iterable".into()))?
        }
    };
    let mut ids: HashMap<Key, usize> = HashMap::new();
    for claim in &claims {
        let fields_ok = claim.as_dict().is_some_and(|c| {
            c.len() == CLAIM_FIELDS.len() && CLAIM_FIELDS.iter().all(|f| c.contains_key(f))
        });
        require!(
            fields_ok,
            "{}: claim fields",
            claim
                .get("id")
                .map_or_else(|| "None".to_owned(), Value::py_str)
        );
        let field = |name: &str| claim.get(name).expect("claim fields were checked");
        let cid_value = field("id");
        let cid = cid_value.py_str();
        let key = Key::of(cid_value)
            .ok_or_else(|| Error::Shape(format!("unhashable claim id {}", cid_value.repr())))?;
        let count = ids.entry(key).or_default();
        *count += 1;
        require!(*count == 1, "{cid}: duplicate claim");
        let stage = field("stage").as_str();
        let kinds = stage.and_then(stage_kinds);
        require!(kinds.is_some(), "{cid}: stage");
        let (stage, kinds) = (stage.expect("checked above"), kinds.expect("checked above"));
        let status = field("status");
        require!(
            status.as_str().is_some_and(|s| STATUSES.contains(&s)),
            "{cid}: status"
        );
        require!(
            [
                "statement",
                "subject",
                "metric",
                "units",
                "denominator",
                "scope",
                "uncertainty"
            ]
            .iter()
            .all(|f| field(f).truthy()),
            "{cid}: incomplete binding"
        );
        require!(
            field("limitations")
                .as_list()
                .is_some_and(|l| !l.is_empty()),
            "{cid}: limitations"
        );
        require!(
            py::is_hex(Some(field("source_commit")), 40),
            "{cid}: source commit"
        );
        let cited = field("evidence").as_list().filter(|list| {
            !list.is_empty()
                && list
                    .iter()
                    .all(|e| e.as_str().is_some_and(|e| nodes.contains_key(e)))
        });
        require!(cited.is_some(), "{cid}: evidence references");
        let cited: Vec<&str> = cited
            .expect("checked above")
            .iter()
            .map(|e| e.as_str().expect("checked above"))
            .collect();
        for node_id in &cited {
            let node = nodes.get(node_id).expect("checked above");
            require!(
                node.get("subject") == Some(field("subject"))
                    && node.get("scope") == Some(field("scope"))
                    && node.get("source_commit") == Some(field("source_commit")),
                "{cid}: subject, scope or source differs from evidence {node_id}"
            );
        }
        let kind_of = |node_id: &str| {
            nodes
                .get(node_id)
                .and_then(|n| n.get("kind"))
                .and_then(Value::as_str)
                .unwrap_or_default()
        };
        require!(
            cited.iter().any(|e| kinds.contains(&kind_of(e))),
            "{cid}: no {stage} evidence"
        );
        if NOT_SYNTHETIC_STAGES.contains(&stage) {
            require!(
                !cited
                    .iter()
                    .any(|e| kind_of(e) == "synthetic-fixture" || synthetic_nodes.contains(e)),
                "{cid}: synthetic evidence cannot support a {stage} claim"
            );
        }
        if cited.iter().any(|e| revoked.contains(*e)) {
            require!(
                status.as_str() == Some("revoked"),
                "{cid}: cites revoked evidence but is {}",
                status.py_str()
            );
        }
    }
    let mut revoked_evidence: Vec<String> = revoked.into_iter().collect();
    revoked_evidence.sort();
    let mut statuses: BTreeMap<String, usize> = BTreeMap::new();
    for claim in &claims {
        let status = claim
            .get("status")
            .map_or_else(|| "None".to_owned(), Value::py_str);
        *statuses.entry(status).or_default() += 1;
    }
    Ok(Summary {
        evidence: nodes.len(),
        verified_bytes: verified,
        custody_unverified: nodes.len() - verified,
        revoked_evidence,
        claims: statuses,
    })
}
