use super::*;

/// One revoked evidence node, the digest its bytes were pinned at, and the revocation that
/// reaches it: the node's own, or that of its revoked ancestor with the lowest sequence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RevokedEvidence {
    pub node: String,
    pub sha256: String,
    /// The node the revocation names (this node, or an ancestor).
    pub revoked_node: String,
    pub issuer: String,
    pub cause: String,
    pub effective_utc: String,
    pub sequence: i64,
}

/// The revocation closure of one verified claims record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RevokedClosure {
    /// sha256 of the exact record bytes that were verified and read.
    pub claims_sha256: String,
    /// Every revoked node, sorted by node id.
    pub revoked: Vec<RevokedEvidence>,
}

/// The revocation closure of the claims record at `path`, as live consumers need it: every
/// revoked node with the digest its bytes were pinned at, attributed to the revocation that
/// reaches it. The record is verified exactly as [`verify`] verifies it, and the closure is taken
/// from the same bytes (a record that changes while it is verified is refused), so a record
/// `verify` refuses yields no revocations at all, never a partial set. Revoking a node revokes
/// exactly its descendants, as in [`verify`].
pub fn revoked_evidence(
    path: &Path,
    root: Option<&Path>,
    require_custody: bool,
) -> Result<RevokedClosure> {
    let before = py::read_bytes(path)?;
    verify(path, root, require_custody)?;
    let bytes = py::read_bytes(path)?;
    require!(
        bytes == before,
        "{}: the claims record changed while it was verified",
        path.display()
    );
    let document = py::load_json_bytes(path, &bytes)?;
    let empty = Value::Dict(Dict::new());
    let nodes = document
        .get("evidence")
        .unwrap_or(&empty)
        .as_dict()
        .expect("verify checked the evidence map");
    let mut children: HashMap<String, Vec<String>> = HashMap::new();
    for (node_id, node) in nodes.iter() {
        for parent in parents(node) {
            if let Some(parent) = parent.as_str() {
                children
                    .entry(parent.to_owned())
                    .or_default()
                    .push(node_id.to_owned());
            }
        }
    }
    let mut revocations = match document.get("revocations") {
        None => Vec::new(),
        Some(value) => py::iterate(value).expect("verify checked the revocations"),
    };
    revocations.sort_by_key(|r| r.get("sequence").and_then(Value::as_int));
    let text = |v: &Value, field: &str| {
        v.get(field)
            .and_then(Value::as_str)
            .map(str::to_owned)
            .unwrap_or_else(|| v.get(field).map_or_else(String::new, Value::py_str))
    };
    let mut reached: BTreeMap<String, RevokedEvidence> = BTreeMap::new();
    for revocation in &revocations {
        let root_node = text(revocation, "node");
        let mut pending = vec![root_node.clone()];
        while let Some(node_id) = pending.pop() {
            if reached.contains_key(&node_id) {
                continue;
            }
            let node = nodes
                .get(node_id.as_str())
                .expect("verify checked every node");
            reached.insert(
                node_id.clone(),
                RevokedEvidence {
                    node: node_id.clone(),
                    sha256: text(node, "sha256"),
                    revoked_node: root_node.clone(),
                    issuer: text(revocation, "issuer"),
                    cause: text(revocation, "cause"),
                    effective_utc: text(revocation, "effective_utc"),
                    sequence: revocation
                        .get("sequence")
                        .and_then(Value::as_int)
                        .expect("verify checked the sequence"),
                },
            );
            pending.extend(children.get(&node_id).into_iter().flatten().cloned());
        }
    }
    Ok(RevokedClosure {
        claims_sha256: py::sha256_hex(&bytes),
        revoked: reached.into_values().collect(),
    })
}
