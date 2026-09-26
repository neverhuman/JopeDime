//! `conformance`: the public synthetic conformance pack and its private-canary absence scan
//! (W12-T07).
//!
//! The pack under `docs/internal/core-ip/conformance/` is generated here, deterministically, so
//! its public replay is reproducible: the committed bytes must equal [`pack`]'s. It holds one
//! synthetic fixture per refusal class of the claim-to-evidence boundary, plus an accepted
//! control. Every fixture:
//!
//! * carries the conformance `synthetic_identity` and lives in the conformance tree, so the
//!   receiving validator itself treats it as synthetic;
//! * is exercised through that same validator ([`claims::verify`]), so its acceptance or refusal
//!   is the validator's own.
//!
//! The index binds each fixture to its expected outcome. [`scan`] refuses any file in the tree
//! that does not carry the synthetic identity, or that carries a private canary: a credential
//! shape, a private custody path or host, or an object-store URI.
//!
//! Scope. The pack covers the claims boundary only. The dispositions, fitted-package, proposal
//! ledger and lineage boundaries keep their synthetic cases in their own crates' tests.
//! Synthetic identity bars every fixture from backing a real-corpus, hosted, live, measurement,
//! promotion or historical claim. The pack establishes none of them, and no hardware or
//! operational behaviour. Replaying the private evidence package is W12-T06 and is not part of
//! this pack.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use regex::Regex;

use crate::pyjson::{self, Dict, Value};
use crate::{claims, py, require, synthetic, Error, Result};

/// Schema of the pack index.
pub const PACK_SCHEMA: &str = "jope.core-ip-conformance-pack/v1";
/// The synthetic identity every pack file carries.
pub const IDENTITY: &str = "jope.core-ip-conformance/v1";
/// The pack index, at the root of the conformance tree.
pub const INDEX: &str = "index.json";
/// A synthetic source commit (no repository commit has it).
const COMMIT: &str = "5e1f5e1f5e1f5e1f5e1f5e1f5e1f5e1f5e1f5e1f";
const SUBJECT: &str = "synthetic conformance subject";
const SCOPE: &str = "synthetic conformance fixture; public record-contract replay only";

/// Private canaries: none may appear in any public pack file.
pub const CANARIES: [(&str, &str); 8] = [
    ("aws-access-key", r"AKIA[0-9A-Z]{16}"),
    ("private-key", r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    ("github-token", r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    ("aws-secret-name", r"(?i)aws_secret_access_key"),
    ("private-home-path", r"/home/[A-Za-z0-9_.-]+/"),
    ("private-host", r"(?i)\b(xbabe[0-9]+|atomicsoul)\b"),
    ("object-store-uri", r"\bs3://"),
    ("custody-root", r"jope-(rescue|evidence)"),
];

/// The conformance tree under `repo`.
pub fn tree(repo: &Path) -> PathBuf {
    synthetic::CONFORMANCE_TREE
        .iter()
        .fold(repo.to_path_buf(), |dir, part| dir.join(part))
}

/// A pack path as the repository (and the claims verifier) names it.
fn repository_path(relative: &str) -> String {
    format!("{}/{relative}", synthetic::CONFORMANCE_TREE.join("/"))
}

fn json_bytes(value: &Value) -> Vec<u8> {
    let mut text = pyjson::dumps(value, true, Some(2));
    text.push('\n');
    text.into_bytes()
}

fn marked(mut dict: Dict) -> Value {
    dict.insert("synthetic_identity", IDENTITY.into());
    Value::Dict(dict)
}

fn strings(items: &[&str]) -> Value {
    Value::List(items.iter().map(|s| Value::from(*s)).collect())
}

/// One fixture being built: its evidence files and nodes, its claims and revocations.
struct Fixture {
    directory: String,
    files: BTreeMap<String, Vec<u8>>,
    nodes: Dict,
    claims: Vec<Value>,
    revocations: Vec<Value>,
}

impl Fixture {
    fn new(id: &str) -> Self {
        Fixture {
            directory: format!("claims/{id}"),
            files: BTreeMap::new(),
            nodes: Dict::new(),
            claims: Vec::new(),
            revocations: Vec::new(),
        }
    }

    /// A synthetic evidence file and its node of `kind`, bound to its exact bytes.
    fn node(&mut self, id: &str, kind: &str) -> &mut Dict {
        let relative = format!("{}/evidence/{id}.json", self.directory);
        let mut body = Dict::new();
        body.insert("node", id.into());
        body.insert(
            "description",
            "synthetic conformance evidence; attests nothing real".into(),
        );
        let bytes = json_bytes(&marked(body));
        let mut node = Dict::new();
        node.insert("kind", kind.into());
        node.insert("location", "repository".into());
        node.insert("path", repository_path(&relative).into());
        node.insert("sha256", py::sha256_hex(&bytes).into());
        node.insert("source_commit", COMMIT.into());
        node.insert("subject", SUBJECT.into());
        node.insert("scope", SCOPE.into());
        node.insert("depends_on", Value::List(Vec::new()));
        self.files.insert(relative, bytes);
        self.nodes.insert(id, Value::Dict(node));
        match self.nodes.get_mut(id) {
            Some(Value::Dict(node)) => node,
            _ => unreachable!("inserted above"),
        }
    }

    /// A complete claim of `stage` citing `evidence`.
    fn claim(&mut self, id: &str, stage: &str, evidence: &[&str], status: &str) -> &mut Dict {
        let mut claim = Dict::new();
        claim.insert("id", id.into());
        claim.insert(
            "statement",
            "A synthetic conformance subject satisfies a synthetic contract.".into(),
        );
        claim.insert("subject", SUBJECT.into());
        claim.insert("metric", "synthetic contract outcome".into());
        claim.insert("units", "outcomes".into());
        claim.insert("denominator", "one synthetic fixture".into());
        claim.insert("comparison", Value::Null);
        claim.insert("baseline", Value::Null);
        claim.insert("scope", SCOPE.into());
        claim.insert("source_commit", COMMIT.into());
        claim.insert("observed_utc", "2026-09-26T00:00:00Z".into());
        claim.insert("uncertainty", "exact; synthetic".into());
        claim.insert(
            "limitations",
            strings(&["Synthetic conformance fixture; attests no real subject."]),
        );
        claim.insert("stage", stage.into());
        claim.insert("evidence", strings(evidence));
        claim.insert("status", status.into());
        self.claims.push(Value::Dict(claim));
        match self.claims.last_mut() {
            Some(Value::Dict(claim)) => claim,
            _ => unreachable!("pushed above"),
        }
    }

    /// The fixture's `claims.json` and evidence files.
    fn finish(mut self) -> (String, BTreeMap<String, Vec<u8>>) {
        let mut document = Dict::new();
        document.insert("schema", claims::SCHEMA.into());
        document.insert("evidence", Value::Dict(self.nodes));
        document.insert("revocations", Value::List(self.revocations));
        document.insert("claims", Value::List(self.claims));
        let record = format!("{}/claims.json", self.directory);
        self.files
            .insert(record.clone(), json_bytes(&marked(document)));
        (record, self.files)
    }
}

/// `(id, refusal class, message the validator must refuse with)`; no class: accepted.
const CASES: [(&str, Option<&str>, Option<&str>); 12] = [
    ("accepted-fixture-claim", None, None),
    (
        "synthetic-as-live-host",
        Some("synthetic identity as live evidence"),
        Some("synthetic identity cannot be live-host evidence"),
    ),
    (
        "synthetic-as-real-corpus",
        Some("synthetic identity as real-corpus evidence"),
        Some("synthetic identity cannot be real-corpus-fixture evidence"),
    ),
    (
        "synthetic-as-measurement",
        Some("synthetic identity as measurement evidence"),
        Some("synthetic identity cannot be measurement evidence"),
    ),
    (
        "synthetic-as-decision",
        Some("synthetic identity as a promotion decision"),
        Some("synthetic identity cannot be decision evidence"),
    ),
    (
        "live-claim-on-fixture-evidence",
        Some("stage without evidence of its kind"),
        Some("no live evidence"),
    ),
    (
        "evidence-digest-mismatch",
        Some("evidence bytes changed"),
        Some("evidence digest mismatch"),
    ),
    (
        "evidence-path-traversal",
        Some("evidence outside the repository"),
        Some("repository path traversal"),
    ),
    (
        "claim-scope-drift",
        Some("claim scope differs from its evidence"),
        Some("subject, scope or source differs from evidence"),
    ),
    (
        "revoked-evidence-active-claim",
        Some("revoked evidence still cited"),
        Some("cites revoked evidence but is active"),
    ),
    (
        "incomplete-binding",
        Some("incomplete claim binding"),
        Some("incomplete binding"),
    ),
    (
        "unknown-evidence-kind",
        Some("unknown evidence kind"),
        Some("E-synthetic: kind"),
    ),
];

fn build(id: &str) -> (String, BTreeMap<String, Vec<u8>>) {
    let mut f = Fixture::new(id);
    let claim = "C-synthetic";
    let node = "E-synthetic";
    match id {
        "accepted-fixture-claim" => {
            f.node(node, "synthetic-fixture");
            f.claim(claim, "fixture", &[node], "active");
        }
        "synthetic-as-live-host" => {
            f.node(node, "live-host");
            f.claim(claim, "live", &[node], "active");
        }
        "synthetic-as-real-corpus" => {
            f.node(node, "real-corpus-fixture");
            f.claim(claim, "fixture", &[node], "active");
        }
        "synthetic-as-measurement" => {
            f.node(node, "measurement");
            f.claim(claim, "measurement", &[node], "active");
        }
        "synthetic-as-decision" => {
            f.node(node, "decision");
            f.claim(claim, "promotion", &[node], "active");
        }
        "live-claim-on-fixture-evidence" => {
            f.node(node, "synthetic-fixture");
            f.claim(claim, "live", &[node], "active");
        }
        "evidence-digest-mismatch" => {
            f.node(node, "synthetic-fixture")
                .insert("sha256", py::sha256_hex(b"other bytes").into());
            f.claim(claim, "fixture", &[node], "active");
        }
        "evidence-path-traversal" => {
            f.node(node, "synthetic-fixture").insert(
                "path",
                repository_path("claims/../outside/E-synthetic.json").into(),
            );
            f.claim(claim, "fixture", &[node], "active");
        }
        "claim-scope-drift" => {
            f.node(node, "synthetic-fixture");
            f.claim(claim, "fixture", &[node], "active")
                .insert("scope", "a wider scope the evidence never covered".into());
        }
        "revoked-evidence-active-claim" => {
            f.node(node, "synthetic-fixture");
            f.claim(claim, "fixture", &[node], "active");
            let mut revocation = Dict::new();
            revocation.insert("node", node.into());
            revocation.insert("issuer", "synthetic conformance issuer".into());
            revocation.insert("cause", "synthetic revocation".into());
            revocation.insert("effective_utc", "2026-09-26T00:00:00Z".into());
            revocation.insert("sequence", 1i64.into());
            f.revocations.push(Value::Dict(revocation));
        }
        "incomplete-binding" => {
            f.node(node, "synthetic-fixture");
            f.claim(claim, "fixture", &[node], "active")
                .insert("denominator", "".into());
        }
        "unknown-evidence-kind" => {
            f.node(node, "unknown-kind");
            f.claim(claim, "fixture", &[node], "active");
        }
        other => unreachable!("no conformance case {other}"),
    }
    f.finish()
}

/// Every pack file, by path relative to the conformance tree, index included.
pub fn pack() -> BTreeMap<String, Vec<u8>> {
    let mut files = BTreeMap::new();
    let mut fixtures = Vec::new();
    for (id, class, message) in CASES {
        let (record, built) = build(id);
        files.extend(built);
        let mut expect = Dict::new();
        match message {
            None => expect.insert("outcome", "accepted".into()),
            Some(message) => {
                expect.insert("outcome", "refused".into());
                expect.insert("message_contains", message.into())
            }
        };
        let mut fixture = Dict::new();
        fixture.insert("id", id.into());
        fixture.insert("boundary", "claims".into());
        fixture.insert("validator", "jope_records::claims::verify".into());
        fixture.insert(
            "refusal_class",
            class.map_or(Value::Null, |class| class.into()),
        );
        fixture.insert("record", record.into());
        fixture.insert("expect", Value::Dict(expect));
        fixtures.push(Value::Dict(fixture));
    }
    let mut index = Dict::new();
    index.insert("schema", PACK_SCHEMA.into());
    index.insert(
        "generator",
        "cargo run -q --locked -p jope-records -- conformance --write".into(),
    );
    index.insert(
        "scope",
        "claim-to-evidence boundary: one synthetic fixture per refusal class plus an accepted control"
            .into(),
    );
    index.insert(
        "limitations",
        strings(&[
            "Synthetic fixtures only: they cannot back a real-corpus, hosted, live, measurement, promotion or historical claim.",
            "The dispositions, fitted-package, proposal-ledger and lineage boundaries keep their synthetic cases in their own crates' tests.",
            "Replaying the private evidence package (W12-T06) is not part of this pack.",
        ]),
    );
    index.insert("fixtures", Value::List(fixtures));
    files.insert(INDEX.into(), json_bytes(&marked(index)));
    files
}

/// Every regular file under `dir`, by path relative to `dir` (`/`-separated).
fn committed(dir: &Path, prefix: &str, out: &mut BTreeMap<String, Vec<u8>>) -> Result<()> {
    let entries = std::fs::read_dir(dir).map_err(|e| Error::io(dir, e))?;
    for entry in entries {
        let entry = entry.map_err(|e| Error::io(dir, e))?;
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        let relative = if prefix.is_empty() {
            name
        } else {
            format!("{prefix}/{name}")
        };
        let kind = std::fs::symlink_metadata(&path).map_err(|e| Error::io(&path, e))?;
        require!(
            !kind.file_type().is_symlink(),
            "conformance pack entry is a link: {relative}"
        );
        if kind.is_dir() {
            committed(&path, &relative, out)?;
        } else {
            out.insert(relative, py::read_bytes(&path)?);
        }
    }
    Ok(())
}

/// Check that the committed pack is exactly [`pack`]'s bytes (no drift, no extra or missing
/// file), or rewrite it.
pub fn check_or_write(repo: &Path, write: bool) -> Result<usize> {
    let root = tree(repo);
    let expected = pack();
    if write {
        // Writes the generated files only; a stray file is still refused below, never
        // deleted here.
        for (relative, bytes) in &expected {
            let path = root.join(relative);
            let parent = path.parent().expect("pack files have a parent");
            std::fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
            std::fs::write(&path, bytes).map_err(|e| Error::io(&path, e))?;
        }
    }
    let mut present = BTreeMap::new();
    committed(&root, "", &mut present)?;
    for relative in expected.keys().chain(present.keys()) {
        require!(
            expected.get(relative) == present.get(relative),
            "conformance pack differs from its generator at {relative}; regenerate with --write"
        );
    }
    Ok(present.len())
}

/// Refuse any pack file without the synthetic identity or with a private canary.
pub fn scan(repo: &Path) -> Result<usize> {
    let mut present = BTreeMap::new();
    committed(&tree(repo), "", &mut present)?;
    let canaries: Vec<(&str, Regex)> = CANARIES
        .iter()
        .map(|(name, pattern)| (*name, Regex::new(pattern).expect("canary pattern")))
        .collect();
    for (relative, bytes) in &present {
        let text = String::from_utf8_lossy(bytes);
        for (name, pattern) in &canaries {
            require!(
                !pattern.is_match(&text),
                "private canary {name} in public conformance file {relative}"
            );
        }
        let body = pyjson::parse_bytes(bytes)
            .map_err(|message| Error::parse(&tree(repo).join(relative), message))?;
        require!(
            synthetic::marked(&body),
            "public conformance file {relative} lacks the synthetic identity"
        );
    }
    Ok(present.len())
}

/// Replay every fixture of the committed index through its receiving validator and require
/// the recorded outcome.
pub fn replay(repo: &Path) -> Result<Value> {
    let root = tree(repo);
    let index = py::load_json(&root.join(INDEX))?;
    require!(
        index.get("schema") == Some(&Value::from(PACK_SCHEMA)) && synthetic::marked(&index),
        "conformance pack index schema or identity"
    );
    let fixtures = index
        .get("fixtures")
        .and_then(Value::as_list)
        .ok_or_else(|| Error::Shape("conformance fixtures".into()))?;
    let (mut accepted, mut refused) = (0usize, 0usize);
    for fixture in fixtures {
        let id = fixture
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or_default();
        require!(
            fixture.get("boundary") == Some(&Value::from("claims")),
            "{id}: unknown conformance boundary"
        );
        let record = fixture
            .get("record")
            .and_then(Value::as_str)
            .ok_or_else(|| Error::Shape(format!("{id}: record")))?;
        let outcome = claims::verify(&root.join(record), Some(repo), false);
        let expect = fixture.get("expect");
        match (
            expect
                .and_then(|e| e.get("outcome"))
                .and_then(Value::as_str),
            outcome,
        ) {
            (Some("accepted"), Ok(_)) => accepted += 1,
            (Some("refused"), Err(error)) => {
                let needle = expect
                    .and_then(|e| e.get("message_contains"))
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let message = error.to_string();
                require!(
                    error.is_value_error() && !needle.is_empty() && message.contains(needle),
                    "{id}: refused for another reason: {message}"
                );
                refused += 1;
            }
            (Some("accepted"), Err(error)) => {
                return Err(Error::Invalid(format!(
                    "{id}: expected acceptance, refused: {error}"
                )))
            }
            (Some("refused"), Ok(_)) => {
                return Err(Error::Invalid(format!("{id}: expected refusal, accepted")))
            }
            _ => return Err(Error::Shape(format!("{id}: expected outcome"))),
        }
    }
    let mut summary = Dict::new();
    summary.insert("fixtures", fixtures.len().into());
    summary.insert("accepted", accepted.into());
    summary.insert("refused", refused.into());
    summary.insert(
        "scope",
        "synthetic claim-boundary conformance; qualifies nothing real".into(),
    );
    Ok(Value::Dict(summary))
}
