//! Product vocabulary restricted to the operators jain-math can execute.
//!
//! The jain product adapter admits a search program only when every stage op
//! is executable and not quarantined in jain-math's feat-math registry. The
//! frozen `featmath.json` grammar also offers heads that jain-math does not
//! implement, so a search driven by it wastes evaluations on programs the
//! product must discard. `featmath-jain-v1.json` is `featmath.json` with those
//! heads/transforms removed from the legal productions (`ops`,
//! `engine_heads`, `engine_transforms`, `head_tasks`). The token table is
//! untouched, so token ids, checkpoints and corpora stay compatible.
//!
//! The executable op list is a checked-in snapshot of the jain-math registry
//! (with its revision and source location as provenance); this crate does
//! not depend on jain-math.
use anyhow::{bail, ensure, Context, Result};
use serde::Deserialize;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

pub const BASE_VOCABULARY: &str = "assets/grammar/featmath.json";
pub const EXECUTABLE_OPS: &str = "assets/grammar/jain-math-executable-9fa594e2.json";
pub const RESTRICTED_VOCABULARY: &str = "assets/grammar/featmath-jain-v1.json";

const VOCAB_SCHEMA: &str = "parallel.action_vocab.v1";
const EXECUTABLE_SCHEMA: &str = "parallel.jain_math_executable_ops.v1";
const ENGINE: &str = "featmath";

#[derive(Debug, Deserialize)]
pub struct ExecutableOps {
    pub schema: String,
    pub engine: String,
    pub provenance: Provenance,
    pub executable: Vec<RegistryOp>,
    pub not_executable: Vec<RegistryOp>,
}

#[derive(Debug, Deserialize)]
pub struct Provenance {
    pub repository: String,
    pub rev: String,
    pub registry_path: String,
    pub registry_hash: String,
}

#[derive(Debug, Deserialize)]
pub struct RegistryOp {
    pub name: String,
    pub role: String,
    pub lifecycle: String,
}

impl ExecutableOps {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let list: Self = serde_json::from_slice(bytes).context("parse executable op list")?;
        ensure!(
            list.schema == EXECUTABLE_SCHEMA,
            "executable op list schema"
        );
        ensure!(list.engine == ENGINE, "executable op list engine");
        ensure!(
            list.provenance.rev.len() == 40,
            "executable op list must pin a full jain-math revision"
        );
        ensure!(!list.executable.is_empty(), "executable op list is empty");
        // Executable means executable and not quarantined in the registry:
        // only the `generatable` and `compatibility` lifecycles qualify.
        for op in &list.executable {
            ensure!(
                matches!(op.lifecycle.as_str(), "generatable" | "compatibility"),
                "op {} is listed executable with lifecycle {}",
                op.name,
                op.lifecycle
            );
        }
        for op in &list.not_executable {
            ensure!(
                op.lifecycle == "quarantined",
                "op {} is listed not executable with lifecycle {}",
                op.name,
                op.lifecycle
            );
        }
        let mut seen = BTreeSet::new();
        for op in list.executable.iter().chain(&list.not_executable) {
            ensure!(
                seen.insert(op.name.as_str()),
                "op {} is listed more than once",
                op.name
            );
        }
        Ok(list)
    }

    pub fn names(&self) -> BTreeSet<&str> {
        self.executable.iter().map(|op| op.name.as_str()).collect()
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

/// Rebuild every object with sorted keys so the output bytes do not depend
/// on serde_json's map implementation.
fn canonical(value: Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut entries: Vec<_> = map.into_iter().collect();
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            Value::Object(
                entries
                    .into_iter()
                    .map(|(k, v)| (k, canonical(v)))
                    .collect::<Map<_, _>>(),
            )
        }
        Value::Array(items) => Value::Array(items.into_iter().map(canonical).collect()),
        other => other,
    }
}

fn retain_ids(value: &mut Value, keep: &BTreeSet<u64>, what: &str) -> Result<()> {
    let ids = value
        .get_mut(ENGINE)
        .and_then(Value::as_array_mut)
        .with_context(|| format!("{what} lacks {ENGINE}"))?;
    ids.retain(|id| id.as_u64().is_some_and(|id| keep.contains(&id)));
    Ok(())
}

/// Derive the restricted product vocabulary from the frozen base vocabulary
/// bytes and the executable op list bytes. Deterministic: the output is
/// pretty JSON with sorted keys and a trailing newline.
pub fn derive_restricted_vocabulary(base: &[u8], executable: &[u8]) -> Result<Vec<u8>> {
    let list = ExecutableOps::parse(executable)?;
    let allowed = list.names();
    let mut vocab: Value = serde_json::from_slice(base).context("parse base vocabulary")?;
    ensure!(vocab["schema"] == VOCAB_SCHEMA, "base vocabulary schema");
    let base_version = vocab["version"]
        .as_str()
        .context("base vocabulary version")?
        .to_string();

    let ops = vocab["ops"]
        .as_object_mut()
        .context("base vocabulary ops")?;
    let mut removed = Vec::new();
    let mut kept = BTreeSet::new();
    for (key, op) in ops.iter() {
        let id: u64 = key.parse().context("op key is not a token id")?;
        let name = op["name"].as_str().context("op name")?;
        let role = op["role"].as_str().context("op role")?;
        ensure!(op["engine"] == ENGINE, "op {name} outside {ENGINE}");
        if !matches!(role, "head" | "transform") {
            bail!("op {name} has unsupported role {role}");
        }
        match list.executable.iter().find(|entry| entry.name == name) {
            Some(entry) => {
                ensure!(
                    entry.role == role,
                    "op {name} is a {role} in the vocabulary but a {} in jain-math",
                    entry.role
                );
                kept.insert(id);
            }
            None => removed.push((id, name.to_string())),
        }
    }
    ops.retain(|key, _| key.parse().is_ok_and(|id: u64| kept.contains(&id)));
    ensure!(
        ops.values().all(|op| op["name"]
            .as_str()
            .is_some_and(|name| allowed.contains(name))),
        "non-executable op survived restriction"
    );
    ensure!(
        ops.values()
            .any(|op| op["role"] == "head" && op["name"] == "ridge"),
        "restricted vocabulary must keep the ridge seed head"
    );

    retain_ids(&mut vocab["engine_heads"], &kept, "engine_heads")?;
    retain_ids(&mut vocab["engine_transforms"], &kept, "engine_transforms")?;
    vocab["head_tasks"]
        .as_object_mut()
        .context("base vocabulary head_tasks")?
        .retain(|key, _| key.parse().is_ok_and(|id: u64| kept.contains(&id)));

    let rev = &list.provenance.rev;
    vocab["version"] = Value::String(format!("{base_version}+jain-math-{}", &rev[..8]));
    let sources = vocab["source_sha256"]
        .as_object_mut()
        .context("base vocabulary source_sha256")?;
    sources.insert(
        format!("parallel/{BASE_VOCABULARY}"),
        Value::String(sha256_hex(base)),
    );
    sources.insert(
        format!("parallel/{EXECUTABLE_OPS}"),
        Value::String(sha256_hex(executable)),
    );
    vocab["restriction"] = serde_json::json!({
        "schema": "parallel.vocab_restriction.v1",
        "rule": "keep a head/transform op only if jain-math's feat-math registry marks it executable and not quarantined",
        "base_version": base_version,
        "base_vocabulary_sha256": sha256_hex(base),
        "executable_ops_sha256": sha256_hex(executable),
        "jain_math_repository": list.provenance.repository,
        "jain_math_rev": rev,
        "jain_math_registry_path": list.provenance.registry_path,
        "jain_math_registry_hash": list.provenance.registry_hash,
        "removed_ops": removed
            .iter()
            .map(|(id, name)| serde_json::json!({"token_id": id, "name": name}))
            .collect::<Vec<_>>(),
        "token_table": "unchanged",
    });

    let mut bytes = serde_json::to_vec_pretty(&canonical(vocab))?;
    bytes.push(b'\n');
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::Vocab;
    use crate::manifest::read_json;
    use std::path::{Path, PathBuf};

    fn asset(relative: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join(relative)
    }

    fn read(relative: &str) -> Vec<u8> {
        std::fs::read(asset(relative)).unwrap()
    }

    fn restricted() -> Vocab {
        read_json(&asset(RESTRICTED_VOCABULARY)).unwrap()
    }

    #[test]
    fn committed_restricted_vocabulary_is_the_derivation() {
        let derived =
            derive_restricted_vocabulary(&read(BASE_VOCABULARY), &read(EXECUTABLE_OPS)).unwrap();
        assert!(
            derived == read(RESTRICTED_VOCABULARY),
            "{RESTRICTED_VOCABULARY} is stale; regenerate it with derive_restricted_vocabulary"
        );
    }

    #[test]
    fn every_restricted_head_and_transform_is_executable() {
        let list = ExecutableOps::parse(&read(EXECUTABLE_OPS)).unwrap();
        let allowed = list.names();
        let vocab = restricted();
        let ids = vocab.engine_heads[ENGINE]
            .iter()
            .chain(&vocab.engine_transforms[ENGINE]);
        for id in ids {
            let op = &vocab.ops[id];
            assert!(allowed.contains(op.name.as_str()), "{} kept", op.name);
        }
        for op in vocab.ops.values() {
            assert!(allowed.contains(op.name.as_str()), "{} kept", op.name);
            assert!(
                vocab.engine_heads[ENGINE].contains(&op.token_id)
                    || vocab.engine_transforms[ENGINE].contains(&op.token_id)
            );
        }
        assert!(vocab.head_tasks.keys().all(|id| vocab.ops.contains_key(id)));
    }

    #[test]
    fn restriction_removes_exactly_the_non_executable_ops() {
        let list = ExecutableOps::parse(&read(EXECUTABLE_OPS)).unwrap();
        let allowed = list.names();
        let base: Vocab = read_json(&asset(BASE_VOCABULARY)).unwrap();
        let vocab = restricted();
        for (id, op) in &base.ops {
            assert_eq!(
                vocab.ops.contains_key(id),
                allowed.contains(op.name.as_str()),
                "{}",
                op.name
            );
        }
        let removed: BTreeSet<&str> = base
            .ops
            .values()
            .filter(|op| !vocab.ops.contains_key(&op.token_id))
            .map(|op| op.name.as_str())
            .collect();
        let expected: BTreeSet<&str> = [
            "extra_trees",
            "gam_boost",
            "logit_ridge",
            "multiclass_diagonal_lda",
            "multiclass_nearest_centroid",
            "multiclass_ovr_logistic",
            "multiclass_ridge",
            "portfolio",
            "ridge_gcv",
            "tab_boost",
        ]
        .into_iter()
        .collect();
        assert_eq!(removed, expected);
    }

    #[test]
    fn token_table_and_automaton_constants_are_unchanged() {
        let base: Vocab = read_json(&asset(BASE_VOCABULARY)).unwrap();
        let vocab = restricted();
        assert_eq!(vocab.tokens.len(), 401);
        assert_eq!(vocab.tokens, base.tokens);
        assert_eq!(vocab.engine_ids, base.engine_ids);
        assert_eq!(vocab.task_ids, base.task_ids);
        assert_eq!(vocab.tasks, base.tasks);
        assert_eq!(vocab.append_ids, base.append_ids);
        assert_eq!(vocab.max_transforms, base.max_transforms);
        assert_eq!(vocab.max_expr_stages, base.max_expr_stages);
        assert_eq!(vocab.expr.symbol, base.expr.symbol);
        assert_eq!(vocab.expr.const_values, base.expr.const_values);
        let base_json: Value = serde_json::from_slice(&read(BASE_VOCABULARY)).unwrap();
        let json: Value = serde_json::from_slice(&read(RESTRICTED_VOCABULARY)).unwrap();
        assert_eq!(json["index"], base_json["index"]);
        assert_eq!(json["counters"], base_json["counters"]);
        for (id, op) in &vocab.ops {
            assert_eq!(
                serde_json::to_value(op).unwrap(),
                serde_json::to_value(&base.ops[id]).unwrap()
            );
        }
    }

    #[test]
    fn search_loader_projects_only_executable_programs() {
        let vocab = restricted();
        assert_eq!(
            vocab.project_proposal("(pipe X (ridge))").unwrap(),
            "(pipe X (ridge))"
        );
        for head in ["extra_trees", "ridge_gcv", "portfolio", "tab_boost"] {
            assert!(vocab
                .project_proposal(&format!("(pipe X (standardize) ({head}))"))
                .is_err());
        }
        // For every task, the replay automaton never offers a removed op, and
        // its Stage masks still offer a head.
        let base: Vocab = read_json(&asset(BASE_VOCABULARY)).unwrap();
        let program = "(pipe X (standardize) (ridge))";
        let ridge = vocab.ops.values().find(|op| op.name == "ridge").unwrap();
        let mut base_offered_removed = BTreeSet::new();
        assert_eq!(vocab.tasks.len(), 4);
        let regression = base.encode(program).unwrap();
        assert_eq!(regression, vocab.encode(program).unwrap());
        for task in &vocab.tasks {
            // `encode_for_task` admits only regression (task_contract), so
            // swap the task token (tokens[2]) to replay the other tasks.
            let mut tokens = regression.clone();
            tokens[2] = vocab.task_ids[task];
            assert_eq!(tokens[2], base.task_ids[task]);
            let masks = vocab.replay(&tokens).unwrap();
            assert!(
                masks
                    .iter()
                    .flatten()
                    .filter(|id| base.ops.contains_key(id))
                    .all(|id| vocab.ops.contains_key(id)),
                "{task}: removed op offered"
            );
            // masks[i] is the legal set for tokens[i + 1]; tokens[3] is the
            // first stage and the ridge token is chosen in a Stage phase.
            let ridge_at = tokens.iter().position(|&t| t == ridge.token_id).unwrap();
            for stage_mask in [&masks[2], &masks[ridge_at - 1]] {
                assert!(
                    stage_mask
                        .iter()
                        .any(|id| vocab.engine_heads[ENGINE].contains(id)),
                    "{task}: Stage mask offers no head"
                );
            }
            assert!(masks[ridge_at - 1].contains(&ridge.token_id));
            for mask in base.replay(&tokens).unwrap() {
                base_offered_removed.extend(
                    mask.into_iter()
                        .filter(|id| base.ops.contains_key(id) && !vocab.ops.contains_key(id)),
                );
            }
        }
        // The base grammar does offer the removed heads on these replays, so
        // the check above is not vacuous.
        let offered: BTreeSet<&str> = base_offered_removed
            .iter()
            .map(|id| base.ops[id].name.as_str())
            .collect();
        for head in [
            "extra_trees",
            "logit_ridge",
            "multiclass_ridge",
            "tab_boost",
        ] {
            assert!(offered.contains(head), "{head} never offered by base");
        }
    }

    #[test]
    fn derivation_fails_closed_on_bad_inputs() {
        let base = read(BASE_VOCABULARY);
        let mut list: Value = serde_json::from_slice(&read(EXECUTABLE_OPS)).unwrap();
        list["executable"]
            .as_array_mut()
            .unwrap()
            .retain(|op| op["name"] != "ridge");
        assert!(derive_restricted_vocabulary(&base, &serde_json::to_vec(&list).unwrap()).is_err());
        let mut list: Value = serde_json::from_slice(&read(EXECUTABLE_OPS)).unwrap();
        list["schema"] = Value::String("other".into());
        assert!(derive_restricted_vocabulary(&base, &serde_json::to_vec(&list).unwrap()).is_err());

        // A quarantined op moved into `executable` is rejected, whether it
        // stays in `not_executable` or is removed from it.
        let original: Value = serde_json::from_slice(&read(EXECUTABLE_OPS)).unwrap();
        let quarantined = original["not_executable"]
            .as_array()
            .unwrap()
            .iter()
            .find(|op| op["name"] == "rff_orthogonal")
            .unwrap()
            .clone();
        let mut list = original.clone();
        list["executable"]
            .as_array_mut()
            .unwrap()
            .push(quarantined.clone());
        let err = derive_restricted_vocabulary(&base, &serde_json::to_vec(&list).unwrap())
            .unwrap_err()
            .to_string();
        assert!(err.contains("lifecycle quarantined"), "{err}");
        list["not_executable"]
            .as_array_mut()
            .unwrap()
            .retain(|op| op["name"] != "rff_orthogonal");
        assert!(derive_restricted_vocabulary(&base, &serde_json::to_vec(&list).unwrap()).is_err());

        // Relabelling it does not help: the name is then listed twice.
        let mut list = original.clone();
        let mut relabelled = quarantined;
        relabelled["lifecycle"] = Value::String("generatable".into());
        list["executable"].as_array_mut().unwrap().push(relabelled);
        let err = derive_restricted_vocabulary(&base, &serde_json::to_vec(&list).unwrap())
            .unwrap_err()
            .to_string();
        assert!(err.contains("more than once"), "{err}");

        // A not-executable entry must be quarantined, and lifecycle is required.
        let mut list = original.clone();
        list["not_executable"][0]["lifecycle"] = Value::String("compatibility".into());
        assert!(derive_restricted_vocabulary(&base, &serde_json::to_vec(&list).unwrap()).is_err());
        let mut list = original;
        list["executable"][0]
            .as_object_mut()
            .unwrap()
            .remove("lifecycle");
        assert!(derive_restricted_vocabulary(&base, &serde_json::to_vec(&list).unwrap()).is_err());
    }
}
