use anyhow::{bail, Result};
use std::collections::BTreeMap;

use super::construction::{build_op, build_slots};
use super::spec::{append_unique, ActionVocab, GrammarSpec, CANONICAL_TASKS};

/// Append one engine without modifying any existing token id.
pub fn extend_action_vocab_append_only(
    base: &ActionVocab,
    spec: &GrammarSpec,
) -> Result<ActionVocab> {
    if base.engines.contains(&spec.engine) {
        bail!("engine already present: {}", spec.engine);
    }
    let mut vocab = base.clone();
    let engine_id = append_unique(
        &mut vocab.tokens,
        &mut vocab.index,
        format!("ENGINE={}", spec.engine),
    )?;
    vocab.engines.push(spec.engine.clone());
    vocab.engine_ids.insert(spec.engine.clone(), engine_id);
    let mut transforms = Vec::new();
    let mut heads = Vec::new();
    for (group, bucket) in [
        (&spec.transforms, &mut transforms),
        (&spec.heads, &mut heads),
    ] {
        for (name, op) in group {
            let op_id = append_unique(
                &mut vocab.tokens,
                &mut vocab.index,
                format!("OP={}:{}", spec.engine, name),
            )?;
            let slots = build_slots(
                op,
                name,
                &spec.engine,
                &mut vocab.tokens,
                &mut vocab.index,
                true,
            )?;
            vocab
                .ops
                .insert(op_id, build_op(op, name, &spec.engine, op_id, slots));
            vocab
                .op_id_by_name
                .insert((spec.engine.clone(), name.clone()), op_id);
            bucket.push(op_id);
            if op.role == "head" {
                vocab.head_tasks.insert(op_id, op.tasks.clone());
            }
        }
    }
    vocab
        .engine_transforms
        .insert(spec.engine.clone(), transforms);
    vocab.engine_heads.insert(spec.engine.clone(), heads);
    vocab
        .max_transforms
        .insert(spec.engine.clone(), spec.max_transforms);
    vocab
        .max_expr_stages
        .insert(spec.engine.clone(), spec.max_expr_stages);
    vocab
        .version
        .push_str(&format!("+{}:{}", spec.engine, spec.version));
    Ok(vocab)
}

/// Append the canonical task-conditioning tokens, preserving every pre-existing id.
pub fn append_missing_task_tokens(base: &ActionVocab) -> Result<ActionVocab> {
    append_missing_task_tokens_from(
        base,
        &CANONICAL_TASKS
            .iter()
            .map(|task| (*task).to_owned())
            .collect::<Vec<_>>(),
    )
}

/// Append selected task-conditioning tokens, preserving every pre-existing id.
pub fn append_missing_task_tokens_from(
    base: &ActionVocab,
    tasks: &[String],
) -> Result<ActionVocab> {
    let mut vocab = base.clone();
    for task in tasks {
        if vocab.task_ids.contains_key(task) {
            continue;
        }
        let token = format!("TASK={task}");
        let id = append_unique(&mut vocab.tokens, &mut vocab.index, token.clone())
            .map_err(|_| anyhow::anyhow!("token collision while appending task token: {token}"))?;
        vocab.task_ids.insert(task.clone(), id);
        vocab.tasks.push(task.clone());
    }
    Ok(vocab)
}

/// Update head task capabilities from live grammar without changing any token identity.
pub fn reconcile_head_tasks_from_grammar(
    base: &ActionVocab,
    spec: &GrammarSpec,
    grant: &[String],
    exclude: &[String],
) -> Result<ActionVocab> {
    let mut vocab = base.clone();
    let mut granted = Vec::new();
    for task in grant {
        if !granted.contains(task) {
            granted.push(task.clone());
        }
    }
    let mut unknown = granted
        .iter()
        .filter(|task| !CANONICAL_TASKS.contains(&task.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    if !unknown.is_empty() {
        unknown.sort();
        let listed = unknown
            .iter()
            .map(|task| format!("'{}'", task.replace('\\', "\\\\").replace('\'', "\\'")))
            .collect::<Vec<_>>()
            .join(", ");
        bail!("cannot grant non-canonical task(s): [{listed}]");
    }
    let order = CANONICAL_TASKS
        .iter()
        .enumerate()
        .map(|(i, task)| ((*task).to_owned(), i))
        .collect::<BTreeMap<_, _>>();
    for (name, _) in &spec.heads {
        if exclude.contains(name) {
            continue;
        }
        let Some(op) = spec.op(name) else { continue };
        let Some(op_id) = vocab
            .op_id_by_name
            .get(&(spec.engine.clone(), name.clone()))
            .copied()
        else {
            continue;
        };
        let mut current = vocab.head_tasks.get(&op_id).cloned().unwrap_or_default();
        let additions = granted
            .iter()
            .filter(|task| op.tasks.contains(task) && !current.contains(task))
            .cloned()
            .collect::<Vec<_>>();
        current.extend(additions);
        current.sort_by_key(|task| order.get(task).copied().unwrap_or(usize::MAX));
        vocab.head_tasks.insert(op_id, current);
    }
    Ok(vocab)
}

/// Append parse-compatibility operators to an existing engine without id drift.
pub fn extend_engine_ops_append_only(
    base: &ActionVocab,
    spec: &GrammarSpec,
    names: &[String],
) -> Result<ActionVocab> {
    if !base.engines.contains(&spec.engine) {
        bail!("engine is not present: {}", spec.engine);
    }
    let mut vocab = base.clone();
    for name in names {
        let op = spec
            .op(name)
            .ok_or_else(|| anyhow::anyhow!("unknown compatibility op: {}:{name}", spec.engine))?;
        if vocab
            .op_id_by_name
            .contains_key(&(spec.engine.clone(), name.clone()))
        {
            bail!("operator already present: {}:{name}", spec.engine);
        }
        let op_id = append_unique(
            &mut vocab.tokens,
            &mut vocab.index,
            format!("OP={}:{}", spec.engine, name),
        )?;
        let slots = build_slots(
            op,
            name,
            &spec.engine,
            &mut vocab.tokens,
            &mut vocab.index,
            true,
        )?;
        vocab
            .ops
            .insert(op_id, build_op(op, name, &spec.engine, op_id, slots));
        vocab
            .op_id_by_name
            .insert((spec.engine.clone(), name.clone()), op_id);
        if op.role == "head" {
            vocab
                .engine_heads
                .entry(spec.engine.clone())
                .or_default()
                .push(op_id);
            vocab.head_tasks.insert(op_id, op.tasks.clone());
        } else {
            vocab
                .engine_transforms
                .entry(spec.engine.clone())
                .or_default()
                .push(op_id);
        }
    }
    vocab.version.push_str(&format!("+{}:compat", spec.engine));
    Ok(vocab)
}
