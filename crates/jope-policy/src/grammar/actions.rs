//! Structured action/token conversion corresponding to `jope.policy.tokens_actions`.
use super::{render_expr, Vocab, BOS, EOS};
use anyhow::{bail, Context, Result};
use serde_json::{Map, Number, Value};
use std::collections::BTreeMap;

/// Encode parsed action objects as `[BOS, ENGINE, TASK, ... stages ..., EOS]`.
/// Counters are additive to match the Python vocabulary's mutable counter map.
pub fn encode_actions(
    engine: &str,
    actions: &Value,
    vocab: &Vocab,
    task: &str,
    counters: &mut BTreeMap<String, u64>,
) -> Result<Vec<usize>> {
    let task = if vocab.task_ids.contains_key(task) {
        task
    } else {
        vocab.tasks.first().context("vocabulary has no tasks")?
    };
    let mut out = vec![
        BOS,
        *vocab
            .engine_ids
            .get(engine)
            .with_context(|| engine.to_string())?,
        *vocab.task_ids.get(task).with_context(|| task.to_string())?,
    ];
    let mut stages = actions
        .get("transforms")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    stages.push(actions.get("head").cloned().context("head")?);
    for stage in stages {
        let name = stage.get("op").and_then(Value::as_str).context("op")?;
        let op = vocab
            .ops
            .values()
            .find(|op| op.engine == engine && op.name == name)
            .with_context(|| format!("unknown op {engine}:{name}"))?;
        out.push(op.token_id);
        let params = stage
            .get("params")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        let append_flag = params.get("append").and_then(Value::as_bool);
        for slot in &op.slots {
            let value = params.get(&slot.param);
            if slot.is_expr {
                let value = value
                    .filter(|value| !value.is_null())
                    .or_else(|| (!slot.default.is_null()).then_some(&slot.default));
                let text = value.map(value_text).unwrap_or_default();
                if let Some(i) = slot.bin_texts.iter().position(|s| s == &text) {
                    out.push(slot.val_ids[i]);
                } else {
                    out.extend(super::parse_expr_tokens(&text, &vocab.expr)?);
                    out.push(vocab.expr.end_id);
                }
                continue;
            }
            let Some(value) = value.filter(|v| !v.is_null()) else {
                if let Some(id) = slot.absent_id {
                    out.push(id);
                } else if let Some(id) = slot.val_ids.first() {
                    *counters.entry("required_defaulted".into()).or_default() += 1;
                    out.push(*id);
                } else {
                    bail!("{engine}:{name}.{}: no bins and no absent", slot.param);
                }
                continue;
            };
            let text = format_param_token(value);
            if let Some(i) = slot.bin_texts.iter().position(|s| s == &text) {
                out.push(slot.val_ids[i]);
            } else if let Some(id) = snap_bin(value, slot) {
                *counters.entry("param_snapped".into()).or_default() += 1;
                out.push(id);
            } else if let Some(id) = slot.absent_id {
                *counters.entry("param_dropped".into()).or_default() += 1;
                out.push(id);
            } else {
                bail!("{engine}:{name}.{}={value:?} not encodable", slot.param);
            }
        }
        if op.allow_append {
            out.push(
                vocab.append_ids[match append_flag {
                    None => 0,
                    Some(false) => 1,
                    Some(true) => 2,
                }],
            );
        }
    }
    out.push(EOS);
    Ok(out)
}

/// Decode token ids to `(engine, actions)`; action field names mirror Python.
pub fn decode_tokens(ids: &[usize], vocab: &Vocab) -> Result<(String, Value)> {
    if ids.first() != Some(&BOS) {
        bail!("sequence must start with BOS");
    }
    let mut i = 1;
    let engine = vocab
        .engine_ids
        .iter()
        .find(|(_, id)| ids.get(i) == Some(*id))
        .map(|(name, _)| name.clone())
        .context("missing ENGINE token")?;
    i += 1;
    let mut task = "regression".to_owned();
    if i < ids.len() {
        if let Some((name, _)) = vocab
            .task_ids
            .iter()
            .find(|(_, id)| ids.get(i) == Some(*id))
        {
            task = name.clone();
            i += 1;
        }
    }
    let mut transforms = Vec::new();
    let mut head = None;
    while i < ids.len() && ids[i] != EOS {
        let tid = ids[i];
        let op = vocab.ops.get(&tid).with_context(|| {
            format!(
                "expected OP token at {i}, got {}",
                vocab
                    .tokens
                    .get(tid)
                    .map(String::as_str)
                    .unwrap_or("<out-of-range>")
            )
        })?;
        i += 1;
        let mut params = Map::new();
        for slot in &op.slots {
            if slot.is_expr {
                if i < ids.len() && slot.val_ids.contains(&ids[i]) {
                    let pos = slot.val_ids.iter().position(|id| *id == ids[i]).unwrap();
                    params.insert(
                        slot.param.clone(),
                        Value::String(slot.bin_texts[pos].clone()),
                    );
                    i += 1;
                    continue;
                }
                let start = i;
                while i < ids.len() && ids[i] != vocab.expr.end_id {
                    i += 1;
                }
                if i >= ids.len() {
                    bail!("unterminated EXPR block");
                }
                let mut pos = 0;
                let text = render_expr(&ids[start..i], &vocab.expr, &mut pos)?;
                i += 1;
                params.insert(slot.param.clone(), Value::String(text));
                continue;
            }
            let tid = *ids.get(i).context("value token absent")?;
            i += 1;
            if slot.absent_id == Some(tid) {
                continue;
            }
            let pos = slot
                .val_ids
                .iter()
                .position(|id| *id == tid)
                .with_context(|| {
                    format!(
                        "bad VAL token for {}.{}: {}",
                        op.name,
                        slot.param,
                        vocab
                            .tokens
                            .get(tid)
                            .map(String::as_str)
                            .unwrap_or("<out-of-range>")
                    )
                })?;
            params.insert(slot.param.clone(), parse_param_token(&slot.bin_texts[pos])?);
        }
        if op.allow_append {
            let tid = *ids.get(i).context("append token absent")?;
            i += 1;
            if tid == vocab.append_ids[2] {
                params.insert("append".into(), Value::Bool(true));
            } else if tid == vocab.append_ids[1] {
                params.insert("append".into(), Value::Bool(false));
            } else if tid != vocab.append_ids[0] {
                bail!("expected APPEND token");
            }
        }
        let stage = serde_json::json!({"op": op.name, "params": params});
        if op.role == "head" {
            head = Some(stage);
            break;
        }
        transforms.push(stage);
    }
    let head = head.context("sequence has no head stage")?;
    Ok((
        engine.clone(),
        serde_json::json!({"engine": engine, "task": task, "transform_count": transforms.len(), "transforms": transforms, "head": head}),
    ))
}

fn value_text(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Null => "None".into(),
        _ => value.to_string(),
    }
}

fn format_param_token(value: &Value) -> String {
    let text = match value {
        Value::Bool(v) => return v.to_string(),
        Value::Number(n) if n.is_i64() => return n.to_string(),
        Value::Number(n) if n.is_u64() => return n.to_string(),
        Value::Number(n) => {
            let x = n.as_f64().unwrap_or_default();
            if (x - x.round()).abs() < 1e-12 {
                return format!("{x:.1}");
            }
            return x.to_string();
        }
        Value::String(s) => s.clone(),
        _ => value.to_string(),
    };
    if !text.is_empty()
        && text
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "_+*.-".contains(c))
    {
        text
    } else {
        format!("\"{}\"", text.replace('\\', "\\\\").replace('"', "\\\""))
    }
}

fn parse_param_token(raw: &str) -> Result<Value> {
    if raw == "true" {
        return Ok(Value::Bool(true));
    }
    if raw == "false" {
        return Ok(Value::Bool(false));
    }
    if raw.starts_with('"') && raw.ends_with('"') {
        return Ok(Value::String(
            raw[1..raw.len() - 1]
                .replace("\\\"", "\"")
                .replace("\\\\", "\\"),
        ));
    }
    if let Ok(v) = raw.parse::<i64>() {
        return Ok(Value::Number(v.into()));
    }
    if let Ok(v) = raw.parse::<f64>() {
        if !v.is_finite() {
            bail!("non-finite param: {raw}");
        }
        return Ok(Number::from_f64(v)
            .map(Value::Number)
            .unwrap_or(Value::Null));
    }
    Ok(Value::String(raw.to_owned()))
}

fn snap_bin(value: &Value, slot: &super::Slot) -> Option<usize> {
    let v = value.as_f64()?;
    slot.val_ids
        .iter()
        .zip(&slot.bin_texts)
        .filter_map(|(&id, text)| {
            text.parse::<f64>().ok().map(|b| {
                let distance = if v != 0. && b != 0. && v.is_sign_positive() == b.is_sign_positive()
                {
                    (v.abs() / b.abs()).ln().abs()
                } else {
                    (v - b).abs() + 1.
                };
                (id, distance)
            })
        })
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(id, _)| id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::{Expr, Op, Slot};

    #[test]
    fn param_text_preserves_python_numeric_and_quoting_rules() {
        assert_eq!(format_param_token(&Value::Bool(false)), "false");
        assert_eq!(format_param_token(&Value::from(3.0)), "3.0");
        assert_eq!(format_param_token(&Value::String("a b".into())), "\"a b\"");
        assert_eq!(parse_param_token("true").unwrap(), Value::Bool(true));
        assert_eq!(parse_param_token("2.5").unwrap(), Value::from(2.5));
    }

    #[test]
    fn structured_actions_snap_and_decode_like_the_python_callers() {
        let vocab = Vocab {
            tokens: vec![
                "PAD".into(),
                "BOS".into(),
                "EOS".into(),
                "ENGINE=e".into(),
                "TASK=regression".into(),
                "OP=e:head".into(),
                "VAL=1".into(),
                "VAL=2".into(),
            ],
            engines: vec!["e".into()],
            engine_ids: BTreeMap::from([("e".into(), 3)]),
            ops: BTreeMap::from([(
                5,
                Op {
                    token_id: 5,
                    engine: "e".into(),
                    name: "head".into(),
                    role: "head".into(),
                    slots: vec![Slot {
                        param: "x".into(),
                        kind: "float".into(),
                        val_ids: vec![6, 7],
                        bin_texts: vec!["1".into(), "2".into()],
                        absent_id: None,
                        is_expr: false,
                        required: true,
                        default: Value::Null,
                    }],
                    allow_append: false,
                    takes_expr: false,
                    generate: true,
                },
            )]),
            engine_transforms: BTreeMap::from([("e".into(), vec![])]),
            engine_heads: BTreeMap::from([("e".into(), vec![5])]),
            append_ids: [0, 0, 0],
            expr: Expr {
                var_map_ids: vec![],
                var_col_ids: vec![],
                var_common_ids: vec![],
                const_ids: vec![],
                const_values: vec![],
                func_arity: BTreeMap::new(),
                end_id: 0,
                symbol: BTreeMap::new(),
                is_binop: BTreeMap::new(),
                max_depth: 0,
                max_nodes: 0,
            },
            max_transforms: BTreeMap::from([("e".into(), 0)]),
            max_expr_stages: BTreeMap::from([("e".into(), 0)]),
            tasks: vec!["regression".into()],
            task_ids: BTreeMap::from([("regression".into(), 4)]),
            head_tasks: BTreeMap::new(),
            restriction: None,
        };
        let actions = serde_json::json!({"head": {"op":"head", "params":{"x":1.8}}});
        let tokens =
            encode_actions("e", &actions, &vocab, "regression", &mut BTreeMap::new()).unwrap();
        assert_eq!(tokens, vec![1, 3, 4, 5, 7, 2]);
        let (engine, decoded) = decode_tokens(&tokens, &vocab).unwrap();
        assert_eq!(engine, "e");
        assert_eq!(decoded["head"]["params"]["x"], 2);
    }
}
