use anyhow::Result;
use std::collections::BTreeMap;

use super::spec::{append_token, ActionVocab, ExprSpec, GrammarSpec, OpSpec, ABSENT, SPECIALS};
use crate::grammar::{Expr, Op, Slot};

fn build_expr(
    spec: &ExprSpec,
    tokens: &mut Vec<String>,
    index: &mut BTreeMap<String, usize>,
) -> Result<Expr> {
    let var_map_ids = spec
        .vars_map
        .iter()
        .map(|name| append_token(tokens, index, format!("EXPR_VAR={name}")))
        .collect::<Vec<_>>();
    let var_common_ids = spec
        .vars_common
        .iter()
        .map(|name| append_token(tokens, index, format!("EXPR_VAR={name}")))
        .collect::<Vec<_>>();
    let var_col_ids = (0..spec.max_col)
        .map(|i| append_token(tokens, index, format!("EXPR_VAR=col{i}")))
        .collect::<Vec<_>>();
    let const_ids = spec
        .consts
        .iter()
        .map(|c| append_token(tokens, index, format!("EXPR_CONST={c}")))
        .collect::<Vec<_>>();
    let const_values = spec
        .consts
        .iter()
        .map(|c| c.parse::<f64>().map_err(anyhow::Error::from))
        .collect::<Result<Vec<_>>>()?;
    let mut func_arity = BTreeMap::new();
    let mut is_binop = BTreeMap::new();
    let mut symbol = BTreeMap::new();
    for (names, arity, binop) in [
        (&spec.funcs1, 1, false),
        (&spec.funcs2, 2, false),
        (&spec.funcs3, 3, false),
        (&spec.binops, 2, true),
    ] {
        let prefix = if binop { "EXPR_BINOP=" } else { "EXPR_FUNC=" };
        for name in names {
            let id = append_token(tokens, index, format!("{prefix}{name}"));
            func_arity.insert(id, arity);
            is_binop.insert(id, binop);
            symbol.insert(id, name.clone());
        }
    }
    let end_id = append_token(tokens, index, "EXPR_END".to_owned());
    for (id, name) in var_map_ids.iter().zip(&spec.vars_map) {
        symbol.insert(*id, name.clone());
    }
    for (id, name) in var_common_ids.iter().zip(&spec.vars_common) {
        symbol.insert(*id, name.clone());
    }
    for (id, i) in var_col_ids.iter().zip(0..) {
        symbol.insert(*id, format!("col({i})"));
    }
    for (id, value) in const_ids.iter().zip(&spec.consts) {
        symbol.insert(*id, value.clone());
    }
    Ok(Expr {
        var_map_ids,
        var_col_ids,
        var_common_ids,
        const_ids,
        const_values,
        func_arity,
        end_id,
        symbol,
        is_binop,
        max_depth: spec.max_depth,
        max_nodes: spec.max_nodes,
    })
}

pub(super) fn add_spec_token(
    tokens: &mut Vec<String>,
    index: &mut BTreeMap<String, usize>,
    token: String,
    unique: bool,
) -> Result<usize> {
    if unique {
        append_unique(tokens, index, token)
    } else {
        Ok(append_token(tokens, index, token))
    }
}

pub(super) fn build_slots(
    op: &OpSpec,
    name: &str,
    engine: &str,
    tokens: &mut Vec<String>,
    index: &mut BTreeMap<String, usize>,
    unique: bool,
) -> Result<Vec<Slot>> {
    op.params
        .iter()
        .map(|param| {
            let bin_texts = param
                .bins
                .iter()
                .map(format_param_token)
                .collect::<Vec<_>>();
            let val_ids = bin_texts
                .iter()
                .map(|text| {
                    add_spec_token(
                        tokens,
                        index,
                        format!("VAL={engine}:{name}:{}={text}", param.name),
                        unique,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            let absent_id = if param.required {
                None
            } else {
                Some(add_spec_token(
                    tokens,
                    index,
                    format!("VAL={engine}:{name}:{}={ABSENT}", param.name),
                    unique,
                )?)
            };
            Ok(Slot {
                param: param.name.clone(),
                kind: param.kind.clone(),
                val_ids,
                bin_texts,
                absent_id,
                is_expr: param.kind == "expr",
                required: param.required,
                default: param.default.clone(),
            })
        })
        .collect()
}

pub(super) fn build_op(
    op: &OpSpec,
    name: &str,
    engine: &str,
    token_id: usize,
    slots: Vec<Slot>,
) -> Op {
    Op {
        token_id,
        engine: engine.to_owned(),
        name: name.to_owned(),
        role: op.role.clone(),
        slots,
        allow_append: op.allow_append && op.role == "transform",
        takes_expr: op.takes_expr,
        generate: op.generate,
    }
}

/// Build the canonical action table. Operators in each group are sorted by name, as in Python.
pub fn build_action_vocab(specs: &[GrammarSpec]) -> Result<ActionVocab> {
    let mut tokens = SPECIALS
        .iter()
        .map(|token| (*token).to_owned())
        .collect::<Vec<_>>();
    let mut index = tokens
        .iter()
        .enumerate()
        .map(|(i, token)| (token.clone(), i))
        .collect::<BTreeMap<_, _>>();
    let engines = specs
        .iter()
        .map(|spec| spec.engine.clone())
        .collect::<Vec<_>>();
    let mut engine_ids = BTreeMap::new();
    for engine in &engines {
        let id = append_token(&mut tokens, &mut index, format!("ENGINE={engine}"));
        engine_ids.insert(engine.clone(), id);
    }
    let mut head_tasks = BTreeMap::new();
    let expr_spec = specs
        .iter()
        .find_map(|spec| spec.expr.as_ref())
        .cloned()
        .unwrap_or_default();
    let expr = build_expr(&expr_spec, &mut tokens, &mut index)?;
    let append_ids = [
        append_token(&mut tokens, &mut index, format!("APPEND={ABSENT}")),
        append_token(&mut tokens, &mut index, "APPEND=0".to_owned()),
        append_token(&mut tokens, &mut index, "APPEND=1".to_owned()),
    ];
    let mut ops = BTreeMap::new();
    let mut op_id_by_name = BTreeMap::new();
    let mut engine_transforms = BTreeMap::new();
    let mut engine_heads = BTreeMap::new();
    for spec in specs {
        let mut transforms = Vec::new();
        let mut heads = Vec::new();
        for (group, bucket) in [
            (&spec.transforms, &mut transforms),
            (&spec.heads, &mut heads),
        ] {
            let mut ordered = group.iter().collect::<Vec<_>>();
            ordered.sort_by(|a, b| a.0.cmp(&b.0));
            for (name, op) in ordered {
                let id = append_token(
                    &mut tokens,
                    &mut index,
                    format!("OP={}:{}", spec.engine, name),
                );
                let slots = build_slots(op, name, &spec.engine, &mut tokens, &mut index, false)?;
                ops.insert(id, build_op(op, name, &spec.engine, id, slots));
                op_id_by_name.insert((spec.engine.clone(), name.clone()), id);
                bucket.push(id);
                if op.role == "head" {
                    head_tasks.insert(id, op.tasks.clone());
                }
            }
        }
        engine_transforms.insert(spec.engine.clone(), transforms);
        engine_heads.insert(spec.engine.clone(), heads);
    }
    let mut tasks = Vec::new();
    for spec in specs {
        for (_, op) in &spec.heads {
            for task in &op.tasks {
                if !tasks.contains(task) {
                    tasks.push(task.clone());
                }
            }
        }
    }
    if tasks.is_empty() {
        tasks.push("regression".to_owned());
    }
    let mut task_ids = BTreeMap::new();
    for task in &tasks {
        let id = append_token(&mut tokens, &mut index, format!("TASK={task}"));
        task_ids.insert(task.clone(), id);
    }
    let version = specs
        .iter()
        .map(|spec| format!("{}:{}", spec.engine, spec.version))
        .collect::<Vec<_>>()
        .join("+");
    Ok(ActionVocab {
        tokens,
        index,
        engines,
        engine_ids,
        ops,
        op_id_by_name,
        engine_transforms,
        engine_heads,
        append_ids,
        expr,
        max_transforms: specs
            .iter()
            .map(|spec| (spec.engine.clone(), spec.max_transforms))
            .collect(),
        max_expr_stages: specs
            .iter()
            .map(|spec| (spec.engine.clone(), spec.max_expr_stages))
            .collect(),
        version,
        tasks,
        task_ids,
        head_tasks,
        counters: BTreeMap::new(),
    })
}
