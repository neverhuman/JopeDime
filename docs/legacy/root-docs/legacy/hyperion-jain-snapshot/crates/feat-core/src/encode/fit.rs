use super::{
    CatEncoding, ColumnEncoding, ColumnPlan, Dictionary, Encoders, MissingPolicy, NumericScaler,
    PreprocessPlan, TargetScaler, DEFAULT_HASH_BUCKETS, DEFAULT_WIDTH_CAP, LOW_CARDINALITY_MAX,
    MIN_HASH_BUCKETS, PREPROCESS_VERSION,
};
use crate::data::parse_field;
use crate::ingest::{is_missing_token, ColKind, Frame, RawColumn};
use crate::task::{TaskKind, TaskSpec};
use std::collections::HashMap;

pub(super) fn parse_numeric_token(s: &str) -> Option<f32> {
    let t = s.trim();
    if is_missing_token(t) {
        return None;
    }
    match t.to_ascii_lowercase().as_str() {
        "true" | "yes" => return Some(1.0),
        "false" | "no" => return Some(0.0),
        "inf" | "+inf" | "-inf" | "infinity" | "+infinity" | "-infinity" => return None,
        _ => {}
    }
    let v = t.parse::<f64>().ok()?;
    if v.is_finite() && v >= f32::MIN as f64 && v <= f32::MAX as f64 {
        Some(v as f32)
    } else {
        None
    }
}

pub(super) fn normalize_category(s: &str) -> Option<String> {
    let t = s.trim();
    if is_missing_token(t) {
        None
    } else {
        Some(t.to_string())
    }
}

fn fit_numeric(col: &RawColumn) -> NumericScaler {
    let mut vals: Vec<f32> = col
        .tokens
        .iter()
        .filter_map(|t| parse_numeric_token(t))
        .collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = vals.first().copied().unwrap_or(0.0);
    let max = vals.last().copied().unwrap_or(min);
    let constant = vals.is_empty() || (max - min).abs() <= 1e-12;
    NumericScaler {
        min,
        max,
        fill: min,
        constant,
    }
}

pub(super) fn fit_dictionary(col: &RawColumn) -> Dictionary {
    let n = col.tokens.len().max(1) as f32;
    let mut levels: Vec<String> = col
        .tokens
        .iter()
        .filter_map(|t| normalize_category(t))
        .collect();
    levels.sort();
    levels.dedup();
    let mut map = HashMap::with_capacity(levels.len());
    for (code, lv) in levels.iter().enumerate() {
        map.insert(lv.clone(), code as u32);
    }
    let mut counts = vec![0u32; levels.len()];
    for t in &col.tokens {
        if let Some(tok) = normalize_category(t) {
            if let Some(&c) = map.get(&tok) {
                counts[c as usize] += 1;
            }
        }
    }
    let freq = counts.iter().map(|&c| c as f32 / n).collect();
    Dictionary { map, freq }
}

pub(super) fn output_names(input: &str, enc: &ColumnEncoding) -> Vec<String> {
    match enc {
        ColumnEncoding::Numeric { .. }
        | ColumnEncoding::Ordinal { .. }
        | ColumnEncoding::Frequency { .. } => vec![input.to_string()],
        ColumnEncoding::OneHot { levels } => {
            let mut out = Vec::with_capacity(levels.len() + 2);
            out.push(format!("{input}::__MISSING__"));
            for lv in levels {
                out.push(format!("{input}=={lv}"));
            }
            out.push(format!("{input}::__OTHER__"));
            out
        }
        ColumnEncoding::Hash { buckets } => {
            let mut out = Vec::with_capacity(buckets + 1);
            out.push(format!("{input}::__MISSING__"));
            for b in 0..*buckets {
                out.push(format!("{input}::hash{b:02}"));
            }
            out
        }
    }
}

fn encoded_width(columns: &[ColumnPlan]) -> usize {
    columns.iter().map(|c| c.output_names.len()).sum()
}

fn enforce_width_cap(columns: &mut [ColumnPlan], cap: usize, issues: &mut Vec<String>) {
    while encoded_width(columns) > cap {
        let candidate = columns
            .iter()
            .enumerate()
            .filter_map(|(i, c)| match c.encoding {
                ColumnEncoding::Hash { buckets } if buckets > MIN_HASH_BUCKETS => {
                    Some((i, buckets))
                }
                _ => None,
            })
            .max_by_key(|&(i, buckets)| (buckets, std::cmp::Reverse(i)));
        let Some((idx, buckets)) = candidate else {
            break;
        };
        let new_buckets = (buckets / 2).max(MIN_HASH_BUCKETS);
        columns[idx].encoding = ColumnEncoding::Hash {
            buckets: new_buckets,
        };
        columns[idx].output_names = output_names(&columns[idx].input_name, &columns[idx].encoding);
        issues.push(format!(
            "categorical column '{}' hash buckets shrank from {buckets} to {new_buckets} to honor width cap {cap}",
            columns[idx].input_name
        ));
    }

    while encoded_width(columns) > cap {
        let candidate = columns
            .iter()
            .enumerate()
            .filter_map(|(i, c)| match &c.encoding {
                ColumnEncoding::OneHot { .. } | ColumnEncoding::Hash { .. } => {
                    Some((i, c.output_names.len()))
                }
                _ => None,
            })
            .max_by_key(|&(i, width)| (width, std::cmp::Reverse(i)));
        let Some((idx, width)) = candidate else {
            break;
        };
        let dict = Dictionary {
            map: HashMap::new(),
            freq: Vec::new(),
        };
        columns[idx].encoding = ColumnEncoding::Frequency { dict };
        columns[idx].output_names = output_names(&columns[idx].input_name, &columns[idx].encoding);
        issues.push(format!(
            "categorical column '{}' fell back from width {width} to scalar frequency to honor width cap {cap}",
            columns[idx].input_name
        ));
    }
}

pub fn fit_preprocess_plan(train: &Frame, task: &TaskSpec, policy: CatEncoding) -> PreprocessPlan {
    fit_preprocess_plan_with_cap(train, task, policy, DEFAULT_WIDTH_CAP)
}

pub fn fit_preprocess_plan_with_cap(
    train: &Frame,
    task: &TaskSpec,
    policy: CatEncoding,
    width_cap: usize,
) -> PreprocessPlan {
    let mut issues = train.issues.clone();
    let mut columns = Vec::with_capacity(train.feats.len());
    for col in &train.feats {
        let encoding = match col.kind {
            ColKind::Numeric => ColumnEncoding::Numeric {
                scaler: fit_numeric(col),
            },
            ColKind::Categorical => {
                let dict = fit_dictionary(col);
                match policy {
                    CatEncoding::Ordinal => ColumnEncoding::Ordinal { dict },
                    CatEncoding::Frequency => ColumnEncoding::Frequency { dict },
                    CatEncoding::Hybrid => {
                        let mut levels: Vec<String> = dict.map.keys().cloned().collect();
                        levels.sort();
                        if levels.len() <= LOW_CARDINALITY_MAX {
                            ColumnEncoding::OneHot { levels }
                        } else {
                            ColumnEncoding::Hash {
                                buckets: DEFAULT_HASH_BUCKETS,
                            }
                        }
                    }
                }
            }
        };
        columns.push(ColumnPlan {
            input_name: col.name.clone(),
            kind: col.kind,
            output_names: output_names(&col.name, &encoding),
            encoding,
        });
    }
    enforce_width_cap(&mut columns, width_cap, &mut issues);

    // If cap backup created empty frequency dicts, refit the dictionary for those columns.
    for plan in &mut columns {
        if matches!(
            &plan.encoding,
            ColumnEncoding::Frequency { dict } if dict.map.is_empty()
        ) {
            if let Some(col) = train.feats.iter().find(|c| c.name == plan.input_name) {
                plan.encoding = ColumnEncoding::Frequency {
                    dict: fit_dictionary(col),
                };
                plan.output_names = output_names(&plan.input_name, &plan.encoding);
            }
        }
    }

    let target_scaler = if task.kind == TaskKind::Regression {
        let raw: Vec<f32> = train.targets[train.primary]
            .tokens
            .iter()
            .map(|s| parse_field(s))
            .collect();
        TargetScaler::fit(&raw)
    } else {
        TargetScaler::identity()
    };

    PreprocessPlan {
        version: PREPROCESS_VERSION,
        policy,
        low_cardinality_max: LOW_CARDINALITY_MAX,
        default_hash_buckets: DEFAULT_HASH_BUCKETS,
        width_cap,
        missing: MissingPolicy::default(),
        columns,
        target_scaler,
        issues,
    }
}

/// Legacy wrapper retained for older internal tests. New code should call `fit_preprocess_plan`.
pub fn fit_encoders(train: &Frame, enc: CatEncoding) -> Encoders {
    let task = TaskSpec {
        kind: TaskKind::Binary,
        n_targets: train.targets.len(),
        n_classes: 2,
        class_values: vec![0.0, 1.0],
        prevalence: f64::NAN,
    };
    fit_preprocess_plan(train, &task, enc)
}
