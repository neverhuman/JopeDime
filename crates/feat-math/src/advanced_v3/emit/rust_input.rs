use super::*;

pub(crate) fn rust_f64(value: f64) -> String {
    format!("{value:.17e}")
}

pub(crate) fn rust_f64_slice(values: &[f64]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(|value| rust_f64(*value))
            .collect::<Vec<_>>()
            .join(",")
    )
}

pub(crate) fn rust_usize_slice(values: &[usize]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",")
    )
}

pub(crate) fn rust_string(value: &str) -> String {
    format!("{value:?}")
}

pub(crate) fn rust_input_body(input: &FittedInput, missing_aware: bool) -> String {
    let mut source = String::from(
        "    let mut x: Vec<f64> = Vec::new();\n    let mut missing_mask: Vec<bool> = Vec::new();\n",
    );
    for (j, column) in input.columns.iter().enumerate() {
        match column {
            FittedColumn::Numeric { fill, indicator } => {
                source.push_str(&format!(
                    "    let raw_missing_{j} = is_missing(&row[{j}]);\n    let parsed_{j} = if raw_missing_{j} {{ {fill} }} else {{ row[{j}].trim().parse::<f64>().map_err(|_| format!(\"invalid numeric value {{}}\", row[{j}]))? }};\n    let absent_{j} = raw_missing_{j} || !parsed_{j}.is_finite();\n    let value_{j} = if parsed_{j}.is_finite() {{ parsed_{j} }} else {{ {fill} }};\n    x.push(if absent_{j} && {missing_aware} {{ f64::NAN }} else {{ value_{j} }});\n    missing_mask.push(absent_{j});\n",
                    fill = rust_f64(*fill)
                ));
                if *indicator {
                    source.push_str(&format!(
                        "    x.push(if absent_{j} {{ 1.0 }} else {{ 0.0 }});\n    missing_mask.push(false);\n"
                    ));
                }
            }
            FittedColumn::Categorical {
                mode,
                vocabulary,
                counts,
                sums,
                total,
                global,
                onehot,
            } => {
                let vocab = vocabulary
                    .iter()
                    .map(|value| rust_string(value))
                    .collect::<Vec<_>>()
                    .join(",");
                source.push_str(&format!(
                    "    let absent_{j} = is_category_missing(&row[{j}]);\n    let key_{j} = category(&row[{j}]);\n    let vocabulary_{j}: &[&str] = &[{vocab}];\n    let index_{j} = vocabulary_{j}.binary_search(&key_{j}.as_str()).ok();\n"
                ));
                match mode {
                    CategoricalMode::Native => source.push_str(&format!(
                        "    x.push(index_{j}.map_or(-1.0, |index| index as f64));\n    missing_mask.push(absent_{j});\n"
                    )),
                    CategoricalMode::Frequency => source.push_str(&format!(
                        "    let counts_{j}: &[usize] = &{counts};\n    x.push(index_{j}.map_or(0.0, |index| counts_{j}[index] as f64 / {total}.0));\n    missing_mask.push(absent_{j});\n",
                        counts = rust_usize_slice(counts)
                    )),
                    CategoricalMode::Onehot => {
                        for category in onehot {
                            source.push_str(&format!(
                                "    x.push(if index_{j} == Some({category}) {{ 1.0 }} else {{ 0.0 }});\n    missing_mask.push(false);\n"
                            ));
                        }
                    }
                    CategoricalMode::Ordered | CategoricalMode::Hybrid => {
                        source.push_str(&format!(
                            "    let counts_{j}: &[usize] = &{counts};\n    let sums_{j}: &[f64] = &{sums};\n    let ordered_{j} = index_{j}.map_or({global}, |index| (sums_{j}[index] + {prior} * {global}) / (counts_{j}[index] as f64 + {prior}));\n",
                            counts = rust_usize_slice(counts),
                            sums = rust_f64_slice(sums),
                            global = rust_f64(*global),
                            prior = rust_f64(input.config.prior),
                        ));
                        if *mode == CategoricalMode::Hybrid {
                            source.push_str(&format!(
                                "    x.push(index_{j}.map_or(0.0, |index| counts_{j}[index] as f64 / {total}.0));\n    missing_mask.push(absent_{j});\n"
                            ));
                        }
                        source.push_str(&format!(
                            "    x.push(ordered_{j});\n    missing_mask.push({});\n",
                            if *mode == CategoricalMode::Ordered {
                                format!("absent_{j}")
                            } else {
                                "false".to_string()
                            }
                        ));
                    }
                }
            }
        }
    }
    source.push_str("    Ok((x, missing_mask))\n");
    source
}
