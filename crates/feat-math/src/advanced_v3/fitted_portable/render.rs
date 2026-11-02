use super::*;

// Substitute only the template itself; user strings resembling markers remain literal.
pub(super) fn render(mut template: &str, values: &[(&str, String)]) -> String {
    for (key, _) in values {
        assert!(template.contains(key), "missing template marker {key}");
    }
    let mut remaining = template.to_owned();
    for (key, _) in values {
        remaining = remaining.replace(key, "");
    }
    for fragment in remaining.split("__").skip(1).step_by(2) {
        assert!(
            !fragment
                .bytes()
                .all(|b| b.is_ascii_uppercase() || b == b'_'),
            "unknown template marker"
        );
    }
    let mut output = String::new();
    while let Some((offset, key, value)) = values
        .iter()
        .filter_map(|(key, value)| template.find(key).map(|offset| (offset, *key, value)))
        .min_by_key(|item| item.0)
    {
        output.push_str(&template[..offset]);
        output.push_str(value);
        template = &template[offset + key.len()..];
    }
    output.push_str(template);
    output
}

pub(super) fn floats(values: &[f64]) -> String {
    format!(
        "&[{}]",
        values
            .iter()
            .map(|value| format!("f64::from_bits(0x{:016x})", value.to_bits()))
            .collect::<Vec<_>>()
            .join(",")
    )
}
pub(super) fn linear_call(value: &Linear) -> String {
    format!(
        "linear(x,{},{},{},{:.17e},{})",
        floats(&value.means),
        floats(&value.scales),
        floats(&value.weights),
        value.bound,
        value.finite_input
    )
}
pub(super) fn rust_source(state: &State) -> String {
    let mut encode = String::from("let mut x = Vec::new();\n");
    for (index, input) in state.input.iter().enumerate() {
        match input {
            Input::Numeric {
                fill,
                indicator,
                restore_missing,
            } => {
                encode.push_str(&format!("let raw=&row[{index}]; let absent=missing(raw); let parsed=if absent {{ f64::NAN }} else {{ raw.trim().parse::<f64>().map_err(|_| \"invalid numeric input\".to_string())? }}; let absent=absent || !parsed.is_finite(); x.push(if absent && {restore_missing} {{ f64::NAN }} else if absent {{ f64::from_bits(0x{:016x}) }} else {{ parsed }});\n", fill.to_bits()));
                if *indicator {
                    encode.push_str("x.push(if absent { 1.0 } else { 0.0 });\n");
                }
            }
            Input::Onehot {
                vocabulary,
                indices,
            } => {
                let values = vocabulary
                    .iter()
                    .map(|value| format!("{value:?}"))
                    .collect::<Vec<_>>()
                    .join(",");
                encode.push_str(&format!("let key=category(&row[{index}]); let vocabulary: &[&str]=&[{values}]; let found=vocabulary.iter().position(|value| *value == key);\n"));
                for position in indices {
                    encode.push_str(&format!(
                        "x.push(if found==Some({position}) {{ 1.0 }} else {{ 0.0 }});\n"
                    ));
                }
            }
        }
    }
    encode.push_str("Ok(x)");
    let predict = match &state.head {
        Head::Scalar { binary, state } => {
            let value = scalar::rust(state, "x");
            if *binary {
                format!("{{let p={value};vec![1.0-p,p]}} ")
            } else {
                format!("vec![finite({value})]")
            }
        }
        Head::Regression(value) => {
            format!("vec![finite({}.clamp(-60.0,60.0))]", linear_call(value))
        }
        Head::Binary(value) => {
            format!("{{ let p=sigmoid({}); vec![1.0-p,p] }}", linear_call(value))
        }
        Head::Softmax(values) => format!(
            "softmax(vec![{}])",
            values
                .iter()
                .map(|value| format!("finite({})", linear_call(value)))
                .collect::<Vec<_>>()
                .join(",")
        ),
        Head::Ovr(values) => format!(
            "normalize(vec![{}])",
            values
                .iter()
                .map(|value| format!("sigmoid({})", linear_call(value)))
                .collect::<Vec<_>>()
                .join(",")
        ),
        Head::Lda {
            centers,
            scales,
            priors,
        } => format!(
            "softmax(vec![{}])",
            centers
                .iter()
                .zip(priors)
                .map(|(center, prior)| format!(
                    "f64::from_bits(0x{:016x})-0.5*distance(x,{},{},true)",
                    prior.to_bits(),
                    floats(center),
                    floats(scales)
                ))
                .collect::<Vec<_>>()
                .join(",")
        ),
        Head::Centroid { centers, scales } => format!(
            "softmax(vec![{}])",
            centers
                .iter()
                .map(|center| format!("-distance(x,{},{},false)", floats(center), floats(scales)))
                .collect::<Vec<_>>()
                .join(",")
        ),
    };
    render(
        include_str!("../fitted_portable.rs.txt"),
        &[
            ("__WIDTH__", state.input.len().to_string()),
            ("__ENCODE__", encode),
            ("__PREDICT__", predict),
        ],
    )
}
