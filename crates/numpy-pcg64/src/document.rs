//! `sampling.json` writers in the exact byte layout of the jope-dime
//! exporters: `json.dumps(document, indent=2) + "\n"` with the document keys
//! in insertion order (`schema`, `seed`, `numpy_version`, then `window` and
//! `views` for targets, then `recipes`) and the recipes in the caller's key
//! order.

use crate::generator::SampleError;
use crate::recipes::{
    foundation_recipe, target_key, target_recipe, FOUNDATION_SCHEMA, FOUNDATION_SEED,
    NUMPY_VERSION, TARGET_SCHEMA, TARGET_SEED, TARGET_VIEWS, TARGET_WINDOW,
};

/// A value in the small JSON subset these documents use.
enum Json<'a> {
    Str(&'a str),
    Int(u64),
    List(Vec<Json<'a>>),
    Object(Vec<(String, Json<'a>)>),
}

fn indices(values: &[u64]) -> Json<'static> {
    Json::List(values.iter().map(|&v| Json::Int(v)).collect())
}

fn windows(values: &[Vec<u64>]) -> Json<'static> {
    Json::List(values.iter().map(|w| indices(w)).collect())
}

fn push_str(out: &mut String, text: &str) {
    // `json.dumps` with the default `ensure_ascii=True`.
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            ' '..='~' => out.push(ch),
            _ => {
                let mut units = [0u16; 2];
                for unit in ch.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
        }
    }
    out.push('"');
}

fn newline(out: &mut String, depth: usize) {
    out.push('\n');
    out.extend(std::iter::repeat_n(' ', 2 * depth));
}

fn write(out: &mut String, value: &Json<'_>, depth: usize) {
    match value {
        Json::Str(text) => push_str(out, text),
        Json::Int(v) => out.push_str(&v.to_string()),
        Json::List(items) if items.is_empty() => out.push_str("[]"),
        Json::Object(items) if items.is_empty() => out.push_str("{}"),
        Json::List(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                newline(out, depth + 1);
                write(out, item, depth + 1);
            }
            newline(out, depth);
            out.push(']');
        }
        Json::Object(items) => {
            out.push('{');
            for (i, (key, item)) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                newline(out, depth + 1);
                push_str(out, key);
                out.push_str(": ");
                write(out, item, depth + 1);
            }
            newline(out, depth);
            out.push('}');
        }
    }
}

fn render(document: Json<'_>) -> String {
    let mut out = String::new();
    write(&mut out, &document, 0);
    out.push('\n');
    out
}

/// The foundation `sampling.json` for cut lengths `lengths`, in that order.
pub fn foundation_document(lengths: &[usize]) -> String {
    let recipes = lengths
        .iter()
        .map(|&n| {
            let recipe = foundation_recipe(n);
            let body = Json::Object(vec![
                ("context".to_owned(), indices(&recipe.context)),
                ("query".to_owned(), indices(&recipe.query)),
            ]);
            (n.to_string(), body)
        })
        .collect();
    render(Json::Object(vec![
        ("schema".to_owned(), Json::Str(FOUNDATION_SCHEMA)),
        ("seed".to_owned(), Json::Int(FOUNDATION_SEED)),
        ("numpy_version".to_owned(), Json::Str(NUMPY_VERSION)),
        ("recipes".to_owned(), Json::Object(recipes)),
    ]))
}

/// The target `sampling.json` for `(task, length)` pairs, in that order.
pub fn target_document(keys: &[(u32, usize)]) -> Result<String, SampleError> {
    let recipes = keys
        .iter()
        .map(|&(task, n)| {
            let recipe = target_recipe(task, n)?;
            let body = Json::Object(vec![
                ("classical".to_owned(), windows(&recipe.classical)),
                ("neural".to_owned(), windows(&recipe.neural)),
            ]);
            Ok((target_key(task, n), body))
        })
        .collect::<Result<Vec<_>, SampleError>>()?;
    Ok(render(Json::Object(vec![
        ("schema".to_owned(), Json::Str(TARGET_SCHEMA)),
        ("seed".to_owned(), Json::Int(TARGET_SEED)),
        ("numpy_version".to_owned(), Json::Str(NUMPY_VERSION)),
        ("window".to_owned(), Json::Int(TARGET_WINDOW as u64)),
        ("views".to_owned(), Json::Int(TARGET_VIEWS as u64)),
        ("recipes".to_owned(), Json::Object(recipes)),
    ])))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_matches_python_indent_two() {
        let doc = Json::Object(vec![
            ("a".to_owned(), Json::List(vec![Json::Int(1), Json::Int(2)])),
            ("b".to_owned(), Json::List(vec![])),
            ("c".to_owned(), Json::Object(vec![])),
            ("d".to_owned(), Json::Str("q\"\u{e9}")),
        ]);
        assert_eq!(
            render(doc),
            "{\n  \"a\": [\n    1,\n    2\n  ],\n  \"b\": [],\n  \"c\": {},\n  \"d\": \"q\\\"\\u00e9\"\n}\n"
        );
    }

    #[test]
    fn empty_target_length_is_rejected() {
        assert_eq!(
            target_document(&[(0, 0)]),
            Err(SampleError::EmptyPopulation)
        );
    }
}
