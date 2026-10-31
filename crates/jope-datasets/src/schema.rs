//! A small JSON Schema 2020-12 validator covering the vocabulary the committed schemas use.
//!
//! Supported: `$ref` (local pointers), `type`, `enum`, `const`, `properties`, `required`,
//! `additionalProperties`, `items`, `uniqueItems`, `minItems`, `maxItems`, `pattern`,
//! `minLength`, `maxLength`, `minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum`,
//! `allOf`, `anyOf`, `oneOf`, `not`, `if`/`then`/`else`. Any other assertion keyword in a schema
//! is reported as an error (fail closed) rather than silently ignored.

use serde_json::Value;
use std::collections::HashMap;

const SUPPORTED: &[&str] = &[
    "$ref",
    "type",
    "enum",
    "const",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "uniqueItems",
    "minItems",
    "maxItems",
    "pattern",
    "minLength",
    "maxLength",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "allOf",
    "anyOf",
    "oneOf",
    "not",
    "if",
    "then",
    "else",
];
const ANNOTATIONS: &[&str] = &[
    "$schema",
    "$id",
    "$defs",
    "$comment",
    "title",
    "description",
    "default",
    "examples",
    "format",
    "deprecated",
    "readOnly",
    "writeOnly",
];

type Bound = fn(f64, f64) -> bool;

struct Ctx<'a> {
    root: &'a Value,
    errors: Vec<String>,
    regexes: HashMap<String, regex::Regex>,
}

/// Validate `instance` against `schema`; an empty vector means valid.
pub fn validate(schema: &Value, instance: &Value) -> Vec<String> {
    let mut cx = Ctx {
        root: schema,
        errors: Vec::new(),
        regexes: HashMap::new(),
    };
    cx.check(schema, instance, "");
    cx.errors
}

fn type_matches(ty: &str, v: &Value) -> bool {
    match ty {
        "object" => v.is_object(),
        "array" => v.is_array(),
        "string" => v.is_string(),
        "boolean" => v.is_boolean(),
        "null" => v.is_null(),
        "number" => v.is_number(),
        "integer" => v.is_i64() || v.is_u64() || v.as_f64().is_some_and(|f| f.fract() == 0.0),
        _ => false,
    }
}

impl Ctx<'_> {
    fn resolve(&self, reference: &str) -> Option<Value> {
        let pointer = reference.strip_prefix('#')?;
        self.root.pointer(pointer).cloned()
    }

    fn is_valid(&mut self, schema: &Value, instance: &Value, path: &str) -> bool {
        let saved = std::mem::take(&mut self.errors);
        self.check(schema, instance, path);
        let ok = self.errors.is_empty();
        self.errors = saved;
        ok
    }

    fn regex(&mut self, pattern: &str) -> Option<regex::Regex> {
        if let Some(r) = self.regexes.get(pattern) {
            return Some(r.clone());
        }
        // ECMA-262 has no `\z`; schemas spell "end of input" as the lookahead `(?![\s\S])`,
        // which the `regex` crate cannot parse. It is exactly `\z`, so translate that one idiom.
        match regex::Regex::new(&pattern.replace("(?![\\s\\S])", "\\z")) {
            Ok(r) => {
                self.regexes.insert(pattern.to_string(), r.clone());
                Some(r)
            }
            Err(e) => {
                self.errors
                    .push(format!("schema: invalid pattern {pattern:?}: {e}"));
                None
            }
        }
    }

    #[allow(clippy::too_many_lines)]
    fn check(&mut self, schema: &Value, instance: &Value, path: &str) {
        let here = if path.is_empty() { "/" } else { path };
        let obj = match schema {
            Value::Bool(true) => return,
            Value::Bool(false) => {
                self.errors.push(format!("{here}: schema is false"));
                return;
            }
            Value::Object(o) => o,
            _ => {
                self.errors.push(format!("{here}: schema is not an object"));
                return;
            }
        };
        for key in obj.keys() {
            if !SUPPORTED.contains(&key.as_str()) && !ANNOTATIONS.contains(&key.as_str()) {
                self.errors
                    .push(format!("{here}: unsupported schema keyword {key:?}"));
            }
        }
        if let Some(Value::String(reference)) = obj.get("$ref") {
            match self.resolve(reference) {
                Some(target) => self.check(&target, instance, path),
                None => self
                    .errors
                    .push(format!("{here}: unresolvable $ref {reference:?}")),
            }
        }
        if let Some(ty) = obj.get("type") {
            let ok = match ty {
                Value::String(t) => type_matches(t, instance),
                Value::Array(ts) => ts
                    .iter()
                    .any(|t| t.as_str().is_some_and(|t| type_matches(t, instance))),
                _ => false,
            };
            if !ok {
                self.errors.push(format!(
                    "{here}: expected type {ty}, got {}",
                    kind(instance)
                ));
            }
        }
        if let Some(Value::Array(allowed)) = obj.get("enum") {
            if !allowed.contains(instance) {
                self.errors.push(format!(
                    "{here}: {instance} is not one of {}",
                    Value::Array(allowed.clone())
                ));
            }
        }
        if let Some(c) = obj.get("const") {
            if c != instance {
                self.errors
                    .push(format!("{here}: expected const {c}, got {instance}"));
            }
        }
        if let Value::String(s) = instance {
            if let Some(Value::String(p)) = obj.get("pattern") {
                if let Some(re) = self.regex(p) {
                    if !re.is_match(s) {
                        self.errors
                            .push(format!("{here}: {s:?} does not match {p:?}"));
                    }
                }
            }
            let len = s.chars().count() as u64;
            if let Some(min) = obj.get("minLength").and_then(Value::as_u64) {
                if len < min {
                    self.errors
                        .push(format!("{here}: shorter than minLength {min}"));
                }
            }
            if let Some(max) = obj.get("maxLength").and_then(Value::as_u64) {
                if len > max {
                    self.errors
                        .push(format!("{here}: longer than maxLength {max}"));
                }
            }
        }
        if let Some(n) = instance.as_f64() {
            let bounds: [(&str, Bound); 4] = [
                ("minimum", |n, m| n < m),
                ("maximum", |n, m| n > m),
                ("exclusiveMinimum", |n, m| n <= m),
                ("exclusiveMaximum", |n, m| n >= m),
            ];
            for (kw, bad) in bounds {
                if let Some(m) = obj.get(kw).and_then(Value::as_f64) {
                    if bad(n, m) {
                        self.errors.push(format!("{here}: {n} violates {kw} {m}"));
                    }
                }
            }
        }
        if let Value::Object(map) = instance {
            if let Some(Value::Array(req)) = obj.get("required") {
                for r in req.iter().filter_map(Value::as_str) {
                    if !map.contains_key(r) {
                        self.errors
                            .push(format!("{here}: missing required property {r:?}"));
                    }
                }
            }
            let props = obj.get("properties").and_then(Value::as_object);
            for (k, v) in map {
                let child = format!("{path}/{k}");
                match props.and_then(|p| p.get(k)) {
                    Some(sub) => self.check(sub, v, &child),
                    None => match obj.get("additionalProperties") {
                        Some(Value::Bool(false)) => {
                            self.errors
                                .push(format!("{here}: additional property {k:?} is not allowed"));
                        }
                        Some(sub @ Value::Object(_)) => self.check(sub, v, &child),
                        _ => {}
                    },
                }
            }
        }
        if let Value::Array(items) = instance {
            if let Some(sub) = obj.get("items") {
                for (i, v) in items.iter().enumerate() {
                    self.check(sub, v, &format!("{path}/{i}"));
                }
            }
            if obj.get("uniqueItems") == Some(&Value::Bool(true)) {
                for i in 0..items.len() {
                    if items[i + 1..].contains(&items[i]) {
                        self.errors
                            .push(format!("{here}: items are not unique ({})", items[i]));
                        break;
                    }
                }
            }
            if let Some(min) = obj.get("minItems").and_then(Value::as_u64) {
                if (items.len() as u64) < min {
                    self.errors
                        .push(format!("{here}: fewer than minItems {min}"));
                }
            }
            if let Some(max) = obj.get("maxItems").and_then(Value::as_u64) {
                if (items.len() as u64) > max {
                    self.errors
                        .push(format!("{here}: more than maxItems {max}"));
                }
            }
        }
        if let Some(Value::Array(all)) = obj.get("allOf") {
            for sub in all {
                self.check(sub, instance, path);
            }
        }
        if let Some(Value::Array(any)) = obj.get("anyOf") {
            if !any.iter().any(|sub| self.is_valid(sub, instance, path)) {
                self.errors.push(format!("{here}: matches none of anyOf"));
            }
        }
        if let Some(Value::Array(one)) = obj.get("oneOf") {
            let n = one
                .iter()
                .filter(|sub| self.is_valid(sub, instance, path))
                .count();
            if n != 1 {
                self.errors
                    .push(format!("{here}: matches {n} of oneOf, expected exactly 1"));
            }
        }
        if let Some(sub) = obj.get("not") {
            if self.is_valid(sub, instance, path) {
                self.errors.push(format!("{here}: matches the not schema"));
            }
        }
        if let Some(cond) = obj.get("if") {
            let branch = if self.is_valid(cond, instance, path) {
                obj.get("then")
            } else {
                obj.get("else")
            };
            if let Some(b) = branch {
                self.check(b, instance, path);
            }
        }
    }
}

fn kind(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

#[cfg(test)]
mod tests {
    use super::validate;
    use serde_json::json;

    #[test]
    fn ecma_end_of_input_lookahead_validates_exactly() {
        // The `DecimalU64` wire pattern ends `$(?![\s\S])`; the regex crate lacks lookahead.
        let schema =
            serde_json::json!({"type": "string", "pattern": "^(?:0|[1-9][0-9]*)$(?![\\s\\S])"});
        assert!(validate(&schema, &serde_json::json!("123")).is_empty());
        assert!(!validate(&schema, &serde_json::json!("123\n")).is_empty());
        assert!(!validate(&schema, &serde_json::json!("0123")).is_empty());
    }

    #[test]
    fn basic_keywords() {
        let schema = json!({
            "type": "object",
            "required": ["a"],
            "properties": {
                "a": {"type": "string", "pattern": "^[0-9a-f]{4}$"},
                "b": {"type": ["integer", "null"], "minimum": 0},
                "c": {"enum": [false, null]},
                "d": {"$ref": "#/$defs/d"}
            },
            "additionalProperties": false,
            "if": {"properties": {"a": {"const": "ffff"}}},
            "then": {"required": ["b"]},
            "$defs": {"d": {"type": "array", "items": {"type": "string"}, "uniqueItems": true}}
        });
        assert!(validate(
            &schema,
            &json!({"a": "0abc", "b": 3, "c": false, "d": ["x"]})
        )
        .is_empty());
        let errs = validate(
            &schema,
            &json!({"a": "zzzz", "b": -1, "c": true, "d": ["x", "x"], "e": 1}),
        );
        assert_eq!(errs.len(), 5, "{errs:?}");
        let errs = validate(&schema, &json!({"a": "ffff"}));
        assert_eq!(errs, vec!["/: missing required property \"b\""]);
        let errs = validate(&json!({"contains": {}}), &json!([]));
        assert!(errs[0].contains("unsupported schema keyword"));
    }

    #[test]
    fn integer_accepts_whole_floats_only() {
        assert!(validate(&json!({"type": "integer"}), &json!(3.0)).is_empty());
        assert!(!validate(&json!({"type": "integer"}), &json!(3.5)).is_empty());
    }
}
