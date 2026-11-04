use super::*;
use std::fmt::Write as _;

/// Python's `json.dumps(value, sort_keys=True, separators=(",", ":"))` (with `ensure_ascii`)
/// for the values these fixtures hold. They hold no floats.
pub(crate) fn py_compact(value: &Value) -> String {
    fn string(s: &str, out: &mut String) {
        out.push('"');
        for c in s.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                '\u{8}' => out.push_str("\\b"),
                '\u{c}' => out.push_str("\\f"),
                ' '..='~' => out.push(c),
                _ => {
                    for unit in c.encode_utf16(&mut [0u16; 2]) {
                        write!(out, "\\u{unit:04x}").unwrap();
                    }
                }
            }
        }
        out.push('"');
    }
    fn walk(value: &Value, out: &mut String) {
        match value {
            Value::Null => out.push_str("null"),
            Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Value::Number(n) => {
                assert!(
                    n.is_i64() || n.is_u64(),
                    "fixture values hold no floats: {n}"
                );
                out.push_str(&n.to_string());
            }
            Value::String(s) => string(s, out),
            Value::Array(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    walk(item, out);
                }
                out.push(']');
            }
            Value::Object(map) => {
                out.push('{');
                for (i, (k, v)) in map.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    string(k, out);
                    out.push(':');
                    walk(v, out);
                }
                out.push('}');
            }
        }
    }
    let mut out = String::new();
    walk(value, &mut out);
    out
}

/// Files of one scenario root, referenced relative to it or (as the Python fixture did)
/// absolutely.
pub(crate) struct World {
    pub(super) root: PathBuf,
    pub(super) absolute: bool,
}

impl World {
    fn reference(&self, rel: &str) -> String {
        if self.absolute {
            self.root.join(rel).to_str().unwrap().to_owned()
        } else {
            rel.to_owned()
        }
    }
    pub(super) fn put_bytes(&self, rel: &str, bytes: &[u8]) -> Value {
        let path = self.root.join(rel);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, bytes).unwrap();
        json!({"path": self.reference(rel), "sha256": hash(bytes)})
    }
    /// `GrowthFixture.put`: compact sorted JSON plus a newline.
    pub(crate) fn put(&self, rel: &str, value: &Value) -> Value {
        self.put_bytes(rel, format!("{}\n", py_compact(value)).as_bytes())
    }
}
