//! A small ordered JSON value for records whose keys are dynamic (validation tasks, DOPE
//! submodules). `serde_json::Map` sorts keys; the Python hooks wrote keys in insertion order, and
//! `J` keeps that order so the record bodies read the same as before.

use serde::ser::{Serialize, SerializeMap, SerializeSeq, Serializer};

#[derive(Clone, Debug, PartialEq)]
pub enum J {
    Null,
    Bool(bool),
    Int(i64),
    UInt(u64),
    /// Non-finite floats serialize as `null`.
    Float(f64),
    Str(String),
    Arr(Vec<J>),
    Obj(Vec<(String, J)>),
}

impl J {
    pub fn obj<K: Into<String>>(fields: impl IntoIterator<Item = (K, J)>) -> Self {
        J::Obj(fields.into_iter().map(|(k, v)| (k.into(), v)).collect())
    }

    pub fn get(&self, key: &str) -> Option<&J> {
        match self {
            J::Obj(fields) => fields.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// Append or replace a key, keeping the first position of an existing key.
    pub fn set(&mut self, key: &str, value: J) {
        if let J::Obj(fields) = self {
            if let Some(slot) = fields.iter_mut().find(|(k, _)| k == key) {
                slot.1 = value;
            } else {
                fields.push((key.to_owned(), value));
            }
        }
    }
}

impl Serialize for J {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            J::Null => s.serialize_unit(),
            J::Bool(b) => s.serialize_bool(*b),
            J::Int(i) => s.serialize_i64(*i),
            J::UInt(u) => s.serialize_u64(*u),
            J::Float(f) if f.is_finite() => s.serialize_f64(*f),
            J::Float(_) => s.serialize_unit(),
            J::Str(v) => s.serialize_str(v),
            J::Arr(items) => {
                let mut seq = s.serialize_seq(Some(items.len()))?;
                for item in items {
                    seq.serialize_element(item)?;
                }
                seq.end()
            }
            J::Obj(fields) => {
                let mut map = s.serialize_map(Some(fields.len()))?;
                for (k, v) in fields {
                    map.serialize_entry(k, v)?;
                }
                map.end()
            }
        }
    }
}

impl From<bool> for J {
    fn from(v: bool) -> Self {
        J::Bool(v)
    }
}
impl From<f64> for J {
    fn from(v: f64) -> Self {
        J::Float(v)
    }
}
impl From<u64> for J {
    fn from(v: u64) -> Self {
        J::UInt(v)
    }
}
impl From<u32> for J {
    fn from(v: u32) -> Self {
        J::UInt(v.into())
    }
}
impl From<i64> for J {
    fn from(v: i64) -> Self {
        J::Int(v)
    }
}
impl From<usize> for J {
    fn from(v: usize) -> Self {
        J::UInt(v as u64)
    }
}
impl From<&str> for J {
    fn from(v: &str) -> Self {
        J::Str(v.to_owned())
    }
}
impl From<String> for J {
    fn from(v: String) -> Self {
        J::Str(v)
    }
}
impl<T: Into<J>> From<Option<T>> for J {
    fn from(v: Option<T>) -> Self {
        v.map_or(J::Null, Into::into)
    }
}
impl<T: Into<J>> From<Vec<T>> for J {
    fn from(v: Vec<T>) -> Self {
        J::Arr(v.into_iter().map(Into::into).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_insertion_order_and_nulls_non_finite() {
        let mut v = J::obj([("z", J::from(1u64)), ("a", J::from(f64::NAN))]);
        v.set("m", J::from("x"));
        v.set("z", J::from(2u64));
        assert_eq!(
            serde_json::to_string(&v).unwrap(),
            r#"{"z":2,"a":null,"m":"x"}"#
        );
    }
}
