//! Bounded-memory JSON validation shared by persisted state and the local protocol.
use crate::Failure;
use anyhow::Result;
use serde::de::{DeserializeOwned, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use std::collections::BTreeSet;
use std::fmt;

struct Unique;
impl<'de> Deserialize<'de> for Unique {
    fn deserialize<D: Deserializer<'de>>(decoder: D) -> std::result::Result<Self, D::Error> {
        struct Check;
        impl<'de> Visitor<'de> for Check {
            type Value = Unique;
            fn expecting(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
                out.write_str("JSON with unique object keys")
            }
            fn visit_map<M: MapAccess<'de>>(
                self,
                mut map: M,
            ) -> std::result::Result<Unique, M::Error> {
                let mut keys = BTreeSet::new();
                while let Some(key) = map.next_key::<String>()? {
                    if !keys.insert(key) {
                        return Err(serde::de::Error::custom("duplicate JSON object key"));
                    }
                    map.next_value::<Unique>()?;
                }
                Ok(Unique)
            }
            fn visit_seq<S: SeqAccess<'de>>(
                self,
                mut seq: S,
            ) -> std::result::Result<Unique, S::Error> {
                while seq.next_element::<Unique>()?.is_some() {}
                Ok(Unique)
            }
            fn visit_bool<E: serde::de::Error>(self, _: bool) -> std::result::Result<Unique, E> {
                Ok(Unique)
            }
            fn visit_i64<E: serde::de::Error>(self, _: i64) -> std::result::Result<Unique, E> {
                Ok(Unique)
            }
            fn visit_u64<E: serde::de::Error>(self, _: u64) -> std::result::Result<Unique, E> {
                Ok(Unique)
            }
            fn visit_f64<E: serde::de::Error>(self, _: f64) -> std::result::Result<Unique, E> {
                Ok(Unique)
            }
            fn visit_str<E: serde::de::Error>(self, _: &str) -> std::result::Result<Unique, E> {
                Ok(Unique)
            }
            fn visit_unit<E: serde::de::Error>(self) -> std::result::Result<Unique, E> {
                Ok(Unique)
            }
        }
        decoder.deserialize_any(Check)
    }
}

/// Refuse duplicate keys even in arbitrary extension metadata. Does not retain another state tree.
pub fn validate(bytes: &[u8]) -> Result<()> {
    serde_json::from_slice::<Unique>(bytes)
        .map(|_| ())
        .map_err(|e| Failure::integrity(format!("JSON: {e}")))
}

/// A known decoder must consume every field; an older reader cannot silently drop new state.
pub fn decode<T: DeserializeOwned>(bytes: &[u8]) -> Result<T> {
    validate(bytes)?;
    let mut decoder = serde_json::Deserializer::from_slice(bytes);
    let mut ignored = None;
    let value = serde_ignored::deserialize(&mut decoder, |path| {
        if ignored.is_none() {
            ignored = Some(path.to_string());
        }
    })
    .map_err(|e| Failure::integrity(format!("state JSON: {e}")))?;
    decoder
        .end()
        .map_err(|e| Failure::integrity(format!("state JSON: {e}")))?;
    if let Some(path) = ignored {
        return Err(Failure::integrity(format!(
            "unsupported state field: {path}"
        )));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn escaped_duplicate_keys_nested_arrays_and_trailing_values_are_refused() {
        for bytes in [
            br#"{"a":1,"\u0061":2}"#.as_slice(),
            br#"[{"metadata":{"a":1,"a":2}}]"#,
            b"{} {}",
        ] {
            assert!(validate(bytes).is_err());
        }
        validate(br#"{"different":{"a":1},"another":{"a":2},"array":[1,true,null]}"#).unwrap();
    }
}
