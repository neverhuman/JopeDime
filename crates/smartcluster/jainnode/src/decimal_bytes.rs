use std::fmt;

use serde::{
    de::{self, Visitor},
    Deserializer, Serializer,
};

pub fn serialize<S>(value: &u64, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    if serializer.is_human_readable() {
        serializer.serialize_str(&value.to_string())
    } else {
        serializer.serialize_u64(*value)
    }
}

pub fn deserialize<'de, D>(deserializer: D) -> Result<u64, D::Error>
where
    D: Deserializer<'de>,
{
    if deserializer.is_human_readable() {
        deserializer.deserialize_str(DecimalBytesVisitor)
    } else {
        deserializer.deserialize_u64(DecimalBytesVisitor)
    }
}

struct DecimalBytesVisitor;

impl Visitor<'_> for DecimalBytesVisitor {
    type Value = u64;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("an unsigned decimal byte count without leading zeroes")
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        if value.is_empty()
            || (value.len() > 1 && value.starts_with('0'))
            || !value.bytes().all(|byte| byte.is_ascii_digit())
        {
            return Err(E::custom("invalid decimal byte count"));
        }
        value.parse().map_err(E::custom)
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(value)
    }
}
