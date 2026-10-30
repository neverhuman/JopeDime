use schemars::{JsonSchema, Schema, SchemaGenerator};
use serde::{de, Deserialize, Deserializer, Serialize, Serializer};
use std::borrow::Cow;

/// A `u64` represented on the wire as a canonical decimal string.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct DecimalU64(pub u64);

impl JsonSchema for DecimalU64 {
    fn inline_schema() -> bool {
        true
    }

    fn schema_name() -> Cow<'static, str> {
        String::schema_name()
    }

    fn json_schema(generator: &mut SchemaGenerator) -> Schema {
        String::json_schema(generator)
    }
}

impl Serialize for DecimalU64 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.0.to_string())
    }
}

impl<'de> Deserialize<'de> for DecimalU64 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        let bytes = value.as_bytes();
        let canonical = value == "0"
            || matches!(bytes.first(), Some(b'1'..=b'9'))
                && bytes[1..].iter().all(u8::is_ascii_digit);
        if !canonical {
            return Err(de::Error::custom(
                "expected a canonical unsigned decimal string",
            ));
        }

        value
            .parse::<u64>()
            .map(Self)
            .map_err(|_| de::Error::custom("decimal string is outside the u64 range"))
    }
}

#[cfg(test)]
mod tests {
    use super::DecimalU64;

    #[test]
    fn serializes_and_deserializes_boundary_values_as_strings() {
        for value in [0, 9_007_199_254_740_993, u64::MAX] {
            let encoded = serde_json::to_string(&DecimalU64(value)).unwrap();
            assert_eq!(encoded, format!("\"{value}\""));
            assert_eq!(
                serde_json::from_str::<DecimalU64>(&encoded).unwrap(),
                DecimalU64(value)
            );
        }
    }

    #[test]
    fn rejects_noncanonical_decimal_strings() {
        for value in ["\"01\"", "\"-1\"", "\"1e3\"", "\"+1\"", "\"\""] {
            assert!(
                serde_json::from_str::<DecimalU64>(value).is_err(),
                "accepted {value}"
            );
        }
    }

    #[test]
    fn rejects_json_numbers() {
        assert!(serde_json::from_str::<DecimalU64>("0").is_err());
        assert!(serde_json::from_str::<DecimalU64>("9007199254740993").is_err());
    }

    #[test]
    fn json_schema_describes_a_string() {
        let schema = serde_json::to_value(schemars::schema_for!(DecimalU64)).unwrap();
        assert_eq!(schema["type"], "string");
    }
}
