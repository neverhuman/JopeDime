use schemars::{JsonSchema, Schema, SchemaGenerator};
use serde::{de, Deserialize, Deserializer, Serialize, Serializer};
use std::borrow::Cow;

const U64_DECIMAL_PATTERN: &str = concat!(
    "^(?:0|[1-9][0-9]{0,18}|1[0-7][0-9]{18}|18[0-3][0-9]{17}|",
    "184[0-3][0-9]{16}|1844[0-5][0-9]{15}|18446[0-6][0-9]{14}|",
    "184467[0-3][0-9]{13}|1844674[0-3][0-9]{12}|184467440[0-6][0-9]{10}|",
    "1844674407[0-2][0-9]{9}|18446744073[0-6][0-9]{8}|",
    "1844674407370[0-8][0-9]{6}|18446744073709[0-4][0-9]{5}|",
    "184467440737095[0-4][0-9]{4}|18446744073709550[0-9]{3}|",
    "18446744073709551[0-5][0-9]{2}|1844674407370955160[0-9]|",
    "1844674407370955161[0-4]|18446744073709551615)$(?![\\s\\S])"
);

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
        let mut schema = String::json_schema(generator);
        let object = schema.ensure_object();
        object.insert("maxLength".into(), serde_json::Value::Number(20.into()));
        object.insert(
            "pattern".into(),
            serde_json::Value::String(U64_DECIMAL_PATTERN.into()),
        );
        schema
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
        for value in [
            "\"01\"",
            "\"-1\"",
            "\"1e3\"",
            "\"+1\"",
            "\"\"",
            "\"١\"",
            "\"1\\n\"",
            "\"18446744073709551616\"",
        ] {
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
        assert_eq!(schema["maxLength"], 20);
        assert_eq!(schema["pattern"], super::U64_DECIMAL_PATTERN);
    }

    #[test]
    fn optional_decimal_schema_keeps_nullability_and_the_value_constraint() {
        let schema = serde_json::to_value(schemars::schema_for!(Option<DecimalU64>)).unwrap();
        let types = schema["type"].as_array().unwrap();
        assert!(types.iter().any(|value| value == "null"));
        assert!(types.iter().any(|value| value == "string"));
        assert_eq!(schema["maxLength"], 20);
        assert_eq!(schema["pattern"], super::U64_DECIMAL_PATTERN);
    }

    #[test]
    fn every_optional_decimal_field_retains_nullability() {
        fn assert_optional_field(schema: &serde_json::Value, field: &str) {
            let property = &schema["properties"][field];
            let types = property["type"]
                .as_array()
                .unwrap_or_else(|| panic!("{field} is not represented as a nullable schema"));
            assert!(
                types.iter().any(|value| value == "null"),
                "{field} lost nullability"
            );
            assert!(
                types.iter().any(|value| value == "string")
                    && property["maxLength"] == 20
                    && property["pattern"] == super::U64_DECIMAL_PATTERN,
                "{field} lost the decimal constraint"
            );
        }

        let budget = serde_json::to_value(schemars::schema_for!(crate::runs::Budget)).unwrap();
        assert_optional_field(&budget, "evaluations");
        assert_optional_field(&budget, "wall_seconds");
        let counts = serde_json::to_value(schemars::schema_for!(crate::runs::RowCounts)).unwrap();
        assert_optional_field(&counts, "test");
    }

    #[test]
    fn accepts_deterministic_arithmetic_oracle_values() {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        for _ in 0..256 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let encoded = format!("\"{state}\"");
            assert_eq!(
                serde_json::from_str::<DecimalU64>(&encoded).unwrap(),
                DecimalU64(state)
            );
        }
    }
}
