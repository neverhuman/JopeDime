//! Validated identifiers used by the JopeDime v1 API.

use std::{fmt, str::FromStr};

use serde::{de, Deserialize, Deserializer, Serialize, Serializer};
use ulid::Ulid;

/// An identifier that does not match the corresponding v1 wire format.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ParseIdError;

impl fmt::Display for ParseIdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("invalid identifier")
    }
}

impl std::error::Error for ParseIdError {}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, schemars::JsonSchema)]
#[schemars(transparent)]
pub struct RunId(String);

impl RunId {
    pub fn new() -> Self {
        Self(format!("run_{}", Ulid::new()))
    }
}

impl Default for RunId {
    fn default() -> Self {
        Self::new()
    }
}

impl FromStr for RunId {
    type Err = ParseIdError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let encoded = value.strip_prefix("run_").ok_or(ParseIdError)?;
        let id = Ulid::from_string(encoded).map_err(|_| ParseIdError)?;
        if id.to_string() != encoded {
            return Err(ParseIdError);
        }
        Ok(Self(value.to_owned()))
    }
}

impl fmt::Display for RunId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, schemars::JsonSchema)]
#[schemars(transparent)]
pub struct DatasetId(String);

impl FromStr for DatasetId {
    type Err = ParseIdError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        parse_hex_id(value, "ds_")?;
        Ok(Self(value.to_owned()))
    }
}

impl fmt::Display for DatasetId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, schemars::JsonSchema)]
#[schemars(transparent)]
pub struct ArtifactId(String);

impl FromStr for ArtifactId {
    type Err = ParseIdError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        parse_hex_id(value, "art_")?;
        Ok(Self(value.to_owned()))
    }
}

impl fmt::Display for ArtifactId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, schemars::JsonSchema)]
#[schemars(transparent)]
pub struct AlgorithmId(String);

impl FromStr for AlgorithmId {
    type Err = ParseIdError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let digits = value.strip_prefix("alg_").ok_or(ParseIdError)?;
        if digits.len() < 4 || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(ParseIdError);
        }
        Ok(Self(digits.to_owned()))
    }
}

impl fmt::Display for AlgorithmId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "alg_{}", self.0)
    }
}

macro_rules! string_serde {
    ($($id:ty),+ $(,)?) => {$ (
        impl Serialize for $id {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: Serializer,
            {
                serializer.collect_str(self)
            }
        }

        impl<'de> Deserialize<'de> for $id {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                let value = String::deserialize(deserializer)?;
                value.parse().map_err(de::Error::custom)
            }
        }
    )+};
}

string_serde!(RunId, DatasetId, AlgorithmId, ArtifactId);

fn parse_hex_id(value: &str, prefix: &str) -> Result<[u8; 12], ParseIdError> {
    let digits = value.strip_prefix(prefix).ok_or(ParseIdError)?;
    if digits.len() != 24 {
        return Err(ParseIdError);
    }

    let mut bytes = [0; 12];
    for (byte, pair) in bytes.iter_mut().zip(digits.as_bytes().chunks_exact(2)) {
        let high = lowercase_hex_nibble(pair[0]).ok_or(ParseIdError)?;
        let low = lowercase_hex_nibble(pair[1]).ok_or(ParseIdError)?;
        *byte = high << 4 | low;
    }
    Ok(bytes)
}

fn lowercase_hex_nibble(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::{AlgorithmId, ArtifactId, DatasetId, RunId};

    #[test]
    fn identifiers_round_trip_and_serialize_as_strings() {
        let run = RunId::new();
        let dataset: DatasetId = "ds_0123456789abcdef01234567".parse().unwrap();
        let algorithm: AlgorithmId = "alg_0001".parse().unwrap();
        let artifact: ArtifactId = "art_0123456789abcdef01234567".parse().unwrap();

        for (encoded, decoded) in [
            (
                run.to_string(),
                serde_json::from_str::<RunId>(&serde_json::to_string(&run).unwrap())
                    .unwrap()
                    .to_string(),
            ),
            (
                dataset.to_string(),
                serde_json::from_str::<DatasetId>(&serde_json::to_string(&dataset).unwrap())
                    .unwrap()
                    .to_string(),
            ),
            (
                algorithm.to_string(),
                serde_json::from_str::<AlgorithmId>(&serde_json::to_string(&algorithm).unwrap())
                    .unwrap()
                    .to_string(),
            ),
            (
                artifact.to_string(),
                serde_json::from_str::<ArtifactId>(&serde_json::to_string(&artifact).unwrap())
                    .unwrap()
                    .to_string(),
            ),
        ] {
            assert_eq!(encoded, decoded);
            assert_eq!(encoded.parse::<String>().unwrap(), decoded);
        }
    }

    #[test]
    fn parsers_reject_bad_prefixes_lengths_and_uppercase_hex() {
        assert!("bad_0123456789abcdef01234567".parse::<DatasetId>().is_err());
        assert!("ds_0123456789abcdef0123456".parse::<DatasetId>().is_err());
        assert!("ds_0123456789ABCDEF01234567".parse::<DatasetId>().is_err());
        assert!("ds_💥💥💥💥💥💥".parse::<DatasetId>().is_err());
        assert!("ds_0123456789abcdef01234567".parse::<ArtifactId>().is_err());
        assert!("art_0123456789abcdef0123456".parse::<ArtifactId>().is_err());
        assert!("art_0123456789abcdef0123456A"
            .parse::<ArtifactId>()
            .is_err());
        assert!("bad_01ARZ3NDEKTSV4RRFFQ69G5FAV".parse::<RunId>().is_err());
        assert!("run_01ARZ3NDEKTSV4RRFFQ69G5FA".parse::<RunId>().is_err());
        assert!("bad_0001".parse::<AlgorithmId>().is_err());
        assert!("alg_123".parse::<AlgorithmId>().is_err());
        assert!("alg_12a4".parse::<AlgorithmId>().is_err());
    }

    proptest! {
        #[test]
        fn dataset_id_round_trip(bytes in any::<[u8; 12]>()) {
            let encoded = format!("ds_{}", bytes.iter().map(|byte| format!("{byte:02x}")).collect::<String>());
            prop_assert_eq!(encoded.parse::<DatasetId>().unwrap().to_string(), encoded);
        }

        #[test]
        fn artifact_id_round_trip(bytes in any::<[u8; 12]>()) {
            let encoded = format!("art_{}", bytes.iter().map(|byte| format!("{byte:02x}")).collect::<String>());
            prop_assert_eq!(encoded.parse::<ArtifactId>().unwrap().to_string(), encoded);
        }

        #[test]
        fn run_id_round_trip(value in any::<u128>()) {
            let encoded = format!("run_{}", ulid::Ulid::from(value));
            prop_assert_eq!(encoded.parse::<RunId>().unwrap().to_string(), encoded);
        }

        #[test]
        fn algorithm_id_round_trip(digits in "[0-9]{4,30}") {
            let encoded = format!("alg_{digits}");
            prop_assert_eq!(encoded.parse::<AlgorithmId>().unwrap().to_string(), encoded);
        }
    }
}
