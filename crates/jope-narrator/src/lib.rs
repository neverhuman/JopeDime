//! Evidence-grounded search commentary. This crate observes owned facts; it cannot call a
//! producer, fit a model, draw search randomness, or execute emitted source.
pub mod facts;
pub mod journal;
pub mod program;
pub mod render;
pub mod tap;

pub use facts::{LineageFact, ProgramRef, Survivor, FACT_SCHEMA};
pub use journal::{
    commit_path, journal_from_env, read_commit, read_journal, spawn_journal, spawn_journal_with,
    Cursor, Envelope, FactJournal, JournalSummary, ReadPage, DIR_ENV, SCHEMA,
};
pub use program::{
    diff_programs, AtomicEdit, EditSite, ExactValue, ProgramDelta, ProgramSnapshot, StageView,
};
pub use render::{Claim, Metric, Narration};
pub use tap::{
    fact_channel, EmitStatus, FactReceiver, Limits, Links, Tap, DEFAULT_MAX_BYTES,
    DEFAULT_MAX_FACTS, MAX_FRAME_BYTES, MIN_FRAME_BYTES,
};

pub(crate) fn digest(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(bytes))
}

/// Serialize without ever retaining a payload larger than the admitted frame bound.
pub(crate) fn encode_bounded(
    value: &impl serde::Serialize,
    limit: usize,
) -> anyhow::Result<Vec<u8>> {
    struct Bounded {
        bytes: Vec<u8>,
        limit: usize,
    }
    impl std::io::Write for Bounded {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if bytes.len() > self.limit.saturating_sub(self.bytes.len()) {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "frame exceeds configured bound",
                ));
            }
            self.bytes.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut writer = Bounded {
        bytes: Vec::new(),
        limit,
    };
    serde_json::to_writer(&mut writer, value)?;
    Ok(writer.bytes.into_boxed_slice().into_vec())
}

pub(crate) fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

pub(crate) fn decimal_u64(value: &str) -> anyhow::Result<u64> {
    anyhow::ensure!(
        !value.is_empty() && value.bytes().all(|c| c.is_ascii_digit()),
        "invalid decimal sequence"
    );
    let number: u64 = value.parse()?;
    anyhow::ensure!(number.to_string() == value, "noncanonical decimal sequence");
    Ok(number)
}

pub(crate) mod decimal {
    use serde::{Deserialize, Deserializer, Serializer};
    pub fn serialize<S: Serializer>(value: &u64, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&value.to_string())
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(de: D) -> Result<u64, D::Error> {
        let text = String::deserialize(de)?;
        super::decimal_u64(&text).map_err(serde::de::Error::custom)
    }
}

pub(crate) mod decimal_option {
    use serde::{Deserialize, Deserializer, Serializer};
    pub fn serialize<S: Serializer>(value: &Option<u64>, serializer: S) -> Result<S::Ok, S::Error> {
        match value {
            Some(v) => serializer.serialize_some(&v.to_string()),
            None => serializer.serialize_none(),
        }
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(de: D) -> Result<Option<u64>, D::Error> {
        Option::<String>::deserialize(de)?
            .map(|v| super::decimal_u64(&v).map_err(serde::de::Error::custom))
            .transpose()
    }
}

#[cfg(feature = "testing")]
pub use tap::RecordingTap;
