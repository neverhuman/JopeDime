use super::*;
use crate::kernel::codec::{Reader, Writer};
use crate::kernel::ordered_axis::codec::{read_group, write_group};
impl TimeGradientState {
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut writer = Writer::default();
        writer.raw(b"FMKTGR\0\x02")?;
        writer.word(self.config.width as u64)?;
        writer.word(self.config.minimum_gap.to_bits())?;
        let cursor = self.cursor.to_bytes()?;
        writer.word(cursor.len() as u64)?;
        writer.raw(&cursor)?;
        writer.word(self.groups.len() as u64)?;
        for (group, (timestamp, previous)) in &self.groups {
            write_group(&mut writer, group)?;
            writer.word(timestamp.to_bits())?;
            for value in previous {
                writer.word(value.to_bits())?;
            }
        }
        Ok(writer.finish())
    }
    pub fn from_bytes(
        bytes: &[u8],
        task: TaskMode,
        input: &OrderedInputIdentity,
        fold: FoldBoundary,
    ) -> Result<Self> {
        let mut reader = Reader::new(bytes)?;
        if reader.take(8)? != b"FMKTGR\0\x02" {
            return Err(Error::Unsupported(
                "time-gradient state codec version".into(),
            ));
        }
        let width = reader.length()?;
        let minimum_gap = f64::from_bits(reader.word()?);
        let config = Config::new(2, 75, width, minimum_gap)?;
        let n = reader.length()?;
        let stage = StageAxis::new(Axis::SampleTime, Causality::Causal, FitScope::Stateless);
        let cursor = CausalCursor::from_bytes(reader.take(n)?, stage, task, input, fold)?;
        let count = reader.length()?;
        if count != cursor.progress_counts().count() {
            return Err(Error::Domain("time-gradient group count".into()));
        }
        let mut groups = BTreeMap::new();
        for (expected_group, _, expected_timestamp) in cursor.progress_timestamps() {
            let group = read_group(&mut reader)?;
            let timestamp = f64::from_bits(reader.word()?);
            if &group != expected_group || timestamp.to_bits() != expected_timestamp.to_bits() {
                return Err(Error::Domain("time-gradient group/time binding".into()));
            }
            let previous = (0..width)
                .map(|_| reader.word().map(f64::from_bits))
                .collect::<Result<Vec<_>>>()?;
            groups.insert(group, (timestamp, previous));
        }
        reader.finish()?;
        let state = Self {
            config,
            cursor,
            groups,
        };
        if state.to_bytes()? != bytes {
            return Err(Error::Syntax("noncanonical time-gradient state".into()));
        }
        Ok(state)
    }
}
