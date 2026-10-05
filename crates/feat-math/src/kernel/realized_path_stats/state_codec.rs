use super::*;
use crate::kernel::codec::{Reader, Writer};
use crate::kernel::ordered_axis::codec::{read_group, write_group};
impl PathState {
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut writer = Writer::default();
        writer.raw(b"FMKPTH\0\x02")?;
        writer.word(self.config.window as u64)?;
        let cursor = self.cursor.to_bytes()?;
        writer.word(cursor.len() as u64)?;
        writer.raw(&cursor)?;
        writer.word(self.history.len() as u64)?;
        for (group, history) in &self.history {
            write_group(&mut writer, group)?;
            writer.word(history.len() as u64)?;
            for value in history {
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
        if reader.take(8)? != b"FMKPTH\0\x02" {
            return Err(Error::Unsupported("path-stats state codec version".into()));
        }
        let window = usize::try_from(reader.word()?)
            .map_err(|_| Error::Shape("path-stats window".into()))?;
        let config = Config::new(2, 64, window)?;
        let n = reader.length()?;
        let stage = StageAxis::new(Axis::SampleTime, Causality::Causal, FitScope::Stateless);
        let cursor = CausalCursor::from_bytes(reader.take(n)?, stage, task, input, fold)?;
        let count = reader.length()?;
        if count != cursor.progress_counts().count() {
            return Err(Error::Domain("path-stats group count".into()));
        }
        let mut history = BTreeMap::new();
        for (expected_group, prior_rows) in cursor.progress_counts() {
            let group = read_group(&mut reader)?;
            let length = reader.length()?;
            if &group != expected_group || length != prior_rows.min(config.window) {
                return Err(Error::Domain("path-stats group/history binding".into()));
            }
            let values = (0..length)
                .map(|_| reader.word().map(f64::from_bits))
                .collect::<Result<VecDeque<_>>>()?;
            history.insert(group, values);
        }
        reader.finish()?;
        let state = Self {
            config,
            cursor,
            history,
        };
        if state.to_bytes()? != bytes {
            return Err(Error::Syntax("noncanonical path-stats state".into()));
        }
        Ok(state)
    }
}
