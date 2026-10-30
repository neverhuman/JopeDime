//! State includes the cursor identity and raw history words, never decimal floats.
use super::*;
use crate::kernel::codec::{Reader, Writer};
use crate::kernel::ordered_axis::codec::{read_group, write_group};

impl DifferenceState {
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut writer = Writer::default();
        writer.raw(b"FMKDIFF\x02")?;
        writer.byte(self.config.order)?;
        writer.word(self.config.step.to_bits())?;
        writer.word(self.width as u64)?;
        let cursor = self.cursor.to_bytes()?;
        writer.word(cursor.len() as u64)?;
        writer.raw(&cursor)?;
        writer.word(self.history.len() as u64)?;
        for (group, rows) in &self.history {
            write_group(&mut writer, group)?;
            writer.word(rows.len() as u64)?;
            for row in rows {
                for value in row {
                    writer.word(value.to_bits())?;
                }
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
        if reader.take(8)? != b"FMKDIFF\x02" {
            return Err(Error::Unsupported("difference state codec version".into()));
        }
        let order = reader.byte()?;
        let config = DifferenceConfig::new(2, 59, order, f64::from_bits(reader.word()?))?;
        let width = usize::try_from(reader.word()?)
            .map_err(|_| Error::Bounds("difference width".into()))?;
        validate_width(width)?;
        let n = reader.length()?;
        let stage = StageAxis::new(Axis::SampleTime, Causality::Causal, FitScope::Stateless);
        let cursor = CausalCursor::from_bytes(reader.take(n)?, stage, task, input, fold)?;
        let count = reader.length()?;
        if count != cursor.progress_counts().count() {
            return Err(Error::Domain("difference history group count".into()));
        }
        let mut history = BTreeMap::new();
        for (expected_group, prior_rows) in cursor.progress_counts() {
            let group = read_group(&mut reader)?;
            let rows = reader.length()?;
            if &group != expected_group || rows != prior_rows.min(usize::from(order)) {
                return Err(Error::Domain("difference history progress".into()));
            }
            let mut values = VecDeque::new();
            for _ in 0..rows {
                let mut row = Vec::with_capacity(width);
                for _ in 0..width {
                    row.push(f64::from_bits(reader.word()?));
                }
                values.push_back(row);
            }
            history.insert(group, values);
        }
        reader.finish()?;
        let state = Self {
            config,
            width,
            cursor,
            history,
        };
        if state.to_bytes()? != bytes {
            return Err(Error::Syntax("noncanonical difference state".into()));
        }
        Ok(state)
    }
}
