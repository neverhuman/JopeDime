//! Lossless accumulator checkpoints bound to the declared ordered cursor.
use super::*;
use crate::kernel::codec::{Reader, Writer};
use crate::kernel::ordered_axis::codec::{read_group, write_group};

impl IntegrationState {
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut writer = Writer::default();
        writer.raw(b"FMKINT\0\x02")?;
        writer.byte(match self.config.mode {
            Mode::Rect => 0,
            Mode::Trap => 1,
        })?;
        writer.word(self.config.initial.len() as u64)?;
        for value in &self.config.initial {
            writer.word(value.to_bits())?;
        }
        let cursor = self.cursor.to_bytes()?;
        writer.word(cursor.len() as u64)?;
        writer.raw(&cursor)?;
        writer.word(self.groups.len() as u64)?;
        for (group, state) in &self.groups {
            write_group(&mut writer, group)?;
            writer.word(state.timestamp.to_bits())?;
            for value in &state.previous {
                writer.word(value.to_bits())?;
            }
            for value in &state.accumulator {
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
        if reader.take(8)? != b"FMKINT\0\x02" {
            return Err(Error::Unsupported("integral state codec version".into()));
        }
        let mode = match reader.byte()? {
            0 => Mode::Rect,
            1 => Mode::Trap,
            _ => return Err(Error::Unsupported("integral mode".into())),
        };
        let width = reader.length()?;
        if width == 0 || width > crate::MAX_WIDTH {
            return Err(Error::Shape("integral width".into()));
        }
        let initial = (0..width)
            .map(|_| reader.word().map(f64::from_bits))
            .collect::<Result<Vec<_>>>()?;
        let config = Config::new(2, 60, mode, initial)?;
        let n = reader.length()?;
        let stage = StageAxis::new(Axis::SampleTime, Causality::Causal, FitScope::Stateless);
        let cursor = CausalCursor::from_bytes(reader.take(n)?, stage, task, input, fold)?;
        let count = reader.length()?;
        if count != cursor.progress_counts().count() {
            return Err(Error::Domain("integral history group count".into()));
        }
        let mut groups = BTreeMap::new();
        // Bind each raw group timestamp to the restored cursor progression.
        for (expected_group, prior_rows, expected_timestamp) in cursor.progress_timestamps() {
            let group = read_group(&mut reader)?;
            let timestamp = f64::from_bits(reader.word()?);
            if &group != expected_group || timestamp.to_bits() != expected_timestamp.to_bits() {
                return Err(Error::Domain("integral group/time binding".into()));
            }
            let previous = (0..width)
                .map(|_| reader.word().map(f64::from_bits))
                .collect::<Result<Vec<_>>>()?;
            let accumulator = (0..width)
                .map(|_| reader.word().map(f64::from_bits))
                .collect::<Result<Vec<_>>>()?;
            if accumulator
                .iter()
                .any(|v| !v.is_finite() || v.abs() > MAX_ABS)
                || (prior_rows == 1
                    && accumulator
                        .iter()
                        .zip(&config.initial)
                        .any(|(a, b)| a.to_bits() != b.to_bits()))
            {
                return Err(Error::Domain("integral accumulator binding".into()));
            }
            groups.insert(
                group,
                GroupState {
                    timestamp,
                    previous,
                    accumulator,
                },
            );
        }
        reader.finish()?;
        let state = Self {
            config,
            cursor,
            groups,
        };
        if state.to_bytes()? != bytes {
            return Err(Error::Syntax("noncanonical integral state".into()));
        }
        Ok(state)
    }
}
