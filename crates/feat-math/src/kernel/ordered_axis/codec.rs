//! Checked, lossless continuation of one declared ordered cut.
use super::*;
use crate::kernel::codec::{Reader, Writer};

pub(crate) fn write_group(writer: &mut Writer, group: &GroupKey) -> Result<()> {
    match group {
        GroupKey::NumericBits(bits) => {
            writer.byte(0)?;
            writer.word(*bits)
        }
        GroupKey::Categorical(value) => {
            writer.byte(1)?;
            writer.text(value)
        }
    }
}

pub(crate) fn read_group(reader: &mut Reader<'_>) -> Result<GroupKey> {
    match reader.byte()? {
        0 => {
            let bits = reader.word()?;
            let group = GroupKey::numeric(f64::from_bits(bits))?;
            if group != GroupKey::NumericBits(bits) {
                return Err(Error::Domain("noncanonical group key".into()));
            }
            Ok(group)
        }
        1 => GroupKey::categorical(reader.text()?),
        _ => Err(Error::Unsupported("ordered group codec".into())),
    }
}

impl CausalCursor {
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut writer = Writer::default();
        writer.raw(b"FMKCUR\0\x01")?;
        for metadata in [
            serde_json::to_string(&self.stage),
            serde_json::to_string(&self.input),
            serde_json::to_string(&self.fold),
        ] {
            writer.text(&metadata.map_err(|e| Error::Syntax(e.to_string()))?)?;
        }
        writer.word(self.next_ordinal as u64)?;
        writer.word(self.groups.len() as u64)?;
        for (group, progress) in &self.groups {
            write_group(&mut writer, group)?;
            writer.word(
                progress
                    .last_timestamp
                    .ok_or_else(|| Error::Domain("empty ordered group".into()))?
                    .to_bits(),
            )?;
            writer.word(progress.prior_rows as u64)?;
            writer.word(progress.prior_train_rows as u64)?;
        }
        Ok(writer.finish())
    }

    pub fn from_bytes(
        bytes: &[u8],
        stage: StageAxis,
        task: TaskMode,
        input: &OrderedInputIdentity,
        fold: FoldBoundary,
    ) -> Result<Self> {
        let mut cursor = Self::new(stage, task, input, fold)?;
        let mut reader = Reader::new(bytes)?;
        if reader.take(8)? != b"FMKCUR\0\x01" {
            return Err(Error::Unsupported("ordered cursor codec version".into()));
        }
        let saved_stage: StageAxis =
            serde_json::from_str(&reader.text()?).map_err(|e| Error::Syntax(e.to_string()))?;
        let saved_input: OrderedInputIdentity =
            serde_json::from_str(&reader.text()?).map_err(|e| Error::Syntax(e.to_string()))?;
        let saved_fold: FoldBoundary =
            serde_json::from_str(&reader.text()?).map_err(|e| Error::Syntax(e.to_string()))?;
        if saved_stage != stage || saved_input != *input || saved_fold != cursor.fold {
            return Err(Error::Domain(
                "ordered continuation identity mismatch".into(),
            ));
        }
        cursor.next_ordinal =
            usize::try_from(reader.word()?).map_err(|_| Error::Bounds("ordered ordinal".into()))?;
        if cursor.next_ordinal > cursor.fold.total_rows {
            return Err(Error::Bounds("ordered ordinal exceeds fold".into()));
        }
        let count = reader.length()?;
        if count > cursor.next_ordinal {
            return Err(Error::Bounds("ordered group count".into()));
        }
        let mut rows = 0usize;
        let mut training = 0usize;
        let mut previous = None;
        for _ in 0..count {
            let group = read_group(&mut reader)?;
            if previous.as_ref().is_some_and(|p| p >= &group) {
                return Err(Error::Domain("duplicate or unordered groups".into()));
            }
            previous = Some(group.clone());
            let timestamp = f64::from_bits(reader.word()?);
            let prior_rows = usize::try_from(reader.word()?)
                .map_err(|_| Error::Bounds("ordered rows".into()))?;
            let prior_train_rows = usize::try_from(reader.word()?)
                .map_err(|_| Error::Bounds("ordered train rows".into()))?;
            if !timestamp.is_finite()
                || prior_rows == 0
                || prior_rows > cursor.next_ordinal
                || prior_train_rows > prior_rows
            {
                return Err(Error::Domain("invalid ordered group progress".into()));
            }
            rows = rows
                .checked_add(prior_rows)
                .ok_or_else(|| Error::Bounds("ordered row sum".into()))?;
            training = training
                .checked_add(prior_train_rows)
                .ok_or_else(|| Error::Bounds("ordered train sum".into()))?;
            cursor.groups.insert(
                group,
                GroupProgress {
                    last_timestamp: Some(timestamp),
                    prior_rows,
                    prior_train_rows,
                },
            );
        }
        reader.finish()?;
        if rows != cursor.next_ordinal || training != rows.min(cursor.fold.train_end) {
            return Err(Error::Domain("ordered progress totals".into()));
        }
        // Require the canonical encoder, including metadata field order.
        if cursor.to_bytes()? != bytes {
            return Err(Error::Syntax("noncanonical ordered cursor".into()));
        }
        Ok(cursor)
    }

    pub(crate) fn progress_counts(&self) -> impl Iterator<Item = (&GroupKey, usize)> {
        self.groups
            .iter()
            .map(|(group, progress)| (group, progress.prior_rows))
    }

    pub(crate) fn ordinal(&self) -> usize {
        self.next_ordinal
    }
}

impl CausalCursor {
    pub(crate) fn progress_timestamps(&self) -> impl Iterator<Item = (&GroupKey, usize, f64)> {
        self.groups.iter().map(|(g, p)| {
            (
                g,
                p.prior_rows,
                p.last_timestamp.expect("ordered group invariant"),
            )
        })
    }
}
