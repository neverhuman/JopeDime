use super::*;

pub(super) fn segment_path(directory: &Path, start: u64) -> PathBuf {
    directory.join(format!("{start:020}.{SEGMENT_EXTENSION}"))
}

pub(super) fn segment_start_from_path(path: &Path) -> Option<u64> {
    if path.extension()?.to_str()? != SEGMENT_EXTENSION {
        return None;
    }
    path.file_stem()?.to_str()?.parse().ok()
}

pub(super) fn segment_paths(directory: &Path) -> Result<Vec<PathBuf>, EventLogError> {
    let entries = fs::read_dir(directory).map_err(|error| EventLogError::io(directory, error))?;
    let mut paths = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|error| EventLogError::io(directory, error))?;
        let path = entry.path();
        if segment_start_from_path(&path).is_some() {
            paths.push(path);
        }
    }
    paths.sort_by_key(|path| segment_start_from_path(path).expect("filtered segment path"));
    Ok(paths)
}

pub(super) fn create_segment(directory: &Path, start: u64) -> Result<PathBuf, EventLogError> {
    let path = segment_path(directory, start);
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .map_err(|error| EventLogError::io(&path, error))?;
    let header = segment_header(start);
    file.write_all(&header)
        .and_then(|()| file.sync_all())
        .map_err(|error| EventLogError::io(&path, error))?;
    sync_directory(directory)?;
    Ok(path)
}

pub(super) fn segment_header(start: u64) -> [u8; HEADER_LEN as usize] {
    let mut header = [0_u8; HEADER_LEN as usize];
    header[..8].copy_from_slice(MAGIC);
    header[8..10].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
    header[12..20].copy_from_slice(&start.to_le_bytes());
    let checksum = crc32fast::hash(&header[..20]);
    header[20..24].copy_from_slice(&checksum.to_le_bytes());
    header
}

pub(super) fn valid_header(header: &[u8; HEADER_LEN as usize], expected_start: u64) -> bool {
    header[..8] == MAGIC[..]
        && u16::from_le_bytes([header[8], header[9]]) == FORMAT_VERSION
        && header[10..12] == [0, 0]
        && u64::from_le_bytes(header[12..20].try_into().expect("fixed slice")) == expected_start
        && u32::from_le_bytes(header[20..24].try_into().expect("fixed slice"))
            == crc32fast::hash(&header[..20])
}

pub(super) enum RecoveredSegment {
    Usable {
        end: u64,
        bytes: u64,
        truncated_bytes: u64,
    },
    InvalidHeader {
        removed_bytes: u64,
    },
}

pub(super) fn recover_segment(
    path: &Path,
    start: u64,
    max_event_bytes: usize,
) -> Result<RecoveredSegment, EventLogError> {
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .map_err(|error| EventLogError::io(path, error))?;
    let original_len = file
        .metadata()
        .map_err(|error| EventLogError::io(path, error))?
        .len();
    let mut header = [0_u8; HEADER_LEN as usize];
    if file.read_exact(&mut header).is_err() || !valid_header(&header, start) {
        return Ok(RecoveredSegment::InvalidHeader {
            removed_bytes: original_len,
        });
    }

    let mut expected_offset = start;
    let mut valid_bytes = HEADER_LEN;
    loop {
        let mut prefix = [0_u8; RECORD_PREFIX_LEN as usize];
        match read_prefix(&mut file, &mut prefix).map_err(|error| EventLogError::io(path, error))? {
            PrefixRead::End => break,
            PrefixRead::Partial => break,
            PrefixRead::Full => {}
        }
        let payload_len = u32::from_le_bytes(prefix[..4].try_into().expect("fixed slice"));
        let offset = u64::from_le_bytes(prefix[4..12].try_into().expect("fixed slice"));
        let expected_checksum = u32::from_le_bytes(prefix[12..16].try_into().expect("fixed slice"));
        if usize::try_from(payload_len).unwrap_or(usize::MAX) > max_event_bytes
            || offset != expected_offset
            || offset == u64::MAX
        {
            break;
        }
        let mut payload = vec![0_u8; payload_len as usize];
        if file.read_exact(&mut payload).is_err()
            || record_checksum(offset, &payload) != expected_checksum
        {
            break;
        }
        expected_offset += 1;
        valid_bytes = valid_bytes
            .saturating_add(RECORD_PREFIX_LEN)
            .saturating_add(u64::from(payload_len));
    }

    let truncated_bytes = original_len.saturating_sub(valid_bytes);
    if truncated_bytes > 0 {
        file.set_len(valid_bytes)
            .and_then(|()| file.sync_all())
            .map_err(|error| EventLogError::io(path, error))?;
    }
    Ok(RecoveredSegment::Usable {
        end: expected_offset,
        bytes: valid_bytes,
        truncated_bytes,
    })
}

pub(super) enum PrefixRead {
    End,
    Partial,
    Full,
}

pub(super) fn read_prefix(file: &mut File, prefix: &mut [u8]) -> std::io::Result<PrefixRead> {
    let mut read = 0;
    while read < prefix.len() {
        match file.read(&mut prefix[read..])? {
            0 if read == 0 => return Ok(PrefixRead::End),
            0 => return Ok(PrefixRead::Partial),
            count => read += count,
        }
    }
    Ok(PrefixRead::Full)
}

pub(super) fn record_checksum(offset: u64, payload: &[u8]) -> u32 {
    let mut hasher = Hasher::new();
    hasher.update(&offset.to_le_bytes());
    hasher.update(payload);
    hasher.finalize()
}

pub(super) fn read_segment<T: DeserializeOwned>(
    segment: &Segment,
    cursor: &mut u64,
    limit: usize,
    items: &mut Vec<LogItem<T>>,
) -> Result<(), EventLogError> {
    let mut file =
        File::open(&segment.path).map_err(|error| EventLogError::io(&segment.path, error))?;
    file.seek(SeekFrom::Start(HEADER_LEN))
        .map_err(|error| EventLogError::io(&segment.path, error))?;
    loop {
        let mut prefix = [0_u8; RECORD_PREFIX_LEN as usize];
        if !matches!(
            read_prefix(&mut file, &mut prefix)
                .map_err(|error| EventLogError::io(&segment.path, error))?,
            PrefixRead::Full
        ) {
            break;
        }
        let payload_len = u32::from_le_bytes(prefix[..4].try_into().expect("fixed slice"));
        let event_offset = u64::from_le_bytes(prefix[4..12].try_into().expect("fixed slice"));
        let mut payload = vec![0_u8; payload_len as usize];
        file.read_exact(&mut payload)
            .map_err(|error| EventLogError::io(&segment.path, error))?;
        if event_offset < *cursor {
            continue;
        }
        if event_offset > *cursor {
            items.push(LogItem::Gap(StreamGap {
                first_missing: *cursor,
                next_available: event_offset,
            }));
            *cursor = event_offset;
            if items.len() == limit {
                break;
            }
        }
        let event = from_bytes(&payload).map_err(|source| EventLogError::Decode {
            offset: event_offset,
            source,
        })?;
        items.push(LogItem::Event(LoggedEvent {
            offset: event_offset,
            event,
        }));
        *cursor = event_offset.saturating_add(1);
        if items.len() == limit {
            break;
        }
    }
    Ok(())
}

pub(super) fn remove_oldest_segment(
    directory: &Path,
    segments: &mut Vec<Segment>,
) -> Result<(), EventLogError> {
    let oldest = segments.remove(0);
    fs::remove_file(&oldest.path).map_err(|error| EventLogError::io(&oldest.path, error))?;
    sync_directory(directory)
}

pub(super) fn sync_directory(directory: &Path) -> Result<(), EventLogError> {
    let file = File::open(directory).map_err(|error| EventLogError::io(directory, error))?;
    file.sync_all()
        .map_err(|error| EventLogError::io(directory, error))
}
