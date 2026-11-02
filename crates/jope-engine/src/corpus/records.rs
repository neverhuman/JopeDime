use super::*;

#[derive(Clone, Serialize, Deserialize)]
pub struct ContextRow {
    pub dataset: String,
    pub family: String,
    pub cut: usize,
    pub partition: String,
    pub path: PathBuf,
    pub sha256: String,
    pub prepared: PathBuf,
    pub prepared_sha256: String,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Group {
    pub context: usize,
    pub family: String,
    pub partition: String,
    pub records: Vec<usize>,
    pub max_length: usize,
}
#[derive(Serialize, Deserialize)]
pub struct Index {
    pub schema: String,
    pub qualification: bool,
    pub vocabulary_sha256: String,
    pub context_index_sha256: String,
    pub records_sha256: String,
    pub record_bytes: usize,
    pub records: usize,
    pub legal_sets: Vec<Vec<usize>>,
    pub contexts: Vec<ContextRow>,
    pub groups: Vec<Group>,
    /// W1 round 8: digest of a legacy (`parallel.teaching_corpus.v1`) corpus's
    /// `labels.jsonl`, bound by the index itself, so teacher labels are
    /// authenticated by the index a caller verified. `provenance.json` is
    /// bound only by its own back-reference to this index and never supplies
    /// it. Absent in corpora written before the binding: their labels are
    /// unbound (and the serialized index is unchanged when absent).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub labels_sha256: Option<String>,
}
#[derive(Clone, Debug)]
pub struct Record {
    pub tokens: [i64; SEQUENCE],
    pub legal: [i64; SEQUENCE],
    pub length: usize,
    pub context: usize,
    pub value: f32,
    pub percentile: f32,
    pub runtime: f32,
}
impl Record {
    pub fn bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(RECORD_BYTES);
        for x in self.tokens.iter().chain(self.legal.iter()) {
            out.extend((*x as u16).to_le_bytes());
        }
        out.extend((self.length as u16).to_le_bytes());
        out.extend(0u16.to_le_bytes());
        out.extend((self.context as u32).to_le_bytes());
        for x in [self.value, self.percentile, self.runtime] {
            out.extend(x.to_le_bytes());
        }
        out
    }
    pub fn parse(b: &[u8]) -> Self {
        assert_eq!(b.len(), RECORD_BYTES);
        let mut r = Self {
            tokens: [0; SEQUENCE],
            legal: [0; SEQUENCE],
            length: u16::from_le_bytes(b[512..514].try_into().unwrap()) as usize,
            context: u32::from_le_bytes(b[516..520].try_into().unwrap()) as usize,
            value: f32::from_le_bytes(b[520..524].try_into().unwrap()),
            percentile: f32::from_le_bytes(b[524..528].try_into().unwrap()),
            runtime: f32::from_le_bytes(b[528..532].try_into().unwrap()),
        };
        for (i, x) in r.tokens.iter_mut().chain(r.legal.iter_mut()).enumerate() {
            *x = u16::from_le_bytes(b[i * 2..i * 2 + 2].try_into().unwrap()) as i64;
        }
        r
    }
}
fn record_byte_len(count: usize) -> Result<usize> {
    ensure!(count > 0, "record artifact dimensions changed");
    let bytes = count
        .checked_mul(RECORD_BYTES)
        .context("record count exceeds addressable byte capacity")?;
    ensure!(
        bytes <= isize::MAX as usize,
        "record count exceeds addressable byte capacity"
    );
    Ok(bytes)
}

/// Copy exactly the advertised record bytes and reject growth without retaining
/// or writing any trailing payload. The caller still verifies the returned hash.
fn copy_record_bytes(
    source: &mut impl std::io::Read,
    destination: &mut impl std::io::Write,
    bytes: usize,
) -> Result<[u8; 32]> {
    use sha2::{Digest, Sha256};
    let mut digest = Sha256::new();
    let mut buffer = vec![0u8; bytes.min(1024 * 1024)];
    let mut remaining = bytes;
    while remaining > 0 {
        let capacity = remaining.min(buffer.len());
        let n = source.read(&mut buffer[..capacity])?;
        ensure!(n > 0, "record artifact truncated during verification");
        digest.update(&buffer[..n]);
        destination.write_all(&buffer[..n])?;
        remaining -= n;
    }
    let mut trailing = [0u8; 1];
    ensure!(
        source.read(&mut trailing)? == 0,
        "record artifact grew during verification"
    );
    Ok(digest.finalize().into())
}

/// Read-only memory mapping of the immutable training artifact.
pub struct Records {
    mapping: memmap2::Mmap,
}
impl Records {
    pub fn open(path: &Path, count: usize) -> Result<Self> {
        let bytes = record_byte_len(count)?;
        let source = fs::File::open(path)?;
        Self::from_file(sealed_record_copy(source, bytes, None)?, count)
    }
    /// Hash the bytes actually copied into sealed shared storage. A subsequent
    /// source mutation cannot change training inputs or cause mmap SIGBUS.
    pub fn open_verified(path: &Path, count: usize, expected: &str) -> Result<Self> {
        let bytes = record_byte_len(count)?;
        if let Some(file) = crate::sealed_storage::shared(path, expected, bytes)? {
            return Self::from_file(file, count);
        }
        let source = fs::File::open(path)?;
        Self::from_file(sealed_record_copy(source, bytes, Some(expected))?, count)
    }
    fn from_file(file: fs::File, count: usize) -> Result<Self> {
        use std::os::fd::AsRawFd;
        let expected = record_byte_len(count)?;
        let len = usize::try_from(file.metadata()?.len())
            .context("record artifact exceeds addressable byte capacity")?;
        ensure!(len == expected, "record artifact dimensions changed");
        let seals =
            libc::F_SEAL_WRITE | libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL;
        // SAFETY: F_GET_SEALS reads flags from this live borrowed descriptor and takes no pointer.
        let actual_seals = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GET_SEALS) };
        ensure!(
            actual_seals >= 0 && actual_seals & seals == seals,
            "record artifact is not fully sealed"
        );
        // SAFETY: verified seals prevent writes and size changes; `len` is positive, <= isize::MAX,
        // and equals the initialized file length for this read-only mapping.
        let mapping = unsafe { memmap2::MmapOptions::new().len(len).map(&file) }
            .map_err(|_| anyhow::anyhow!("mmap failed"))?;
        Ok(Self { mapping })
    }
    pub fn get(&self, i: usize) -> Record {
        assert!(
            i < self.mapping.len() / RECORD_BYTES,
            "record index out of bounds"
        );
        let start = i * RECORD_BYTES;
        Record::parse(&self.mapping[start..start + RECORD_BYTES])
    }
}

fn sealed_record_copy(
    mut source: fs::File,
    bytes: usize,
    expected_digest: Option<&str>,
) -> Result<fs::File> {
    use std::os::fd::{AsRawFd, FromRawFd};
    let metadata = source.metadata()?;
    ensure!(
        metadata.is_file() && metadata.len() == bytes as u64,
        "record artifact dimensions changed"
    );
    // SAFETY: the static name is NUL-terminated and flags are supported integer constants.
    let fd = unsafe {
        libc::memfd_create(
            c"jope-teaching-records".as_ptr(),
            libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING,
        )
    };
    ensure!(fd >= 0, "cannot create sealed teaching storage");
    // SAFETY: memfd_create returned this fresh nonnegative descriptor; ownership is transferred
    // exactly once and File closes it on every error path.
    let mut sealed = unsafe { fs::File::from_raw_fd(fd) };
    // Copy exactly the bounded artifact and reject truncation or growth after the initial stat.
    let digest = copy_record_bytes(&mut source, &mut sealed, bytes)?;
    if let Some(expected) = expected_digest {
        ensure!(
            hex::encode(digest) == expected,
            "record content changed during verification"
        );
    }
    let seals = libc::F_SEAL_WRITE | libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL;
    // SAFETY: `sealed` owns the fresh descriptor; F_ADD_SEALS takes integer flags, not pointers.
    ensure!(
        unsafe { libc::fcntl(sealed.as_raw_fd(), libc::F_ADD_SEALS, seals) } == 0,
        "cannot seal teaching storage"
    );
    Ok(sealed)
}

#[cfg(test)]
mod record_bounds_tests {
    use super::*;
    use std::panic::{catch_unwind, AssertUnwindSafe};

    fn record(value: f32) -> Record {
        Record {
            tokens: [0; SEQUENCE],
            legal: [0; SEQUENCE],
            length: 1,
            context: 0,
            value,
            percentile: 0.5,
            runtime: 1.0,
        }
    }

    #[test]
    fn sealed_record_copy_rejects_unending_growth_with_bounded_io() {
        struct GrowingSource(usize);
        impl std::io::Read for GrowingSource {
            fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
                buffer.fill(7);
                self.0 += buffer.len();
                Ok(buffer.len())
            }
        }
        let mut source = GrowingSource(0);
        let mut copied = Vec::new();
        let error = copy_record_bytes(&mut source, &mut copied, RECORD_BYTES).unwrap_err();
        assert!(error.to_string().contains("grew during verification"));
        assert_eq!(source.0, RECORD_BYTES + 1);
        assert_eq!(copied, vec![7u8; RECORD_BYTES]);
    }

    #[test]
    fn sealed_record_copy_hashes_exact_bytes_and_refuses_truncation() {
        use sha2::{Digest, Sha256};
        let expected = record(0.125).bytes();
        let mut copied = Vec::new();
        let actual =
            copy_record_bytes(&mut expected.as_slice(), &mut copied, expected.len()).unwrap();
        let expected_digest: [u8; 32] = Sha256::digest(&expected).into();
        assert_eq!(actual, expected_digest);
        assert_eq!(copied, expected);

        let mut copied = Vec::new();
        let mut truncated = &expected[..expected.len() - 1];
        let error = copy_record_bytes(&mut truncated, &mut copied, expected.len()).unwrap_err();
        assert!(error.to_string().contains("truncated during verification"));
        assert_eq!(copied.len(), expected.len() - 1);
    }

    #[test]
    fn mapped_records_preserve_first_and_last_record_values() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&record(0.125).bytes()).unwrap();
        file.write_all(&record(0.75).bytes()).unwrap();
        let records = Records::open(file.path(), 2).unwrap();
        assert_eq!(records.get(0).value.to_bits(), 0.125f32.to_bits());
        assert_eq!(records.get(1).value.to_bits(), 0.75f32.to_bits());
        assert!(catch_unwind(AssertUnwindSafe(|| records.get(2))).is_err());
    }

    #[test]
    fn wrapped_record_index_is_refused_before_mapping_access() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&record(0.125).bytes()).unwrap();
        let records = Records::open(file.path(), 1).unwrap();
        // RECORD_BYTES is divisible by four. The old release guard wrapped this
        // index to byte offset zero; a regression safely reads record zero, then fails.
        let wrapped = 1usize << (usize::BITS - 2);
        assert!(catch_unwind(AssertUnwindSafe(|| records.get(wrapped))).is_err());
    }

    #[test]
    fn impossible_record_counts_return_errors_without_panicking() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&record(0.125).bytes()).unwrap();
        let wrapped_count = (1usize << (usize::BITS - 2)) + 1;
        for count in [
            wrapped_count,
            0,
            isize::MAX as usize / RECORD_BYTES + 1,
            usize::MAX / RECORD_BYTES + 1,
            usize::MAX,
        ] {
            assert!(Records::open(file.path(), count).is_err());
            let error = Records::open_verified(file.path(), count, "not-used")
                .err()
                .expect("invalid dimensions must fail before content verification");
            let message = error.to_string();
            assert!(if count == 0 {
                message.contains("record artifact dimensions changed")
            } else {
                message.contains("record count exceeds addressable byte capacity")
            });
        }
        assert!(Records::open(file.path(), 2).is_err());
    }
}
