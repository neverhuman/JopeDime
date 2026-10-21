//! Streaming sha256 with CSV row/column counting in the same pass.

use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock};

const BUFFER_BYTES: usize = 1 << 20;
const FIRST_LINE_BYTES: usize = 64 * 1024;

/// Content identity for binary blobs; no CSV scanning or header allocation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContentDigest {
    pub sha256: String,
    pub bytes: u64,
}

fn content_hasher(path: &Path) -> io::Result<(Sha256, u64)> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0; BUFFER_BYTES];
    let mut bytes = 0;
    loop {
        let n = match file.read(&mut buffer) {
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            result => result?,
        };
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
        bytes += n as u64;
    }
    Ok((hasher, bytes))
}

pub fn digest_content(path: &Path) -> io::Result<ContentDigest> {
    let (hasher, bytes) = content_hasher(path)?;
    Ok(ContentDigest {
        sha256: hex::encode(hasher.finalize()),
        bytes,
    })
}

// A process-wide bound also applies when callers use different Tokio runtimes.
// Hold the permit in the admitted read worker: cancelling its future cannot stop a file
// read, and must not admit a replacement hash before that read has finished.
static HASH_SLOTS: LazyLock<Arc<tokio::sync::Semaphore>> = LazyLock::new(|| {
    let workers = std::thread::available_parallelism()
        .map_or(1, usize::from)
        .min(4);
    Arc::new(tokio::sync::Semaphore::new(workers))
});

/// Hash on an admitted dedicated read worker. At most four hashes run at
/// once (fewer under a smaller CPU quota), each with one 1 MiB read buffer.
pub async fn digest_content_async(path: PathBuf) -> io::Result<ContentDigest> {
    on_hash_worker(Arc::clone(&HASH_SLOTS), move || digest_content(&path)).await
}

/// Resume from the bytes actually read, retaining SHA-256 state without a
/// second pass. The destination lock is held by the caller throughout.
pub(crate) async fn content_hasher_async(path: PathBuf) -> io::Result<(Sha256, u64)> {
    on_hash_worker(Arc::clone(&HASH_SLOTS), move || content_hasher(&path)).await
}

async fn on_hash_worker<T: Send + 'static>(
    slots: Arc<tokio::sync::Semaphore>,
    work: impl FnOnce() -> io::Result<T> + Send + 'static,
) -> io::Result<T> {
    // Keep admitted reads off the shared blocking pool: destination-lock waiters
    // can occupy every blocking worker while their lock holder awaits this hash.
    let _runtime = tokio::runtime::Handle::try_current().map_err(io::Error::other)?;
    let permit = slots.acquire_owned().await.map_err(io::Error::other)?;
    let (send, receive) = tokio::sync::oneshot::channel();
    std::thread::Builder::new()
        .name("jope-blob-hash".into())
        .spawn(move || {
            // Cancelling delivery cannot cancel the actual read or free its slot.
            let result = work();
            // Publish completion only after the read has relinquished admission.
            drop(permit);
            let _ = send.send(result);
        })?;
    receive.await.map_err(io::Error::other)?
}

/// What one pass over a file yields.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileDigest {
    pub sha256: String,
    pub bytes: u64,
    /// Number of lines (a final line without `\n` counts).
    pub rows: u64,
    /// Comma-separated fields on the first line (0 for an empty file).
    pub columns: u64,
    /// The first line, capped at 64 KiB (columns are still counted over the whole line), for
    /// the headerless check.
    pub first_line: String,
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

/// Hash a file in 1 MiB chunks, counting CSV rows and first-line columns as it goes.
pub fn digest_file(path: &Path) -> io::Result<FileDigest> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; BUFFER_BYTES];
    let mut bytes = 0u64;
    let mut newlines = 0u64;
    let mut last = 0u8;
    let mut first_line: Vec<u8> = Vec::new();
    let mut first_commas = 0u64;
    let mut first_done = false;
    loop {
        let n = match file.read(&mut buf) {
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            result => result?,
        };
        if n == 0 {
            break;
        }
        let chunk = &buf[..n];
        hasher.update(chunk);
        bytes += n as u64;
        newlines += chunk.iter().filter(|&&b| b == b'\n').count() as u64;
        last = chunk[n - 1];
        if !first_done {
            let head = match chunk.iter().position(|&b| b == b'\n') {
                Some(i) => {
                    first_done = true;
                    &chunk[..i]
                }
                None => chunk,
            };
            first_commas += head.iter().filter(|&&b| b == b',').count() as u64;
            let take = head.len().min(FIRST_LINE_BYTES - first_line.len());
            first_line.extend_from_slice(&head[..take]);
        }
    }
    if first_line.last() == Some(&b'\r') {
        first_line.pop();
    }
    let rows = newlines + u64::from(bytes > 0 && last != b'\n');
    let columns = if bytes == 0 { 0 } else { first_commas + 1 };
    Ok(FileDigest {
        sha256: hex::encode(hasher.finalize()),
        bytes,
        rows,
        columns,
        first_line: String::from_utf8_lossy(&first_line).into_owned(),
    })
}

/// A CSV is "headerless" when every field of its first line parses as a number
/// (empty fields and NaN/inf spellings count as numeric).
pub fn looks_headerless(first_line: &str) -> bool {
    first_line.split(',').all(|f| {
        let f = f.trim();
        f.is_empty() || f.parse::<f64>().is_ok()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_rows_and_columns() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.csv");
        std::fs::write(&p, "1,2,3\n4,5,6\n").unwrap();
        let d = digest_file(&p).unwrap();
        assert_eq!((d.rows, d.columns, d.bytes), (2, 3, 12));
        assert_eq!(d.sha256, sha256_hex(b"1,2,3\n4,5,6\n"));
        assert_eq!(d.first_line, "1,2,3");
        std::fs::write(&p, "1,2\n4,5").unwrap();
        assert_eq!(digest_file(&p).unwrap().rows, 2);
        std::fs::write(&p, "").unwrap();
        let d = digest_file(&p).unwrap();
        assert_eq!((d.rows, d.columns), (0, 0));
    }

    #[test]
    fn counts_columns_beyond_the_first_line_cap() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("wide.csv");
        let line: Vec<String> = (0..20_000).map(|i| format!("0.{i}")).collect();
        std::fs::write(&p, format!("{}\n{}\n", line.join(","), line.join(","))).unwrap();
        let d = digest_file(&p).unwrap();
        assert_eq!((d.rows, d.columns), (2, 20_000));
        assert!(d.first_line.len() <= 64 * 1024);
    }

    #[test]
    fn headerless_detection() {
        assert!(looks_headerless("0,0.5,-1e3,NaN,,inf"));
        assert!(!looks_headerless("a1,a2,target"));
    }

    #[test]
    fn binary_content_identity_matches_csv_digest_across_chunk_boundaries() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("weights.bin");
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        for size in [0, 1, BUFFER_BYTES - 1, BUFFER_BYTES, BUFFER_BYTES + 37] {
            let bytes: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
            std::fs::write(&path, &bytes).unwrap();
            let digest = digest_content(&path).unwrap();
            assert_eq!(digest.sha256, sha256_hex(&bytes));
            assert_eq!(digest.bytes, size as u64);
            let csv = digest_file(&path).unwrap();
            assert_eq!(
                (csv.sha256, csv.bytes),
                (digest.sha256.clone(), digest.bytes)
            );
            assert_eq!(
                runtime
                    .block_on(digest_content_async(path.clone()))
                    .unwrap(),
                digest
            );
        }
        let missing = dir.path().join("missing");
        assert_eq!(
            runtime
                .block_on(digest_content_async(missing))
                .unwrap_err()
                .kind(),
            io::ErrorKind::NotFound
        );
    }

    #[test]
    fn wide_first_line_keeps_all_column_counts_and_only_the_header_prefix() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wide.csv");
        let mut bytes = vec![b','; BUFFER_BYTES + 17];
        bytes.extend_from_slice(b"\r\n1,2");
        std::fs::write(&path, &bytes).unwrap();
        let digest = digest_file(&path).unwrap();
        assert_eq!(digest.first_line.as_bytes(), &bytes[..FIRST_LINE_BYTES]);
        assert_eq!(digest.columns, (BUFFER_BYTES + 18) as u64);
        assert_eq!(digest.rows, 2);
        assert_eq!(digest.sha256, sha256_hex(&bytes));
    }

    #[test]
    fn cancelled_hash_keeps_its_slot_until_the_blocking_job_finishes() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap();
        runtime.block_on(async {
            let slots = Arc::new(tokio::sync::Semaphore::new(1));
            let (started, wait_started) = tokio::sync::oneshot::channel();
            let (release, wait_release) = std::sync::mpsc::sync_channel(1);
            let first_slots = Arc::clone(&slots);
            let first = tokio::spawn(on_hash_worker(first_slots, move || {
                started.send(()).unwrap();
                wait_release
                    .recv_timeout(std::time::Duration::from_secs(5))
                    .unwrap();
                Ok(())
            }));
            // Receiving this on a current-thread runtime also proves blocking
            // hashing leaves the executor free to poll another task.
            tokio::time::timeout(std::time::Duration::from_secs(2), wait_started)
                .await
                .unwrap()
                .unwrap();
            first.abort();
            assert!(first.await.unwrap_err().is_cancelled());
            assert!(
                slots.try_acquire().is_err(),
                "running cancelled work still owns the slot"
            );
            let next_slots = Arc::clone(&slots);
            let second = tokio::spawn(on_hash_worker(next_slots, || Ok(7)));
            release.send(()).unwrap();
            assert_eq!(
                tokio::time::timeout(std::time::Duration::from_secs(2), second)
                    .await
                    .unwrap()
                    .unwrap()
                    .unwrap(),
                7
            );
            assert_eq!(slots.available_permits(), 1);
        });
    }

    #[test]
    fn hashing_progresses_while_a_destination_waiter_occupies_the_blocking_pool() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cached-blob");
        let bytes = b"hashing must progress under destination lock contention";
        std::fs::write(&path, bytes).unwrap();
        let lock_path = dir.path().join("destination.partial.lock");
        let owner = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&lock_path)
            .unwrap();
        owner.lock().unwrap();
        let waiter = std::fs::OpenOptions::new()
            .write(true)
            .open(lock_path)
            .unwrap();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .max_blocking_threads(1)
            .enable_time()
            .build()
            .unwrap();
        let (started, wait_started) = tokio::sync::oneshot::channel();
        let waiting = runtime.spawn_blocking(move || {
            started.send(()).unwrap();
            waiter.lock().map(|()| waiter)
        });
        let result = runtime.block_on(async {
            wait_started.await.unwrap();
            tokio::time::timeout(
                std::time::Duration::from_secs(2),
                digest_content_async(path),
            )
            .await
        });
        // Release before asserting: the old circular wait becomes a bounded failure,
        // rather than hanging runtime destruction on a still-held filesystem lock.
        drop(owner);
        drop(runtime.block_on(waiting).unwrap().unwrap());
        let digest = result
            .expect("hashing must progress independently of lock waiters")
            .unwrap();
        assert_eq!(digest.sha256, sha256_hex(bytes));
        assert_eq!(digest.bytes, bytes.len() as u64);
    }

    #[test]
    fn admitted_read_keeps_its_slot_across_runtime_shutdown() {
        let slots = Arc::new(tokio::sync::Semaphore::new(1));
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap();
        let (started, wait_started) = tokio::sync::oneshot::channel();
        let (release, wait_release) = std::sync::mpsc::sync_channel(1);
        let first = runtime.spawn(on_hash_worker(Arc::clone(&slots), move || {
            let _ = started.send(());
            wait_release
                .recv_timeout(std::time::Duration::from_secs(5))
                .map_err(io::Error::other)?;
            Ok(())
        }));
        runtime.block_on(async {
            tokio::time::timeout(std::time::Duration::from_secs(2), wait_started)
                .await
                .unwrap()
                .unwrap();
            first.abort();
            assert!(first.await.unwrap_err().is_cancelled());
        });
        drop(runtime);
        assert_eq!(
            slots.available_permits(),
            0,
            "the actual read retains admission"
        );
        let next_runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap();
        next_runtime.block_on(async {
            let mut next = tokio::spawn(on_hash_worker(Arc::clone(&slots), || Ok(7)));
            assert!(
                tokio::time::timeout(std::time::Duration::from_millis(50), &mut next)
                    .await
                    .is_err(),
                "replacement work must await the old read"
            );
            release.send(()).unwrap();
            assert_eq!(
                tokio::time::timeout(std::time::Duration::from_secs(2), next)
                    .await
                    .unwrap()
                    .unwrap()
                    .unwrap(),
                7
            );
        });
        assert_eq!(slots.available_permits(), 1);
    }
}
