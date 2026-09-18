//! Streaming sha256 with CSV row/column counting in the same pass.

use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

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
    let mut buf = vec![0u8; 1 << 20];
    let mut bytes = 0u64;
    let mut newlines = 0u64;
    let mut last = 0u8;
    let mut first_line: Vec<u8> = Vec::new();
    let mut first_commas = 0u64;
    let mut first_done = false;
    loop {
        let n = file.read(&mut buf)?;
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
            if first_line.len() < 64 * 1024 {
                first_line.extend_from_slice(head);
                first_line.truncate(64 * 1024);
            }
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
}
