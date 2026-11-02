//! Hyperion Vector Cache V2 (`.hvc2`) framing and receipt contracts.
//!
//! Files are deliberately simple and deterministic: an `HVC2` preamble, a little-endian format
//! version and JSON-header length, the canonical compact JSON header, then exact row-major
//! little-endian `f32` payload bytes. The header binds the source CSV and frozen Orig V2 profile.

mod format;
mod io;

pub use format::*;
pub use io::*;

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    fn temp_dir(label: &str) -> PathBuf {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        std::env::temp_dir().join(format!(
            "feat-core-hvc2-{label}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn metadata(split: Hvc2Split) -> Hvc2Metadata {
        Hvc2Metadata::encoded("dataset", "regression", split, "a".repeat(64), 123)
    }

    fn write_sample(root: &Path) -> (PathBuf, Hvc2Header, Vec<f32>) {
        std::fs::create_dir_all(root).unwrap();
        let values = vec![1.0, 2.0, 3.5, -4.25];
        let header = Hvc2Header::new(metadata(Hvc2Split::Train), 2, 2, &values).unwrap();
        let path = root.join(HVC2_TRAIN_FILENAME);
        write_hvc2_atomic(&path, &header, &values).unwrap();
        (path, header, values)
    }

    #[test]
    fn deterministic_round_trip_is_row_major_and_exact() {
        let left = temp_dir("det-left");
        let right = temp_dir("det-right");
        let (left_path, header, values) = write_sample(&left);
        let (right_path, _, _) = write_sample(&right);
        assert_eq!(
            std::fs::read(&left_path).unwrap(),
            std::fs::read(&right_path).unwrap()
        );
        let decoded = read_hvc2(&left_path).unwrap();
        assert_eq!(decoded.header, header);
        assert_eq!(decoded.values, values);
        let _ = std::fs::remove_dir_all(left);
        let _ = std::fs::remove_dir_all(right);
    }

    #[test]
    fn rejects_corruption_truncation_trailing_bytes_and_version_drift() {
        let root = temp_dir("corruption");
        let (path, _, _) = write_sample(&root);
        let original = std::fs::read(&path).unwrap();

        let corrupt = root.join("corrupt.hvc2");
        let mut bytes = original.clone();
        *bytes.last_mut().unwrap() ^= 1;
        std::fs::write(&corrupt, bytes).unwrap();
        assert!(read_hvc2(&corrupt)
            .unwrap_err()
            .to_string()
            .contains("SHA-256"));

        let truncated = root.join("truncated.hvc2");
        std::fs::write(&truncated, &original[..original.len() - 1]).unwrap();
        assert!(read_hvc2(&truncated)
            .unwrap_err()
            .to_string()
            .contains("truncated"));

        let trailing = root.join("trailing.hvc2");
        let mut bytes = original.clone();
        bytes.push(0);
        std::fs::write(&trailing, bytes).unwrap();
        assert!(read_hvc2(&trailing)
            .unwrap_err()
            .to_string()
            .contains("trailing"));

        let unsupported = root.join("unsupported.hvc2");
        let mut bytes = original;
        bytes[4..6].copy_from_slice(&3u16.to_le_bytes());
        std::fs::write(&unsupported, bytes).unwrap();
        assert!(read_hvc2(&unsupported)
            .unwrap_err()
            .to_string()
            .contains("unsupported"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn zero_column_not_applicable_cache_is_valid() {
        let root = temp_dir("empty");
        std::fs::create_dir_all(&root).unwrap();
        let metadata = Hvc2Metadata::no_exogenous_features(
            "target-only",
            "timeseries",
            Hvc2Split::Test,
            "b".repeat(64),
            44,
        );
        let header = Hvc2Header::new(metadata, 9, 0, &[]).unwrap();
        let path = root.join(HVC2_TEST_FILENAME);
        write_hvc2_atomic(&path, &header, &[]).unwrap();
        let decoded = read_hvc2(&path).unwrap();
        assert_eq!(decoded.header.columns, 0);
        assert_eq!(decoded.header.rows, 9);
        assert!(decoded.values.is_empty());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn rejects_non_finite_and_wrong_model_hashes() {
        assert!(Hvc2Header::new(metadata(Hvc2Split::Train), 1, 1, &[f32::NAN]).is_err());
        let mut header = Hvc2Header::new(metadata(Hvc2Split::Train), 1, 1, &[1.0]).unwrap();
        header.weights_sha256 = "c".repeat(64);
        assert!(header.validate().is_err());
    }

    #[test]
    fn atomic_write_refuses_to_replace_an_existing_cache() {
        let root = temp_dir("no-clobber");
        let (path, header, values) = write_sample(&root);
        let before = std::fs::read(&path).unwrap();
        assert!(write_hvc2_atomic(&path, &header, &values).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), before);
        let _ = std::fs::remove_dir_all(root);
    }
}
