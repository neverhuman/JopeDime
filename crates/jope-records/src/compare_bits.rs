//! `compare-bits`: compare every `pred-*.txt` in a directory as IEEE-754 `f64` bit patterns,
//! ported from `deploy/ci/emitters/compare-bits.py` (`PORT-RUST-03`).
//!
//! Used by `deploy/ci/emitters.sh` and `deploy/ci/e2e-smoke.sh`: an emitted program is only
//! equivalent to its siblings when the IEEE-754 bits agree, not merely the decimal text.
//! Prints the agreed bit stream on stdout; exits 1 naming the first file that disagrees on stderr.

use std::collections::BTreeMap;
use std::path::Path;

use crate::py;
use crate::{Error, Result};

/// Parse one prediction line as `f64` and return its little-endian IEEE-754 bytes as 16-char
/// lowercase hex (`struct.pack("<d", float(v)).hex()`).
pub fn float_le_hex(token: &str) -> Result<String> {
    let trimmed = py::strip(token);
    let value: f64 = match trimmed.to_ascii_lowercase().as_str() {
        "inf" | "+inf" | "infinity" | "+infinity" => f64::INFINITY,
        "-inf" | "-infinity" => f64::NEG_INFINITY,
        "nan" | "+nan" | "-nan" => {
            // CPython float("nan") / float("-nan") bit pattern on x86_64/aarch64 little-endian:
            let bits = if trimmed.starts_with('-') {
                0xfff8_0000_0000_0000_u64
            } else {
                0x7ff8_0000_0000_0000_u64
            };
            f64::from_bits(bits)
        }
        _ => trimmed
            .parse::<f64>()
            .map_err(|_| Error::Invalid(format!("invalid float literal: {trimmed}")))?,
    };
    Ok(value
        .to_le_bytes()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

/// Compare all `pred-*.txt` files in `dir`. Returns `(stdout, stderr, ok)`.
pub fn compare_dir(dir: &Path) -> Result<(String, String, bool)> {
    let entries = std::fs::read_dir(dir).map_err(|e| Error::io(dir, e))?;
    let mut bits: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for entry in entries {
        let entry = entry.map_err(|e| Error::io(dir, e))?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.starts_with("pred-") || !name.ends_with(".txt") {
            continue;
        }
        let text = py::read_text(&entry.path())?;
        let mut file_bits = Vec::new();
        for line in text.lines() {
            let s = py::strip(line);
            if !s.is_empty() {
                file_bits.push(float_le_hex(s)?);
            }
        }
        bits.insert(name, file_bits);
    }
    let Some((_, first)) = bits.iter().next() else {
        return Err(Error::Invalid(format!(
            "no pred-*.txt files found in {}",
            dir.display()
        )));
    };
    for (name, value) in &bits {
        if value != first {
            return Ok((String::new(), format!("MISMATCH {name}\n"), false));
        }
    }
    let mut out = first.join("\n");
    out.push('\n');
    Ok((out, String::new(), true))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn agrees_when_all_pred_files_have_identical_ieee754_bits() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("pred-rs.txt"), "1.5\n-0.25\n0.0\n").unwrap();
        fs::write(
            dir.path().join("pred-py.txt"),
            "1.5000000000000000\n-0.25\n0\n",
        )
        .unwrap();
        let (out, err, ok) = compare_dir(dir.path()).unwrap();
        assert!(ok);
        assert!(err.is_empty());
        assert_eq!(
            out,
            "000000000000f83f\n000000000000d0bf\n0000000000000000\n"
        );
    }

    #[test]
    fn reports_first_mismatching_file_on_stderr() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("pred-a.txt"), "1.0\n2.0\n").unwrap();
        fs::write(dir.path().join("pred-b.txt"), "1.0\n2.0000000000000004\n").unwrap();
        let (out, err, ok) = compare_dir(dir.path()).unwrap();
        assert!(!ok);
        assert!(out.is_empty());
        assert_eq!(err, "MISMATCH pred-b.txt\n");
    }
}
