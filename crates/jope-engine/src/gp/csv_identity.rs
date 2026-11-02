/// The SHA-256 of `csv_bytes(rows)`, the rows' identity in cuts, feature views and score
/// identities, computed while the CSV is written: the table is never rendered into memory
/// (about 20 bytes per cell, several times the table) and each row's cell strings reuse one set
/// of buffers instead of allocating a `String` per cell. Same writer, same field text
/// (`f64` `Display`), so the same bytes reach the hasher.
fn csv_sha256(rows: &[Vec<f64>]) -> Result<String> {
    use sha2::Digest as _;
    use std::fmt::Write as _;
    let mut writer = csv::WriterBuilder::new()
        .has_headers(false)
        .from_writer(sha2::Sha256::new());
    let mut cells: Vec<String> = Vec::new();
    for row in rows {
        cells.resize_with(row.len(), String::new);
        for (cell, value) in cells.iter_mut().zip(row) {
            cell.clear();
            write!(cell, "{value}")?;
        }
        writer.write_record(&cells)?;
    }
    let hasher = writer
        .into_inner()
        .map_err(|error| anyhow::anyhow!("csv digest: {}", error.error()))?;
    Ok(hex::encode(hasher.finalize()))
}

#[cfg(test)]
mod csv_digest_tests {
    use super::*;

    #[test]
    fn streamed_csv_digest_equals_the_digest_of_the_rendered_csv() {
        let specials = [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            0.0,
            -0.0,
            1e300,
            -1e-300,
            5e-324,
            f64::MAX,
            0.1 + 0.2,
            -123456.789,
            42.0,
        ];
        let tables: Vec<Vec<Vec<f64>>> = vec![
            vec![],
            vec![vec![]],
            vec![specials.to_vec()],
            specials.chunks(4).map(<[f64]>::to_vec).collect(),
            (0..200)
                .map(|i| (0..7).map(|j| (i * 7 + j) as f64 / 3.0 - 50.0).collect())
                .collect(),
        ];
        for rows in tables {
            assert_eq!(csv_sha256(&rows).unwrap(), hash(&csv_bytes(&rows).unwrap()));
        }
        // The csv writer refuses a record whose width differs from the first one; both paths
        // use it, so a ragged table is refused by both rather than given a digest.
        let ragged: Vec<Vec<f64>> = specials.chunks(5).map(<[f64]>::to_vec).collect();
        assert!(csv_bytes(&ragged).is_err());
        assert!(csv_sha256(&ragged).is_err());
    }
}

#[cfg(test)]
mod cpu_evaluator_tests {
    use super::*;

    #[test]
    fn explicit_evaluator_has_no_current_executable_fallback() {
        assert!(cpu_evaluator_binding(None).is_err());
        assert!(cpu_evaluator_binding(Some("relative-executable".into())).is_err());
        let executable = std::env::current_exe().unwrap();
        assert_eq!(
            cpu_evaluator_binding(Some(executable.clone().into_os_string())).unwrap(),
            executable.canonicalize().unwrap()
        );
        let directory = tempfile::tempdir().unwrap();
        assert!(cpu_evaluator_binding(Some(directory.path().as_os_str().into())).is_err());
        let plain = directory.path().join("not-executable");
        fs::write(&plain, b"not an executable").unwrap();
        assert!(cpu_evaluator_binding(Some(plain.into_os_string())).is_err());
    }
}
