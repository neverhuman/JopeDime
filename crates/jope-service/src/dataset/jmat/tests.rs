use super::*;

use super::*;
const REAL: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../test-support/datasets/v1/messy-csv-like/raw-source-subset.csv"
));

#[test]
fn real_catalog_tables_round_trip_every_cell_and_encode_deterministically() {
    for source in [REAL] {
        let parsed = jope_runtime::table::parse_csv(source, &Default::default()).unwrap();
        let encoded = encode(&parsed).unwrap();
        assert_eq!(encoded, encode(&parsed).unwrap());
        let matrix = Matrix::open(encoded).unwrap();
        assert_eq!(matrix.rows(), parsed.rows.len());
        assert_eq!(matrix.columns().len(), parsed.headers.len());
        let mut missing = 0;
        for (column, header) in parsed.headers.iter().enumerate() {
            assert_eq!(&matrix.columns()[column].name, header);
            assert!(matrix.metadata.columns[column].values.is_multiple_of(64));
            assert!(matrix.metadata.columns[column].present.is_multiple_of(64));
            for (row, values) in parsed.rows.iter().enumerate() {
                assert_eq!(matrix.row_id(row), Some(row as u64));
                match (&values[column], matrix.cell(row, column).unwrap()) {
                    (None, JmatCell::Missing) => missing += 1,
                    (Some(text), JmatCell::Numeric(number)) => {
                        assert_eq!(numeric(text).unwrap().to_bits(), number.to_bits())
                    }
                    (Some(text), JmatCell::Categorical(value)) => assert_eq!(text, value),
                    other => panic!("roundtrip kind mismatch: {other:?}"),
                }
            }
        }
        if source == REAL {
            assert!(missing > 0);
        }
        assert_eq!(matrix.row_id(matrix.rows()), None);
        assert_eq!(matrix.cell(0, matrix.columns().len()), None);
    }
}

fn sign(bytes: &mut [u8]) {
    let end = bytes.len() - CHECKSUM;
    let hash = Sha256::digest(&bytes[..end]);
    bytes[end..].copy_from_slice(&hash);
}
#[test]
fn checksum_version_lengths_offsets_padding_and_row_ids_are_validated() {
    let parsed = jope_runtime::table::parse_csv(REAL, &Default::default()).unwrap();
    let valid = encode(&parsed).unwrap();
    let matrix = Matrix::open(valid.as_slice()).unwrap();
    for offset in [
        0,
        8,
        16,
        24,
        32,
        40,
        48,
        56,
        matrix.row_ids,
        matrix.metadata.columns[0].present as usize,
    ] {
        let mut corrupt = valid.clone();
        corrupt[offset] ^= 0xff;
        sign(&mut corrupt);
        assert!(Matrix::open(corrupt).is_err(), "offset {offset}");
    }
    let mut bad = valid.clone();
    let padding = HEADER + integer(&bad, 24).unwrap();
    bad[padding] = 1;
    sign(&mut bad);
    assert!(Matrix::open(bad).is_err());
    let mut bad = valid.clone();
    let last = bad.len() - 1;
    bad[last] ^= 1;
    assert!(Matrix::open(bad).is_err());
    for length in [0, 7, 63, 64, valid.len() - CHECKSUM, valid.len() - 1] {
        assert!(Matrix::open(&valid[..length]).is_err());
    }
    let mut bad = valid.clone();
    bad[32..40].copy_from_slice(&u64::MAX.to_le_bytes());
    sign(&mut bad);
    assert!(Matrix::open(bad).is_err());
}

#[test]
fn binary64_signed_zero_and_missing_are_distinct_format_values() {
    // Encoding protocol values, not a scientific/benchmark dataset.
    let table = CsvTable {
        headers: vec!["number".into()],
        rows: vec![vec![Some("-0".into())], vec![Some("0".into())], vec![None]],
        sha256: String::new(),
        bytes: 0,
    };
    let matrix = Matrix::open(encode(&table).unwrap()).unwrap();
    for (row, bits) in [(0, (-0.0_f64).to_bits()), (1, 0.0_f64.to_bits())] {
        let Some(JmatCell::Numeric(value)) = matrix.cell(row, 0) else {
            panic!("numeric expected")
        };
        assert_eq!(value.to_bits(), bits);
    }
    assert_eq!(matrix.cell(2, 0), Some(JmatCell::Missing));
}

#[test]
fn category_encoding_is_injective_and_rejects_invalid_codes() {
    // Format protocol vectors; no synthetic data is used to measure a model.
    let table = CsvTable {
        headers: vec!["category".into()],
        rows: [
            Some("NA"),
            Some("null"),
            Some("?"),
            Some(" category "),
            Some("λ"),
            None,
        ]
        .into_iter()
        .map(|v| vec![v.map(str::to_owned)])
        .collect(),
        sha256: String::new(),
        bytes: 0,
    };
    let encoded = encode(&table).unwrap();
    let matrix = Matrix::open(encoded.as_slice()).unwrap();
    assert_eq!(matrix.columns()[0].dictionary.len(), 5);
    for (row, original) in table.rows.iter().enumerate() {
        let expected = original[0]
            .as_deref()
            .map(JmatCell::Categorical)
            .unwrap_or(JmatCell::Missing);
        assert_eq!(matrix.cell(row, 0), Some(expected));
    }
    let mut bad = encoded.clone();
    let start = matrix.columns()[0].values as usize;
    bad[start..start + 4].copy_from_slice(&u32::MAX.to_le_bytes());
    sign(&mut bad);
    assert!(Matrix::open(bad).is_err());
    let mut bad = encoded.clone();
    let mut metadata: serde_json::Value =
        serde_json::from_slice(&encoded[HEADER..HEADER + integer(&encoded, 24).unwrap()]).unwrap();
    metadata["columns"][0]["dictionary"][0] = metadata["columns"][0]["dictionary"][1].clone();
    let json = serde_json::to_vec(&metadata).unwrap();
    bad[HEADER..matrix.row_ids].fill(0);
    bad[24..32].copy_from_slice(&(json.len() as u64).to_le_bytes());
    bad[HEADER..HEADER + json.len()].copy_from_slice(&json);
    sign(&mut bad);
    assert!(Matrix::open(bad).is_err());
}
fn semantic_digest<B: AsRef<[u8]>>(matrix: &Matrix<B>) -> String {
    let mut hash = Sha256::new();
    for column in matrix.columns() {
        hash.update((column.name.len() as u64).to_le_bytes());
        hash.update(column.name.as_bytes());
    }
    for row in 0..matrix.rows() {
        hash.update(matrix.row_id(row).unwrap().to_le_bytes());
        for col in 0..matrix.columns().len() {
            match matrix.cell(row, col).unwrap() {
                JmatCell::Missing => hash.update([0]),
                JmatCell::Numeric(v) => {
                    hash.update([1]);
                    hash.update(v.to_bits().to_le_bytes());
                }
                JmatCell::Categorical(s) => {
                    hash.update([2]);
                    hash.update((s.len() as u64).to_le_bytes());
                    hash.update(s.as_bytes());
                }
            }
        }
    }
    hash.finalize().iter().map(|v| format!("{v:02x}")).collect()
}

#[test]
#[allow(unsafe_code)]
fn separate_process_read_only_mapping_has_identical_values() {
    const CHILD_PATH: &str = "JOPE_SVC11_CHILD_MATRIX";
    const CHILD_DIGEST: &str = "JOPE_SVC11_CHILD_DIGEST";
    if let Some(path) = std::env::var_os(CHILD_PATH) {
        let file = std::fs::File::open(path).unwrap();
        // SAFETY: the parent owns this file and never mutates/truncates it;
        // it waits for this process to exit before removing the directory.
        let matrix = unsafe { map_read_only(&file) }.unwrap();
        assert_eq!(
            semantic_digest(&matrix),
            std::env::var(CHILD_DIGEST).unwrap()
        );
        assert!(matrix.rows() > 0);
        println!("SVC11_NATIVE_MMAP_VALUES_VERIFIED");
        return;
    }
    let mut sources = vec![REAL.to_vec()];
    // Optional additional catalog input supplements the always-run pinned
    // fixture, rather than making a missing capability silently pass.
    if let Some(path) = std::env::var_os("JOPE_SVC11_CATALOG_CSV") {
        sources.push(std::fs::read(path).unwrap());
    }
    for source in sources {
        let parsed = jope_runtime::table::parse_csv(&source, &Default::default()).unwrap();
        let bytes = encode(&parsed).unwrap();
        let parent = Matrix::open(bytes.as_slice()).unwrap();
        let home = tempfile::tempdir().unwrap();
        let path = home.path().join("matrix.jmat");
        std::fs::write(&path, &bytes).unwrap();
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "dataset::jmat::tests::separate_process_read_only_mapping_has_identical_values",
                "--nocapture",
            ])
            .env(CHILD_PATH, &path)
            .env(CHILD_DIGEST, semantic_digest(&parent))
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            String::from_utf8_lossy(&output.stdout).contains("SVC11_NATIVE_MMAP_VALUES_VERIFIED")
        );
    }
}
