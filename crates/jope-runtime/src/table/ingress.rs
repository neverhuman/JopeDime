use super::*;

/// Fitted envelopes must retain this policy and replay `raw_table` at inference.
pub const RAW_CELL_ENCODING: &str = "jope.raw-cell-encoding/v1";
pub const CATEGORY_PREFIX: &str = "jope.category.v1:";

/// Encode one nullable cell. Every present category is namespaced, including literal
/// null/NA/empty, so RawTable's missing-token vocabulary cannot collapse it.
pub fn encode_cell(kind: ColumnKind, value: Option<&str>) -> Result<String> {
    let Some(value) = value else {
        return Ok(String::new());
    };
    match kind {
        ColumnKind::Numeric => {
            let number: f64 = value.trim().parse().context("invalid numeric feature")?;
            ensure!(
                number.is_finite(),
                "numeric features must be finite or null"
            );
            Ok(value.to_owned())
        }
        ColumnKind::Categorical => Ok(format!(
            "{CATEGORY_PREFIX}{}",
            serde_json::to_string(value)?
        )),
    }
}

/// Reusable label-free preprocessing for fitting, workers and fitted inference.
/// `None` is the only missing cell; original categorical strings are injective.
pub fn raw_table(schema: &DataSchema, values: &[Vec<Option<String>>]) -> Result<RawTable> {
    let rows = values
        .iter()
        .map(|row| {
            ensure!(row.len() == schema.columns.len(), "ragged raw feature row");
            row.iter()
                .zip(&schema.columns)
                .map(|(value, column)| encode_cell(column.kind, value.as_deref()))
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?;
    let table = RawTable::new(
        schema
            .columns
            .iter()
            .map(|column| column.name.clone())
            .collect(),
        rows,
    )?;
    table.validate_against(schema)?;
    Ok(table)
}

/// The schema of independent rows over `columns`.
pub fn iid_schema(columns: Vec<ColumnSpec>) -> Result<DataSchema> {
    let schema = DataSchema {
        columns,
        row_semantics: RowSemantics::Iid,
    };
    schema.validate()?;
    Ok(schema)
}

/// W3-T04: the schema of [`fitting_input_sha256`].
pub const FITTING_INPUT: &str = "jope.fitting-input/v1";

/// W3-T04: the digest a fitted artifact binds its fitting input by: the exact training table and
/// targets, in row order, that a fit worker receives. A struct rather than a JSON map, so the
/// byte order never depends on serde_json features.
pub fn fitting_input_sha256(table: &RawTable, targets: &[f64]) -> Result<String> {
    #[derive(Serialize)]
    struct FittingInput<'a> {
        schema: &'static str,
        table: &'a RawTable,
        targets: &'a [f64],
    }
    ensure!(
        table.rows.len() == targets.len() && targets.iter().all(|t| t.is_finite()),
        "fitting input targets differ from its training rows"
    );
    Ok(crate::sha256_hex(&serde_json::to_vec(&FittingInput {
        schema: FITTING_INPUT,
        table,
        targets,
    })?))
}
/// A parsed CSV file: header and nullable cells.
#[derive(Clone, Debug)]
pub struct CsvTable {
    pub headers: Vec<String>,
    pub rows: Vec<Vec<Option<String>>>,
    pub sha256: String,
    pub bytes: u64,
}

/// Read and bound a CSV file before any per-row allocation beyond the limits.
pub fn read_csv(path: &Path, limits: &Limits) -> Result<CsvTable> {
    let metadata =
        std::fs::metadata(path).map_err(|e| Failure::input(format!("{}: {e}", path.display())))?;
    if !metadata.is_file() {
        return Err(Failure::input(format!("{} is not a file", path.display())));
    }
    if metadata.len() > limits.max_input_bytes {
        return Err(Failure::input(format!(
            "{} is {} bytes; the input limit is {}",
            path.display(),
            metadata.len(),
            limits.max_input_bytes
        )));
    }
    let bytes =
        std::fs::read(path).map_err(|e| Failure::input(format!("{}: {e}", path.display())))?;
    parse_csv(&bytes, limits).with_context(|| path.display().to_string())
}

/// Parse CSV bytes under the supplied input, row, and column limits.
///
/// A UTF-8 BOM is accepted and removed before parsing. The header is required and must contain
/// non-empty column names; malformed CSV and inputs exceeding `limits` return an input error.
pub fn parse_csv(bytes: &[u8], limits: &Limits) -> Result<CsvTable> {
    let body = bytes.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(bytes);
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(false)
        .from_reader(body);
    let headers: Vec<String> = reader
        .headers()
        .map_err(|e| Failure::input(format!("CSV header: {e}")))?
        .iter()
        .map(|h| h.trim().to_owned())
        .collect();
    if headers.is_empty() || headers.iter().any(String::is_empty) {
        return Err(Failure::input("CSV header has an empty column name"));
    }
    if headers.len() > limits.max_columns {
        return Err(Failure::input(format!(
            "{} columns; the limit is {}",
            headers.len(),
            limits.max_columns
        )));
    }
    let mut seen = BTreeSet::new();
    if let Some(duplicate) = headers.iter().find(|h| !seen.insert(h.as_str())) {
        return Err(Failure::input(format!("duplicate column {duplicate:?}")));
    }
    let mut rows = Vec::new();
    for (index, record) in reader.records().enumerate() {
        let record =
            record.map_err(|e| Failure::input(format!("CSV record {}: {e}", index + 2)))?;
        if rows.len() == limits.max_rows {
            return Err(Failure::input(format!(
                "more than {} rows; raise --max-rows to accept more",
                limits.max_rows
            )));
        }
        rows.push(
            record
                .iter()
                .map(|cell| {
                    let trimmed = cell.trim_matches(|c: char| c.is_ascii_whitespace());
                    (!trimmed.is_empty()).then(|| cell.to_owned())
                })
                .collect(),
        );
    }
    Ok(CsvTable {
        headers,
        rows,
        sha256: crate::sha256_hex(bytes),
        bytes: bytes.len() as u64,
    })
}
