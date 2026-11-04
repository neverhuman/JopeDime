// ---- Streaming reader: many row groups and pages, dictionaries, Snappy (S3 Inventory) ----

/// Snappy, as S3 Inventory writes it.
const CODEC_SNAPPY: i64 = 1;
const CODEC_UNCOMPRESSED: i64 = 0;
const CODEC_ZSTD: i64 = 6;
const ENC_PLAIN: i64 = 0;
const ENC_PLAIN_DICTIONARY: i64 = 2;
const ENC_RLE: i64 = 3;
const ENC_RLE_DICTIONARY: i64 = 8;

/// One leaf column of a flat schema.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColumnMeta {
    /// Name recorded for this column meta.
    pub name: String,
    /// Parquet physical type (0 BOOLEAN, 1 INT32, 2 INT64, 4 FLOAT, 5 DOUBLE, 6 BYTE_ARRAY).
    pub physical: i64,
    /// Required recorded for this column meta.
    pub required: bool,
    /// For INT64 timestamps: divide by this to get milliseconds (1 millis, 1000 micros, 10^6
    /// nanos); `None` when the column is not a timestamp.
    pub timestamp_ms_divisor: Option<i64>,
}

#[derive(Clone, Debug)]
struct ChunkMeta {
    codec: i64,
    num_values: u64,
    start: usize,
    end: usize,
    statistics: Option<ColumnStatistics>,
}

#[derive(Clone, Debug)]
/// `RowGroupMeta` is a record for the row group meta contract in this module: A minimal Apache Parquet writer and reader for flat tables (the gpdistilled/v1 table).
pub struct RowGroupMeta {
    /// Number of rows represented by this value.
    pub rows: u64,
    chunks: Vec<ChunkMeta>,
}

/// Decoded values of one column chunk.
#[derive(Clone, Debug, PartialEq)]
pub enum Values {
    /// The `Bool` case of `Values`.
    Bool(Vec<Option<bool>>),
    /// The `I64` case of `Values`.
    I64(Vec<Option<i64>>),
    /// The `F64` case of `Values`.
    F64(Vec<Option<f64>>),
    /// The `Bytes` case of `Values`.
    Bytes(Vec<Option<Vec<u8>>>),
}

impl Values {
    /// Performs the len operation.
    pub fn len(&self) -> usize {
        match self {
            Values::Bool(v) => v.len(),
            Values::I64(v) => v.len(),
            Values::F64(v) => v.len(),
            Values::Bytes(v) => v.len(),
        }
    }

    /// Returns whether empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// What a chunk's pages looked like (for receipts: which decoder paths real files used).
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct PageCounts {
    /// Dictionary recorded for this page counts.
    pub dictionary: u64,
    /// Data v1 recorded for this page counts.
    pub data_v1: u64,
    /// Data v2 recorded for this page counts.
    pub data_v2: u64,
    /// Index recorded for this page counts.
    pub index: u64,
    /// Encodings recorded for this page counts.
    pub encodings: std::collections::BTreeSet<i64>,
    /// Codecs recorded for this page counts.
    pub codecs: std::collections::BTreeSet<i64>,
}

impl PageCounts {
    /// Performs the add operation.
    pub fn add(&mut self, o: &PageCounts) {
        self.dictionary += o.dictionary;
        self.data_v1 += o.data_v1;
        self.data_v2 += o.data_v2;
        self.index += o.index;
        self.encodings.extend(&o.encodings);
        self.codecs.extend(&o.codecs);
    }
}

/// A Parquet file opened for column-chunk reads. Flat schemas only; anything this reader does
/// not implement (nested columns, INT96, DELTA encodings, other codecs) is an error.
pub struct FileReader<'a> {
    bytes: &'a [u8],
    /// Collection of columns values carried by this file reader.
    pub columns: Vec<ColumnMeta>,
    /// Number of num rows represented here.
    pub num_rows: u64,
    /// Collection of row groups values carried by this file reader.
    pub row_groups: Vec<RowGroupMeta>,
    /// Optional created by value; `None` means it is not available.
    pub created_by: Option<String>,
}
