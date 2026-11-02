/// Page compression for [`write_with`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Codec {
    /// The `Uncompressed` case of `Codec`.
    Uncompressed,
    /// The `Zstd` case of `Codec`.
    Zstd,
    /// The `Snappy` case of `Codec`.
    Snappy,
}

impl Codec {
    fn id(self) -> i32 {
        match self {
            Codec::Uncompressed => 0,
            Codec::Snappy => 1,
            Codec::Zstd => 6,
        }
    }

    fn apply(self, page: &[u8]) -> Vec<u8> {
        match self {
            Codec::Uncompressed => page.to_vec(),
            Codec::Zstd => jope_records::zstd_frame::compress(page),
            Codec::Snappy => super::snappy::compress(page),
        }
    }
}

/// Layout knobs. The defaults (one row group, one PLAIN page per column) are what
/// `gp_distilled/v1` uses; the others exist so fixtures can look like S3 Inventory files.
#[derive(Clone, Copy, Debug)]
pub struct WriteOptions {
    /// Codec recorded for this write options.
    pub codec: Codec,
    /// Rows per row group (0: one row group).
    pub row_group_rows: usize,
    /// Rows per data page (0: one page per column chunk).
    pub page_rows: usize,
    /// Dictionary-encode UTF8, INT64 and timestamp columns (PLAIN_DICTIONARY, as parquet-mr).
    pub dictionary: bool,
}

/// `values` of `width` bits as one bit-packed hybrid run (padded to a multiple of 8 values).
fn bitpacked_run(values: &[u32], width: u32) -> Vec<u8> {
    let groups = values.len().div_ceil(8);
    let mut run = Tw::new();
    run.varint(((groups as u64) << 1) | 1);
    let mut out = run.buf;
    let mut packed = vec![0u8; groups * width as usize];
    let mut bit = 0usize;
    for i in 0..groups * 8 {
        let v = values.get(i).copied().unwrap_or(0);
        for b in 0..width {
            if v >> b & 1 == 1 {
                packed[bit / 8] |= 1 << (bit % 8);
            }
            bit += 1;
        }
    }
    out.extend(packed);
    out
}

/// Distinct values in first-seen order and each present cell's index into them.
fn dictionary(present: &[&Cell]) -> (Vec<Cell>, Vec<u32>) {
    let mut seen: BTreeMap<String, u32> = BTreeMap::new();
    let mut dict = Vec::new();
    let mut idx = Vec::with_capacity(present.len());
    for c in present {
        let k = format!("{c:?}");
        let i = *seen.entry(k).or_insert_with(|| {
            dict.push((*c).clone());
            (dict.len() - 1) as u32
        });
        idx.push(i);
    }
    (dict, idx)
}

struct ChunkOut {
    file_offset: i64,
    data_offset: i64,
    dict_offset: Option<i64>,
    uncompressed: i64,
    compressed: i64,
    dictionary: bool,
    statistics: Option<ColumnStatistics>,
}

/// Appends one page (header + body) and returns (uncompressed, compressed) sizes incl. header.
fn put_page(
    out: &mut Vec<u8>,
    codec: Codec,
    raw: &[u8],
    header: impl FnOnce(&mut Tw),
    page_type: i32,
) -> Result<(i64, i64)> {
    let body = codec.apply(raw);
    let mut h = Tw::new();
    h.i32f(1, page_type);
    h.i32f(2, i32::try_from(raw.len())?);
    h.i32f(3, i32::try_from(body.len())?);
    header(&mut h);
    let header = h.finish();
    out.extend(&header);
    out.extend(&body);
    Ok((
        (header.len() + raw.len()) as i64,
        (header.len() + body.len()) as i64,
    ))
}

/// Writes `table` as Parquet. `zstd` selects ZSTD page compression, else UNCOMPRESSED.
pub fn write(table: &Table, zstd: bool, created_by: &str) -> Result<Vec<u8>> {
    let codec = if zstd {
        Codec::Zstd
    } else {
        Codec::Uncompressed
    };
    write_with(
        table,
        &WriteOptions {
            codec,
            row_group_rows: 0,
            page_rows: 0,
            dictionary: false,
        },
        created_by,
    )
}

/// Writes with to the configured destination.
pub fn write_with(table: &Table, opts: &WriteOptions, created_by: &str) -> Result<Vec<u8>> {
    write_table(table, opts, created_by, false)
}

/// Additive statistics footer; legacy write/write_with bytes remain unchanged.
pub fn write_with_statistics(
    table: &Table,
    opts: &WriteOptions,
    created_by: &str,
) -> Result<Vec<u8>> {
    write_table(table, opts, created_by, true)
}

fn write_table(
    table: &Table,
    opts: &WriteOptions,
    created_by: &str,
    statistics: bool,
) -> Result<Vec<u8>> {
    let ncol = table.columns.len();
    ensure!(ncol > 0, "a table needs at least one column");
    for (r, row) in table.rows.iter().enumerate() {
        ensure!(
            row.len() == ncol,
            "row {r} has {} cells, the schema {ncol}",
            row.len()
        );
    }
    let nrows = table.rows.len();
    let group_rows = if opts.row_group_rows == 0 {
        nrows.max(1)
    } else {
        opts.row_group_rows
    };
    let mut out = MAGIC.to_vec();
    let mut groups: Vec<(usize, Vec<ChunkOut>)> = Vec::new();
    let mut start = 0usize;
    loop {
        let rows = &table.rows[start..(start + group_rows).min(nrows)];
        let mut chunks = Vec::with_capacity(ncol);
        for (ci, col) in table.columns.iter().enumerate() {
            let cells: Vec<&Cell> = rows.iter().map(|r| &r[ci]).collect();
            if col.required {
                ensure!(
                    cells.iter().all(|c| !matches!(c, Cell::Null)),
                    "required column {} holds a null",
                    col.name
                );
            }
            let file_offset = out.len() as i64;
            let use_dict =
                opts.dictionary && col.ty != ColumnType::Boolean && col.ty != ColumnType::Double;
            let all_present: Vec<&Cell> = cells
                .iter()
                .copied()
                .filter(|c| !matches!(c, Cell::Null))
                .collect();
            let (dict, all_idx) = if use_dict {
                dictionary(&all_present)
            } else {
                (Vec::new(), Vec::new())
            };
            let (mut unc, mut comp) = (0i64, 0i64);
            let dict_offset = if use_dict {
                let off = out.len() as i64;
                let refs: Vec<&Cell> = dict.iter().collect();
                let raw = plain(col.ty, &refs).with_context(|| format!("column {}", col.name))?;
                let n = i32::try_from(dict.len())?;
                let (u, c) = put_page(
                    &mut out,
                    opts.codec,
                    &raw,
                    |h| {
                        h.struct_begin(7);
                        h.i32f(1, n);
                        h.i32f(2, 0); // PLAIN
                        h.struct_end();
                    },
                    2,
                )?;
                unc += u;
                comp += c;
                Some(off)
            } else {
                None
            };
            let width = (32 - (dict.len().max(2) as u32 - 1).leading_zeros()).max(1);
            let data_offset = out.len() as i64;
            let page_rows = if opts.page_rows == 0 {
                cells.len().max(1)
            } else {
                opts.page_rows
            };
            let mut p0 = 0usize;
            let mut idx_pos = 0usize;
            loop {
                let slice = &cells[p0..(p0 + page_rows).min(cells.len())];
                let defined: Vec<bool> = slice.iter().map(|c| !matches!(c, Cell::Null)).collect();
                let present: Vec<&Cell> = slice
                    .iter()
                    .copied()
                    .filter(|c| !matches!(c, Cell::Null))
                    .collect();
                let mut raw = if col.required {
                    Vec::new()
                } else {
                    def_levels(&defined)
                };
                if use_dict {
                    raw.push(width as u8);
                    raw.extend(bitpacked_run(
                        &all_idx[idx_pos..idx_pos + present.len()],
                        width,
                    ));
                    idx_pos += present.len();
                } else {
                    raw.extend(
                        plain(col.ty, &present).with_context(|| format!("column {}", col.name))?,
                    );
                }
                let n = i32::try_from(slice.len())?;
                let (u, c) = put_page(
                    &mut out,
                    opts.codec,
                    &raw,
                    |h| {
                        h.struct_begin(5);
                        h.i32f(1, n);
                        h.i32f(2, if use_dict { 2 } else { 0 }); // PLAIN_DICTIONARY | PLAIN
                        h.i32f(3, 3); // RLE definition levels
                        h.i32f(4, 3); // RLE repetition levels
                        h.struct_end();
                    },
                    0,
                )?;
                unc += u;
                comp += c;
                p0 += slice.len();
                if p0 >= cells.len() {
                    break;
                }
            }
            chunks.push(ChunkOut {
                file_offset: if use_dict { file_offset } else { data_offset },
                data_offset,
                dict_offset,
                uncompressed: unc,
                compressed: comp,
                dictionary: use_dict,
                statistics: statistics
                    .then(|| column_statistics(col.ty, &cells))
                    .transpose()?,
            });
        }
        groups.push((rows.len(), chunks));
        start += rows.len();
        if start >= nrows {
            break;
        }
    }
    let mut f = Tw::new();
    f.i32f(1, 1);
    f.list_begin(2, 12, ncol + 1);
    f.elem_struct_begin();
    f.binf(4, b"schema");
    f.i32f(5, i32::try_from(ncol)?);
    f.struct_end();
    for col in &table.columns {
        f.elem_struct_begin();
        f.i32f(1, col.ty.physical());
        f.i32f(3, if col.required { 0 } else { 1 });
        f.binf(4, col.name.as_bytes());
        match col.ty {
            ColumnType::Utf8 => f.i32f(6, 0),            // ConvertedType UTF8
            ColumnType::TimestampMillis => f.i32f(6, 9), // ConvertedType TIMESTAMP_MILLIS
            _ => {}
        }
        f.struct_end();
    }
    f.i64f(3, nrows as i64);
    f.list_begin(4, 12, groups.len());
    for (rows, chunks) in &groups {
        f.elem_struct_begin();
        f.list_begin(1, 12, ncol);
        for (col, m) in table.columns.iter().zip(chunks) {
            f.elem_struct_begin();
            f.i64f(2, m.file_offset);
            f.struct_begin(3);
            f.i32f(1, col.ty.physical());
            if m.dictionary {
                f.list_begin(2, 5, 3);
                f.elem_i32(0);
                f.elem_i32(2);
                f.elem_i32(3);
            } else {
                f.list_begin(2, 5, 2);
                f.elem_i32(0);
                f.elem_i32(3);
            }
            f.list_begin(3, 8, 1);
            f.elem_bin(col.name.as_bytes());
            f.i32f(4, opts.codec.id());
            f.i64f(5, *rows as i64);
            f.i64f(6, m.uncompressed);
            f.i64f(7, m.compressed);
            f.i64f(9, m.data_offset);
            if let Some(d) = m.dict_offset {
                f.i64f(11, d);
            }
            if let Some(stats) = &m.statistics {
                f.struct_begin(12);
                f.i64f(
                    3,
                    i64::try_from(stats.null_count.context("statistics null count")?)?,
                );
                if let Some(value) = &stats.max {
                    f.binf(5, &value.encode());
                }
                if let Some(value) = &stats.min {
                    f.binf(6, &value.encode());
                }
                if let Some(exact) = stats.max_is_exact {
                    f.field(7, if exact { 1 } else { 2 });
                }
                if let Some(exact) = stats.min_is_exact {
                    f.field(8, if exact { 1 } else { 2 });
                }
                if let Some(count) = stats.nan_count {
                    f.i64f(9, i64::try_from(count)?);
                }
                f.struct_end();
            }
            f.struct_end();
            f.struct_end();
        }
        f.i64f(2, chunks.iter().map(|m| m.uncompressed).sum());
        f.i64f(3, *rows as i64);
        f.struct_end();
    }
    f.list_begin(5, 12, table.metadata.len());
    for (k, v) in &table.metadata {
        f.elem_struct_begin();
        f.binf(1, k.as_bytes());
        f.binf(2, v.as_bytes());
        f.struct_end();
    }
    f.binf(6, created_by.as_bytes());
    if statistics {
        // Modern bounds have meaning only under an explicit ColumnOrder. TYPE_ORDER uses
        // signed numeric/Boolean order and unsigned lexicographic UTF8 bytes.
        f.list_begin(7, 12, ncol);
        for _ in &table.columns {
            f.elem_struct_begin();
            f.struct_begin(1); // ColumnOrder.TYPE_ORDER: empty TypeDefinedOrder
            f.struct_end();
            f.struct_end();
        }
    }
    let footer = f.finish();
    out.extend(&footer);
    out.extend((u32::try_from(footer.len())?).to_le_bytes());
    out.extend(MAGIC);
    Ok(out)
}
