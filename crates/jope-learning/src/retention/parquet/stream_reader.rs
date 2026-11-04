fn timestamp_divisor(m: &BTreeMap<i16, Tv>) -> Result<Option<i64>> {
    // converted_type 9 TIMESTAMP_MILLIS, 10 TIMESTAMP_MICROS.
    match int(m, 6) {
        Some(9) => return Ok(Some(1)),
        Some(10) => return Ok(Some(1000)),
        _ => {}
    }
    // logicalType (10) -> TIMESTAMP (8) -> unit (2) -> MILLIS (1) | MICROS (2) | NANOS (3).
    let Some(Tv::Struct(lt)) = m.get(&10) else {
        return Ok(None);
    };
    let Some(Tv::Struct(ts)) = lt.get(&8) else {
        return Ok(None);
    };
    let Some(Tv::Struct(unit)) = ts.get(&2) else {
        bail!("parquet: timestamp without a unit");
    };
    Ok(Some(if unit.contains_key(&1) {
        1
    } else if unit.contains_key(&2) {
        1000
    } else if unit.contains_key(&3) {
        1_000_000
    } else {
        bail!("parquet: unknown timestamp unit")
    }))
}

fn decompress_page(codec: i64, body: &[u8], size: usize) -> Result<Vec<u8>> {
    ensure!(size <= MAX_PAGE, "parquet: page larger than the reader cap");
    let page = match codec {
        CODEC_UNCOMPRESSED => body.to_vec(),
        CODEC_SNAPPY => super::snappy::decompress(body, size)?,
        CODEC_ZSTD => {
            jope_records::zstd_frame::decompress(body, size).map_err(anyhow::Error::msg)?
        }
        c => bail!("parquet: codec {c} unsupported"),
    };
    ensure!(
        page.len() == size,
        "parquet: page decodes to {} bytes, header says {size}",
        page.len()
    );
    Ok(page)
}

/// PLAIN-decodes `n` values of one physical type.
fn plain_values(physical: i64, data: &[u8], n: usize, p: &mut usize) -> Result<Values> {
    Ok(match physical {
        0 => {
            let bits = take_bytes(data, p, n.div_ceil(8))?;
            Values::Bool(
                (0..n)
                    .map(|i| Some(bits[i / 8] >> (i % 8) & 1 == 1))
                    .collect(),
            )
        }
        1 => Values::I64(
            (0..n)
                .map(|_| {
                    Ok(Some(i64::from(i32::from_le_bytes(
                        take_bytes(data, p, 4)?.try_into()?,
                    ))))
                })
                .collect::<Result<_>>()?,
        ),
        2 => Values::I64(
            (0..n)
                .map(|_| {
                    Ok(Some(i64::from_le_bytes(
                        take_bytes(data, p, 8)?.try_into()?,
                    )))
                })
                .collect::<Result<_>>()?,
        ),
        5 => Values::F64(
            (0..n)
                .map(|_| {
                    Ok(Some(f64::from_le_bytes(
                        take_bytes(data, p, 8)?.try_into()?,
                    )))
                })
                .collect::<Result<_>>()?,
        ),
        6 => Values::Bytes(
            (0..n)
                .map(|_| {
                    let len = u32::from_le_bytes(take_bytes(data, p, 4)?.try_into()?) as usize;
                    Ok(Some(take_bytes(data, p, len)?.to_vec()))
                })
                .collect::<Result<_>>()?,
        ),
        t => bail!("parquet: physical type {t} unsupported"),
    })
}

fn gather(dict: &Values, idx: &[u32]) -> Result<Values> {
    let get = |i: u32| -> Result<usize> {
        let i = i as usize;
        ensure!(i < dict.len(), "parquet: dictionary index {i} out of range");
        Ok(i)
    };
    Ok(match dict {
        Values::Bool(d) => {
            Values::Bool(idx.iter().map(|&i| Ok(d[get(i)?])).collect::<Result<_>>()?)
        }
        Values::I64(d) => Values::I64(idx.iter().map(|&i| Ok(d[get(i)?])).collect::<Result<_>>()?),
        Values::F64(d) => Values::F64(idx.iter().map(|&i| Ok(d[get(i)?])).collect::<Result<_>>()?),
        Values::Bytes(d) => Values::Bytes(
            idx.iter()
                .map(|&i| Ok(d[get(i)?].clone()))
                .collect::<Result<_>>()?,
        ),
    })
}

/// Spreads `present` non-null values over `defined` slots.
fn spread(present: Values, defined: &[bool]) -> Result<Values> {
    fn go<T: Clone>(v: Vec<Option<T>>, defined: &[bool]) -> Result<Vec<Option<T>>> {
        let mut it = v.into_iter();
        let out: Vec<Option<T>> = defined
            .iter()
            .map(|d| if *d { it.next().flatten() } else { None })
            .collect();
        ensure!(
            it.next().is_none(),
            "parquet: more values than defined slots"
        );
        Ok(out)
    }
    Ok(match present {
        Values::Bool(v) => Values::Bool(go(v, defined)?),
        Values::I64(v) => Values::I64(go(v, defined)?),
        Values::F64(v) => Values::F64(go(v, defined)?),
        Values::Bytes(v) => Values::Bytes(go(v, defined)?),
    })
}

fn append(acc: &mut Option<Values>, v: Values) -> Result<()> {
    match (acc.as_mut(), v) {
        (None, v) => *acc = Some(v),
        (Some(Values::Bool(a)), Values::Bool(b)) => a.extend(b),
        (Some(Values::I64(a)), Values::I64(b)) => a.extend(b),
        (Some(Values::F64(a)), Values::F64(b)) => a.extend(b),
        (Some(Values::Bytes(a)), Values::Bytes(b)) => a.extend(b),
        _ => bail!("parquet: pages of one column decode to different types"),
    }
    Ok(())
}

impl<'a> FileReader<'a> {
    /// Constructs or opens a open value from the supplied inputs.
    pub fn open(bytes: &'a [u8]) -> Result<FileReader<'a>> {
        ensure!(
            bytes.len() >= 12 && &bytes[..4] == MAGIC && &bytes[bytes.len() - 4..] == MAGIC,
            "not a parquet file"
        );
        let flen = u32::from_le_bytes(bytes[bytes.len() - 8..bytes.len() - 4].try_into()?) as usize;
        ensure!(
            flen + 12 <= bytes.len(),
            "parquet footer length {flen} exceeds the file"
        );
        let footer = &bytes[bytes.len() - 8 - flen..bytes.len() - 8];
        let meta = Tr {
            b: footer,
            p: 0,
            depth: 0,
        }
        .strukt()?;
        let num_rows = u64::try_from(int(&meta, 3).context("parquet: num_rows missing")?)?;
        let schema = list(&meta, 2);
        ensure!(!schema.is_empty(), "parquet: empty schema");
        let root = strukt(&schema[0])?;
        ensure!(
            int(root, 5) == Some(schema.len() as i64 - 1),
            "parquet: nested schema (root has {:?} children for {} elements)",
            int(root, 5),
            schema.len() - 1
        );
        let mut columns = Vec::new();
        for el in &schema[1..] {
            let m = strukt(el)?;
            let name = String::from_utf8(bin(m, 4).context("parquet: column name")?.to_vec())?;
            ensure!(
                int(m, 5).is_none(),
                "parquet: column {name} is a group (nested)"
            );
            let physical = int(m, 1).context("parquet: column type")?;
            let required = match int(m, 3) {
                Some(0) => true,
                Some(1) => false,
                other => bail!("parquet: column {name} repetition {other:?} unsupported"),
            };
            let timestamp_ms_divisor = timestamp_divisor(m)?;
            ensure!(
                timestamp_ms_divisor.is_none() || physical == 2,
                "parquet: column {name} is a non-INT64 timestamp (INT96 unsupported)"
            );
            columns.push(ColumnMeta {
                name,
                physical,
                required,
                timestamp_ms_divisor,
            });
        }
        let orders = list(&meta, 7);
        ensure!(
            orders.is_empty() || orders.len() == columns.len(),
            "parquet: column order count mismatch"
        );
        let mut row_groups = Vec::new();
        let mut total = 0u64;
        for g in list(&meta, 4) {
            let g = strukt(g)?;
            let rows = u64::try_from(int(g, 3).context("parquet: row group num_rows")?)?;
            total += rows;
            let chunks = list(g, 1);
            ensure!(
                chunks.len() == columns.len(),
                "parquet: {} column chunks for {} columns",
                chunks.len(),
                columns.len()
            );
            let mut metas = Vec::new();
            for (ci, (col, chunk)) in columns.iter().zip(chunks).enumerate() {
                let cm = strukt(
                    strukt(chunk)?
                        .get(&3)
                        .context("parquet: column metadata missing (external chunk?)")?,
                )?;
                let path: Vec<&[u8]> = list(cm, 3)
                    .iter()
                    .filter_map(|v| match v {
                        Tv::Bin(b) => Some(b.as_slice()),
                        _ => None,
                    })
                    .collect();
                ensure!(
                    path == [col.name.as_bytes()],
                    "parquet: chunk path does not match column {}",
                    col.name
                );
                let data_off = usize::try_from(int(cm, 9).context("parquet: data_page_offset")?)?;
                let start = match int(cm, 11) {
                    Some(d) if d > 0 && (d as usize) < data_off => d as usize,
                    _ => data_off,
                };
                let size = usize::try_from(int(cm, 7).context("parquet: total_compressed_size")?)?;
                let end = start.checked_add(size).context("parquet: chunk overflow")?;
                ensure!(
                    end <= bytes.len() - 8 - flen,
                    "parquet: column {} chunk runs into the footer",
                    col.name
                );
                let num_values = u64::try_from(int(cm, 5).context("parquet: chunk num_values")?)?;
                ensure!(
                    num_values == rows,
                    "parquet: column {} has {num_values} values for {rows} rows",
                    col.name
                );
                metas.push(ChunkMeta {
                    codec: int(cm, 4).context("parquet: codec")?,
                    num_values,
                    start,
                    end,
                    statistics: cm
                        .get(&12)
                        .map(|s| {
                            decode_statistics(
                                strukt(s)?,
                                col.physical,
                                rows,
                                orders
                                    .get(ci)
                                    .and_then(|o| strukt(o).ok())
                                    .is_some_and(|m| m.contains_key(&1)),
                            )
                        })
                        .transpose()?,
                });
            }
            row_groups.push(RowGroupMeta {
                rows,
                chunks: metas,
            });
        }
        ensure!(
            total == num_rows,
            "parquet: row groups hold {total} rows, footer says {num_rows}"
        );
        let created_by = bin(&meta, 6).map(|b| String::from_utf8_lossy(b).to_string());
        Ok(FileReader {
            bytes,
            columns,
            num_rows,
            row_groups,
            created_by,
        })
    }

    /// Performs the column index operation.
    pub fn column_index(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|c| c.name == name)
    }

    /// Returns decoded footer statistics for this column chunk when they are present.
    ///
    /// Legacy files without footer statistics return `None`; see
    /// `retention_pack::optional_statistics_preserve_legacy_distilled_bytes_and_reader`.
    pub fn statistics(&self, group: usize, col: usize) -> Result<Option<&ColumnStatistics>> {
        let chunk = self
            .row_groups
            .get(group)
            .context("parquet: no such row group")?
            .chunks
            .get(col)
            .context("parquet: no such column")?;
        Ok(chunk.statistics.as_ref())
    }

    /// Decodes column `col` of row group `group`: every page, with its dictionary.
    pub fn read_column(&self, group: usize, col: usize) -> Result<(Values, PageCounts)> {
        let rg = self
            .row_groups
            .get(group)
            .context("parquet: no such row group")?;
        let c = self.columns.get(col).context("parquet: no such column")?;
        let ch = &rg.chunks[col];
        let chunk = &self.bytes[ch.start..ch.end];
        let mut counts = PageCounts::default();
        counts.codecs.insert(ch.codec);
        let mut dict: Option<Values> = None;
        let mut acc: Option<Values> = None;
        let mut done = 0u64;
        let mut p = 0usize;
        while done < ch.num_values {
            ensure!(
                p < chunk.len(),
                "parquet: column {} ends before its values",
                c.name
            );
            let mut r = Tr {
                b: &chunk[p..],
                p: 0,
                depth: 0,
            };
            let ph = r.strukt()?;
            let header_len = r.p;
            let ty = int(&ph, 1).context("parquet: page type")?;
            let usize_ = usize::try_from(int(&ph, 2).context("parquet: page size")?)?;
            let csize = usize::try_from(int(&ph, 3).context("parquet: page size")?)?;
            let body = take_bytes(chunk, &mut (p + header_len), csize)?;
            p += header_len + csize;
            match ty {
                2 => {
                    counts.dictionary += 1;
                    ensure!(
                        dict.is_none(),
                        "parquet: column {} has two dictionaries",
                        c.name
                    );
                    let dh = strukt(ph.get(&7).context("parquet: dictionary page header")?)?;
                    let n = usize::try_from(int(dh, 1).context("parquet: dictionary size")?)?;
                    let enc = int(dh, 2).unwrap_or(ENC_PLAIN);
                    ensure!(
                        enc == ENC_PLAIN || enc == ENC_PLAIN_DICTIONARY,
                        "parquet: dictionary encoding {enc} unsupported"
                    );
                    let page = decompress_page(ch.codec, body, usize_)?;
                    dict = Some(plain_values(c.physical, &page, n, &mut 0)?);
                }
                0 | 3 => {
                    let (n, enc, defined, values): (usize, i64, Vec<bool>, Vec<u8>) = if ty == 0 {
                        counts.data_v1 += 1;
                        let dh = strukt(ph.get(&5).context("parquet: data page header")?)?;
                        let n = usize::try_from(int(dh, 1).context("parquet: num_values")?)?;
                        let page = decompress_page(ch.codec, body, usize_)?;
                        let (defined, used) = if c.required {
                            (vec![true; n], 0)
                        } else {
                            ensure!(
                                int(dh, 3) == Some(ENC_RLE),
                                "parquet: column {} definition levels not RLE",
                                c.name
                            );
                            read_levels(&page, n)?
                        };
                        let enc = int(dh, 2).context("parquet: encoding")?;
                        (n, enc, defined, page[used..].to_vec())
                    } else {
                        counts.data_v2 += 1;
                        let dh = strukt(ph.get(&8).context("parquet: data page v2 header")?)?;
                        let n = usize::try_from(int(dh, 1).context("parquet: num_values")?)?;
                        let dlen = usize::try_from(int(dh, 5).unwrap_or(0))?;
                        let rlen = usize::try_from(int(dh, 6).unwrap_or(0))?;
                        ensure!(rlen == 0, "parquet: repetition levels in a flat column");
                        let compressed = !matches!(dh.get(&7), Some(Tv::Bool(false)));
                        ensure!(
                            dlen <= body.len() && dlen <= usize_,
                            "parquet: v2 levels past the page"
                        );
                        let defined = if c.required {
                            vec![true; n]
                        } else {
                            rle_hybrid(&body[..dlen], 1, n)?
                                .into_iter()
                                .map(|v| v == 1)
                                .collect()
                        };
                        let rest = &body[dlen..];
                        let values = if compressed {
                            decompress_page(ch.codec, rest, usize_ - dlen)?
                        } else {
                            rest.to_vec()
                        };
                        (n, int(dh, 4).context("parquet: encoding")?, defined, values)
                    };
                    counts.encodings.insert(enc);
                    let present = defined.iter().filter(|d| **d).count();
                    let vals = match enc {
                        ENC_PLAIN => plain_values(c.physical, &values, present, &mut 0)?,
                        ENC_PLAIN_DICTIONARY | ENC_RLE_DICTIONARY => {
                            let d = dict
                                .as_ref()
                                .context("parquet: dictionary-encoded page without a dictionary")?;
                            let width = u32::from(
                                *values.first().context("parquet: empty dictionary page")?,
                            );
                            gather(d, &rle_hybrid(&values[1..], width, present)?)?
                        }
                        ENC_RLE if c.physical == 0 => {
                            let (_, b) = length_prefixed(&values)?;
                            Values::Bool(
                                rle_hybrid(b, 1, present)?
                                    .into_iter()
                                    .map(|v| Some(v == 1))
                                    .collect(),
                            )
                        }
                        e => bail!("parquet: column {} encoding {e} unsupported", c.name),
                    };
                    ensure!(
                        vals.len() == present,
                        "parquet: column {} decoded {} of {present} values",
                        c.name,
                        vals.len()
                    );
                    append(&mut acc, spread(vals, &defined)?)?;
                    done += n as u64;
                }
                1 => counts.index += 1,
                t => bail!("parquet: page type {t} unsupported"),
            }
        }
        ensure!(
            done == ch.num_values,
            "parquet: column {} decoded {done} values, chunk says {}",
            c.name,
            ch.num_values
        );
        let vals = match acc {
            Some(v) => v,
            None => plain_values(c.physical, &[], 0, &mut 0)?,
        };
        Ok((vals, counts))
    }
}
