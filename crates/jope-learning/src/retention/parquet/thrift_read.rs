// ---- Thrift compact protocol: reader ----

#[derive(Clone, Debug)]
enum Tv {
    /// A list boolean or a double the reader consumes but does not use.
    Other,
    /// A struct-field boolean (compact protocol: the value is the field type).
    Bool(bool),
    Int(i64),
    Bin(Vec<u8>),
    List(Vec<Tv>),
    Struct(BTreeMap<i16, Tv>),
}

struct Tr<'a> {
    b: &'a [u8],
    p: usize,
    depth: usize,
}

impl<'a> Tr<'a> {
    fn byte(&mut self) -> Result<u8> {
        let v = *self.b.get(self.p).context("thrift: unexpected end")?;
        self.p += 1;
        Ok(v)
    }
    fn varint(&mut self) -> Result<u64> {
        let mut v = 0u64;
        for shift in (0..64).step_by(7) {
            let b = self.byte()?;
            v |= u64::from(b & 0x7f) << shift;
            if b & 0x80 == 0 {
                return Ok(v);
            }
        }
        bail!("thrift: varint too long")
    }
    fn zigzag(&mut self) -> Result<i64> {
        let v = self.varint()?;
        Ok(((v >> 1) as i64) ^ -((v & 1) as i64))
    }
    fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.p.checked_add(n).context("thrift: length overflow")?;
        ensure!(end <= self.b.len(), "thrift: binary runs past the end");
        let s = &self.b[self.p..end];
        self.p = end;
        Ok(s)
    }
    fn value(&mut self, ty: u8) -> Result<Tv> {
        Ok(match ty {
            1 | 2 => Tv::Bool(ty == 1),
            3 => Tv::Int(i64::from(self.byte()? as i8)),
            4..=6 => Tv::Int(self.zigzag()?),
            7 => {
                self.bytes(8)?;
                Tv::Other
            }
            8 => {
                let n = usize::try_from(self.varint()?)?;
                Tv::Bin(self.bytes(n)?.to_vec())
            }
            9 | 10 => {
                let h = self.byte()?;
                let et = h & 0x0f;
                let mut n = u64::from(h >> 4);
                if n == 15 {
                    n = self.varint()?;
                }
                ensure!(
                    n <= self.b.len() as u64,
                    "thrift: list longer than the buffer"
                );
                let mut items = Vec::with_capacity(n as usize);
                for _ in 0..n {
                    items.push(match et {
                        1 | 2 => {
                            self.byte()?;
                            Tv::Other
                        }
                        _ => self.value(et)?,
                    });
                }
                Tv::List(items)
            }
            12 => Tv::Struct(self.strukt()?),
            _ => bail!("thrift: unsupported type {ty}"),
        })
    }
    fn strukt(&mut self) -> Result<BTreeMap<i16, Tv>> {
        self.depth += 1;
        ensure!(self.depth < 32, "thrift: nesting too deep");
        let mut out = BTreeMap::new();
        let mut last = 0i16;
        loop {
            let h = self.byte()?;
            if h == 0 {
                break;
            }
            let ty = h & 0x0f;
            let delta = i16::from(h >> 4);
            let id = if delta == 0 {
                i16::try_from(self.zigzag()?)?
            } else {
                last + delta
            };
            last = id;
            out.insert(id, self.value(ty)?);
        }
        self.depth -= 1;
        Ok(out)
    }
}

fn int(m: &BTreeMap<i16, Tv>, id: i16) -> Option<i64> {
    match m.get(&id) {
        Some(Tv::Int(v)) => Some(*v),
        _ => None,
    }
}

fn bin(m: &BTreeMap<i16, Tv>, id: i16) -> Option<&[u8]> {
    match m.get(&id) {
        Some(Tv::Bin(v)) => Some(v),
        _ => None,
    }
}

fn list(m: &BTreeMap<i16, Tv>, id: i16) -> &[Tv] {
    match m.get(&id) {
        Some(Tv::List(v)) => v,
        _ => &[],
    }
}

fn strukt(v: &Tv) -> Result<&BTreeMap<i16, Tv>> {
    match v {
        Tv::Struct(m) => Ok(m),
        _ => bail!("parquet: expected a struct"),
    }
}

fn read_levels(data: &[u8], n: usize) -> Result<(Vec<bool>, usize)> {
    let (len, body) = length_prefixed(data)?;
    let out = rle_hybrid(body, 1, n)?;
    Ok((out.into_iter().map(|v| v == 1).collect(), 4 + len))
}

/// A 4-byte little-endian length and the bytes it covers.
fn length_prefixed(data: &[u8]) -> Result<(usize, &[u8])> {
    ensure!(data.len() >= 4, "parquet: RLE section truncated");
    let len = u32::from_le_bytes(data[..4].try_into()?) as usize;
    ensure!(
        4 + len <= data.len(),
        "parquet: RLE section runs past the page"
    );
    Ok((len, &data[4..4 + len]))
}

/// The RLE/bit-packed hybrid encoding (levels, dictionary indices, RLE booleans): `n` values of
/// `width` bits. Values decoded past `n` (padding of the last bit-packed group) are dropped.
fn rle_hybrid(data: &[u8], width: u32, n: usize) -> Result<Vec<u32>> {
    ensure!(width <= 32, "parquet: RLE bit width {width} too large");
    let mut r = Tr {
        b: data,
        p: 0,
        depth: 0,
    };
    let mut out = Vec::with_capacity(n);
    let vbytes = width.div_ceil(8) as usize;
    while out.len() < n {
        let h = r.varint()?;
        if h & 1 == 1 {
            let groups = usize::try_from(h >> 1)?;
            let packed = r.bytes(
                groups
                    .checked_mul(width as usize)
                    .context("parquet: RLE overflow")?,
            )?;
            let (mut acc, mut bits) = (0u64, 0u32);
            let mask = if width == 32 {
                u64::from(u32::MAX)
            } else {
                (1u64 << width) - 1
            };
            let mut it = packed.iter();
            for _ in 0..groups * 8 {
                while bits < width {
                    acc |=
                        u64::from(*it.next().context("parquet: bit-packed run truncated")?) << bits;
                    bits += 8;
                }
                out.push((acc & mask) as u32);
                acc >>= width;
                bits -= width;
            }
        } else {
            let run = usize::try_from(h >> 1)?;
            ensure!(run > 0, "parquet: empty RLE run");
            ensure!(
                out.len() + run <= n + 8,
                "parquet: RLE run past the page's values"
            );
            let v = u32::try_from(le_value(r.bytes(vbytes)?))?;
            out.extend(std::iter::repeat_n(v, run));
        }
    }
    out.truncate(n);
    Ok(out)
}

fn le_value(b: &[u8]) -> u64 {
    b.iter()
        .rev()
        .fold(0u64, |acc, &x| (acc << 8) | u64::from(x))
}

fn take_bytes<'a>(data: &'a [u8], p: &mut usize, k: usize) -> Result<&'a [u8]> {
    let end = p
        .checked_add(k)
        .context("parquet: value length overflows")?;
    ensure!(end <= data.len(), "parquet: values run past the page");
    let s = &data[*p..end];
    *p = end;
    Ok(s)
}

fn read_plain(ty: ColumnType, data: &[u8], n: usize) -> Result<Vec<Cell>> {
    let mut out = Vec::with_capacity(n);
    let mut p = 0usize;
    let mut take = |k: usize| take_bytes(data, &mut p, k);
    match ty {
        ColumnType::Boolean => {
            let bits = take(n.div_ceil(8))?;
            for i in 0..n {
                out.push(Cell::Bool(bits[i / 8] >> (i % 8) & 1 == 1));
            }
        }
        ColumnType::Int64 | ColumnType::TimestampMillis => {
            for _ in 0..n {
                out.push(Cell::I64(i64::from_le_bytes(take(8)?.try_into()?)));
            }
        }
        ColumnType::Double => {
            for _ in 0..n {
                out.push(Cell::F64(f64::from_le_bytes(take(8)?.try_into()?)));
            }
        }
        ColumnType::Utf8 => {
            for _ in 0..n {
                let len = u32::from_le_bytes(take(4)?.try_into()?) as usize;
                out.push(Cell::Str(
                    String::from_utf8(take(len)?.to_vec()).context("parquet: utf8 value")?,
                ));
            }
        }
    }
    Ok(out)
}

/// Reads a file written by [`write()`] (one row group, PLAIN, UNCOMPRESSED or ZSTD pages).
pub fn read(bytes: &[u8]) -> Result<Table> {
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
    let nrows = usize::try_from(int(&meta, 3).context("parquet: num_rows missing")?)?;
    let schema = list(&meta, 2);
    ensure!(!schema.is_empty(), "parquet: empty schema");
    let mut columns = Vec::new();
    for el in &schema[1..] {
        let m = strukt(el)?;
        let name = String::from_utf8(bin(m, 4).context("parquet: column name")?.to_vec())?;
        let ty = ColumnType::from_physical(int(m, 1).context("parquet: column type")?, int(m, 6))?;
        let required = match int(m, 3) {
            Some(0) => true,
            Some(1) => false,
            other => bail!("parquet: column {name} repetition {other:?} unsupported"),
        };
        columns.push(ColumnSpec { name, ty, required });
    }
    let groups = list(&meta, 4);
    ensure!(
        nrows == 0 || !groups.is_empty(),
        "parquet: {nrows} rows but no row group"
    );
    ensure!(
        groups.len() <= 1,
        "parquet: {} row groups; this reader takes one",
        groups.len()
    );
    let mut cols: Vec<Vec<Cell>> = Vec::new();
    if let Some(g) = groups.first() {
        let chunks = list(strukt(g)?, 1);
        ensure!(
            chunks.len() == columns.len(),
            "parquet: {} column chunks for {} columns",
            chunks.len(),
            columns.len()
        );
        for (col, chunk) in columns.iter().zip(chunks) {
            let cm = strukt(
                strukt(chunk)?
                    .get(&3)
                    .context("parquet: column metadata missing")?,
            )?;
            let codec = int(cm, 4).unwrap_or(-1);
            let off = usize::try_from(int(cm, 9).context("parquet: data_page_offset")?)?;
            ensure!(off < bytes.len(), "parquet: page offset outside the file");
            let mut r = Tr {
                b: &bytes[off..],
                p: 0,
                depth: 0,
            };
            let ph = r.strukt()?;
            ensure!(
                int(&ph, 1) == Some(0),
                "parquet: column {} first page is not a v1 data page",
                col.name
            );
            let usize_ = usize::try_from(int(&ph, 2).context("parquet: page size")?)?;
            let csize = usize::try_from(int(&ph, 3).context("parquet: page size")?)?;
            ensure!(
                usize_ <= MAX_PAGE,
                "parquet: page larger than the reader cap"
            );
            let dph = strukt(ph.get(&5).context("parquet: data page header")?)?;
            ensure!(
                int(dph, 2) == Some(0),
                "parquet: column {} is not PLAIN",
                col.name
            );
            let n = usize::try_from(int(dph, 1).context("parquet: num_values")?)?;
            ensure!(
                n == nrows,
                "parquet: column {} has {n} values for {nrows} rows",
                col.name
            );
            let body = r.bytes(csize)?;
            let page = match codec {
                0 => body.to_vec(),
                6 => jope_records::zstd_frame::decompress(body, usize_)
                    .map_err(anyhow::Error::msg)?,
                c => bail!("parquet: codec {c} unsupported"),
            };
            ensure!(
                page.len() == usize_,
                "parquet: page decodes to {} bytes, header says {usize_}",
                page.len()
            );
            let (defined, used) = if col.required {
                (vec![true; n], 0)
            } else {
                read_levels(&page, n)?
            };
            let present = defined.iter().filter(|d| **d).count();
            let mut values = read_plain(col.ty, &page[used..], present)?.into_iter();
            cols.push(
                defined
                    .iter()
                    .map(|d| {
                        if *d {
                            values.next().unwrap_or(Cell::Null)
                        } else {
                            Cell::Null
                        }
                    })
                    .collect(),
            );
        }
    }
    let rows = (0..nrows)
        .map(|r| cols.iter().map(|c| c[r].clone()).collect())
        .collect();
    let metadata = list(&meta, 5)
        .iter()
        .map(|kv| {
            let m = strukt(kv)?;
            Ok((
                String::from_utf8(bin(m, 1).unwrap_or_default().to_vec())?,
                String::from_utf8(bin(m, 2).unwrap_or_default().to_vec())?,
            ))
        })
        .collect::<Result<_>>()?;
    Ok(Table {
        columns,
        rows,
        metadata,
    })
}
