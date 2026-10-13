//! A minimal Apache Parquet writer and reader for flat tables (the `gp_distilled/v1` table).
//!
//! No Parquet crate is in `Cargo.lock`, and adding one would change the ADR-0004 evaluator
//! identity, so this module writes the subset it needs by hand:
//! - one row group, one data page (v1) per column;
//! - physical types BOOLEAN, INT64, DOUBLE, BYTE_ARRAY (UTF8), each REQUIRED or OPTIONAL;
//! - PLAIN values; definition levels in the RLE/bit-packed hybrid (bit width 1);
//! - pages either UNCOMPRESSED or ZSTD (one zstd frame per page);
//! - the footer (`FileMetaData`) in the Thrift compact protocol, with key/value metadata.
//!
//! The reader decodes exactly that subset (plus RLE runs in definition levels) and refuses
//! anything else, so a table can be reconciled against its raw source (strategy §8.2 E3).

use anyhow::{bail, ensure, Context, Result};
use std::collections::BTreeMap;

const MAGIC: &[u8; 4] = b"PAR1";
/// Decoded page size cap for the reader (a hostile footer cannot request more).
const MAX_PAGE: usize = 1 << 30;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColumnType {
    Boolean,
    Int64,
    Double,
    Utf8,
}

impl ColumnType {
    fn physical(self) -> i32 {
        match self {
            ColumnType::Boolean => 0,
            ColumnType::Int64 => 2,
            ColumnType::Double => 5,
            ColumnType::Utf8 => 6,
        }
    }

    fn from_physical(t: i64, converted: Option<i64>) -> Result<ColumnType> {
        Ok(match (t, converted) {
            (0, None) => ColumnType::Boolean,
            (2, None) => ColumnType::Int64,
            (5, None) => ColumnType::Double,
            (6, Some(0)) => ColumnType::Utf8,
            _ => bail!("unsupported parquet column type {t} (converted {converted:?})"),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColumnSpec {
    pub name: String,
    pub ty: ColumnType,
    pub required: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Cell {
    Null,
    Bool(bool),
    I64(i64),
    F64(f64),
    Str(String),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Table {
    pub columns: Vec<ColumnSpec>,
    pub rows: Vec<Vec<Cell>>,
    pub metadata: Vec<(String, String)>,
}

// ---- Thrift compact protocol: writer ----

#[derive(Default)]
struct Tw {
    buf: Vec<u8>,
    last: Vec<i16>,
}

fn zz32(v: i32) -> u64 {
    u64::from(((v << 1) ^ (v >> 31)) as u32)
}

fn zz64(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}

impl Tw {
    fn new() -> Tw {
        Tw {
            buf: Vec::new(),
            last: vec![0],
        }
    }
    fn varint(&mut self, mut v: u64) {
        loop {
            let b = (v & 0x7f) as u8;
            v >>= 7;
            if v == 0 {
                self.buf.push(b);
                return;
            }
            self.buf.push(b | 0x80);
        }
    }
    fn field(&mut self, id: i16, ty: u8) {
        let last = self.last.last_mut().expect("struct context");
        let delta = id - *last;
        if (1..=15).contains(&delta) {
            self.buf.push(((delta as u8) << 4) | ty);
        } else {
            self.buf.push(ty);
            let z = zz32(i32::from(id));
            self.varint(z);
        }
        *self.last.last_mut().expect("struct context") = id;
    }
    fn i32f(&mut self, id: i16, v: i32) {
        self.field(id, 5);
        self.varint(zz32(v));
    }
    fn i64f(&mut self, id: i16, v: i64) {
        self.field(id, 6);
        self.varint(zz64(v));
    }
    fn binf(&mut self, id: i16, b: &[u8]) {
        self.field(id, 8);
        self.varint(b.len() as u64);
        self.buf.extend_from_slice(b);
    }
    fn struct_begin(&mut self, id: i16) {
        self.field(id, 12);
        self.last.push(0);
    }
    fn struct_end(&mut self) {
        self.buf.push(0);
        self.last.pop();
    }
    fn list_begin(&mut self, id: i16, elem: u8, n: usize) {
        self.field(id, 9);
        if n < 15 {
            self.buf.push(((n as u8) << 4) | elem);
        } else {
            self.buf.push(0xf0 | elem);
            self.varint(n as u64);
        }
    }
    fn elem_struct_begin(&mut self) {
        self.last.push(0);
    }
    fn elem_i32(&mut self, v: i32) {
        self.varint(zz32(v));
    }
    fn elem_bin(&mut self, b: &[u8]) {
        self.varint(b.len() as u64);
        self.buf.extend_from_slice(b);
    }
    fn finish(mut self) -> Vec<u8> {
        self.buf.push(0);
        self.buf
    }
}

// ---- encoders ----

fn bitpack_bools(values: &[bool]) -> Vec<u8> {
    let mut out = vec![0u8; values.len().div_ceil(8)];
    for (i, v) in values.iter().enumerate() {
        if *v {
            out[i / 8] |= 1 << (i % 8);
        }
    }
    out
}

/// Definition levels (bit width 1) as one bit-packed run, with the v1 4-byte length prefix.
fn def_levels(defined: &[bool]) -> Vec<u8> {
    let mut run = Tw::new();
    let groups = defined.len().div_ceil(8) as u64;
    run.varint((groups << 1) | 1);
    let mut body = run.buf;
    body.extend(bitpack_bools(defined));
    let mut out = (body.len() as u32).to_le_bytes().to_vec();
    out.extend(body);
    out
}

fn plain(ty: ColumnType, cells: &[&Cell]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    match ty {
        ColumnType::Boolean => {
            let vals: Vec<bool> = cells
                .iter()
                .map(|c| match c {
                    Cell::Bool(b) => Ok(*b),
                    other => bail!("boolean column holds {other:?}"),
                })
                .collect::<Result<_>>()?;
            out = bitpack_bools(&vals);
        }
        ColumnType::Int64 => {
            for c in cells {
                let Cell::I64(v) = c else {
                    bail!("int64 column holds {c:?}")
                };
                out.extend(v.to_le_bytes());
            }
        }
        ColumnType::Double => {
            for c in cells {
                let Cell::F64(v) = c else {
                    bail!("double column holds {c:?}")
                };
                out.extend(v.to_le_bytes());
            }
        }
        ColumnType::Utf8 => {
            for c in cells {
                let Cell::Str(s) = c else {
                    bail!("utf8 column holds {c:?}")
                };
                out.extend((u32::try_from(s.len())?).to_le_bytes());
                out.extend(s.as_bytes());
            }
        }
    }
    Ok(out)
}

/// Writes `table` as Parquet. `zstd` selects ZSTD page compression, else UNCOMPRESSED.
pub fn write(table: &Table, zstd: bool, created_by: &str) -> Result<Vec<u8>> {
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
    let mut out = MAGIC.to_vec();
    struct Meta {
        offset: i64,
        uncompressed: i64,
        compressed: i64,
    }
    let mut metas = Vec::with_capacity(ncol);
    for (ci, col) in table.columns.iter().enumerate() {
        let cells: Vec<&Cell> = table.rows.iter().map(|r| &r[ci]).collect();
        let defined: Vec<bool> = cells.iter().map(|c| !matches!(c, Cell::Null)).collect();
        if col.required {
            ensure!(
                defined.iter().all(|d| *d),
                "required column {} holds a null",
                col.name
            );
        }
        let present: Vec<&Cell> = cells
            .iter()
            .copied()
            .filter(|c| !matches!(c, Cell::Null))
            .collect();
        let mut page = if col.required {
            Vec::new()
        } else {
            def_levels(&defined)
        };
        page.extend(plain(col.ty, &present).with_context(|| format!("column {}", col.name))?);
        let body = if zstd {
            jope_records::zstd_frame::compress(&page)
        } else {
            page.clone()
        };
        let mut h = Tw::new();
        h.i32f(1, 0); // DATA_PAGE
        h.i32f(2, i32::try_from(page.len())?);
        h.i32f(3, i32::try_from(body.len())?);
        h.struct_begin(5);
        h.i32f(1, i32::try_from(nrows)?);
        h.i32f(2, 0); // PLAIN
        h.i32f(3, 3); // RLE definition levels
        h.i32f(4, 3); // RLE repetition levels
        h.struct_end();
        let header = h.finish();
        let offset = out.len() as i64;
        out.extend(&header);
        out.extend(&body);
        metas.push(Meta {
            offset,
            uncompressed: (header.len() + page.len()) as i64,
            compressed: (header.len() + body.len()) as i64,
        });
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
        if col.ty == ColumnType::Utf8 {
            f.i32f(6, 0); // ConvertedType UTF8
        }
        f.struct_end();
    }
    f.i64f(3, nrows as i64);
    f.list_begin(4, 12, 1);
    f.elem_struct_begin();
    f.list_begin(1, 12, ncol);
    for (col, m) in table.columns.iter().zip(&metas) {
        f.elem_struct_begin();
        f.i64f(2, m.offset);
        f.struct_begin(3);
        f.i32f(1, col.ty.physical());
        f.list_begin(2, 5, 2);
        f.elem_i32(0);
        f.elem_i32(3);
        f.list_begin(3, 8, 1);
        f.elem_bin(col.name.as_bytes());
        f.i32f(4, if zstd { 6 } else { 0 });
        f.i64f(5, nrows as i64);
        f.i64f(6, m.uncompressed);
        f.i64f(7, m.compressed);
        f.i64f(9, m.offset);
        f.struct_end();
        f.struct_end();
    }
    f.i64f(2, metas.iter().map(|m| m.uncompressed).sum());
    f.i64f(3, nrows as i64);
    f.struct_end();
    f.list_begin(5, 12, table.metadata.len());
    for (k, v) in &table.metadata {
        f.elem_struct_begin();
        f.binf(1, k.as_bytes());
        f.binf(2, v.as_bytes());
        f.struct_end();
    }
    f.binf(6, created_by.as_bytes());
    let footer = f.finish();
    out.extend(&footer);
    out.extend((u32::try_from(footer.len())?).to_le_bytes());
    out.extend(MAGIC);
    Ok(out)
}

// ---- Thrift compact protocol: reader ----

#[derive(Clone, Debug)]
enum Tv {
    /// A boolean or double the reader consumes but does not use.
    Other,
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
            1 | 2 => Tv::Other,
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
    ensure!(data.len() >= 4, "parquet: definition levels truncated");
    let len = u32::from_le_bytes(data[..4].try_into()?) as usize;
    ensure!(
        4 + len <= data.len(),
        "parquet: definition levels run past the page"
    );
    let mut r = Tr {
        b: &data[4..4 + len],
        p: 0,
        depth: 0,
    };
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let h = r.varint()?;
        if h & 1 == 1 {
            let groups = usize::try_from(h >> 1)?;
            for byte in r.bytes(groups)? {
                for bit in 0..8 {
                    out.push(byte >> bit & 1 == 1);
                }
            }
        } else {
            let run = usize::try_from(h >> 1)?;
            let v = r.byte()? & 1 == 1;
            out.extend(std::iter::repeat_n(v, run));
        }
    }
    out.truncate(n);
    Ok((out, 4 + len))
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
        ColumnType::Int64 => {
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

/// Reads a file written by [`write`] (one row group, PLAIN, UNCOMPRESSED or ZSTD pages).
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
