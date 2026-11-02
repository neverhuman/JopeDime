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
        ColumnType::Int64 | ColumnType::TimestampMillis => {
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
