use std::collections::HashSet;
use std::path::Path;

use crate::py;
use crate::pyjson::{self, Dict, Value};
use crate::{require, Error, Result};

/// Byte width of each supported dtype.
fn dtype_bytes(dtype: &Value) -> Option<i128> {
    (dtype.as_str() == Some("F32")).then_some(4)
}

/// A parsed safetensors object: its header (without `__metadata__`) and payload.
pub struct Safetensors<'a> {
    pub header: Dict,
    pub payload: &'a [u8],
}

/// `(begin, end)` of an entry's `data_offsets`, as Python's two-name unpacking read it.
pub(crate) fn offsets(name: &str, entry: &Value) -> Result<(i128, i128)> {
    let items = entry
        .get("data_offsets")
        .and_then(py::iterate)
        .ok_or_else(|| Error::Shape(format!("{name}: data_offsets is not a pair")))?;
    require!(
        items.len() <= 2,
        "too many values to unpack (expected 2, got {})",
        items.len()
    );
    require!(
        items.len() == 2,
        "not enough values to unpack (expected 2, got {})",
        items.len()
    );
    let int = |v: &Value| {
        v.as_int()
            .map(i128::from)
            .ok_or_else(|| Error::Shape(format!("{name}: data_offsets must be integers")))
    };
    Ok((int(&items[0])?, int(&items[1])?))
}

/// `math.prod(shape)`; `None` when the product leaves the representable range.
pub(crate) fn product(name: &str, shape: &Value) -> Result<Option<i128>> {
    let dims = shape
        .as_list()
        .ok_or_else(|| Error::Shape(format!("{name}: shape is not a list")))?;
    let mut product: i128 = 1;
    for dim in dims {
        let dim = dim
            .as_int()
            .ok_or_else(|| Error::Shape(format!("{name}: shape entries must be integers")))?;
        match product.checked_mul(i128::from(dim)) {
            Some(p) => product = p,
            None => return Ok(None),
        }
    }
    Ok(Some(product))
}

/// Header and payload; every byte of the payload belongs to exactly one tensor.
pub fn parse_safetensors(raw: &[u8]) -> Result<Safetensors<'_>> {
    require!(raw.len() >= 8, "truncated header length");
    let size = u64::from_le_bytes(raw[..8].try_into().expect("eight bytes"));
    let end = usize::try_from(size).ok().and_then(|s| s.checked_add(8));
    require!(end.is_some_and(|end| end <= raw.len()), "truncated header");
    let end = end.expect("checked above");
    let header = pyjson::parse_bytes(&raw[8..end])
        .map_err(|message| Error::parse(Path::new("<safetensors header>"), message))?;
    let Value::Dict(mut header) = header else {
        return Err(Error::Shape("safetensors header is not an object".into()));
    };
    header.remove("__metadata__");
    let payload = &raw[end..];
    let length = payload.len() as i128;
    let fields: HashSet<&str> = ["dtype", "shape", "data_offsets"].into();
    let mut spans: Vec<(i128, i128, String)> = Vec::new();
    for (name, entry) in header.iter() {
        require!(
            entry
                .as_dict()
                .is_some_and(|e| e.len() == fields.len() && e.keys().all(|k| fields.contains(k))),
            "{name}: header fields"
        );
        let dtype = entry.get("dtype").expect("fields checked");
        let width = dtype_bytes(dtype);
        require!(width.is_some(), "{name}: dtype {}", dtype.py_str());
        let (begin, end) = offsets(name, entry)?;
        require!(
            0 <= begin && begin <= end && end <= length,
            "{name}: offsets outside payload"
        );
        let count = product(name, entry.get("shape").expect("fields checked"))?;
        require!(
            count.and_then(|c| c.checked_mul(width.expect("checked above"))) == Some(end - begin),
            "{name}: byte length differs from shape"
        );
        spans.push((begin, end, name.to_owned()));
    }
    spans.sort();
    let mut cursor = 0;
    for (begin, end, name) in &spans {
        require!(
            *begin == cursor,
            "{name}: payload gap or overlap at byte {begin}"
        );
        cursor = *end;
    }
    require!(cursor == length, "trailing payload bytes");
    Ok(Safetensors { header, payload })
}
