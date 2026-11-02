use super::*;

pub(super) fn refuse<T>(why: impl Into<String>) -> Result<T> {
    Err(Error::Invalid(format!(
        "{}\nThe committed benchmark evidence disagrees with itself; refusing to publish.",
        why.into()
    )))
}

pub(super) fn shape(what: &str) -> Error {
    Error::Shape(format!("{what} missing"))
}

/// `value[key]`: a missing key (Python's `KeyError`) or a value that is not a dict (its
/// `TypeError`) refuses, naming the key and, for a value that is not a dict, its type.
pub(super) fn at<'a>(value: &'a Value, key: &str) -> Result<&'a Value> {
    match value {
        Value::Dict(dict) => dict.get(key).ok_or_else(|| shape(&pyrepr_key(key))),
        other => Err(raised(
            pyrepr_key(key),
            py::exception::not_subscriptable(other),
        )),
    }
}

pub(super) fn pyrepr_key(key: &str) -> String {
    crate::pyjson::str_repr(key)
}

pub(super) fn get<'a>(value: &'a Value, key: &str) -> &'a Value {
    static NULL: Value = Value::Null;
    value.get(key).unwrap_or(&NULL)
}

/// `str(value)` as an f-string prints it.
pub(super) fn s(value: &Value) -> String {
    value.py_str()
}

pub(super) fn cell<'a>(row: &'a Dict, key: &str) -> Result<&'a str> {
    row.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| shape(&pyrepr_key(key)))
}

/// `int(row[key])` for a CSV cell through the strict number grammar ([`py::int`]): text that is
/// not an integer refuses (Python's `ValueError`), and so does one of more than 4300 digits,
/// leading zeros counted, in CPython's words. Underscores, Unicode digits and Unicode whitespace,
/// which Python read, are refused too (stricter), as are U+001C..U+001F around the digits, which
/// Python refused although `str.strip()` removes them.
pub(super) fn int_cell(row: &Dict, key: &str) -> Result<i64> {
    let text = cell(row, key)?;
    py::int(text).map_err(|e| match e {
        py::IntError::Limit(message) => raised(format_args!("CSV column {key}"), message),
        py::IntError::Invalid => Error::Shape(format!(
            "CSV column {key}: expected an integer, found {}",
            pyrepr_key(text)
        )),
    })
}

/// `float(row[key])` for a CSV cell through the strict number grammar ([`py::float`]); text that
/// is not a number refuses (Python's `ValueError`). `nan`, `inf`, underscores, Unicode digits and
/// Unicode whitespace, which Python read, are refused too (stricter; no committed ladder writes
/// them).
pub(super) fn float_cell(row: &Dict, key: &str) -> Result<f64> {
    let text = cell(row, key)?;
    py::float(text).ok_or_else(|| {
        Error::Shape(format!(
            "CSV column {key}: expected a number, found {}",
            pyrepr_key(text)
        ))
    })
}

/// A JSON number formatted with a float format (`f"{value:.2f}"`, which converts an `int` to the
/// nearest double); any other type refuses (the Python raised `ValueError` or `TypeError` in
/// `format`).
pub(super) fn float_of(value: &Value, what: &str) -> Result<f64> {
    Ok(num_of(value, what)?.as_f64())
}

/// `f"{valid['rankable_rows']:,}"`: an integer. A float refuses, which is stricter than the
/// Python (it printed `format(8718.0, ",")`, `8,718.0`); any other type refuses as it did.
pub(super) fn rankable_rows(valid: &Value) -> Result<i64> {
    let value = at(valid, "rankable_rows")?;
    value.as_int().ok_or_else(|| {
        Error::Shape(format!(
            "{VALID_SNAPSHOT} rankable_rows: expected an integer, found {} ({})",
            value.repr(),
            py::type_name(value)
        ))
    })
}

/// Malformed evidence the Python crashed on (a traceback, exit status 1): the field, then what
/// could not be done with it and its type (see [`py::exception`]; the text is not CPython's).
pub(super) fn raised(context: impl std::fmt::Display, exception: String) -> Error {
    Error::Shape(format!("{context}: {exception}"))
}

/// `value[key]` (a string key); refuses where the Python raised (missing key, not a dict).
pub(super) fn sub<'a>(value: &'a Value, key: &str, context: &str) -> Result<&'a Value> {
    py::subscript(value, key).map_err(|e| raised(context, e))
}

/// `for x in value`: the items (a list's, a dict's keys, a string's characters), or Python's
/// `TypeError`.
pub(super) fn items(value: &Value, context: &str) -> Result<Vec<Value>> {
    py::iterate(value).ok_or_else(|| raised(context, py::exception::not_iterable(value)))
}

/// `value.get(key)` on what the Python used as a dict (an `AttributeError` otherwise).
pub(super) fn dict_get<'a>(
    value: &'a Value,
    key: &str,
    context: &str,
) -> Result<Option<&'a Value>> {
    match value {
        Value::Dict(dict) => Ok(dict.get(key)),
        other => Err(raised(context, py::exception::no_attribute(other, "get"))),
    }
}

/// `f"{value[:9]}"`: a string's first nine characters, or a list's first nine items.
pub(super) fn first_nine(value: &Value, context: &str) -> Result<String> {
    match value {
        Value::Str(text) => Ok(text.chars().take(9).collect()),
        Value::List(list) => Ok(Value::List(list.iter().take(9).cloned().collect()).repr()),
        Value::Dict(_) => Err(Error::Shape(format!(
            "{context}: expected a string, found {}",
            value.repr()
        ))),
        other => Err(raised(
            context,
            format!(
                "TypeError: '{}' object is not subscriptable",
                py::type_name(other)
            ),
        )),
    }
}

/// `new < current` for two method ids that tie on RMSE, as the tuple comparison inside `min`
/// made it: equal ids are not less; strings compare, and numbers compare exactly ([`py_cmp`]: an
/// int and a float that round to the same double are still ordered); anything else raised.
pub(super) fn id_less(new: &Value, current: &Value) -> Result<bool> {
    if new == current {
        return Ok(false);
    }
    match (new, current) {
        (Value::Str(a), Value::Str(b)) => Ok(a < b),
        _ => match (Num::of(new), Num::of(current)) {
            (Some(a), Some(b)) => Ok(py_lt(a, b)),
            _ => Err(raised(
                format_args!("{SAMECUT} method id"),
                format!(
                    "TypeError: '<' not supported between instances of '{}' and '{}'",
                    py::type_name(new),
                    py::type_name(current)
                ),
            )),
        },
    }
}

/// A dict built as `{key: value for ...}`: a repeated key keeps its first position (and its first
/// key object) and takes the last value; keys compare as Python hashes them.
pub(super) type PyDict = Vec<(py::Key, Value, Value)>;

pub(super) fn py_insert(dict: &mut PyDict, key: &Value, value: Value, context: &str) -> Result<()> {
    let hashed = py::Key::of(key).ok_or_else(|| raised(context, py::exception::unhashable(key)))?;
    match dict.iter_mut().find(|(k, _, _)| *k == hashed) {
        Some(entry) => entry.2 = value,
        None => dict.push((hashed, key.clone(), value)),
    }
    Ok(())
}

pub(super) fn py_lookup<'a>(dict: &'a PyDict, key: &Value) -> Option<&'a Value> {
    let hashed = py::Key::of(key)?;
    dict.iter()
        .find(|(k, _, _)| *k == hashed)
        .map(|(_, _, v)| v)
}

/// `{a["id"]: a["rows"] for a in meta["applied"]}`.
pub(super) fn applied_rows(meta: &Value, context: &str) -> Result<PyDict> {
    let mut applied = PyDict::new();
    for a in items(
        sub(meta, "applied", context)?,
        &format!("{context} applied"),
    )? {
        let id = sub(&a, "id", &format!("{context} applied"))?;
        let rows = sub(&a, "rows", &format!("{context} applied"))?.clone();
        py_insert(&mut applied, id, rows, &format!("{context} applied id"))?;
    }
    Ok(applied)
}
