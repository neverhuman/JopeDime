use super::*;

/// A hashable JSON scalar under Python's hash equality (`1 == 1.0 == True`).
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Key {
    None,
    Int(i64),
    /// An integer beyond `i64` (a [`Value::BigInt`], or a float equal to one), as its decimal.
    Big(String),
    Float(u64),
    Str(String),
}

impl Key {
    /// `None` for an unhashable value (a list or a dict).
    pub fn of(value: &Value) -> Option<Key> {
        Some(match value {
            Value::Null => Key::None,
            Value::Bool(b) => Key::Int(i64::from(*b)),
            Value::Int(i) => Key::Int(*i),
            Value::BigInt(i) => Key::Big(i.clone()),
            Value::Float(f) => {
                if value == &Value::Int(*f as i64) {
                    Key::Int(*f as i64)
                } else if let Some(integer) = pyjson::integral_float_text(*f) {
                    Key::Big(integer)
                } else {
                    Key::Float(f.to_bits())
                }
            }
            Value::Str(s) => Key::Str(s.clone()),
            Value::List(_) | Value::Dict(_) => return None,
        })
    }

    pub fn str(s: &str) -> Key {
        Key::Str(s.to_owned())
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Key::Str(s) => Some(s),
            _ => None,
        }
    }
}

/// Python `iter(value)` as a list of values: a list's items, a dict's keys, a string's
/// characters. `None` when Python would raise `TypeError`.
pub fn iterate(value: &Value) -> Option<Vec<Value>> {
    match value {
        Value::List(list) => Some(list.clone()),
        Value::Dict(dict) => Some(dict.keys().map(Value::from).collect()),
        Value::Str(s) => Some(s.chars().map(|c| Value::Str(c.to_string())).collect()),
        _ => None,
    }
}

/// `set(value)` as hash keys; `None` when not iterable or an item is unhashable.
pub fn key_set(value: &Value) -> Option<std::collections::HashSet<Key>> {
    iterate(value)?.iter().map(Key::of).collect()
}

/// Python `item in container`; `None` when Python would raise `TypeError`. A list is searched
/// with identity first ([`pyjson::eq_or_identical`]), so a JSON `NaN` is in a list holding one.
pub fn contains(container: &Value, item: &Value) -> Option<bool> {
    match container {
        Value::List(list) => Some(list.iter().any(|x| pyjson::eq_or_identical(x, item))),
        Value::Dict(dict) => {
            let key = Key::of(item)?;
            Some(key.as_str().is_some_and(|k| dict.contains_key(k)))
        }
        Value::Str(s) => item.as_str().map(|needle| s.contains(needle)),
        _ => None,
    }
}

/// Python `item in container`, or the refusal for the pair Python could not search (see
/// [`exception::membership`]): which of the two values was the wrong type, and what it was.
pub fn contains_or_raise(container: &Value, item: &Value) -> std::result::Result<bool, String> {
    contains(container, item).ok_or_else(|| exception::membership(container, item))
}

/// `type(value).__name__` for a JSON value.
pub fn type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Int(_) | Value::BigInt(_) => "int",
        Value::Float(_) => "float",
        Value::Str(_) => "str",
        Value::List(_) => "list",
        Value::Dict(_) => "dict",
    }
}

/// Refusals for the malformed values a retired Python tool crashed on (a traceback and exit
/// status 1, no report). A port that meets such a value refuses instead of skipping it or carrying
/// on with a default. The caller prefixes the field; each message names the operation that could
/// not be applied and the offending value's type. The wording is modelled on the exception
/// CPython raised so a reader can recognise the case, but it is not a copy of CPython's text and
/// no test compares it with CPython's: the negative parity fixtures assert only that both sides
/// refuse (exit status 1, nothing printed on stdout).
pub mod exception {
    use super::type_name;
    use crate::pyjson::{str_repr, Value};

    /// `item in container` where Python could not search `container` for `item`: a dict and an
    /// unhashable item, a string and an item that is not a string, or a container that is neither
    /// a list, a dict nor a string.
    pub fn membership(container: &Value, item: &Value) -> String {
        match container {
            Value::Dict(_) => unhashable(item),
            Value::Str(_) => format!(
                "TypeError: 'in <string>' requires string as left operand, not {}",
                type_name(item)
            ),
            other => not_container(other),
        }
    }

    /// `Path(value)` on a value that is not a string.
    pub fn path_argument(value: &Value) -> String {
        format!(
            "TypeError: a path must be a string, not '{}'",
            type_name(value)
        )
    }

    /// `os.path.*(value)` (`os.fspath`) on a value that is not a string.
    pub fn fspath(value: &Value) -> String {
        format!(
            "TypeError: expected str, bytes or os.PathLike object, not {}",
            type_name(value)
        )
    }

    /// `for x in value`, `set(value)` or `s.intersection(value)` on a value that is not iterable.
    pub fn not_iterable(value: &Value) -> String {
        format!("TypeError: '{}' object is not iterable", type_name(value))
    }

    /// `item in value` on a value that is not a container.
    pub fn not_container(value: &Value) -> String {
        format!(
            "TypeError: argument of type '{}' is not iterable",
            type_name(value)
        )
    }

    /// Hashing a list or a dict (set membership, dict lookup).
    pub fn unhashable(value: &Value) -> String {
        format!("TypeError: unhashable type: '{}'", type_name(value))
    }

    /// `value.<attribute>` on a type that has no such attribute.
    pub fn no_attribute(value: &Value, attribute: &str) -> String {
        format!(
            "AttributeError: '{}' object has no attribute '{attribute}'",
            type_name(value)
        )
    }

    /// `d[key]` with the key missing.
    pub fn key_error(key: &str) -> String {
        format!("KeyError: {}", str_repr(key))
    }

    /// `value[key]` with a string key on a value that is not a dict.
    pub fn not_subscriptable(value: &Value) -> String {
        match value {
            Value::Str(_) => "TypeError: string indices must be integers, not 'str'".to_owned(),
            Value::List(_) => {
                "TypeError: list indices must be integers or slices, not str".to_owned()
            }
            other => format!(
                "TypeError: '{}' object is not subscriptable",
                type_name(other)
            ),
        }
    }
}

/// Python `separator.join(value)`: the items `for x in value` yields, joined, or the exception
/// line Python raised (a non-iterable, or an item that is not a string).
pub fn join(separator: &str, value: &Value) -> std::result::Result<String, String> {
    let items = iterate(value).ok_or_else(|| "TypeError: can only join an iterable".to_owned())?;
    let mut parts = Vec::with_capacity(items.len());
    for (i, item) in items.iter().enumerate() {
        match item.as_str() {
            Some(text) => parts.push(text.to_owned()),
            None => {
                return Err(format!(
                    "TypeError: sequence item {i}: expected str instance, {} found",
                    type_name(item)
                ))
            }
        }
    }
    Ok(parts.join(separator))
}

/// Python `value[0]`: the first item (a list's, or a string's first character), or the exception
/// line Python raised.
pub fn first_item(value: &Value) -> std::result::Result<Value, String> {
    match value {
        Value::List(list) => list
            .first()
            .cloned()
            .ok_or_else(|| "IndexError: list index out of range".to_owned()),
        Value::Str(text) => text
            .chars()
            .next()
            .map(|c| Value::from(c.to_string()))
            .ok_or_else(|| "IndexError: string index out of range".to_owned()),
        Value::Dict(_) => Err("KeyError: 0".to_owned()),
        other => Err(format!(
            "TypeError: '{}' object is not subscriptable",
            type_name(other)
        )),
    }
}

/// Python `value[key]` for a string key: the item, or the refusal where Python raised (a missing
/// key, or a value that is not a dict).
pub fn subscript<'a>(value: &'a Value, key: &str) -> std::result::Result<&'a Value, String> {
    match value {
        Value::Dict(dict) => dict.get(key).ok_or_else(|| exception::key_error(key)),
        other => Err(exception::not_subscriptable(other)),
    }
}

/// Python `repr` of a tuple of values.
pub fn tuple_repr(items: &[Value]) -> String {
    let parts: Vec<String> = items.iter().map(Value::repr).collect();
    if parts.len() == 1 {
        format!("({},)", parts[0])
    } else {
        format!("({})", parts.join(", "))
    }
}

/// Python `repr` of a list of strings (as `sorted(...)[:n]` printed them).
pub fn str_list_repr<S: AsRef<str>>(items: &[S]) -> String {
    let parts: Vec<String> = items.iter().map(|s| pyjson::str_repr(s.as_ref())).collect();
    format!("[{}]", parts.join(", "))
}
