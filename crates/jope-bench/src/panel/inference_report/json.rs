use super::*;

// ---------------------------------------------------------------------------------------------
// Order-preserving JSON with Python's comparison rules.

/// A JSON value that keeps member order, as Python's `json` module does.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Json {
    Null,
    Bool(bool),
    Int(i128),
    Float(f64),
    Str(String),
    Array(Vec<Json>),
    Object(Object),
}

/// Members in first-insertion order; a repeated key keeps its first position and last value.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct Object(pub(super) Vec<(String, Json)>);

pub(super) static NULL: Json = Json::Null;
pub(super) static ZERO: Json = Json::Int(0);
pub(super) static EMPTY_ARRAY: Json = Json::Array(Vec::new());
pub(super) static EMPTY_OBJECT: Json = Json::Object(Object(Vec::new()));
pub(super) static EMPTY_STRING: Json = Json::Str(String::new());

impl Object {
    pub(super) fn get(&self, key: &str) -> Option<&Json> {
        self.0.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }
    pub(super) fn set(&mut self, key: &str, value: Json) {
        match self.0.iter_mut().find(|(k, _)| k == key) {
            Some(slot) => slot.1 = value,
            None => self.0.push((key.to_owned(), value)),
        }
    }
}

impl From<&str> for Json {
    fn from(value: &str) -> Self {
        Json::Str(value.to_owned())
    }
}
impl From<String> for Json {
    fn from(value: String) -> Self {
        Json::Str(value)
    }
}
impl From<bool> for Json {
    fn from(value: bool) -> Self {
        Json::Bool(value)
    }
}
impl From<u64> for Json {
    fn from(value: u64) -> Self {
        Json::Int(value.into())
    }
}
impl From<usize> for Json {
    fn from(value: usize) -> Self {
        Json::Int(value as i128)
    }
}

impl<'de> Deserialize<'de> for Json {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct JsonVisitor;
        impl<'de> Visitor<'de> for JsonVisitor {
            type Value = Json;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a JSON value")
            }
            fn visit_unit<E: serde::de::Error>(self) -> Result<Json, E> {
                Ok(Json::Null)
            }
            fn visit_bool<E: serde::de::Error>(self, v: bool) -> Result<Json, E> {
                Ok(Json::Bool(v))
            }
            fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<Json, E> {
                Ok(Json::Int(v.into()))
            }
            fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<Json, E> {
                Ok(Json::Int(v.into()))
            }
            fn visit_f64<E: serde::de::Error>(self, v: f64) -> Result<Json, E> {
                if v.is_finite() {
                    Ok(Json::Float(v))
                } else {
                    Err(E::custom("nonfinite number"))
                }
            }
            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Json, E> {
                Ok(Json::Str(v.to_owned()))
            }
            fn visit_string<E: serde::de::Error>(self, v: String) -> Result<Json, E> {
                Ok(Json::Str(v))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Json, A::Error> {
                let mut items = Vec::new();
                while let Some(item) = seq.next_element()? {
                    items.push(item);
                }
                Ok(Json::Array(items))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Json, A::Error> {
                let mut members = Object::default();
                while let Some((key, value)) = map.next_entry::<String, Json>()? {
                    members.set(&key, value);
                }
                Ok(Json::Object(members))
            }
        }
        deserializer.deserialize_any(JsonVisitor)
    }
}

pub(super) fn members<const N: usize>(entries: [(&str, Json); N]) -> Object {
    Object(
        entries
            .into_iter()
            .map(|(k, v)| (k.to_owned(), v))
            .collect(),
    )
}
pub(super) fn object<const N: usize>(entries: [(&str, Json); N]) -> Json {
    Json::Object(members(entries))
}

pub(super) fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// `json.loads` of a file, or an empty object when it cannot be read or parsed.
pub(super) fn read(path: &Path) -> Json {
    fs::read(path)
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or_else(|| EMPTY_OBJECT.clone())
}

pub(super) fn kind(value: &Json) -> &'static str {
    match value {
        Json::Null => "null",
        Json::Bool(_) => "bool",
        Json::Int(_) => "int",
        Json::Float(_) => "float",
        Json::Str(_) => "string",
        Json::Array(_) => "array",
        Json::Object(_) => "object",
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) enum JsonNumber {
    Int(i128),
    Float(f64),
}

impl JsonNumber {
    pub(super) fn to_f64(self) -> f64 {
        match self {
            JsonNumber::Int(i) => i as f64,
            JsonNumber::Float(f) => f,
        }
    }
    pub(super) fn is_finite(self) -> bool {
        match self {
            JsonNumber::Int(_) => true,
            JsonNumber::Float(f) => f.is_finite(),
        }
    }
}

/// `isinstance(value, (int, float))`; a bool is an int.
pub(super) fn number(value: &Json) -> Option<JsonNumber> {
    match value {
        Json::Bool(b) => Some(JsonNumber::Int(i128::from(*b))),
        Json::Int(i) => Some(JsonNumber::Int(*i)),
        Json::Float(f) => Some(JsonNumber::Float(*f)),
        _ => None,
    }
}

/// Exact numeric comparison across ints and floats, as Python compares them.
pub(super) fn compare(a: JsonNumber, b: JsonNumber) -> Option<Ordering> {
    match (a, b) {
        (JsonNumber::Int(x), JsonNumber::Int(y)) => Some(x.cmp(&y)),
        (JsonNumber::Float(x), JsonNumber::Float(y)) => x.partial_cmp(&y),
        (JsonNumber::Int(x), JsonNumber::Float(y)) => int_float(x, y),
        (JsonNumber::Float(x), JsonNumber::Int(y)) => int_float(y, x).map(Ordering::reverse),
    }
}
pub(super) fn int_float(i: i128, f: f64) -> Option<Ordering> {
    const LIMIT: f64 = 1.701_411_834_604_692_3e38; // 2^127
    if f.is_nan() {
        return None;
    }
    if f >= LIMIT {
        return Some(Ordering::Less);
    }
    if f < -LIMIT {
        return Some(Ordering::Greater);
    }
    let whole = f.trunc();
    Some(match i.cmp(&(whole as i128)) {
        Ordering::Equal if f > whole => Ordering::Less,
        Ordering::Equal if f < whole => Ordering::Greater,
        other => other,
    })
}
pub(super) fn at_least_zero(n: JsonNumber) -> bool {
    compare(n, JsonNumber::Int(0)).is_some_and(Ordering::is_ge)
}
pub(super) fn above_zero(n: JsonNumber) -> bool {
    compare(n, JsonNumber::Int(0)).is_some_and(Ordering::is_gt)
}

/// Python `==` between JSON values: numbers (and bools) by value, containers structurally,
/// objects regardless of member order.
pub(super) fn py_eq(a: &Json, b: &Json) -> bool {
    if let (Some(x), Some(y)) = (number(a), number(b)) {
        return compare(x, y) == Some(Ordering::Equal);
    }
    match (a, b) {
        (Json::Null, Json::Null) => true,
        (Json::Str(x), Json::Str(y)) => x == y,
        (Json::Array(x), Json::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| py_eq(p, q))
        }
        (Json::Object(x), Json::Object(y)) => {
            x.0.len() == y.0.len()
                && x.0
                    .iter()
                    .all(|(k, v)| y.get(k).is_some_and(|w| py_eq(v, w)))
        }
        _ => false,
    }
}
pub(super) fn is(value: &Json, text: &str) -> bool {
    matches!(value, Json::Str(s) if s == text)
}
/// `value is True`.
pub(super) fn is_true(value: Option<&Json>) -> bool {
    matches!(value, Some(Json::Bool(true)))
}
pub(super) fn truthy(value: &Json) -> bool {
    match value {
        Json::Null => false,
        Json::Bool(b) => *b,
        Json::Int(i) => *i != 0,
        Json::Float(f) => *f != 0.0,
        Json::Str(s) => !s.is_empty(),
        Json::Array(items) => !items.is_empty(),
        Json::Object(o) => !o.0.is_empty(),
    }
}
pub(super) fn py_len(value: &Json) -> Result<usize> {
    Ok(match value {
        Json::Str(s) => s.chars().count(),
        Json::Array(items) => items.len(),
        Json::Object(o) => o.0.len(),
        other => bail!("TypeError: a JSON {} has no len()", kind(other)),
    })
}
/// Python iteration: array items, object keys or string characters.
pub(super) fn py_iter(value: &Json) -> Result<Vec<Json>> {
    Ok(match value {
        Json::Str(s) => s.chars().map(|c| Json::Str(c.to_string())).collect(),
        Json::Array(items) => items.clone(),
        Json::Object(o) => o.0.iter().map(|(k, _)| Json::Str(k.clone())).collect(),
        other => bail!("TypeError: a JSON {} is not iterable", kind(other)),
    })
}
/// `needle in container`.
pub(super) fn py_contains(container: &Json, needle: &str) -> Result<bool> {
    Ok(match container {
        Json::Str(s) => s.contains(needle),
        Json::Array(items) => items.iter().any(|x| is(x, needle)),
        Json::Object(o) => o.get(needle).is_some(),
        other => bail!(
            "TypeError: argument of type {} is not iterable",
            kind(other)
        ),
    })
}
/// `all(needle in x for x in container)`.
pub(super) fn all_contain(container: &Json, needle: &str) -> Result<bool> {
    for item in py_iter(container)? {
        if !py_contains(&item, needle)? {
            return Ok(false);
        }
    }
    Ok(true)
}
pub(super) fn hashable(value: &Json) -> Result<()> {
    ensure!(
        !matches!(value, Json::Array(_) | Json::Object(_)),
        "TypeError: unhashable JSON {}",
        kind(value)
    );
    Ok(())
}
/// Distinct members of a Python set built from the values.
pub(super) fn distinct(values: Vec<Json>) -> Result<Vec<Json>> {
    let mut seen: Vec<Json> = Vec::new();
    for value in values {
        hashable(&value)?;
        if !seen.iter().any(|s| py_eq(s, &value)) {
            seen.push(value);
        }
    }
    Ok(seen)
}
/// Python `<` ordering where it is defined.
pub(super) fn py_order(a: &Json, b: &Json) -> Result<Ordering> {
    if let (Some(x), Some(y)) = (number(a), number(b)) {
        return compare(x, y).context("unordered numbers");
    }
    match (a, b) {
        (Json::Str(x), Json::Str(y)) => Ok(x.cmp(y)),
        (Json::Array(x), Json::Array(y)) => {
            for (p, q) in x.iter().zip(y) {
                if !py_eq(p, q) {
                    return py_order(p, q);
                }
            }
            Ok(x.len().cmp(&y.len()))
        }
        _ => bail!(
            "TypeError: '<' is not supported between JSON {} and {}",
            kind(a),
            kind(b)
        ),
    }
}

// ---------------------------------------------------------------------------------------------
// Member access with Python's failure modes.

/// `value.get(key)`: only an object has `get`; anything else fails the report.
pub(super) fn get<'a>(value: &'a Json, key: &str) -> Result<Option<&'a Json>> {
    match value {
        Json::Object(o) => Ok(o.get(key)),
        other => bail!("AttributeError: a JSON {} has no get({key:?})", kind(other)),
    }
}
/// `value.get(key, default)`.
pub(super) fn get_or<'a>(value: &'a Json, key: &str, default: &'a Json) -> Result<&'a Json> {
    Ok(get(value, key)?.unwrap_or(default))
}
/// `value.get(key)`, absent as `None`.
pub(super) fn at<'a>(value: &'a Json, key: &str) -> Result<&'a Json> {
    get_or(value, key, &NULL)
}
/// `value[key]`: a missing member or a non-object fails the report.
pub(super) fn index<'a>(value: &'a Json, key: &str) -> Result<&'a Json> {
    match value {
        Json::Object(o) => o.get(key).with_context(|| format!("KeyError: {key:?}")),
        other => bail!(
            "TypeError: a JSON {} is not subscriptable by {key:?}",
            kind(other)
        ),
    }
}
/// `value[key]` inside a `try` that treats KeyError and TypeError as absence.
pub(super) fn lookup<'a>(value: &'a Json, key: &str) -> Option<&'a Json> {
    match value {
        Json::Object(o) => o.get(key),
        _ => None,
    }
}
/// `value[key] = item` on a receipt.
pub(super) fn assign(value: &mut Json, key: &str, item: Json) -> Result<()> {
    match value {
        Json::Object(o) => {
            o.set(key, item);
            Ok(())
        }
        other => bail!(
            "TypeError: a JSON {} does not support item assignment",
            kind(other)
        ),
    }
}

// ---------------------------------------------------------------------------------------------
// Artifact identity and exact-bit parity.

/// A pin names a readable file whose SHA-256 matches; anything else is unverified.
pub(super) fn verified(pin: &Json) -> Result<bool> {
    let Some(Json::Str(path)) = lookup(pin, "path") else {
        return Ok(false);
    };
    ensure!(
        !path.contains('\0'),
        "ValueError: embedded null byte in {path:?}"
    );
    let Ok(bytes) = fs::read(path) else {
        return Ok(false);
    };
    Ok(
        matches!(lookup(pin, "sha256"), Some(Json::Str(expected)) if *expected == sha256_hex(&bytes)),
    )
}

/// The JSON behind an already verified pin, `read(Path(owner[key]['path']))`.
pub(super) fn pinned(owner: &Json, key: &str) -> Result<Json> {
    match lookup(index(owner, key)?, "path") {
        Some(Json::Str(path)) => Ok(read(Path::new(path))),
        _ => bail!("TypeError: pin {key:?} has no path"),
    }
}

/// `struct.pack('>d', value)`: the IEEE-754 bits of a number; anything else fails the report.
pub(super) fn pack(value: &Json) -> Result<u64> {
    match number(value) {
        Some(JsonNumber::Int(i)) => Ok((i as f64).to_bits()),
        Some(JsonNumber::Float(f)) => Ok(f.to_bits()),
        None => bail!(
            "struct.error: required argument is not a float (JSON {})",
            kind(value)
        ),
    }
}

/// A backend's predictions equal its qualification plan's expected values bit for bit, and
/// the plan is bound to the pinned bundle and its backend.
pub(super) fn parity_verified(b: &Json) -> Result<bool> {
    if !verified(get_or(b, "qualification_plan", &EMPTY_OBJECT)?)?
        || !verified(get_or(b, "prediction", &EMPTY_OBJECT)?)?
    {
        return Ok(false);
    }
    let q = pinned(b, "qualification_plan")?;
    let actual = pinned(b, "prediction")?;
    let expected = get_or(&q, "expected", &EMPTY_OBJECT)?.clone();
    if !verified(get_or(b, "bundle", &EMPTY_OBJECT)?)? {
        return Ok(false);
    }
    let bundle = pinned(b, "bundle")?;
    let identity = Json::Str(sha256_hex(dumps(&bundle, Style::IDENTITY)?.as_bytes()));
    if !py_eq(at(&q, "bundle_sha256")?, &identity) {
        return Ok(false);
    }
    let backend = at(&q, "backend")?;
    if !py_eq(
        backend,
        at(get_or(&bundle, "manifest", &EMPTY_OBJECT)?, "backend")?,
    ) {
        return Ok(false);
    }
    // KeyError and TypeError in this comparison mean "not verified".
    let (Some(kind_actual), Some(kind_expected)) =
        (lookup(&actual, "kind"), lookup(&expected, "kind"))
    else {
        return Ok(false);
    };
    if !(py_eq(kind_actual, kind_expected) && is(kind_expected, "values")) {
        return Ok(false);
    }
    let (Some(data_actual), Some(data_expected)) =
        (lookup(&actual, "data"), lookup(&expected, "data"))
    else {
        return Ok(false);
    };
    let (Ok(n_actual), Ok(n_expected)) = (py_len(data_actual), py_len(data_expected)) else {
        return Ok(false);
    };
    if n_actual != n_expected || n_actual == 0 {
        return Ok(false);
    }
    for (a, e) in py_iter(data_actual)?
        .iter()
        .zip(py_iter(data_expected)?.iter())
    {
        let bits = pack(a)?;
        if bits != pack(e)? {
            return Ok(false);
        }
    }
    Ok(true)
}
