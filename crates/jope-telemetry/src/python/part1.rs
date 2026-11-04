// ---- JSON bridging -------------------------------------------------------------------------

/// `json.dumps(obj, allow_nan=False, separators=(",", ":"))` in Rust: dicts keep their order,
/// non-finite floats and unsupported types raise like `json` does.
fn write_json(obj: &Bound<'_, PyAny>, out: &mut String, depth: u32) -> PyResult<()> {
    if depth > MAX_JSON_DEPTH {
        return Err(value_error("Circular reference detected"));
    }
    if obj.is_none() {
        out.push_str("null");
    } else if let Ok(b) = obj.cast::<PyBool>() {
        out.push_str(if b.is_true() { "true" } else { "false" });
    } else if obj.cast::<PyInt>().is_ok() {
        out.push_str(&obj.str()?.to_cow()?);
    } else if let Ok(f) = obj.cast::<PyFloat>() {
        let v = f.value();
        if !v.is_finite() {
            return Err(value_error(format!(
                "Out of range float values are not JSON compliant: {v}"
            )));
        }
        out.push_str(&serde_json::to_string(&v).map_err(value_error)?);
    } else if let Ok(s) = obj.cast::<PyString>() {
        out.push_str(&serde_json::to_string(&*s.to_cow()?).map_err(value_error)?);
    } else if let Ok(d) = obj.cast::<PyDict>() {
        out.push('{');
        for (i, (k, v)) in d.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            let key = if let Ok(s) = k.cast::<PyString>() {
                s.to_cow()?.into_owned()
            } else if k.is_none() {
                "null".to_owned()
            } else if let Ok(b) = k.cast::<PyBool>() {
                (if b.is_true() { "true" } else { "false" }).to_owned()
            } else if k.cast::<PyInt>().is_ok() || k.cast::<PyFloat>().is_ok() {
                k.str()?.to_cow()?.into_owned()
            } else {
                return Err(PyTypeError::new_err(format!(
                    "keys must be str, int, float, bool or None, not {}",
                    k.get_type().name()?
                )));
            };
            out.push_str(&serde_json::to_string(&key).map_err(value_error)?);
            out.push(':');
            write_json(&v, out, depth + 1)?;
        }
        out.push('}');
    } else if obj.cast::<PyList>().is_ok() || obj.cast::<PyTuple>().is_ok() {
        out.push('[');
        for (i, item) in obj.try_iter()?.enumerate() {
            if i > 0 {
                out.push(',');
            }
            write_json(&item?, out, depth + 1)?;
        }
        out.push(']');
    } else {
        return Err(PyTypeError::new_err(format!(
            "Object of type {} is not JSON serializable",
            obj.get_type().name()?
        )));
    }
    Ok(())
}

fn to_json(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    let mut out = String::new();
    write_json(obj, &mut out, 0)?;
    Ok(out)
}

/// `json.dumps(obj, sort_keys=True)` with the default separators, `repr` floats and `NaN`: the
/// spelling `live-metrics.jsonl` had, for the fields a live row passes through.
fn write_py_json(obj: &Bound<'_, PyAny>, out: &mut String, depth: u32) -> PyResult<()> {
    if depth > MAX_JSON_DEPTH {
        return Err(value_error("Circular reference detected"));
    }
    if obj.is_none() {
        out.push_str("null");
    } else if let Ok(b) = obj.cast::<PyBool>() {
        out.push_str(if b.is_true() { "true" } else { "false" });
    } else if obj.cast::<PyInt>().is_ok() {
        out.push_str(&obj.str()?.to_cow()?);
    } else if let Ok(f) = obj.cast::<PyFloat>() {
        out.push_str(&live::py_float(f.value()));
    } else if let Ok(s) = obj.cast::<PyString>() {
        out.push_str(&live::py_str(&s.to_cow()?));
    } else if let Ok(d) = obj.cast::<PyDict>() {
        let mut items = Vec::with_capacity(d.len());
        for (k, v) in d.iter() {
            let key = k
                .cast::<PyString>()
                .map_err(|_| PyTypeError::new_err("live metrics dict keys must be str"))?
                .to_cow()?
                .into_owned();
            items.push((key, v));
        }
        items.sort_by(|a, b| a.0.cmp(&b.0));
        out.push('{');
        for (i, (k, v)) in items.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            out.push_str(&live::py_str(k));
            out.push_str(": ");
            write_py_json(v, out, depth + 1)?;
        }
        out.push('}');
    } else if obj.cast::<PyList>().is_ok() || obj.cast::<PyTuple>().is_ok() {
        out.push('[');
        for (i, item) in obj.try_iter()?.enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            write_py_json(&item?, out, depth + 1)?;
        }
        out.push(']');
    } else {
        return Err(PyTypeError::new_err(format!(
            "Object of type {} is not JSON serializable",
            obj.get_type().name()?
        )));
    }
    Ok(())
}

/// Serialize in Rust, then `json.loads`, so the dict keeps the serialized key order.
fn to_py<'py, T: Serialize + ?Sized>(py: Python<'py>, value: &T) -> PyResult<Bound<'py, PyAny>> {
    let text = serde_json::to_string(value).map_err(runtime_error)?;
    py.import("json")?.call_method1("loads", (text,))
}

fn emit_record(sink: &Option<RecordHandle>, kind: &str, body: &impl Serialize) -> bool {
    sink.as_ref()
        .is_some_and(|s| crate::record::record(s, kind, body))
}

fn emit_json_line(sink: &dyn RecordSink, kind: &str, body_json: &str) -> bool {
    match crate::record::encode_record_json_line(kind, body_json) {
        Ok(line) => sink.record_line(kind, line),
        Err(error) => {
            sink.record_encoding_error(error.contains("exceeds size cap"));
            false
        }
    }
}
