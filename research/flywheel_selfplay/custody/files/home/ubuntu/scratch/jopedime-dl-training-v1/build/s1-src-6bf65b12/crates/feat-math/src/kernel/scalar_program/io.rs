//! Preserve scalar-v2 raw IEEE input and width refusal in the existing recipe IO.
use super::emit::ScalarLanguage;
use crate::kernel::{Error, Result};

pub(crate) fn io_source(language: ScalarLanguage, existing: &str) -> Result<String> {
    let replacements: &[(&str,&str)] = match language {
        ScalarLanguage::Rust => &[
            ("blank, malformed, or non-finite cells become 0.0.", "blank or malformed cells become 0.0; non-finite IEEE inputs are retained."),
            ("Ok(value) => {\n            if value.is_finite() {\n                value\n            } else {\n                0.0\n            }\n        }", "Ok(value) => value,"),
            ("x_train.push(pad(row, width));", "assert_eq!(row.len(),width,\"scalar training width\"); x_train.push(row);"),
            ("x_test.push(pad(row, width));", "assert_eq!(row.len(),width,\"scalar prediction width\"); x_test.push(row);"),
        ],
        ScalarLanguage::Python => &[
            ("blank, malformed, or non-finite cells become 0.0.", "blank or malformed cells become 0.0; non-finite IEEE inputs are retained."),
            ("return value if math.isfinite(value) else 0.0", "return value"),
            ("x_train = [_pad(row, width) for row in features]", "x_train = features\n    if any(len(row)!=width for row in x_train): raise ValueError('scalar training width')"),
            ("x_test = [_pad(row, width) for row in _read_csv(argv[1])]", "x_test = _read_csv(argv[1])\n    if any(len(row)!=width for row in x_test): raise ValueError('scalar prediction width')"),
        ],
        ScalarLanguage::Cpp17 => &[], // Existing C ABI already passes raw cells and explicit dimensions.
    };
    let mut source = existing.to_owned();
    for &(from, to) in replacements {
        if source.matches(from).count() != 1 {
            return Err(Error::Unsupported(
                "scalar recipe IO template boundary".into(),
            ));
        }
        source = source.replacen(from, to, 1);
    }
    Ok(source)
}
