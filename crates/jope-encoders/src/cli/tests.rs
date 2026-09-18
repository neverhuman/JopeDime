use super::input::{capped_features, keep_feature_prefix, parse_device, parse_dtype};
use super::*;
use candle_core::{DType, Device};

#[test]
fn parses_cpu_device_and_supported_dtypes() {
    assert!(matches!(parse_device("cpu").unwrap(), Device::Cpu));
    assert_eq!(parse_dtype("f32").unwrap(), DType::F32);
    assert_eq!(parse_dtype("float32").unwrap(), DType::F32);
    assert_eq!(parse_dtype("bf16").unwrap(), DType::BF16);
    assert_eq!(parse_dtype("bfloat16").unwrap(), DType::BF16);
}

#[test]
fn rejects_invalid_device_dtype_and_cli_arguments() {
    assert!(parse_device("tpu:0").is_err());
    assert!(parse_device("cuda:not-a-number").is_err());
    assert!(parse_dtype("f16").is_err());

    assert!(Args::try_parse_from(["starforge", "unknown"]).is_err());
    assert!(Args::try_parse_from(["starforge", "bench", "--limit", "not-a-number",]).is_err());
}

#[test]
fn caps_feature_prefix_for_regression_bench() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
    assert_eq!(capped_features(4, 2), 2);
    assert_eq!(capped_features(4, 0), 4);
    assert_eq!(
        keep_feature_prefix(x.clone(), 2, 4, 2),
        vec![1.0, 2.0, 10.0, 20.0]
    );
    assert_eq!(keep_feature_prefix(x.clone(), 2, 4, 4), x);
}
