use super::{forwarded_args, parse_native_validation, ValidationKind, ValidationRequest};
use std::ffi::OsString;
use std::path::PathBuf;

#[test]
fn broad_shadow_subcommand_precedes_unchanged_python_arguments() {
    let args = [
        "freeze-catalog",
        "--rows",
        "rows.json",
        "--out",
        "catalog.json",
    ];
    assert_eq!(
        forwarded_args(args),
        [
            "broad-shadow",
            "freeze-catalog",
            "--rows",
            "rows.json",
            "--out",
            "catalog.json"
        ]
        .map(OsString::from)
    );
}

#[test]
fn broad_shadow_arguments_keep_non_utf8_paths_intact() {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStringExt;
        let path = OsString::from_vec(vec![b'r', b'o', b'w', 0xff]);
        let args = forwarded_args([
            OsString::from("validate"),
            OsString::from("catalog"),
            path.clone(),
        ]);
        assert_eq!(args[0], "broad-shadow");
        assert_eq!(args[1], "validate");
        assert_eq!(args[2], "catalog");
        assert_eq!(args[3], path);
    }
}

#[test]
fn catalog_and_slate_validation_parse_the_python_cli_arguments() {
    let catalog = parse_native_validation(&[
        "validate".into(),
        "catalog".into(),
        "catalog.json".into(),
        "--production".into(),
    ])
    .unwrap();
    assert_eq!(
        catalog,
        ValidationRequest {
            kind: ValidationKind::Catalog,
            path: PathBuf::from("catalog.json"),
            production: true,
        }
    );

    let slate =
        parse_native_validation(&["validate".into(), "slate".into(), "slate.json".into()]).unwrap();
    assert_eq!(slate.kind, ValidationKind::ProbeSlate);
    assert_eq!(slate.path, PathBuf::from("slate.json"));
    assert!(!slate.production);
}

#[test]
fn unsupported_or_malformed_commands_remain_on_the_python_fallback() {
    assert!(
        parse_native_validation(&["validate".into(), "receipt".into(), "x.json".into()]).is_none()
    );
    assert!(parse_native_validation(&["validate".into(), "catalog".into()]).is_none());
    assert!(parse_native_validation(&[
        "validate".into(),
        "catalog".into(),
        "x.json".into(),
        "--unknown".into(),
    ])
    .is_none());
}
