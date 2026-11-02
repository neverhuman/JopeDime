use super::forwarded_args;
use std::ffi::OsString;

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
