use super::*;

#[test]
fn compressed_inputs_are_decoded_and_bound_to_their_decoded_digest() {
    // Addition: `.gz` and `.zst` inputs decode in process and must match `decoded_sha256`;
    // without a zstd decoder a `.zst` input is unavailable, never trusted.
    let directory = scratch();
    let root = directory.path();
    let data = b"track_id,dataset_id\nt,d1\n".repeat(50);
    let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    gz.write_all(&data).unwrap();
    std::fs::write(root.join("x.csv.gz"), gz.finish().unwrap()).unwrap();
    let zst =
        ruzstd::encoding::compress_to_vec(&data[..], ruzstd::encoding::CompressionLevel::Fastest);
    std::fs::write(root.join("x.csv.zst"), zst).unwrap();
    let item = |name: &str, decoded: &str| {
        json(&format!(
            r#"{{"path": "{name}", "sha256": "{}", "decoded_sha256": "{decoded}"}}"#,
            digest(&root.join(name))
        ))
    };
    let decoded = py::sha256_hex(&data);
    for name in ["x.csv.gz", "x.csv.zst"] {
        let loaded = elo_history::load(root, &item(name, &decoded), Zstd::Builtin).unwrap();
        assert_eq!(loaded, data, "{name}");
        assert_refused(
            elo_history::load(root, &item(name, &"0".repeat(64)), Zstd::Builtin),
            &format!("^{}: decoded digest mismatch$", regex::escape(name)),
        );
    }
    match elo_history::load(root, &item("x.csv.zst", &decoded), Zstd::Unavailable) {
        Err(jope_records::Error::Unavailable(reason)) => {
            assert_eq!(reason, "x.csv.zst: no zstd decoder on this host")
        }
        other => panic!("expected unavailable, got {other:?}"),
    }
    // A plain input needs no decoded digest.
    std::fs::write(root.join("plain.csv"), &data).unwrap();
    let plain = json(&format!(
        r#"{{"path": "plain.csv", "sha256": "{}"}}"#,
        digest(&root.join("plain.csv"))
    ));
    assert_eq!(
        elo_history::load(root, &plain, Zstd::Unavailable).unwrap(),
        data
    );
    assert_refused(
        elo_history::load(
            root,
            &json(r#"{"path": "plain.csv", "sha256": null}"#),
            Zstd::Builtin,
        ),
        r#"^\{'path': 'plain\.csv', 'sha256': None\}: input reference$"#,
    );
}
