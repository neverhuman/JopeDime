#[test]
fn multiline_metadata_remains_documentation() {
    if !support::enabled() {
        return;
    }
    let (x, y, probe) = data();
    let genome = GenomeIr::parse("(pipe X (ridge))").unwrap();
    let options = EmitOptions {
        title: Some("Title\r\nSecond line\u{2028}third line".into()),
        ..EmitOptions::default()
    };
    let bundle = emit(&genome, &options).unwrap();
    let root = std::env::current_dir()
        .unwrap()
        .join("../../target/cuda-metadata");
    support::run(
        &root,
        &bundle,
        &serde_json::json!({"train":x,"targets":y,"test":probe}),
        &fit_predict(&genome, &x, &y, &probe).unwrap(),
        false,
    );
    eprintln!("CUDA C++ metadata: strict heading qualified");
}
