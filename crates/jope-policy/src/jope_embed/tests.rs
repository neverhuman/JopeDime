use super::*;
use std::path::Path;
#[test]
#[ignore = "requires the reviewed Starforge model-bundle artifact"]
fn reviewed_preprocessing_bundle_loads_with_exact_v1_shapes() {
    let root = std::env::var_os("JAIN_TEST_MODEL_BUNDLE_ROOT")
        .map(std::path::PathBuf::from)
        .expect("JAIN_TEST_MODEL_BUNDLE_ROOT must identify the reviewed bundle");
    let encoder = JopeEncoder::from_safetensors(root.join("preprocessing.safetensors"))
        .expect("load reviewed preprocessing.safetensors");
    let preprocessing = encoder.preprocessing();
    assert_eq!(preprocessing.hyperion_mean.len(), 512);
    assert_eq!(preprocessing.hyperion_scale.len(), 512);
    assert_eq!(preprocessing.foundation_pca_mean.len(), 2048);
    assert_eq!(preprocessing.foundation_components.len(), 1024 * 2048);
    assert_eq!(preprocessing.foundation_mean.len(), 1024);
    assert_eq!(preprocessing.foundation_scale.len(), 1024);
}

#[test]
fn versioned_encoder_rejects_missing_target_sidecar() {
    let preprocessing = JopePreprocessingV1 {
        hyperion_mean: vec![0.0; 512],
        hyperion_scale: vec![1.0; 512],
        foundation_pca_mean: vec![0.0; 2048],
        foundation_components: vec![0.0; 1024 * 2048],
        foundation_mean: vec![0.0; 1024],
        foundation_scale: vec![1.0; 1024],
    };
    let encoder = JopeEncoder::new(preprocessing).unwrap();
    let input = JopeDatasetInputV1 {
        dataset_id: "fixture".into(),
        input_digest: "input".into(),
        target_sidecar_digest: "target".into(),
        hyperion_rows: vec![vec![0.0; 128]],
        foundation_raw: vec![0.0; 2048],
        target_rows: Vec::new(),
    };
    let options = JopeEncodeOptions {
        model_bundle_digest: "bundle".into(),
        ..Default::default()
    };
    assert!(encoder.encode(&input, &options).is_err());
}

#[test]
fn versioned_encoder_assembles_blocks_deterministically() {
    let mut components = vec![0.0; 1024 * 2048];
    for index in 0..1024 {
        components[index * 2048 + index] = 1.0;
    }
    let encoder = JopeEncoder::new(JopePreprocessingV1 {
        hyperion_mean: vec![0.0; 512],
        hyperion_scale: vec![1.0; 512],
        foundation_pca_mean: vec![0.0; 2048],
        foundation_components: components,
        foundation_mean: vec![0.0; 1024],
        foundation_scale: vec![1.0; 1024],
    })
    .unwrap();
    let mut first = vec![0.0; 128];
    let mut second = vec![0.0; 128];
    first[0] = 1.0;
    second[0] = 3.0;
    let mut foundation = vec![0.0; 2048];
    foundation[0] = 5.0;
    let input = JopeDatasetInputV1 {
        dataset_id: "fixture".into(),
        input_digest: String::new(),
        target_sidecar_digest: "target".into(),
        hyperion_rows: vec![first, second],
        foundation_raw: foundation,
        target_rows: vec![vec![2.0; 512], vec![4.0; 512]],
    };
    let options = JopeEncodeOptions {
        model_bundle_digest: "bundle".into(),
        ..Default::default()
    };
    let a = encoder.encode(&input, &options).unwrap();
    let b = encoder.encode(&input, &options).unwrap();
    assert_eq!(a.embedding, b.embedding);
    assert_eq!(a.target, vec![3.0; 512]);
    assert_eq!(a.embedding[0], 2.0);
    assert_eq!(a.embedding[512], 5.0);
    assert_eq!(a.embedding.len(), 1536);
    assert_eq!(a.input_digest.len(), 64);
}

// Parity vs the Python reference dumped for hash 0014b9dad5725a1e (full rows).
#[test]
#[ignore = "requires reviewed JOPE parity artifacts and reference output"]
fn embedding_matches_python_reference() {
    let art_buf = std::env::var_os("JOPE_PARITY_ART_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| Path::new("artifacts/jope").to_path_buf());
    let ds_buf = std::env::var_os("JOPE_PARITY_DATASET_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| Path::new("target/jope-fixture").to_path_buf());
    let art = art_buf.as_path();
    let ds = ds_buf.as_path();
    let ref_path =
        std::env::var("JOPE_PARITY_REF").unwrap_or_else(|_| "target/emb_ref.json".into());
    assert!(
        ds.is_dir(),
        "JOPE parity dataset directory is missing: {}",
        ds.display()
    );
    assert!(
        std::path::Path::new(&ref_path).is_file(),
        "JOPE parity reference is missing: {ref_path}"
    );
    let cfg = load_std_cfg(art).unwrap();
    let proj = load_projection(&cfg.proj_npz).unwrap();
    let (emb, tgt) = jope_embedding(ds, &cfg, &proj, 1.0, 0).unwrap();
    let refj: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(ref_path).unwrap()).unwrap();
    let re: Vec<f32> = refj["embedding"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect();
    let rt: Vec<f32> = refj["target"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect();
    assert_eq!(emb.len(), 1536);
    assert_eq!(tgt.len(), 512);
    let maxd = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
    };
    let de = maxd(&emb, &re);
    let dt = maxd(&tgt, &rt);
    eprintln!("max|Δ| embedding={de:.2e} target={dt:.2e}");
    assert!(de < 1e-4, "embedding parity: max diff {de}");
    assert!(dt < 1e-4, "target parity: max diff {dt}");
}
