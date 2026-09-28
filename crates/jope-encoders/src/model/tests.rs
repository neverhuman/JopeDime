use super::*;
use candle_core::{Device, Tensor};

#[test]
fn rank_aggregate_uses_average_one_based_ranks() {
    let preds = vec![vec![0.1, 0.4, 0.4, 0.9], vec![0.8, 0.1, 0.2, 0.2]];
    let got = rank_aggregate(&preds, 4);
    assert_eq!(got, vec![0.625, 0.4375, 0.625, 0.8125]);
}

// Row-stability lock: the shipped view aggregation must score a row identically whether it is
// sent alone or inside a larger batch. `rank_aggregate` violated this (each query was ranked
// against the OTHER queries in the same request), which is why the ensemble heads aggregate by
// per-row median instead.
#[test]
fn median_of_views_is_row_stable_single_vs_batch() {
    // 6 views x 3 rows; row values are deliberately spread so batch-ranking WOULD reshuffle them.
    let batch: Vec<Vec<f64>> = vec![
        vec![0.10, 0.90, 0.50],
        vec![0.20, 0.80, 0.55],
        vec![0.30, 0.70, 0.45],
        vec![0.40, 0.60, 0.52],
        vec![0.15, 0.85, 0.48],
        vec![0.35, 0.65, 0.51],
    ];
    let batched = median_of_views(&batch, 3);
    for row in 0..3 {
        let alone: Vec<Vec<f64>> = batch.iter().map(|view| vec![view[row]]).collect();
        assert_eq!(
            median_of_views(&alone, 1)[0],
            batched[row],
            "row {row} must score identically alone and in a batch"
        );
    }
    // And it is the median of an even count: mean of the two middle views.
    assert!((batched[0] - 0.25).abs() < 1e-12);
    assert!((batched[1] - 0.75).abs() < 1e-12);
}

// A zstd-compressed `*.safetensors.zst` must decompress and load to exactly the
// same tensors as the equivalent plain `*.safetensors` (lossless image shrink).
#[test]
fn loads_plain_and_zstd_safetensors_identically() {
    let dev = Device::Cpu;
    let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = Tensor::from_vec(values.clone(), (2, 3), &dev).unwrap();
    let head = Tensor::from_vec(vec![0.0f32], (1, 1), &dev).unwrap();
    let mut map = HashMap::new();
    map.insert("w".to_string(), tensor);
    map.insert("output_projection.0.weight".to_string(), head);

    let dir = std::env::temp_dir().join(format!("starforge_zst_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let plain = dir.join("m.safetensors");
    candle_core::safetensors::save(&map, &plain).unwrap();

    let raw = std::fs::read(&plain).unwrap();
    let compressed =
        ruzstd::encoding::compress_to_vec(&raw[..], ruzstd::encoding::CompressionLevel::Fastest);
    // Glue check: our decoder round-trips the frame back to the original bytes.
    assert_eq!(decompress_zstd(&compressed).unwrap(), raw);
    let zst = dir.join("m.safetensors.zst");
    std::fs::write(&zst, &compressed).unwrap();

    let plain_model = StarforgeModel::load(&plain, dev.clone()).unwrap();
    let zst_model = StarforgeModel::load(&zst, dev).unwrap();

    let a = plain_model.weights["w"]
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let b = zst_model.weights["w"]
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(a, b);
    assert_eq!(b, values);

    let _ = std::fs::remove_dir_all(&dir);
}

// Production path: the release weights are compressed by the `zstd` CLI
// (`scripts/compress-starforge-weights.sh`, zstd -19). Confirm the pure-Rust
// ruzstd decoder loads a real zstd-CLI frame identically. Skips when `zstd`
// is unavailable (the ruzstd round-trip above already covers the glue).
#[test]
fn loads_zstd_cli_compressed_safetensors() {
    use std::process::Command;
    let dev = Device::Cpu;
    let values = vec![7.0f32, 8.0, 9.0, 10.0];
    let tensor = Tensor::from_vec(values.clone(), (2, 2), &dev).unwrap();
    let head = Tensor::from_vec(vec![0.0f32], (1, 1), &dev).unwrap();
    let mut map = HashMap::new();
    map.insert("k".to_string(), tensor);
    map.insert("output_projection.0.weight".to_string(), head);

    let dir = std::env::temp_dir().join(format!("starforge_zstdcli_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let plain = dir.join("m.safetensors");
    candle_core::safetensors::save(&map, &plain).unwrap();
    let zst = dir.join("m.safetensors.zst");

    let ran = Command::new("zstd")
        .args(["-q", "-19", "-f"])
        .arg(&plain)
        .arg("-o")
        .arg(&zst)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !ran {
        let _ = std::fs::remove_dir_all(&dir);
        return; // zstd CLI unavailable on this runner; skip.
    }

    let model = StarforgeModel::load(&zst, dev).unwrap();
    let got = model.weights["k"]
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(got, values);
    let _ = std::fs::remove_dir_all(&dir);
}

// Real-weights round-trip (opt-in, slow): exercise the exact production path on the shipped
// Chimera component weights — compress with the `zstd -19` CLI and decompress with the
// pure-Rust runtime decoder, asserting byte-for-byte identity. `#[ignore]` so it stays out of
// the fast lane; run it with
// `cargo test -p starforge -- --ignored roundtrips_real_starforge_weights_if_present`.
// Skips when `zstd` is absent or the weights are LFS pointers.
#[test]
#[ignore = "slow: processes the real ~200MB weights with zstd -19"]
fn roundtrips_real_starforge_weights_if_present() {
    use std::process::Command;
    let candidates = [
        "../../artifacts/starforge/starlight_core_regressor.safetensors",
        "../../artifacts/foundation/tabicl-regressor-v2-20260212.safetensors",
        "../../artifacts/starforge/chimera_stackfusionreg.safetensors",
        "../../artifacts/starforge/chimera_classification_expert_a.safetensors",
        "../../artifacts/foundation/tabicl-classifier-v2-chimera-shuffled.safetensors",
        "../../artifacts/starforge/chimera_classification_attnfusion.safetensors",
    ];
    let mut checked = 0usize;
    for rel in candidates {
        let p = std::path::Path::new(rel);
        let meta = match std::fs::metadata(p) {
            Ok(m) => m,
            Err(_) => continue,
        };
        // Git LFS pointer files are a few hundred bytes; the real tensors are >100MB.
        if meta.len() < 1_000_000 {
            continue;
        }
        let dir =
            std::env::temp_dir().join(format!("sf_real_zst_{}_{}", std::process::id(), checked));
        std::fs::create_dir_all(&dir).unwrap();
        let zst = dir.join("w.safetensors.zst");
        // Production compression path: zstd -19 CLI.
        let ran = Command::new("zstd")
            .args(["-q", "-19", "-f"])
            .arg(p)
            .arg("-o")
            .arg(&zst)
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if !ran {
            let _ = std::fs::remove_dir_all(&dir);
            continue; // zstd CLI unavailable; the synthetic tests still cover the codec.
        }
        let raw = std::fs::read(p).unwrap();
        let compressed = std::fs::read(&zst).unwrap();
        // Runtime decompression path: pure-Rust ruzstd.
        let back = decompress_zstd(&compressed).unwrap();
        assert_eq!(
            back.len(),
            raw.len(),
            "decompressed length differs for {rel}"
        );
        assert!(back == raw, "decompressed bytes differ for {rel}");
        eprintln!(
            "roundtrip OK {rel}: {} -> {} bytes ({:.1}% of original)",
            raw.len(),
            compressed.len(),
            100.0 * compressed.len() as f64 / raw.len() as f64
        );
        checked += 1;
        let _ = std::fs::remove_dir_all(&dir);
    }
    eprintln!("verified {checked} real weight file(s)");
}
