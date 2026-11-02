use super::*;

fn normalize_router(router: &[f64]) -> Vec<f64> {
    let mean = router.iter().sum::<f64>() / router.len().max(1) as f64;
    let var = router
        .iter()
        .map(|v| {
            let d = *v - mean;
            d * d
        })
        .sum::<f64>()
        / router.len().max(1) as f64;
    let scale = var.sqrt().max(1e-6);
    router
        .iter()
        .map(|v| ((*v - mean) / scale).clamp(-4.0, 4.0))
        .collect()
}

fn router_hash(router: &[f64]) -> String {
    let mut hasher = Sha256::new();
    for v in router {
        hasher.update(v.to_le_bytes());
    }
    hex_digest(hasher.finalize())
}

fn router_seed(router: &[f64]) -> u64 {
    let mut h = 0x726f_7574_6572_3132;
    for v in router {
        h = mix64(h ^ v.to_bits());
    }
    h
}

fn kernel_seed(seed: u64, sample: usize, step: usize, label: &str) -> i64 {
    let mut h = mix64(seed ^ mix64(sample as u64) ^ ((step as u64) << 24));
    for b in label.bytes() {
        h = mix64(h ^ b as u64);
    }
    (h % 9999 + 1) as i64
}

fn sha256_hex(data: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data.as_bytes());
    hex_digest(hasher.finalize())
}

fn hex_digest(digest: impl AsRef<[u8]>) -> String {
    let bytes = digest.as_ref();
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

fn finite(v: f64) -> f64 {
    if v.is_finite() {
        v
    } else {
        0.0
    }
}

fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}
