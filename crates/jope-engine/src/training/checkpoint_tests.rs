use super::hash_parameter_values;
use sha2::{Digest, Sha256};

#[test]
fn batched_parameter_hash_preserves_every_float_bit() {
    let values = [
        0u32, 0x80000000, 0x3f800000, 0xbf800000, 0x00000001, 0x7f800000, 0xff800000, 0x7fc00001,
        0x7f800001, 0xffffffff,
    ]
    .map(f32::from_bits);
    let mut old = Sha256::new();
    let mut new = Sha256::new();
    old.update("tensor-name");
    new.update("tensor-name");
    for value in values {
        old.update(value.to_le_bytes());
    }
    hash_parameter_values(&mut new, &values);
    assert_eq!(old.finalize(), new.finalize());
}

#[test]
#[ignore = "explicit full-policy CPU checkpoint hash benchmark"]
fn benchmark_parameter_hash() {
    let values = vec![0.25f32; 27_796_494];
    let mut timings = Vec::new();
    let mut digests = Vec::new();
    for batched in [false, true] {
        let begun = std::time::Instant::now();
        for _ in 0..3 {
            let mut h = Sha256::new();
            if batched {
                hash_parameter_values(&mut h, &values);
            } else {
                for value in &values {
                    h.update(value.to_le_bytes());
                }
            }
            digests.push(h.finalize());
        }
        timings.push(begun.elapsed().as_secs_f64() / 3.0);
    }
    assert!(digests.iter().all(|x| *x == digests[0]));
    eprintln!("checkpoint SHA256 benchmark: 27,796,494 FP32 parameters, per-float {:.3} ms, batched {:.3} ms, {:.2}x speedup, identical digests",
            timings[0]*1000.0,timings[1]*1000.0,timings[0]/timings[1]);
}
