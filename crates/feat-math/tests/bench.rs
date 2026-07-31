//! Manual emitter micro-benchmark (run with `--ignored --nocapture`).
//! Reports pure string-assembly tokens/sec on a representative 8-stage genome.

use feat_math::transpile::{emit, EmitOptions, GenomeIr};

#[test]
#[ignore]
fn emitter_speed() {
    let src = "(pipe X (standardize append=true) (rff_cos n=32 seed=7 gamma=1.0 append=true) \
               (cheb degree=3 append=true) (rank append=true) (winsor_clip q=0.02 append=true) \
               (poly_interact append=true) (row_stats append=true) (ridge))";
    let genome = GenomeIr::parse(src).unwrap();
    let opts = EmitOptions::default();
    // Warm up.
    for _ in 0..200 {
        let _ = emit(&genome, &opts).unwrap();
    }
    let iters = 4000usize;
    let mut ns: Vec<u128> = Vec::with_capacity(iters);
    let mut tokens = 0usize;
    for _ in 0..iters {
        let e = emit(&genome, &opts).unwrap();
        ns.push(e.timing.emit_ns);
        tokens = e.timing.py_tokens + e.timing.rs_tokens;
    }
    ns.sort_unstable();
    let min = ns[0];
    let median = ns[iters / 2];
    let tps_min = tokens as f64 / (min as f64 / 1e9);
    let tps_med = tokens as f64 / (median as f64 / 1e9);
    println!(
        "EMIT-BENCH stages=8 tokens={tokens} min_ns={min} median_ns={median} \
         tokens/sec(min)={tps_min:.3e} tokens/sec(median)={tps_med:.3e}"
    );
}
