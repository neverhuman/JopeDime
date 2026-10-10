//! Fix-1 fixture: native bin search is the Rust 1.97.1 monobound, and every
//! emitter writes that same loop.
//!
//! The 41 historical failing pairs and the 1,554 native evaluation SHAs were
//! not replayed. Equality with `slice::partition_point` on this toolchain is
//! the proof that native scores cannot move. A mismatch fails closed.

use std::process::Command;

use feat_math::ops::{lower_bound, upper_bound};
use feat_math::transpile::{emit, emit_cpp17, emit_javascript, lint_shipped_text, EmitOptions};
use feat_math::GenomeIr;

fn classic_lower(s: &[f64], v: f64) -> usize {
    let mut lo = 0usize;
    let mut hi = s.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if s[mid] < v {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

fn classic_upper(s: &[f64], v: f64) -> usize {
    let mut lo = 0usize;
    let mut hi = s.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if s[mid] <= v {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

fn lcg(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    *state
}

fn sample(state: &mut u64) -> f64 {
    let draw = lcg(state);
    let mag = ((draw >> 8) % 17) as f64;
    match draw % 7 {
        0 => 0.0,
        1 => mag,
        2 => -mag,
        3 => mag + 0.5,
        4 => f64::NAN,
        5 => f64::INFINITY,
        _ => f64::NEG_INFINITY,
    }
}

fn assert_matches_std(s: &[f64], v: f64) {
    assert_eq!(
        lower_bound(s, v),
        s.partition_point(|e| *e < v),
        "lower_bound drifted from partition_point len={} v={v:?}",
        s.len()
    );
    assert_eq!(
        upper_bound(s, v),
        s.partition_point(|e| *e <= v),
        "upper_bound drifted from partition_point len={} v={v:?}",
        s.len()
    );
}

#[test]
fn vendored_monobound_matches_partition_point_and_not_the_old_loop() {
    let a = 0.2007432314976106_f64;
    let down = f64::from_bits(a.to_bits() - 1);
    let up = f64::from_bits(a.to_bits() + 1);
    let plateau = [a, a, down, up, a];
    assert_ne!(
        lower_bound(&plateau, up),
        classic_lower(&plateau, up),
        "the plateau must still separate the old lower loop from std"
    );
    assert_ne!(
        upper_bound(&plateau, a),
        classic_upper(&plateau, a),
        "the plateau must still separate the old upper loop from std"
    );
    for v in [down, a, up, 0.0, 1.0] {
        assert_matches_std(&plateau, v);
    }

    assert_matches_std(&[], 0.0);
    assert_matches_std(&[1.0], 1.0);
    let mut state = 0x5eed_u64;
    for n in 0..=32 {
        let mut values = Vec::with_capacity(n);
        for _ in 0..n {
            values.push(sample(&mut state));
        }
        let mut probes = values.clone();
        probes.extend([0.0, 1.0, -1.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN]);
        for v in probes {
            assert_matches_std(&values, v);
        }
        let mut sorted = values.clone();
        sorted.retain(|v| v.is_finite());
        sorted.sort_by(|x, y| x.total_cmp(y));
        for v in [0.0, 1.0, -1.0, f64::NAN] {
            assert_matches_std(&sorted, v);
        }
    }
}

#[test]
fn emitters_write_the_monobound_and_python_matches_native() {
    let genome = GenomeIr::parse("(pipe X (rank) (quantile_bins bins=8) (ridge))").unwrap();
    let opts = EmitOptions::default();
    let artifact = emit(&genome, &opts).unwrap();
    let cpp = emit_cpp17(&genome, &opts).unwrap();
    let js = emit_javascript(&genome, &opts).unwrap();
    for (lang, src) in [
        ("rust", artifact.rust.as_str()),
        ("python", artifact.python.as_str()),
        ("cpp", cpp.as_str()),
        ("js", js.as_str()),
    ] {
        let hits = lint_shipped_text(src);
        assert!(hits.is_empty(), "{lang} shipped-text hits: {hits:?}");
    }
    assert!(artifact.rust.contains("while size > 1"));
    assert!(artifact.python.contains("while size > 1"));
    assert!(cpp.contains("while (size > 1)"));
    assert!(js.contains("while(size>1)"));
    assert!(!artifact.rust.contains("(lo + hi) / 2"));
    assert!(!artifact.python.contains("(lo + hi) // 2"));
    assert!(!cpp.contains("(lo + hi) / 2"));
    assert!(!js.contains("(lo+hi)/2"));

    let a = 0.2007432314976106_f64;
    let down = f64::from_bits(a.to_bits() - 1);
    let up = f64::from_bits(a.to_bits() + 1);
    let plateau = [a, a, down, up, a];
    let monotone = [0.0, 0.0, 1.0, 2.0, 2.0];
    let mut cases = Vec::new();
    for (s, v) in [
        (plateau.as_slice(), up),
        (plateau.as_slice(), a),
        (monotone.as_slice(), 0.0),
        (monotone.as_slice(), 2.0),
    ] {
        cases.push((
            s.iter()
                .map(|x| format!("{:x}", x.to_bits()))
                .collect::<Vec<_>>(),
            format!("{:x}", v.to_bits()),
            lower_bound(s, v),
            upper_bound(s, v),
        ));
    }
    let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("binning_search");
    std::fs::create_dir_all(&dir).unwrap();
    let py_path = dir.join("helpers.py");
    let lower = slice_fn(&artifact.python, "def _lower_bound(");
    let upper = slice_fn(&artifact.python, "def _upper_bound(");
    std::fs::write(&py_path, format!("{lower}\n\n{upper}\n")).unwrap();
    let script = dir.join("check.py");
    std::fs::write(
        &script,
        r#"
import json, struct, sys
ns = {}
exec(open(sys.argv[1]).read(), ns)
cases = json.loads(sys.argv[2])

def f(bits):
    return struct.unpack("<d", struct.pack("<Q", int(bits, 16)))[0]

out = []
for raw, vbits, _, _ in cases:
    s = [f(b) for b in raw]
    v = f(vbits)
    out.append([ns["_lower_bound"](s, v), ns["_upper_bound"](s, v)])
print(json.dumps(out))
"#,
    )
    .unwrap();
    let payload = serde_json::to_string(&cases).unwrap();
    let run = Command::new("python3")
        .arg(&script)
        .arg(&py_path)
        .arg(&payload)
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "python helper failed: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    let got: Vec<[usize; 2]> = serde_json::from_slice(&run.stdout).unwrap();
    for (i, (_, _, lower_i, upper_i)) in cases.iter().enumerate() {
        assert_eq!(got[i][0], *lower_i, "python lower case {i}");
        assert_eq!(got[i][1], *upper_i, "python upper case {i}");
    }
}

fn slice_fn(src: &str, marker: &str) -> String {
    let start = src
        .find(marker)
        .unwrap_or_else(|| panic!("missing {marker}"));
    let rest = &src[start..];
    let end = rest[1..]
        .find("\ndef ")
        .map(|i| i + 1)
        .unwrap_or(rest.len());
    rest[..end].trim_end().to_string()
}
