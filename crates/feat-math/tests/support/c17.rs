//! Execute an actual C caller over exact-bit array inputs, linked to emitted sources.
use feat_math::{transpile::EmitOptions, GenomeIr};
use std::{fs, path::Path, process::Command};

pub fn run(command: &mut Command) -> String {
    let result = command
        .output()
        .unwrap_or_else(|e| panic!("{command:?}: {e}"));
    assert!(
        result.status.success(),
        "{command:?} exited {}: {}\n{}",
        result.status,
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    String::from_utf8(result.stdout).unwrap()
}

pub fn build(dir: &Path, genome: &GenomeIr, options: &EmitOptions) {
    fs::create_dir_all(dir).unwrap();
    for (path, source) in feat_math::transpile::emit_c17(genome, options)
        .unwrap()
        .files
    {
        fs::write(dir.join(path.strip_prefix("c17/").unwrap()), source).unwrap();
    }
    compile(dir);
}

pub fn compile(dir: &Path) {
    run(Command::new("rustc").current_dir(dir).args([
        "--edition=2021",
        "-O",
        "-C",
        "panic=unwind",
        "--crate-type=staticlib",
        "-Dwarnings",
        "model_ffi.rs",
        "-o",
        "libmodel.a",
    ]));
}

pub fn consumer(dir: &Path, compiler: &str, source: &str) -> String {
    fs::write(dir.join("consumer.c"), source).unwrap();
    run(Command::new(compiler).current_dir(dir).args([
        "-std=c17",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-pedantic-errors",
        "-fno-fast-math",
        "-ffp-contract=off",
        "consumer.c",
        "libmodel.a",
        "-ldl",
        "-lpthread",
        "-lm",
        "-o",
        "consumer",
    ]));
    run(&mut Command::new(dir.join("consumer")))
}

fn array(name: &str, values: impl Iterator<Item = f64>) -> String {
    let bits: Vec<_> = values
        .map(|v| format!("UINT64_C(0x{:016x})", v.to_bits()))
        .collect();
    format!(
        "static const uint64_t {name}[] = {{{}}};\n",
        if bits.is_empty() {
            "0".into()
        } else {
            bits.join(",")
        }
    )
}

pub fn predictions(
    dir: &Path,
    genome: &GenomeIr,
    options: &EmitOptions,
    train: &[Vec<f64>],
    y: &[f64],
    test: &[Vec<f64>],
) -> Vec<u64> {
    build(dir, genome, options);
    let rows = train.len();
    let cols = train[0].len();
    let ntest = test.len();
    let mut c = String::from(
        "#include \"model.h\"\n#include <inttypes.h>\n#include <stdio.h>\n#include <string.h>\n",
    );
    c.push_str(&array("xb", train.iter().flatten().copied()));
    c.push_str(&array("yb", y.iter().copied()));
    c.push_str(&array("tb", test.iter().flatten().copied()));
    c.push_str(&format!(r#"
int main(void) {{
    double x[{nx}], y[{rows}], t[{nt}], out[{nout}]; uint64_t h=0;
    memcpy(x, xb, sizeof(x)); memcpy(y, yb, sizeof(y)); memcpy(t, tb, sizeof(t));
    if(jope_model_fit(x,{nx},y,{rows},{rows},{cols},&h)) return 11;
    if(jope_model_predict(h,t,{nt},{ntest},{cols},out,{ntest})) return 12;
    for(size_t i=0;i<{ntest};i++) {{ uint64_t bits; memcpy(&bits,&out[i],8); printf("%016" PRIx64 "\n",bits); }}
    return jope_model_free(h);
}}
"#, nx=rows*cols, nt=(ntest*cols).max(1), nout=ntest.max(1)));
    consumer(dir, "gcc", &c)
        .lines()
        .map(|v| u64::from_str_radix(v, 16).unwrap())
        .collect()
}
