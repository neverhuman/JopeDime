//! Emit one program to every portable target.
//!
//! `cargo run -p feat-math --example emit -- [--all] <program.dsl> <out-dir>` writes `model.py` and
//! `model.rs` (the bit-identical pair, always), `model.cpp` (C++17, also a valid C++20 program) and
//! `model.js` when the independently certified C++17/ES2020 emitters cover the program, the sidecar
//! `manifest.json`, and `targets.txt` naming what was emitted. With `--all` an uncovered target is
//! an error naming the unsupported elements; without it the reason is printed and recorded, never
//! silently dropped. The sources are byte-deterministic — `deploy/ci/emitters.sh` pins their
//! sha256 and the bit pattern of the predictions they produce.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use feat_math::genome::GenomeIr;
use feat_math::transpile::{self, EmitOptions};

fn main() -> Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let cuda = args.iter().any(|arg| arg == "--cuda");
    args.retain(|arg| arg != "--cuda");
    let c17 = args.iter().any(|arg| arg == "--c17");
    args.retain(|arg| arg != "--c17");
    let require_all = args.iter().any(|arg| arg == "--all");
    let typescript = args
        .iter()
        .any(|arg| arg == "--typescript" || arg == "--ts");
    args.retain(|arg| !["--all", "--typescript", "--ts"].contains(&arg.as_str()));
    let [dsl, out] = args.as_slice() else {
        bail!("usage: emit [--all] [--c17] [--cuda] [--typescript] <program.dsl> <out-dir>");
    };
    let (dsl, out) = (dsl.clone(), out.clone());
    let source = std::fs::read_to_string(&dsl).with_context(|| format!("read {dsl}"))?;
    let genome = GenomeIr::parse(source.trim()).with_context(|| format!("parse {dsl}"))?;
    let opts = EmitOptions::default();
    let out = PathBuf::from(out);
    std::fs::create_dir_all(&out)?;

    let emitted = transpile::emit(&genome, &opts)?;
    std::fs::write(out.join("model.py"), &emitted.python)?;
    std::fs::write(out.join("model.rs"), &emitted.rust)?;
    std::fs::write(out.join("manifest.json"), &emitted.manifest)?;
    let mut targets = vec!["py", "rs"];
    match transpile::supports_cpp17(&genome) {
        Ok(()) => {
            std::fs::write(
                out.join("model.cpp"),
                transpile::emit_cpp17(&genome, &opts)?,
            )?;
            targets.push("cpp");
        }
        Err(unsupported) if require_all => {
            bail!("C++17 target unsupported: {}", unsupported.join(", "))
        }
        Err(unsupported) => eprintln!("C++17 target not emitted: {}", unsupported.join(", ")),
    }
    match transpile::supports_javascript(&genome) {
        Ok(()) => {
            std::fs::write(
                out.join("model.js"),
                transpile::emit_javascript(&genome, &opts)?,
            )?;
            targets.push("js");
        }
        Err(unsupported) if require_all => {
            bail!("JavaScript target unsupported: {}", unsupported.join(", "))
        }
        Err(unsupported) => eprintln!("JavaScript target not emitted: {}", unsupported.join(", ")),
    }
    if cuda {
        for (path, source) in transpile::emit_cuda(&genome, &opts)?.files {
            let dest = out.join(path);
            std::fs::create_dir_all(dest.parent().context("CUDA output parent")?)?;
            std::fs::write(dest, source)?;
        }
        targets.push("cuda");
    }
    if c17 {
        for (path, source) in transpile::emit_c17(&genome, &opts)?.files {
            let dest = out.join(path);
            std::fs::create_dir_all(dest.parent().context("C17 output parent")?)?;
            std::fs::write(dest, source)?;
        }
        targets.push("c17");
    }
    if typescript {
        let bundle = transpile::emit_typescript(&genome, &opts)?;
        std::fs::write(out.join("model.ts"), bundle.source)?;
        std::fs::write(out.join("model.kernel.rs"), bundle.kernel)?;
        targets.push("ts");
    }
    std::fs::write(out.join("targets.txt"), format!("{}\n", targets.join(" ")))?;
    println!("emitted {} to {}", targets.join(" "), out.display());
    Ok(())
}
