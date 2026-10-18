#[path = "../registry/codegen.rs"]
mod codegen;

fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let write = match args.as_slice() {
        [arg] if arg == "--write" => true,
        [arg] if arg == "--check" => false,
        _ => return Err("usage: registry_codegen --check|--write".into()),
    };
    codegen::run(std::path::Path::new(env!("CARGO_MANIFEST_DIR")), write)?;
    feat_math::kernel::registry_v2::validate_inheritance()
}
