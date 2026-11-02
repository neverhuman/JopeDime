fn config(w: u8, p: u8, r: u8) -> Config {
    Config::new(2, 58, Axis::FeatureWithinRow, w, p, r, "shrink").unwrap()
}
fn program(width: usize, c: Config) -> Program {
    Program::new(2, width, vec![RowStage::Savgol(c)], 0.02).unwrap()
}
fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}
fn training(width: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let x = (0..24)
        .map(|i| {
            (0..width)
                .map(|j| ((i * 13 + j * 7) % 37) as f64 / 9.0 - 2.0)
                .collect::<Vec<f64>>()
        })
        .collect::<Vec<_>>();
    let y = x
        .iter()
        .map(|r| r[0] * 0.5 + r[width - 1] * 0.2 - 0.3)
        .collect();
    (x, y)
}
fn run(command: &mut Command) -> Output {
    let output = command
        .env("LC_ALL", "C")
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .output()
        .expect("required compiler/interpreter missing");
    assert!(
        output.status.success(),
        "{command:?}: {}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

fn cpp(root: &Path, file: &str, bin: &str, compiler: &str, standard: &str, extra: &[&str]) {
    let mut cmd = Command::new(compiler);
    cmd.args(["-O2", "-ffp-contract=off", "-fno-fast-math", standard])
        .args(extra)
        .arg(root.join(file))
        .arg("-o")
        .arg(root.join(bin));
    run(&mut cmd);
}
