use super::*;

pub(super) fn rows_by_name(rows: &[Value]) -> BTreeMap<&str, &Value> {
    rows.iter()
        .map(|row| (row["name"].as_str().unwrap(), row))
        .collect()
}

pub(super) fn witness_sources() -> BTreeMap<String, String> {
    let historical: Value = serde_json::from_str(HISTORICAL).unwrap();
    let emitted: Value = serde_json::from_str(EMITTED).unwrap();
    let programs = rows_by_name(emitted["programs"].as_array().unwrap());
    historical["registry"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|row| row["evidence"]["kind"] == "fitted_existing")
        .map(|row| {
            let name = row["evidence"]["name"].as_str().unwrap();
            (
                name.to_owned(),
                programs[name]["canonical"].as_str().unwrap().to_owned(),
            )
        })
        .collect()
}

pub(super) fn toy_data() -> (Vec<Vec<f64>>, Vec<f64>) {
    toy_data_seed(42)
}

pub(super) fn toy_data_seed(seed: u64) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = Pcg32::new(seed, 17);
    let mut x = Vec::new();
    let mut y = Vec::new();
    for _ in 0..80 {
        let row: Vec<f64> = (0..5).map(|_| rng.next_f64() * 4.0 - 2.0).collect();
        let target = 1.7 * row[0] - 0.9 * row[1]
            + 0.5 * (3.0 * row[2]).sin()
            + 0.3 * row[3] * row[4]
            + 0.1 * rng.gaussian();
        x.push(row);
        y.push(target);
    }
    (x, y)
}

pub(super) fn write_recipe_csvs(dir: &Path, x: &[Vec<f64>], y: &[f64]) {
    let mut train = String::new();
    for (row, target) in x[..TRAINING_ROWS].iter().zip(&y[..TRAINING_ROWS]) {
        for value in row {
            train.push_str(&format!("{value:?},"));
        }
        train.push_str(&format!("{target:?}\n"));
    }
    fs::write(dir.join("train.csv"), train).unwrap();
    let mut test = String::new();
    for row in &x[TRAINING_ROWS..] {
        test.push_str(
            &row.iter()
                .map(|value| format!("{value:?}"))
                .collect::<Vec<_>>()
                .join(","),
        );
        test.push('\n');
    }
    fs::write(dir.join("test.csv"), test).unwrap();
    assert_eq!(
        fs::read_to_string(dir.join("train.csv"))
            .unwrap()
            .lines()
            .count(),
        TRAINING_ROWS
    );
    assert_eq!(
        fs::read_to_string(dir.join("test.csv"))
            .unwrap()
            .lines()
            .count(),
        x.len() - TRAINING_ROWS
    );
}

pub(super) fn require_success(what: &str, command: &mut Command) {
    let output = command
        .output()
        .unwrap_or_else(|error| panic!("{what}: {error}"));
    assert!(
        output.status.success(),
        "{what}: exit {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

pub(super) fn recipe_bits(path: &Path) -> Vec<u64> {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| line.trim().parse::<f64>().unwrap().to_bits())
        .collect()
}

pub(super) fn run_emitted_pair(
    name: &str,
    genome: &GenomeIr,
    dir: &Path,
    x: &[Vec<f64>],
    y: &[f64],
) -> (Vec<u64>, Vec<u64>) {
    fs::create_dir_all(dir).unwrap();
    write_recipe_csvs(dir, x, y);
    let artifact = emit(genome, &EmitOptions::default()).unwrap();
    let python = dir.join("model.py");
    let python_out = dir.join("python.txt");
    fs::write(&python, artifact.python).unwrap();
    require_success(
        &format!("{name}: emitted Python"),
        Command::new("python3")
            .arg(&python)
            .arg(dir.join("train.csv"))
            .arg(dir.join("test.csv"))
            .arg(&python_out)
            .env("LC_ALL", "C")
            .env("PYTHONDONTWRITEBYTECODE", "1"),
    );
    let rust = dir.join("model.rs");
    let binary = dir.join("model_bin");
    let rust_out = dir.join("rust.txt");
    fs::write(&rust, artifact.rust).unwrap();
    require_success(
        &format!("{name}: rustc -O"),
        Command::new("rustc")
            .arg("-O")
            .arg("-o")
            .arg(&binary)
            .arg(&rust),
    );
    require_success(
        &format!("{name}: emitted Rust"),
        Command::new(&binary)
            .arg(dir.join("train.csv"))
            .arg(dir.join("test.csv"))
            .arg(&rust_out),
    );
    (recipe_bits(&python_out), recipe_bits(&rust_out))
}

pub(super) fn run_cpp_recipe(
    name: &str,
    mut source: String,
    dir: &Path,
    x: &[Vec<f64>],
    y: &[f64],
) -> Vec<u64> {
    assert_eq!(x.len(), 80);
    assert_eq!(y.len(), 80);
    assert_eq!(TRAINING_ROWS, 64, "C++ numeric ABI training rows");
    assert!(x.iter().all(|row| row.len() == 5));
    fs::create_dir_all(dir).unwrap();
    source.push_str("\n#include <cstdio>\nint main() {\n");
    // Decode exact input bits, including signed zero and nonfinite feature payloads.
    // The input reaches the model's numeric ABI rather than the finite-only CSV CLI.
    for (label, values) in [
        (
            "train",
            x[..TRAINING_ROWS]
                .iter()
                .flatten()
                .copied()
                .collect::<Vec<_>>(),
        ),
        ("targets", y[..TRAINING_ROWS].to_vec()),
        (
            "test",
            x[TRAINING_ROWS..]
                .iter()
                .flatten()
                .copied()
                .collect::<Vec<_>>(),
        ),
    ] {
        write!(source, "const std::uint64_t {label}_bits[] = {{").unwrap();
        for value in &values {
            write!(source, "0x{:016x}ULL,", value.to_bits()).unwrap();
        }
        writeln!(
            source,
            "}}; double {label}[{}]; for (std::size_t i=0; i<{}; ++i) std::memcpy(&{label}[i], &{label}_bits[i], 8);",
            values.len(), values.len()
        )
        .unwrap();
    }
    source.push_str(
        "double out[16] = {}; model_handle* model = nullptr;\n\
         if (model_fit_numeric(train, 320, 64, 5, targets, 64, &model) != 0) return 10;\n\
         if (model_predict_numeric_batch(model, test, 80, 16, 5, nullptr, 0, out, 16) != 0) return 11;\n\
         for (double value : out) { std::uint64_t bits; std::memcpy(&bits, &value, 8); std::printf(\"%016llx\\n\", static_cast<unsigned long long>(bits)); }\n\
         model_unload(model); return 0; }\n",
    );
    let path = dir.join("model.cpp");
    let binary = dir.join("model.bin");
    fs::write(&path, source).unwrap();
    require_success(
        &format!("{name}: g++17 O2"),
        Command::new("g++")
            .args(["-std=c++17", "-O2", "-ffp-contract=off", "-fno-fast-math"])
            .arg(&path)
            .arg("-o")
            .arg(&binary),
    );
    let output = Command::new(&binary).output().unwrap();
    assert!(
        output.status.success(),
        "{name}: C++17 recipe exit {}: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    let bits: Vec<u64> = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|word| u64::from_str_radix(word, 16).unwrap())
        .collect();
    assert_eq!(bits.len(), 16, "{name}: C++17 output row count");
    bits
}

pub(super) fn round_trip(source: &str) -> GenomeIr {
    let original = GenomeIr::parse(source).unwrap();
    let canonical = original.canonical();
    let genome = GenomeIr::parse(&canonical).unwrap();
    assert_eq!(canonical, genome.canonical());
    let bytes = Program::import_legacy(&genome).to_bytes().unwrap();
    assert_eq!(
        bytes,
        Program::from_bytes(&bytes).unwrap().to_bytes().unwrap()
    );
    genome
}
