//! Direct C++17 numeric-ABI boundary proof for a generated training recipe.

use feat_math::transpile::{emit_cpp17, EmitOptions};
use feat_math::{fit_predict, GenomeIr, Pcg32};
use std::fmt::Write;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

const PROGRAM: &str = "(pipe X (standardize) (ridge))";
const SEEDS: [u64; 3] = [41, 43, 47];

fn sample(seed: u64) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>) {
    let mut rng = Pcg32::new(seed, 17);
    let mut rows = || {
        (0..3)
            .map(|_| rng.next_f64() * 4.0 - 2.0)
            .collect::<Vec<_>>()
    };
    let mut train = (0..24).map(|_| rows()).collect::<Vec<_>>();
    let mut test = (0..8).map(|_| rows()).collect::<Vec<_>>();
    for data in [&mut train, &mut test] {
        data[0][0] = f64::from_bits(0x7ff8_1234_5678_9abc);
        data[1][1] = f64::INFINITY;
        data[2][2] = f64::NEG_INFINITY;
        data[3][0] = -0.0;
        data[4][1] = f64::from_bits(1);
    }
    let targets = train
        .iter()
        .map(|row| {
            let finite = |value: f64| if value.is_finite() { value } else { 0.0 };
            0.7 * finite(row[0]) - 0.2 * finite(row[1]) + 0.1 * finite(row[2])
        })
        .collect();
    (train, targets, test)
}

fn bit_array(source: &mut String, name: &str, values: impl IntoIterator<Item = f64>) {
    write!(source, "double {name}[] = {{").unwrap();
    for value in values {
        write!(source, "boundary_from_bits(0x{:016x}ULL),", value.to_bits()).unwrap();
    }
    source.push_str("};\n");
}

#[test]
fn cpp17_training_numeric_abi_matches_seeded_nonfinite_features_and_refuses_targets() {
    let genome = GenomeIr::parse(PROGRAM).unwrap();
    let mut source = emit_cpp17(&genome, &EmitOptions::default()).unwrap();
    source.push_str(
        "\n#include <cstdio>\n#include <cstdint>\n#include <cstring>\n\
         static double boundary_from_bits(std::uint64_t bits) {\n\
           double value; std::memcpy(&value, &bits, sizeof(value)); return value;\n\
         }\nint main() {\n\
           if (model_abi_version() != 1U) return 2;\n",
    );

    let mut expected = Vec::new();
    for (index, seed) in SEEDS.into_iter().enumerate() {
        let (train, targets, test) = sample(seed);
        expected.extend(
            fit_predict(&genome, &train, &targets, &test)
                .unwrap()
                .into_iter()
                .map(f64::to_bits),
        );
        bit_array(
            &mut source,
            &format!("train_{index}"),
            train.into_iter().flatten(),
        );
        bit_array(&mut source, &format!("targets_{index}"), targets);
        bit_array(
            &mut source,
            &format!("test_{index}"),
            test.into_iter().flatten(),
        );
        writeln!(source, "double out_{index}[8] = {{}};").unwrap();
        writeln!(source, "model_handle* model_{index} = nullptr;").unwrap();
        writeln!(
            source,
            "if (model_fit_numeric(train_{index}, 72, 24, 3, targets_{index}, 24, &model_{index}) != 0 || model_{index} == nullptr) return 10;"
        )
        .unwrap();
        writeln!(
            source,
            "if (model_predict_numeric_batch(model_{index}, test_{index}, 24, 8, 2, nullptr, 0, out_{index}, 8) != 1) return 11;"
        )
        .unwrap();
        writeln!(
            source,
            "if (model_predict_numeric_batch(model_{index}, test_{index}, 23, 8, 3, nullptr, 0, out_{index}, 8) != 1) return 12;"
        )
        .unwrap();
        writeln!(
            source,
            "if (model_predict_numeric_batch(model_{index}, test_{index}, 24, 8, 3, nullptr, 0, out_{index}, 8) != 0) return 13;"
        )
        .unwrap();
        writeln!(
            source,
            "for (double value : out_{index}) {{ std::uint64_t bits; std::memcpy(&bits, &value, 8); std::printf(\"%016llx\\n\", static_cast<unsigned long long>(bits)); }}"
        )
        .unwrap();
        writeln!(
            source,
            "double invalid_{index}[24]; std::memcpy(invalid_{index}, targets_{index}, sizeof(invalid_{index}));"
        )
        .unwrap();
        writeln!(
            source,
            "for (std::uint64_t bad : {{0x7ff8000000000001ULL, 0x7ff0000000000000ULL, 0xfff0000000000000ULL}}) {{"
        )
        .unwrap();
        writeln!(source, "  invalid_{index}[0] = boundary_from_bits(bad);").unwrap();
        writeln!(source, "  model_handle* rejected = model_{index};").unwrap();
        writeln!(
            source,
            "  if (model_fit_numeric(train_{index}, 72, 24, 3, invalid_{index}, 24, &rejected) != 1 || rejected != nullptr) return 14;"
        )
        .unwrap();
        source.push_str("}\n");
        writeln!(source, "model_unload(model_{index});").unwrap();
    }
    source.push_str("return 0; }\n");

    let scratch_root = std::env::var_os("TMPDIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_TARGET_TMPDIR")));
    fs::create_dir_all(&scratch_root).unwrap();
    let scratch = tempfile::Builder::new()
        .prefix("conversion-boundary-cpp17-")
        .tempdir_in(scratch_root)
        .unwrap();
    let source_path = scratch.path().join("model.cpp");
    let binary = scratch.path().join("model.bin");
    fs::write(&source_path, source).unwrap();
    let compile = Command::new("g++")
        .args(["-std=c++17", "-O2", "-ffp-contract=off", "-fno-fast-math"])
        .arg(&source_path)
        .arg("-o")
        .arg(&binary)
        .output()
        .unwrap();
    assert!(
        compile.status.success(),
        "g++17 O2 failed: {}",
        String::from_utf8_lossy(&compile.stderr)
    );
    let run = Command::new(&binary).output().unwrap();
    assert!(
        run.status.success(),
        "generated C++17 driver exited {}: {}",
        run.status,
        String::from_utf8_lossy(&run.stderr)
    );
    let actual = String::from_utf8(run.stdout)
        .unwrap()
        .lines()
        .map(|line| u64::from_str_radix(line, 16).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(actual.len(), SEEDS.len() * 8);
    assert_eq!(actual, expected, "seeded direct C++17 ABI/evaluator bits");
}
