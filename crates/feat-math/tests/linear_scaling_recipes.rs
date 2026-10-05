//! Explicit training recipes; outer-fold labels never enter their API.
use feat_math::kernel::linear_scaling::{emit_training_recipe, LinearScaledWinner};
use feat_math::math_expr::MathExpr;
use std::fmt::Write;
use std::fs;
use std::path::Path;
use std::process::Command;

#[path = "support/linear_scaling_signed_zero.rs"]
mod signed_zero;
use signed_zero::{encoded, table};

#[test]
fn training_recipe_admission_preserves_hard_error_refusals() {
    for (source, width) in [
        ("x", 1),
        ("c1", 1),
        ("c0", 0),
        ("c0", 513),
        ("clip(c0,c1*c1,100)", 2),
    ] {
        assert!(emit_training_recipe(&MathExpr::parse(source).unwrap(), width).is_err());
    }
    assert!(emit_training_recipe(&MathExpr::parse("c0").unwrap(), 1).is_ok());
    for case in signed_zero::stored_clip_cases() {
        assert!(
            emit_training_recipe(&MathExpr::parse(case.source).unwrap(), case.train[0].len(),)
                .is_err()
        );
    }
}

#[test]
fn calibration_extraction_preserves_existing_winner_words() {
    let rows = vec![vec![-2.0], vec![-1.0], vec![0.0], vec![1.0], vec![2.0]];
    let fit = LinearScaledWinner::fit_training(
        &MathExpr::parse("c0").unwrap(),
        &rows,
        &[1.0, 4.0, 7.0, 10.0, 13.0],
    )
    .unwrap();
    assert_eq!(fit.winner.coefficients(), [3.0, 7.0]);
    assert_eq!(fit.training_mse.to_bits(), 0.0_f64.to_bits());
}

#[test]
fn translated_feature_families_preserve_exact_affine_coefficients() {
    let expression = MathExpr::parse("c0").unwrap();
    for anchor in [
        -1e12_f64, -1e6, -100.0, -3.0, -1.0, 1.0, 3.0, 10.0, 100.0, 1e6, 1e12,
    ] {
        let magnitude = anchor.abs();
        let ulp = f64::from_bits(magnitude.to_bits() + 1) - magnitude;
        let slope = 1.0 / ulp;
        let intercept = -slope * anchor;
        let direction = if anchor > 0.0 { -1.0 } else { 1.0 };
        for width in [1, 2, 7, 32, 512] {
            for length in 2..=32 {
                let rows = (0..length)
                    .map(|i| {
                        let mut row = vec![0.0; width];
                        row[0] = anchor + direction * f64::from(i) * ulp;
                        row
                    })
                    .collect::<Vec<_>>();
                let labels = (0..length)
                    .map(|i| direction * f64::from(i))
                    .collect::<Vec<_>>();
                let fit = LinearScaledWinner::fit_training(&expression, &rows, &labels).unwrap();
                assert_eq!(
                    fit.winner.coefficients(),
                    [slope, intercept],
                    "anchor={anchor}, width={width}, length={length}"
                );
                assert_eq!(fit.training_mse, 0.0);
                let mut outside = vec![0.0; width];
                outside[0] = anchor + direction * 32.0 * ulp;
                assert_eq!(fit.winner.predict(&[outside]).unwrap(), [direction * 32.0]);
            }
        }
    }
}

#[test]
fn representable_subnormal_training_loss_is_not_a_perfect_fit() {
    let expression = MathExpr::parse("2.0").unwrap();
    for length in [2, 100, 8192] {
        let labels = (0..length)
            .map(|i| if i & 1 == 0 { -1e-161 } else { 1e-161 })
            .collect::<Vec<_>>();
        let fit = LinearScaledWinner::fit_training(&expression, &vec![vec![1.0]; length], &labels)
            .unwrap();
        assert_eq!(fit.winner.coefficients(), [0.0, 0.0]);
        // Independent exact-rational rounding of the input words: 20 minimum subnormals.
        assert_eq!(fit.training_mse.to_bits(), 0x14, "length={length}");
    }
    assert!(LinearScaledWinner::fit_training(
        &expression,
        &[vec![1.0], vec![1.0]],
        &[-1e-300, 1e-300]
    )
    .is_err());
}

fn run(command: &mut Command) -> String {
    let output = command.output().unwrap();
    assert!(
        output.status.success(),
        "{command:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

#[test]
fn training_recipes_refuse_an_unrepresentable_nonzero_slope() {
    let root = std::env::var_os("TMPDIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")));
    std::fs::create_dir_all(&root).unwrap();
    let scratch = tempfile::tempdir_in(root).unwrap();
    let dir = scratch.path();
    let artifact = emit_training_recipe(&MathExpr::parse("c0").unwrap(), 1).unwrap();
    let rows = vec![vec![0.0], vec![-1e12], vec![1e12], vec![1e-300]];
    let labels = [0.0, 0.0, 0.0, 1.0];
    let model = dir.join("model.rs");
    let library = dir.join("librecipe.rlib");
    fs::write(&model, artifact.rust).unwrap();
    run(Command::new("rustc")
        .args([
            "--edition=2021",
            "-O",
            "--crate-type=lib",
            "--crate-name=recipe",
        ])
        .arg(&model)
        .arg("-o")
        .arg(&library));
    let driver = dir.join("driver.rs");
    fs::write(&driver, format!(
        "extern crate recipe; fn main() {{ assert!(matches!(recipe::scaled_train(&{}, &{}), Err(\"linear-scaling coefficients\"))); }}",
        table(&rows, 0), encoded(&labels, 0),
    )).unwrap();
    let rust_binary = dir.join("rust-bin");
    run(Command::new("rustc")
        .args(["--edition=2021", "-O", "--extern"])
        .arg(format!("recipe={}", library.display()))
        .arg(&driver)
        .arg("-o")
        .arg(&rust_binary));
    run(&mut Command::new(rust_binary));
    fs::write(dir.join("model.py"), artifact.python).unwrap();
    fs::write(dir.join("driver.py"), format!(
        "from struct import unpack\nfrom model import scaled_train\ntry: scaled_train({}, {})\nexcept ValueError as error: assert str(error) == 'linear-scaling coefficients'\nelse: raise AssertionError('unrepresentable nonzero slope admitted')\n",
        table(&rows, 1), encoded(&labels, 1),
    )).unwrap();
    run(Command::new("python3")
        .arg(dir.join("driver.py"))
        .env("PYTHONDONTWRITEBYTECODE", "1"));
    let cpp = dir.join("driver.cpp");
    fs::write(&cpp, format!(
        "{}\n#include <string>\nstatic double from_bits(std::uint64_t bits) {{ double value; std::memcpy(&value,&bits,8); return value; }}\nint main() {{ try {{ scaled_train({}, {}); }} catch(const std::invalid_argument& error) {{ return std::string(error.what()) == \"linear-scaling coefficients\" ? 0 : 2; }} return 1; }}\n",
        artifact.cpp17, table(&rows, 2), encoded(&labels, 2),
    )).unwrap();
    let binary = dir.join("cpp-bin");
    let mut compiler = Command::new("g++");
    compiler.arg(format!(
        "-std=c++{}",
        std::env::var("JOPE_CPP_STD").unwrap_or_else(|_| "17".into())
    ));
    compiler.args(["-O2", "-ffp-contract=off", "-fno-fast-math"]);
    if let Ok(sanitizer) = std::env::var("JOPE_FITTED_CPP_SANITIZER") {
        compiler.arg(format!("-fsanitize={sanitizer}"));
        compiler.args(["-fno-omit-frame-pointer", "-fno-pie", "-no-pie"]);
    }
    run(compiler.arg(&cpp).arg("-o").arg(&binary));
    run(&mut Command::new(binary));
}

fn execute_recipe(
    dir: &Path,
    source: &str,
    train: &[Vec<f64>],
    labels: &[f64],
    test: &[Vec<f64>],
) -> [String; 3] {
    fs::create_dir_all(dir).unwrap();
    let artifacts =
        emit_training_recipe(&MathExpr::parse(source).unwrap(), train[0].len()).unwrap();
    assert_eq!(
        artifacts.profile,
        "feat-math.scalar-linear-scaling-recipe.v2"
    );
    let width = train[0].len();
    let model = dir.join("model.rs");
    let library = dir.join("librecipe.rlib");
    fs::write(&model, artifacts.rust).unwrap();
    run(Command::new("rustc")
        .args([
            "--edition=2021",
            "-O",
            "--crate-type=lib",
            "--crate-name=recipe",
        ])
        .arg(&model)
        .arg("-o")
        .arg(&library));
    let driver = dir.join("driver.rs");
    fs::write(&driver,format!(r#"extern crate recipe;
fn main() {{
    let train={}; let labels={}; let test={};
    let (model,mse)=recipe::scaled_train(&train,&labels).unwrap();
    for v in model.coefficients() {{ println!("{{:016x}}",v.to_bits()); }}
    println!("{{:016x}}",mse.to_bits());
    let bytes=model.to_bytes(); for b in &bytes {{ print!("{{b:02x}}"); }} println!();
    let restored=recipe::ScaledModel::from_bytes(&bytes).unwrap();
    assert_eq!(bytes,restored.to_bytes());
    for v in restored.predict(&test).unwrap() {{ println!("{{:016x}}",v.to_bits()); }}
    assert_eq!(bytes,restored.to_bytes());
    assert!(restored.predict(&[]).unwrap().is_empty());
    for (rows,y) in [(vec![],vec![]),(vec![vec![]],vec![1.0]),(train.clone(),vec![]),(train.clone(),vec![f64::NAN;train.len()]),(vec![vec![f64::INFINITY;{width}]],vec![1.0]),(vec![vec![1.0;{width}];2],vec![-f64::MAX,f64::MAX]),(vec![vec![1.0;{width}];2],vec![-1e-300,1e-300])] {{ assert!(recipe::scaled_train(&rows,&y).is_err()); }}
    for bad in [vec![vec![]],vec![vec![f64::NAN;{width}]]] {{ assert!(restored.predict(&bad).is_err()); }}
    for length in 0..bytes.len() {{ assert!(recipe::ScaledModel::from_bytes(&bytes[..length]).is_err()); }}
    let mut bad=bytes.clone(); bad.push(0); assert!(recipe::ScaledModel::from_bytes(&bad).is_err());
    for offset in [8,9,10,18,42] {{ let mut bad=bytes.clone(); bad[offset]^=1; assert!(recipe::ScaledModel::from_bytes(&bad).is_err()); }}
    for offset in [26,34] {{ let mut bad=bytes.clone(); bad[offset..offset+8].copy_from_slice(&f64::NAN.to_bits().to_le_bytes()); assert!(recipe::ScaledModel::from_bytes(&bad).is_err()); }}
}}
"#,table(train,0),encoded(labels,0),table(test,0))).unwrap();
    let rs_bin = dir.join("rust-bin");
    run(Command::new("rustc")
        .args(["--edition=2021", "-O", "--extern"])
        .arg(format!("recipe={}", library.display()))
        .arg(&driver)
        .arg("-o")
        .arg(&rs_bin));
    let rust = run(&mut Command::new(&rs_bin));
    fs::write(dir.join("model.py"), artifacts.python).unwrap();
    fs::write(dir.join("driver.py"),format!(r#"from struct import pack,unpack
from model import scaled_train,ScaledModel
train={}
labels={}
test={}
model,mse=scaled_train(train,labels)
for v in model.coefficients(): print(pack('>d',v).hex())
print(pack('>d',mse).hex())
data=model.to_bytes()
print(data.hex())
restored=ScaledModel.from_bytes(data)
assert data==restored.to_bytes()
for v in restored.predict(test): print(pack('>d',v).hex())
assert data==restored.to_bytes() and restored.predict([])==[]
for rows,y in [([],[]),([[]],[1.0]),(train,[]),(train,[float('nan')]*len(train)),([[float('inf')]*{width}],[1.0]),([[1.0]*{width}]*2,[-float.fromhex('0x1.fffffffffffffp1023'),float.fromhex('0x1.fffffffffffffp1023')]),([[1.0]*{width}]*2,[-1e-300,1e-300])]:
    try: scaled_train(rows,y)
    except ValueError: pass
    else: raise AssertionError('invalid training admitted')
for bad in [[[]],[[float('nan')]*{width}]]:
    try: restored.predict(bad)
    except ValueError: pass
    else: raise AssertionError('invalid prediction admitted')
bad_states=[data[:i] for i in range(len(data))]+[data+b'\0']
for offset in [8,9,10,18,42]:
    bad=bytearray(data); bad[offset]^=1; bad_states.append(bytes(bad))
for offset in [26,34]:
    bad=bytearray(data); bad[offset:offset+8]=pack('<d',float('nan')); bad_states.append(bytes(bad))
for bad in bad_states:
    try: ScaledModel.from_bytes(bad)
    except ValueError: pass
    else: raise AssertionError('invalid state admitted')
"#,table(train,1),encoded(labels,1),table(test,1))).unwrap();
    let python = run(Command::new("python3")
        .arg(dir.join("driver.py"))
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .env("LC_ALL", "C"));
    let cpp = dir.join("driver.cpp");
    fs::write(&cpp,format!(r#"{}
#include <cstdio>
#include <string>
static double from_bits(std::uint64_t b) {{ double v; std::memcpy(&v,&b,8); return v; }}
static void print_bits(double v) {{ std::uint64_t b; std::memcpy(&b,&v,8); std::printf("%016llx\n",static_cast<unsigned long long>(b)); }}
template<class F> static void refuses(F function) {{ bool refused=false; try {{function();}} catch(const std::invalid_argument&) {{refused=true;}} if(!refused) throw std::runtime_error("invalid data admitted"); }}
int main() {{
    const std::vector<std::vector<double>> train={},test={}; const std::vector<double> labels={};
    const auto fit=scaled_train(train,labels); const auto model=fit.first;
    for (double v:model.coefficients()) print_bits(v); print_bits(fit.second);
    const auto bytes=model.to_bytes(); for(auto b:bytes) std::printf("%02x",b); std::puts("");
    const auto restored=ScaledModel::from_bytes(bytes);
    if(bytes!=restored.to_bytes())return 2;
    for(double v:restored.predict(test))print_bits(v);
    if(bytes!=restored.to_bytes() || !restored.predict({{}}).empty())return 3;
    refuses([&]{{scaled_train({{}},{{}});}}); refuses([&]{{scaled_train({{{{}}}},{{1.0}});}}); refuses([&]{{scaled_train(train,{{}});}});
    refuses([&]{{scaled_train(train,std::vector<double>(train.size(),std::numeric_limits<double>::quiet_NaN()));}});
    refuses([&]{{scaled_train({{std::vector<double>({width},std::numeric_limits<double>::infinity())}},{{1.0}});}});
    refuses([&]{{scaled_train(std::vector<std::vector<double>>(2,std::vector<double>({width},1.0)),{{-std::numeric_limits<double>::max(),std::numeric_limits<double>::max()}});}});
    refuses([&]{{scaled_train(std::vector<std::vector<double>>(2,std::vector<double>({width},1.0)),{{-1e-300,1e-300}});}});
    refuses([&]{{restored.predict({{{{}}}});}}); refuses([&]{{restored.predict({{std::vector<double>({width},std::numeric_limits<double>::quiet_NaN())}});}});
    for(std::size_t i=0;i<bytes.size();++i)refuses([&]{{ScaledModel::from_bytes(std::vector<std::uint8_t>(bytes.begin(),bytes.begin()+i));}});
    auto bad=bytes;bad.push_back(0);refuses([&]{{ScaledModel::from_bytes(bad);}});
    for(unsigned offset:{{8,9,10,18,42}}){{bad=bytes;bad[offset]^=1;refuses([&]{{ScaledModel::from_bytes(bad);}});}}
    for(unsigned offset:{{26,34}}){{bad=bytes;const std::uint64_t word=0x7ff8000000000000ULL;for(unsigned j=0;j<8;++j)bad[offset+j]=static_cast<std::uint8_t>(word>>(j*8));refuses([&]{{ScaledModel::from_bytes(bad);}});}}
}}
"#,artifacts.cpp17,table(train,2),table(test,2),encoded(labels,2))).unwrap();
    let executable = dir.join("cpp-bin");
    let mut compiler = Command::new("g++");
    let std = std::env::var("JOPE_CPP_STD").unwrap_or_else(|_| "17".into());
    compiler.arg(format!("-std=c++{std}"));
    compiler.args(["-O2", "-ffp-contract=off", "-fno-fast-math"]);
    if let Ok(sanitizer) = std::env::var("JOPE_FITTED_CPP_SANITIZER") {
        compiler.arg(format!("-fsanitize={sanitizer}"));
        compiler.args(["-fno-omit-frame-pointer", "-fno-pie", "-no-pie"]);
    }
    run(compiler.arg(&cpp).arg("-o").arg(&executable));
    let cpp17 = run(&mut Command::new(executable));
    [rust, python, cpp17]
}

#[test]
fn independently_trained_recipes_match_winner_envelopes_loss_and_prediction_bits() {
    let root = std::env::var_os("TMPDIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")));
    fs::create_dir_all(&root).unwrap();
    let scratch = tempfile::Builder::new()
        .prefix("scaled-training-")
        .tempdir_in(root)
        .unwrap();
    let train: Vec<_> = (-3..=3)
        .map(|x| vec![f64::from(x), f64::from(x * x) * 0.5])
        .collect();
    let test = vec![
        vec![-0.0, 0.0],
        vec![f64::from_bits(1), -2.0],
        vec![100.0, 1.0],
        vec![-7.0, 3.0],
    ];
    let mut cases = Vec::new();
    for source in [
        "c0",
        "c0+0.5",
        "c0*c0",
        "sin(c0)",
        "row_sum",
        "2.0",
        "min(c0,c1)",
        "step(c0,c1)",
        "abs(step(c0,c1))",
        "min(step(c0,c1),c1)",
        "step(c0,c1)^2",
        "c0^step(c0,c1)",
        "sqrt(-step(c0,c1))",
    ] {
        let expr = MathExpr::parse(source).unwrap();
        let labels = train
            .iter()
            .map(|r| 3.0 * expr.eval_row(r).unwrap() + 7.0)
            .collect::<Vec<_>>();
        cases.push((source, train.clone(), labels, test.clone()));
    }
    cases.push((
        "c0",
        vec![vec![2.0]; 4],
        vec![1.0, 3.0, 5.0, 7.0],
        vec![vec![100.0]],
    ));
    cases.push((
        "c0",
        vec![vec![-1.0], vec![0.0], vec![1.0]],
        vec![-f64::MAX, 0.0, f64::MAX],
        vec![vec![-0.5], vec![0.5]],
    ));
    cases.push((
        "c0",
        vec![vec![-1e-300], vec![0.0], vec![1e-300]],
        vec![-1.0, 0.0, 1.0],
        vec![vec![2e-300]],
    ));
    cases.push((
        "2.0",
        vec![vec![1.0]; 3],
        vec![f64::MAX; 3],
        vec![vec![2.0]],
    ));
    let anchor = 1e12_f64;
    let ulp = f64::from_bits(anchor.to_bits() + 1) - anchor;
    cases.push((
        "c0",
        vec![vec![anchor - 2.0 * ulp], vec![anchor - ulp], vec![anchor]],
        vec![-2.0, -1.0, 0.0],
        vec![vec![anchor - 4.0 * ulp]],
    ));
    cases.push((
        "2.0",
        vec![vec![1.0]; 100],
        (0..100)
            .map(|i| if i & 1 == 0 { -1e-161 } else { 1e-161 })
            .collect(),
        vec![vec![2.0]],
    ));
    for (i, (source, train, labels, test)) in cases.into_iter().enumerate() {
        let fit =
            LinearScaledWinner::fit_training(&MathExpr::parse(source).unwrap(), &train, &labels)
                .unwrap();
        let mut expected = String::new();
        for v in fit
            .winner
            .coefficients()
            .into_iter()
            .chain([fit.training_mse])
        {
            writeln!(expected, "{:016x}", v.to_bits()).unwrap();
        }
        let bytes = fit.winner.to_bytes().unwrap();
        for b in &bytes {
            write!(expected, "{b:02x}").unwrap();
        }
        expected.push('\n');
        for v in fit.winner.predict(&test).unwrap() {
            writeln!(expected, "{:016x}", v.to_bits()).unwrap();
        }
        for (backend, actual) in execute_recipe(
            &scratch.path().join(i.to_string()),
            source,
            &train,
            &labels,
            &test,
        )
        .into_iter()
        .enumerate()
        {
            assert_eq!(actual, expected, "case {i} {source} backend {backend}");
        }
        assert_eq!(bytes, fit.winner.to_bytes().unwrap());
    }
}

#[test]
fn trained_recipe_compositions_preserve_signed_zero_state_and_prediction_bits() {
    let root = std::env::var_os("TMPDIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")));
    fs::create_dir_all(&root).unwrap();
    let scratch = tempfile::Builder::new()
        .prefix("scaled-zero-recipe-")
        .tempdir_in(root)
        .unwrap();
    for (index, case) in signed_zero::cases().into_iter().enumerate() {
        let expression = MathExpr::parse(case.source).unwrap();
        let fit = LinearScaledWinner::fit_training(&expression, &case.train, &case.labels).unwrap();
        let bytes = fit.winner.to_bytes().unwrap();
        let mut expected = String::new();
        for value in fit
            .winner
            .coefficients()
            .into_iter()
            .chain([fit.training_mse])
        {
            writeln!(expected, "{:016x}", value.to_bits()).unwrap();
        }
        for b in &bytes {
            write!(expected, "{b:02x}").unwrap();
        }
        expected.push('\n');
        for value in fit.winner.predict(&case.test).unwrap() {
            writeln!(expected, "{:016x}", value.to_bits()).unwrap();
        }
        for (backend, actual) in execute_recipe(
            &scratch.path().join(index.to_string()),
            case.source,
            &case.train,
            &case.labels,
            &case.test,
        )
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                actual,
                expected,
                "{} width {} backend {backend}",
                case.source,
                case.train[0].len()
            );
        }
        assert_eq!(fit.winner.to_bytes().unwrap(), bytes);
    }
}
