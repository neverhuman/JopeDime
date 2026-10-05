//! Ordered recipe and fitted continuation through the public v2 export routes.
use feat_math::kernel::ordered_axis::{FoldBoundary, GroupKey, OrderedInputIdentity, TaskMode};
use feat_math::kernel::timescale_nabla::{
    Config as TimeGradientConfig, OrderedRow, TimeGradientFittedCpp, TimeGradientLanguage,
    TimeGradientModel, TimeGradientProgram, TimeGradientState,
};
use feat_math::GenomeIr;
use std::fs;
use std::path::Path;
use std::process::Command;

pub(super) fn rows() -> Vec<OrderedRow> {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/math_boundary/v2/timescale_nabla.json"
    ))
    .unwrap();
    let float = |v: &serde_json::Value| {
        f64::from_bits(u64::from_str_radix(v.as_str().unwrap(), 16).unwrap())
    };
    fixture["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| OrderedRow {
            group: GroupKey::categorical(row["group"].as_str().unwrap()).unwrap(),
            timestamp: float(&row["time_bits"]),
            values: row["values_bits"]
                .as_array()
                .unwrap()
                .iter()
                .map(float)
                .collect(),
        })
        .collect()
}
fn program(gap: f64) -> TimeGradientProgram {
    TimeGradientProgram::new(
        TimeGradientConfig::new(2, 75, 2, gap).unwrap(),
        OrderedInputIdentity::new("generation\"\\\t\u{2028}", "time"),
        GenomeIr::parse("(pipe X (ridge lambda_frac=0.01))").unwrap(),
    )
    .unwrap()
}

fn state(model: &TimeGradientModel) -> Vec<u8> {
    serde_json::from_str::<serde_json::Value>(&model.to_json().unwrap()).unwrap()["state"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u8)
        .collect()
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
pub(super) fn reference(mut model: TimeGradientModel, rows: &[OrderedRow]) -> String {
    let initial = state(&model);
    let prediction = model.predict_with_faults(&rows[4..]).unwrap();
    let text = format!(
        "initial:{}\nvalues:{}\nfaults:{:?}\nfinal:{}\n",
        hex(&initial),
        prediction
            .values
            .iter()
            .map(|v| format!("{:016x}", v.to_bits()))
            .collect::<Vec<_>>()
            .join(","),
        prediction.faults,
        hex(&state(&model))
    );
    model.finish().unwrap();
    text
}
fn checked(command: &mut Command) -> String {
    let output = command.output().expect("required recipe tool must launch");
    assert!(
        output.status.success(),
        "{command:?}\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}
pub(super) fn rust_driver(rows: &[OrderedRow], labels: &[f64], incompatible: &[Vec<u8>]) -> String {
    let labels = labels
        .iter()
        .map(|v| format!("f64::from_bits({})", v.to_bits()))
        .collect::<Vec<_>>()
        .join(",");
    let incompatible = format!(
        "vec![{}]",
        incompatible
            .iter()
            .map(|bytes| format!("vec!{bytes:?}"))
            .collect::<Vec<_>>()
            .join(",")
    );
    let rows = rows
        .iter()
        .map(|row| {
            let group = match &row.group {
                GroupKey::Categorical(text) => format!("OrderedGroup::Text({text:?}.into())"),
                GroupKey::NumericBits(bits) => format!("OrderedGroup::Numeric({bits})"),
            };
            format!(
                "OrderedRow{{group:{group},timestamp:f64::from_bits({}),values:vec![{}]}}",
                row.timestamp.to_bits(),
                row.values
                    .iter()
                    .map(|v| format!("f64::from_bits({})", v.to_bits()))
                    .collect::<Vec<_>>()
                    .join(",")
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!(
        r#"
fn output_hex(b:&[u8])->String {{b.iter().map(|v|format!("{{v:02x}}")).collect()}}
fn main() {{
 let rows=vec![{rows}];let labels=[{labels}];
 let mut model=fit_ordered(&rows[..4],&labels,2,"recipe-cut",4,8).unwrap();
 let initial=model.state_bytes().unwrap();assert!(model.finish().is_err());
 let empty=model.predict(&[]).unwrap();assert!(empty.0.is_empty() && empty.1.is_empty());assert_eq!(initial,model.state_bytes().unwrap());
 let mut bad=rows[4..6].to_vec();bad[1].values.clear();assert!(model.predict(&bad).is_err());assert_eq!(initial,model.state_bytes().unwrap());
 let mut truncated=initial.clone();truncated.pop();assert!(model.replace_state(&truncated).is_err());assert_eq!(initial,model.state_bytes().unwrap());
 for bytes in {incompatible} {{assert!(model.replace_state(&bytes).is_err());assert_eq!(initial,model.state_bytes().unwrap());}}
 let mut backwards=vec![rows[4].clone(),rows[0].clone()];backwards[1].timestamp=-f64::MAX;assert!(model.predict(&backwards).is_err());assert_eq!(initial,model.state_bytes().unwrap());
 backwards[1].timestamp=f64::NAN;assert!(model.predict(&backwards).is_err());assert_eq!(initial,model.state_bytes().unwrap());
 model.predict(&rows[4..5]).unwrap();assert_ne!(initial,model.state_bytes().unwrap());
 model.replace_state(&initial).unwrap();assert_eq!(initial,model.state_bytes().unwrap());
 println!("initial:{{}}",output_hex(&initial));let (values,faults)=model.predict(&rows[4..]).unwrap();
 println!("values:{{}}",values.iter().map(|v|format!("{{:016x}}",v.to_bits())).collect::<Vec<_>>().join(","));
 println!("faults:{{faults:?}}");println!("final:{{}}",output_hex(&model.state_bytes().unwrap()));model.finish().unwrap();
 assert!(fit_ordered(&rows,&[0.;8],2,"recipe-cut",4,8).is_err());
 assert!(fit_ordered(&rows[..4],&[f64::MAX;4],2,"recipe-cut",4,8).is_err());
}}
"#
    )
}
pub(super) fn python_driver(
    rows: &[OrderedRow],
    labels: &[f64],
    incompatible: &[Vec<u8>],
) -> String {
    let labels = labels
        .iter()
        .map(|v| format!("word({})", v.to_bits()))
        .collect::<Vec<_>>()
        .join(",");
    let incompatible = format!("{incompatible:?}");
    let rows = rows
        .iter()
        .map(|row| {
            let group = match &row.group {
                GroupKey::Categorical(text) => {
                    format!("(1,{})", serde_json::to_string(text).unwrap())
                }
                GroupKey::NumericBits(bits) => format!("(0,{bits})"),
            };
            format!(
                "OrderedRow({},word({}),[{}])",
                group,
                row.timestamp.to_bits(),
                row.values
                    .iter()
                    .map(|v| format!("word({})", v.to_bits()))
                    .collect::<Vec<_>>()
                    .join(",")
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!(
        r#"
def word(bits): return struct.unpack('<d',struct.pack('<Q',bits))[0]
rows=[{rows}]
model=fit_ordered(rows[:4],[{labels}],'forecasting','recipe-cut',4,8)
initial=model.state_bytes()
def refused(action):
 try: action()
 except OrderedError: return
 raise AssertionError('operation should refuse')
refused(model.finish)
assert model.predict([])==([],[]) and model.state_bytes()==initial
bad=[rows[4],OrderedRow(rows[5].group,rows[5].timestamp,[])]
refused(lambda:model.predict(bad));assert model.state_bytes()==initial
refused(lambda:model.replace_state(initial[:-1]));assert model.state_bytes()==initial
for data in {incompatible}:
 refused(lambda:model.replace_state(bytes(data)));assert model.state_bytes()==initial
backwards=[rows[4],OrderedRow(rows[0].group,-word(0x7fefffffffffffff),rows[0].values)]
refused(lambda:model.predict(backwards));assert model.state_bytes()==initial
backwards[1]=OrderedRow(rows[0].group,float('nan'),rows[0].values)
refused(lambda:model.predict(backwards));assert model.state_bytes()==initial
model.predict(rows[4:5]);assert model.state_bytes()!=initial
model.replace_state(initial);assert model.state_bytes()==initial
print('initial:'+initial.hex())
values,faults=model.predict(rows[4:])
print('values:'+','.join(format(struct.unpack('<Q',struct.pack('<d',v))[0],'016x') for v in values))
print('faults:'+str(faults));print('final:'+model.state_bytes().hex());model.finish()
refused(lambda:fit_ordered(rows,[0.]*8,'forecasting','recipe-cut',4,8))
refused(lambda:fit_ordered(rows[:4],[word(0x7fefffffffffffff)]*4,'forecasting','recipe-cut',4,8))
"#
    )
}
pub(super) fn cpp_driver(
    rows: &[OrderedRow],
    labels: &[f64],
    incompatible: &[Vec<u8>],
    fitted: bool,
) -> String {
    let labels = labels
        .iter()
        .map(|v| format!("ordered_float({}ULL)", v.to_bits()))
        .collect::<Vec<_>>()
        .join(",");
    let incompatible = incompatible
        .iter()
        .map(|bytes| {
            format!(
                "{{{}}}",
                bytes
                    .iter()
                    .map(|b| b.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let rows = rows
        .iter()
        .map(|row| {
            let group = match &row.group {
                GroupKey::Categorical(text) => format!(
                    "OrderedGroup::categorical({})",
                    serde_json::to_string(text).unwrap()
                ),
                GroupKey::NumericBits(bits) => {
                    format!("OrderedGroup::numeric(ordered_float({bits}ULL))")
                }
            };
            format!(
                "{{{},ordered_float({}ULL),{{{}}}}}",
                group,
                row.timestamp.to_bits(),
                row.values
                    .iter()
                    .map(|v| format!("ordered_float({}ULL)", v.to_bits()))
                    .collect::<Vec<_>>()
                    .join(",")
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let factory = if fitted {
        "make_time_gradient_model()".to_string()
    } else {
        format!("fit_ordered(std::vector<OrderedRow>(rows.begin(),rows.begin()+4),{{{labels}}},2,\"recipe-cut\",4,8)")
    };
    let fit_checks = if fitted {
        ""
    } else {
        r#"
 refused=false;try{fit_ordered(rows,std::vector<double>(8,0.),2,"recipe-cut",4,8);}catch(const OrderedError&){refused=true;}if(!refused)return 8;
 refused=false;try{fit_ordered(std::vector<OrderedRow>(rows.begin(),rows.begin()+4),std::vector<double>(4,std::numeric_limits<double>::max()),2,"recipe-cut",4,8);}catch(const OrderedError&){refused=true;}if(!refused)return 9;
"#
    };
    format!(
        r#"
#include <iostream>
#include <sstream>
#include <iomanip>
static std::string output_hex(const OrderedBytes& bytes) {{std::ostringstream out;for(auto b:bytes)out<<std::hex<<std::setfill('0')<<std::setw(2)<<unsigned(b);return out.str();}}
int main() {{
 const std::vector<OrderedRow> rows={{{rows}}};auto model={factory};auto other={factory};auto initial=model.state_bytes();
 bool refused=false;try{{model.finish();}}catch(const OrderedError&){{refused=true;}}if(!refused)return 2;
 auto empty=model.predict({{}});if(!empty.first.empty()||!empty.second.empty()||model.state_bytes()!=initial)return 3;
 std::vector<OrderedRow> bad={{rows[4],rows[5]}};bad[1].values.clear();refused=false;try{{model.predict(bad);}}catch(const OrderedError&){{refused=true;}}if(!refused||model.state_bytes()!=initial)return 4;
 auto truncated=initial;truncated.pop_back();refused=false;try{{model.replace_state(truncated);}}catch(const OrderedError&){{refused=true;}}if(!refused||model.state_bytes()!=initial)return 5;
 const std::vector<OrderedBytes> incompatible={{{incompatible}}};for(const auto& bytes:incompatible){{refused=false;try{{model.replace_state(bytes);}}catch(const OrderedError&){{refused=true;}}if(!refused||model.state_bytes()!=initial)return 11;}}
 std::vector<OrderedRow> backwards={{rows[4],rows[0]}};backwards[1].timestamp=-std::numeric_limits<double>::max();refused=false;try{{model.predict(backwards);}}catch(const OrderedError&){{refused=true;}}if(!refused||model.state_bytes()!=initial)return 12;
 backwards[1].timestamp=std::numeric_limits<double>::quiet_NaN();refused=false;try{{model.predict(backwards);}}catch(const OrderedError&){{refused=true;}}if(!refused||model.state_bytes()!=initial)return 13;
 model.predict({{rows[4]}});if(model.state_bytes()==initial)return 14;model.replace_state(initial);if(model.state_bytes()!=initial)return 15;const auto continuation=std::vector<OrderedRow>(rows.begin()+4,rows.end());const auto result=model.predict(continuation);if(other.state_bytes()!=initial)return 6;
 const auto replay=other.predict(continuation);if(result.first.size()!=replay.first.size()||result.second!=replay.second)return 7;for(std::size_t i=0;i<result.first.size();i++)if(ordered_bits(result.first[i])!=ordered_bits(replay.first[i]))return 10;
 std::cout<<"initial:"<<output_hex(initial)<<'\n'<<"values:";for(std::size_t i=0;i<result.first.size();i++){{if(i)std::cout<<',';std::cout<<std::hex<<std::setfill('0')<<std::setw(16)<<ordered_bits(result.first[i]);}}
 std::cout<<std::dec<<"\nfaults:[";for(std::size_t i=0;i<result.second.size();i++){{if(i)std::cout<<", ";std::cout<<'[';for(std::size_t j=0;j<result.second[i].size();j++){{if(j)std::cout<<", ";std::cout<<result.second[i][j];}}std::cout<<']';}}
 std::cout<<"]\nfinal:"<<output_hex(model.state_bytes())<<'\n';model.finish();{fit_checks}
}}
"#
    )
}
pub(super) fn execute(dir: &Path, name: &str, source: &str, lang: TimeGradientLanguage) -> String {
    let extension = match lang {
        TimeGradientLanguage::Rust => "rs",
        TimeGradientLanguage::Python => "py",
        TimeGradientLanguage::Cpp17 => "cpp",
    };
    let file = dir.join(format!("{name}.{extension}"));
    fs::write(&file, source).unwrap();
    let exe = dir.join(name);
    match lang {
        TimeGradientLanguage::Rust => {
            checked(
                Command::new("rustc")
                    .args(["--edition=2021", "-O"])
                    .arg(&file)
                    .arg("-o")
                    .arg(&exe),
            );
            checked(&mut Command::new(exe))
        }
        TimeGradientLanguage::Python => checked(
            Command::new("python3")
                .arg(file)
                .env("PYTHONDONTWRITEBYTECODE", "1"),
        ),
        TimeGradientLanguage::Cpp17 => {
            let standard = std::env::var("JOPE_CPP_STD").unwrap_or_else(|_| "17".into());
            let sanitizer = std::env::var("JOPE_FITTED_CPP_SANITIZER").ok();
            let mut compiler = Command::new(feat_math::cxx::compiler());
            compiler.arg(format!("-std=c++{standard}")).args([
                "-O2",
                "-ffp-contract=off",
                "-fno-fast-math",
            ]);
            if let Some(ref flags) = sanitizer {
                compiler
                    .args(["-O1", "-g", "-fno-omit-frame-pointer"])
                    .arg(format!("-fsanitize={flags}"));
            }
            checked(compiler.arg(file).arg("-o").arg(&exe));
            let mut runner = Command::new(exe);
            if sanitizer.is_some() {
                runner
                    .env("ASAN_OPTIONS", "detect_leaks=0:halt_on_error=1")
                    .env("UBSAN_OPTIONS", "halt_on_error=1:print_stacktrace=1");
            }
            checked(&mut runner)
        }
    }
}

#[test]
fn irregular_recipes_fitted_and_checkpoints_match_native_bits() {
    let dir = tempfile::tempdir().unwrap().keep();
    let mut receipts = Vec::new();
    for gap in [0.125, 0.25] {
        for case in ["boundary", "numeric_groups", "minus_zero_labels"] {
            let p = program(gap);
            let mut rows = rows();
            rows[5].timestamp = 1.5;
            if case == "numeric_groups" {
                for row in &mut rows {
                    row.group =
                        GroupKey::numeric(if row.group == GroupKey::categorical("a").unwrap() {
                            -0.
                        } else {
                            3.
                        })
                        .unwrap();
                }
            }
            let labels = if case == "minus_zero_labels" {
                vec![-0.; 4]
            } else {
                vec![1., 2., 3., 4.]
            };
            let fold = FoldBoundary::new("recipe-cut", 4, 8).unwrap();
            let native = p
                .fit_training(&rows[..4], &labels, TaskMode::Forecasting, fold.clone())
                .unwrap();
            let fitted =
                feat_math::kernel::wire_v2::VersionedModel::from_json(&native.to_json().unwrap())
                    .unwrap()
                    .emit_time_gradient_fitted_cpp17()
                    .unwrap();
            assert_eq!(
                TimeGradientFittedCpp::from_codec_json(&fitted.codec)
                    .unwrap()
                    .source,
                fitted.source
            );
            let expected = reference(native, &rows);
            let mut incompatible = Vec::new();
            for (width, other_gap) in [(2, gap / 2.), (1, gap)] {
                let mut state = TimeGradientState::new(
                    TimeGradientConfig::new(2, 75, width, other_gap).unwrap(),
                    TaskMode::Forecasting,
                    p.input_identity(),
                    fold.clone(),
                )
                .unwrap();
                for row in &rows[..4] {
                    state
                        .push(row.group.clone(), row.timestamp, &row.values[..width])
                        .unwrap();
                }
                incompatible.push(state.to_bytes().unwrap());
            }
            for lang in [
                TimeGradientLanguage::Rust,
                TimeGradientLanguage::Python,
                TimeGradientLanguage::Cpp17,
            ] {
                let source =
                    feat_math::kernel::wire_v2::VersionedProgram::TimeGradientV2(p.clone())
                        .emit_time_gradient_recipe(lang)
                        .unwrap()
                        .source;
                let driver = match lang {
                    TimeGradientLanguage::Rust => rust_driver(&rows, &labels, &incompatible),
                    TimeGradientLanguage::Python => python_driver(&rows, &labels, &incompatible),
                    TimeGradientLanguage::Cpp17 => cpp_driver(&rows, &labels, &incompatible, false),
                };
                let name = format!("gap{:016x}-{case}-{lang:?}", gap.to_bits());
                let actual = execute(&dir, &name, &format!("{source}\n{driver}"), lang);
                assert_eq!(actual.replace(' ', ""), expected.replace(' ', ""), "{name}");
                fs::write(dir.join(format!("{name}.stdout")), &actual).unwrap();
                receipts.push(name);
            }
            let name = format!("gap{:016x}-{case}-fitted", gap.to_bits());
            let actual = execute(
                &dir,
                &name,
                &format!(
                    "{}\n{}",
                    fitted.source,
                    cpp_driver(&rows, &labels, &incompatible, true)
                ),
                TimeGradientLanguage::Cpp17,
            );
            assert_eq!(actual.replace(' ', ""), expected.replace(' ', ""), "{name}");
            fs::write(dir.join(format!("{name}.stdout")), actual).unwrap();
            receipts.push(name);
        }
    }
    fs::write(
        dir.join("artifacts.json"),
        serde_json::to_vec_pretty(&receipts).unwrap(),
    )
    .unwrap();
    println!("24 compiled B15 recipes/fitted at {}", dir.display());
}
