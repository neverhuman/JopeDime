//! Ordered recipe and fitted continuation through the public v2 export routes.
use feat_math::kernel::finite_difference::{
    DifferenceConfig, DifferenceFittedCpp, DifferenceLanguage, DifferenceModel, DifferenceProgram,
    DifferenceState, OrderedRow,
};
use feat_math::kernel::ordered_axis::{FoldBoundary, GroupKey, OrderedInputIdentity, TaskMode};
use feat_math::kernel::wire_v2::{export_report, VersionError, VersionedModel, VersionedProgram};
use feat_math::{kernel::Program, GenomeIr};
use std::fs;
use std::path::Path;
use std::process::Command;

pub(super) fn rows() -> Vec<OrderedRow> {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../fixtures/math_boundary/v2/finite_difference.json"
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
fn program(order: u8) -> DifferenceProgram {
    DifferenceProgram::new(
        DifferenceConfig::new(2, 59, order, 0.5).unwrap(),
        2,
        OrderedInputIdentity::new("generation\"\\\t\u{2028}", "time"),
        GenomeIr::parse("(pipe X (ridge lambda_frac=0.01))").unwrap(),
    )
    .unwrap()
}
fn state(model: &DifferenceModel) -> Vec<u8> {
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
pub(super) fn reference(mut model: DifferenceModel, rows: &[OrderedRow]) -> String {
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
        "make_difference_model()".to_string()
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
pub(super) fn execute(dir: &Path, name: &str, source: &str, lang: DifferenceLanguage) -> String {
    let extension = match lang {
        DifferenceLanguage::Rust => "rs",
        DifferenceLanguage::Python => "py",
        DifferenceLanguage::Cpp17 => "cpp",
    };
    let file = dir.join(format!("{name}.{extension}"));
    fs::write(&file, source).unwrap();
    let exe = dir.join(name);
    match lang {
        DifferenceLanguage::Rust => {
            checked(
                Command::new("rustc")
                    .args(["--edition=2021", "-O"])
                    .arg(&file)
                    .arg("-o")
                    .arg(&exe),
            );
            checked(&mut Command::new(exe))
        }
        DifferenceLanguage::Python => checked(
            Command::new("python3")
                .arg(file)
                .env("PYTHONDONTWRITEBYTECODE", "1"),
        ),
        DifferenceLanguage::Cpp17 => {
            checked(
                Command::new(feat_math::cxx::compiler())
                    .args(["-std=c++17", "-O2", "-ffp-contract=off", "-fno-fast-math"])
                    .arg(file)
                    .arg("-o")
                    .arg(&exe),
            );
            checked(&mut Command::new(exe))
        }
    }
}

#[test]
fn four_orders_train_and_continue_through_all_three_public_recipes_and_fitted_cpp() {
    use feat_math::kernel::export::Route;
    let original = rows();
    let mut mixed = original.clone();
    for (i, row) in mixed.iter_mut().enumerate() {
        row.timestamp = i as f64;
        row.group = match i {
            0 => GroupKey::NumericBits((-0.0f64).to_bits()),
            1 => GroupKey::numeric(0.).unwrap(),
            2 | 6 => GroupKey::numeric(-1.).unwrap(),
            3 => GroupKey::numeric(1.).unwrap(),
            4 => GroupKey::numeric(2.5).unwrap(),
            _ => GroupKey::categorical("category").unwrap(),
        };
    }
    let mut negative = original.clone();
    for (i, row) in negative.iter_mut().enumerate() {
        row.values = vec![if i < 4 { 0.0 } else { -(i as f64) }; 2];
    }
    let dir = tempfile::tempdir().unwrap().keep(); // Preserve generated proof artifacts.
    for (case, rows, labels) in [
        ("boundary", original.clone(), vec![0.25, 0.5, 1., 2.]),
        ("groups", mixed, vec![0.25, 0.5, 1., 2.]),
        ("negative-zero", negative, vec![-0.0; 4]),
    ] {
        for order in 1..=4 {
            if case != "boundary" && order != 1 {
                continue;
            }
            let program = program(order);
            let routed = VersionedProgram::from_bytes(&program.to_bytes().unwrap()).unwrap();
            let report = export_report(&routed);
            assert!(matches!(report.rust_recipe, Route::Ok));
            assert!(matches!(report.python_recipe, Route::Ok));
            assert!(matches!(report.cpp17_recipe, Route::Ok));
            assert!(matches!(report.cpp17_fitted, Route::Ok));
            let native = program
                .fit_training(
                    &rows[..4],
                    &labels,
                    TaskMode::Forecasting,
                    FoldBoundary::new("recipe-cut", 4, 8).unwrap(),
                )
                .unwrap();
            let mut incompatible = Vec::new();
            for (other_order, step, width) in [
                (if order == 4 { 1 } else { order + 1 }, 0.5, 2),
                (order, 1., 2),
                (order, 0.5, 1),
            ] {
                let p = DifferenceProgram::new(
                    DifferenceConfig::new(2, 59, other_order, step).unwrap(),
                    width,
                    program.input_identity().clone(),
                    program.downstream().clone(),
                )
                .unwrap();
                let mut training = rows[..4].to_vec();
                for row in &mut training {
                    row.values.truncate(width);
                }
                incompatible.push(state(
                    &p.fit_training(
                        &training,
                        &labels,
                        TaskMode::Forecasting,
                        FoldBoundary::new("recipe-cut", 4, 8).unwrap(),
                    )
                    .unwrap(),
                ));
            }
            incompatible.push(
                DifferenceState::new(
                    program.config(),
                    2,
                    TaskMode::Forecasting,
                    program.input_identity(),
                    FoldBoundary::new("recipe-cut", 4, 8).unwrap(),
                )
                .unwrap()
                .to_bytes()
                .unwrap(),
            );
            let tagged = VersionedModel::from_json(&native.to_json().unwrap()).unwrap();
            let fitted = tagged.emit_ordered_fitted_cpp17().unwrap();
            let expected = reference(native, &rows);
            if case == "negative-zero" {
                assert!(expected.contains("8000000000000000"));
            }
            for (lang, driver, name) in [
                (
                    DifferenceLanguage::Rust,
                    rust_driver(&rows, &labels, &incompatible),
                    "rust",
                ),
                (
                    DifferenceLanguage::Python,
                    python_driver(&rows, &labels, &incompatible),
                    "python",
                ),
                (
                    DifferenceLanguage::Cpp17,
                    cpp_driver(&rows, &labels, &incompatible, false),
                    "cpp",
                ),
            ] {
                let recipe = routed.emit_ordered_recipe(lang).unwrap();
                assert_eq!(recipe.program_identity, program.identity().unwrap());
                assert_eq!(
                    execute(
                        &dir,
                        &format!("{case}-order{order}-{name}"),
                        &(recipe.source + &driver),
                        lang
                    ),
                    expected
                );
            }
            assert_eq!(
                execute(
                    &dir,
                    &format!("{case}-order{order}-fitted"),
                    &(fitted.source + &cpp_driver(&rows, &labels, &incompatible, true)),
                    DifferenceLanguage::Cpp17
                ),
                expected
            );
        }
    }
    eprintln!("ordered recipe proof artifacts: {}", dir.display());
}

#[test]
fn opaque_shapes_stay_parseable_but_never_gain_recipe_support_or_fallback() {
    use feat_math::kernel::export::Route;
    for text in [
        "(pipe X (standardize) (ridge lambda_frac=0.01))",
        "(pipe X (elasticnet lambda_frac=0.01 l1_ratio=0.5))",
    ] {
        let p = DifferenceProgram::new(
            DifferenceConfig::new(2, 59, 1, 1.).unwrap(),
            1,
            OrderedInputIdentity::new("g", "t"),
            GenomeIr::parse(text).unwrap(),
        )
        .unwrap();
        let routed = VersionedProgram::from_bytes(&p.to_bytes().unwrap()).unwrap();
        assert_eq!(routed.to_bytes().unwrap(), p.to_bytes().unwrap());
        let report = export_report(&routed);
        assert!(matches!(report.rust_recipe, Route::Unsupported(_)));
        assert!(matches!(report.python_recipe, Route::Unsupported(_)));
        assert!(matches!(report.cpp17_recipe, Route::Unsupported(_)));
        assert!(matches!(report.cpp17_fitted, Route::Unsupported(_)));
        for lang in [
            DifferenceLanguage::Rust,
            DifferenceLanguage::Python,
            DifferenceLanguage::Cpp17,
        ] {
            assert!(routed.emit_ordered_recipe(lang).is_err());
        }
    }
    // Parsing remains available even when fixed metadata alone cannot fit a
    // native model. Export admission must refuse all routes before emission.
    let large = DifferenceProgram::new(
        DifferenceConfig::new(2, 59, 1, 1.).unwrap(),
        1,
        OrderedInputIdentity::new("g".repeat(3_000_000), "t"),
        GenomeIr::parse("(pipe X (ridge lambda_frac=0.01))").unwrap(),
    )
    .unwrap();
    let bytes = large.to_bytes().unwrap();
    let restored = VersionedProgram::from_bytes(&bytes).unwrap();
    assert_eq!(restored.to_bytes().unwrap(), bytes);
    assert!(large.validate_recipe_closure().is_err());
    let report = export_report(&restored);
    assert!(matches!(report.rust_recipe, Route::Unsupported(_)));
    assert!(matches!(report.python_recipe, Route::Unsupported(_)));
    assert!(matches!(report.cpp17_recipe, Route::Unsupported(_)));
    assert!(matches!(report.cpp17_fitted, Route::Unsupported(_)));
    for language in [
        DifferenceLanguage::Rust,
        DifferenceLanguage::Python,
        DifferenceLanguage::Cpp17,
    ] {
        assert!(restored.emit_ordered_recipe(language).is_err());
    }
    let legacy = VersionedProgram::LegacyV1(Program::LegacyV1(
        GenomeIr::parse("(pipe X (ridge lambda_frac=0.01))").unwrap(),
    ));
    assert!(legacy
        .emit_ordered_recipe(DifferenceLanguage::Rust)
        .is_err());
}

#[test]
fn fitted_codec_binds_source_profile_program_and_native_continuation() {
    let rows = rows();
    let model = program(2)
        .fit_training(
            &rows[..4],
            &[0.25, 0.5, 1., 2.],
            TaskMode::Forecasting,
            FoldBoundary::new("recipe-cut", 4, 8).unwrap(),
        )
        .unwrap();
    let artifact = model.emit_fitted_cpp17().unwrap();
    let replay = DifferenceFittedCpp::from_codec_json(&artifact.codec).unwrap();
    assert_eq!(replay.source, artifact.source);
    assert_eq!(replay.codec, artifact.codec);
    let value: serde_json::Value = serde_json::from_str(&artifact.codec).unwrap();
    for (field, replacement) in [
        ("schema", serde_json::json!("unknown")),
        ("format_version", serde_json::json!(3)),
        ("min_reader_version", serde_json::json!(3)),
        ("required_features", serde_json::json!(64 | 128)),
        ("profile", serde_json::json!("unknown")),
    ] {
        let mut unsupported = value.clone();
        unsupported[field] = replacement;
        assert!(
            matches!(
                DifferenceFittedCpp::from_codec_json(&unsupported.to_string()),
                Err(VersionError::UnsupportedVersion(_))
            ),
            "unsupported {field} must refuse before checkpoint admission"
        );
    }
    for (field, replacement) in [
        ("schema", serde_json::json!("unknown")),
        ("min_reader_version", serde_json::json!(3)),
        ("required_features", serde_json::json!(0)),
        ("profile", serde_json::json!("unknown")),
        ("source_sha256", serde_json::json!("0".repeat(64))),
        ("program_identity", serde_json::json!("0".repeat(64))),
        ("model_json", serde_json::json!("{}")),
        ("unknown", serde_json::json!(true)),
    ] {
        let mut forged = value.clone();
        forged[field] = replacement;
        assert!(
            DifferenceFittedCpp::from_codec_json(&forged.to_string()).is_err(),
            "{field}"
        );
    }
    assert!(DifferenceFittedCpp::from_codec_json("{}").is_err());
    let oversized = format!("{}{}", artifact.codec, " ".repeat(16 * 1024 * 1024));
    assert!(
        matches!(DifferenceFittedCpp::from_codec_json(&oversized),Err(VersionError::Malformed(message)) if message.starts_with("Resource("))
    );
}
