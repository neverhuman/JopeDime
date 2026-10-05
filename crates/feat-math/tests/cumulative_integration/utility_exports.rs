//! Execute each distinct utility winner on its own recorded training split.
use feat_math::kernel::cumulative_integration::{
    IntegrationLanguage, IntegrationProgram, OrderedRow,
};
use feat_math::kernel::ordered_axis::{FoldBoundary, GroupKey, TaskMode};
use feat_math::kernel::wire_v2::VersionedProgram;
use std::path::Path;

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|v| format!("{v:02x}")).collect()
}
fn bits(values: &[f64]) -> String {
    values
        .iter()
        .map(|v| format!("{:016x}\n", v.to_bits()))
        .collect()
}
fn words(values: &[f64], language: IntegrationLanguage) -> String {
    values
        .iter()
        .map(|v| match language {
            IntegrationLanguage::Rust => format!("f64::from_bits({})", v.to_bits()),
            IntegrationLanguage::Python => format!("word({})", v.to_bits()),
            IntegrationLanguage::Cpp17 => format!("utility_float({}ULL)", v.to_bits()),
        })
        .collect::<Vec<_>>()
        .join(",")
}
fn ordered_rows(rows: &[OrderedRow], language: IntegrationLanguage) -> String {
    rows.iter().map(|r| {
        let GroupKey::Categorical(group)=&r.group else {panic!("categorical utility fixture")};
        let name=serde_json::to_string(group).unwrap();
        let values=words(&r.values,language);
        match language {
            IntegrationLanguage::Rust=>format!("OrderedRow{{group:OrderedGroup::Text({name}.into()),timestamp:f64::from_bits({}),values:vec![{values}]}}",r.timestamp.to_bits()),
            IntegrationLanguage::Python=>format!("OrderedRow((1,{name}),word({}),[{values}])",r.timestamp.to_bits()),
            IntegrationLanguage::Cpp17=>format!("{{OrderedGroup::categorical({name}),utility_float({}ULL),{{{values}}}}}",r.timestamp.to_bits()),
        }
    }).collect::<Vec<_>>().join(",")
}
const CPP_WORDS: &str = r#"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cstring>
static double utility_float(std::uint64_t bits){double v;std::memcpy(&v,&bits,8);return v;}
static std::string utility_hex(const OrderedBytes& bytes){std::ostringstream s;for(auto b:bytes)s<<std::hex<<std::setfill('0')<<std::setw(2)<<unsigned(b);return s.str();}
"#;

fn ordered(
    program: &IntegrationProgram,
    rows: &[OrderedRow],
    labels: &[f64],
    train: usize,
    dir: &Path,
    name: &str,
) {
    let model = program
        .fit_training(
            &rows[..train],
            &labels[..train],
            TaskMode::Forecasting,
            FoldBoundary::new("utility-artifact", train, rows.len()).unwrap(),
        )
        .unwrap();
    let fitted = model.emit_fitted_cpp17().unwrap();
    let mut native = model;
    let before: Vec<u8> = serde_json::from_value(
        serde_json::from_str::<serde_json::Value>(&native.to_json().unwrap()).unwrap()["state"]
            .clone(),
    )
    .unwrap();
    let output = native.predict_with_faults(&rows[train..]).unwrap();
    let after: Vec<u8> = serde_json::from_value(
        serde_json::from_str::<serde_json::Value>(&native.to_json().unwrap()).unwrap()["state"]
            .clone(),
    )
    .unwrap();
    let expected = format!(
        "initial:{}\nvalues:{}\nfaults:{:?}\nfinal:{}\n",
        hex(&before),
        output
            .values
            .iter()
            .map(|v| format!("{:016x}", v.to_bits()))
            .collect::<Vec<_>>()
            .join(","),
        output.faults,
        hex(&after)
    );
    native.finish().unwrap();
    for language in [
        IntegrationLanguage::Rust,
        IntegrationLanguage::Python,
        IntegrationLanguage::Cpp17,
    ] {
        let raw = ordered_rows(rows, language);
        let y = words(&labels[..train], language);
        let total = rows.len();
        let driver = match language {
            IntegrationLanguage::Rust => format!(
                r#"
fn utility_hex(b:&[u8])->String{{b.iter().map(|v|format!("{{v:02x}}")).collect()}}
fn main(){{let rows=vec![{raw}];let mut model=fit_ordered(&rows[..{train}],&[{y}],2,"utility-artifact",{train},{total}).unwrap();println!("initial:{{}}",utility_hex(&model.state_bytes().unwrap()));let (values,faults)=model.predict(&rows[{train}..]).unwrap();println!("values:{{}}",values.iter().map(|v|format!("{{:016x}}",v.to_bits())).collect::<Vec<_>>().join(","));println!("faults:{{faults:?}}");println!("final:{{}}",utility_hex(&model.state_bytes().unwrap()));model.finish().unwrap();}}
"#
            ),
            IntegrationLanguage::Python => format!(
                r#"
def word(bits):return struct.unpack('<d',struct.pack('<Q',bits))[0]
rows=[{raw}]
model=fit_ordered(rows[:{train}],[{y}],'forecasting','utility-artifact',{train},{total})
print('initial:'+model.state_bytes().hex())
values,faults=model.predict(rows[{train}:])
print('values:'+','.join(format(struct.unpack('<Q',struct.pack('<d',v))[0],'016x') for v in values))
print('faults:'+str(faults));print('final:'+model.state_bytes().hex());model.finish()
"#
            ),
            IntegrationLanguage::Cpp17 => cpp_driver(&raw, &y, train, total, false),
        };
        let suffix = match language {
            IntegrationLanguage::Rust => "rust",
            IntegrationLanguage::Python => "python",
            IntegrationLanguage::Cpp17 => "cpp",
        };
        assert_eq!(
            super::recipes::execute(
                dir,
                &format!("{name}-{suffix}"),
                &(program.emit_recipe(language).unwrap().source + &driver),
                language
            ),
            expected
        );
    }
    let raw = ordered_rows(rows, IntegrationLanguage::Cpp17);
    assert_eq!(
        super::recipes::execute(
            dir,
            &format!("{name}-fitted"),
            &(fitted.source + &cpp_driver(&raw, "", train, rows.len(), true)),
            IntegrationLanguage::Cpp17
        ),
        expected
    );
}
fn cpp_driver(raw: &str, labels: &str, train: usize, total: usize, fitted: bool) -> String {
    let factory = if fitted {
        "make_integral_model()".into()
    } else {
        format!("fit_ordered(std::vector<OrderedRow>(rows.begin(),rows.begin()+{train}),{{{labels}}},2,\"utility-artifact\",{train},{total})")
    };
    format!(
        r#"{CPP_WORDS}
int main(){{const std::vector<OrderedRow> rows={{{raw}}};auto model={factory};std::cout<<"initial:"<<utility_hex(model.state_bytes())<<'\n';const auto output=model.predict(std::vector<OrderedRow>(rows.begin()+{train},rows.end()));std::cout<<"values:";for(std::size_t i=0;i<output.first.size();++i){{if(i)std::cout<<',';std::cout<<std::hex<<std::setfill('0')<<std::setw(16)<<ordered_bits(output.first[i]);}}std::cout<<std::dec<<"\nfaults:[";for(std::size_t i=0;i<output.second.size();++i){{if(i)std::cout<<", ";std::cout<<'[';for(std::size_t j=0;j<output.second[i].size();++j){{if(j)std::cout<<", ";std::cout<<output.second[i][j];}}std::cout<<']';}}std::cout<<"]\nfinal:"<<utility_hex(model.state_bytes())<<'\n';model.finish();}}
"#
    )
}

pub(super) fn check(
    node: &VersionedProgram,
    rows: &[OrderedRow],
    labels: &[f64],
    train: usize,
    dir: &Path,
    name: &str,
) {
    match node {
        VersionedProgram::IntegrationV2(program) => {
            ordered(program, rows, labels, train, dir, name)
        }
        VersionedProgram::LegacyV1(feat_math::kernel::Program::LegacyV1(genome)) => {
            legacy(genome, rows, labels, train, dir, name)
        }
        _ => panic!("utility profile admits only its two arms"),
    }
}

fn legacy(
    genome: &feat_math::GenomeIr,
    rows: &[OrderedRow],
    labels: &[f64],
    train: usize,
    dir: &Path,
    name: &str,
) {
    let x = rows[..train]
        .iter()
        .map(|r| r.values.clone())
        .collect::<Vec<_>>();
    let test = rows[train..]
        .iter()
        .map(|r| r.values.clone())
        .collect::<Vec<_>>();
    let native = feat_math::fit_legacy_numeric_bound(genome, &x, &labels[..train]).unwrap();
    let table = feat_math::advanced_v3::RawTable {
        headers: native
            .schema
            .columns
            .iter()
            .map(|c| c.name.clone())
            .collect(),
        rows: test
            .iter()
            .map(|r| r.iter().map(f64::to_string).collect())
            .collect(),
    };
    let expected = bits(&native.predict_table(&table).unwrap());
    let options = feat_math::transpile::EmitOptions::default();
    let emitted = feat_math::transpile::emit(genome, &options).unwrap();
    let r_train = x
        .iter()
        .map(|r| format!("vec![{}]", words(r, IntegrationLanguage::Rust)))
        .collect::<Vec<_>>()
        .join(",");
    let r_test = test
        .iter()
        .map(|r| format!("vec![{}]", words(r, IntegrationLanguage::Rust)))
        .collect::<Vec<_>>()
        .join(",");
    let r_y = words(&labels[..train], IntegrationLanguage::Rust);
    let rust=format!("mod utility {{\n{}\npub fn predict()->Vec<f64>{{Model::fit(&[{r_train}],&[{r_y}]).predict(&[{r_test}])}}\n}}\nfn main(){{for value in utility::predict(){{println!(\"{{:016x}}\",value.to_bits());}}}}",emitted.rust);
    let p_train = x
        .iter()
        .map(|r| format!("[{}]", words(r, IntegrationLanguage::Python)))
        .collect::<Vec<_>>()
        .join(",");
    let p_test = test
        .iter()
        .map(|r| format!("[{}]", words(r, IntegrationLanguage::Python)))
        .collect::<Vec<_>>()
        .join(",");
    let p_y = words(&labels[..train], IntegrationLanguage::Python);
    let python = emitted
        .python
        .replace("if __name__ == \"__main__\":", "if False:")
        + &format!(
            r#"
import struct
def word(v):return struct.unpack('<d',struct.pack('<Q',v))[0]
model=Model().fit([{p_train}],[{p_y}])
for value in model.predict([{p_test}]):print('%016x'%struct.unpack('<Q',struct.pack('<d',value))[0])
"#
        );
    let c_train = x
        .iter()
        .map(|r| format!("{{{}}}", words(r, IntegrationLanguage::Cpp17)))
        .collect::<Vec<_>>()
        .join(",");
    let c_test = test
        .iter()
        .map(|r| format!("{{{}}}", words(r, IntegrationLanguage::Cpp17)))
        .collect::<Vec<_>>()
        .join(",");
    let c_y = words(&labels[..train], IntegrationLanguage::Cpp17);
    let primitive = r#"
#include <cstdio>
#include <cstring>
static double utility_float(std::uint64_t v){double out;std::memcpy(&out,&v,8);return out;}
static void utility_print(double value){std::uint64_t bits;std::memcpy(&bits,&value,8);std::printf("%016llx\n",static_cast<unsigned long long>(bits));}
"#;
    let cpp=feat_math::kernel::expression_recipe::emit_cpp17(genome,&options).unwrap().source+primitive+&format!("\nint main(){{auto model=Model::fit({{{c_train}}},{{{c_y}}});for(double value:model.predict({{{c_test}}}))utility_print(value);}}\n");
    let cells = test.iter().flatten().copied().collect::<Vec<_>>();
    let data = words(&cells, IntegrationLanguage::Cpp17);
    let count = cells.len();
    let total = test.len();
    let fitted=feat_math::advanced_v3::emit_fitted_cpp17(&native).unwrap().source+primitive+&format!("\nint main(){{const double values[]={{ {data} }};const std::uint8_t missing[{count}]={{}};double output[{total}]={{}};int status=jope_fitted_predict(values,{count},missing,{count},{total},2,output,{total});if(status)return status;for(double v:output)utility_print(v);}}\n");
    for (suffix, source, language) in [
        ("rust", rust, IntegrationLanguage::Rust),
        ("python", python, IntegrationLanguage::Python),
        ("cpp", cpp, IntegrationLanguage::Cpp17),
        ("fitted", fitted, IntegrationLanguage::Cpp17),
    ] {
        assert_eq!(
            super::recipes::execute(dir, &format!("{name}-{suffix}"), &source, language),
            expected
        );
    }
}
