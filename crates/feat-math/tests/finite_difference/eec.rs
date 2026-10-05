//! GP-created and edited wire-59 candidates retain working export closure.
use feat_math::kernel::finite_difference::{
    DifferenceLanguage, DifferenceSearch, DifferenceState, Edit, ExperimentalProfile,
};
use feat_math::kernel::ordered_axis::{FoldBoundary, OrderedInputIdentity, TaskMode};
use feat_math::kernel::wire_v2::{VersionedModel, VersionedProgram};
use feat_math::GenomeIr;

#[test]
fn grow_change_remove_round_trip_and_compile_three_recipes_and_fitted_cpp() {
    let search = DifferenceSearch::new(
        2,
        OrderedInputIdentity::new("group", "time"),
        GenomeIr::parse("(pipe X (ridge lambda_frac=0.01))").unwrap(),
        TaskMode::Forecasting,
    )
    .unwrap();
    let profile = ExperimentalProfile {
        difference: true,
        prior: 0.125,
    };
    let rows = super::recipes::rows();
    let labels = [0.25, 0.5, 1., 2.];
    let dir = tempfile::tempdir().unwrap().keep();
    let mut executed = 0;
    for seed in [7, 11] {
        let grow = search
            .edit(&search.base(), seed, Edit::Grow, profile)
            .unwrap();
        let change = search.edit(&grow, seed, Edit::Change, profile).unwrap();
        assert_ne!(grow.to_bytes().unwrap(), change.to_bytes().unwrap());
        let removed = search
            .edit(
                &change,
                seed,
                Edit::Drop,
                ExperimentalProfile {
                    difference: false,
                    prior: profile.prior,
                },
            )
            .unwrap();
        assert_eq!(
            removed.to_bytes().unwrap(),
            search.base().to_bytes().unwrap()
        );
        if seed == 7 {
            check_removed(&removed, &dir);
            executed += 4;
        }
        for (name, node) in [("grow", grow), ("change", change)] {
            let bytes = node.to_bytes().unwrap();
            let restored = VersionedProgram::from_bytes(&bytes).unwrap();
            assert_eq!(restored.to_bytes().unwrap(), bytes);
            let text = restored.to_lossless_text().unwrap();
            assert_eq!(
                VersionedProgram::parse_lossless_text(&text)
                    .unwrap()
                    .to_bytes()
                    .unwrap(),
                bytes
            );
            let VersionedProgram::DifferenceV2(ref program) = restored else {
                panic!("created wire59")
            };
            let native = program
                .fit_training(
                    &rows[..4],
                    &labels,
                    TaskMode::Forecasting,
                    FoldBoundary::new("recipe-cut", 4, 8).unwrap(),
                )
                .unwrap();
            let incompatible = vec![DifferenceState::new(
                program.config(),
                2,
                TaskMode::Forecasting,
                program.input_identity(),
                FoldBoundary::new("recipe-cut", 4, 8).unwrap(),
            )
            .unwrap()
            .to_bytes()
            .unwrap()];
            let fitted = VersionedModel::from_json(&native.to_json().unwrap())
                .unwrap()
                .emit_ordered_fitted_cpp17()
                .unwrap();
            let expected = super::recipes::reference(native, &rows);
            for (language, driver, suffix) in [
                (
                    DifferenceLanguage::Rust,
                    super::recipes::rust_driver(&rows, &labels, &incompatible),
                    "rust",
                ),
                (
                    DifferenceLanguage::Python,
                    super::recipes::python_driver(&rows, &labels, &incompatible),
                    "python",
                ),
                (
                    DifferenceLanguage::Cpp17,
                    super::recipes::cpp_driver(&rows, &labels, &incompatible, false),
                    "cpp",
                ),
            ] {
                let recipe = restored.emit_ordered_recipe(language).unwrap();
                assert_eq!(
                    super::recipes::execute(
                        &dir,
                        &format!("seed{seed}-{name}-{suffix}"),
                        &(recipe.source + &driver),
                        language
                    ),
                    expected
                );
                executed += 1;
            }
            assert_eq!(
                super::recipes::execute(
                    &dir,
                    &format!("seed{seed}-{name}-fitted"),
                    &(fitted.source
                        + &super::recipes::cpp_driver(&rows, &labels, &incompatible, true)),
                    DifferenceLanguage::Cpp17
                ),
                expected
            );
            executed += 1;
        }
    }
    assert_eq!(executed, 20);
    eprintln!("wire59 GP closure artifacts: {}", dir.display());
}

fn check_removed(node: &VersionedProgram, dir: &std::path::Path) {
    use feat_math::advanced_v3::RawTable;
    let VersionedProgram::LegacyV1(feat_math::kernel::Program::LegacyV1(genome)) = node else {
        panic!("removed ordered stage")
    };
    let train = vec![vec![0., 1.], vec![1., 2.], vec![2., 4.], vec![3., 8.]];
    let labels = [0.25, 0.5, 1., 2.];
    let test = [vec![4., 16.], vec![5., 32.]];
    let native = feat_math::fit_legacy_numeric_bound(genome, &train, &labels).unwrap();
    let table = RawTable {
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
    let expected = native
        .predict_table(&table)
        .unwrap()
        .iter()
        .map(|v| format!("{:016x}\n", v.to_bits()))
        .collect::<String>();
    let options = feat_math::transpile::EmitOptions::default();
    let recipe = feat_math::transpile::emit(genome, &options).unwrap();
    let rust=format!("mod removed {{\n{}\npub fn predictions()->Vec<f64>{{Model::fit(&[vec![0.,1.],vec![1.,2.],vec![2.,4.],vec![3.,8.]],&[0.25,0.5,1.,2.]).predict(&[vec![4.,16.],vec![5.,32.]])}}\n}}\nfn main(){{for v in removed::predictions(){{println!(\"{{:016x}}\",v.to_bits());}}}}",recipe.rust);
    let python = recipe
        .python
        .replace("if __name__ == \"__main__\":", "if False:")
        + r#"
import struct
model=Model().fit([[0.,1.],[1.,2.],[2.,4.],[3.,8.]],[0.25,0.5,1.,2.])
for value in model.predict([[4.,16.],[5.,32.]]): print('%016x'%struct.unpack('<Q',struct.pack('<d',value))[0])
"#;
    let cpp = feat_math::kernel::expression_recipe::emit_cpp17(genome, &options)
        .unwrap()
        .source
        + r#"
#include <cstdio>
#include <cstring>
int main(){auto model=Model::fit({{0.,1.},{1.,2.},{2.,4.},{3.,8.}},{0.25,0.5,1.,2.});for(double v:model.predict({{4.,16.},{5.,32.}})){std::uint64_t bits;std::memcpy(&bits,&v,8);std::printf("%016llx\n",static_cast<unsigned long long>(bits));}}
"#;
    let fitted = feat_math::advanced_v3::emit_fitted_cpp17(&native)
        .unwrap()
        .source
        + r#"
#include <cstdio>
int main(){const double x[]={4.,16.,5.,32.};const std::uint8_t missing[]={0,0,0,0};double output[2]={};int status=jope_fitted_predict(x,4,missing,4,2,2,output,2);if(status)return status;for(double v:output){std::uint64_t bits;std::memcpy(&bits,&v,8);std::printf("%016llx\n",static_cast<unsigned long long>(bits));}}
"#;
    for (name, source, language) in [
        ("drop-rust", rust, DifferenceLanguage::Rust),
        ("drop-python", python, DifferenceLanguage::Python),
        ("drop-cpp", cpp, DifferenceLanguage::Cpp17),
        ("drop-fitted", fitted, DifferenceLanguage::Cpp17),
    ] {
        assert_eq!(
            super::recipes::execute(dir, name, &source, language),
            expected
        );
    }
}
