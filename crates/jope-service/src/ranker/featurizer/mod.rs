//! DSL-only inputs to the frozen 412-coordinate ranker recipe (§5.8).
//!
//! RNK-03 and RNK-04 implement the structural, parameter and position families.

pub mod parameters;
pub mod parse;
pub mod positions;
pub mod structural;

use anyhow::{ensure, Result};
use std::collections::{BTreeMap, BTreeSet};

pub const FEATURE_COUNT: usize = 412;
pub const REFERENCE_SOURCE_SHA256: &str =
    "534169af0443122600853ef9e96d734d570cd9d77c2a3d21e7b6456b4b19ab40";
pub const FEATURE_NAMES_SHA256: &str =
    "c4e4373d1bc7e99bc38e44eb4acda22d0f9ecc9e332fa8cb9c6a0b463fe37962";

/// The caller supplies the authenticated bundle's name order, without sorting.
/// This is an internal projection helper, not a bundle authentication boundary.
pub struct Featurizer {
    names: [String; FEATURE_COUNT],
}

impl Default for Featurizer {
    fn default() -> Self {
        Self {
            names: FEATURE_NAMES.map(str::to_owned),
        }
    }
}

impl Featurizer {
    pub fn from_names(names: Vec<String>) -> Result<Self> {
        ensure!(names.len() == FEATURE_COUNT, "expected 412 feature names");
        let unique: BTreeSet<_> = names.iter().map(String::as_str).collect();
        ensure!(unique.len() == FEATURE_COUNT, "duplicate feature name");
        ensure!(
            unique == FEATURE_NAMES.into_iter().collect(),
            "feature names differ from the frozen recipe"
        );
        Ok(Self {
            names: names.try_into().expect("length checked above"),
        })
    }

    pub fn feature_names(&self) -> &[String; FEATURE_COUNT] {
        &self.names
    }

    /// Families include explicit zeroes for their own absent counts/flags.
    /// Other families' missing keys remain None, never implicit model inputs.
    /// Unknown dictionary keys are ignored, as in the reference's name projection.
    pub fn project(&self, features: &BTreeMap<String, f64>) -> PartialFeatures {
        PartialFeatures {
            values: std::array::from_fn(|i| features.get(&self.names[i]).copied()),
        }
    }

    pub fn structural(&self, dsl: &str) -> Result<PartialFeatures> {
        let program = parse::parse(dsl)?;
        Ok(self.project(&structural::extract(&program)))
    }

    /// Extract all three frozen feature families and project them in the
    /// authenticated bundle order before the reference FP32 rounding step.
    pub fn featurize(&self, dsl: &str) -> Result<[f32; FEATURE_COUNT]> {
        let program = parse::parse(dsl)?;
        self.project(&extract_all(&program)?).require_complete()
    }
}

fn extract_all(program: &parse::ParsedProgram<'_>) -> Result<BTreeMap<String, f64>> {
    let mut features = structural::extract(program);
    for family in [parameters::extract(program), positions::extract(program)] {
        for (name, value) in family {
            ensure!(
                features.insert(name.clone(), value).is_none(),
                "duplicate feature across frozen families: {name}"
            );
        }
    }
    ensure!(
        features.len() == FEATURE_COUNT,
        "expected {FEATURE_COUNT} composed features, got {}",
        features.len()
    );
    Ok(features)
}

#[derive(Debug)]
pub struct PartialFeatures {
    values: [Option<f64>; FEATURE_COUNT],
}

impl PartialFeatures {
    pub fn values(&self) -> &[Option<f64>; FEATURE_COUNT] {
        &self.values
    }

    /// The reference wrapper rounds to FP32 before lifting into FP64 proj16.
    pub fn require_complete(self) -> Result<[f32; FEATURE_COUNT]> {
        let missing = self.values.iter().filter(|v| v.is_none()).count();
        ensure!(
            missing == 0,
            "incomplete feat412: {missing} coordinates missing"
        );
        let values = self
            .values
            .map(|v| v.expect("all coordinates present") as f32);
        ensure!(
            values.iter().all(|v| v.is_finite()),
            "nonfinite feat412 value"
        );
        Ok(values)
    }
}

#[cfg(test)]
mod test_support {
    use super::*;
    use serde::Deserialize;
    use std::{fs, path::PathBuf};

    #[derive(Deserialize)]
    pub struct ProgramRecord {
        pub context_sha256: String,
        pub dsl: String,
        pub program_sha256: String,
        pub dl_split: Option<String>,
    }

    pub fn custody() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../research/flywheel_selfplay/custody/files")
            .join("home")
            .join("ubuntu")
    }

    pub fn pinned(path: PathBuf, sha256: &str) -> Vec<u8> {
        let bytes = fs::read(&path).unwrap();
        assert_eq!(
            jope_runtime::sha256_hex(&bytes),
            sha256,
            "{}",
            path.display()
        );
        bytes
    }

    pub fn names() -> Vec<String> {
        let path = custody().join("JopeDimeMicro/assets/projections/program-feature-names.json");
        serde_json::from_slice(&pinned(path, FEATURE_NAMES_SHA256)).unwrap()
    }

    pub fn corpus() -> Vec<ProgramRecord> {
        let root = custody().join("jope-data/res-01/input");
        let mut rows: Vec<ProgramRecord> = serde_json::from_slice(&pinned(
            root.join("champions.json"),
            "a444047ffb937dc8d69012cbdbf0891faf3e7d743b127e7cac3baed114cf141d",
        ))
        .unwrap();
        assert_eq!(rows.len(), 1219);
        assert!(rows
            .iter()
            .all(|r| r.dl_split.as_deref() == Some("DL-Train")));
        let train: BTreeSet<_> = rows.iter().map(|r| r.context_sha256.clone()).collect();
        let winners: serde_json::Value = serde_json::from_slice(&pinned(
            root.join("winners.json"),
            "f38b2732d2e28f73972a5c09990029bfdcee45d7f2128609d00dc53625e58f44",
        ))
        .unwrap();
        let winners: Vec<ProgramRecord> =
            serde_json::from_value(winners["all_90_qualified_novel_winner_records"].clone())
                .unwrap();
        assert_eq!(winners.len(), 90);
        assert!(winners.iter().all(|r| train.contains(&r.context_sha256)));
        rows.extend(winners);
        assert_eq!(rows.len(), 1309);
        assert_eq!(
            rows.iter()
                .map(|r| &r.program_sha256)
                .collect::<BTreeSet<_>>()
                .len(),
            1292
        );
        for row in &rows {
            assert_eq!(
                jope_runtime::sha256_hex(row.dsl.as_bytes()),
                row.program_sha256
            );
        }
        rows
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_order_and_reference_are_the_pinned_custody_recipe() {
        assert_eq!(FEATURE_NAMES.to_vec(), test_support::names());
        test_support::pinned(
            test_support::custody().join("JopeDimeMicro/python/rank_prepare.py"),
            REFERENCE_SOURCE_SHA256,
        );
    }

    #[test]
    fn incomplete_families_are_not_silent_zero_filled_ranker_inputs() {
        let extracted = Featurizer::default()
            .structural("(pipe X (ridge))")
            .unwrap();
        assert_eq!(
            extracted.values().iter().filter(|v| v.is_some()).count(),
            30
        );
        assert!(extracted
            .require_complete()
            .unwrap_err()
            .to_string()
            .contains("382"));
    }

    #[test]
    fn bundle_order_is_preserved_and_bad_name_sets_are_refused() {
        let mut names = test_support::names();
        names.reverse();
        let f = Featurizer::from_names(names.clone()).unwrap();
        let result = f.structural("(pipe X (ridge))").unwrap();
        assert_eq!(f.feature_names(), names.as_slice());
        assert_eq!(result.values()[0], Some(1.0)); // steps is last in the frozen order.
        names.pop();
        assert!(Featurizer::from_names(names).is_err());
        let mut names = test_support::names();
        names[1] = names[0].clone();
        assert!(Featurizer::from_names(names).is_err());
        let mut names = test_support::names();
        names[0] = "unknown.feature".into();
        assert!(Featurizer::from_names(names).is_err());
    }

    #[test]
    fn projection_ignores_unknown_keys_and_refuses_nonfinite_complete_inputs() {
        let f = Featurizer::default();
        let mut values: BTreeMap<_, _> = FEATURE_NAMES.iter().map(|&n| (n.into(), 1.0)).collect();
        values.insert("unknown.feature".into(), f64::NAN);
        assert_eq!(
            f.project(&values).require_complete().unwrap(),
            [1.0; FEATURE_COUNT]
        );
        values.insert("steps".into(), f64::INFINITY);
        assert!(f.project(&values).require_complete().is_err());
    }

    #[test]
    fn complete_input_rounds_to_fp32_before_projection_and_refuses_cast_overflow() {
        let f = Featurizer::default();
        let name = "parameter.ridge.lambda_frac.sum";
        let index = FEATURE_NAMES.iter().position(|&n| n == name).unwrap();
        let mut values: BTreeMap<_, _> = FEATURE_NAMES.iter().map(|&n| (n.into(), 0.0)).collect();
        let unrounded = 1.0 + 2.0_f64.powi(-25);
        assert_ne!(unrounded, 1.0);
        values.insert(name.into(), unrounded);
        let complete = f.project(&values).require_complete().unwrap();
        assert_eq!(f64::from(complete[index]), 1.0);
        values.insert(name.into(), f64::MAX);
        assert!(f.project(&values).require_complete().is_err());
    }

    #[test]
    fn full_composition_covers_all_1309_reference_records_in_frozen_order() {
        let rows = test_support::corpus();
        let featurizer = Featurizer::default();
        let mut reversed_names = FEATURE_NAMES.map(str::to_owned).to_vec();
        reversed_names.reverse();
        let reversed = Featurizer::from_names(reversed_names).unwrap();

        for (record, row) in rows.iter().enumerate() {
            let actual = featurizer.featurize(&row.dsl).unwrap_or_else(|error| {
                panic!("record {record}, {}: {error:#}", row.program_sha256)
            });
            let reversed_actual = reversed.featurize(&row.dsl).unwrap();
            assert!(actual.iter().all(|value| value.is_finite()));
            assert_eq!(
                actual.iter().rev().copied().collect::<Vec<_>>(),
                reversed_actual.to_vec(),
                "bundle order for record {record}, {}",
                row.program_sha256
            );

            // The component tests bind all 412 named coordinates against the
            // Python oracle for every record; this checks their full union is
            // present at the corresponding frozen feat412 index.
            let program = parse::parse(&row.dsl).unwrap();
            let all = extract_all(&program).unwrap();
            assert_eq!(all.len(), FEATURE_COUNT);
            for (index, name) in FEATURE_NAMES.iter().enumerate() {
                assert_eq!(
                    actual[index], all[*name] as f32,
                    "record {record}, {}: {name}",
                    row.program_sha256
                );
            }
        }
        assert_eq!(rows.len(), 1309);
    }

    /// Exercise serialized Python outputs, independently of the Rust families.
    /// This model-prior subset does not establish full historical M3 coverage.
    #[test]
    #[ignore = "requires pinned RES-07 model-prior program/feature fixtures"]
    fn real_model_prior_structural_features_match_frozen_oracle() {
        use safetensors::{Dtype, SafeTensors};
        use std::{env, fs, path::PathBuf};

        let root =
            PathBuf::from(env::var_os("JOPE_GOLDEN_DIR").expect("JOPE_GOLDEN_DIR is required"));
        let index_pin = env::var("JOPE_GOLDEN_INDEX_SHA256")
            .expect("explicit trusted index SHA256 is required");
        assert!(
            index_pin.len() == 64
                && index_pin
                    .bytes()
                    .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
        );
        let read_pinned = |name: &str, pin: &str| {
            let path = root.join(name);
            let metadata = fs::symlink_metadata(&path).expect("required real fixture");
            assert!(metadata.is_file() && metadata.len() <= 64 * 1024 * 1024);
            let bytes = fs::read(path).unwrap();
            assert_eq!(
                jope_runtime::sha256_hex(&bytes),
                pin,
                "fixture pin differs: {name}"
            );
            bytes
        };
        let index: serde_json::Value =
            serde_json::from_slice(&read_pinned("index.json", &index_pin)).unwrap();
        assert_eq!(index["schema"], "jope.golden-fixtures/v1");
        assert_eq!(
            index["bundle_digest"],
            "sha256:94fd9587675c7806bd0a0582b6e97540cd2fdc7ebbc51958411b905185cad85c"
        );
        assert_eq!(index["model_prior_programs"], 872);
        let payload = |name: &str| {
            let spec = &index["files"][name];
            let bytes = read_pinned(name, spec["sha256"].as_str().expect("missing fixture pin"));
            assert_eq!(bytes.len() as u64, spec["bytes"].as_u64().unwrap());
            bytes
        };
        let programs = payload("programs.jsonl");
        let programs: Vec<serde_json::Value> = std::str::from_utf8(&programs)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(programs.len(), 872);
        let oracle = payload("programs.safetensors");
        let tensors = SafeTensors::deserialize(&oracle).unwrap();
        let features = tensors.tensor("feat412").unwrap();
        assert_eq!(features.dtype(), Dtype::F32);
        assert_eq!(features.shape(), [programs.len(), FEATURE_COUNT]);
        let featurizer = Featurizer::default();
        let mut unique = BTreeSet::new();
        let mut comparisons = 0;
        for (row, program) in programs.iter().enumerate() {
            let dsl = program["dsl"].as_str().expect("DSL required");
            let pin = program["program_sha256"]
                .as_str()
                .expect("program identity required");
            assert_eq!(jope_runtime::sha256_hex(dsl.as_bytes()), pin);
            assert!(unique.insert(pin));
            let actual = featurizer
                .structural(dsl)
                .unwrap_or_else(|e| panic!("{pin}: {e:#}"));
            let mut per_program = 0;
            for (column, name) in FEATURE_NAMES.iter().enumerate() {
                if !structural::is_feature(name) {
                    assert!(actual.values()[column].is_none());
                    continue;
                }
                let offset = (row * FEATURE_COUNT + column) * 4;
                let expected =
                    f32::from_le_bytes(features.data()[offset..offset + 4].try_into().unwrap());
                let observed =
                    actual.values()[column].expect("missing structural coordinate") as f32;
                assert!(expected.is_finite() && observed.is_finite());
                assert_eq!(observed, expected, "program {row} {pin}: {name}");
                per_program += 1;
                comparisons += 1;
            }
            assert_eq!(per_program, 30);
        }
        assert_eq!(comparisons, 872 * 30);
        println!(
            "{}",
            serde_json::json!({
                "gate":"RNK-03 structural serialized-oracle parity", "scope":"frozen model_prior subset",
                "programs":programs.len(),"coordinates_per_program":30,"comparisons":comparisons,"mismatches":0,
                "index_sha256":index_pin,"full_historical_M3":false,
                "compiled_source_sha256":{
                    "mod.rs":jope_runtime::sha256_hex(include_str!("mod.rs").as_bytes()),
                    "parse.rs":jope_runtime::sha256_hex(include_str!("parse.rs").as_bytes()),
                    "structural.rs":jope_runtime::sha256_hex(include_str!("structural.rs").as_bytes())
                }
            })
        );
    }
}

/// Frozen name order, generated from the hash-pinned custody metadata above.
pub const FEATURE_NAMES: [&str; FEATURE_COUNT] = [
    "count.bin_onehot",
    "count.blom",
    "count.cheb",
    "count.elasticnet",
    "count.hinge",
    "count.histgbm",
    "count.ksmooth",
    "count.oblique_hinge",
    "count.pair_prod",
    "count.poly_interact",
    "count.quantile_bins",
    "count.rand_proj",
    "count.rank",
    "count.rff_cos",
    "count.rff_sin",
    "count.ridge",
    "count.ridge_gcv",
    "count.robust_scale",
    "count.row_stats",
    "count.sigmoid",
    "count.standardize",
    "count.target_bin_encode",
    "count.winsor_clip",
    "dsl_characters",
    "estimator.elasticnet",
    "estimator.histgbm",
    "estimator.ksmooth",
    "estimator.ridge",
    "estimator.ridge_gcv",
    "parameter.bin_onehot.append.count",
    "parameter.bin_onehot.append.sum",
    "parameter.bin_onehot.bins.count",
    "parameter.bin_onehot.bins.sum",
    "parameter.blom.append.count",
    "parameter.blom.append.sum",
    "parameter.cheb.append.count",
    "parameter.cheb.append.sum",
    "parameter.cheb.degree.count",
    "parameter.cheb.degree.sum",
    "parameter.elasticnet.alpha.count",
    "parameter.elasticnet.alpha.sum",
    "parameter.elasticnet.l1_ratio.count",
    "parameter.elasticnet.l1_ratio.sum",
    "parameter.hinge.append.count",
    "parameter.hinge.append.sum",
    "parameter.hinge.knots.count",
    "parameter.hinge.knots.sum",
    "parameter.histgbm.depth.count",
    "parameter.histgbm.depth.sum",
    "parameter.histgbm.lr.count",
    "parameter.histgbm.lr.sum",
    "parameter.histgbm.rounds.count",
    "parameter.histgbm.rounds.sum",
    "parameter.ksmooth.gamma.count",
    "parameter.ksmooth.gamma.sum",
    "parameter.ksmooth.k.count",
    "parameter.ksmooth.k.sum",
    "parameter.oblique_hinge.append.count",
    "parameter.oblique_hinge.append.sum",
    "parameter.oblique_hinge.gamma.count",
    "parameter.oblique_hinge.gamma.sum",
    "parameter.oblique_hinge.n.count",
    "parameter.oblique_hinge.n.sum",
    "parameter.pair_prod.append.count",
    "parameter.pair_prod.append.sum",
    "parameter.pair_prod.topk.count",
    "parameter.pair_prod.topk.sum",
    "parameter.poly_interact.append.count",
    "parameter.poly_interact.append.sum",
    "parameter.quantile_bins.append.count",
    "parameter.quantile_bins.append.sum",
    "parameter.quantile_bins.bins.count",
    "parameter.quantile_bins.bins.sum",
    "parameter.rand_proj.append.count",
    "parameter.rand_proj.append.sum",
    "parameter.rand_proj.k.count",
    "parameter.rand_proj.k.sum",
    "parameter.rank.append.count",
    "parameter.rank.append.sum",
    "parameter.rff_cos.append.count",
    "parameter.rff_cos.append.sum",
    "parameter.rff_cos.gamma.count",
    "parameter.rff_cos.gamma.sum",
    "parameter.rff_cos.n.count",
    "parameter.rff_cos.n.sum",
    "parameter.rff_sin.append.count",
    "parameter.rff_sin.append.sum",
    "parameter.rff_sin.gamma.count",
    "parameter.rff_sin.gamma.sum",
    "parameter.rff_sin.n.count",
    "parameter.rff_sin.n.sum",
    "parameter.ridge.lambda_frac.count",
    "parameter.ridge.lambda_frac.sum",
    "parameter.robust_scale.append.count",
    "parameter.robust_scale.append.sum",
    "parameter.row_stats.append.count",
    "parameter.row_stats.append.sum",
    "parameter.sigmoid.append.count",
    "parameter.sigmoid.append.sum",
    "parameter.standardize.append.count",
    "parameter.standardize.append.sum",
    "parameter.target_bin_encode.append.count",
    "parameter.target_bin_encode.append.sum",
    "parameter.target_bin_encode.bins.count",
    "parameter.target_bin_encode.bins.sum",
    "parameter.winsor_clip.append.count",
    "parameter.winsor_clip.append.sum",
    "parameter.winsor_clip.q.count",
    "parameter.winsor_clip.q.sum",
    "position0.append",
    "position0.bin_onehot",
    "position0.bins",
    "position0.blom",
    "position0.cheb",
    "position0.degree",
    "position0.gamma",
    "position0.hinge",
    "position0.k",
    "position0.knots",
    "position0.n",
    "position0.oblique_hinge",
    "position0.pair_prod",
    "position0.poly_interact",
    "position0.q",
    "position0.quantile_bins",
    "position0.rand_proj",
    "position0.rank",
    "position0.rff_cos",
    "position0.rff_sin",
    "position0.robust_scale",
    "position0.row_stats",
    "position0.sigmoid",
    "position0.standardize",
    "position0.target_bin_encode",
    "position0.topk",
    "position0.winsor_clip",
    "position1.alpha",
    "position1.append",
    "position1.bin_onehot",
    "position1.bins",
    "position1.blom",
    "position1.cheb",
    "position1.degree",
    "position1.depth",
    "position1.elasticnet",
    "position1.gamma",
    "position1.hinge",
    "position1.histgbm",
    "position1.k",
    "position1.knots",
    "position1.ksmooth",
    "position1.l1_ratio",
    "position1.lambda_frac",
    "position1.lr",
    "position1.n",
    "position1.oblique_hinge",
    "position1.pair_prod",
    "position1.poly_interact",
    "position1.q",
    "position1.quantile_bins",
    "position1.rand_proj",
    "position1.rank",
    "position1.rff_cos",
    "position1.rff_sin",
    "position1.ridge",
    "position1.ridge_gcv",
    "position1.robust_scale",
    "position1.rounds",
    "position1.row_stats",
    "position1.sigmoid",
    "position1.standardize",
    "position1.target_bin_encode",
    "position1.topk",
    "position1.winsor_clip",
    "position2.alpha",
    "position2.append",
    "position2.bin_onehot",
    "position2.bins",
    "position2.blom",
    "position2.cheb",
    "position2.degree",
    "position2.depth",
    "position2.elasticnet",
    "position2.gamma",
    "position2.hinge",
    "position2.histgbm",
    "position2.k",
    "position2.knots",
    "position2.ksmooth",
    "position2.l1_ratio",
    "position2.lambda_frac",
    "position2.lr",
    "position2.n",
    "position2.oblique_hinge",
    "position2.pair_prod",
    "position2.poly_interact",
    "position2.q",
    "position2.quantile_bins",
    "position2.rand_proj",
    "position2.rank",
    "position2.rff_cos",
    "position2.rff_sin",
    "position2.ridge",
    "position2.ridge_gcv",
    "position2.robust_scale",
    "position2.rounds",
    "position2.row_stats",
    "position2.sigmoid",
    "position2.standardize",
    "position2.target_bin_encode",
    "position2.topk",
    "position2.winsor_clip",
    "position3.alpha",
    "position3.append",
    "position3.bin_onehot",
    "position3.bins",
    "position3.blom",
    "position3.cheb",
    "position3.degree",
    "position3.depth",
    "position3.elasticnet",
    "position3.gamma",
    "position3.hinge",
    "position3.histgbm",
    "position3.k",
    "position3.knots",
    "position3.ksmooth",
    "position3.l1_ratio",
    "position3.lambda_frac",
    "position3.lr",
    "position3.n",
    "position3.oblique_hinge",
    "position3.pair_prod",
    "position3.poly_interact",
    "position3.q",
    "position3.quantile_bins",
    "position3.rand_proj",
    "position3.rank",
    "position3.rff_cos",
    "position3.rff_sin",
    "position3.ridge",
    "position3.ridge_gcv",
    "position3.robust_scale",
    "position3.rounds",
    "position3.row_stats",
    "position3.sigmoid",
    "position3.standardize",
    "position3.target_bin_encode",
    "position3.topk",
    "position3.winsor_clip",
    "position4.alpha",
    "position4.append",
    "position4.bin_onehot",
    "position4.bins",
    "position4.blom",
    "position4.cheb",
    "position4.degree",
    "position4.depth",
    "position4.elasticnet",
    "position4.gamma",
    "position4.hinge",
    "position4.histgbm",
    "position4.k",
    "position4.knots",
    "position4.ksmooth",
    "position4.l1_ratio",
    "position4.lambda_frac",
    "position4.lr",
    "position4.n",
    "position4.oblique_hinge",
    "position4.pair_prod",
    "position4.poly_interact",
    "position4.q",
    "position4.quantile_bins",
    "position4.rand_proj",
    "position4.rank",
    "position4.rff_cos",
    "position4.rff_sin",
    "position4.ridge",
    "position4.ridge_gcv",
    "position4.robust_scale",
    "position4.rounds",
    "position4.row_stats",
    "position4.sigmoid",
    "position4.standardize",
    "position4.target_bin_encode",
    "position4.topk",
    "position4.winsor_clip",
    "position5.alpha",
    "position5.append",
    "position5.bin_onehot",
    "position5.bins",
    "position5.blom",
    "position5.cheb",
    "position5.degree",
    "position5.depth",
    "position5.elasticnet",
    "position5.gamma",
    "position5.hinge",
    "position5.histgbm",
    "position5.k",
    "position5.knots",
    "position5.ksmooth",
    "position5.l1_ratio",
    "position5.lr",
    "position5.n",
    "position5.oblique_hinge",
    "position5.pair_prod",
    "position5.poly_interact",
    "position5.q",
    "position5.quantile_bins",
    "position5.rand_proj",
    "position5.rank",
    "position5.rff_cos",
    "position5.rff_sin",
    "position5.ridge",
    "position5.ridge_gcv",
    "position5.robust_scale",
    "position5.rounds",
    "position5.row_stats",
    "position5.sigmoid",
    "position5.standardize",
    "position5.target_bin_encode",
    "position5.topk",
    "position5.winsor_clip",
    "position6.alpha",
    "position6.append",
    "position6.bin_onehot",
    "position6.bins",
    "position6.blom",
    "position6.cheb",
    "position6.degree",
    "position6.depth",
    "position6.elasticnet",
    "position6.gamma",
    "position6.hinge",
    "position6.histgbm",
    "position6.k",
    "position6.knots",
    "position6.ksmooth",
    "position6.l1_ratio",
    "position6.lr",
    "position6.n",
    "position6.oblique_hinge",
    "position6.pair_prod",
    "position6.poly_interact",
    "position6.q",
    "position6.quantile_bins",
    "position6.rand_proj",
    "position6.rank",
    "position6.rff_cos",
    "position6.rff_sin",
    "position6.ridge",
    "position6.ridge_gcv",
    "position6.robust_scale",
    "position6.rounds",
    "position6.row_stats",
    "position6.sigmoid",
    "position6.standardize",
    "position6.target_bin_encode",
    "position6.topk",
    "position6.winsor_clip",
    "position7.alpha",
    "position7.append",
    "position7.bin_onehot",
    "position7.bins",
    "position7.blom",
    "position7.cheb",
    "position7.degree",
    "position7.depth",
    "position7.elasticnet",
    "position7.gamma",
    "position7.hinge",
    "position7.histgbm",
    "position7.k",
    "position7.knots",
    "position7.ksmooth",
    "position7.l1_ratio",
    "position7.lr",
    "position7.n",
    "position7.oblique_hinge",
    "position7.pair_prod",
    "position7.poly_interact",
    "position7.q",
    "position7.quantile_bins",
    "position7.rand_proj",
    "position7.rank",
    "position7.rff_cos",
    "position7.rff_sin",
    "position7.ridge",
    "position7.ridge_gcv",
    "position7.robust_scale",
    "position7.rounds",
    "position7.row_stats",
    "position7.sigmoid",
    "position7.standardize",
    "position7.target_bin_encode",
    "position7.topk",
    "position7.winsor_clip",
    "position8.alpha",
    "position8.depth",
    "position8.elasticnet",
    "position8.gamma",
    "position8.histgbm",
    "position8.k",
    "position8.ksmooth",
    "position8.l1_ratio",
    "position8.lr",
    "position8.ridge",
    "position8.ridge_gcv",
    "position8.rounds",
    "steps",
];
