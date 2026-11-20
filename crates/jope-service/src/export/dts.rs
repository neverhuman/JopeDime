//! TypeScript declarations for a fitted JavaScript predictor.

use anyhow::Result;
use jope_runtime::fitted::Header;
use jope_runtime::table::ColumnKind;

/// Generate declarations for the input columns recorded by a fitted model.
pub fn generate(header: &Header) -> Result<String> {
    let mut output = String::from("export interface PredictorRow {\n");
    for column in &header.input_schema {
        let name = serde_json::to_string(&column.name)?;
        let value_type = match column.kind {
            ColumnKind::Numeric => "number",
            ColumnKind::Categorical => "string",
        };
        output.push_str(&format!("  readonly {name}: {value_type} | null;\n"));
    }
    if header.task == "binary" && header.class_order.len() == 2 {
        let c0 = serde_json::to_string(&header.class_order[0])?;
        let c1 = serde_json::to_string(&header.class_order[1])?;
        output.push_str(&format!(
            "}}\n\
             \n\
             export type BinaryProbabilityPair = [number, number];\n\
             export declare const CLASS_ORDER: readonly [{c0}, {c1}];\n\
             export declare function predict(row: PredictorRow): BinaryProbabilityPair;\n\
             export declare function predictBatch(rows: readonly PredictorRow[]): BinaryProbabilityPair[];\n"
        ));
    } else if header.task == "multiclass" && header.class_order.len() >= 3 {
        let tuple_elems = vec!["number"; header.class_order.len()].join(", ");
        let class_literals = header
            .class_order
            .iter()
            .map(serde_json::to_string)
            .collect::<std::result::Result<Vec<_>, _>>()?
            .join(", ");
        output.push_str(&format!(
            "}}\n\
             \n\
             export type MulticlassProbabilityTuple = [{tuple_elems}];\n\
             export declare const CLASS_ORDER: readonly [{class_literals}];\n\
             export declare function predict(row: PredictorRow): MulticlassProbabilityTuple;\n\
             export declare function predictBatch(rows: readonly PredictorRow[]): MulticlassProbabilityTuple[];\n"
        ));
    } else {
        output.push_str(
            "}\n\
             \n\
             export declare function predict(row: PredictorRow): number;\n\
             export declare function predictBatch(rows: readonly PredictorRow[]): number[];\n",
        );
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fixture_header() -> Header {
        serde_json::from_value(json!({
            "schema": "jope.appliance-fitted/v1",
            "minimum_reader_version": 1,
            "product": "jopedime",
            "product_version": "1.0.0",
            "runtime_abi": "jope.appliance-runtime/v1",
            "method": {
                "id": "ridge",
                "family": "linear",
                "configuration": "default",
                "recipe": "ridge",
                "implementation": "fixture"
            },
            "task": "regression",
            "metric": "rmse",
            "class_order": [],
            "target": "target",
            "input_schema": [
                {"name": "age", "kind": "numeric"},
                {"name": "region", "kind": "categorical"}
            ],
            "preprocessing": {
                "raw_cell_encoding": "jope.raw-cell-encoding/v1",
                "input_policy": "named-columns"
            },
            "recipe_sha256": "00",
            "required_asset_hashes": [],
            "fitted_sections": [],
            "training_procedure": {
                "rows": 1,
                "fitting_input_sha256": "00",
                "training_file_sha256": "00",
                "method_choice": "requested",
                "selection": null,
                "budget": {},
                "final_refit": true
            },
            "retained_training_context": {
                "neural_support_rows": 0,
                "note": "fixture"
            },
            "created_by": {
                "source_sha40": "0000000000",
                "package_manifest_sha256": null,
                "package_profile": null
            }
        }))
        .expect("valid fitted header fixture")
    }

    #[test]
    fn output_is_deterministic() {
        let header = fixture_header();
        assert_eq!(generate(&header).unwrap(), generate(&header).unwrap());
    }

    #[test]
    fn fixture_has_golden_declaration() {
        assert_eq!(
            generate(&fixture_header()).unwrap(),
            concat!(
                "export interface PredictorRow {\n",
                "  readonly \"age\": number | null;\n",
                "  readonly \"region\": string | null;\n",
                "}\n\n",
                "export declare function predict(row: PredictorRow): number;\n",
                "export declare function predictBatch(rows: readonly PredictorRow[]): number[];\n"
            )
        );
    }

    #[test]
    fn maps_every_input_column_kind() {
        let declaration = generate(&fixture_header()).unwrap();
        assert!(declaration.contains("readonly \"age\": number | null;"));
        assert!(declaration.contains("readonly \"region\": string | null;"));
    }

    #[test]
    fn binary_fixture_declares_probability_pairs_and_class_order() {
        let mut header = fixture_header();
        header.task = "binary".into();
        header.metric = "auroc".into();
        header.class_order = vec!["no".into(), "yes".into()];
        let declaration = generate(&header).unwrap();
        assert!(declaration.contains("export type BinaryProbabilityPair = [number, number];"));
        assert!(
            declaration.contains("export declare const CLASS_ORDER: readonly [\"no\", \"yes\"];")
        );
        assert!(declaration.contains(
            "export declare function predict(row: PredictorRow): BinaryProbabilityPair;"
        ));
        assert!(declaration.contains("export declare function predictBatch(rows: readonly PredictorRow[]): BinaryProbabilityPair[];"));
    }

    #[test]
    fn multiclass_fixture_declares_probability_tuple_and_class_order() {
        let mut header = fixture_header();
        header.task = "multiclass".into();
        header.metric = "logloss".into();
        header.class_order = vec!["alpha".into(), "beta".into(), "gamma".into()];
        let declaration = generate(&header).unwrap();
        assert!(declaration
            .contains("export type MulticlassProbabilityTuple = [number, number, number];"));
        assert!(declaration.contains(
            "export declare const CLASS_ORDER: readonly [\"alpha\", \"beta\", \"gamma\"];"
        ));
        assert!(declaration.contains(
            "export declare function predict(row: PredictorRow): MulticlassProbabilityTuple;"
        ));
        assert!(declaration.contains(
            "export declare function predictBatch(rows: readonly PredictorRow[]): MulticlassProbabilityTuple[];"
        ));
    }
}
