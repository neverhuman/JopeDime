use super::*;
use crate::ranker::featurizer::parse;
use serde::Deserialize;

#[derive(Deserialize)]
struct Boundary {
    dsl: String,
    operations: Option<Vec<String>>,
    structural: Option<Vec<f64>>,
    error: Option<String>,
}

#[test]
fn reference_boundaries_preserve_unicode_quotes_wrappers_and_empty_group_behavior() {
    let boundaries: Vec<Boundary> = serde_json::from_str(BOUNDARY_GOLDENS).unwrap();
    assert_eq!(boundaries.len(), 13);
    let names: Vec<_> = FEATURE_NAMES
        .iter()
        .copied()
        .filter(|n| is_feature(n))
        .collect();
    for case in boundaries {
        let parsed = parse::parse(&case.dsl);
        if case.error.is_some() {
            assert!(parsed.is_err(), "{:?}", case.dsl);
            continue;
        }
        let parsed = parsed.unwrap();
        assert_eq!(
            parsed.operations(),
            case.operations.unwrap(),
            "{:?}",
            case.dsl
        );
        let actual = extract(&parsed);
        for (name, expected) in names.iter().zip(case.structural.unwrap()) {
            assert_eq!(actual[*name], expected, "{:?}: {name}", case.dsl);
        }
    }
}

const BOUNDARY_GOLDENS: &str = r#"[
  {
    "dsl": "",
    "operations": [],
    "structural": [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0.0,
      0,
      0,
      0,
      0,
      0,
      0.0
    ]
  },
  {
    "dsl": "ridge",
    "operations": [],
    "structural": [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      5.0,
      0,
      0,
      0,
      0,
      0,
      0.0
    ]
  },
  {
    "dsl": "(pipe X)",
    "operations": [
      "pipe"
    ],
    "structural": [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      8.0,
      0,
      0,
      0,
      0,
      0,
      1.0
    ]
  },
  {
    "dsl": "()",
    "operations": [],
    "structural": [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      2.0,
      0,
      0,
      0,
      0,
      0,
      0.0
    ]
  },
  {
    "dsl": "((ridge))",
    "operations": [
      "ridge"
    ],
    "structural": [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      9.0,
      0,
      0,
      0,
      1.0,
      0,
      1.0
    ]
  },
  {
    "dsl": "( )",
    "error": "empty operation"
  },
  {
    "dsl": "prefix (unknown v=2) suffix (ridge)",
    "operations": [
      "unknown",
      "ridge"
    ],
    "structural": [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      35.0,
      0,
      0,
      0,
      1.0,
      0,
      2.0
    ]
  },
  {
    "dsl": "(pipe X (ridge label=\"(inner)\"))",
    "operations": [
      "inner"
    ],
    "structural": [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      32.0,
      0,
      0,
      0,
      0,
      0,
      1.0
    ]
  },
  {
    "dsl": "(pipe X (ridge note=☃))",
    "operations": [
      "ridge"
    ],
    "structural": [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      23.0,
      0,
      0,
      0,
      1.0,
      0,
      1.0
    ]
  },
  {
    "dsl": "(pipe X (ridge\u001cseed=1))",
    "operations": [
      "ridge"
    ],
    "structural": [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      23.0,
      0,
      0,
      0,
      1.0,
      0,
      1.0
    ]
  },
  {
    "dsl": "(pipe X (ridge lambda_frac=1 lambda_frac=2))",
    "operations": [
      "ridge"
    ],
    "structural": [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      44.0,
      0,
      0,
      0,
      1.0,
      0,
      1.0
    ]
  },
  {
    "dsl": "(portfolio (member (pipe X (ridge))) (blend kind=nnls))",
    "operations": [
      "ridge",
      "blend"
    ],
    "structural": [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      55.0,
      0,
      0,
      0,
      0,
      0,
      2.0
    ]
  },
  {
    "dsl": "(pipe X (ridge)",
    "operations": [
      "ridge"
    ],
    "structural": [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      15.0,
      0,
      0,
      0,
      1.0,
      0,
      1.0
    ]
  }
]
"#;
