use std::sync::LazyLock;

use regex::Regex;

use super::{EXPECTED_PARAMETERS, EXPECTED_TENSORS};
use crate::pyjson::{Dict, Value};
use crate::{Error, Result};

/// The tensor roots of each role, in role order.
pub type Roles = Vec<(String, Vec<String>)>;

/// The accepted checkpoint's roles.
pub fn accepted_roles() -> Roles {
    let role = |name: &str, roots: &[&str]| {
        (
            name.to_owned(),
            roots.iter().map(|r| (*r).to_owned()).collect(),
        )
    };
    vec![
        role(
            "decode",
            &[
                "encoder",
                "ctx_proj",
                "token_emb",
                "pos_emb",
                "decoder",
                "out_norm",
            ],
        ),
        role(
            "rank",
            &[
                "reward_encoder",
                "reward_token_emb",
                "reward_norm",
                "z_gamma",
                "z_proj",
                "reward_heads",
                "runtime_head",
            ],
        ),
        role(
            "training_only",
            &["stability_head", "reward_head_ablation", "loss_log_vars"],
        ),
    ]
}

/// Where the expected header geometry comes from.
#[derive(Clone, Debug)]
pub enum Geometry {
    /// [`expected_geometry`] of the model source, derived when the header is compared.
    Documented,
    /// A fixed name → shape map (synthetic fixtures).
    Fixed(Dict),
}

/// What an inventory is checked against.
#[derive(Clone, Debug)]
pub struct Spec {
    pub roles: Roles,
    pub geometry: Geometry,
    pub parameters: i64,
    pub tensors: usize,
}

impl Spec {
    /// The accepted step-60000 checkpoint.
    pub fn accepted() -> Self {
        Spec {
            roles: accepted_roles(),
            geometry: Geometry::Documented,
            parameters: EXPECTED_PARAMETERS,
            tensors: EXPECTED_TENSORS,
        }
    }
}

fn shape(dims: &[i64]) -> Value {
    Value::List(dims.iter().map(|d| Value::Int(*d)).collect())
}

/// Documented mirror of `model.rs` `expected_inventory` for the step-60000 configuration.
pub fn expected_geometry(source: &str) -> Result<Dict> {
    const WIDTH: i64 = 512;
    const EMBEDDING_ROWS: i64 = 657;
    const POSITIONS: i64 = 129;
    const LAYERS: usize = 6;
    const FFN: i64 = 2048;
    const CONTEXT: i64 = 2048;
    const HEADS: usize = 4;
    let mut expected = Dict::new();
    let dense = |expected: &mut Dict, name: &str, inputs: i64, outputs: i64| {
        expected.insert(format!("{name}.weight"), shape(&[outputs, inputs]));
        expected.insert(format!("{name}.bias"), shape(&[outputs]));
    };
    let norm = |expected: &mut Dict, name: &str| {
        expected.insert(format!("{name}.weight"), shape(&[WIDTH]));
        expected.insert(format!("{name}.bias"), shape(&[WIDTH]));
    };
    for name in ["encoder", "reward_encoder"] {
        dense(&mut expected, &format!("{name}.0"), CONTEXT, WIDTH);
        norm(&mut expected, &format!("{name}.2"));
        dense(&mut expected, &format!("{name}.4"), WIDTH, WIDTH);
        norm(&mut expected, &format!("{name}.6"));
    }
    dense(&mut expected, "ctx_proj", WIDTH, WIDTH);
    expected.insert("token_emb.weight", shape(&[EMBEDDING_ROWS, WIDTH]));
    expected.insert("pos_emb.weight", shape(&[POSITIONS, WIDTH]));
    for i in 0..LAYERS {
        let p = format!("decoder.layers.{i}");
        expected.insert(
            format!("{p}.self_attn.in_proj_weight"),
            shape(&[3 * WIDTH, WIDTH]),
        );
        expected.insert(format!("{p}.self_attn.in_proj_bias"), shape(&[3 * WIDTH]));
        dense(
            &mut expected,
            &format!("{p}.self_attn.out_proj"),
            WIDTH,
            WIDTH,
        );
        dense(&mut expected, &format!("{p}.linear1"), WIDTH, FFN);
        dense(&mut expected, &format!("{p}.linear2"), FFN, WIDTH);
        norm(&mut expected, &format!("{p}.norm1"));
        norm(&mut expected, &format!("{p}.norm2"));
    }
    norm(&mut expected, "out_norm");
    expected.insert("reward_token_emb.weight", shape(&[EMBEDDING_ROWS, WIDTH]));
    norm(&mut expected, "reward_norm");
    dense(&mut expected, "z_gamma.0", WIDTH, WIDTH);
    dense(&mut expected, "z_proj", WIDTH, WIDTH);
    for i in 0..HEADS {
        dense(
            &mut expected,
            &format!("reward_heads.{i}.0"),
            3 * WIDTH,
            WIDTH,
        );
        dense(&mut expected, &format!("reward_heads.{i}.3"), WIDTH, 1);
    }
    for (name, inputs, index) in [
        ("runtime_head", 2 * WIDTH, 2),
        ("stability_head", 3 * WIDTH, 2),
        ("reward_head_ablation", WIDTH, 3),
    ] {
        dense(&mut expected, &format!("{name}.0"), inputs, WIDTH);
        dense(&mut expected, &format!("{name}.{index}"), WIDTH, 1);
    }
    static QUOTED: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#""([a-z_]+)""#).expect("regex"));
    let block = source
        .split_once("const LOSS_LOG_VARS:")
        .map(|(_, rest)| rest.split_once("];").map_or(rest, |(block, _)| block))
        .ok_or_else(|| Error::Shape("model source has no `const LOSS_LOG_VARS:`".into()))?;
    for captures in QUOTED.captures_iter(block) {
        expected.insert(
            format!("loss_log_vars.{}", &captures[1]),
            Value::List(Vec::new()),
        );
    }
    Ok(expected)
}
