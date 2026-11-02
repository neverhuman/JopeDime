use serde_json::json;

use super::{param, required_expr, transform, OpSpec};

macro_rules! bins {
    ($($value:expr),* $(,)?) => { vec![$(json!($value)),*] };
}

pub(super) fn definitions() -> Vec<OpSpec> {
    let mut transforms = vec![
        transform("standardize", &["scaling"], vec![]),
        transform("robust_scale", &["scaling"], vec![]),
        transform("norm_features", &["scaling"], vec![]),
        transform("rank", &["rank"], vec![]),
        transform("blom", &["rank"], vec![]),
        transform(
            "quantile_bins",
            &["rank"],
            vec![param("bins", "int", bins![4, 8, 16, 32])],
        ),
        transform(
            "bin_onehot",
            &["rank"],
            vec![param("bins", "int", bins![4, 8, 12, 16])],
        ),
        transform(
            "hinge",
            &["basis"],
            vec![param("knots", "int", bins![2, 4, 6, 8])],
        ),
        transform(
            "winsor_clip",
            &["scaling", "robust"],
            vec![param("q", "float", bins![0.01, 0.02, 0.05, 0.1])],
        ),
        transform(
            "winsor_moments",
            &["scaling", "robust"],
            vec![param("q", "float", bins![0.02, 0.05, 0.1])],
        ),
        transform(
            "cheb",
            &["basis"],
            vec![param("degree", "int", bins![2, 3, 4, 6])],
        ),
        transform(
            "poly_interact",
            &["interaction"],
            vec![param("topk", "int", bins![4, 8, 12])],
        ),
        transform(
            "rff_cos",
            &["random_features", "kernel"],
            vec![
                param("n", "int", bins![8, 16, 32, 64]),
                param("gamma", "float", bins![0.5, 1.0, 2.0]),
                param("seed", "int", bins![2, 7, 11, 23]),
            ],
        ),
        transform(
            "rff_sin",
            &["random_features", "kernel"],
            vec![
                param("n", "int", bins![8, 16, 32, 64]),
                param("gamma", "float", bins![0.5, 1.0, 2.0]),
                param("seed", "int", bins![3, 5, 13, 29]),
            ],
        ),
        transform(
            "rand_proj",
            &["random_features", "kernel"],
            vec![
                param("k", "int", bins![8, 16, 32, 64]),
                param("seed", "int", bins![7, 17, 31]),
            ],
        ),
        transform(
            "oblique_hinge",
            &["random_features", "kernel"],
            vec![
                param("n", "int", bins![8, 16, 32]),
                param("seed", "int", bins![7, 19]),
            ],
        ),
        transform("row_stats", &["rowwise"], vec![]),
        transform("norm_row", &["rowwise"], vec![]),
        transform("row_diff_stats", &["rowwise"], vec![]),
        transform(
            "pair_prod",
            &["interaction"],
            vec![param("topk", "int", bins![4, 8, 12])],
        ),
        transform(
            "target_bin_encode",
            &["target_aware"],
            vec![
                param("bins", "int", bins![4, 8, 16]),
                param("prior", "int", bins![1, 5, 10]),
            ],
        ),
        transform("sigmoid", &["basis"], vec![]),
        transform(
            "step",
            &["basis"],
            vec![param("t", "float", bins![0.25, 0.5, 0.75])],
        ),
        transform("polar", &["geometry", "rowwise"], vec![]),
        transform(
            "fft_mag",
            &["spectral"],
            vec![param("k", "int", bins![4, 8, 16])],
        ),
        transform("row_spectral", &["spectral"], vec![]),
    ];
    let mut svd = transform(
        "svd_project",
        &["projection", "spectral"],
        vec![
            param("k", "int", bins![4, 8, 16, 32]),
            param("iters", "int", bins![8, 16, 24]),
            param("seed", "int", bins![7]),
        ],
    );
    svd.cost_class = "medium".into();
    transforms.push(svd);
    let mut nystrom = transform(
        "nystrom",
        &["random_features", "kernel", "projection"],
        vec![
            param("k", "int", bins![8, 16, 32]),
            param("gamma", "float", bins![0.25, 0.5, 1.0]),
            param("seed", "int", bins![7]),
        ],
    );
    nystrom.cost_class = "medium".into();
    transforms.push(nystrom);
    let mut landmark = transform(
        "landmark_dist",
        &["random_features", "kernel", "projection"],
        vec![param("k", "int", bins![8, 16, 32])],
    );
    landmark.cost_class = "medium".into();
    transforms.push(landmark);
    let mut map = transform(
        "map",
        &["symbolic", "basis"],
        vec![required_expr("expr", "tanh(x)")],
    );
    map.takes_expr = true;
    transforms.push(map);
    let mut derive = transform(
        "derive",
        &["symbolic", "basis"],
        vec![required_expr("expr", "row_mean")],
    );
    derive.takes_expr = true;
    transforms.push(derive);

    transforms
}
