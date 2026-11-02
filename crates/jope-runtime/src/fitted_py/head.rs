use super::render::Py;
use feat_math::heads::FittedHead;
use feat_math::spec::PlannedHead;

/// The state the recipe's head `*_predict` reads, after checking it against the planned head.
pub(super) fn head_state(
    planned: &PlannedHead,
    head: &FittedHead,
    problems: &mut Vec<String>,
) -> Option<Py> {
    use FittedHead as H;
    use PlannedHead as P;
    Some(match (planned, head) {
        (P::Ridge { .. }, H::Ridge(r)) => ridge(r),
        (P::RidgeGcv, H::RidgeGcv(r)) => Py::Tuple(vec![
            Py::ints(&r.cols),
            Py::floats(&r.means),
            Py::floats(&r.scales),
            Py::floats(&r.w),
        ]),
        (P::ElasticNet { .. }, H::ElasticNet(e)) => elastic_net(e),
        (P::Gbm(_), H::Gbm(g)) => gbm(g),
        (
            P::Blend { .. },
            H::Ensemble {
                ridge: r,
                gbm: g,
                w,
            },
        ) => Py::Tuple(vec![ridge(r), gbm(g), Py::F(*w)]),
        (P::HistGbm2 { .. }, H::HistGbm2(h)) => hist_gbm2(h),
        (P::Krr { .. }, H::Krr(k)) => krr(k),
        (P::Gpr { .. }, H::Gpr(g)) => Py::Tuple(vec![
            Py::floats(&g.means),
            Py::floats(&g.scales),
            Py::rows(&g.basis),
            Py::floats(&g.alpha),
            Py::floats(&g.ls),
            Py::F(g.sf2),
            Py::F(g.y_mean),
        ]),
        // The kernel terms are baked into the predict call; `theta` keeps its trailing noise entry,
        // which the recipe's decoder indexes.
        (P::GprKernel { spec, .. }, H::GprKernel(g)) => {
            if *spec != g.spec {
                problems.push("head kernel parameter differs from fitted state".into());
            }
            Py::Tuple(vec![
                Py::floats(&g.means),
                Py::floats(&g.scales),
                Py::rows(&g.basis),
                Py::floats(&g.alpha),
                Py::floats(&g.theta),
                Py::F(g.y_mean),
            ])
        }
        // k and gamma are baked into the predict call.
        (P::Ksmooth { k, gamma }, H::Ksmooth(m)) => {
            if *k != m.k {
                problems.push("head k parameter differs from fitted state".into());
            }
            if !(m.gamma.is_finite() && m.gamma.to_bits() == gamma.to_bits()) {
                problems.push("head gamma parameter differs from fitted state".into());
            }
            Py::Tuple(vec![
                Py::floats(&m.means),
                Py::floats(&m.scales),
                Py::rows(&m.z),
                Py::floats(&m.ys),
                Py::F(m.y_mean),
            ])
        }
        // The library is fixed: (weights, ridge_a, ridge_b, net, trees, kernel), None if unused.
        (P::Stack, H::Stack(m)) => Py::Tuple(vec![
            Py::floats(&m.weights),
            m.ridge_a.as_ref().map_or(Py::None, ridge),
            m.ridge_b.as_ref().map_or(Py::None, ridge),
            m.net.as_ref().map_or(Py::None, elastic_net),
            m.trees.as_ref().map_or(Py::None, hist_gbm2),
            m.kernel.as_ref().map_or(Py::None, krr),
        ]),
        (P::Portfolio(plan), H::Portfolio(m)) => portfolio(plan, m, problems)?,
        _ => {
            problems.push(format!(
                "fitted head {} does not match the planned head {}",
                fitted_head_kind(head),
                planned_head_kind(planned)
            ));
            return None;
        }
    })
}

fn ridge(r: &feat_math::heads::RidgeModel) -> Py {
    Py::Tuple(vec![
        Py::floats(&r.means),
        Py::floats(&r.scales),
        Py::floats(&r.w),
    ])
}

fn elastic_net(e: &feat_math::heads::ElasticNetModel) -> Py {
    Py::Tuple(vec![
        Py::floats(&e.means),
        Py::floats(&e.scales),
        Py::F(e.w0),
        Py::floats(&e.w),
    ])
}

fn krr(k: &feat_math::krr::KrrModel) -> Py {
    Py::Tuple(vec![
        Py::floats(&k.means),
        Py::floats(&k.scales),
        Py::rows(&k.basis),
        Py::floats(&k.alpha),
        Py::F(k.gamma),
        Py::F(k.y_mean),
    ])
}

/// `(init, lr, bags)`. An empty bag stays: the recipe divides by the number of bags.
fn hist_gbm2(h: &feat_math::boost2::HistGbm2Model) -> Py {
    Py::Tuple(vec![
        Py::F(h.init),
        Py::F(h.lr),
        Py::List(
            h.bags
                .iter()
                .map(|bag| Py::List(bag.iter().map(leafwise_tree).collect()))
                .collect(),
        ),
    ])
}

/// `(mean_only, y_mean, width, keep, proj, entries, clip_lo, clip_hi)`, one entry
/// `(component, projected, weight, state)` per fitted member. The component number is the
/// member's 1-based position in the planned roster, which names its `component<n>_predict`; a
/// member's state lists its stateful stages in order, then its head.
fn portfolio(
    plan: &feat_math::spec::PortfolioPlan,
    m: &feat_math::portfolio::PortfolioModel,
    problems: &mut Vec<String>,
) -> Option<Py> {
    use feat_math::portfolio::FittedMemberState as S;
    use feat_math::spec::PlannedMemberKind as K;
    if m.members.len() != m.weights.len() {
        problems.push("portfolio members and weights differ in length".into());
        return None;
    }
    let proj = match &m.proj {
        None => Py::None,
        Some(FittedOp::SvdProject {
            means, components, ..
        }) => Py::Tuple(vec![Py::floats(means), Py::rows(components)]),
        Some(_) => {
            problems.push("portfolio projection is not a principal component projection".into());
            return None;
        }
    };
    let mut entries = Vec::with_capacity(m.members.len());
    for (member, &weight) in m.members.iter().zip(&m.weights) {
        let Some(index) = plan.members.iter().position(|p| p.name == member.name) else {
            problems.push("a fitted portfolio member is not in the planned roster".into());
            continue;
        };
        let state = match (&plan.members[index].kind, &member.state) {
            (_, S::Mean | S::Skipped) => Py::None,
            (K::GroupMean, S::GroupMean(group)) => group_mean(group, problems)?,
            (K::Pipeline { stages, head }, S::Fitted(model)) => {
                let Model::Pipeline {
                    target: None,
                    ops,
                    head: fitted,
                } = model.as_ref()
                else {
                    problems.push("a portfolio member is not a plain pipeline".into());
                    continue;
                };
                if ops.len() != stages.len() {
                    problems.push(format!(
                        "portfolio member: {} fitted stages for {} planned stages",
                        ops.len(),
                        stages.len()
                    ));
                    continue;
                }
                let mut states: Vec<Py> = stages
                    .iter()
                    .zip(ops)
                    .enumerate()
                    .filter_map(|(i, (planned, op))| stage_state(i, planned, op, problems))
                    .collect();
                states.extend(head_state(head, fitted, problems));
                Py::Tuple(states)
            }
            _ => {
                problems.push("a fitted portfolio member does not match its planned kind".into());
                continue;
            }
        };
        entries.push(Py::Tuple(vec![
            Py::I(index + 1),
            Py::B(member.proj_view),
            Py::F(weight),
            state,
        ]));
    }
    Some(Py::Tuple(vec![
        Py::B(m.mean_only),
        Py::F(m.y_mean),
        Py::I(m.width),
        Py::ints(&m.keep),
        proj,
        Py::List(entries),
        Py::F(m.clip_lo),
        Py::F(m.clip_hi),
    ]))
}

/// `(table, g_mean)`, the table keyed by the recipe's `row_signature` tuples. The model's fields
/// are crate-private, so its serialized form is read: `table` is `[(key bits, mean), ...]`.
fn group_mean(
    group: &feat_math::portfolio::GroupMeanModel,
    problems: &mut Vec<String>,
) -> Option<Py> {
    #[derive(serde::Deserialize)]
    struct GroupMean {
        table: Vec<(Vec<u64>, f64)>,
        g_mean: f64,
    }
    let Ok(group) = serde_json::to_value(group).and_then(serde_json::from_value::<GroupMean>)
    else {
        problems.push("unreadable group-mean member state".into());
        return None;
    };
    Some(Py::Tuple(vec![
        Py::Dict(
            group
                .table
                .into_iter()
                .map(|(key, mean)| {
                    let key = key
                        .into_iter()
                        .map(|bits| Py::F(f64::from_bits(bits)))
                        .collect();
                    (Py::Tuple(key), Py::F(mean))
                })
                .collect(),
        ),
        Py::F(group.g_mean),
    ]))
}

/// `(edges, trees, init, lr, n_feat, fallback)`; a node is `(feature, bin, left, right, value)`
/// with feature -1 for a leaf, in the fitted pre-order.
fn gbm(g: &feat_math::heads::GbmModel) -> Py {
    use feat_math::heads::GbmNode;
    let trees = g
        .trees
        .iter()
        .map(|tree| {
            Py::List(
                tree.iter()
                    .map(|node| match node {
                        GbmNode::Leaf(v) => {
                            Py::Tuple(vec![Py::Leaf, Py::I(0), Py::I(0), Py::I(0), Py::F(*v)])
                        }
                        GbmNode::Split {
                            feature,
                            thr_bin,
                            left,
                            right,
                        } => Py::Tuple(vec![
                            Py::I(*feature),
                            Py::I(usize::from(*thr_bin)),
                            Py::I(*left),
                            Py::I(*right),
                            Py::F(0.0),
                        ]),
                    })
                    .collect(),
            )
        })
        .collect();
    Py::Tuple(vec![
        Py::rows(&g.edges),
        Py::List(trees),
        Py::F(g.init),
        Py::F(g.lr),
        Py::I(g.n_feat),
        Py::F(g.constant_value),
    ])
}

/// A leaf-wise tree: nodes `(feature, threshold, left, right, value)`, feature -1 for a leaf.
fn leafwise_tree(tree: &feat_math::boost2::BoostTree) -> Py {
    use feat_math::boost2::BoostNode;
    Py::List(
        tree.iter()
            .map(|node| match node {
                BoostNode::Leaf(v) => {
                    Py::Tuple(vec![Py::Leaf, Py::F(0.0), Py::I(0), Py::I(0), Py::F(*v)])
                }
                BoostNode::Split {
                    feature,
                    threshold,
                    left,
                    right,
                } => Py::Tuple(vec![
                    Py::I(*feature),
                    Py::F(*threshold),
                    Py::I(*left),
                    Py::I(*right),
                    Py::F(0.0),
                ]),
            })
            .collect(),
    )
}

fn fitted_head_kind(head: &FittedHead) -> &'static str {
    use FittedHead as H;
    match head {
        H::Ridge(_) => "ridge",
        H::RidgeGcv(_) => "ridge_gcv",
        H::ElasticNet(_) => "elasticnet",
        H::Gbm(_) => "gbm",
        H::Ensemble { .. } => "ensemble",
        H::HistGbm2(_) => "histgbm2",
        H::Krr(_) => "krr",
        H::Gpr(_) => "gpr",
        H::GprKernel(_) => "gpr with a kernel",
        H::Ksmooth(_) => "ksmooth",
        H::Hurdle2(_) => "hurdle2",
        H::Stack(_) => "stack",
        H::Portfolio(_) => "portfolio",
    }
}

fn planned_head_kind(head: &PlannedHead) -> &'static str {
    use PlannedHead as P;
    match head {
        P::Ridge { .. } => "ridge",
        P::RidgeGcv => "ridge_gcv",
        P::ElasticNet { .. } => "elasticnet",
        P::Gbm(_) => "gbm",
        P::Blend { .. } => "ensemble",
        P::HistGbm2 { .. } => "histgbm2",
        P::Krr { .. } => "krr",
        P::Gpr { .. } => "gpr",
        P::GprKernel { .. } => "gpr with a kernel",
        P::Ksmooth { .. } => "ksmooth",
        P::Stack => "stack",
        P::Portfolio(_) => "portfolio",
    }
}

/// The target transform's `self.target` state, after checking its kind against the plan.
pub(super) fn target_state(
    planned: &feat_math::spec::PlannedTarget,
    fitted: &feat_math::target_transform::FittedTarget,
    problems: &mut Vec<String>,
) -> Option<Py> {
    use feat_math::spec::TargetKind as K;
    use feat_math::target_transform::FittedTarget as T;
    match (planned.kind, fitted) {
        (
            K::RankGauss,
            T::RankGauss {
                sorted_values,
                table_values,
                table_scores,
            },
        ) => Some(Py::Tuple(vec![
            Py::floats(sorted_values),
            Py::floats(table_values),
            Py::floats(table_scores),
        ])),
        // The recipe's log1p inverse reads no state.
        (K::Log1p, T::Log1p) => None,
        (K::Winsor, T::Winsor { lo, hi }) => Some(Py::Tuple(vec![Py::F(*lo), Py::F(*hi)])),
        (
            K::ZScore,
            T::ZScore {
                mu,
                sigma,
                clamp_lo,
                clamp_hi,
            },
        ) => Some(Py::Tuple(vec![
            Py::F(*mu),
            Py::F(*sigma),
            Py::F(*clamp_lo),
            Py::F(*clamp_hi),
        ])),
        _ => {
            problems.push("fitted target transform kind differs from the planned one".into());
            None
        }
    }
}
