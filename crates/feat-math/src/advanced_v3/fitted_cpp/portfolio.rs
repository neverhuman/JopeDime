fn emit_legacy_model(name: &str, model: &crate::model::Model, out: &mut String) -> Result<()> {
    match model {
        crate::model::Model::Pipeline { target, ops, head } => {
            let head_name = format!("{name}_head");
            let transform_name = format!("{name}_transform");
            emit_legacy_head(&head_name, head, out)?;
            out.push_str(&format!(
                "static V {transform_name}(V x){{V mask(x.size(),0.0);\n"
            ));
            for op in ops {
                emit_legacy_op_padded(out, op)?;
            }
            out.push_str("return x;}\n");
            out.push_str(&format!(
                "static double {name}(const V& source){{V x={transform_name}(source);double value={head_name}(x);{}return finitev(value);}}\n",
                emit_target_inverse_value(target)
            ));
            Ok(())
        }
        crate::model::Model::Symbolic { expr, a, b } => {
            let expression = expr.to_cpp17(false)?;
            out.push_str(&format!(
                "static double {name}(const V& values){{const double* row=values.data();const std::size_t row_len=values.size();(void)row;(void)row_len;double value=math_safe({expression});double result={a}*value+{b};return std::isfinite(result)?clampv(result,-60.0,60.0):0.0;}}\n",
                a = f64_lit(*a),
                b = f64_lit(*b),
            ));
            Ok(())
        }
        crate::model::Model::Portfolio(model) => emit_portfolio(name, model, out),
    }
}

fn emit_target_inverse_value(target: &Option<FittedTarget>) -> String {
    match target {
        None => String::new(),
        Some(FittedTarget::RankGauss {
            table_values,
            table_scores,
            ..
        }) => format!(
            "value=clampv(rank_inverse(value,{},{}),-60.0,60.0);",
            f64_vec(table_scores),
            f64_vec(table_values)
        ),
        Some(FittedTarget::Log1p) => {
            "value=clampv(std::exp(clampv(value,-745.0,700.0))-1.0,-60.0,60.0);".to_string()
        }
        Some(FittedTarget::Winsor { .. }) => "value=clampv(value,-60.0,60.0);".to_string(),
        Some(FittedTarget::ZScore {
            mu,
            sigma,
            clamp_lo,
            clamp_hi,
        }) => format!(
            "value=clampv(value*{}+{},{},{});",
            f64_lit(*sigma),
            f64_lit(*mu),
            f64_lit(*clamp_lo),
            f64_lit(*clamp_hi)
        ),
    }
}

fn u64_vec(values: &[u64]) -> String {
    format!(
        "B{{{}}}",
        values
            .iter()
            .map(|value| format!("0x{value:016x}ULL"))
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn emit_group_mean(name: &str, model: &crate::portfolio::GroupMeanModel, out: &mut String) {
    let entries = model
        .table
        .iter()
        .map(|(key, value)| {
            format!(
                "std::pair<B,double>{{{},{}}}",
                u64_vec(key),
                f64_lit(*value)
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    out.push_str(&format!(
        "static double {name}(const V& x){{B key;key.reserve(x.size());for(double value:x){{double rounded=std::floor(value*1e9+0.5)/1e9+0.0;key.push_back(bits(rounded));}}const std::vector<std::pair<B,double>> table{{{entries}}};auto found=std::lower_bound(table.begin(),table.end(),key,[](const auto& item,const B& wanted){{return item.first<wanted;}});return found!=table.end()&&found->first==key?found->second:{};}}\n",
        f64_lit(model.g_mean)
    ));
}

fn emit_portfolio(
    name: &str,
    model: &crate::portfolio::PortfolioModel,
    out: &mut String,
) -> Result<()> {
    if model.mean_only {
        out.push_str(&format!(
            "static double {name}(const V&){{return {};}}\n",
            f64_lit(model.y_mean)
        ));
        return Ok(());
    }
    let projection_name = format!("{name}_project");
    if let Some(projection) = &model.proj {
        out.push_str(&format!(
            "static V {projection_name}(V x){{V mask(x.size(),0.0);\n"
        ));
        emit_legacy_op_padded(out, projection)?;
        out.push_str("return x;}\n");
    }
    let mut member_terms = Vec::new();
    for (index, member) in model.members.iter().enumerate() {
        let weight = model.weights.get(index).copied().unwrap_or(0.0);
        if weight <= 0.0 {
            continue;
        }
        let child = format!("{name}_member_{index}");
        match &member.state {
            FittedMemberState::Fitted(model) => emit_legacy_model(&child, model, out)?,
            FittedMemberState::GroupMean(model) => emit_group_mean(&child, model, out),
            FittedMemberState::Mean | FittedMemberState::Skipped => continue,
        }
        let view = if member.proj_view && model.proj.is_some() {
            "projected"
        } else {
            "raw"
        };
        member_terms.push(format!(
            "{{double member={child}({view});if(!std::isfinite(member))member=y_mean;member=clampv(member,clip_lo,clip_hi);value+={weight}*(member-y_mean);}}",
            weight = f64_lit(weight)
        ));
    }
    let projection = if model.proj.is_some() {
        format!("V projected={projection_name}(raw);")
    } else {
        "V projected;".to_string()
    };
    out.push_str(&format!(
        "static double {name}(const V& source){{V resized=source;resized.resize({},0.0);for(double& cell:resized)if(!std::isfinite(cell))cell=0.0;const U keep={};V raw;raw.reserve(keep.size());for(std::size_t column:keep)raw.push_back(resized[column]);{projection}const double y_mean={},clip_lo={},clip_hi={};double value=y_mean;{}if(!std::isfinite(value))value=y_mean;return clampv(value,clip_lo,clip_hi);}}\n",
        model.width,
        usize_vec(&model.keep),
        f64_lit(model.y_mean),
        f64_lit(model.clip_lo),
        f64_lit(model.clip_hi),
        member_terms.join(""),
    ));
    Ok(())
}

fn extra_expr(node: &ExtraNode) -> String {
    match node {
        ExtraNode::Leaf(value) => f64_lit(*value),
        ExtraNode::Split {
            feature,
            threshold,
            missing_left,
            left,
            right,
        } => format!(
            "((std::isfinite(x[{feature}])?x[{feature}]<={}:{} )?({}):({}))",
            f64_lit(*threshold),
            missing_left,
            extra_expr(left),
            extra_expr(right)
        ),
    }
}

fn emit_gbm(name: &str, model: &crate::heads::GbmModel, out: &mut String) -> Result<()> {
    let mut trees = Vec::new();
    for tree in &model.trees {
        let nodes = tree
            .iter()
            .map(|node| match node {
                GbmNode::Leaf(value) => format!("GNode{{true,{},0,0,0,0}}", f64_lit(*value)),
                GbmNode::Split {
                    feature,
                    thr_bin,
                    left,
                    right,
                } => format!("GNode{{false,0.0,{feature},{thr_bin},{left},{right}}}"),
            })
            .collect::<Vec<_>>()
            .join(",");
        trees.push(format!("std::vector<GNode>{{{nodes}}}"));
    }
    out.push_str(&format!("static double {name}(const V& x){{if({}==0)return {};const M edges={};const std::vector<std::vector<GNode>> trees{{{}}};double value={};for(const auto& tree:trees)value+={}*route_gbm(tree,edges,x);return value;}}\n",model.n_feat,f64_lit(model.constant_value),f64_matrix(&model.edges),trees.join(","),f64_lit(model.init),f64_lit(model.lr)));
    Ok(())
}

fn emit_hist_gbm2(
    name: &str,
    model: &crate::boost2::HistGbm2Model,
    out: &mut String,
) -> Result<()> {
    if model.bags.is_empty() {
        out.push_str(&format!(
            "static double {name}(const V&){{return {};}}\n",
            f64_lit(model.init)
        ));
        return Ok(());
    }
    let mut bag_terms = Vec::new();
    for bag in &model.bags {
        let mut tree_terms = Vec::new();
        for tree in bag {
            tree_terms.push(boost_expr(tree, 0, 0)?);
        }
        bag_terms.push(if tree_terms.is_empty() {
            "0.0".to_string()
        } else {
            format!("({})", tree_terms.join("+"))
        });
    }
    out.push_str(&format!(
        "static double {name}(const V& x){{double acc={};return {}+{}*acc/{}.0;}}\n",
        bag_terms.join("+"),
        f64_lit(model.init),
        f64_lit(model.lr),
        model.bags.len(),
    ));
    Ok(())
}

fn boost_expr(tree: &[BoostNode], index: usize, depth: usize) -> Result<String> {
    if depth > tree.len() {
        bail!("histgbm2 fitted tree contains a cycle");
    }
    match tree.get(index) {
        Some(BoostNode::Leaf(value)) => Ok(f64_lit(*value)),
        Some(BoostNode::Split {
            feature,
            threshold,
            left,
            right,
        }) => Ok(format!(
            "(x[{feature}]<={}?({}):({}))",
            f64_lit(*threshold),
            boost_expr(tree, *left, depth + 1)?,
            boost_expr(tree, *right, depth + 1)?,
        )),
        None => bail!("histgbm2 fitted tree references an invalid node"),
    }
}
