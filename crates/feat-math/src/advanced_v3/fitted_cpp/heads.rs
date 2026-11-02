fn emit_head_function(name: &str, head: &V3Head, out: &mut String) -> Result<()> {
    match head {
        V3Head::Legacy(head) => emit_legacy_head(name, head, out),
        V3Head::LogitRidge(model) => {
            emit_linear(
                name,
                &model.means,
                &model.scales,
                &model.weights,
                None,
                -30.0,
                30.0,
                true,
                out,
            );
            Ok(())
        }
        V3Head::TabBoost(model) => {
            let mut body = format!(
                "static double {name}(const V& x){{double value={};",
                f64_lit(model.base)
            );
            for stump in &model.stumps {
                body.push_str(&format!("{{const double cell=x[{feature}];const bool left=std::isfinite(cell)?cell<={threshold}:{missing};value+={lr}*(left?{left}:{right});}}", feature=stump.feature,threshold=f64_lit(stump.threshold),missing=stump.missing_left,lr=f64_lit(model.learning_rate),left=f64_lit(stump.left),right=f64_lit(stump.right)));
            }
            body.push_str(if model.binary {
                "return sigmoid(value);}\n"
            } else {
                "return finitev(value);}\n"
            });
            out.push_str(&body);
            Ok(())
        }
        V3Head::ExtraTrees(model) => {
            let terms = model.trees.iter().map(extra_expr).collect::<Vec<_>>();
            let sum = if terms.is_empty() {
                "0.0".to_string()
            } else {
                terms.join("+")
            };
            let finish = if model.binary {
                format!("clampv(({sum})/{}.0,0.0,1.0)", terms.len().max(1))
            } else {
                format!("finitev(({sum})/{}.0)", terms.len().max(1))
            };
            out.push_str(&format!(
                "static double {name}(const V& x){{return {finish};}}\n"
            ));
            Ok(())
        }
        V3Head::GamBoost(model) => {
            let mut body = format!(
                "static double {name}(const V& x){{double value={};",
                f64_lit(model.base)
            );
            for term in &model.terms {
                body.push_str(&format!("{{const V edges={};const V values={};const std::size_t bin=std::min(lower(edges,x[{}]),values.size()-1);value+={}*values[bin];}}", f64_vec(&term.edges),f64_vec(&term.values),term.feature,f64_lit(model.learning_rate)));
            }
            body.push_str(if model.binary {
                "return sigmoid(value);}\n"
            } else {
                "return finitev(value);}\n"
            });
            out.push_str(&body);
            Ok(())
        }
        V3Head::Pls(model) => {
            let V3Op::PlsProject { means, weights, .. } = &model.projection else {
                unreachable!("fitted PLS head stores a PLS projection")
            };
            out.push_str(&format!("static double {name}(const V& x){{const V center={};const M projection={};V p;for(const V& w:projection){{double s=0.0;for(std::size_t j=0;j<std::min(x.size(),std::min(center.size(),w.size()));++j)s+=w[j]*(x[j]-center[j]);p.push_back(s);}}const V means={};const V scales={};const V weights={};double value=weights[0];for(std::size_t j=0;j<means.size();++j)value+=weights[j+1]*clampv((p[j]-means[j])/scales[j],-20.0,20.0);return clampv(value,-60.0,60.0);}}\n", f64_vec(means),f64_matrix(weights),f64_vec(&model.ridge.means),f64_vec(&model.ridge.scales),f64_vec(&model.ridge.w)));
            Ok(())
        }
        V3Head::ClassificationPortfolio(model) => {
            let mut terms = Vec::new();
            for (index, (member, weight)) in model.members.iter().zip(&model.weights).enumerate() {
                let child = format!("{name}_{index}");
                emit_head_function(&child, member, out)?;
                terms.push(format!("{}*{}(x)", f64_lit(*weight), child));
            }
            out.push_str(&format!(
                "static double {name}(const V& x){{return clampv({},0.0,1.0);}}\n",
                if terms.is_empty() {
                    "0.0".to_string()
                } else {
                    terms.join("+")
                }
            ));
            Ok(())
        }
        V3Head::MulticlassRidge(_)
        | V3Head::MulticlassOvrLogistic(_)
        | V3Head::MulticlassDiagonalLda(_)
        | V3Head::MulticlassNearestCentroid(_) => {
            bail!("multiclass probability matrices are not supported by the scalar C++ ABI")
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_linear(
    name: &str,
    means: &[f64],
    scales: &[f64],
    weights: &[f64],
    columns: Option<&[usize]>,
    lo: f64,
    hi: f64,
    probability: bool,
    out: &mut String,
) {
    let mapping = columns.map_or_else(
        || "const std::size_t source=j;".to_string(),
        |columns| {
            format!(
                "const U columns={};const std::size_t source=columns[j];",
                usize_vec(columns)
            )
        },
    );
    let finish = if probability {
        "sigmoid(value)"
    } else {
        "clampv(value,-60.0,60.0)"
    };
    out.push_str(&format!("static double {name}(const V& x){{const V means={};const V scales={};const V weights={};double value=weights[0];for(std::size_t j=0;j<means.size();++j){{{mapping}if(source<x.size())value+=weights[j+1]*clampv((x[source]-means[j])/scales[j],{},{});}}return {finish};}}\n",f64_vec(means),f64_vec(scales),f64_vec(weights),f64_lit(lo),f64_lit(hi)));
}

fn emit_legacy_head(name: &str, head: &FittedHead, out: &mut String) -> Result<()> {
    match head {
        FittedHead::Ridge(model) => {
            emit_linear(
                name,
                &model.means,
                &model.scales,
                &model.w,
                None,
                -20.0,
                20.0,
                false,
                out,
            );
            Ok(())
        }
        FittedHead::RidgeGcv(model) => {
            emit_linear(
                name,
                &model.means,
                &model.scales,
                &model.w,
                Some(&model.cols),
                f64::NEG_INFINITY,
                f64::INFINITY,
                false,
                out,
            );
            Ok(())
        }
        FittedHead::ElasticNet(model) => {
            out.push_str(&format!("static double {name}(const V& x){{static const V means={};static const V scales={};static const V weights={};double value={};for(std::size_t j=0;j<means.size();++j)value+=weights[j]*clampv((x[j]-means[j])/scales[j],-20.0,20.0);return clampv(value,-60.0,60.0);}}\n",f64_vec(&model.means),f64_vec(&model.scales),f64_vec(&model.w),f64_lit(model.w0)));
            Ok(())
        }
        FittedHead::Gbm(model) => emit_gbm(name, model, out),
        FittedHead::HistGbm2(model) => emit_hist_gbm2(name, model, out),
        FittedHead::Ensemble { ridge, gbm, w } => {
            let ridge_name = format!("{name}_ridge");
            let gbm_name = format!("{name}_gbm");
            emit_linear(
                &ridge_name,
                &ridge.means,
                &ridge.scales,
                &ridge.w,
                None,
                -20.0,
                20.0,
                false,
                out,
            );
            emit_gbm(&gbm_name, gbm, out)?;
            out.push_str(&format!("static double {name}(const V& x){{return {}*{ridge_name}(x)+(1.0-{})*{gbm_name}(x);}}\n",f64_lit(*w),f64_lit(*w)));
            Ok(())
        }
        FittedHead::Krr(model) => {
            out.push_str(&format!("static double {name}(const V& x){{const V means={};const V scales={};const M basis={};const V alpha={};if(basis.empty())return clampv({},-60.0,60.0);V z;for(std::size_t j=0;j<std::min(x.size(),means.size());++j)z.push_back((x[j]-means[j])/scales[j]);double value={};for(std::size_t i=0;i<basis.size();++i){{double r2=0.0;for(std::size_t j=0;j<std::min(z.size(),basis[i].size());++j){{double q=z[j]-basis[i][j];r2+=q*q;}}value+=alpha[i]*std::exp(-{}*r2);}}return clampv(value,-60.0,60.0);}}\n",f64_vec(&model.means),f64_vec(&model.scales),f64_matrix(&model.basis),f64_vec(&model.alpha),f64_lit(model.y_mean),f64_lit(model.y_mean),f64_lit(model.gamma)));
            Ok(())
        }
        FittedHead::Gpr(model) => {
            let inverse = model
                .ls
                .iter()
                .map(|length| 1.0 / (length * length))
                .collect::<Vec<_>>();
            out.push_str(&format!("static double {name}(const V& x){{const V means={};const V scales={};const M basis={};const V alpha={};const V inv={};if(basis.empty())return clampv({},-60.0,60.0);V z;for(std::size_t j=0;j<std::min(x.size(),means.size());++j)z.push_back((x[j]-means[j])/scales[j]);double value={};for(std::size_t i=0;i<basis.size();++i){{double r2=0.0;for(std::size_t j=0;j<std::min(z.size(),basis[i].size());++j){{double q=z[j]-basis[i][j];r2+=q*q*inv[j];}}value+=alpha[i]*{}*std::exp(-0.5*r2);}}return clampv(value,-60.0,60.0);}}\n",f64_vec(&model.means),f64_vec(&model.scales),f64_matrix(&model.basis),f64_vec(&model.alpha),f64_vec(&inverse),f64_lit(model.y_mean),f64_lit(model.y_mean),f64_lit(model.sf2)));
            Ok(())
        }
        FittedHead::GprKernel(model) => emit_gpr_kernel(name, model, out),
        FittedHead::Ksmooth(model) => {
            out.push_str(&format!("static double {name}(const V& x){{const V means={};const V scales={};const M basis={};const V ys={};if(basis.empty())return clampv({},-60.0,60.0);V z;for(std::size_t j=0;j<std::min(x.size(),means.size());++j)z.push_back((x[j]-means[j])/scales[j]);std::vector<std::pair<double,std::size_t>> order;for(std::size_t i=0;i<basis.size();++i){{double r2=0.0;for(std::size_t j=0;j<std::min(z.size(),basis[i].size());++j){{double q=z[j]-basis[i][j];r2+=q*q;}}order.push_back({{r2,i}});}}std::sort(order.begin(),order.end(),[](const auto&a,const auto&b){{return a.first<b.first||(a.first==b.first&&a.second<b.second);}});const double base=order[0].first;double num=0.0,den=0.0;for(std::size_t q=0;q<std::min<std::size_t>({},order.size());++q){{double w=std::exp(-{}*(order[q].first-base)/static_cast<double>(means.size()));num+=w*ys[order[q].second];den+=w;}}return clampv(num/den,-60.0,60.0);}}\n",f64_vec(&model.means),f64_vec(&model.scales),f64_matrix(&model.z),f64_vec(&model.ys),f64_lit(model.y_mean),model.k,f64_lit(model.gamma)));
            Ok(())
        }
        FittedHead::Hurdle2(model) => emit_hurdle(name, model, out),
        FittedHead::Stack(model) => emit_stack(name, model, out),
        FittedHead::Portfolio(model) => emit_portfolio(name, model, out),
    }
}

fn emit_unclamped_ridge(name: &str, model: &crate::heads::RidgeModel, out: &mut String) {
    out.push_str(&format!(
        "static double {name}(const V& x){{const V means={};const V scales={};const V weights={};double value=weights[0];for(std::size_t j=0;j<std::min(x.size(),means.size());++j)value+=weights[j+1]*clampv((x[j]-means[j])/scales[j],-20.0,20.0);return value;}}\n",
        f64_vec(&model.means),
        f64_vec(&model.scales),
        f64_vec(&model.w),
    ));
}

type DecodedGprKernel = (Vec<f64>, Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>, Vec<Vec<f64>>);

fn decode_gpr_kernel(model: &crate::gpr::GprKernelModel) -> Result<DecodedGprKernel> {
    let width = model.means.len();
    let mut signal = Vec::new();
    let mut inverse = Vec::new();
    let mut alpha = Vec::new();
    let mut period = Vec::new();
    let mut offset = 0usize;
    for term in &model.spec {
        let amplitude = *model
            .theta
            .get(offset)
            .context("gpr kernel theta is truncated at amplitude")?;
        signal.push((2.0 * amplitude).exp());
        offset += 1;
        let mut inverse_term = Vec::new();
        let mut alpha_term = Vec::new();
        let mut period_term = Vec::new();
        for primitive in term {
            match primitive {
                KernelPrim::Rbf => {
                    let mut values = Vec::with_capacity(width);
                    for dimension in 0..width {
                        let length = model
                            .theta
                            .get(offset + dimension)
                            .context("gpr kernel theta is truncated at RBF lengthscale")?
                            .exp();
                        values.push(1.0 / (length * length));
                    }
                    offset += width;
                    inverse_term.push(values);
                    alpha_term.push(0.0);
                    period_term.push(0.0);
                }
                KernelPrim::Rq => {
                    let mut values = Vec::with_capacity(width);
                    for dimension in 0..width {
                        let length = model
                            .theta
                            .get(offset + dimension)
                            .context("gpr kernel theta is truncated at RQ lengthscale")?
                            .exp();
                        values.push(1.0 / (length * length));
                    }
                    let shape = model
                        .theta
                        .get(offset + width)
                        .context("gpr kernel theta is truncated at RQ alpha")?
                        .exp();
                    offset += width + 1;
                    inverse_term.push(values);
                    alpha_term.push(shape);
                    period_term.push(0.0);
                }
                KernelPrim::Periodic => {
                    let cycle = model
                        .theta
                        .get(offset)
                        .context("gpr kernel theta is truncated at period")?
                        .exp();
                    let length = model
                        .theta
                        .get(offset + 1)
                        .context("gpr kernel theta is truncated at periodic lengthscale")?
                        .exp();
                    offset += 2;
                    inverse_term.push(vec![1.0 / (length * length)]);
                    alpha_term.push(0.0);
                    period_term.push(cycle);
                }
                KernelPrim::Linear => {
                    inverse_term.push(Vec::new());
                    alpha_term.push(0.0);
                    period_term.push(0.0);
                }
            }
        }
        inverse.push(inverse_term);
        alpha.push(alpha_term);
        period.push(period_term);
    }
    if model.theta.get(offset).is_none() {
        bail!("gpr kernel theta is truncated at observation noise");
    }
    Ok((signal, inverse, alpha, period))
}

fn emit_gpr_kernel(name: &str, model: &crate::gpr::GprKernelModel, out: &mut String) -> Result<()> {
    if model.basis.is_empty() {
        out.push_str(&format!(
            "static double {name}(const V&){{return clampv({},-60.0,60.0);}}\n",
            f64_lit(model.y_mean)
        ));
        return Ok(());
    }
    let (signal, inverse, alpha, period) = decode_gpr_kernel(model)?;
    let mut kernel = String::from("double kernel=0.0;");
    for (term_index, term) in model.spec.iter().enumerate() {
        kernel.push_str(&format!("{{double term={};", f64_lit(signal[term_index])));
        for (factor_index, primitive) in term.iter().enumerate() {
            match primitive {
                KernelPrim::Rbf => kernel.push_str(&format!(
                    "{{const V inv={};double distance=0.0;for(std::size_t d=0;d<std::min(z.size(),basis_row.size());++d){{double delta=z[d]-basis_row[d];distance+=delta*delta*inv[d];}}term*=std::exp(-0.5*distance);}}",
                    f64_vec(&inverse[term_index][factor_index])
                )),
                KernelPrim::Rq => kernel.push_str(&format!(
                    "{{const V inv={};double distance=0.0;for(std::size_t d=0;d<std::min(z.size(),basis_row.size());++d){{double delta=z[d]-basis_row[d];distance+=delta*delta*inv[d];}}const double shape={};const double scaled=distance/(2.0*shape);term*=std::exp(-shape*std::log(1.0+scaled));}}",
                    f64_vec(&inverse[term_index][factor_index]),
                    f64_lit(alpha[term_index][factor_index]),
                )),
                KernelPrim::Periodic => kernel.push_str(&format!(
                    "{{const double period={};const double inv={};double sine2=0.0;for(std::size_t d=0;d<std::min(z.size(),basis_row.size());++d){{double angle={}*(z[d]-basis_row[d])/period;double sine=std::sin(angle);sine2+=sine*sine;}}term*=std::exp(-2.0*sine2*inv);}}",
                    f64_lit(period[term_index][factor_index]),
                    f64_lit(inverse[term_index][factor_index][0]),
                    f64_lit(std::f64::consts::PI),
                )),
                KernelPrim::Linear => kernel.push_str(&format!(
                    "{{double product=0.0;for(std::size_t d=0;d<std::min(z.size(),basis_row.size());++d)product+=z[d]*basis_row[d];term*=product/{}.0;}}",
                    model.means.len()
                )),
            }
        }
        kernel.push_str("kernel+=term;}");
    }
    out.push_str(&format!(
        "static double {name}(const V& x){{const V means={};const V scales={};const M basis={};const V coefficients={};if(basis.empty())return clampv({},-60.0,60.0);V z;for(std::size_t j=0;j<std::min(x.size(),means.size());++j)z.push_back((x[j]-means[j])/scales[j]);double value={};for(std::size_t row_index=0;row_index<basis.size();++row_index){{const V& basis_row=basis[row_index];{kernel}value+=coefficients[row_index]*kernel;}}return clampv(value,-60.0,60.0);}}\n",
        f64_vec(&model.means),
        f64_vec(&model.scales),
        f64_matrix(&model.basis),
        f64_vec(&model.alpha),
        f64_lit(model.y_mean),
        f64_lit(model.y_mean),
    ));
    Ok(())
}

fn emit_hurdle(name: &str, model: &crate::heads::Hurdle2Model, out: &mut String) -> Result<()> {
    if model.components.is_empty() {
        out.push_str(&format!(
            "static double {name}(const V&){{return {};}}\n",
            f64_lit(model.fallback_mean)
        ));
        return Ok(());
    }
    let mut terms = Vec::new();
    for (index, component) in model.components.iter().enumerate() {
        let gate = format!("{name}_gate_{index}");
        let low = format!("{name}_low_{index}");
        let high = format!("{name}_high_{index}");
        emit_linear(
            &gate,
            &component.gate.means,
            &component.gate.scales,
            &component.gate.w,
            None,
            -20.0,
            20.0,
            true,
            out,
        );
        emit_conditional_ridge(&low, &component.low, out);
        emit_conditional_ridge(&high, &component.high, out);
        terms.push(format!(
            "{{double probability={gate}(x);value+=probability*{high}(x)+(1.0-probability)*{low}(x);}}"
        ));
    }
    out.push_str(&format!(
        "static double {name}(const V& x){{double value=0.0;{}value*=1.0/{}.0;return std::isfinite(value)?value:{};}}\n",
        terms.join(""),
        model.components.len(),
        f64_lit(model.fallback_mean),
    ));
    Ok(())
}

fn emit_conditional_ridge(name: &str, model: &ConditionalRidge, out: &mut String) {
    match model {
        ConditionalRidge::Ridge(model) => emit_unclamped_ridge(name, model, out),
        ConditionalRidge::Mean(value) => out.push_str(&format!(
            "static double {name}(const V&){{return {};}}\n",
            f64_lit(*value)
        )),
    }
}

fn emit_stack(name: &str, model: &crate::stack::StackModel, out: &mut String) -> Result<()> {
    let mut terms = Vec::new();
    let members = [
        model
            .ridge_a
            .as_ref()
            .map(|model| FittedHead::Ridge(model.clone())),
        model
            .ridge_b
            .as_ref()
            .map(|model| FittedHead::Ridge(model.clone())),
        model
            .net
            .as_ref()
            .map(|model| FittedHead::ElasticNet(model.clone())),
        model
            .trees
            .as_ref()
            .map(|model| FittedHead::HistGbm2(model.clone())),
        model
            .kernel
            .as_ref()
            .map(|model| FittedHead::Krr(model.clone())),
    ];
    for (index, member) in members.into_iter().enumerate() {
        let Some(member) = member else { continue };
        let child = format!("{name}_{index}");
        emit_legacy_head(&child, &member, out)?;
        let weight = model.weights.get(index).copied().unwrap_or(0.0);
        terms.push(format!("{}*{child}(x)", f64_lit(weight)));
    }
    out.push_str(&format!(
        "static double {name}(const V& x){{return {};}}\n",
        if terms.is_empty() {
            "0.0".to_string()
        } else {
            terms.join("+")
        }
    ));
    Ok(())
}
