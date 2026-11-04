fn emit_row_operation_stage(
    model: &BoundModel,
    k: usize,
    append: bool,
    order: &FittedTableOrder,
    tails: &[GroupTail],
    roll: bool,
) -> Result<String> {
    let group_columns = order
        .group_columns
        .iter()
        .map(|(column, _)| *column)
        .collect::<Vec<_>>();
    let order_columns = order
        .order_columns
        .iter()
        .map(|(column, _)| *column)
        .collect::<Vec<_>>();
    let mut initializers = String::new();
    for tail in tails {
        let group = row_key_literal(model, &order.group_columns, &tail.key)?;
        let cutoff = row_key_literal(model, &order.order_columns, &tail.last_order)?;
        initializers.push_str(&format!(
            "if(key_equal(active_group,{group})){{history={};cutoff={cutoff};has_cutoff=true;}}",
            tail_rows_literal(tail)
        ));
    }
    let generation = if roll {
        format!(
            "for(std::size_t feature=0;feature<width;++feature){{double sum=0.0;std::size_t count=0;std::size_t start=history.size()>{window}?history.size()-{window}:0;for(std::size_t prior=start;prior<history.size();++prior)if(history[prior].mask[feature]==0.0&&std::isfinite(history[prior].x[feature])){{sum+=history[prior].x[feature];++count;}}if(masks[row][feature]==0.0&&std::isfinite(table[row][feature])){{sum+=table[row][feature];++count;}}if(count>0){{generated[feature]=sum/static_cast<double>(count);generated_mask[feature]=0.0;}}}}",
            window = k.saturating_sub(1)
        )
    } else {
        format!(
            "if(history.size()>={k}){{const Block& prior=history[history.size()-{k}];generated=prior.x;generated_mask=prior.mask;}}"
        )
    };
    let tail_limit = if roll { k.saturating_sub(1) } else { k };
    Ok(format!(
        "{{const U group_columns={groups};const U order_columns={orders};std::vector<Key> group_keys(rows),order_keys(rows);for(std::size_t row=0;row<rows;++row){{group_keys[row]=raw_key(raw+row*columns,missing+row*columns,group_columns);order_keys[row]=raw_key(raw+row*columns,missing+row*columns,order_columns);}}U chronological(rows);for(std::size_t row=0;row<rows;++row)chronological[row]=row;std::sort(chronological.begin(),chronological.end(),[&](std::size_t left,std::size_t right){{if(key_less(group_keys[left],group_keys[right]))return true;if(key_less(group_keys[right],group_keys[left]))return false;if(key_less(order_keys[left],order_keys[right]))return true;if(key_less(order_keys[right],order_keys[left]))return false;return left<right;}});Key active_group;Key cutoff;bool active=false,has_cutoff=false;std::vector<Block> history;for(std::size_t row:chronological){{if(!active||!key_equal(active_group,group_keys[row])){{active=true;active_group=group_keys[row];has_cutoff=false;history.clear();{initializers}}}if(has_cutoff&&!key_less(cutoff,order_keys[row]))return false;const std::size_t width=table[row].size();if(masks[row].size()!=width)throw FittedWidthError{{}};for(const Block& prior:history)if(prior.x.size()!=width||prior.mask.size()!=width)throw FittedWidthError{{}};V generated(width,0.0),generated_mask(width,1.0);{generation}Block current{{table[row],masks[row]}};history.push_back(std::move(current));if(history.size()>{tail_limit})history.erase(history.begin(),history.begin()+(history.size()-{tail_limit}));V input=table[row],input_mask=masks[row];finish_block(input,input_mask,std::move(generated),std::move(generated_mask),{append},table[row],masks[row]);}}}}\n",
        groups = usize_vec(&group_columns),
        orders = usize_vec(&order_columns),
    ))
}

fn emit_table_transforms(model: &BoundModel, ops: &[V3Op]) -> Result<String> {
    let mut helpers = String::new();
    let mut body = String::from(
        "static bool transform_table(M& table,M& masks,const double* raw,const std::uint8_t* missing,std::size_t rows,std::size_t columns){\n",
    );
    let mut segment = Vec::new();
    let mut segment_index = 0usize;
    let flush = |segment: &mut Vec<V3Op>,
                 segment_index: &mut usize,
                 helpers: &mut String,
                 body: &mut String|
     -> Result<()> {
        if segment.is_empty() {
            return Ok(());
        }
        let name = format!("table_segment_{}", *segment_index);
        helpers.push_str(&emit_transforms_named(segment, &name, true)?);
        body.push_str(&format!("for(std::size_t row=0;row<rows;++row){{Block block={name}(std::move(table[row]),std::move(masks[row]));table[row]=std::move(block.x);masks[row]=std::move(block.mask);}}\n"));
        segment.clear();
        *segment_index += 1;
        Ok(())
    };
    for op in ops {
        match op {
            V3Op::Lag {
                k,
                append,
                order,
                tails,
            } => {
                flush(&mut segment, &mut segment_index, &mut helpers, &mut body)?;
                body.push_str(&emit_row_operation_stage(
                    model, *k, *append, order, tails, false,
                )?);
            }
            V3Op::Roll {
                k,
                append,
                order,
                tails,
            } => {
                flush(&mut segment, &mut segment_index, &mut helpers, &mut body)?;
                body.push_str(&emit_row_operation_stage(
                    model, *k, *append, order, tails, true,
                )?);
            }
            other => segment.push(other.clone()),
        }
    }
    flush(&mut segment, &mut segment_index, &mut helpers, &mut body)?;
    body.push_str("return true;}\n");
    Ok(format!("{helpers}{body}"))
}

/// A legacy stage of a legacy predictor, whose runtime is `Model::predict`: `ops::finish` pads an
/// empty output row to `[0.0]`, so the emitted stage does too. The padded cell's mask is 0.0, since
/// the legacy model has no missing mask, and the mask is reset with it, since it may be longer
/// than the row. The V3 pipeline path emits `emit_legacy_op` alone: its runtime
/// (`finish_generated_from_sets`) does not pad.
fn emit_legacy_op_padded(body: &mut String, op: &FittedOp) -> Result<()> {
    emit_legacy_op(body, op)?;
    body.push_str("  if(x.empty()){x.assign(1,0.0);mask.assign(1,0.0);}\n");
    Ok(())
}

fn emit_legacy_op(body: &mut String, op: &FittedOp) -> Result<()> {
    let append = legacy_append(op);
    let generated_mask = match op {
        FittedOp::Standardize { .. }
        | FittedOp::RobustScale { .. }
        | FittedOp::WinsorClip { .. }
        | FittedOp::TargetBinEncode { .. }
        | FittedOp::Sigmoid { .. }
        | FittedOp::Step { .. }
        | FittedOp::NormRow { .. } => "V gm=mask;".to_string(),
        FittedOp::SelectCorr { indices, .. } => format!(
            "V gm;const U mask_indices={};for(std::size_t j:mask_indices)gm.push_back(mask[fitted_index(j,mask.size())]);",
            usize_vec(indices)
        ),
        FittedOp::Impute { .. } => "V gm(g.size(),0.0);".to_string(),
        _ => "V gm(g.size(),0.0);".to_string(),
    };
    let generated = match op {
        FittedOp::Standardize { means, scales, .. } => format!(
            "const V a={};const V b={};for(std::size_t j=0;j<std::min(x.size(),std::min(a.size(),b.size()));++j)g.push_back(clampv((x[j]-a[j])/b[j],-20.0,20.0));",
            f64_vec(means), f64_vec(scales)
        ),
        FittedOp::RobustScale {
            medians, scales, ..
        } => format!(
            "const V a={};const V b={};for(std::size_t j=0;j<std::min(x.size(),std::min(a.size(),b.size()));++j)g.push_back((x[j]-a[j])/b[j]);",
            f64_vec(medians), f64_vec(scales)
        ),
        FittedOp::Rank { sorted_cols, .. } => format!(
            "const M a={};for(std::size_t j=0;j<std::min(x.size(),a.size());++j)g.push_back(mid_rank(a[j],x[j]));",
            f64_matrix(sorted_cols)
        ),
        FittedOp::Blom { sorted_cols, .. } => format!(
            "static const M a={};for(std::size_t j=0;j<std::min(x.size(),a.size());++j){{const double n=static_cast<double>(a[j].size());const double mid=mid_rank(a[j],x[j])*n;g.push_back(probit((mid+0.125)/(n+0.25)));}}",
            f64_matrix(sorted_cols)
        ),
        FittedOp::QuantileBins { edges, bins, .. } => format!(
            "static const M a={};for(std::size_t j=0;j<std::min(x.size(),a.size());++j)g.push_back(static_cast<double>(lower(a[j],x[j]))/({}.0-1.0));",
            f64_matrix(edges), bins
        ),
        FittedOp::BinOnehot { edges, bins, .. } => format!(
            "const M a={};for(std::size_t j=0;j<std::min(x.size(),a.size());++j){{const std::size_t bin=std::min(lower(a[j],x[j]),static_cast<std::size_t>({}-1));for(std::size_t q=0;q<{};++q)g.push_back(q==bin?1.0:0.0);}}",
            f64_matrix(edges), bins, bins
        ),
        FittedOp::Hinge { knots, .. } => format!(
            "const M a={};for(std::size_t j=0;j<std::min(x.size(),a.size());++j)for(double knot:a[j])g.push_back(std::max(x[j]-knot,0.0));",
            f64_matrix(knots)
        ),
        FittedOp::WinsorClip { lo, hi, .. } => format!(
            "const V a={};const V b={};for(std::size_t j=0;j<std::min(x.size(),std::min(a.size(),b.size()));++j)g.push_back(clampv(x[j],a[j],b[j]));",
            f64_vec(lo), f64_vec(hi)
        ),
        FittedOp::TargetBinEncode { edges, enc, .. } => format!(
            "const M a={};const M b={};for(std::size_t j=0;j<std::min(x.size(),std::min(a.size(),b.size()));++j){{if(b[j].empty())throw FittedWidthError{{}};g.push_back(b[j][std::min(lower(a[j],x[j]),b[j].size()-1)]);}}",
            f64_matrix(edges), f64_matrix(enc)
        ),
        FittedOp::Sigmoid { .. } => "for(double v:x)g.push_back(1.0/(1.0+std::exp(-clampv(v,-30.0,30.0))));".to_string(),
        FittedOp::Step { t, .. } => format!("for(double v:x)g.push_back(v>{}?1.0:0.0);", f64_lit(*t)),
        FittedOp::NormRow { .. } => "double s=0.0;for(double v:x)s+=v*v;double inv=1.0/std::max(std::sqrt(s),1e-9);for(double v:x)g.push_back(v*inv);".to_string(),
        FittedOp::RowStats { .. } => "g=row_stats(x);".to_string(),
        FittedOp::FftMag { k, .. } => format!("g=dft_mags(x,{k});"),
        FittedOp::RowSpectral { .. } => "g=row_spectral(x);".to_string(),
        FittedOp::RowDiffStats { .. } => "g=row_diff_stats(x);".to_string(),
        FittedOp::Cheb { degree, .. } => format!("for(double v:x){{const double u=clampv(2.0*v-1.0,-1.0,1.0);double previous=1.0,current=u;for(std::size_t q=0;q<{};++q){{g.push_back(current);const double next=2.0*u*current-previous;previous=current;current=next;}}}}", degree),
        FittedOp::SelectCorr { indices, .. } => format!("const U a={};for(std::size_t j:a)g.push_back(x[fitted_index(j,x.size())]);", usize_vec(indices)),
        FittedOp::Impute { fill, indicator_cols, .. } => format!("const V a={};const U cols={};for(std::size_t j=0;j<x.size();++j)g.push_back(x[j]==0.0?a[fitted_index(j,a.size())]:x[j]);for(std::size_t j:cols)g.push_back(x[fitted_index(j,x.size())]==0.0?1.0:0.0);", f64_vec(fill), usize_vec(indicator_cols)),
        FittedOp::PairProd { pairs, .. } | FittedOp::PolyInteract { pairs, .. } => {
            let pairs = pairs.iter().map(|(a,b)| format!("{{{a},{b}}}")).collect::<Vec<_>>().join(",");
            format!("static const std::vector<std::pair<std::size_t,std::size_t>> pairs{{{pairs}}};for(const auto& p:pairs)g.push_back(x[fitted_index(p.first,x.size())]*x[fitted_index(p.second,x.size())]);")
        }
        FittedOp::PairRatio { pairs, eps, .. } => {
            let pairs = pairs.iter().map(|(a,b)| format!("{{{a},{b}}}")).collect::<Vec<_>>().join(",");
            format!("const std::vector<std::pair<std::size_t,std::size_t>> pairs{{{pairs}}};const V eps={};for(std::size_t i=0;i<pairs.size();++i){{double a=x[fitted_index(pairs[i].first,x.size())],b=x[fitted_index(pairs[i].second,x.size())];g.push_back(a*b/(b*b+eps[fitted_index(i,eps.size())]));}}", f64_vec(eps))
        }
        FittedOp::RffCos { w, b, .. } => format!("const M a={};const V b={};const double scale=std::sqrt(2.0/static_cast<double>(a.size()));for(std::size_t i=0;i<a.size();++i)g.push_back(scale*std::cos(dot(a[i],x)+b[fitted_index(i,b.size())]));", f64_matrix(w), f64_vec(b)),
        FittedOp::RffSin { w, b, .. } => format!("const M a={};const V b={};const double scale=std::sqrt(2.0/static_cast<double>(a.size()));for(std::size_t i=0;i<a.size();++i)g.push_back(scale*std::sin(dot(a[i],x)+b[fitted_index(i,b.size())]));", f64_matrix(w), f64_vec(b)),
        FittedOp::RandProj { w, .. } => format!("const M a={};double inv=1.0/std::sqrt(static_cast<double>(a.size()));for(const V& row:a)g.push_back(dot(row,x)*inv);", f64_matrix(w)),
        FittedOp::LandmarkDist { landmarks, gamma, .. } | FittedOp::Nystrom { landmarks, gamma, .. } => format!("const M a={};for(const V& row:a){{double s=0.0;for(std::size_t j=0;j<std::min(row.size(),x.size());++j){{double q=x[j]-row[j];s+=q*q;}}g.push_back(std::exp(-{}*s));}}", f64_matrix(landmarks), f64_lit(*gamma)),
        FittedOp::ObliqueHinge { w, t, .. } => format!("const M a={};const V b={};for(std::size_t i=0;i<a.size();++i)g.push_back(std::max(dot(a[i],x)-b[fitted_index(i,b.size())],0.0));", f64_matrix(w), f64_vec(t)),
        FittedOp::SvdProject { means, components, .. } => format!("const V a={};const M b={};for(const V& row:b){{double s=0.0;for(std::size_t j=0;j<std::min(x.size(),std::min(a.size(),row.size()));++j)s+=(x[j]-a[j])*row[j];g.push_back(s);}}", f64_vec(means), f64_matrix(components)),
        FittedOp::Symlog { c, .. } => format!("for(double v:x){{const double q=v/{};const double a=std::abs(q),s=q>0.0?1.0:(q<0.0?-1.0:0.0);g.push_back(s*std::log(a+std::sqrt(q*q+1.0)));}}", f64_lit(*c)),
        FittedOp::Polar { .. } => "for(std::size_t j=0;j+1<x.size();j+=2){const double a=x[j],b=x[j+1];g.push_back(std::sqrt(a*a+b*b));g.push_back(std::atan2(b,a));}if(x.size()%2==1)g.push_back(x.back());".to_string(),
        FittedOp::Lag { k, .. } => format!("for(std::size_t j=0;j<x.size();++j)g.push_back(j>={0}?x[j-{0}]:0.0);", k),
        FittedOp::Roll { k, .. } => format!("if(x.empty())throw FittedWidthError{{}};const std::size_t shift={}%x.size();for(std::size_t j=0;j<x.size();++j)g.push_back(x[(j+x.size()-shift)%x.size()]);", k),
        FittedOp::Spline { knots, .. } => format!("const M knots={};for(std::size_t j=0;j<std::min(x.size(),knots.size());++j){{const V& xi=knots[j];const std::size_t n=xi.size();if(n<2)throw FittedWidthError{{}};const double v=x[j];if(xi[n-1]-xi[0]<1e-9){{g.push_back(v);continue;}}g.push_back(v);const double pl=std::max(v-xi[n-1],0.0),pk=std::max(v-xi[n-2],0.0);const double dlast=(pk*pk*pk-pl*pl*pl)/std::max(xi[n-1]-xi[n-2],1e-9);for(std::size_t q=0;q<n-2;++q){{const double pt=std::max(v-xi[q],0.0);const double d=(pt*pt*pt-pl*pl*pl)/std::max(xi[n-1]-xi[q],1e-9);g.push_back(d-dlast);}}}}", f64_matrix(knots)),
        FittedOp::MapExpr { expr, .. } => {
            let expression = expr.to_cpp17(true)?;
            format!("const double* row=x.data();const std::size_t row_len=x.size();(void)row;(void)row_len;for(double element:x){{double x=element;(void)x;g.push_back(math_safe({expression}));}}")
        }
        FittedOp::DeriveExpr { expr, .. } => {
            let expression = expr.to_cpp17(false)?;
            format!("const double* row=x.data();const std::size_t row_len=x.size();(void)row;(void)row_len;g.push_back(math_safe({expression}));")
        }
    };
    body.push_str(&format!(
        "  {{V input=x,input_mask=mask,g;{generated}{generated_mask}finish_block(input,input_mask,std::move(g),std::move(gm),{append},x,mask);}}\n"
    ));
    Ok(())
}
