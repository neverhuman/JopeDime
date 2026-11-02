use super::*;

impl Program {
    pub const ABI: u16 = 1;

    pub fn validate(&self, schema: &Schema, rows: usize) -> Result<Validation, GraphError> {
        if self.abi != Self::ABI
            || self.nodes.is_empty()
            || self.output >= self.nodes.len()
            || self.nodes.len() > self.limits.max_nodes
            || rows == 0
            || rows > self.limits.max_rows
            || self.limits.max_width == 0
            || self.limits.max_depth == 0
            || self.limits.max_fuel == 0
        {
            return Err(GraphError::Limit(
                "invalid ABI, graph shape, or declared limit".into(),
            ));
        }
        let mut widths = vec![None; self.nodes.len()];
        let mut depths = vec![None; self.nodes.len()];
        let mut state = vec![0u8; self.nodes.len()];
        let mut reachable = BTreeSet::new();
        let mut route_count = 0usize;
        let mut fuel = 0u64;
        let mut fits: BTreeMap<&str, &FitEffect> = BTreeMap::new();
        let mut shared_fit_ids = BTreeSet::new();
        let output_width = self.measure(
            self.output,
            schema,
            rows,
            &mut widths,
            &mut depths,
            &mut state,
            &mut reachable,
            &mut route_count,
            &mut fuel,
            &mut fits,
            &mut shared_fit_ids,
        )?;
        if output_width == 0 || reachable.len() != self.nodes.len() {
            return Err(GraphError::InvalidGraph(
                "every node must be reachable from a nonempty output".into(),
            ));
        }
        if fuel > self.limits.max_fuel {
            return Err(GraphError::Limit(format!(
                "worst-case eager fuel {fuel} exceeds {}",
                self.limits.max_fuel
            )));
        }
        Ok(Validation {
            node_widths: widths.into_iter().map(Option::unwrap).collect(),
            node_depths: depths.into_iter().map(Option::unwrap).collect(),
            worst_case_fuel: fuel,
            eager_route_branches: route_count,
            shared_fit_ids: shared_fit_ids.into_iter().map(str::to_owned).collect(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn measure<'a>(
        &'a self,
        id: usize,
        schema: &Schema,
        rows: usize,
        widths: &mut [Option<usize>],
        depths: &mut [Option<usize>],
        state: &mut [u8],
        reachable: &mut BTreeSet<usize>,
        route_count: &mut usize,
        fuel: &mut u64,
        fits: &mut BTreeMap<&'a str, &'a FitEffect>,
        shared_fit_ids: &mut BTreeSet<&'a str>,
    ) -> Result<usize, GraphError> {
        if id >= self.nodes.len() {
            return Err(GraphError::InvalidGraph(format!(
                "node reference {id} is absent"
            )));
        }
        if state[id] == 1 {
            return Err(GraphError::Cycle(id));
        }
        if let Some(width) = widths[id] {
            return Ok(width);
        }
        state[id] = 1;
        let node = &self.nodes[id];
        reachable.insert(id);
        let children = node.inputs();
        let mut child_widths = Vec::with_capacity(children.len());
        let mut child_depth = 0usize;
        for child in children {
            let width = self.measure(
                child,
                schema,
                rows,
                widths,
                depths,
                state,
                reachable,
                route_count,
                fuel,
                fits,
                shared_fit_ids,
            )?;
            child_widths.push(width);
            child_depth = child_depth.max(depths[child].unwrap());
        }
        let width = match &node.action {
            Action::Column(reference) => {
                if reference.schema_id != schema.id {
                    return Err(GraphError::InvalidSchema(
                        "column reference belongs to a different schema".into(),
                    ));
                }
                if schema.index_of(&reference.key).is_none() {
                    return Err(GraphError::InvalidSchema(format!(
                        "bound column {:?} is not in this schema",
                        reference.key
                    )));
                }
                1
            }
            Action::Union(inputs) => {
                if inputs.is_empty() {
                    return Err(GraphError::InvalidGraph("union must have inputs".into()));
                }
                child_widths.iter().try_fold(0usize, |sum, next| {
                    sum.checked_add(*next)
                        .ok_or_else(|| GraphError::Limit("union width overflow".into()))
                })?
            }
            Action::Route { .. } => {
                *route_count = route_count.saturating_add(1);
                if child_widths.len() != 3 || child_widths[0] != 1 {
                    return Err(GraphError::InvalidGraph(
                        "route predicate must have width one".into(),
                    ));
                }
                if child_widths[1] != child_widths[2] {
                    return Err(GraphError::InvalidGraph(
                        "eager route branches must have equal widths".into(),
                    ));
                }
                child_widths[1]
            }
            Action::Let(_) => child_widths[0],
            Action::SelectFeature {
                column_index,
                input: _,
            } => {
                if *column_index >= child_widths[0] {
                    return Err(GraphError::InvalidGraph(format!(
                        "node {id} selects column {column_index} of width {}",
                        child_widths[0]
                    )));
                }
                1
            }
            Action::ConcatFeatures(inputs) => {
                if inputs.is_empty() {
                    return Err(GraphError::InvalidGraph(
                        "concatenation must have inputs".into(),
                    ));
                }
                child_widths.iter().try_fold(0usize, |sum, next| {
                    sum.checked_add(*next)
                        .ok_or_else(|| GraphError::Limit("concatenation width overflow".into()))
                })?
            }
            Action::Affine { scale, offset, .. } => {
                affine_parameters(id, scale, offset)?;
                child_widths[0]
            }
        };
        if width == 0 || width > self.limits.max_width {
            return Err(GraphError::Limit(format!(
                "node {id} width {width} exceeds {}",
                self.limits.max_width
            )));
        }
        let depth = child_depth.saturating_add(1);
        if depth > self.limits.max_depth {
            return Err(GraphError::Limit(format!(
                "node {id} depth {depth} exceeds {}",
                self.limits.max_depth
            )));
        }
        let node_fuel = (rows as u64)
            .checked_mul(width as u64)
            .ok_or_else(|| GraphError::Limit("fuel arithmetic overflow".into()))?;
        *fuel = fuel
            .checked_add(node_fuel)
            .ok_or_else(|| GraphError::Limit("fuel arithmetic overflow".into()))?;
        if let Some(effect) = &node.fit_effect {
            if effect.fit_id.trim().is_empty()
                || effect.target_id.trim().is_empty()
                || effect.train_row_ids.is_empty()
                || effect
                    .train_row_ids
                    .iter()
                    .copied()
                    .collect::<BTreeSet<_>>()
                    .len()
                    != effect.train_row_ids.len()
            {
                return Err(GraphError::InvalidGraph(
                    "incomplete target-aware fit effect".into(),
                ));
            }
            let first_fit = if let Some(previous) = fits.get(effect.fit_id.as_str()) {
                if *previous != effect {
                    return Err(GraphError::InvalidGraph(format!(
                        "shared fit identity {:?} crosses incompatible target, role, or rows",
                        effect.fit_id
                    )));
                }
                shared_fit_ids.insert(effect.fit_id.as_str());
                false
            } else {
                fits.insert(effect.fit_id.as_str(), effect);
                true
            };
            if first_fit {
                *fuel = fuel
                    .checked_add(effect.fuel)
                    .ok_or_else(|| GraphError::Limit("fit fuel arithmetic overflow".into()))?;
            }
        }
        if *fuel > self.limits.max_fuel {
            return Err(GraphError::Limit(format!(
                "worst-case eager fuel exceeds {}",
                self.limits.max_fuel
            )));
        }
        widths[id] = Some(width);
        depths[id] = Some(depth);
        state[id] = 2;
        Ok(width)
    }
}
