use super::*;

/// Decode the exact affine parameters. A malformed or non-finite value is refused.
pub(in crate::combinators) fn affine_parameters(
    id: usize,
    scale: &ExactScalar,
    offset: &ExactScalar,
) -> Result<(f64, f64), GraphError> {
    let scale = scale
        .value()
        .map_err(|e| GraphError::InvalidGraph(format!("node {id} affine scale: {e}")))?;
    let offset = offset
        .value()
        .map_err(|e| GraphError::InvalidGraph(format!("node {id} affine offset: {e}")))?;
    Ok((scale, offset))
}

impl Program {
    /// No portable backend is registered for this draft representation. Refuse
    /// explicitly instead of relabeling interpreter execution as emitted output.
    pub fn emit_for_target(&self, target: &str) -> Result<String, GraphError> {
        Err(GraphError::UnsupportedEmitter(format!(
            "DAG ABI {} has no registered emitter for target {target:?}",
            self.abi
        )))
    }

    pub fn evaluate(&self, input: &DenseInput) -> Result<(Matrix, Validation), GraphError> {
        let (output, validation, _counts) = self.evaluate_counted(input)?;
        Ok((output, validation))
    }

    /// Evaluate and report how many times each node was computed. A node shared by several
    /// consumers is cached after its first computation, so its count is 1.
    pub fn evaluate_counted(
        &self,
        input: &DenseInput,
    ) -> Result<(Matrix, Validation, Vec<usize>), GraphError> {
        if self.nodes.iter().any(|node| node.fit_effect.is_some()) {
            return Err(GraphError::InvalidGraph(
                "pure feature DAG evaluation refuses fit_effect nodes; fit/label capabilities require scoped fit execution"
                    .into(),
            ));
        }
        let validation = self.validate(&input.schema, input.values.len())?;
        if input.values.iter().any(|row| {
            row.len() != input.schema.columns.len() || row.iter().any(|value| !value.is_finite())
        }) {
            return Err(GraphError::Input(
                "invalid dense input shape or value".into(),
            ));
        }
        let mut cache = vec![None; self.nodes.len()];
        let mut counts = vec![0usize; self.nodes.len()];
        let output =
            self.evaluate_node(self.output, input, &validation, &mut cache, &mut counts)?;
        Ok((output, validation, counts))
    }

    fn evaluate_node(
        &self,
        id: usize,
        input: &DenseInput,
        validation: &Validation,
        cache: &mut [Option<Matrix>],
        counts: &mut [usize],
    ) -> Result<Matrix, GraphError> {
        if let Some(matrix) = &cache[id] {
            return Ok(matrix.clone());
        }
        counts[id] += 1;
        let node = &self.nodes[id];
        let rows = input.values.len();
        let matrix = match &node.action {
            Action::Column(reference) => {
                if reference.schema_id != input.schema.id {
                    return Err(GraphError::InvalidSchema("schema identity changed".into()));
                }
                let column = input
                    .schema
                    .index_of(&reference.key)
                    .ok_or_else(|| GraphError::InvalidSchema("column key missing".into()))?;
                Matrix {
                    rows,
                    width: 1,
                    values: input.values.iter().map(|row| row[column]).collect(),
                }
            }
            Action::Union(children) => {
                let matrices = children
                    .iter()
                    .map(|child| self.evaluate_node(*child, input, validation, cache, counts))
                    .collect::<Result<Vec<_>, _>>()?;
                let width = validation.node_widths[id];
                let mut values = Vec::with_capacity(rows * width);
                for row in 0..rows {
                    for matrix in &matrices {
                        let start = row * matrix.width;
                        values.extend_from_slice(&matrix.values[start..start + matrix.width]);
                    }
                }
                Matrix {
                    rows,
                    width,
                    values,
                }
            }
            Action::Route {
                predicate,
                when_true,
                when_false,
            } => {
                // Both branches are deliberately evaluated; validation charged the
                // complete eager branch DAG before any output allocation.
                let predicate = self.evaluate_node(*predicate, input, validation, cache, counts)?;
                let yes = self.evaluate_node(*when_true, input, validation, cache, counts)?;
                let no = self.evaluate_node(*when_false, input, validation, cache, counts)?;
                route_rows(&predicate, &yes, &no)?
            }
            Action::Let(bound) => self.evaluate_node(*bound, input, validation, cache, counts)?,
            Action::SelectFeature {
                input: source,
                column_index,
            } => {
                let source = self.evaluate_node(*source, input, validation, cache, counts)?;
                Matrix {
                    rows,
                    width: 1,
                    values: (0..rows)
                        .map(|row| source.values[row * source.width + column_index])
                        .collect(),
                }
            }
            Action::ConcatFeatures(children) => {
                let matrices = children
                    .iter()
                    .map(|child| self.evaluate_node(*child, input, validation, cache, counts))
                    .collect::<Result<Vec<_>, _>>()?;
                let width = validation.node_widths[id];
                let mut values = Vec::with_capacity(rows * width);
                for row in 0..rows {
                    for matrix in &matrices {
                        let start = row * matrix.width;
                        values.extend_from_slice(&matrix.values[start..start + matrix.width]);
                    }
                }
                Matrix {
                    rows,
                    width,
                    values,
                }
            }
            Action::Affine {
                input: source,
                scale,
                offset,
            } => {
                let (scale, offset) = affine_parameters(id, scale, offset)?;
                let source = self.evaluate_node(*source, input, validation, cache, counts)?;
                Matrix {
                    rows: source.rows,
                    width: source.width,
                    values: source.values.iter().map(|v| v * scale + offset).collect(),
                }
            }
        };
        if matrix.width != validation.node_widths[id]
            || matrix.values.len() != matrix.rows.saturating_mul(matrix.width)
            || matrix.values.iter().any(|value| !value.is_finite())
        {
            return Err(GraphError::Evaluation(
                "node returned malformed matrix".into(),
            ));
        }
        cache[id] = Some(matrix.clone());
        Ok(matrix)
    }
}
