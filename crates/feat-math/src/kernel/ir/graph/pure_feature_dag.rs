use super::*;

impl InputSchema {
    /// Opt-in semantic-key schema. The union schema sorts keys canonically, so its id is
    /// permutation-invariant. `identity()` above is unchanged and stays order-sensitive.
    pub fn semantic_key_schema(
        &self,
    ) -> std::result::Result<crate::combinators::union::Schema, crate::combinators::union::GraphError>
    {
        self.validate()
            .map_err(|e| crate::combinators::union::GraphError::InvalidSchema(e.to_string()))?;
        if self.rows != RowSemantics::Iid {
            return Err(crate::combinators::union::GraphError::InvalidSchema(
                "pure-feature semantic-key schema requires Iid row semantics; Ordered requires ordered-row execution"
                    .into(),
            ));
        }
        crate::combinators::union::Schema::new(self.columns.clone())
    }

    /// Opt-in binding by semantic key, for use with `combinators::union::Program`.
    pub fn bind_semantic_column(
        &self,
        key: &str,
    ) -> std::result::Result<
        crate::combinators::union::ColumnRef,
        crate::combinators::union::GraphError,
    > {
        let schema = self.semantic_key_schema()?;
        crate::combinators::union::ColumnRef::bind(&schema, key)
    }
}

impl ProgramGraph {
    /// Opt-in admitted construction for the pure feature DAG. Refuses any graph that
    /// `lower_pure_feature_dag` refuses. `new` and the v1 identity are unchanged.
    pub fn new_pure_feature_dag(
        input_schema: InputSchema,
        nodes: Vec<Node>,
        output: usize,
        limits: &Constraints,
        dag_limits: crate::combinators::union::Limits,
    ) -> Result<Self> {
        let graph = Self {
            schema: PURE_FEATURE_DAG_SCHEMA.into(),
            version: 1,
            input_schema,
            profile_id: registry_v2::PROFILE_ID.into(),
            profile_revision: registry_v2::REVISION,
            registry_sha256: registry_v2::source_sha256(),
            nodes,
            output,
        };
        graph.lower_pure_feature_dag(limits, dag_limits)?;
        Ok(graph)
    }

    /// Canonical bytes of an admitted pure feature DAG, bounded by `limits.max_bytes`.
    pub fn to_pure_feature_dag_bytes(
        &self,
        limits: &Constraints,
        dag_limits: crate::combinators::union::Limits,
    ) -> Result<Vec<u8>> {
        self.lower_pure_feature_dag(limits, dag_limits)?;
        let bytes = canon::encode(self)?;
        if bytes.len() > limits.max_bytes {
            return Err("pure feature DAG: graph byte bound".into());
        }
        Ok(bytes)
    }

    /// Parses only canonical pure feature DAG bytes. Legacy v1 graph bytes are refused.
    pub fn from_pure_feature_dag_bytes(
        bytes: &[u8],
        limits: &Constraints,
        dag_limits: crate::combinators::union::Limits,
    ) -> Result<Self> {
        limits.validate()?;
        if bytes.len() > limits.max_bytes {
            return Err("pure feature DAG: graph byte bound".into());
        }
        let graph: Self = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        if graph.to_pure_feature_dag_bytes(limits, dag_limits)? != bytes {
            return Err(
                "noncanonical pure feature DAG bytes, duplicate key or unknown representation"
                    .into(),
            );
        }
        Ok(graph)
    }

    /// Identity under the pure feature DAG domain, separate from the v1 `identity`.
    pub fn pure_feature_dag_identity(
        &self,
        limits: &Constraints,
        dag_limits: crate::combinators::union::Limits,
    ) -> Result<canon::ProgramIdV2> {
        canon::ProgramIdV2::compute(
            PURE_FEATURE_DAG_SCHEMA,
            &self.to_pure_feature_dag_bytes(limits, dag_limits)?,
        )
    }

    /// Opt-in pure-feature route. Lowers the supported subset of `nodes` into a bounded
    /// `combinators::union::Program`, which is validated before it is returned. `lower()` and
    /// the default identity are unchanged.
    ///
    /// Supported: `Input` (raw, not lowered on its own), `Columns` and single-column
    /// `Scalar` over a raw `Input`, and `Stack` over lowered members. Refused: `Operator`,
    /// `Residual`, affine `Scalar`, non-column `Scalar` expressions, and any node that is not
    /// reachable from the output.
    pub fn lower_pure_feature_dag(
        &self,
        limits: &Constraints,
        dag_limits: crate::combinators::union::Limits,
    ) -> Result<crate::combinators::union::Program> {
        use crate::combinators::union::{Action, Node as UNode, Program};
        limits.validate()?;
        self.input_schema.validate()?;
        if self.input_schema.rows != RowSemantics::Iid {
            return Err("pure feature DAG requires Iid row semantics".into());
        }
        if self.schema != PURE_FEATURE_DAG_SCHEMA || self.version != 1 {
            return Err("pure feature DAG: graph schema or version differs".into());
        }
        if self.profile_id != registry_v2::PROFILE_ID
            || self.profile_revision != registry_v2::REVISION
            || self.registry_sha256 != registry_v2::source_sha256()
        {
            return Err("pure feature DAG: profile or registry identity differs".into());
        }
        let n = self.nodes.len();
        if n < 2
            || n > limits.max_nodes
            || self.output >= n
            || !matches!(self.nodes.first(), Some(Node::Input))
        {
            return Err(
                "pure feature DAG: node bound, output, or leading input out of range".into(),
            );
        }
        let mut edges = 0usize;
        let mut depths: Vec<usize> = Vec::with_capacity(n);
        let mut widths: Vec<usize> = Vec::with_capacity(n);
        for (index, node) in self.nodes.iter().enumerate() {
            let deps = match node {
                Node::Input if index != 0 => {
                    return Err("pure feature DAG: multiple graph inputs".into());
                }
                Node::Input => Vec::new(),
                Node::Columns { input, .. } | Node::Scalar { input, .. } => vec![*input],
                Node::Stack { members } => members.clone(),
                Node::Operator { .. } | Node::Residual { .. } => {
                    return Err(format!(
                        "pure feature DAG refuses node {index}: operator and residual nodes need fit or label scope"
                    ));
                }
            };
            for dep in &deps {
                if *dep >= index {
                    return Err(format!(
                        "pure feature DAG: node {index} depends on {dep}, not earlier"
                    ));
                }
            }
            edges = edges
                .checked_add(deps.len())
                .ok_or("pure feature DAG: edge overflow")?;
            if edges > limits.max_edges {
                return Err("pure feature DAG: edge bound".into());
            }
            let depth = deps
                .iter()
                .map(|dep| depths[*dep])
                .max()
                .unwrap_or(0)
                .checked_add(1)
                .ok_or("pure feature DAG: depth overflow")?;
            if depth > limits.max_depth {
                return Err(format!("pure feature DAG: node {index} exceeds max_depth"));
            }
            let width = match node {
                Node::Input => self.input_schema.columns.len(),
                Node::Columns { columns, .. } => columns.len(),
                Node::Scalar { .. } => 1,
                Node::Stack { members } => members.iter().try_fold(0usize, |acc, member| {
                    acc.checked_add(widths[*member])
                        .ok_or("pure feature DAG: width overflow")
                })?,
                Node::Operator { .. } | Node::Residual { .. } => unreachable!("refused above"),
            };
            if width == 0 || width > limits.max_width {
                return Err(format!(
                    "pure feature DAG: node {index} width outside bounds"
                ));
            }
            depths.push(depth);
            widths.push(width);
        }
        // Full reachability from the output, so no node is silently dropped.
        let mut reachable = vec![false; n];
        let mut stack = vec![self.output];
        while let Some(index) = stack.pop() {
            if reachable[index] {
                continue;
            }
            reachable[index] = true;
            match &self.nodes[index] {
                Node::Input => {}
                Node::Columns { input, .. } | Node::Scalar { input, .. } => stack.push(*input),
                Node::Stack { members } => stack.extend(members.iter().copied()),
                Node::Operator { .. } | Node::Residual { .. } => unreachable!("refused above"),
            }
        }
        if reachable.iter().any(|seen| !seen) {
            return Err("pure feature DAG: every node must be reachable from the output".into());
        }
        if canon::encode(self)?.len() > limits.max_bytes {
            return Err("pure feature DAG: graph byte bound".into());
        }

        let mut out: Vec<UNode> = Vec::new();
        // For each graph node: the union node that carries it, or None for the raw input.
        let mut map: Vec<Option<usize>> = vec![None; n];
        // One union `Column` node per semantic key, shared by every consumer.
        let mut column_nodes: BTreeMap<String, usize> = BTreeMap::new();
        for (index, node) in self.nodes.iter().enumerate() {
            match node {
                Node::Input => {}
                Node::Columns { input, columns } => {
                    if !matches!(self.nodes[*input], Node::Input) {
                        return Err(format!(
                            "pure feature DAG: Columns node {index} must read the raw input"
                        ));
                    }
                    if columns.is_empty() {
                        return Err(format!("pure feature DAG: Columns node {index} is empty"));
                    }
                    let mut members = Vec::with_capacity(columns.len());
                    for column in columns {
                        column.validate(&self.input_schema)?;
                        let shared = self.intern_column(&mut column_nodes, &mut out, column)?;
                        members.push(shared);
                    }
                    out.push(UNode::new(Action::ConcatFeatures(members)));
                    map[index] = Some(out.len() - 1);
                }
                Node::Scalar {
                    input,
                    expression,
                    affine,
                } => {
                    if *affine {
                        return Err(format!(
                            "pure feature DAG refuses affine Scalar node {index}: needs the scalar expression scope"
                        ));
                    }
                    if !matches!(self.nodes[*input], Node::Input) {
                        return Err(format!(
                            "pure feature DAG: Scalar node {index} must read the raw input"
                        ));
                    }
                    let crate::kernel::ir::graph::GraphExpression::Column { column } = expression
                    else {
                        return Err(format!(
                            "pure feature DAG refuses Scalar node {index}: only a bare column expression is lowered"
                        ));
                    };
                    column.validate(&self.input_schema)?;
                    let shared = self.intern_column(&mut column_nodes, &mut out, column)?;
                    map[index] = Some(shared);
                }
                Node::Stack { members } => {
                    let mut lowered = Vec::with_capacity(members.len());
                    for member in members {
                        match map[*member] {
                            Some(union_index) => lowered.push(union_index),
                            None => {
                                return Err(format!(
                                    "pure feature DAG: Stack node {index} member {member} is the raw input"
                                ))
                            }
                        }
                    }
                    if lowered.is_empty() {
                        return Err(format!("pure feature DAG: Stack node {index} is empty"));
                    }
                    out.push(UNode::new(Action::ConcatFeatures(lowered)));
                    map[index] = Some(out.len() - 1);
                }
                Node::Operator { .. } | Node::Residual { .. } => unreachable!("refused above"),
            }
        }
        let output = map[self.output].ok_or_else(|| {
            "pure feature DAG: output must be a feature node, not the raw input".to_string()
        })?;
        if out.len() > limits.max_nodes {
            return Err(
                "pure feature DAG: expanded union node count exceeds graph max_nodes".into(),
            );
        }
        let program = Program {
            abi: Program::ABI,
            nodes: out,
            output,
            limits: dag_limits,
        };
        let semantic_schema = self
            .input_schema
            .semantic_key_schema()
            .map_err(|e| e.to_string())?;
        program
            .validate(&semantic_schema, 1)
            .map_err(|e| e.to_string())?;
        Ok(program)
    }

    fn intern_column(
        &self,
        column_nodes: &mut BTreeMap<String, usize>,
        out: &mut Vec<crate::combinators::union::Node>,
        column: &ColumnRef,
    ) -> Result<usize> {
        if let Some(index) = column_nodes.get(&column.key) {
            return Ok(*index);
        }
        let bound = self
            .input_schema
            .bind_semantic_column(&column.key)
            .map_err(|e| e.to_string())?;
        out.push(crate::combinators::union::Node::new(
            crate::combinators::union::Action::Column(bound),
        ));
        let index = out.len() - 1;
        column_nodes.insert(column.key.clone(), index);
        Ok(index)
    }

    /// Lower and evaluate in one step. Every node is counted, and the cache is invocation-local.
    pub fn evaluate_pure_feature_dag_counted(
        &self,
        input: &crate::combinators::union::DenseInput,
        limits: &Constraints,
        dag_limits: crate::combinators::union::Limits,
    ) -> Result<(
        crate::combinators::union::Matrix,
        crate::combinators::union::Validation,
        Vec<usize>,
    )> {
        let program = self.lower_pure_feature_dag(limits, dag_limits)?;
        program
            .evaluate_counted(input)
            .map_err(|error| error.to_string())
    }
}
