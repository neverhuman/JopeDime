use super::*;

impl ProgramGraph {
    pub fn new(
        input_schema: InputSchema,
        nodes: Vec<Node>,
        output: usize,
        limits: &Constraints,
    ) -> Result<Self> {
        let graph = Self {
            schema: GRAPH_SCHEMA.into(),
            version: 1,
            input_schema,
            profile_id: registry_v2::PROFILE_ID.into(),
            profile_revision: registry_v2::REVISION,
            registry_sha256: registry_v2::source_sha256(),
            nodes,
            output,
        };
        graph.validate(limits)?;
        Ok(graph)
    }
    pub fn input_schema(&self) -> &InputSchema {
        &self.input_schema
    }
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }
    pub fn output(&self) -> usize {
        self.output
    }
    pub fn validate(&self, limits: &Constraints) -> Result<Effects> {
        Ok(self.lower(limits)?.effects)
    }
    pub fn to_bytes(&self, limits: &Constraints) -> Result<Vec<u8>> {
        self.validate(limits)?;
        canon::encode(self)
    }
    pub fn from_bytes(bytes: &[u8], limits: &Constraints) -> Result<Self> {
        limits.validate()?;
        if bytes.len() > limits.max_bytes {
            return Err("graph byte bound".into());
        }
        let graph: Self = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        if graph.to_bytes(limits)? != bytes {
            return Err("noncanonical graph bytes, duplicate key or unknown representation".into());
        }
        Ok(graph)
    }
    pub fn identity(&self, limits: &Constraints) -> Result<canon::ProgramIdV2> {
        canon::ProgramIdV2::compute(GRAPH_SCHEMA, &self.to_bytes(limits)?)
    }
    pub(crate) fn lower(&self, limits: &Constraints) -> Result<Lowered> {
        limits.validate()?;
        self.input_schema.validate()?;
        if self.schema != GRAPH_SCHEMA
            || self.version != 1
            || self.profile_id != registry_v2::PROFILE_ID
            || self.profile_revision != registry_v2::REVISION
            || self.registry_sha256 != registry_v2::source_sha256()
        {
            return Err("graph version/profile/registry identity mismatch".into());
        }
        if self.nodes.len() < 2
            || self.nodes.len() > limits.max_nodes
            || self.output != self.nodes.len() - 1
            || !matches!(self.nodes.first(), Some(Node::Input))
        {
            return Err("graph node/output bound or missing input".into());
        }
        let mut ports = Vec::new();
        let mut effects = Vec::new();
        let mut depths = Vec::new();
        let mut edges = 0usize;
        let mut stages = Vec::new();
        let mut columns: Vec<usize> = (0..self.input_schema.columns.len()).collect();
        let mut symbolic = None;
        for (index, node) in self.nodes.iter().enumerate() {
            let deps = node.dependencies();
            edges = edges.checked_add(deps.len()).ok_or("edge overflow")?;
            if edges > limits.max_edges || deps.iter().any(|d| *d >= index) {
                return Err("cycle, forward/unknown node reference or edge bound".into());
            }
            let depth = deps.iter().map(|d| depths[*d]).max().unwrap_or(0) + 1;
            if depth > limits.max_depth {
                return Err("graph depth bound".into());
            }
            let mut effect = deps
                .iter()
                .fold(Effects::default(), |a, d| a.union(effects[*d]));
            let port=match node {
                Node::Input if index==0=>Port::Matrix(self.input_schema.columns.len()),
                Node::Input=>return Err("multiple graph inputs".into()),
                Node::Columns{input,columns:refs}=>{
                    if *input!=0||index!=1||refs.is_empty()||refs.len()>limits.max_width{return Err("column projection must directly follow input within width bound".into());}
                    for column in refs{column.validate(&self.input_schema)?;}
                    columns=refs.iter().map(|c|c.index).collect();Port::Matrix(columns.len())
                },
                Node::Operator{input,operator,parameters}=>{
                    if *input!=index-1{return Err("executable adapter requires a linear stage chain".into());}
                    let Port::Matrix(width)=ports[*input]else{return Err("matrix input required".into());};
                    if width>limits.max_width{return Err("fit width bound; select schema columns before native fitting".into());}
                    let entry=operator.entry()?;let facts=entry.facts();let materialized=stage(operator,parameters)?;
                    effect.fit_x|=facts.fitted.ok_or("unknown fitting effect")?;effect.fit_y|=facts.requires_labels;effect.target_derived|=facts.requires_labels;
                    effect.ordered_rows|=entry.axis=="ordered_rows";
                    effect.seeded|=materialized.params.contains_key("seed");
                    let port=if facts.role=="head"{
                        if !supported_head(facts.canonical){return Err(format!("{}: head/composition adapter unavailable; nested cross-fitting not proven",operator.name));}
                        Port::Prediction
                    }else if facts.role=="transform"&&entry.inherit=="registry_v1"{
                        let row=registry::operator(&operator.name).ok_or("inherited shape missing")?;
                        let width=match row.shape{
                            ShapeRule::Preserve=>width,
                            ShapeRule::Parameter(key)=>materialized.get_usize(key,0),
                            ShapeRule::Fixed(n)=>n,
                            _=>return Err(format!("{}: exact width adapter unavailable",operator.name)),
                        };
                        let width=if matches!(materialized.params.get("append"),Some(crate::dsl::ParamValue::Bool(true))){width.checked_add(match ports[*input]{Port::Matrix(n)=>n,_=>0}).ok_or("width overflow")?}else{width};
                        if width==0||width>limits.max_width{return Err("operator output width bound".into());}Port::Matrix(width)
                    }else{return Err("target/directive/expansion layout adapter unavailable".into());};
                    stages.push(materialized);port
                },
                Node::Scalar{input,expression,affine}=>{
                    if *input!=0||index!=1{return Err("scalar adapter requires direct schema input".into());}
                    let expr=MathExpr::from_ast(expression.to_ast(&self.input_schema,limits)?).map_err(|e|e.to_string())?;
                    effect.fit_x|=*affine;effect.fit_y|=*affine;effect.target_derived|=*affine;
                    symbolic=Some(GenomeIr::Symbolic{source:expr.to_source().map_err(|e|e.to_string())?,expr,affine:*affine});Port::Prediction
                },
                Node::Stack{..}|Node::Residual{..}=>return Err("nested cross-fitting adapter unavailable for stack/residual; target-derived paths cannot be admitted".into()),
            };
            ports.push(port);
            effects.push(effect);
            depths.push(depth);
        }
        if ports[self.output] != Port::Prediction {
            return Err("graph output must be a prediction port".into());
        }
        let mut live = std::collections::BTreeSet::new();
        let mut pending = vec![self.output];
        while let Some(i) = pending.pop() {
            if live.insert(i) {
                pending.extend(self.nodes[i].dependencies());
            }
        }
        if live.len() != self.nodes.len() {
            return Err("unreachable graph nodes".into());
        }
        if canon::encode(self)?.len() > limits.max_bytes {
            return Err("graph byte bound".into());
        }
        Ok(Lowered {
            genome: symbolic.unwrap_or(GenomeIr::Pipeline(DslExpr { stages })),
            columns,
            effects: effects[self.output],
        })
    }
}
