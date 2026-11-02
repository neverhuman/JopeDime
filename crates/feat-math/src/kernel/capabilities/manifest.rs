#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Kernel2Use {
    Baseline,
    Experimental,
    NotGenerated { reason: &'static str },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Drift {
    None,
    LegacyDrift { reason: &'static str },
    GrammarOnly { reason: &'static str },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StageVocab {
    Legacy,
    Expansion,
    None,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RegistryLifecycle {
    Generatable,
    Compatibility,
    Quarantined,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum V3Admission {
    Baseline,
    RequiresReceipt,
    Disabled,
    Absent,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OpKind {
    Transform,
    Head,
    Target,
    Input,
    Directive,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WidthEffect {
    Registry(ShapeRule),
    None,
    Input,
    OneToOne,
    GeneratedPerFeature,
    ExpansionPerFeature { parameter: &'static str },
    Select { parameter: &'static str },
    Projection { parameter: &'static str },
    Impute,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct GrammarFlags {
    pub v3: Option<bool>,
    pub jain_v1: Option<bool>,
}

#[derive(Clone, Debug, Serialize)]
pub struct OpCap {
    pub name: &'static str,
    pub canonical: &'static str,
    pub kind: OpKind,
    pub wire_id: Option<u16>,
    pub registry_lifecycle: Option<RegistryLifecycle>,
    pub v3_admission: V3Admission,
    pub grammar: GrammarFlags,
    pub stage_vocab: StageVocab,
    pub kernel2: Kernel2Use,
    pub drift: Drift,
    pub width_effect: WidthEffect,
    pub requires_labels: bool,
    pub requires_row_order: bool,
    pub emitters: EmitterSupport,
    pub notes: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct FormCap {
    pub name: &'static str,
    pub kernel2: Kernel2Use,
    pub emitters: EmitterSupport,
    pub notes: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct PolicyCap {
    pub name: &'static str,
    pub class: Class,
    pub birth_bins: Vec<f64>,
    pub emitters: EmitterSupport,
    pub notes: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct GeneratorCap {
    pub id: &'static str,
    pub stage_ops: Vec<&'static str>,
    pub functions: Vec<&'static str>,
    pub binops: Vec<&'static str>,
    pub variables: Vec<&'static str>,
    pub unary: Vec<&'static str>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Manifest {
    pub expression: Vec<ElementRow>,
    pub policies: Vec<PolicyCap>,
    pub ops: Vec<OpCap>,
    pub forms: Vec<FormCap>,
    pub legacy_generators: Vec<GeneratorCap>,
    pub v2_reservations: Vec<wire_v2::ReservationRow>,
}

impl Manifest {
    pub fn current() -> Self {
        let v3 = grammar_ops(include_str!(
            "../../../../jope-engine/assets/grammar/featmath.json"
        ));
        let jain = grammar_ops(include_str!(
            "../../../../jope-engine/assets/grammar/featmath-jain-v1.json"
        ));
        let mut names = BTreeSet::new();
        names.extend(registry::operator_registry().iter().map(|row| row.name));
        names.extend(GRAMMAR_ONLY.iter().copied());
        names.extend(crate::full_rust::TRANSFORMS);
        names.extend(crate::full_rust::HEADS);
        names.extend(crate::full_rust::EXPANSION_TRANSFORMS);
        names.extend(crate::evolve::EXPANSION_TRANSFORMS);
        names.extend(V3_ONLY);
        names.extend(["target", "input", "hurdle2"]);
        let ops = names
            .into_iter()
            .map(|name| build_op(name, &v3, &jain))
            .collect();
        let forms = vec![
            FormCap {
                name: "pipeline",
                kernel2: Kernel2Use::Baseline,
                emitters: EmitterSupport {
                    rust_recipe: YES,
                    python_recipe: YES,
                    cpp17_recipe: YES,
                    cpp17_fitted: YES,
                    javascript: JS_SUBSET,
                },
                notes: "each stage determines its own route",
            },
            FormCap {
                name: "symbolic",
                kernel2: Kernel2Use::NotGenerated { reason: "deferred" },
                emitters: EmitterSupport {
                    rust_recipe: YES,
                    python_recipe: YES,
                    cpp17_recipe: Support::Planned { slice: "X2a" },
                    cpp17_fitted: YES,
                    javascript: NO_JS,
                },
                notes: "closed formula",
            },
            FormCap {
                name: "nested_portfolio",
                kernel2: Kernel2Use::NotGenerated { reason: "deferred" },
                emitters: EmitterSupport {
                    rust_recipe: Support::Deferred { reason: "planning" },
                    python_recipe: Support::Deferred { reason: "planning" },
                    cpp17_recipe: Support::Deferred { reason: "planning" },
                    cpp17_fitted: YES,
                    javascript: NO_JS,
                },
                notes: "explicit nested form",
            },
        ];
        let bins = vec![
            -2.0, -1.0, -0.5, -0.25, -0.1, 0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0,
            20.0,
        ];
        let policies = vec![
            PolicyCap {
                name: "columns",
                class: BASE,
                birth_bins: vec![],
                emitters: EXPR_EMITTERS,
                notes: "indices under 8; wider indices wait for Phase 4",
            },
            PolicyCap {
                name: "constants",
                class: BASE,
                birth_bins: bins,
                emitters: EXPR_EMITTERS,
                notes: "birth bins; finite values remain continuous afterward",
            },
            PolicyCap {
                name: "negation",
                class: BASE,
                birth_bins: vec![],
                emitters: EXPR_EMITTERS,
                notes: "a negative constant only",
            },
        ];
        Self {
            expression: expression_elements(),
            policies,
            ops,
            forms,
            legacy_generators: legacy_generators(),
            v2_reservations: wire_v2::manifest_rows(),
        }
    }

    pub fn op(&self, name: &str) -> Option<&OpCap> {
        self.ops.iter().find(|row| row.name == name)
    }
    pub fn form(&self, name: &str) -> Option<&FormCap> {
        self.forms.iter().find(|row| row.name == name)
    }
}

pub fn op_capability(name: &str) -> Option<&'static OpCap> {
    static MANIFEST: OnceLock<Manifest> = OnceLock::new();
    MANIFEST.get_or_init(Manifest::current).op(name)
}

/// The legacy recipe spelling of lag/roll operates on feature positions in
/// one row. Advanced-v3's separate ordered-row transforms are not this route.
pub fn legacy_recipe_axis(name: &str) -> Option<StageAxis> {
    matches!(name, "lag" | "roll").then_some(StageAxis::legacy_within_row())
}

pub fn form_capability(program: &GenomeIr) -> &'static FormCap {
    static MANIFEST: OnceLock<Manifest> = OnceLock::new();
    let name = match program {
        GenomeIr::Pipeline(_) => "pipeline",
        GenomeIr::Symbolic { .. } => "symbolic",
        GenomeIr::Portfolio(_) => "nested_portfolio",
    };
    MANIFEST
        .get_or_init(Manifest::current)
        .form(name)
        .expect("form table")
}

/// The two shipped token samplers retain their registry-era policy. New grammars
/// use the kernel declaration, including the later v4 grammar.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Authority {
    Registry,
    Manifest,
}

pub const V3_GRAMMAR_SHA256: &str =
    "8d55fee4366acd2c175cf55bfd769cebc904770f5227a6fd50a613723dc85787";
pub const JAIN_V1_GRAMMAR_SHA256: &str =
    "570b2e5e8d859a73b888f5d0bee27f59da0e2376536babce24c241604f818e81";

pub fn authority(grammar_sha256: &str) -> Authority {
    if matches!(grammar_sha256, V3_GRAMMAR_SHA256 | JAIN_V1_GRAMMAR_SHA256) {
        Authority::Registry
    } else {
        Authority::Manifest
    }
}
