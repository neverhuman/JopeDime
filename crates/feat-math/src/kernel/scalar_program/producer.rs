//! Canonical Experimental scalar producer declarations for model readers.
//! Reading a declaration does not register or enable its generation consumer.
use crate::kernel::profile::{profile, SearchProfile};
use crate::kernel::scalar_program::ScalarCarrier;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const BIRTH_STREAM: u64 = 0x7363616c62697274;
const SITE_STREAM: u64 = 0x6d7574617632;
const OPT_IN_EDIT_STREAM: u64 = 0x7363616c65646974;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScalarProducerKind {
    InverseLinks,
    StableExponentials,
    NumericalPolicy,
    All,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProducerDeclaration {
    schema: String,
    revision: u8,
    class: String,
    kind: ScalarProducerKind,
    base_profile: String,
    base_profile_sha256: String,
    carrier: ScalarCarrier,
    input_width: usize,
    max_nodes: usize,
    max_depth: usize,
    enable_functions: bool,
    enable_inverse_links: bool,
    enable_stable_exponentials: bool,
    enable_numerical_policy: bool,
    function_catalog: Vec<u16>,
    reduction_catalog: Vec<String>,
    policy_birth_choices: Vec<String>,
    admission_contract: String,
    opt_in_numerator: u8,
    opt_in_denominator: u8,
    birth_stream: u64,
    site_stream: u64,
    opt_in_edit_stream: u64,
    rejection: String,
}

#[derive(Clone, Debug)]
pub struct ScalarProducerProfile {
    base: SearchProfile,
    declaration: ProducerDeclaration,
}

impl ScalarProducerProfile {
    pub fn new(
        kind: ScalarProducerKind,
        carrier: ScalarCarrier,
        width: usize,
    ) -> Result<Self, String> {
        if !(1..=512).contains(&width) {
            return Err("scalar profile width".into());
        }
        let base = profile("kernel2-baseline", 3).ok_or("existing scalar base profile")?;
        let functions = match kind {
            ScalarProducerKind::InverseLinks => vec![19, 20],
            ScalarProducerKind::StableExponentials => vec![19, 20, 21, 22],
            ScalarProducerKind::NumericalPolicy => vec![13, 19, 20, 21, 22],
            ScalarProducerKind::All => vec![13, 19, 20, 21, 22],
        };
        let numerical = matches!(
            kind,
            ScalarProducerKind::NumericalPolicy | ScalarProducerKind::All
        );
        let declaration = ProducerDeclaration {
            schema: "feat-math.scalar-search-profile.v1".into(),
            revision: 1,
            class: "experimental".into(),
            kind,
            base_profile: base.canonical_json(),
            base_profile_sha256: base.sha256(),
            carrier,
            input_width: width,
            max_nodes: 15,
            max_depth: 4,
            enable_functions: true,
            enable_inverse_links: true,
            enable_stable_exponentials: kind != ScalarProducerKind::InverseLinks,
            enable_numerical_policy: numerical,
            function_catalog: functions,
            reduction_catalog: if numerical {
                ["sum", "mean", "l1", "l2"].map(String::from).to_vec()
            } else {
                vec![]
            },
            policy_birth_choices: if numerical {
                ["preserve_body", "sum", "mean", "l1", "l2"].map(String::from).to_vec()
            } else {
                vec![]
            },
            admission_contract: "existing-rev3-projection;typed-catalog;cap15/4;variable-leaf;canonical-literals;binary-text-fixed-point".into(),
            opt_in_numerator: 1,
            opt_in_denominator: 4,
            birth_stream: BIRTH_STREAM,
            site_stream: SITE_STREAM,
            opt_in_edit_stream: OPT_IN_EDIT_STREAM,
            rejection: "retain_parent_without_redraw".into(),
        };
        Ok(Self { base, declaration })
    }

    pub fn generation_disabled(mut self) -> Self {
        self.declaration.enable_functions = false;
        self.declaration.enable_inverse_links = false;
        self.declaration.enable_stable_exponentials = false;
        self.declaration.enable_numerical_policy = false;
        self
    }

    /// Disable this family while retaining its declared catalog slots.
    /// Consumers must reject disabled draws without redrawing.
    pub fn inverse_links_disabled(mut self) -> Self {
        self.declaration.enable_inverse_links = false;
        self
    }

    pub fn stable_exponentials_disabled(mut self) -> Self {
        self.declaration.enable_stable_exponentials = false;
        self
    }

    pub fn numerical_policy_disabled(mut self) -> Self {
        self.declaration.enable_numerical_policy = false;
        self
    }

    pub fn canonical_json(&self) -> String {
        serde_json::to_string(&self.declaration).expect("owned scalar declaration")
    }

    pub fn sha256(&self) -> String {
        format!("{:x}", Sha256::digest(self.canonical_json().as_bytes()))
    }

    pub fn from_canonical(text: &str) -> Result<Self, String> {
        if text.len() > 16 * 1024 * 1024 {
            return Err("scalar profile size".into());
        }
        let declaration: ProducerDeclaration =
            serde_json::from_str(text).map_err(|e| e.to_string())?;
        let mut expected = Self::new(
            declaration.kind,
            declaration.carrier,
            declaration.input_width,
        )?;
        expected.declaration.enable_functions = declaration.enable_functions;
        expected.declaration.enable_inverse_links = declaration.enable_inverse_links;
        expected.declaration.enable_stable_exponentials = declaration.enable_stable_exponentials;
        expected.declaration.enable_numerical_policy = declaration.enable_numerical_policy;
        if declaration.kind == ScalarProducerKind::InverseLinks
            && declaration.enable_stable_exponentials
        {
            return Err("exponentials outside scalar profile".into());
        }
        if !matches!(
            declaration.kind,
            ScalarProducerKind::NumericalPolicy | ScalarProducerKind::All
        ) && declaration.enable_numerical_policy
        {
            return Err("policy outside scalar profile".into());
        }
        if expected.declaration != declaration || expected.canonical_json() != text {
            return Err("unsupported or noncanonical scalar profile".into());
        }
        Ok(expected)
    }

    pub fn carrier(&self) -> ScalarCarrier {
        self.declaration.carrier
    }
    pub fn input_width(&self) -> usize {
        self.declaration.input_width
    }

    /// The immutable base declaration reused by typed scalar generation.
    pub fn base_profile(&self) -> &SearchProfile {
        &self.base
    }
}
