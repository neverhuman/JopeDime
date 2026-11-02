use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Target {
    Python,
    Rust,
    Cpp17,
    Cpp20,
    #[serde(rename = "javascript")]
    JavaScript,
    #[serde(rename = "typescript")]
    TypeScript,
    #[serde(rename = "go")]
    Go,
    Cpp17RecipeV2,
    Cpp17Fitted,
    C17,
    Cuda,
}
impl Target {
    pub const RECIPES: [Self; 5] = [
        Self::Python,
        Self::Rust,
        Self::Cpp17,
        Self::Cpp20,
        Self::JavaScript,
    ];
    pub fn parse(value: &str) -> Result<Self> {
        Ok(match value.trim().to_ascii_lowercase().as_str() {
            "python" | "py" => Self::Python,
            "rust" | "rs" => Self::Rust,
            "cpp17" | "c++17" => Self::Cpp17,
            "cpp20" | "c++20" => Self::Cpp20,
            "javascript" | "js" => Self::JavaScript,
            "typescript" | "ts" => Self::TypeScript,
            "go" | "golang" => Self::Go,
            "cpp17-recipe-v2" => Self::Cpp17RecipeV2,
            "cpp17-fitted" => Self::Cpp17Fitted,
            "c" | "c17" => Self::C17,
            "cuda" | "cuda-cpp" | "cu" => Self::Cuda,
            _ => anyhow::bail!("unknown requested target {value:?}"),
        })
    }
    /// New requests refuse duplicates, including aliases. Legacy parsers are unchanged.
    pub fn parse_list(value: &str) -> Result<Vec<Self>> {
        let mut out = Vec::new();
        for word in value.split(',') {
            if word.trim().eq_ignore_ascii_case("all") {
                out.extend(Self::RECIPES);
            } else {
                out.push(Self::parse(word)?);
            }
        }
        validate_requests(&out)?;
        out.sort();
        Ok(out)
    }
    pub fn path(self) -> &'static str {
        match self {
            Self::C17 => "c17/model.h",
            Self::Cuda => "cuda/model.cu",
            Self::Python => "model.py",
            Self::Rust => "model.rs",
            Self::Cpp17 | Self::Cpp20 => "model.cpp",
            Self::JavaScript => "model.js",
            Self::TypeScript => "model.ts",
            Self::Go => "model.go",
            Self::Cpp17RecipeV2 => "cpp17-recipe-v2/model.cpp",
            Self::Cpp17Fitted => "cpp17-fitted/model.cpp",
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Self::C17 => "c17",
            Self::Cuda => "cuda",
            Self::Python => "python",
            Self::Rust => "rust",
            Self::Cpp17 => "cpp17",
            Self::Cpp20 => "cpp20",
            Self::JavaScript => "javascript",
            Self::TypeScript => "typescript",
            Self::Go => "go",
            Self::Cpp17RecipeV2 => "cpp17-recipe-v2",
            Self::Cpp17Fitted => "cpp17-fitted",
        }
    }
    pub fn registry_target(self) -> &'static str {
        match self {
            // The C boundary invokes the safe Rust recipe without changing arithmetic.
            Self::C17 | Self::Cuda => "rust_recipe",
            Self::Python => "python_recipe",
            Self::Rust | Self::Go | Self::TypeScript => "rust_recipe",
            Self::Cpp17 | Self::Cpp20 => "cpp17_recipe",
            Self::JavaScript => "javascript",
            Self::Cpp17RecipeV2 => "cpp17_recipe_v2",
            Self::Cpp17Fitted => "cpp17_fitted",
        }
    }
    pub fn product(self, dialect: &str) -> &'static str {
        match self {
            Self::Cpp17RecipeV2 => "expression-recipe-v2",
            Self::Cpp17Fitted => "fitted-cpp17",
            _ if dialect == "advanced_v3" => "bound-predictor-v2",
            _ => "training-recipe-v1",
        }
    }
}
fn validate_requests(requested: &[Target]) -> Result<()> {
    ensure!(!requested.is_empty(), "empty requested target set");
    ensure!(
        requested.iter().collect::<BTreeSet<_>>().len() == requested.len(),
        "duplicate requested target"
    );
    Ok(())
}
