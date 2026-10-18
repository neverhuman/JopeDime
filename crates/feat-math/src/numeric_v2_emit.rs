//! Additive fitted prediction artifacts. No legacy templates are changed.
use crate::numeric_v2::{NumericModelV2, TransformV2, PROFILE_ID, SEMANTICS_ID};
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const EMITTER_ID: &str = "fitted-prediction-v2";
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TargetV2 {
    Python,
    Rust,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OptionsV2 {
    pub target: TargetV2,
    pub numeric_profile: String,
    pub semantics: String,
}
impl OptionsV2 {
    pub fn new(target: TargetV2) -> Self {
        Self {
            target,
            numeric_profile: PROFILE_ID.into(),
            semantics: SEMANTICS_ID.into(),
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FittedArtifactV2 {
    pub schema: String,
    pub emitter: String,
    pub options: OptionsV2,
    pub options_sha256: String,
    pub template_sha256: String,
    pub numeric_source_sha256: String,
    pub model_sha256: String,
    pub source_sha256: String,
    pub codec: String,
    pub source: String,
}
pub fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn numeric_source_hash() -> String {
    let mut bytes = Vec::new();
    for source in [
        include_bytes!("numeric_v2.rs").as_slice(),
        include_bytes!("heads.rs").as_slice(),
        include_bytes!("ops.rs").as_slice(),
        include_bytes!("stack.rs").as_slice(),
        include_bytes!("rng.rs").as_slice(),
    ] {
        bytes.extend((source.len() as u64).to_le_bytes());
        bytes.extend(source);
    }
    hash(&bytes)
}
fn template_hash() -> String {
    hash(include_bytes!("numeric_v2_emit.rs"))
}

pub fn emit(model: &NumericModelV2, options: OptionsV2) -> Result<FittedArtifactV2> {
    ensure!(
        options.numeric_profile == PROFILE_ID && options.semantics == SEMANTICS_ID,
        "numeric options mismatch"
    );
    // A canonical refit-independent codec binds all fitted state and schema.
    let codec = serde_json::to_string(model)?;
    let clean: NumericModelV2 = serde_json::from_str(&codec)?;
    clean.predict(&clean.profile.schema, &Vec::new())?;
    let source = render(&clean, options.target)?;
    ensure!(
        crate::transpile::lint_shipped_text(&source).is_empty(),
        "emitted text hygiene failure"
    );
    Ok(FittedArtifactV2 {
        schema: "numeric-fitted-artifact-v2".into(),
        emitter: EMITTER_ID.into(),
        options_sha256: hash(&serde_json::to_vec(&options)?),
        template_sha256: template_hash(),
        numeric_source_sha256: numeric_source_hash(),
        model_sha256: hash(codec.as_bytes()),
        source_sha256: hash(source.as_bytes()),
        options,
        codec,
        source,
    })
}
impl FittedArtifactV2 {
    /// Callers must supply expected options and fitted-state digest from their request.
    pub fn reload(&self, expected: &OptionsV2, model_digest: &str) -> Result<NumericModelV2> {
        ensure!(
            self.schema == "numeric-fitted-artifact-v2" && self.emitter == EMITTER_ID,
            "artifact version mismatch"
        );
        ensure!(
            &self.options == expected
                && self.options_sha256 == hash(&serde_json::to_vec(expected)?),
            "requested target/options mismatch"
        );
        ensure!(
            self.numeric_source_sha256 == numeric_source_hash(),
            "numeric source identity mismatch"
        );
        ensure!(
            self.template_sha256 == template_hash(),
            "template identity mismatch"
        );
        ensure!(
            self.model_sha256 == model_digest && self.model_sha256 == hash(self.codec.as_bytes()),
            "fitted state mismatch"
        );
        ensure!(
            self.source_sha256 == hash(self.source.as_bytes()),
            "artifact bytes mismatch"
        );
        let model: NumericModelV2 = serde_json::from_str(&self.codec)?;
        let regenerated = emit(&model, expected.clone())?;
        ensure!(
            regenerated.source == self.source,
            "source/fitted-state mismatch"
        );
        Ok(model)
    }
}
fn render(m: &NumericModelV2, target: TargetV2) -> Result<String> {
    let py = target == TargetV2::Python;
    let mut s = if py {
        format!("import math\n\ndef predict(rows):\n    out = []\n    for row in rows:\n        if len(row) != {}: raise ValueError('width mismatch')\n        a = [row[j] if math.isfinite(row[j]) else f for j,f in zip({:?},{:?})]\n",m.profile.schema.len(),m.indices,m.fill)
    } else {
        format!("pub fn predict(rows: &[Vec<f64>]) -> Result<Vec<f64>, String> {{\nlet mut out=Vec::new();\nfor row in rows {{\nif row.len() != {} {{ return Err(\"width mismatch\".into()); }}\nlet idx: &[usize]=&{:?}; let fill: &[f64]=&{:?};\nlet mut a: Vec<f64>=idx.iter().enumerate().map(|(k,&j)|if row[j].is_finite(){{row[j]}}else{{fill[k]}}).collect();\n",m.profile.schema.len(),m.indices,m.fill)
    };
    for t in &m.transforms {
        let ap;
        match t {
            TransformV2::Poly { pairs, append } => {
                ap = *append;
                if py {
                    s += &format!("        b = [a[i]*a[j] for i,j in {:?}]\n", pairs);
                } else {
                    s+=&format!("let pairs: &[(usize,usize)]=&{:?}; let mut b:Vec<f64>=pairs.iter().map(|&(i,j)|a[i]*a[j]).collect();\n",pairs);
                }
            }
            TransformV2::Quantile {
                edges,
                bins,
                onehot,
                append,
            } => {
                ap = *append;
                if py {
                    s+=&format!("        b = []\n        for j,e in enumerate({edges:?}):\n            lo,hi = 0,len(e)\n            while lo < hi:\n                mid = lo+(hi-lo)//2\n                if e[mid] < a[j]: lo = mid+1\n                else: hi = mid\n");
                    if *onehot {
                        s += &format!(
                            "            b.extend([float(k == lo) for k in range({bins})])\n"
                        );
                    } else {
                        s += &format!("            b.append(lo/{})\n", bins - 1);
                    }
                } else {
                    let e = edges
                        .iter()
                        .map(|e| format!("vec!{e:?}"))
                        .collect::<Vec<_>>()
                        .join(",");
                    s+=&format!("let edges: Vec<Vec<f64>>=vec![{e}]; let mut b=Vec::new(); for (j,e) in edges.iter().enumerate() {{ let k=e.partition_point(|v| *v < a[j]);\n");
                    if *onehot {
                        s += &format!(
                            "b.extend((0..{bins}).map(|v| if v==k {{1.0}}else{{0.0}}));\n"
                        );
                    } else {
                        s += &format!("b.push(k as f64 / {:?});\n", (bins - 1) as f64);
                    }
                    s += "}\n";
                }
            }
        }
        if py {
            s += if ap {
                "        a = a+b\n"
            } else {
                "        a = b\n"
            };
            s+="        if len(a) > 512 or not all(math.isfinite(v) for v in a): raise ValueError('transform overflow')\n";
        } else {
            s += if ap { "a.append(&mut b);\n" } else { "a=b;\n" };
            s+="if a.len()>512 || a.iter().any(|v| !v.is_finite()) { return Err(\"transform overflow\".into()); }\n";
        }
    }
    if py {
        s+=&format!("        z = {:?}\n        for j,(mu,scale,w) in enumerate(zip({:?},{:?},{:?})):\n            z += w*max(-20.0,min(20.0,(a[j]-mu)/scale))\n        if not math.isfinite(z): raise ValueError('prediction overflow')\n        out.append(max({:?},min({:?},z)))\n    return out\n",m.ridge.w[0],m.ridge.means,m.ridge.scales,&m.ridge.w[1..],m.clamp_lo,m.clamp_hi);
    } else {
        s+=&format!("let means: &[f64]=&{:?}; let scales: &[f64]=&{:?}; let weights: &[f64]=&{:?}; let mut z={:?};\nfor j in 0..means.len() {{ z+=weights[j+1]*((a[j]-means[j])/scales[j]).clamp(-20.0,20.0); }}\nif !z.is_finite() {{return Err(\"prediction overflow\".into());}}\nout.push(z.clamp({:?},{:?}));\n}} Ok(out) }}\n",m.ridge.means,m.ridge.scales,m.ridge.w,m.ridge.w[0],m.clamp_lo,m.clamp_hi);
    }
    Ok(s)
}
