//! Scalar v2 instruction lowering through existing legacy AST emitters.
use crate::kernel::numerical_policy::NumericalPolicy;
use crate::kernel::scalar_execution::Instruction;
use crate::kernel::scalar_program::{ScalarCarrier, ScalarExpressionProgram, ScalarFunction};
use crate::math_expr::{BinOp, Expr, FaultSet, Func, UnOp, Var};
use crate::mathexpr::ExprFeatures;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalarLanguage {
    Rust,
    Python,
    Cpp17,
}

fn register(index: u8) -> Expr {
    Expr::Var(Var::Col(usize::from(index)))
}

fn fragment(instruction: Instruction) -> Option<Expr> {
    Some(match instruction {
        Instruction::Constant(value) => Expr::Const(value),
        Instruction::Variable(var) => Expr::Var(var),
        Instruction::Neg(a) => Expr::Unary(UnOp::Neg, Box::new(register(a))),
        Instruction::Binary(op, a, b) => {
            Expr::Binary(op, Box::new(register(a)), Box::new(register(b)))
        }
        Instruction::Call(ScalarFunction::V1(func), args, count) => Expr::Call(
            func,
            args[..usize::from(count)]
                .iter()
                .map(|&a| register(a))
                .collect(),
        ),
        Instruction::Call(_, _, _) => return None,
    })
}

pub(crate) fn old_helper_features(
    program: &ScalarExpressionProgram,
) -> crate::kernel::Result<ExprFeatures> {
    let mut out = ExprFeatures::default();
    for instruction in program.compile()?.code {
        if let Some(expr) = fragment(instruction) {
            let f = expr.features();
            out.funcs.extend(f.funcs);
            out.vars.extend(f.vars);
            out.uses_div |= f.uses_div;
            out.uses_pow |= f.uses_pow;
        }
    }
    Ok(out)
}

fn legacy_value(instruction: Instruction, language: ScalarLanguage) -> Option<String> {
    if let Instruction::Constant(value) = instruction {
        return Some(match language {
            ScalarLanguage::Rust => format!("f64::from_bits({}u64)", value.to_bits()),
            ScalarLanguage::Python => format!("scalar_bits_v2({})", value.to_bits()),
            ScalarLanguage::Cpp17 => format!("f64_from_bits({}ULL)", value.to_bits()),
        });
    }
    if language == ScalarLanguage::Rust {
        if let Instruction::Call(ScalarFunction::V1(Func::Clip), args, _) = instruction {
            // Clip sorts its bounds with the same native runtime Min/Max
            // kernels. Explicit ties also retain signed zero after folding.
            let v = args[0];
            let a = args[1];
            let b = args[2];
            return Some(format!("registers[{v}].clamp((if registers[{b}].is_nan() || registers[{a}] < registers[{b}] {{ registers[{a}] }} else {{ registers[{b}] }}), (if registers[{b}].is_nan() || registers[{a}] > registers[{b}] {{ registers[{a}] }} else {{ registers[{b}] }}))"));
        }
        if let Instruction::Call(ScalarFunction::V1(function @ (Func::Min | Func::Max)), args, _) =
            instruction
        {
            // The native runtime kernel selects the second operand on ties.
            // LLVM may fold f64::min/max to a different signed-zero selection
            // when a preceding register is constant; keep that tie explicit.
            let a = args[0];
            let b = args[1];
            let cmp = if function == Func::Min { "<" } else { ">" };
            return Some(format!("(if registers[{b}].is_nan() || registers[{a}] {cmp} registers[{b}] {{ registers[{a}] }} else {{ registers[{b}] }})"));
        }
        if let Instruction::Binary(BinOp::Pow, a, b) = instruction {
            // Native operands are runtime values. Hide both operands so LLVM
            // cannot replace a literal power-of-two base with exp2, or fold a
            // literal exponent into another kernel with different IEEE bits.
            return Some(format!("std::hint::black_box(registers[{a}]).powf(std::hint::black_box(registers[{b}]).clamp(-8.0,8.0))"));
        }
    }
    // Native old min/max choose the second operand on equal values. Clang's
    // fmin/fmax constant folding can choose the other signed zero. This local
    // scalar-v2 comparison preserves that observed contract and v1 source.
    if language == ScalarLanguage::Cpp17 {
        if let Instruction::Binary(BinOp::Pow, a, b) = instruction {
            // Clang also propagates constant register values into pow. Volatile
            // operand loads retain the native runtime kernel without changing
            // the frozen legacy helper or its exponent clamp.
            return Some(format!("([](double a,double b){{volatile double base=a;volatile double exponent=b;return m_pow(base,m_clamp(exponent,-8.0,8.0));}})(registers[{a}],registers[{b}])"));
        }
        if let Instruction::Variable(var @ (Var::RowMin | Var::RowMax)) = instruction {
            return Some(format!(
                "scalar_row_extrema_v2(raw, {})",
                var == Var::RowMax
            ));
        }
        if let Instruction::Call(ScalarFunction::V1(function), args, _) = instruction {
            if function == Func::Clip {
                let v = args[0];
                let a = args[1];
                let b = args[2];
                return Some(format!("clampv(registers[{v}], ((std::isnan(registers[{b}]) || registers[{a}] < registers[{b}]) ? registers[{a}] : registers[{b}]), ((std::isnan(registers[{b}]) || registers[{a}] > registers[{b}]) ? registers[{a}] : registers[{b}]))"));
            }
            if matches!(function, Func::Min | Func::Max) {
                let a = args[0];
                let b = args[1];
                let cmp = if function == Func::Min { "<" } else { ">" };
                return Some(format!("((std::isnan(registers[{b}]) || registers[{a}] {cmp} registers[{b}]) ? registers[{a}] : registers[{b}])"));
            }
        }
    }
    // Python's integer floor/ceil lose negative zero, which old native calls
    // can expose beneath a new function. Preserve it in this scalar-v2 route.
    if language == ScalarLanguage::Python {
        if let Instruction::Call(ScalarFunction::V1(function), args, _) = instruction {
            let a = args[0];
            if function == Func::Floor {
                return Some(format!(
                    "(registers[{a}] if registers[{a}]==0.0 else _floor(registers[{a}]))"
                ));
            }
            if function == Func::Round {
                return Some(format!("(math.copysign(_round_half_away(registers[{a}]), registers[{a}]) if registers[{a}]<0.0 else _round_half_away(registers[{a}]))"));
            }
        }
    }
    let expr = fragment(instruction)?;
    let rendered = match language {
        ScalarLanguage::Rust => expr.to_rust(),
        ScalarLanguage::Python => expr.to_python(),
        ScalarLanguage::Cpp17 => expr.to_cpp(false),
    };
    Some(
        if matches!(
            instruction,
            Instruction::Neg(_) | Instruction::Binary(_, _, _) | Instruction::Call(_, _, _)
        ) {
            rendered.replace("row[", "registers[")
        } else {
            rendered
        },
    )
}

fn guarded_call(
    instruction: Instruction,
    policy: NumericalPolicy,
    language: ScalarLanguage,
) -> Option<String> {
    let scalar = match instruction {
        Instruction::Call(function, args, _) => {
            let input = format!("registers[{}]", args[0]);
            match function {
                ScalarFunction::InverseLink(link) => {
                    format!("inverse_link_v2({}, {input})", link.id())
                }
                ScalarFunction::Expm1 | ScalarFunction::Log1mexp => {
                    format!("stable_exponential_v2({}, {input})", function.id())
                }
                ScalarFunction::V1(Func::Asinh) if policy == NumericalPolicy::StableV2 => {
                    format!("numerical_asinh_v2({input})")
                }
                _ => return None,
            }
        }
        Instruction::Variable(var) if policy == NumericalPolicy::StableV2 => {
            let kind = match var {
                Var::RowSum => "sum",
                Var::RowMean => "mean",
                Var::RowL1 => "l1",
                Var::RowL2 => "l2",
                _ => return None,
            };
            format!("numerical_reduce_v2(\"{kind}\", row)")
        }
        Instruction::Variable(Var::RowSum | Var::RowMean) => {
            let kind = if matches!(instruction, Instruction::Variable(Var::RowSum)) {
                "sum"
            } else {
                "mean"
            };
            format!("scalar_legacy_reduce_v2(\"{kind}\", row)")
        }
        _ => return None,
    };
    Some(
        if language == ScalarLanguage::Rust
            && !matches!(
                instruction,
                Instruction::Call(ScalarFunction::V1(Func::Asinh), _, _)
            )
        {
            format!("{scalar}.expect(\"admitted scalar function\")")
        } else {
            scalar
        },
    )
}

// Conditions are rendered per language from the permanent fault assignments.
// Call arguments are registers, already evaluated in native postorder.
fn guard_conditions(instruction: Instruction) -> Vec<(String, FaultSet)> {
    let r = |index| format!("registers[{index}]");
    match instruction {
        Instruction::Variable(Var::X) => vec![("NONFINITE(x)".into(), FaultSet::NONFINITE_INPUT)],
        Instruction::Variable(Var::Col(index)) => vec![(
            format!("NONFINITE(row[{index}])"),
            FaultSet::NONFINITE_INPUT,
        )],
        Instruction::Variable(_) => vec![("ROW_NONFINITE".into(), FaultSet::NONFINITE_INPUT)],
        Instruction::Binary(BinOp::Div, _, b) => {
            vec![(format!("ABS({}) < 1e-12", r(b)), FaultSet::DIV_NEAR_ZERO)]
        }
        Instruction::Binary(BinOp::Pow, _, b) => {
            vec![(format!("ABS({}) > 8.0", r(b)), FaultSet::POW_EXP_CLAMPED)]
        }
        Instruction::Call(ScalarFunction::V1(function), args, _) => {
            let a = r(args[0]);
            match function {
                Func::Sqrt => vec![(format!("{a} < 0.0 OR ISNAN({a})"), FaultSet::SQRT_PROJECTED)],
                Func::Log1p => vec![(format!("{a} <= -1.0"), FaultSet::LOG1P_DOMAIN)],
                Func::Log => vec![(format!("{a} <= 1e-300"), FaultSet::LOG_DOMAIN)],
                Func::Exp => vec![(format!("ABS({a}) > 30.0"), FaultSet::EXP_ARG_CLAMPED)],
                Func::Sigmoid => vec![(format!("ABS({a}) > 30.0"), FaultSet::SIGMOID_ARG_CLAMPED)],
                _ => vec![],
            }
        }
        _ => vec![],
    }
}

fn condition(source: &str, language: ScalarLanguage) -> String {
    let (finite, nan, abs, row) = match language {
        ScalarLanguage::Rust => (
            "f64::is_finite(",
            "f64::is_nan(",
            "f64::abs(",
            "row.iter().any(|&v| !v.is_finite())",
        ),
        ScalarLanguage::Python => (
            "math.isfinite(",
            "math.isnan(",
            "abs(",
            "any(not math.isfinite(v) for v in row)",
        ),
        ScalarLanguage::Cpp17 => (
            "std::isfinite(",
            "std::isnan(",
            "std::abs(",
            "std::any_of(raw.begin(),raw.end(),[](double v){return !std::isfinite(v);})",
        ),
    };
    let no = if language == ScalarLanguage::Python {
        "not "
    } else {
        "!"
    };
    source
        .replace("ROW_NONFINITE", row)
        .replace("NONFINITE(", &format!("{no}{finite}"))
        .replace("FINITE(", finite)
        .replace("ISNAN(", nan)
        .replace("ABS(", abs)
        .replace(
            " OR ",
            if language == ScalarLanguage::Python {
                " or "
            } else {
                " || "
            },
        )
        .replace(
            " AND ",
            if language == ScalarLanguage::Python {
                " and "
            } else {
                " && "
            },
        )
        .replace("NOTNAN(", &format!("{no}{nan}"))
}

fn guard(source: &str, fault: FaultSet, language: ScalarLanguage) -> String {
    let check = condition(source, language);
    let bit = fault.bits();
    match language {
        ScalarLanguage::Python => format!("    if {check}: faults |= {bit}\n"),
        _ => format!("    if ({check}) {{ faults |= {bit}; }}\n"),
    }
}

pub(crate) fn emit_value_function(
    program: &ScalarExpressionProgram,
    language: ScalarLanguage,
    stage: usize,
) -> crate::kernel::Result<String> {
    let compiled = program.compile()?;
    let mut source = match language {
        ScalarLanguage::Rust => format!("#[allow(unused_variables, unused_mut)]\nfn scalar_stage{stage}_value(row: &[f64], x: f64) -> (f64,u32) {{\n    let mut registers=[0.0f64;64]; let mut faults=0u32; let mut math_failed=false;\n"),
        ScalarLanguage::Python => format!("def scalar_stage{stage}_value(row,x):\n    registers=[0.0]*64\n    faults=0\n"),
        ScalarLanguage::Cpp17 => format!("static std::pair<double,std::uint32_t> scalar_stage{stage}_value(const std::vector<double>& raw,[[maybe_unused]] double x) {{\n    [[maybe_unused]] const double* row=raw.data();[[maybe_unused]] const auto row_len=raw.size();double registers[64]={{}};std::uint32_t faults=0;\n"),
    };
    for (index, instruction) in compiled.code.iter().copied().enumerate() {
        for (check, fault) in guard_conditions(instruction) {
            source.push_str(&guard(&check, fault, language));
        }
        if let Instruction::Call(ScalarFunction::V1(Func::Clip), args, _) = instruction {
            let check = condition(
                &format!(
                    "ISNAN(registers[{}]) AND ISNAN(registers[{}])",
                    args[1], args[2]
                ),
                language,
            );
            source.push_str(&match language {
                ScalarLanguage::Rust => {
                    format!("    assert!(!({check}),\"invalid clip bounds\");\n")
                }
                ScalarLanguage::Python => {
                    format!("    if {check}: raise ArithmeticError('invalid clip bounds')\n")
                }
                ScalarLanguage::Cpp17 => {
                    format!("    if({check})throw std::domain_error(\"invalid clip bounds\");\n")
                }
            });
        }
        if let Some(mut call) = guarded_call(instruction, compiled.numerical_policy, language) {
            if language == ScalarLanguage::Cpp17 && matches!(instruction, Instruction::Variable(_))
            {
                call = call.replace(", row)", ", raw)");
            }
            source.push_str(&match language {
                ScalarLanguage::Rust => format!("    let (value,bits)={call};faults|=bits;registers[{index}]=value;\n"),
                ScalarLanguage::Python => format!("    value,bits={call}\n    faults |= bits\n    registers[{index}]=value\n"),
                ScalarLanguage::Cpp17 => format!("    const auto value{index}={call};faults|=value{index}.second;registers[{index}]=value{index}.first;\n"),
            });
        } else {
            let value =
                legacy_value(instruction, language).expect("admitted version-one instruction");
            let end = if language == ScalarLanguage::Python {
                "\n"
            } else {
                ";\n"
            };
            source.push_str(&format!("    registers[{index}]={value}{end}"));
        }
        if let Instruction::Binary(BinOp::Pow, a, b) = instruction {
            source.push_str(&guard(&format!("NONFINITE(registers[{index}]) AND FINITE(registers[{a}]) AND NOTNAN(registers[{b}])"),FaultSet::POW_NONFINITE,language));
        }
    }
    let last = compiled.code.len() - 1;
    source.push_str(&match language {
        ScalarLanguage::Rust => format!("    assert!(!math_failed,\"invalid clip bounds\");let value=registers[{last}];if !value.is_finite() {{(0.0,faults|256)}} else {{(value.clamp(-1e12,1e12),faults|if value.abs()>1e12{{512}}else{{0}})}}\n}}\n"),
        ScalarLanguage::Python => format!("    value=registers[{last}]\n    if not math.isfinite(value): return (0.0,faults|256)\n    return (max(-1e12,min(1e12,value)),faults|(512 if abs(value)>1e12 else 0))\n"),
        ScalarLanguage::Cpp17 => format!("    const double value=registers[{last}];if(!std::isfinite(value))return {{0.0,faults|256u}};return {{std::max(-1e12,std::min(1e12,value)),faults|(std::abs(value)>1e12?512u:0u)}};\n}}\n"),
    });
    Ok(source)
}

pub(crate) fn emit_v2_helpers(
    programs: &[&ScalarExpressionProgram],
    language: ScalarLanguage,
) -> crate::kernel::Result<String> {
    use crate::kernel::{inverse_links, numerical_policy, stable_exponential};
    let compiled = programs
        .iter()
        .map(|p| p.compile())
        .collect::<crate::kernel::Result<Vec<_>>>()?;
    let mut source = String::new();
    if language == ScalarLanguage::Cpp17
        && compiled
            .iter()
            .flat_map(|p| p.code.iter())
            .any(|i| matches!(i, Instruction::Variable(Var::RowMin | Var::RowMax)))
    {
        source.push_str("static double scalar_row_extrema_v2(const std::vector<double>& row,bool maximum){double value=maximum?-std::numeric_limits<double>::infinity():std::numeric_limits<double>::infinity();for(double v:row){if(!std::isnan(v)&&(maximum?v>=value:v<=value))value=v;}return value;}\n");
    }
    if compiled
        .iter()
        .flat_map(|p| p.code.iter())
        .any(|instruction| {
            matches!(
                instruction,
                Instruction::Call(ScalarFunction::InverseLink(_), _, _)
            )
        })
    {
        source.push_str(&inverse_links::emit_helpers(match language {
            ScalarLanguage::Rust => inverse_links::InverseLinkLanguage::Rust,
            ScalarLanguage::Python => inverse_links::InverseLinkLanguage::Python,
            ScalarLanguage::Cpp17 => inverse_links::InverseLinkLanguage::Cpp17,
        }));
    }
    if compiled
        .iter()
        .flat_map(|p| p.code.iter())
        .any(|instruction| {
            matches!(
                instruction,
                Instruction::Call(ScalarFunction::Expm1 | ScalarFunction::Log1mexp, _, _)
            )
        })
    {
        source.push_str(&stable_exponential::emit_helpers(match language {
            ScalarLanguage::Rust => stable_exponential::StableExponentialLanguage::Rust,
            ScalarLanguage::Python => stable_exponential::StableExponentialLanguage::Python,
            ScalarLanguage::Cpp17 => stable_exponential::StableExponentialLanguage::Cpp17,
        })?);
    }
    if programs
        .iter()
        .any(|p| p.numerical_policy() == NumericalPolicy::StableV2)
    {
        source.push_str(&numerical_policy::emit_helpers(match language {
            ScalarLanguage::Rust => numerical_policy::NumericalLanguage::Rust,
            ScalarLanguage::Python => numerical_policy::NumericalLanguage::Python,
            ScalarLanguage::Cpp17 => numerical_policy::NumericalLanguage::Cpp17,
        }));
    }
    // Native f64 iterator sum starts at negative zero. Legacy emitted sources
    // stay frozen; this helper is local to a scalar-v2 legacy-policy variable.
    if compiled.iter().any(|p| {
        p.numerical_policy == NumericalPolicy::LegacyV1
            && p.code
                .iter()
                .any(|i| matches!(i, Instruction::Variable(Var::RowSum | Var::RowMean)))
    }) {
        source.push_str(match language {
            ScalarLanguage::Rust => "fn scalar_legacy_reduce_v2(kind: &str,row: &[f64])->Result<(f64,u32), &'static str>{let mut value=-0.0;for &v in row{value+=v;}if kind==\"mean\"{value/=row.len().max(1) as f64;}Ok((value,0))}\n",
            ScalarLanguage::Python => "def scalar_legacy_reduce_v2(kind,row):\n    value=-0.0\n    for v in row: value+=v\n    if kind=='mean': value/=max(1,len(row))\n    return (value,0)\n",
            ScalarLanguage::Cpp17 => "static std::pair<double,std::uint32_t> scalar_legacy_reduce_v2(const char* kind,const std::vector<double>& row){double value=-0.0;for(double v:row)value+=v;if(std::strcmp(kind,\"mean\")==0)value/=static_cast<double>(std::max(std::size_t{1},row.size()));return {value,0u};}\n",
        });
    }
    if language == ScalarLanguage::Python {
        source.push_str("\nimport struct\ndef scalar_bits_v2(bits):\n    return struct.unpack('<d',struct.pack('<Q',bits))[0]\n");
    }
    // Helper bodies are appended after recipe items, so Rust module docs become comments.
    if language == ScalarLanguage::Rust {
        source = source.replace("//!", "//");
    }
    Ok(source)
}

pub(crate) fn emit_stage(
    program: &ScalarExpressionProgram,
    carrier: ScalarCarrier,
    width: usize,
    stage: usize,
    language: ScalarLanguage,
) -> crate::kernel::Result<String> {
    let mut source = emit_value_function(program, language, stage)?;
    let count = if carrier == ScalarCarrier::Row {
        1
    } else {
        width
    };
    let binding = if carrier == ScalarCarrier::Row {
        "0.0".to_owned()
    } else {
        "row[column]".to_owned()
    };
    source.push_str(&match language {
        ScalarLanguage::Rust => format!("#[allow(unused_variables)]\nfn scalar_stage{stage}_values(rows: &[Vec<f64>]) -> (Vec<Vec<f64>>,Vec<u32>) {{\n    assert!(rows.iter().all(|row|row.len()=={width}),\"scalar input width\");\n    let mut features=Vec::with_capacity(rows.len());let mut faults=Vec::with_capacity(rows.len());\n    for row in rows {{let mut values=Vec::with_capacity({count});let mut bits=0u32;for column in 0..{count} {{let (value,fault)=scalar_stage{stage}_value(row,{binding});values.push(value);bits|=fault;}}features.push(values);faults.push(bits);}}\n    (features,faults)\n}}\nfn scalar_stage{stage}_apply(rows: &[Vec<f64>],append: bool)->Vec<Vec<f64>>{{assert!(!append,\"scalar stage replaces features\");scalar_stage{stage}_values(rows).0}}\n"),
        ScalarLanguage::Python => format!("def scalar_stage{stage}_values(rows):\n    if any(len(row)!={width} for row in rows): raise ValueError('scalar input width')\n    features,faults=[],[]\n    for row in rows:\n        values,bits=[],0\n        for column in range({count}):\n            value,fault=scalar_stage{stage}_value(row,{binding})\n            values.append(value)\n            bits |= fault\n        features.append(values)\n        faults.append(bits)\n    return features,faults\ndef scalar_stage{stage}_apply(rows,append):\n    if append: raise ValueError('scalar stage replaces features')\n    return scalar_stage{stage}_values(rows)[0]\n"),
        ScalarLanguage::Cpp17 => format!("static std::pair<std::vector<std::vector<double>>,std::vector<std::uint32_t>> scalar_stage{stage}_values(const std::vector<std::vector<double>>& rows){{\n    for(const auto& row:rows)if(row.size()!={width})throw std::invalid_argument(\"scalar input width\");\n    std::vector<std::vector<double>> features;std::vector<std::uint32_t> faults;\n    for(const auto& row:rows){{std::vector<double> values;std::uint32_t bits=0;for(std::size_t column=0;column<{count};++column){{const auto result=scalar_stage{stage}_value(row,{binding});values.push_back(result.first);bits|=result.second;}}features.push_back(values);faults.push_back(bits);}}\n    return {{features,faults}};\n}}\nstatic std::vector<std::vector<double>> scalar_stage{stage}_apply(const std::vector<std::vector<double>>& rows,bool append){{if(append)throw std::invalid_argument(\"scalar stage replaces features\");return scalar_stage{stage}_values(rows).first;}}\n"),
    });
    Ok(source)
}

pub(crate) fn programs(plan: &crate::spec::Plan) -> Vec<&ScalarExpressionProgram> {
    match plan {
        crate::spec::Plan::Pipeline { stages, .. } => stages
            .iter()
            .filter_map(|s| match &s.op {
                crate::spec::PlannedOp::ScalarV2 { program, .. } => Some(program),
                _ => None,
            })
            .collect(),
        _ => vec![],
    }
}

pub(crate) fn raw_input(plan: &crate::spec::Plan) -> bool {
    matches!(plan, crate::spec::Plan::Pipeline { stages, .. } if matches!(stages.first().map(|s|&s.op),Some(crate::spec::PlannedOp::ScalarV2 { .. })))
}
