//! Bounded pure interpreter for disabled experimental kernel candidates.
//!
//! This validates and interprets a small straight-line numeric instruction
//! subset. It does not compile or execute candidate source, create receipts,
//! issue certificates, inspect splits, or authorize scientific promotion.

use sha2::{Digest, Sha256};

pub const INTERPRETER_ID: &str = "restricted-numeric-interpreter.v1";
pub const MAX_PROGRAM_INSTRUCTIONS: usize = 1_024;
pub const MAX_SOURCE_BYTES: usize = 1_048_576;
pub const MAX_INPUT_WIDTH: usize = 512;
pub const MAX_ROWS: usize = 4_096;
pub const MAX_STACK: usize = 256;
pub const MAX_FUEL: u64 = 4_202_496; // MAX_ROWS * (MAX_PROGRAM_INSTRUCTIONS + scan + fold)

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FuelBound {
    pub max_program_instructions: usize,
    pub max_source_bytes: usize,
    pub max_input_width: usize,
    pub max_rows: usize,
    pub max_stack: usize,
    pub max_fuel: u64,
}

impl FuelBound {
    pub fn validate(self) -> Result<(), KernelError> {
        if self.max_program_instructions == 0
            || self.max_program_instructions > MAX_PROGRAM_INSTRUCTIONS
            || self.max_source_bytes == 0
            || self.max_source_bytes > MAX_SOURCE_BYTES
            || self.max_input_width == 0
            || self.max_input_width > MAX_INPUT_WIDTH
            || self.max_rows == 0
            || self.max_rows > MAX_ROWS
            || self.max_stack == 0
            || self.max_stack > MAX_STACK
            || self.max_fuel == 0
            || self.max_fuel > MAX_FUEL
        {
            return Err(KernelError::InvalidFuelBound);
        }
        let required = (self.max_rows as u64)
            .checked_mul(self.max_program_instructions as u64 + 2)
            .ok_or(KernelError::InvalidFuelBound)?;
        if self.max_fuel < required {
            return Err(KernelError::InvalidFuelBound);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Instruction {
    LoadFeature(usize),
    Constant(f64),
    Add,
    Subtract,
    Multiply,
    Divide,
    Negate,
    Absolute,
    Minimum,
    Maximum,
    SquareRoot,
    Tanh,
    /// Explicitly represent forbidden source capabilities so the validator
    /// can refuse them before an instruction is run.
    Forbidden(ForbiddenOperation),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ForbiddenOperation {
    FileSystem,
    Network,
    Process,
    Clock,
    Randomness,
    Environment,
    HiddenMutableState,
    Recursion,
    Jump,
    UnboundedLoop,
    ReadOuterOrSealedSplit,
    ModifyEvaluator,
    ModifyTrustedKernel,
    IssueCertificate,
    PromoteCandidate,
}

#[derive(Clone, Debug)]
pub struct CandidateArtifact {
    /// Source bytes are hashed and compared, never compiled or interpreted.
    pub source: Vec<u8>,
    pub toolchain_id: String,
    pub target_id: String,
    pub instructions: Vec<Instruction>,
    /// Candidates cannot certify themselves. Any value here is rejected.
    pub self_issued_certificate: Option<Vec<u8>>,
}

/// Caller-pinned identity and bounds. This is a validation policy, not an OS
/// sandbox credential or proof that an external compiler/runtime is confined.
#[derive(Clone, Debug)]
pub struct SandboxCapability {
    pub toolchain_id: String,
    pub target_id: String,
    pub source_sha256: [u8; 32],
    pub artifact_sha256: [u8; 32],
    pub fuel: FuelBound,
}

impl SandboxCapability {
    pub fn new(
        toolchain_id: impl Into<String>,
        target_id: impl Into<String>,
        source_sha256: [u8; 32],
        artifact_sha256: [u8; 32],
        fuel: FuelBound,
    ) -> Self {
        Self {
            toolchain_id: toolchain_id.into(),
            target_id: target_id.into(),
            source_sha256,
            artifact_sha256,
            fuel,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KernelError {
    InvalidFuelBound,
    InvalidBinding,
    EmptyProgram,
    WrongToolchain,
    WrongTarget,
    WrongSourceDigest,
    WrongArtifactDigest,
    SourceTooLarge,
    SelfIssuedCertificate,
    ForbiddenInstruction(ForbiddenOperation),
    InvalidInstructionStream,
    FeatureOutOfBounds(usize),
    InvalidInputShape,
    TooManyRows,
    FuelExhausted,
    NonFiniteValue,
    DivisionByZero,
}

impl std::fmt::Display for KernelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for KernelError {}

#[derive(Clone, Debug)]
pub struct RestrictedKernel {
    instructions: Vec<Instruction>,
    fuel: FuelBound,
    required_input_width: usize,
}

impl RestrictedKernel {
    /// Validate exact source/toolchain/target/artifact identities, bounds and
    /// instruction safety before producing an in-memory interpreter handle.
    pub fn admit(
        candidate: CandidateArtifact,
        capability: &SandboxCapability,
    ) -> Result<Self, KernelError> {
        capability.fuel.validate()?;
        if capability.toolchain_id.is_empty() || capability.target_id.is_empty() {
            return Err(KernelError::InvalidBinding);
        }
        if candidate.toolchain_id != capability.toolchain_id {
            return Err(KernelError::WrongToolchain);
        }
        if candidate.target_id != capability.target_id {
            return Err(KernelError::WrongTarget);
        }
        if candidate.source.len() > capability.fuel.max_source_bytes {
            return Err(KernelError::SourceTooLarge);
        }
        if candidate.instructions.len() > capability.fuel.max_program_instructions {
            return Err(KernelError::InvalidFuelBound);
        }
        if sha256(&candidate.source) != capability.source_sha256 {
            return Err(KernelError::WrongSourceDigest);
        }
        if instruction_digest(&candidate.instructions) != capability.artifact_sha256 {
            return Err(KernelError::WrongArtifactDigest);
        }
        if candidate.self_issued_certificate.is_some() {
            return Err(KernelError::SelfIssuedCertificate);
        }
        if candidate.instructions.is_empty() {
            return Err(KernelError::EmptyProgram);
        }

        let mut stack_depth = 0_usize;
        let mut required_input_width = 0_usize;
        for instruction in &candidate.instructions {
            match instruction {
                Instruction::LoadFeature(index) => {
                    if *index >= capability.fuel.max_input_width {
                        return Err(KernelError::FeatureOutOfBounds(*index));
                    }
                    required_input_width = required_input_width.max(index + 1);
                    stack_depth += 1;
                }
                Instruction::Constant(value) => {
                    if !value.is_finite() {
                        return Err(KernelError::NonFiniteValue);
                    }
                    stack_depth += 1;
                }
                Instruction::Negate
                | Instruction::Absolute
                | Instruction::SquareRoot
                | Instruction::Tanh => {
                    if stack_depth < 1 {
                        return Err(KernelError::InvalidInstructionStream);
                    }
                }
                Instruction::Add
                | Instruction::Subtract
                | Instruction::Multiply
                | Instruction::Divide
                | Instruction::Minimum
                | Instruction::Maximum => {
                    if stack_depth < 2 {
                        return Err(KernelError::InvalidInstructionStream);
                    }
                    stack_depth -= 1;
                }
                Instruction::Forbidden(operation) => {
                    return Err(KernelError::ForbiddenInstruction(*operation));
                }
            }
            if stack_depth > capability.fuel.max_stack {
                return Err(KernelError::InvalidFuelBound);
            }
        }
        if stack_depth != 1 {
            return Err(KernelError::InvalidInstructionStream);
        }

        Ok(Self {
            instructions: candidate.instructions,
            fuel: capability.fuel,
            required_input_width,
        })
    }

    /// Deterministic bounded scan over input rows. Every row is evaluated in
    /// caller order; the candidate has no control over iteration or row roles.
    pub fn evaluate_rows(&self, rows: &[Vec<f64>]) -> Result<Vec<f64>, KernelError> {
        self.check_batch(rows, 1)?;

        let mut remaining_fuel = self.fuel.max_fuel;
        let mut output = Vec::with_capacity(rows.len());
        for row in rows {
            remaining_fuel = remaining_fuel
                .checked_sub(1)
                .ok_or(KernelError::FuelExhausted)?;
            output.push(self.evaluate_row(row, &mut remaining_fuel)?);
        }
        Ok(output)
    }

    /// Deterministic bounded fold of per-row scalar outputs. The interpreter
    /// owns row order and the sum is accumulated in that same order.
    pub fn evaluate_sum(&self, rows: &[Vec<f64>]) -> Result<f64, KernelError> {
        self.check_batch(rows, 2)?;
        let mut remaining_fuel = self.fuel.max_fuel;
        let mut sum = 0.0;
        for row in rows {
            remaining_fuel = remaining_fuel
                .checked_sub(1)
                .ok_or(KernelError::FuelExhausted)?;
            let value = self.evaluate_row(row, &mut remaining_fuel)?;
            remaining_fuel = remaining_fuel
                .checked_sub(1)
                .ok_or(KernelError::FuelExhausted)?;
            sum += value;
            if !sum.is_finite() {
                return Err(KernelError::NonFiniteValue);
            }
        }
        Ok(sum)
    }

    fn check_batch(&self, rows: &[Vec<f64>], row_overhead: u64) -> Result<(), KernelError> {
        if rows.len() > self.fuel.max_rows {
            return Err(KernelError::TooManyRows);
        }
        let fuel_needed = (rows.len() as u64)
            .checked_mul(self.instructions.len() as u64 + row_overhead)
            .ok_or(KernelError::FuelExhausted)?;
        if fuel_needed > self.fuel.max_fuel {
            return Err(KernelError::FuelExhausted);
        }
        if rows.iter().any(|row| {
            row.len() < self.required_input_width
                || row.len() > self.fuel.max_input_width
                || row.iter().any(|value| !value.is_finite())
        }) {
            return Err(KernelError::InvalidInputShape);
        }
        Ok(())
    }

    fn evaluate_row(&self, row: &[f64], fuel: &mut u64) -> Result<f64, KernelError> {
        let mut stack = Vec::with_capacity(self.fuel.max_stack.min(32));
        for instruction in &self.instructions {
            *fuel = fuel.checked_sub(1).ok_or(KernelError::FuelExhausted)?;
            match instruction {
                Instruction::LoadFeature(index) => stack.push(row[*index]),
                Instruction::Constant(value) => stack.push(*value),
                Instruction::Negate => unary(&mut stack, |v| -v)?,
                Instruction::Absolute => unary(&mut stack, f64::abs)?,
                Instruction::SquareRoot => unary(&mut stack, f64::sqrt)?,
                Instruction::Tanh => unary(&mut stack, f64::tanh)?,
                Instruction::Add => binary(&mut stack, |a, b| a + b)?,
                Instruction::Subtract => binary(&mut stack, |a, b| a - b)?,
                Instruction::Multiply => binary(&mut stack, |a, b| a * b)?,
                Instruction::Divide => {
                    let right = stack.pop().ok_or(KernelError::InvalidInstructionStream)?;
                    let left = stack.pop().ok_or(KernelError::InvalidInstructionStream)?;
                    if right == 0.0 {
                        return Err(KernelError::DivisionByZero);
                    }
                    stack.push(left / right);
                }
                Instruction::Minimum => binary(&mut stack, f64::min)?,
                Instruction::Maximum => binary(&mut stack, f64::max)?,
                Instruction::Forbidden(operation) => {
                    return Err(KernelError::ForbiddenInstruction(*operation));
                }
            }
            if stack.iter().any(|value| !value.is_finite()) {
                return Err(KernelError::NonFiniteValue);
            }
        }
        if stack.len() != 1 {
            return Err(KernelError::InvalidInstructionStream);
        }
        stack.pop().ok_or(KernelError::InvalidInstructionStream)
    }
}

pub fn source_digest(source: &[u8]) -> [u8; 32] {
    sha256(source)
}

pub fn artifact_digest(instructions: &[Instruction]) -> [u8; 32] {
    instruction_digest(instructions)
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn instruction_digest(instructions: &[Instruction]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"jopedime.restricted-kernel-bytecode.v1\0");
    hasher.update((instructions.len() as u64).to_le_bytes());
    for instruction in instructions {
        match instruction {
            Instruction::LoadFeature(index) => {
                hasher.update([0]);
                hasher.update((*index as u64).to_le_bytes());
            }
            Instruction::Constant(value) => {
                hasher.update([1]);
                hasher.update(value.to_bits().to_le_bytes());
            }
            Instruction::Add => hasher.update([2]),
            Instruction::Subtract => hasher.update([3]),
            Instruction::Multiply => hasher.update([4]),
            Instruction::Divide => hasher.update([5]),
            Instruction::Negate => hasher.update([6]),
            Instruction::Absolute => hasher.update([7]),
            Instruction::Minimum => hasher.update([8]),
            Instruction::Maximum => hasher.update([9]),
            Instruction::SquareRoot => hasher.update([10]),
            Instruction::Tanh => hasher.update([11]),
            Instruction::Forbidden(operation) => {
                hasher.update([255, *operation as u8]);
            }
        }
    }
    hasher.finalize().into()
}

fn unary(stack: &mut Vec<f64>, operation: impl FnOnce(f64) -> f64) -> Result<(), KernelError> {
    let value = stack.pop().ok_or(KernelError::InvalidInstructionStream)?;
    stack.push(operation(value));
    Ok(())
}

fn binary(
    stack: &mut Vec<f64>,
    operation: impl FnOnce(f64, f64) -> f64,
) -> Result<(), KernelError> {
    let right = stack.pop().ok_or(KernelError::InvalidInstructionStream)?;
    let left = stack.pop().ok_or(KernelError::InvalidInstructionStream)?;
    stack.push(operation(left, right));
    Ok(())
}
