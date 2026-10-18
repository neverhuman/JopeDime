# Restricted kernel candidate contract

This is a default-disabled experimental design. It provides a small typed,
straight-line numeric bytecode interpreter; it does not compile or execute
candidate source. Source bytes are provenance input only. The admitted program
is the explicit `Instruction` vector, whose digest is separately caller-pinned.

## Admitted behavior

The total subset loads finite numeric features, uses finite constants, and
applies fixed arithmetic and elementary operations. `FuelBound` caps source
bytes, instructions, input width, row count, stack depth, and total interpreter
steps. Rows are scanned in caller order; the optional sum fold is accumulated
in that same order. Invalid shape, arithmetic overflow, division by zero,
excess fuel, and forbidden instructions return errors.

The caller must pin the expected toolchain ID, target ID, source SHA-256,
bytecode artifact SHA-256, and bounds. The interpreter checks these values
before evaluation. These IDs are claims supplied by the caller: this module
does not discover or verify a compiler installation, execute a compiler, or
attest a build. The `SandboxCapability` is a validation policy, not an OS
sandbox credential or proof.

## Authority boundary

Candidates have no file, network, process, clock, randomness, environment,
hidden-state, recursion, jump, or unbounded-loop operation. They cannot read
outer or sealed splits, modify the evaluator or trusted kernel, issue a
certificate, or promote themselves. No receipt, certificate, split access,
training, campaign, or serving capability is created here. The trusted
scientific evaluator and all acceptance, split, receipt, and promotion
authority remain outside this module.

A forbidden operation in the typed program is rejected before evaluation.
This is not a compiler/runtime escape test: there is no arbitrary-code
execution path here. cgroups alone would not establish confinement for a future
compiler/runtime. Any such path needs a separately admitted, independently
reviewed isolation boundary and adversarial escape evidence before use.

## Status and integration

This module is not exported from `kernel` and no production caller is wired.
Its focused test target includes the source privately to prove the bounded
interpreter contract. Integration owns the module export, test inventory,
coverage registration, and any caller. Passing local tests establishes fixture
behavior only; it does not establish dependency acceptance, hosted CI, runtime
qualification, scientific validity, or serving promotion. No code-invention
campaign is authorized by this contract.
