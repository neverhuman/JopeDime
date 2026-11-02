# P26 exact-path fallback exception

This exception covers only `crates/jope-engine/native/runtime.cpp`, finding `HLT-001-DEAD-MARKER` at line 55, fingerprint `sha256:59d09c86954365434cb41de239734c38c259eafcd1ba11837338836c390de04e` in the full audit of source head `dccadcd98f174271bffec7a337e41422c263ca89`.

The flagged branch handles `cudaSetDevice(device)` failure before NCCL initialization. It stores `cudaGetErrorString(cr)` in the calling thread's `last_error` and returns a null communicator. The Rust consumer `crates/jope-engine/src/neural.rs::Nccl::create` rejects that null pointer and reports `jp_error()`. The C ABI uses this null-plus-error channel; it does not proceed with an unselected device or return success. Replacing it with a default communicator or retrying another device would violate the selected-device contract.

The exception excludes this exact file from Jankurai finding generation only. It does not change runtime behavior, test coverage, or other scanners. No GPU or CUDA execution was performed for this review. Jepson authorized P26 ownership of this exact policy entry on 2026-11-02; the entry is disjoint from the merged P29/P34 policy hunks. Remove it if the ABI error path changes or the finding is otherwise resolved.
