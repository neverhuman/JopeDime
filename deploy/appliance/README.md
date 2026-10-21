# Single-file Rust deployment

This directory versions the engineering plan for shipping JopeDime as one signed,
offline Rust executable with its selected pretrained models and native AutoML
alternatives. The proposed installed command is `jopedime`; the initial target is
Linux x86-64 with glibc 2.36 or newer.

**Status: design ready for review; appliance implementation and qualification are
pending.** This documentation PR adds no executable, model bundle, deployment job,
or new runtime capability. The existing [`jope` release procedure](../../docs/release/README.md)
continues to describe the currently implemented release.

| File | Purpose |
| --- | --- |
| [SPECIFICATION.md](SPECIFICATION.md) | Complete 22-section engineering specification, architecture diagrams, exact file changes, formats, CI, release flow, and acceptance criteria. |
| [IMPLEMENTATION_CHECKLIST.md](IMPLEMENTATION_CHECKLIST.md) | Ordered implementation work packages and the evidence required to close each one. |

The specification was prepared on October 20, 2026, America/Denver, against commit
[`95dcb0c0e44d95184607740aacd387bf5c007e89`](https://github.com/neverhuman/JopeDime/commit/95dcb0c0e44d95184607740aacd387bf5c007e89)
and is imported unchanged. Its statements about work performed describe that review.
Future implementation PRs must re-audit their merge base and record any design
changes here and in the specification.

## Implementation contract

- Embed the selected safetensors, encoder configurations, preprocessing assets,
  vocabulary, notices, and runtime in the signed executable. Training and conversion
  may continue to use other languages.
- Keep application execution in Rust and offline. Reuse the existing native methods
  and model ports; external Python benchmark adapters are not Rust alternatives.
- Make the same GPU-capable Linux artifact start on a host without CUDA and select
  CPU when an eligible GPU path is unavailable. A driver-only NVIDIA host must pass
  the full GPU qualification using approved bundled user-space libraries.
- Preserve accepted model identities through lossless packaging. Treat precision
  changes and quantization as separately qualified model variants.
- Resolve every method's exact model dependency graph and distribution rights.
  An internal profile name does not grant redistribution rights.
- Extend the existing CI receipt/aggregate system and release authority. Bind
  qualification and publication to the exact signed candidate bytes.

The specification distinguishes Rust application code from host ABI dependencies,
GPU drivers, vendor libraries, and GPU kernels. Each supported platform requires
its own artifact and qualification.

## Review map

| Review area | Specification |
| --- | --- |
| Source audit and product scope | [Sections 2–3](SPECIFICATION.md#2-evidence-boundary-and-repository-audit) |
| Runtime ownership and model dependencies | [Sections 4–6](SPECIFICATION.md#4-architecture-and-ownership) |
| Exact new and modified files | [Section 7](SPECIFICATION.md#7-repository-layout) |
| Export, verified assets, and package signature | [Sections 8–10](SPECIFICATION.md#8-export-and-safetensors-contract) |
| GPU/CPU execution | [Section 11](SPECIFICATION.md#11-gpucpu-deployment-engineering) |
| CI and release orchestration | [Sections 15–16](SPECIFICATION.md#15-ci-design) |
| Qualification and staged PRs | [Sections 17–18](SPECIFICATION.md#17-acceptance-and-qualification) |
| Examples, unresolved evidence, and sources | [Sections 19–21](SPECIFICATION.md#19-machine-readable-contract-examples) |
| Product completion | [Section 22](SPECIFICATION.md#22-definition-of-done) |

Retire CUDA linkage, full model dependency resolution, distribution rights, and
artifact-size uncertainty early. Follow [P0–P1](IMPLEMENTATION_CHECKLIST.md#p0--inventory-and-contracts)
for the first executable PR. The examples in the specification are proposed
contracts, not production locks or release-ready configuration.
