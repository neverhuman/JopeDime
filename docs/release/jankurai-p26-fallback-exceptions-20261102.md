# P26 fallback finding disposition history

The initial P26 audit flagged `crates/jope-engine/native/runtime.cpp:55`, finding `HLT-001-DEAD-MARKER`, fingerprint `sha256:59d09c86954365434cb41de239734c38c259eafcd1ba11837338836c390de04e`, at source head `dccadcd98f174271bffec7a337e41422c263ca89`.

The branch handles `cudaSetDevice(device)` failure before NCCL initialization. It stores `cudaGetErrorString(cr)` in the calling thread's `last_error`. The original C ABI exposed failure by returning a null pointer, which the Rust `Nccl::create` caller rejected and reported through `jp_error()`. The full Jankurai audit continued to report the finding even with a P26 exact-path policy entry, so that temporary entry was removed.

P26 now changes the internal C ABI to return an integer status and write the communicator through an output pointer. Rust checks both the status and that success produced a non-null communicator. The selected-device refusal and native error text remain intact. No GPU or CUDA execution was performed; the new ABI still requires normal compile and non-GPU test verification.
