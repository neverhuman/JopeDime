# Export and prediction

The CLI help advertises exporting a selected run model or generated source with `jopedime export`. Supported export kinds are `model`, `source`, and `bundle`; supported languages are Rust, Python, JavaScript, and TypeScript. `jopedime predict` accepts either a local model path or a completed run ID, plus input and output CSV paths.

These command surfaces were confirmed from the current executable's help output. They have not been executed here because a completed run requires an available registered dataset and run.

Dataset registration is now available, but the current service router still returns `CAPABILITY_UNAVAILABLE` for run creation, `model_export`, `artifact_download`, and `model_predict`. No export or prediction output is available to show from this build.

**TODO(SVC-05-F01):** execute and document export from a completed run and prediction from its exported model after run, export, and prediction handlers are implemented. Include the generated artifact names and prediction output from the permitted fixture.
