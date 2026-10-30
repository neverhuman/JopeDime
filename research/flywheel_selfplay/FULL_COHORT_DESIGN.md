# Full-cohort adapter repair

Jepson's flywheel_fix_2150 and regbin_2215 authorize this bounded repair. Keep the
existing loop running while preparing the change; after local checks and the
owner-authorized squash merge, transfer only that worker at a completed round.
No GitHub CI is dispatched. The frozen DOPE checksum is a hard startup/reload guard.

The packet contains all 1219 DL-Train and 91 DL-Val contexts, hashed pool files and
training-row-only JTT1 tables. Roles and family disjointness are checked before
model use. A seeded shuffle keeps family blocks together and covers every training
context each optimizer step. Rotate 16 training families for at most 64 native GP
fits. Max2 AdamW updates, max8192 batch, CPU85%, GPU90%, 10GiB combined output and
50GiB free reserve continue unchanged. Copy the append-only promotions journal;
replay it before mutation. Continue global round numbers and complete optimizer
state, but initialize a new DL-Val91 admission identity instead of comparing the
old DL-Train8 development score with the new validation score.

After an incumbent restore, a private seeded generator perturbs only trainable
adapter tensors and sets a bounded absolute warm-restart LR. The frozen anchor and
DOPE remain bit identical. DL-Val is evaluated under no_grad and never supplies
margin constraints, optimizer batches or mutation parents. A two-restored-proposal
fixture proves distinct parameter SHAs and unchanged frozen tensors.

Reporting and clocks are native Rust through thin PyO3 calls: full-context/per-family
validation, distinct contexts/s, completed-round promotions/hour, same-definition
train/validation gap, actual transferred-buffer bytes / pack delivery seconds,
score-history append with flock, and summary JSONL. Optional native telemetry uses
JOPE_TELEMETRY_RUN_DIR and the 1Hz host collector. Phase2 is unused; Rust, all-language
and reasoning token rates are explicitly unavailable. GP DSL token rate is named
separately. Golden45/panel Elo is unavailable within the unchanged fit budget.

Current S3 manifests identify every approved context as regression. Report regression
and binary separately with binary count zero and unavailable binary validation/Elo.
Refuse binary/multiclass through this regression GP evaluator. No blended joint score
is introduced. Raw scores append to the local historical store with exact context,
DL split and metric identities; panel test/AUC histories are incomparable. Percentiles
remain pending JRB-02. Cheap joint-vs-single interference evidence is unavailable;
JRB-01 owns the equal-budget three-arm study. One joint adapter remains the goal.

Feature columns exceeding float32 range are rescaled only for the frozen landmark
engine's per-column standardization. Original labels and raw GP evaluator bytes are
unchanged. All1310 CPU landmark bundles must be finite before deployment.

Local acceptance: Rust telemetry tests and Python rollback/cache/promotion/Phase3
fixtures, Python Ruff/format, native Clippy with python feature, exact-source wheel
import/metric-registry smoke, all1310 data hashes and landmark preflight. After merge,
verify at least five live rounds, full training coverage, <=64 fits, frozen DOPE,
changing candidate SHAs and the validation curve. An unavailable quality quantity
stays unavailable; no panel or sealed generalization claim follows from this run.

Full-cohort qualification exposed a residency limit: keeping all1219 raw pools on
the GPU while materializing the8192-context model signatures exceeds the90% cap.
The gathered slate owns its tensors, so full mode releases raw cached pools and
unused allocator segments before forward. Every effective batch and both optimizer
updates remain unchanged; cold transfers are included in native byte counters.
Startup checks host headroom every16 contexts with the same fixed4 threads and
80% wait/85% hard budget, avoiding a mandatory5.5min sampling delay per restart.

Owner update t528u (regfirst_2250): train, select and judge regression only for now.
Headline regression validation, regression Elo and per-dataset/split percentile
trend; binary/JRB-01 work is deferred until sustained regression validation gains.
This supersedes the immediate joint-training priority above. No binary blend.

Inter-round qualification also exposed allocator fragmentation: after a real8192
slate, only38MB remained allocated but free device capacity fell below the20GiB
guard. Releasing final training references helps; expandable allocator segments
restore more than20GiB free in the locked real-input diagnostic. Configure this
before importing Torch, record it in the run manifest, and retain the20GiB wait
and90% cap. No batch, optimizer or frozen-weight adjustment is authorized by this.

Resume precision: older native JSON parsing could shift a round receipt's f64
validation utility by one ULP while every selected program stayed identical. The
validated best pointer is committed admission state and always takes precedence
over observational round maxima. Native serde_json uses float_roundtrip so future
receipt serialization preserves exact f64 bits; old receipts remain immutable.
The original failure and corrected bit-exact test are retained in custody.
