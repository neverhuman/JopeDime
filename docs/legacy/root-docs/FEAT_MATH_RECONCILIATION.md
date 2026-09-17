# feat-math reconciliation: jope-dime search vs jain-math execution

jope-dime searches with the vendored feat-math grammar
(`parallel/assets/grammar/featmath.json`, 401 tokens). The jain product runs
with jain-math's feat-math, and its adapter
(`jain-core crates/feat-core/src/jope_invention.rs`, `admit()`) admits a
search program only if every stage op is in jain-math's registry, is
`executable`, and is not `quarantined`. The two op sets differ, so part of a
search can go to programs the product will reject.

## Launch: restricted vocabulary (this change)

- `parallel/assets/grammar/jain-math-executable-9fa594e2.json` is a snapshot
  of the ops jain-math can execute at rev
  `9fa594e2eedb1916cc0e1ef4038c79adeccd6832`
  (`crates/feat-math/src/registry.rs`, `OPERATORS` table at line 443,
  registry hash `8173b07f…`). It lists 46 executable ops and 7 that are not
  (`bag` and six quarantined sketch/RFF variants).
- `parallel/assets/grammar/featmath-jain-v1.json` is `featmath.json` with the
  non-executable heads removed from `ops`, `engine_heads` and `head_tasks`.
  Ten heads are removed: `extra_trees`, `gam_boost`, `logit_ridge`, the four
  `multiclass_*` heads, `portfolio`, `ridge_gcv` and `tab_boost`. Every
  `featmath.json` transform is executable in jain-math, so none is removed.
  The token table and `index` stay the same, so token ids, checkpoints and
  corpora remain valid. `featmath.json` and the pins in `parity.json` are
  unchanged.
- `jope_parallel::jain_vocabulary::derive_restricted_vocabulary` builds the
  restricted file from those two inputs. Lib tests check that the committed
  file equals the derived bytes and that it loads through the loader `search`
  uses. To regenerate it, run
  `cargo run --example derive_jain_vocabulary` (add `-- --check` to compare
  only).
- The kernel-v1 product binary rejects a proposal whose op is not in the
  vocabulary before evaluating it (`Vocab::project_proposal`). As a result,
  `search --vocabulary featmath-jain-v1.json` only evaluates programs the
  product can admit.

## Later: reviewed ports into jain-math (not implemented here)

To bring a removed head back, port it into jain-math through review, behind a
parity harness. The harness would run the vendored and jain-math
implementations on the same fits and compare the predictions. After the port
lands, refresh the executable snapshot to the new jain-math rev and derive a
new `featmath-jain-vN.json`. This change does not include the harness,
because it would fit models outside the eval-registry authority, which is
fenced.

## Long-term: consume jain-math by git pin

jope-dime should depend on jain-math's feat-math through a git pin instead of
the vendored copy. Then the grammar, the executor and the product adapter
would all use one registry, and a snapshot file would no longer be needed.
