# EurekaZero: The Tabular Black Hole (`docs/paper/blackhole`)

Modular LaTeX manuscript reporting receipt-pinned measured results for the EurekaZero tabular system:
problem encodings ($z_{\mathrm{p}}$), kernel solution embeddings ($z_{\mathrm{s}}$), all-data random-split neural training (`Bank r5` / `Bank r6`),
and strict multi-regime evaluation (`fair` / unseen, `in-corpus`, and `recall` / absorbed) across `Dev91`, `TabArena v0.1`, and `BeyondArena 2026`.

## Building and Verifying

```sh
make -C docs/paper/blackhole check       # Source, citation, claim-ledger, privacy, and IP-vocabulary checks
make -C docs/paper/blackhole self-test   # Proves every refusal fires on broken fixtures
make -C docs/paper/blackhole pdf         # Compiles target/blackhole-paper/EurekaZero-BlackHole.pdf and checks 0 overfull boxes
```

## Disclosure & Regime Rules

- **Three Separated Regimes:** Every headline number carries its explicit data-exposure regime (`fair` 0% board tables: `1821.5` pre-registered / `1842.0` adaptive; `in-corpus` generalist stack: `1859.0`; `recall` absorbed family specialists: `2343.8`; `recall` absorbed per-table dossiers: `2408.5`). Ladder Elo values from separate MLE fits are never merged into a single column.
- **Protected Solution Representation:** Internal discrete program representations are modeled strictly through continuous kernel embeddings $\Psi_{\mathcal{K}}(s) \in \mathbb{R}^{d_s}$ and similarity kernel $K_{\mathcal{K}}$; `scripts/check.sh` enforces the repository shipped-text rule across all `.tex`, `.md`, `.csv`, and `.json` files.
