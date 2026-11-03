# H261: Diversity-Aware 4-Reveal Selection (`h261_diverse_reveal_set`)

- **Wave**: `H261` (`h261-diverse-reveal-set-20261103` / `h261_diverse_reveal_set`)
- **Owner**: Argon
- **Status**: `PREREGISTERED` (`0` outcome values read)
- **Primary Anchor**: `R882` (`round-882.pt`, SHA256 `0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`)
- **Diagnostic Anchor**: `R3175` (`round-3175.pt`, SHA256 `1d6ea900ebed8d2a05c3b3f37576adc15aadd2b53c585bfd5cb71a103e9ff97b`)
- **Matched Budget**: `0` optimizer updates, CPU-only (`taskset -c 57-60`), `4` unique reveals on gated contexts (`top_zgap < 0.25`) and `0` reveals elsewhere (`mean_reveals_dl_val = 0.7032967032967034`).
- **Framing**: Equal-budget search-policy comparison (`4` reveals vs `4` reveals on gated contexts), **NOT** a zero-reveal checkpoint lift.

See [`PREREG.md`](PREREG.md) for the locked experimental design, cross-validation protocol (`fold_seed = 261140`), Holm (`m = 2`) testing plan (`MMR4_CV - TOP4_PICK` primary, `STRUCT4 - TOP4_PICK` secondary), and decision criteria.
