# H265: Zero-Update Listwise LambdaMART Re-Ranker of `r882` Top-4 (`h265_top4_listwise_reranker`)

- **Wave**: `H265` (`h265-top4-listwise-reranker-20261103` / `h265_top4_listwise_reranker`)
- **Owner**: Argon
- **Status**: `PREREGISTERED` (`0` outcome values read)
- **Primary Anchor (`CONTROL`)**: `r882` (`round-882.pt`, SHA256 `0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`)
- **Test-Time Reveal Budget**: `0` reveals at test time (single-pick evaluation of `r882` + `FIT`-only `LGBMRanker` over `r882`'s `top-4`; `0` optimizer updates to `r882` or DOPE).
- **Folds & Seeds**: `fold_seed = 265140` (`5` family-grouped folds across `205` `DL-Train` families), `15` cells (`5` folds $\times$ `3` seeds `265101–265103`), `boot_seed = 265150` (`10,000` paired family bootstrap draws), `Holm m = 2` (`RERANK4`, `RERANK4_SHUF`).

See [`PREREG.md`](PREREG.md) for the locked experimental protocol, feature specification, strict `H257` FIT selector, global fsync lock before `DL-Val` reads, and decision rule.
