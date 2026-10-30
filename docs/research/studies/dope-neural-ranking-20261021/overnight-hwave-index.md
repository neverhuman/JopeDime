| Wave | Date (MDT) | Hypothesis | Status | Verdict | Dir |
|---|---|---|---|---|---|
| H46 | 2026-10-30 00:35 | Multi-step+EMA / warmup-cosine / robust and family-balanced losses vs restore-on-reject plateau | complete | No selected win; A/B show objective mismatch, C final no gain, D negative; seeds near-pseudoreplicates | h46/ |
| H47 | 2026-10-30 01:15 | Utility-aligned ranking losses (listwise CE / centred Huber / pairwise / listwise+L2-SP) after H46 showed loss-down/utility-down mismatch | complete | No win; all final median deltas negative; RP best but below r785; seeds near-pseudoreplicates | h47/ |
| H48 | 2026-10-30 09:24 | Inner-family-CV utility alignment: control, persistent AdamW/EMA, ES, plus gradient diagnostic | complete | No strict win; ES only lead, fails family CI and Elo; 0/PF fold-3 winner’s-curse signature | h48/ |
