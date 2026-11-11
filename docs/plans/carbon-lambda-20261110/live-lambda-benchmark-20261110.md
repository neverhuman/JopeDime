# Carbon Serverless Benchmark Receipt

- **Endpoint**: `https://lchlyzekra3i75jvurq3msfnde0jlcfl.lambda-url.us-east-2.on.aws`
- **Region**: `us-east-2` (`enterprise_self_hosted=false`, `zero_knowledge_kek=true`)
- **Timestamp (epoch ms)**: `1794415107004`
- **Ping RTT**: initial `190.95 ms`, warm p50 `71.70 ms`, warm min `71.31 ms`
- **Suite Verdict**: **PASS**

| Scenario | Mode | Shards | Wall (ms) | Prep (ms) | Upload (ms) | Search (ms) | Predict (ms) | Cands | Cands/s | CV RMSE | Holdout RMSE | Status | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `free_anon_r882_wave0_cuen` | `SoloSingleLambda` | 1 | 5595.4 | 1.6 | 441.2 | 4402.8 | 223.4 | 25 | 5.7 | 0.07438 | 0.06010 | PASS | waves=1/1, guidance=wave0_seed_only, model=sha256:057e50a9a741, cuen=true |
| `pro_solo_fast_warm` | `SoloSingleLambda` | 1 | 4604.8 | 1.4 | 320.8 | 3593.9 | 201.6 | 25 | 7.0 | 0.08443 | 0.06092 | PASS | waves=1/1, guidance=multi_wave_learned, model=sha256:057e50a9a741, cuen=true |
| `pro_medium_async_multiwave` | `SoloSingleLambda` | 1 | 20952.7 | 3.0 | 315.3 | 19974.3 | 221.0 | 37 | 1.9 | 0.05348 | 0.04811 | PASS | waves=4/4, guidance=multi_wave_learned, model=sha256:057e50a9a741, cuen=true |
| `pro_hybrid_fanout_stop_predict_resume_shred` | `HybridFanOut` | 12 | 22906.5 | 0.0 | 457.6 | 21986.9 | 226.5 | 37 | 1.7 | 0.05810 | 0.04643 | PASS | paused=true, wave_pause=2, wave_end=3, guidance=multi_wave_learned, cuen=true, shredded_objs=8 |
| `free_tier_row_cap_403` | `EdgeAdmissionGuard` | 0 | 151.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0.0 | - | - | PASS | 5,200 rows rejected at edge before S3 presign |
