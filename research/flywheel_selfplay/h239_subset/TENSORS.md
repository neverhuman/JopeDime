# H239 frozen tensor census

See [PREREG.md](PREREG.md) for the branch-depth convention. All rows are CONTROL and LLRD trainable.

| Tensor (`adapter_10tok.` prefix) | Shape | Parameters | BIAS_ONLY | HEAD_ONLY | LLRD d | LR multiplier |
|---|---|---:|---|---|---:|---:|
| `modality_type_emb` | [10, 32] | 320 | no | no | 4 | 0.0625 |
| `modality_gate_logits` | [10] | 10 | no | no | 4 | 0.0625 |
| `mm_prior_scale` | [] | 1 | no | no | 2 | 0.25 |
| `null_kernel_w0_raw` | [] | 1 | no | no | 1 | 0.5 |
| `sem_kernel_scale_raw` | [] | 1 | no | no | 1 | 0.5 |
| `emb_kernel_scale_raw` | [] | 1 | no | no | 1 | 0.5 |
| `traj_kernel_blend` | [] | 1 | no | no | 0 | 1.0 |
| `null_hist_token` | [1, 1, 32] | 32 | no | no | 2 | 0.25 |
| `traj_attn_scale` | [] | 1 | no | no | 0 | 1.0 |
| `modality_norm.weight` | [32] | 32 | yes | no | 4 | 0.0625 |
| `modality_norm.bias` | [32] | 32 | yes | no | 4 | 0.0625 |
| `prog_412_bottleneck.0.weight` | [16, 412] | 6592 | no | no | 5 | 0.03125 |
| `prog_412_bottleneck.0.bias` | [16] | 16 | yes | no | 5 | 0.03125 |
| `prog_412_bottleneck.1.weight` | [16] | 16 | yes | no | 5 | 0.03125 |
| `prog_412_bottleneck.1.bias` | [16] | 16 | yes | no | 5 | 0.03125 |
| `cand_query_proj.0.weight` | [32, 66] | 2112 | no | no | 4 | 0.0625 |
| `cand_query_proj.0.bias` | [32] | 32 | yes | no | 4 | 0.0625 |
| `cand_query_proj.1.weight` | [32] | 32 | yes | no | 4 | 0.0625 |
| `cand_query_proj.1.bias` | [32] | 32 | yes | no | 4 | 0.0625 |
| `mm_cross_attn.in_proj_weight` | [96, 32] | 3072 | no | no | 3 | 0.125 |
| `mm_cross_attn.in_proj_bias` | [96] | 96 | yes | no | 3 | 0.125 |
| `mm_cross_attn.out_proj.weight` | [32, 32] | 1024 | no | no | 3 | 0.125 |
| `mm_cross_attn.out_proj.bias` | [32] | 32 | yes | no | 3 | 0.125 |
| `ctx_bilinear_u.weight` | [16, 32] | 512 | no | no | 3 | 0.125 |
| `prog_bilinear_v.weight` | [16, 24] | 384 | no | no | 3 | 0.125 |
| `mm_prior_out.0.weight` | [24, 50] | 1200 | no | no | 3 | 0.125 |
| `mm_prior_out.0.bias` | [24] | 24 | yes | no | 3 | 0.125 |
| `mm_prior_out.2.weight` | [1, 24] | 24 | no | no | 2 | 0.25 |
| `mm_prior_out.2.bias` | [1] | 1 | yes | no | 2 | 0.25 |
| `hist_token_proj.0.weight` | [32, 60] | 1920 | no | no | 2 | 0.25 |
| `hist_token_proj.0.bias` | [32] | 32 | yes | no | 2 | 0.25 |
| `hist_token_proj.1.weight` | [32] | 32 | yes | no | 2 | 0.25 |
| `hist_token_proj.1.bias` | [32] | 32 | yes | no | 2 | 0.25 |
| `traj_cross_attn.in_proj_weight` | [96, 32] | 3072 | no | no | 1 | 0.5 |
| `traj_cross_attn.in_proj_bias` | [96] | 96 | yes | no | 1 | 0.5 |
| `traj_cross_attn.out_proj.weight` | [32, 32] | 1024 | no | no | 1 | 0.5 |
| `traj_cross_attn.out_proj.bias` | [32] | 32 | yes | no | 1 | 0.5 |
| `traj_delta_head.0.weight` | [24, 67] | 1608 | no | no | 1 | 0.5 |
| `traj_delta_head.0.bias` | [24] | 24 | yes | no | 1 | 0.5 |
| `traj_delta_head.2.weight` | [1, 24] | 24 | no | yes | 0 | 1.0 |
| `traj_delta_head.2.bias` | [1] | 1 | yes | yes | 0 | 1.0 |
| `parent_delta_net.0.weight` | [48, 114] | 5472 | no | no | 1 | 0.5 |
| `parent_delta_net.0.bias` | [48] | 48 | yes | no | 1 | 0.5 |
| `parent_delta_net.1.weight` | [48] | 48 | yes | no | 1 | 0.5 |
| `parent_delta_net.1.bias` | [48] | 48 | yes | no | 1 | 0.5 |
| `parent_delta_net.3.weight` | [1, 48] | 48 | no | no | 0 | 1.0 |
| `parent_delta_net.3.bias` | [1] | 1 | yes | no | 0 | 1.0 |
